"""Unified, reproducible trainer for the paired SegFormer/ECSegFormer experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.segformer_enhancements import create_enhanced_segformer
from models.segformer_model import create_segformer
from models.segnext_tiny import create_segnext_tiny
from models.review_baselines import (
    DeepLabV3PlusBaseline,
    PSPNetBaseline,
    UNetBaseline,
)
from utils.advanced_loss_functions import CombinedFocalDiceLoss
from utils.class_balanced_sampler import ClassBalancedBatchSampler
from utils.segmentation_metrics import ConfusionMatrixMeter


COLOR_TO_ID = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
    (255, 255, 0): 4,
}
CLASS_COLORS = np.asarray(list(COLOR_TO_ID), dtype=np.int32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by CUDA >= 10.2 for deterministic CuBLAS matrix products.
        # This is set before the first CUDA tensor/model is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class ManifestDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, split: str, transform) -> None:
        self.root = root
        self.transform = transform
        with manifest.open(newline="", encoding="utf-8") as handle:
            self.records = [
                row for row in csv.DictReader(handle) if row["split"] == split
            ]
        if not self.records:
            raise ValueError(f"No records found for split={split}")
        self.contains_red = [
            int(row.get("class_1_pixels", 0)) > 0 for row in self.records
        ]
        self.contains_green = [
            int(row.get("class_2_pixels", 0)) > 0 for row in self.records
        ]
        self.red_indices = [
            index for index, value in enumerate(self.contains_red) if value
        ]
        self.green_indices = [
            index for index, value in enumerate(self.contains_green) if value
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image = cv2.imread(str(self.root / row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / row["mask_path"]), cv2.IMREAD_COLOR)
        if image is None or mask is None:
            raise ValueError(f"Cannot load sample: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        pixels = mask_rgb.astype(np.int32)
        distances = (
            (pixels[:, :, None, :] - CLASS_COLORS[None, None, :, :]) ** 2
        ).sum(axis=3)
        mask_id = np.argmin(distances, axis=2).astype(np.uint8)
        transformed = self.transform(image=image, mask=mask_id)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]
        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.from_numpy(mask_tensor)
        return image_tensor, mask_tensor.long(), row["filename"]


def create_transforms(image_size: int):
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.4),
            A.VerticalFlip(p=0.25),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=8, p=0.2, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=12,
                val_shift_limit=8,
                p=0.3,
            ),
            A.RandomGamma(gamma_limit=(95, 105), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
    )
    eval_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
    )
    return train_transform, eval_transform


def unwrap_outputs(output):
    if isinstance(output, (tuple, list)):
        return (
            output[0],
            output[1] if len(output) > 1 else None,
            output[2] if len(output) > 2 else None,
        )
    if isinstance(output, dict):
        logits = output.get("final", output.get("logits"))
        if logits is None:
            logits = next(iter(output.values()))
        return logits, output.get("green_aux"), output.get("edge_aux")
    return output, None, None


def make_boundary_target(
    masks: torch.Tensor, num_classes: int, classwise: bool = False
) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(
        masks, num_classes=num_classes
    ).permute(0, 3, 1, 2).float()
    dilated = torch.nn.functional.max_pool2d(
        one_hot, kernel_size=3, stride=1, padding=1
    )
    eroded = -torch.nn.functional.max_pool2d(
        -one_hot, kernel_size=3, stride=1, padding=1
    )
    class_boundaries = (dilated - eroded).clamp_(0.0, 1.0)
    if classwise:
        return class_boundaries
    return class_boundaries.amax(dim=1, keepdim=True)


def supervised_boundary_loss(
    boundary_logits: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    pos_weight: float,
    bce_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    target = make_boundary_target(
        masks,
        num_classes,
        classwise=boundary_logits.shape[1] == num_classes,
    )
    positive_weight = torch.tensor(
        [pos_weight],
        dtype=boundary_logits.dtype,
        device=boundary_logits.device,
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        boundary_logits, target, pos_weight=positive_weight
    )
    probability = torch.sigmoid(boundary_logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(
        dim=(1, 2, 3)
    )
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    return bce_weight * bce + dice_weight * dice.mean()


def replace_batch_norm_with_group_norm(module: torch.nn.Module) -> None:
    """Replace BatchNorm in small-batch enhancement branches in-place."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm2d):
            channels = child.num_features
            groups = next(
                (
                    candidate
                    for candidate in (8, 4, 2)
                    if channels % candidate == 0
                ),
                1,
            )
            setattr(module, name, torch.nn.GroupNorm(groups, channels))
        else:
            replace_batch_norm_with_group_norm(child)


class StabilizedEnhancedModel(torch.nn.Module):
    """Keep enhancements residual and bounded around the base prediction."""

    def __init__(self, enhanced_model: torch.nn.Module, residual_blend: float):
        super().__init__()
        if not 0.0 < residual_blend <= 1.0:
            raise ValueError("residual_blend must be in (0, 1]")
        self.enhanced_model = enhanced_model
        self.residual_blend = float(residual_blend)

    def forward(self, images):
        outputs = self.enhanced_model(images, return_intermediates=True)
        base_logits = outputs["original"]
        enhanced_logits = outputs["final"]
        stable_logits = base_logits + self.residual_blend * (
            enhanced_logits - base_logits
        )
        aux_logits = None
        if (
            self.enhanced_model.use_green_aux_head
            and self.enhanced_model.green_aux_head is not None
        ):
            aux_logits = self.enhanced_model.green_aux_head(stable_logits)
        edge_aux_logits = outputs.get("edge_aux")
        if edge_aux_logits is not None:
            return stable_logits, aux_logits, edge_aux_logits
        return stable_logits, aux_logits


def load_encoder_pretraining(model, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")
    state = checkpoint.get(
        "state_dict", checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    )
    encoder = model.encoder
    target = encoder.state_dict()
    matched = {}
    for raw_key, value in state.items():
        key = raw_key
        for prefix in (
            "module.",
            "backbone.",
            "encoder.",
            "original_model.encoder.",
        ):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key in target and hasattr(value, "shape") and value.shape == target[key].shape:
            matched[key] = value
    checkpoint_format = "native"
    if not matched and any(key.startswith("layers.") for key in state):
        # Convert torchvision-style SegFormer MiT keys to the naming used by
        # this repository. Its attention layer stores Q/K/V together, while
        # the local implementation stores Q and KV separately.
        converted = {}
        for raw_key, value in state.items():
            patch_match = re.fullmatch(
                r"layers\.(\d)\.0\.(projection|norm)\.(weight|bias)", raw_key
            )
            if patch_match:
                stage = int(patch_match.group(1)) + 1
                component = "proj" if patch_match.group(2) == "projection" else "norm"
                converted[
                    f"patch_embed{stage}.{component}.{patch_match.group(3)}"
                ] = value
                continue
            stage_norm_match = re.fullmatch(
                r"layers\.(\d)\.2\.(weight|bias)", raw_key
            )
            if stage_norm_match:
                stage = int(stage_norm_match.group(1)) + 1
                # The local encoder exposes only the stage-4 output norm.
                if stage == 4:
                    converted[f"norm.{stage_norm_match.group(2)}"] = value
                continue
            block_match = re.fullmatch(
                r"layers\.(\d)\.1\.(\d)\.(.+)", raw_key
            )
            if not block_match:
                continue
            stage = int(block_match.group(1)) + 1
            block = int(block_match.group(2))
            suffix = block_match.group(3)
            local_prefix = f"block{stage}.{block}."
            direct_replacements = {
                "attn.attn.out_proj.": "attn.proj.",
                "ffn.layers.0.": "mlp.fc1.",
                "ffn.layers.1.": "mlp.dwconv.",
                "ffn.layers.4.": "mlp.fc2.",
            }
            if suffix.startswith("attn.attn.in_proj_"):
                parameter = suffix.rsplit("_", 1)[-1]
                q_key = f"{local_prefix}attn.q.{parameter}"
                kv_key = f"{local_prefix}attn.kv.{parameter}"
                q_size = target[q_key].shape[0]
                converted[q_key] = value[:q_size]
                converted[kv_key] = value[q_size:]
                continue
            local_suffix = suffix
            for source_prefix, target_prefix in direct_replacements.items():
                if suffix.startswith(source_prefix):
                    local_suffix = target_prefix + suffix[len(source_prefix) :]
                    break
            local_key = local_prefix + local_suffix
            if local_key in target:
                if (
                    value.ndim == 4
                    and value.shape[-2:] == (1, 1)
                    and target[local_key].ndim == 2
                ):
                    value = value[:, :, 0, 0]
                converted[local_key] = value
        matched = {
            key: value
            for key, value in converted.items()
            if key in target
            and hasattr(value, "shape")
            and value.shape == target[key].shape
        }
        checkpoint_format = "torchvision_layers"
    if not matched:
        raise ValueError(
            f"No encoder tensors from {checkpoint_path} match the local MiT encoder"
        )
    missing, unexpected = encoder.load_state_dict(matched, strict=False)
    return {
        "matched_encoder_tensors": len(matched),
        "encoder_tensor_count": len(target),
        "missing_encoder_tensors": len(missing),
        "missing_encoder_keys": list(missing),
        "unexpected_encoder_tensors": len(unexpected),
        "checkpoint_format": checkpoint_format,
    }


def create_model(config: dict, model_name: str, device: torch.device):
    model_cfg = config["model"]
    num_classes = int(model_cfg["num_classes"])
    if model_name == "unet":
        return UNetBaseline(num_classes=num_classes).to(device), {"pretrained": False}
    if model_name == "deeplabv3plus":
        return DeepLabV3PlusBaseline(num_classes=num_classes).to(device), {
            "pretrained": False
        }
    if model_name == "pspnet":
        return PSPNetBaseline(num_classes=num_classes).to(device), {"pretrained": False}
    if model_name == "segnext_tiny":
        model = create_segnext_tiny(
            num_classes=model_cfg["num_classes"],
            drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        )
        pretraining = model.load_backbone_pretraining(
            PROJECT_ROOT / model_cfg["pretrained"]
        )
        return model.to(device), pretraining
    base = create_segformer(
        model_size=model_cfg["model_size"],
        num_classes=model_cfg["num_classes"],
        img_size=config["data"]["image_size"],
        pretrained=False,
        drop_rate=model_cfg.get("dropout_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )
    pretraining = load_encoder_pretraining(
        base, PROJECT_ROOT / model_cfg["pretrained"]
    )
    if model_name == "segformer":
        model = base
    else:
        model = create_enhanced_segformer(
            base,
            model_cfg.get("use_edge_enhancement", True),
            model_cfg.get("use_color_attention", True),
            model_cfg.get("use_green_aux_head", True),
            model_cfg.get("edge_enhancement_version", "v1"),
            model_cfg.get("edge_dilations", (1, 2, 4)),
            model_cfg.get("cam_strength", 1.0),
        )
        if model_cfg.get("enhancement_group_norm", False):
            if getattr(model, "edge_enhance", None) is not None:
                replace_batch_norm_with_group_norm(model.edge_enhance)
            if getattr(model, "color_attention", None) is not None:
                replace_batch_norm_with_group_norm(model.color_attention)
        residual_blend = float(model_cfg.get("enhancement_residual_blend", 1.0))
        if residual_blend < 1.0:
            model = StabilizedEnhancedModel(model, residual_blend)
    return model.to(device), pretraining


def make_criterion(config: dict, model_name: str):
    loss_cfg = config["loss"]
    if model_name == "segnext_tiny":
        return CrossEntropyBaseline(loss_cfg["class_weights"])
    use_aux = model_name == "ecsegformer" and config["model"].get(
        "use_green_aux_head", False
    )
    return CombinedFocalDiceLoss(
        alpha=loss_cfg["class_weights"],
        gamma=loss_cfg["focal_gamma"],
        sparse_class_weights={
            1: loss_cfg["red_pixel_weight"],
            2: loss_cfg["green_pixel_weight"],
        },
        focal_weight=loss_cfg["focal_weight"],
        dice_weight=loss_cfg["dice_weight"],
        iou_weight=loss_cfg["iou_weight"],
        use_green_aux_loss=use_aux,
        green_aux_weight=loss_cfg["green_aux_weight"] if use_aux else 0.0,
        green_aux_pos_weight=loss_cfg["green_aux_pos_weight"],
        iou_class_weights=loss_cfg["iou_class_weights"],
    )


class CrossEntropyDiceLoss(torch.nn.Module):
    """Standard stable baseline loss; no ECSegFormer-specific auxiliary terms."""

    def __init__(self, class_weights) -> None:
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, target, _unused_aux=None):
        cross_entropy = torch.nn.functional.cross_entropy(
            logits, target, weight=self.class_weights
        )
        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = torch.nn.functional.one_hot(
            target, num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).to(probabilities.dtype)
        intersection = (probabilities * target_one_hot).sum(dim=(0, 2, 3))
        denominator = (probabilities + target_one_hot).sum(dim=(0, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        return cross_entropy + dice_loss


class CrossEntropyBaseline(torch.nn.Module):
    """Baseline-compatible CE loss that accepts the trainer's optional aux output."""

    def __init__(self, class_weights) -> None:
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, target, _unused_aux=None):
        return torch.nn.functional.cross_entropy(logits, target, weight=self.class_weights)


def create_optimizer(model, config: dict, model_name: str):
    training_cfg = config["training"]
    weight_decay = training_cfg["weight_decay"]
    base_lr = float(training_cfg["learning_rate"])
    differential_cfg = training_cfg.get("differential_learning_rates")
    if model_name != "ecsegformer" or not differential_cfg:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )
        for index, group in enumerate(optimizer.param_groups):
            group["base_lr"] = base_lr
            group["group_name"] = f"group_{index}"
        return optimizer

    grouped_parameters = {
        "encoder": [],
        "decoder": [],
        "enhancements": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "original_model.encoder" in name:
            grouped_parameters["encoder"].append(parameter)
        elif "original_model" in name:
            grouped_parameters["decoder"].append(parameter)
        else:
            grouped_parameters["enhancements"].append(parameter)

    learning_rates = {
        "encoder": float(differential_cfg.get("encoder", base_lr)),
        "decoder": float(differential_cfg.get("decoder", base_lr)),
        "enhancements": float(differential_cfg.get("enhancements", base_lr)),
    }
    parameter_groups = []
    for group_name, parameters in grouped_parameters.items():
        if parameters:
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": learning_rates[group_name],
                    "base_lr": learning_rates[group_name],
                    "group_name": group_name,
                }
            )
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def set_warmup_cosine_learning_rates(
    optimizer,
    epoch: int,
    total_epochs: int,
    scheduler_config: dict,
) -> None:
    warmup_epochs = int(scheduler_config.get("warmup_epochs", 5))
    warmup_start_factor = float(
        scheduler_config.get("warmup_start_factor", 0.2)
    )
    minimum_factor = float(scheduler_config.get("minimum_factor", 0.05))
    decay_epochs = min(
        total_epochs,
        int(scheduler_config.get("decay_epochs", total_epochs)),
    )
    if epoch < warmup_epochs:
        denominator = max(1, warmup_epochs - 1)
        factor = warmup_start_factor + (
            1.0 - warmup_start_factor
        ) * epoch / denominator
    else:
        denominator = max(1, decay_epochs - warmup_epochs - 1)
        progress = min(1.0, (epoch - warmup_epochs) / denominator)
        factor = minimum_factor + 0.5 * (1.0 - minimum_factor) * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", group["lr"]))
        group["lr"] = base_lr * factor


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    gradient_clip_norm=0.0,
    max_batches=None,
    edge_aux_config=None,
):
    training = optimizer is not None
    model.train(training)
    meter = ConfusionMatrixMeter(5)
    total_loss = 0.0
    total_edge_loss = 0.0
    processed_batches = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, masks, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, aux_logits, edge_aux_logits = unwrap_outputs(model(images))
            loss = criterion(logits, masks, aux_logits)
            edge_loss = None
            edge_aux_weight = float(
                (edge_aux_config or {}).get("weight", 0.0)
            )
            if edge_aux_logits is not None and edge_aux_weight > 0.0:
                edge_loss = supervised_boundary_loss(
                    edge_aux_logits,
                    masks,
                    num_classes=logits.shape[1],
                    pos_weight=float(
                        edge_aux_config.get("pos_weight", 4.0)
                    ),
                    bce_weight=float(
                        edge_aux_config.get("bce_weight", 0.5)
                    ),
                    dice_weight=float(
                        edge_aux_config.get("dice_weight", 0.5)
                    ),
                )
                loss = loss + edge_aux_weight * edge_loss
            if training:
                loss.backward()
                # The lightweight MSCAN baseline uses a very small dataset and
                # large class weights; protect the isolated baseline run from
                # a single non-finite gradient corrupting all later epochs.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        torch.nan_to_num_(parameter.grad, nan=0.0, posinf=1e4, neginf=-1e4)
                        parameter.grad.clamp_(-10.0, 10.0)
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                optimizer.step()
                # Keep a failed numerical update from contaminating batch-norm
                # running statistics or the next validation pass.
                with torch.no_grad():
                    for tensor in list(model.parameters()) + list(model.buffers()):
                        if tensor.is_floating_point():
                            torch.nan_to_num_(tensor, nan=0.0, posinf=1e4, neginf=-1e4)
                            if isinstance(tensor, torch.nn.Parameter):
                                tensor.clamp_(-10.0, 10.0)
            total_loss += float(loss.detach().cpu())
            if edge_loss is not None:
                total_edge_loss += float(edge_loss.detach().cpu())
            processed_batches += 1
            predictions = torch.argmax(logits.detach(), dim=1)
            meter.update(predictions.cpu().numpy(), masks.cpu().numpy())
    if not processed_batches:
        raise ValueError("No batches were processed")
    metrics = meter.compute()
    metrics["loss"] = total_loss / processed_batches
    metrics["edge_aux_loss"] = total_edge_loss / processed_batches
    metrics["processed_batches"] = processed_batches
    return metrics


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "segformer",
            "ecsegformer",
            "segnext_tiny",
            "unet",
            "deeplabv3plus",
            "pspnet",
        ],
        required=True,
    )
    parser.add_argument(
        "--config", default="config/experiments/main_comparison.yaml"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}; use --overwrite only for smoke tests"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = (PROJECT_ROOT / config["data"]["manifest"]).resolve()
    uses_pretrained = args.model in {"segformer", "ecsegformer", "segnext_tiny"}
    pretrained = (PROJECT_ROOT / config["model"]["pretrained"]).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if uses_pretrained and not pretrained.is_file():
        raise FileNotFoundError(pretrained)

    deterministic = bool(config["training"].get("deterministic", True))
    set_reproducibility(args.seed, deterministic)
    requested_device = config.get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    train_transform, eval_transform = create_transforms(config["data"]["image_size"])
    datasets = {
        "train": ManifestDataset(PROJECT_ROOT, manifest, "train", train_transform),
        "val": ManifestDataset(PROJECT_ROOT, manifest, "val", eval_transform),
        "test": ManifestDataset(PROJECT_ROOT, manifest, "test", eval_transform),
    }
    sampler = ClassBalancedBatchSampler(
        datasets["train"],
        config["training"]["batch_size"],
        min_red=config["data"].get("sampler_min_red", 1),
        min_green=config["data"].get("sampler_min_green", 1),
        shuffle=True,
        seed=args.seed,
        green_requirement_probability=config["data"].get(
            "sampler_green_requirement_probability", 1.0
        ),
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common_loader = {
        "num_workers": config.get("num_workers", 0),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": config.get("num_workers", 0) > 0,
    }
    train_loader = DataLoader(
        datasets["train"], batch_sampler=sampler, **common_loader
    )
    eval_loader_args = {
        **common_loader,
        "batch_size": config["training"]["batch_size"],
        "shuffle": False,
    }
    val_loader = DataLoader(datasets["val"], **eval_loader_args)
    test_loader = DataLoader(datasets["test"], **eval_loader_args)

    model, pretraining = create_model(config, args.model, device)
    criterion = make_criterion(config, args.model).to(device)
    optimizer = create_optimizer(model, config, args.model)
    scheduler_cfg = config["training"]["scheduler"]
    edge_aux_config = config.get("loss", {}).get("edge_aux", {})
    scheduler_type = scheduler_cfg.get("type", "reduce_on_plateau")
    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
            min_lr=scheduler_cfg["min_lr"],
        )
    elif scheduler_type == "warmup_cosine":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    smoke_test = args.max_train_batches is not None or args.max_eval_batches is not None
    metadata = {
        "status": "running",
        "experiment": config["experiment"]["name"],
        "model": args.model,
        "seed": args.seed,
        "split_id": config["experiment"]["split_id"],
        "manifest_sha256": sha256_file(manifest),
        "config_sha256": sha256_file(config_path),
        "pretrained_sha256": sha256_file(pretrained) if uses_pretrained else None,
        "pretraining_load": pretraining,
        "smoke_test": smoke_test,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "dataset_sizes": {key: len(value) for key, value in datasets.items()},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "started_at_unix": time.time(),
    }
    save_json(run_dir / "run_metadata.json", metadata)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    best_miou = -1.0
    history = []
    epochs = config["training"]["num_epochs"]
    for epoch in range(epochs):
        if scheduler_type == "warmup_cosine":
            set_warmup_cosine_learning_rates(
                optimizer, epoch, epochs, scheduler_cfg
            )
        sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            gradient_clip_norm=config["training"]["gradient_clip_norm"],
            max_batches=args.max_train_batches,
            edge_aux_config=edge_aux_config,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_eval_batches,
            edge_aux_config=edge_aux_config,
        )
        if scheduler is not None:
            scheduler.step(val_metrics["mean_iou"])
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rates": json.dumps(
                {
                    group.get("group_name", f"group_{index}"): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                },
                sort_keys=True,
            ),
            "train_loss": train_metrics["loss"],
            "train_mean_iou": train_metrics["mean_iou"],
            "train_edge_aux_loss": train_metrics["edge_aux_loss"],
            "val_loss": val_metrics["loss"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_edge_aux_loss": val_metrics["edge_aux_loss"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {
            "epoch": epoch,
            "model": args.model,
            "seed": args.seed,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        }
        torch.save(checkpoint, run_dir / "last.pth")
        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            torch.save(checkpoint, run_dir / "best.pth")

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best_checkpoint = torch.load(
        run_dir / "best.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        max_batches=args.max_eval_batches,
        edge_aux_config=edge_aux_config,
    )
    test_metrics.update(
        {
            "dataset_split": "test",
            "model": args.model,
            "seed": args.seed,
            "split_id": config["experiment"]["split_id"],
            "manifest_sha256": metadata["manifest_sha256"],
            "checkpoint": str(run_dir / "best.pth"),
            "smoke_test": smoke_test,
        }
    )
    save_json(run_dir / "test_metrics.json", test_metrics)
    metadata["status"] = "completed"
    metadata["completed_at_unix"] = time.time()
    metadata["best_val_mean_iou"] = best_miou
    metadata["test_mean_iou"] = test_metrics["mean_iou"]
    save_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
