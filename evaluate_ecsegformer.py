"""Official test-set evaluation for ECSegFormer.

Metrics are computed once from the complete test-set confusion matrix.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from models.segformer_model import create_segformer
from models.segformer_enhancements import create_enhanced_segformer
from utils.improved_data_loader import ImprovedCoalMicroscopyDataset
from utils.data_loader_utils import create_balanced_transforms
from utils.segmentation_metrics import ConfusionMatrixMeter


def unwrap_logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def build_model(config, checkpoint_path, device):
    model_cfg = config["model"]
    base = create_segformer(
        model_size=model_cfg["model_size"], num_classes=model_cfg["num_classes"],
        img_size=config["data"]["image_size"], pretrained=False,
        drop_rate=model_cfg.get("dropout_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )
    model = create_enhanced_segformer(
        base, model_cfg.get("use_edge_enhancement", True),
        model_cfg.get("use_color_attention", True),
        model_cfg.get("use_green_aux_head", False),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    if missing:
        print(f"Warning: {len(missing)} missing keys: {missing[:5]}")
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_ecsegformer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="evaluation_results/ecsegformer_test_metrics.json")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data = config["data"]
    _, test_transform = create_balanced_transforms(enable_green_augmentation=False)
    dataset = ImprovedCoalMicroscopyDataset(data["test_images"], data["test_masks"], test_transform, is_training=False)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                        num_workers=config.get("num_workers", 0), pin_memory=device.type == "cuda")
    model = build_model(config, args.checkpoint, device)
    meter = ConfusionMatrixMeter(config["model"]["num_classes"])
    with torch.inference_mode():
        for images, masks in loader:
            predictions = torch.argmax(unwrap_logits(model(images.to(device))), dim=1)
            meter.update(predictions.cpu().numpy(), masks.numpy())
    results = meter.compute()
    results.update({"dataset_split": "test", "num_images": len(dataset), "checkpoint": str(args.checkpoint)})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
