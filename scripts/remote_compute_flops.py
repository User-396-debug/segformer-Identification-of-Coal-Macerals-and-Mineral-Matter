"""Compute static parameter counts and FLOPs for the final B1 configurations."""

import json
import os
import sys
from pathlib import Path

import torch
import yaml
from fvcore.nn import FlopCountAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.segformer_model import create_segformer
from models.segformer_enhancements import create_enhanced_segformer


CONFIG_PATH = Path(
    os.environ.get("ECS_CONFIG", ROOT / "config/config_ecsegformer.yaml")
)
CHECKPOINT_PATH = Path(
    os.environ.get("ECS_CHECKPOINT", ROOT / "checkpoints/best_model.pth")
)
OUTPUT_PATH = Path(
    os.environ.get(
        "ECS_FLOPS_OUTPUT",
        ROOT / "evaluation_results/flops_segformer_ecsegformer_512.json",
    )
)


class LogitWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        output = self.model(image)
        return output[0] if isinstance(output, (tuple, list)) else output


def trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def analyse(name, model, input_tensor):
    wrapped = LogitWrapper(model).eval()
    with torch.inference_mode():
        output = wrapped(input_tensor)
    analysis = FlopCountAnalysis(wrapped, input_tensor).unsupported_ops_warnings(False)
    flops = int(analysis.total())
    unsupported = dict(analysis.unsupported_ops())
    return {
        "model": name,
        "output_shape": list(output.shape),
        "trainable_parameters": trainable_parameters(model),
        "flops": flops,
        "gflops": flops / 1e9,
        "unsupported_ops": unsupported,
    }


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = config["model"]
    image_size = int(config["data"]["image_size"])
    base = create_segformer(
        model_size=model_config["model_size"],
        num_classes=model_config["num_classes"],
        img_size=image_size,
        pretrained=False,
        drop_rate=model_config.get("dropout_rate", 0.0),
        drop_path_rate=model_config.get("drop_path_rate", 0.0),
    ).cpu().eval()
    enhanced = create_enhanced_segformer(
        create_segformer(
            model_size=model_config["model_size"],
            num_classes=model_config["num_classes"],
            img_size=image_size,
            pretrained=False,
            drop_rate=model_config.get("dropout_rate", 0.0),
            drop_path_rate=model_config.get("drop_path_rate", 0.0),
        ),
        model_config.get("use_edge_enhancement", True),
        model_config.get("use_color_attention", True),
        model_config.get("use_green_aux_head", False),
    ).cpu().eval()

    checkpoint_info = {"checkpoint_path": str(CHECKPOINT_PATH), "loaded": False}
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = enhanced.load_state_dict(state, strict=False)
        checkpoint_info.update({
            "loaded": True,
            "missing_key_count": len(missing),
            "unexpected_key_count": len(unexpected),
        })

    input_tensor = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    base_result = analyse("SegFormer-B1", base, input_tensor)
    enhanced_result = analyse("ECSegFormer", enhanced, input_tensor)
    increase = {
        "parameter_difference": enhanced_result["trainable_parameters"] - base_result["trainable_parameters"],
        "parameter_increase_percent": 100 * (
            enhanced_result["trainable_parameters"] / base_result["trainable_parameters"] - 1
        ),
        "flop_difference": enhanced_result["flops"] - base_result["flops"],
        "flop_increase_percent": 100 * (enhanced_result["flops"] / base_result["flops"] - 1),
    }
    result = {
        "protocol": {
            "input_shape": [1, 3, image_size, image_size],
            "batch_size": 1,
            "mode": "eval",
            "device": "cpu",
            "tool": "fvcore.nn.FlopCountAnalysis",
            "counting_scope": "single forward pass; standard fvcore operator coverage",
        },
        "config": model_config,
        "checkpoint": checkpoint_info,
        "SegFormer": base_result,
        "ECSegFormer": enhanced_result,
        "difference": increase,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
