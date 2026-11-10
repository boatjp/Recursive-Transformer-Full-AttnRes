from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from data import DataConfig, create_dataloaders, prepare_split
from inference import load_model_checkpoint
from train import evaluate as evaluate_loss


def evaluate_checkpoint(checkpoint: str | Path, batch_size: int = 4,
                        device: torch.device | None = None) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, identities = load_model_checkpoint(checkpoint, device)
    data_config = DataConfig(data_dir=Path(config.data_dir),
                             sequence_length=config.sequence_length, seed=config.seed)
    metadata = prepare_split(data_config, "test")
    identity = identities.get("test")
    if identity is None or any(identity.get(name) != metadata[name] for name in (
            "source_sha256", "tokenizer_sha256", "token_count", "tokenization_method")):
        raise ValueError("Checkpoint test data identity does not match current data")
    loader = create_dataloaders(data_config, batch_size=batch_size,
                                num_workers=config.num_workers, pin_memory=config.pin_memory)["test"]
    use_bf16 = config.precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    loss, perplexity = evaluate_loss(model, loader, device, use_bf16)
    if not math.isfinite(perplexity):
        raise FloatingPointError("Test perplexity is non-finite")
    return {"checkpoint": str(Path(checkpoint).resolve()), "residual_type": config.residual_type,
            "test_loss": loss, "test_perplexity": perplexity,
            "test_tokens": metadata["token_count"], "sequence_length": config.sequence_length,
            "batch_size": batch_size, "precision": "bf16" if use_bf16 else "fp32"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the test split")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/recursive-ja-residual/best.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.batch_size)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    output = args.output or args.checkpoint.parent / "test_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
