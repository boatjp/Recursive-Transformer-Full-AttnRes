from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import torch

from recursive_transformer import ModelConfig, RecursiveTransformerLM
from train import TrainConfig


def load_model_checkpoint(checkpoint: str | Path, device: torch.device) -> tuple[RecursiveTransformerLM, TrainConfig, dict]:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = TrainConfig(**{**state["config"], "betas": tuple(state["config"]["betas"])})
        tokenizer_path = Path(config.data_dir) / "tokenizer.model"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        identities = state["data_identity"]
        if not identities or any(identity.get("tokenizer_sha256") != digest
                                 for identity in identities.values()):
            raise ValueError("Checkpoint tokenizer identity does not match tokenizer.model")
        overrides = config.model_overrides or {}
        if "residual_type" in overrides or "vocab_size" in overrides:
            raise ValueError("Invalid model overrides in checkpoint")
        model_config = ModelConfig.from_tokenizer(tokenizer_path,
                                                  residual_type=config.residual_type, **overrides)
        model = RecursiveTransformerLM(model_config)
        model.load_state_dict(state["model"], strict=True)
    except (KeyError, TypeError, AttributeError, RuntimeError, pickle.UnpicklingError, EOFError) as exc:
        raise ValueError(f"Invalid or incompatible checkpoint: {path}") from exc
    return model.to(device).eval(), config, identities
