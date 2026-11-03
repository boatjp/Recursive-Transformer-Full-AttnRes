from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import DataConfig, create_dataloaders, prepare_split
from recursive_transformer import ModelConfig, RMSNorm, RecursiveTransformerLM, set_seed


@dataclass(frozen=True)
class TrainConfig:
    experiment_name: str = "recursive-ja"
    residual_type: str = "residual"
    data_dir: str = "tokenizer/output-full"
    output_dir: str = "runs/recursive-ja"
    sequence_length: int = 512
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_epochs: int = 10
    seed: int = 42
    precision: str = "bf16"
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.05
    min_lr_fraction: float = 0.1
    validation_interval: int = 200
    checkpoint_interval: int = 500
    num_workers: int = 0
    pin_memory: bool = False
    model_overrides: dict | None = None

    def __post_init__(self) -> None:
        if self.residual_type not in ("residual", "attnres"):
            raise ValueError("Invalid residual_type")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        for name in ("sequence_length", "micro_batch_size", "gradient_accumulation_steps",
                     "max_epochs", "validation_interval", "checkpoint_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.sequence_length < 2 or self.num_workers < 0:
            raise ValueError("Invalid sequence_length or num_workers")
        for name in ("learning_rate", "eps", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.weight_decay < 0 or not 0 <= self.warmup_fraction <= 1 or not 0 <= self.min_lr_fraction <= 1:
            raise ValueError("Invalid weight decay or scheduler fraction")
        if len(self.betas) != 2 or any(not 0 <= value < 1 for value in self.betas):
            raise ValueError("Invalid AdamW betas")

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        if "betas" in values:
            values["betas"] = tuple(values["betas"])
        return cls(**values)


def optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    seen = set()
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise ValueError(f"Parameter registered twice: {module_name}.{name}")
            seen.add(id(parameter))
            if name == "bias" or isinstance(module, RMSNorm) or module_name == "attnres":
                no_decay.append(parameter)
            elif isinstance(module, (nn.Linear, nn.Embedding)) and name == "weight":
                decay.append(parameter)
            else:
                raise ValueError(f"Unclassified parameter: {module_name}.{name}")
    if seen != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise ValueError("Optimizer parameter coverage is incomplete")
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int,
                 peak_lr: float, warmup_fraction: float, min_lr_fraction: float):
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.warmup_steps = round(total_steps * warmup_fraction)
        self.min_lr_fraction = min_lr_fraction
        self.completed_steps = 0
        self._set_next_lr()

    def rate(self, update_number: int) -> float:
        if self.warmup_steps and update_number <= self.warmup_steps:
            return self.peak_lr * update_number / self.warmup_steps
        progress = (update_number - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        fraction = self.min_lr_fraction + (1 - self.min_lr_fraction) * (
            1 + math.cos(math.pi * min(1.0, progress))) / 2
        return self.peak_lr * fraction

    def _set_next_lr(self) -> None:
        rate = self.rate(min(self.completed_steps + 1, self.total_steps))
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def step(self) -> None:
        self.completed_steps += 1
        self._set_next_lr()

    def state_dict(self) -> dict:
        return {"completed_steps": self.completed_steps, "total_steps": self.total_steps,
                "peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps,
                "min_lr_fraction": self.min_lr_fraction}

    def load_state_dict(self, state: dict) -> None:
        if any(state[name] != getattr(self, name) for name in (
                "total_steps", "peak_lr", "warmup_steps", "min_lr_fraction")):
            raise ValueError("Scheduler configuration differs from checkpoint")
        self.completed_steps = state["completed_steps"]
        self._set_next_lr()


def valid_token_count(batch: dict[str, torch.Tensor]) -> int:
    mask = batch["attention_mask"].bool()
    return int(((batch["labels"][:, 1:] != -100) & mask[:, 1:] & mask[:, :-1]).sum())


def evaluate(model: RecursiveTransformerLM, loader: DataLoader,
             device: torch.device, use_bf16: bool = False) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    weighted_loss, total_tokens = 0.0, 0
    try:
        with torch.inference_mode():
            for batch in loader:
                count = valid_token_count(batch)
                if not count:
                    continue
                moved = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    result = model(**moved)
                if not torch.isfinite(result.loss):
                    raise FloatingPointError("Non-finite validation loss")
                weighted_loss += result.loss.item() * count
                total_tokens += count
    finally:
        model.train(was_training)
    if not total_tokens:
        raise ValueError("Validation has no valid prediction tokens")
    mean = weighted_loss / total_tokens
    return mean, math.exp(mean) if mean < 709 else float("inf")


class Trainer:
    def __init__(self, config: TrainConfig, loaders: dict[str, DataLoader] | None = None,
                 model: RecursiveTransformerLM | None = None, data_identity: dict | None = None,
                 device: torch.device | None = None):
        self.config = config
        set_seed(config.seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = (config.precision == "bf16" and self.device.type == "cuda"
                         and torch.cuda.is_bf16_supported())
        data_config = DataConfig(data_dir=Path(config.data_dir), sequence_length=config.sequence_length,
                                 seed=config.seed)
        self.loaders = loaders or create_dataloaders(
            data_config, config.micro_batch_size, num_workers=config.num_workers,
            pin_memory=config.pin_memory, train_shuffle=False)
        if data_identity is None:
            data_identity = {split: {name: prepare_split(data_config, split)[name] for name in (
                "source_sha256", "tokenizer_sha256", "token_count", "tokenization_method")}
                for split in ("train", "validation", "test")}
        self.data_identity = data_identity
        if model is None:
            overrides = config.model_overrides or {}
            if "residual_type" in overrides or "vocab_size" in overrides:
                raise ValueError("residual_type and vocab_size come from the config and tokenizer")
            model_config = ModelConfig.from_tokenizer(
                Path(config.data_dir) / "tokenizer.model", residual_type=config.residual_type, **overrides)
            model = RecursiveTransformerLM(model_config)
        if model.config.residual_type != config.residual_type:
            raise ValueError("Model residual_type differs from training config")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            optimizer_groups(self.model, config.weight_decay), lr=config.learning_rate,
            betas=config.betas, eps=config.eps)
        batches_per_epoch = len(self.loaders["train"])
        if batches_per_epoch == 0:
            raise ValueError("Training loader has no batches")
        total_updates = config.max_epochs * math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
        self.scheduler = WarmupCosineScheduler(self.optimizer, total_updates, config.learning_rate,
                                               config.warmup_fraction, config.min_lr_fraction)
        self.epoch = 0
        self.batch_cursor = 0
        self.global_step = 0
        self.best_validation_loss = float("inf")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.monotonic()

    def _epoch_loader(self) -> DataLoader:
        source = self.loaders["train"]
        generator = torch.Generator().manual_seed(self.config.seed + self.epoch)
        order = torch.randperm(len(source.dataset), generator=generator).tolist()
        batch_size = self.config.micro_batch_size
        if source.drop_last:
            order = order[:len(order) // batch_size * batch_size]
        order = order[self.batch_cursor * batch_size:]
        return DataLoader(Subset(source.dataset, order), batch_size=batch_size, shuffle=False,
                          num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
                          drop_last=False)

    def _log(self, values: dict) -> None:
        record = {"experiment_name": self.config.experiment_name,
                  "residual_type": self.config.residual_type, "seed": self.config.seed,
                  "config": asdict(self.config), **values}
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(encoded, flush=True)
        with (self.output_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")

    def save_checkpoint(self, name: str = "last.pt") -> Path:
        path = self.output_dir / name
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "scaler": None,
                 "epoch": self.epoch, "batch_cursor": self.batch_cursor,
                 "global_step": self.global_step, "config": asdict(self.config),
                 "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "data_identity": self.data_identity,
                 "shuffle": {"seed": self.config.seed, "epoch": self.epoch,
                             "batch_cursor": self.batch_cursor},
                 "best_validation_loss": self.best_validation_loss}
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["config"] != asdict(self.config) or state["data_identity"] != self.data_identity:
            raise ValueError("Checkpoint config or data identity differs")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = state["epoch"]
        self.batch_cursor = state["batch_cursor"]
        self.global_step = state["global_step"]
        self.best_validation_loss = state["best_validation_loss"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def _start_step_measurement(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.monotonic()

    def _finish_step_measurement(self, started: float) -> tuple[float, int]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device)
        else:
            peak_memory = 0
        return max(time.monotonic() - started, 1e-9), peak_memory

    def run(self, max_steps: int | None = None) -> None:
        limit = max_steps if max_steps is not None else self.scheduler.total_steps
        while self.epoch < self.config.max_epochs and self.global_step < limit:
            iterator = iter(self._epoch_loader())
            remaining = len(self.loaders["train"]) - self.batch_cursor
            while remaining and self.global_step < limit:
                group = []
                for _ in range(min(self.config.gradient_accumulation_steps, remaining)):
                    group.append(next(iterator))
                counts = [valid_token_count(batch) for batch in group]
                token_count = sum(counts)
                if token_count == 0:
                    raise ValueError("Accumulation group has no valid prediction tokens")
                started = self._start_step_measurement()
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                for batch, count in zip(group, counts):
                    if not count:
                        continue
                    moved = {name: value.to(self.device, non_blocking=True) for name, value in batch.items()}
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                        enabled=self.use_bf16):
                        loss = self.model(**moved).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    (loss * (count / token_count)).backward()
                    weighted_loss += loss.item() * count / token_count
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm,
                                                     error_if_nonfinite=True)
                self.optimizer.step()
                used_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                step_elapsed, peak_memory = self._finish_step_measurement(started)
                self.global_step += 1
                self.batch_cursor += len(group)
                remaining -= len(group)
                finished_epoch = self.batch_cursor == len(self.loaders["train"])
                if finished_epoch:
                    self.epoch += 1
                    self.batch_cursor = 0
                validation_loss = None
                perplexity = None
                if self.global_step % self.config.validation_interval == 0 or finished_epoch:
                    validation_loss, perplexity = evaluate(
                        self.model, self.loaders["validation"], self.device, self.use_bf16)
                    if validation_loss < self.best_validation_loss:
                        self.best_validation_loss = validation_loss
                        self.save_checkpoint("best.pt")
                self._log({"epoch": self.epoch, "optimizer_step": self.global_step,
                           "train_loss": weighted_loss, "validation_loss": validation_loss,
                           "validation_perplexity": perplexity, "learning_rate": used_lr,
                           "gradient_norm_before_clipping": float(grad_norm),
                           "tokens_per_second": token_count / step_elapsed,
                           "gpu_peak_memory_bytes": peak_memory,
                           "elapsed_seconds": time.monotonic() - self.started_at})
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint("last.pt")
            if self.batch_cursor == 0 and self.epoch < self.config.max_epochs:
                continue
        self.save_checkpoint("last.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a recursive Japanese causal LM")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = TrainConfig.from_json(args.config)
    trainer = Trainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    trainer.run()


if __name__ == "__main__":
    main()
