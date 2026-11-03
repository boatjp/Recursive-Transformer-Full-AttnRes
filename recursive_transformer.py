from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16384
    hidden_size: int = 640
    num_steps: int = 16
    num_heads: int = 10
    intermediate_size: int = 10240
    embedding_rank: int = 192
    residual_type: str = "residual"
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    initializer_std: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    unk_token_id: int | None = None

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "num_steps", "num_heads",
                     "intermediate_size", "embedding_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads or self.head_dim % 2:
            raise ValueError("hidden_size must divide into even-sized heads")
        if self.residual_type not in ("residual", "attnres"):
            raise ValueError("residual_type must be residual or attnres")
        for name in ("norm_eps", "rope_theta", "initializer_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            token_id = getattr(self, name)
            if token_id is not None and not 0 <= token_id < self.vocab_size:
                raise ValueError(f"{name} is outside the vocabulary")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_tokenizer(cls, model_path: str | Path, **kwargs) -> ModelConfig:
        import sentencepiece as sentencepiece

        processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        metadata = {"vocab_size": processor.vocab_size()}
        for field, method in (("pad_token_id", "pad_id"), ("bos_token_id", "bos_id"),
                              ("eos_token_id", "eos_id"), ("unk_token_id", "unk_id")):
            token_id = getattr(processor, method)()
            metadata[field] = token_id if token_id >= 0 else None
        if metadata.keys() & kwargs.keys():
            raise ValueError("Tokenizer vocabulary and special IDs cannot be overridden")
        return cls(**metadata, **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rms_normalize(hidden: Tensor, eps: float) -> Tensor:
    values = hidden.float()
    return (values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)).to(hidden.dtype)


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, hidden: Tensor) -> Tensor:
        normalized = rms_normalize(hidden, self.eps) * self.weight.to(hidden.dtype)
        return normalized if self.bias is None else normalized + self.bias.to(hidden.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, hidden: Tensor, positions: Tensor) -> Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            inverse = self.theta ** (-torch.arange(
                0, self.head_dim, 2, device=hidden.device, dtype=torch.float32
            ) / self.head_dim)
            angles = positions.float().unsqueeze(-1) * inverse
            cosine = angles.cos().unsqueeze(1)
            sine = angles.sin().unsqueeze(1)
            even, odd = hidden.float()[..., 0::2], hidden.float()[..., 1::2]
            rotated = torch.stack((even * cosine - odd * sine,
                                   even * sine + odd * cosine), dim=-1)
        return rotated.flatten(-2).to(hidden.dtype)


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate = nn.Linear(config.hidden_size, config.num_heads, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, hidden: Tensor, query_norm: RMSNorm, key_norm: RMSNorm,
                positions: Tensor, allowed: Tensor, valid: Tensor) -> Tensor:
        batch, length, width = hidden.shape
        query, key, value = [projection(hidden).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2) for projection in (self.query, self.key, self.value)]
        query = self.rope(query_norm(query), positions)
        key = self.rope(key_norm(key), positions)
        attended = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=0.0
        ).transpose(1, 2)
        attended = attended * self.gate(hidden).sigmoid().unsqueeze(-1)
        return self.output(attended.reshape(batch, length, width)) * valid.unsqueeze(-1)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(functional.relu(self.up(hidden)).square())


class SharedBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = CausalAttention(config)
        self.ffn = FeedForward(config)


class FullAttentionResiduals(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(2 * config.num_steps + 1, config.hidden_size)) #qyeryを33個に修正
        self.eps = config.norm_eps

    def forward(self, history: list[Tensor], query_index: int) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=history[0].device.type, enabled=False):
            values = torch.stack(history, dim=2).float()
            keys = rms_normalize(values, self.eps)
            scores = torch.einsum("btdh,h->btd", keys, self.queries[query_index].float())
            weights = scores.softmax(dim=-1)
            mixed = torch.einsum("btd,btdh->bth", weights, values)
        return mixed.to(history[0].dtype), weights


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    depth_weights: tuple[Tensor, ...] = ()


class RecursiveTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_rank,
                                      padding_idx=config.pad_token_id)
        self.input_projection = nn.Linear(config.embedding_rank, config.hidden_size, bias=False)
        self.block = SharedBlock(config)
        self.attn_norms = nn.ModuleList([
            RMSNorm(config.hidden_size, config.norm_eps, bias=True) for _ in range(config.num_steps)
        ])
        self.ffn_norms = nn.ModuleList([
            RMSNorm(config.hidden_size, config.norm_eps, bias=True) for _ in range(config.num_steps)
        ])
        self.query_norms = nn.ModuleList([
            RMSNorm(config.head_dim, config.norm_eps) for _ in range(config.num_steps)
        ])
        self.key_norms = nn.ModuleList([
            RMSNorm(config.head_dim, config.norm_eps) for _ in range(config.num_steps)
        ])
        self.attnres = FullAttentionResiduals(config) if config.residual_type == "attnres" else None
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.head_projection = nn.Linear(config.hidden_size, config.embedding_rank, bias=False)
        self.lm_head = nn.Linear(config.embedding_rank, config.vocab_size, bias=False)
        self.apply(self._initialize)
        output_std = config.initializer_std / math.sqrt(2 * config.num_steps)
        nn.init.normal_(self.block.attention.output.weight, std=output_std)
        nn.init.normal_(self.block.ffn.down.weight, std=output_std)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.initializer_std)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                labels: Tensor | None = None, return_details: bool = False
                ) -> Tensor | CausalLMOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty [batch, sequence] shape")
        if attention_mask is None:
            valid = (torch.ones_like(input_ids, dtype=torch.bool) if self.config.pad_token_id is None
                     else input_ids.ne(self.config.pad_token_id))
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask must contain only 0 and 1")
            valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        length = input_ids.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.input_projection(self.embedding(input_ids)) * valid.unsqueeze(-1)
        history = [hidden]
        depth_weights = []
        for step in range(self.config.num_steps):
            if self.attnres is not None:
                hidden, weights = self.attnres(history, 2 * step)
                if return_details:
                    depth_weights.append(weights)
            attended = self.block.attention(
                self.attn_norms[step](hidden), self.query_norms[step], self.key_norms[step],
                positions, allowed, valid
            )
            if self.attnres is None:
                hidden = hidden + attended
            else:
                history.append(attended)
                hidden, weights = self.attnres(history, 2 * step + 1)
                if return_details:
                    depth_weights.append(weights)
            transformed = self.block.ffn(self.ffn_norms[step](hidden)) * valid.unsqueeze(-1)
            if self.attnres is None:
                hidden = hidden + transformed
            else:
                history.append(transformed)
        if self.attnres is not None:
            hidden, weights = self.attnres(history, -1)
            if return_details:
                depth_weights.append(weights)
        logits = self.lm_head(self.head_projection(self.final_norm(hidden)))
        loss = None
        if labels is not None:
            targets = labels[:, 1:].clone()
            targets.masked_fill_(~(valid[:, :-1] & valid[:, 1:]), -100)
            losses = functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                targets.reshape(-1), ignore_index=-100, reduction="sum"
            ) if length > 1 else logits.float().sum() * 0.0
            loss = losses / targets.ne(-100).sum().clamp_min(1)
        if return_details or labels is not None:
            return CausalLMOutput(logits, loss, tuple(depth_weights))
        return logits

    def parameter_counts(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        counts = {
            "attention": count(self.block.attention),
            "ffn": count(self.block.ffn),
            "embedding": count(self.embedding) + count(self.input_projection),
            "lm_head": count(self.head_projection) + count(self.lm_head),
            "step_norms_and_biases": sum(count(module) for module in (
                self.attn_norms, self.ffn_norms, self.query_norms, self.key_norms
            )),
            "final_norm": count(self.final_norm),
            "attnres_queries": count(self.attnres) if self.attnres is not None else 0,
        }
        counts["total"] = sum(counts.values())
        return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model sizes without training")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/output-full/tokenizer.model"))
    args = parser.parse_args()
    for residual_type in ("residual", "attnres"):
        config = ModelConfig.from_tokenizer(args.tokenizer, residual_type=residual_type)
        with torch.device("meta"):
            model = RecursiveTransformerLM(config)
        print(json.dumps({"residual_type": residual_type, "vocab_size": config.vocab_size,
                          "special_ids": {name: getattr(config, name) for name in (
                              "pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id")},
                          "parameters": model.parameter_counts()}, indent=2))


if __name__ == "__main__":
    main()
