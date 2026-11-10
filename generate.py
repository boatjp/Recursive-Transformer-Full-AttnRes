from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as sentencepiece
import torch
from torch import Tensor

from inference import load_model_checkpoint


def choose_token(logits: Tensor, *, sampling: bool = True, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.95,
                 generator: torch.Generator | None = None) -> int:
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive, top_k nonnegative, and top_p in (0, 1]")
    if not sampling:
        return int(logits.argmax())
    scaled = logits.float() / temperature
    if top_k:
        threshold = torch.topk(scaled, min(top_k, scaled.numel())).values[-1]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    ranked, indices = scaled.sort(descending=True)
    probabilities = ranked.softmax(dim=-1)
    excluded = probabilities.cumsum(dim=-1) - probabilities > top_p
    ranked = ranked.masked_fill(excluded, float("-inf"))
    filtered = torch.full_like(scaled, float("-inf")).scatter(0, indices, ranked)
    if not torch.isfinite(filtered).any():
        raise FloatingPointError("No finite sampling logits")
    return int(torch.multinomial(filtered.softmax(dim=-1), 1, generator=generator))


def generate_token_ids(model, tokenizer: sentencepiece.SentencePieceProcessor, prompt: str,
                       *, max_new_tokens: int = 256, temperature: float = 0.8,
                       top_k: int = 50, top_p: float = 0.95,
                       repetition_penalty: float = 1.0, sampling: bool = True,
                       seed: int | None = None, use_cache: bool = False) -> list[int]:
    if use_cache:
        raise NotImplementedError("KV Cache is not implemented for the recursive model")
    if max_new_tokens < 0 or repetition_penalty <= 0:
        raise ValueError("max_new_tokens must be nonnegative and repetition_penalty positive")
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid sampling settings")
    tokens = tokenizer.encode(prompt, out_type=int)
    if not tokens:
        raise ValueError("Prompt must produce at least one token")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    eos_id = tokenizer.eos_id()
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            try:
                input_ids = torch.tensor([tokens], device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                    logits = model(input_ids)[0, -1].float()
            except torch.OutOfMemoryError as exc:
                raise RuntimeError(f"Generation ran out of memory at context length {len(tokens)}") from exc
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite generation logits")
            if repetition_penalty != 1.0:
                repeated = torch.tensor(list(set(tokens)), device=device)
                selected = logits[repeated]
                logits[repeated] = torch.where(selected < 0, selected * repetition_penalty,
                                               selected / repetition_penalty)
            next_token = choose_token(logits, sampling=sampling, temperature=temperature,
                                      top_k=top_k, top_p=top_p, generator=generator)
            if next_token == eos_id:
                break
            tokens.append(next_token)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Japanese text from a saved model")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/recursive-ja-residual/best.pt"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--greedy", action="store_true")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--use-cache", dest="use_cache", action="store_true")
    cache.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=False)
    args = parser.parse_args()
    if args.use_cache:
        parser.error("KV Cache is not implemented for the recursive model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_model_checkpoint(args.checkpoint, device)
    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=str(Path(config.data_dir) / "tokenizer.model"))
    tokens = generate_token_ids(model, tokenizer, args.prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_k=args.top_k,
                                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                                sampling=not args.greedy, seed=args.seed)
    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
