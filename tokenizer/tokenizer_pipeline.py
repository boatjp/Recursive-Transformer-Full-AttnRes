#!/usr/bin/env python3
"""Build and evaluate a Japanese SentencePiece Unigram tokenizer from FineWiki."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import sentencepiece as spm
from datasets import load_dataset
from sudachipy import dictionary, tokenizer


LOGGER = logging.getLogger("finewiki_tokenizer")
SPLITS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}
EXPECTED_FIELDS = {"text", "id", "title", "page_id", "in_language"}
# SudachiPy rejects inputs above 49,149 bytes. Keep some headroom in case its
# internal representation adds bytes, and split only the text passed to Sudachi.
SUDACHI_MAX_INPUT_BYTES = 48_000
SUDACHI_SENTENCE_BOUNDARIES = frozenset("。！？!?\n")


@dataclass(frozen=True)
class Config:
    output_dir: str
    mode: str
    target_train_words: int
    vocab_size: int
    seed: int
    shuffle_buffer_size: int
    min_document_words: int
    low_frequency_threshold: int
    character_coverage: float = 0.9995
    normalization_rule_name: str = "nmt_nfkc"
    byte_fallback: bool = True
    dataset: str = "HuggingFaceFW/finewiki"
    dataset_config: str = "ja"
    dataset_split: str = "train"
    split_mode: str = "SudachiPy SplitMode.C"
    split_ratios: dict[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_ratios", dict(SPLIT_RATIOS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Japanese FineWiki corpus and SentencePiece tokenizer."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--seed", type=int, default=20251031)
    parser.add_argument("--target-train-words", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--shuffle-buffer-size", type=int)
    parser.add_argument("--min-document-words", type=int, default=50)
    parser.add_argument("--low-frequency-threshold", type=int, default=5)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    smoke = args.mode == "smoke"
    return Config(
        output_dir=str(args.output_dir.resolve()),
        mode=args.mode,
        target_train_words=args.target_train_words or (50_000 if smoke else 10_000_000),
        vocab_size=args.vocab_size or (1_024 if smoke else 16_384),
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size or (1_000 if smoke else 20_000),
        min_document_words=args.min_document_words,
        low_frequency_threshold=args.low_frequency_threshold,
    )


def stable_digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def choose_split(document_id: str, seed: int) -> str:
    value = int.from_bytes(stable_digest(f"{seed}:{document_id}")[:8], "big") / 2**64
    if value < SPLIT_RATIOS["train"]:
        return "train"
    if value < SPLIT_RATIOS["train"] + SPLIT_RATIOS["validation"]:
        return "validation"
    return "test"


def deduplication_text(text: str) -> str:
    """Canonicalize only for hashing; returned corpus text is never replaced by this."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


class DiskHashSet:
    """An exact, disk-backed SHA-256 set with bounded Python memory usage."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS hashes (digest BLOB PRIMARY KEY)")

    def add(self, digest: bytes) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO hashes(digest) VALUES (?)", (digest,)
        )
        return cursor.rowcount == 1

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def load_stream(config: Config) -> Iterable[dict[str, Any]]:
    try:
        stream = load_dataset(
            config.dataset,
            name=config.dataset_config,
            split=config.dataset_split,
            streaming=True,
        )
        return stream.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer_size)
    except Exception as exc:
        raise RuntimeError(
            f"FineWikiの取得開始に失敗しました: dataset={config.dataset!r}, "
            f"config={config.dataset_config!r}, streaming=True. 原因: {exc}"
        ) from exc


def checked_stream(stream: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    iterator = iter(stream)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("FineWikiのストリームが空でした。") from exc
    except Exception as exc:
        raise RuntimeError(f"FineWikiの最初の文書取得に失敗しました。原因: {exc}") from exc

    missing = EXPECTED_FIELDS - set(first)
    if missing:
        raise RuntimeError(
            f"FineWikiのスキーマが想定と異なります。欠落列={sorted(missing)}, "
            f"実際の列={sorted(first)}"
        )
    LOGGER.info("FineWiki schema fields: %s", sorted(first))
    yield first
    yield from iterator


def write_document(handle: TextIO, text: str) -> None:
    # Preserve article content. A single blank line is added only as a document separator.
    handle.write(text)
    if not text.endswith("\n"):
        handle.write("\n")
    handle.write("\n")


def iter_sudachi_chunks(
    text: str, max_bytes: int = SUDACHI_MAX_INPUT_BYTES
) -> Iterator[str]:
    """Yield UTF-8-safe chunks, preferring natural Japanese text boundaries.

    Sentence endings and newlines are preferred, followed by whitespace. If a
    span has no such boundary within ``max_bytes``, it is split between Python
    Unicode characters, which cannot cut through a UTF-8 encoded character.
    Joining the yielded chunks always reconstructs ``text`` exactly.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    start = 0
    text_length = len(text)
    while start < text_length:
        byte_count = 0
        end = start
        last_sentence_boundary: int | None = None
        last_whitespace_boundary: int | None = None

        while end < text_length:
            character = text[end]
            character_bytes = len(character.encode("utf-8"))
            if byte_count + character_bytes > max_bytes:
                break
            byte_count += character_bytes
            end += 1
            if character in SUDACHI_SENTENCE_BOUNDARIES:
                last_sentence_boundary = end
            elif character.isspace():
                last_whitespace_boundary = end

        if end == text_length:
            yield text[start:end]
            return
        if end == start:
            raise ValueError(
                f"max_bytes={max_bytes} is too small for one UTF-8 character"
            )

        split_at = last_sentence_boundary or last_whitespace_boundary or end
        yield text[start:split_at]
        start = split_at


def count_sudachi_words(
    sudachi: Any,
    text: str,
    mode: Any,
    max_bytes: int = SUDACHI_MAX_INPUT_BYTES,
) -> int:
    """Count morphemes without passing oversized input to SudachiPy."""
    return sum(
        1
        for chunk in iter_sudachi_chunks(text, max_bytes=max_bytes)
        for _ in sudachi.tokenize(chunk, mode)
    )


def collect_corpus(config: Config, output_dir: Path) -> dict[str, Any]:
    sudachi = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    counts: dict[str, dict[str, int]] = {
        split: {"documents": 0, "sudachi_words": 0} for split in SPLITS
    }
    rejected = Counter()
    samples: list[str] = []
    sample_rng = random.Random(config.seed)
    seen = DiskHashSet(output_dir / ".dedup.sqlite3")
    temp_paths = {split: output_dir / f".{split}.txt.part" for split in SPLITS}

    try:
        with ExitStack() as stack:
            handles = {
                split: stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
                for split, path in temp_paths.items()
            }
            for row_number, row in enumerate(checked_stream(load_stream(config)), start=1):
                raw_text = row.get("text")
                if not isinstance(raw_text, str) or not raw_text.strip():
                    rejected["empty"] += 1
                    continue

                canonical = deduplication_text(raw_text)
                digest = stable_digest(canonical)
                if not seen.add(digest):
                    rejected["duplicate"] += 1
                    continue

                try:
                    word_count = count_sudachi_words(sudachi, raw_text, mode)
                except Exception as exc:
                    raise RuntimeError(
                        f"SudachiPy解析に失敗しました: row={row_number}, id={row.get('id')!r}. "
                        f"原因: {exc}"
                    ) from exc
                if word_count < config.min_document_words:
                    rejected["too_short"] += 1
                    continue

                document_id = str(row.get("id") or row.get("page_id") or digest.hex())
                split = choose_split(document_id, config.seed)
                write_document(handles[split], raw_text)
                counts[split]["documents"] += 1
                counts[split]["sudachi_words"] += word_count

                if split == "train":
                    nonempty_lines = [line for line in raw_text.splitlines() if line.strip()]
                    for line in nonempty_lines:
                        if len(samples) < 100:
                            samples.append(line)
                        else:
                            index = sample_rng.randrange(counts["train"]["documents"] + 100)
                            if index < 100:
                                samples[index] = line

                if row_number % 100 == 0:
                    seen.connection.commit()
                if row_number % 1_000 == 0:
                    LOGGER.info(
                        "rows=%d accepted_docs=%d train_words=%d/%d",
                        row_number,
                        sum(item["documents"] for item in counts.values()),
                        counts["train"]["sudachi_words"],
                        config.target_train_words,
                    )
                if counts["train"]["sudachi_words"] >= config.target_train_words:
                    break
            else:
                raise RuntimeError(
                    "FineWikiを最後まで読みましたが目標Train語数に到達しませんでした: "
                    f"{counts['train']['sudachi_words']:,}/{config.target_train_words:,}"
                )
    finally:
        seen.close()

    for split in SPLITS:
        os.replace(temp_paths[split], output_dir / f"{split}.txt")
    (output_dir / ".dedup.sqlite3").unlink(missing_ok=True)
    (output_dir / ".dedup.sqlite3-wal").unlink(missing_ok=True)
    (output_dir / ".dedup.sqlite3-shm").unlink(missing_ok=True)
    return {"splits": counts, "rejected_documents": dict(rejected), "roundtrip_samples": samples}


def train_tokenizer(config: Config, output_dir: Path) -> None:
    model_prefix = str(output_dir / "tokenizer")
    try:
        spm.SentencePieceTrainer.train(
            input=str(output_dir / "train.txt"),
            model_prefix=model_prefix,
            model_type="unigram",
            vocab_size=config.vocab_size,
            character_coverage=config.character_coverage,
            byte_fallback=config.byte_fallback,
            normalization_rule_name=config.normalization_rule_name,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>",
            input_sentence_size=0,
            shuffle_input_sentence=False,
            num_threads=1,
            hard_vocab_limit=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "SentencePiece学習に失敗しました。設定は自動変更していません。"
            f" vocab_size={config.vocab_size}, model_type='unigram', "
            f"character_coverage={config.character_coverage}, byte_fallback=True, "
            f"normalization_rule_name={config.normalization_rule_name!r}. 原因: {exc}"
        ) from exc


def iter_nonempty_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if text.strip():
                yield text


def evaluate(config: Config, output_dir: Path, collection: dict[str, Any]) -> dict[str, Any]:
    processor = spm.SentencePieceProcessor(model_file=str(output_dir / "tokenizer.model"))
    token_counts: dict[str, int] = {}
    train_frequencies = [0] * processor.vocab_size()

    for split in SPLITS:
        total = 0
        for line in iter_nonempty_lines(output_dir / f"{split}.txt"):
            ids = processor.encode(line, out_type=int)
            total += len(ids)
            if split == "train":
                for token_id in ids:
                    train_frequencies[token_id] += 1
        token_counts[split] = total

    used = sum(frequency > 0 for frequency in train_frequencies)
    threshold = config.low_frequency_threshold
    low_frequency = sum(0 < frequency <= threshold for frequency in train_frequencies)
    roundtrip_results = []
    for text in collection.pop("roundtrip_samples")[:20]:
        ids = processor.encode(text, out_type=int)
        decoded = processor.decode(ids)
        roundtrip_results.append(
            {
                "input": text,
                "decoded": decoded,
                "exact_match": decoded == text,
                "token_ids_stable": processor.encode(decoded, out_type=int) == ids,
            }
        )

    train_words = collection["splits"]["train"]["sudachi_words"]
    return {
        **collection,
        "token_counts": token_counts,
        "average_tokens_per_train_word": token_counts["train"] / train_words,
        "vocabulary": {
            "size": processor.vocab_size(),
            "used_on_train": used,
            "usage_rate": used / processor.vocab_size(),
            "unused_on_train": processor.vocab_size() - used,
            "low_frequency_threshold": threshold,
            "low_frequency_tokens_on_train": low_frequency,
        },
        "roundtrip": {
            "tested": len(roundtrip_results),
            "exact_matches": sum(item["exact_match"] for item in roundtrip_results),
            "token_id_stable": sum(item["token_ids_stable"] for item in roundtrip_results),
            "note": "nmt_nfkc正規化と空白処理のため、raw textの完全一致は必須ではありません。token_id_stableも併記します。",
            "cases": roundtrip_results,
        },
    }


def print_summary(stats: dict[str, Any]) -> None:
    print("\n=== Evaluation ===")
    print(f"Train Sudachi words: {stats['splits']['train']['sudachi_words']:,}")
    for split in SPLITS:
        print(
            f"{split}: documents={stats['splits'][split]['documents']:,}, "
            f"tokens={stats['token_counts'][split]:,}"
        )
    print(f"Average tokens/train word: {stats['average_tokens_per_train_word']:.6f}")
    vocab = stats["vocabulary"]
    print(
        f"Vocabulary usage: {vocab['used_on_train']:,}/{vocab['size']:,} "
        f"({vocab['usage_rate']:.2%}); frequency <= {vocab['low_frequency_threshold']}: "
        f"{vocab['low_frequency_tokens_on_train']:,}"
    )
    roundtrip = stats["roundtrip"]
    print(
        f"Encode -> Decode: exact={roundtrip['exact_matches']}/{roundtrip['tested']}, "
        f"token-id-stable={roundtrip['token_id_stable']}/{roundtrip['tested']}"
    )


def main() -> int:
    args = parse_args()
    config = make_config(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8"),
        ],
    )
    started = time.time()
    stats_path = output_dir / "tokenizer_stats.json"
    stats: dict[str, Any] = {
        "status": "running",
        "config": asdict(config),
        "counting_method": "SudachiPy dictionary.Dictionary().create(), Tokenizer.SplitMode.C; sum of morphemes per accepted document",
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        collection = collect_corpus(config, output_dir)
        train_tokenizer(config, output_dir)
        stats.update(evaluate(config, output_dir, collection))
        stats["status"] = "completed"
        stats["elapsed_seconds"] = time.time() - started
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(stats)
        return 0
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        stats["status"] = "failed"
        stats["error_type"] = type(exc).__name__
        stats["error"] = str(exc)
        stats["elapsed_seconds"] = time.time() - started
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
