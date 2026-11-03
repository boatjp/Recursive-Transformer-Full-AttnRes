from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import sentencepiece as sentencepiece
import torch
from torch.utils.data import DataLoader, Dataset


SPLITS = ("train", "validation", "test")
CACHE_VERSION = 2
TOKENIZATION_METHOD = "nonempty_lines_without_newlines_v1"


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = Path("tokenizer/output-full")
    sequence_length: int = 512   #512に変更
    seed: int = 0
    add_bos: bool = False
    add_eos: bool = False
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2 for causal loss")

    @property
    def resolved_cache_dir(self) -> Path:
        return self.cache_dir if self.cache_dir is not None else self.data_dir / "token_cache"


def _file_signature(path: Path) -> dict[str, int]:
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required data file is missing: {path}") from exc
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer(config: DataConfig) -> sentencepiece.SentencePieceProcessor:
    path = config.data_dir / "tokenizer.model"
    _file_signature(path)
    processor = sentencepiece.SentencePieceProcessor(model_file=str(path))
    if processor.vocab_size() <= 0:
        raise ValueError("Tokenizer vocabulary is empty")
    for requested, name, token_id in ((config.add_bos, "BOS", processor.bos_id()),
                                      (config.add_eos, "EOS", processor.eos_id())):
        if requested and token_id < 0:
            raise ValueError(f"Tokenizer has no {name} token")
    return processor


def _text_lines(path: Path) -> Iterator[tuple[str, str]]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        for line in handle:
            text = line.rstrip("\r\n")
            yield line, text


def _cache_paths(config: DataConfig, split: str) -> tuple[Path, Path]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    base = config.resolved_cache_dir / split
    return base.with_suffix(".tokens"), base.with_suffix(".json")


def prepare_split(config: DataConfig, split: str) -> dict:
    token_path, metadata_path = _cache_paths(config, split)
    text_path = config.data_dir / f"{split}.txt"
    tokenizer_path = config.data_dir / "tokenizer.model"
    source_signature = _file_signature(text_path)
    if source_signature["size"] == 0:
        raise ValueError(f"Source split is empty: {text_path}")
    tokenizer_signature = _file_signature(tokenizer_path)
    settings = {"version": CACHE_VERSION, "split": split,
                "source_path": str(text_path.resolve()), "source_signature": source_signature,
                "tokenizer_path": str(tokenizer_path.resolve()),
                "tokenizer_signature": tokenizer_signature,
                "add_bos": config.add_bos, "add_eos": config.add_eos,
                "tokenization_method": TOKENIZATION_METHOD}
    if metadata_path.is_file() and token_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (all(metadata.get(key) == value for key, value in settings.items())
                    and token_path.stat().st_size == metadata["token_count"] * 4
                    and metadata["token_count"] > 0):
                return metadata
        except (OSError, ValueError, KeyError, TypeError):
            pass

    processor = _load_tokenizer(config)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = token_path.with_name(token_path.name + f".{os.getpid()}.{uuid.uuid4().hex}.part")
    digest = hashlib.sha256()
    token_count = 0
    content_token_count = 0
    try:
        with temporary.open("wb") as output:
            if config.add_bos:
                output.write(struct.pack("<i", processor.bos_id()))
                token_count += 1
            for raw_line, text in _text_lines(text_path):
                digest.update(raw_line.encode("utf-8"))
                if not text.strip():
                    continue
                tokens = processor.encode(text, out_type=int)
                if tokens:
                    if min(tokens) < 0 or max(tokens) >= processor.vocab_size():
                        raise ValueError("Tokenizer produced an out-of-range token ID")
                    output.write(struct.pack(f"<{len(tokens)}i", *tokens))
                    token_count += len(tokens)
                    content_token_count += len(tokens)
            if config.add_eos:
                output.write(struct.pack("<i", processor.eos_id()))
                token_count += 1
        if content_token_count == 0:
            raise ValueError(f"Tokenized split is empty: {text_path}")
        if _file_signature(text_path) != source_signature or _file_signature(tokenizer_path) != tokenizer_signature:
            raise RuntimeError("Source or tokenizer changed while creating cache")
        metadata = {**settings, "vocab_size": processor.vocab_size(),
                    "bos_token_id": processor.bos_id(), "eos_token_id": processor.eos_id(),
                    "pad_token_id": processor.pad_id(), "unk_token_id": processor.unk_id(),
                    "token_count": token_count, "source_sha256": digest.hexdigest(),
                    "tokenizer_sha256": _sha256(tokenizer_path), "cache_path": str(token_path.resolve())}
        metadata_temporary = metadata_path.with_name(metadata_path.name + f".{uuid.uuid4().hex}.part")
        try:
            metadata_temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, token_path)
            os.replace(metadata_temporary, metadata_path)
        finally:
            metadata_temporary.unlink(missing_ok=True)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


class TokenStreamDataset(Dataset[int]):
    def __init__(self, path: Path, token_count: int):
        self.path = path
        self.token_count = token_count
        self._handle = None
        self._mapping = None

    def __len__(self) -> int:
        return self.token_count

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < self.token_count:
            raise IndexError(index)
        return self.read_range(index, 1)[0]

    def read_range(self, start: int, length: int) -> tuple[int, ...]:
        if start < 0 or length < 0 or start + length > self.token_count:
            raise IndexError((start, length))
        if length == 0:
            return ()
        if self._mapping is None:
            self._handle = self.path.open("rb")
            self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        return struct.unpack_from(f"<{length}i", self._mapping, start * 4)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_mapping"] = None
        return state

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()


class FixedBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, stream: TokenStreamDataset, sequence_length: int):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.stream = stream
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.stream) // self.sequence_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        tokens = torch.tensor(self.stream.read_range(start, self.sequence_length), dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone(),
                "attention_mask": torch.ones(self.sequence_length, dtype=torch.long)}


def create_dataloaders(config: DataConfig, batch_size: int, num_workers: int = 0,
                       pin_memory: bool = False, train_shuffle: bool = True,
                       train_drop_last: bool = True) -> dict[str, DataLoader]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    loaders = {}
    for split in SPLITS:
        metadata = prepare_split(config, split)
        dataset = FixedBlockDataset(TokenStreamDataset(Path(metadata["cache_path"]),
                                                       metadata["token_count"]), config.sequence_length)
        if len(dataset) == 0:
            raise ValueError(f"No full-length samples in {split}")
        shuffle = train_shuffle if split == "train" else False
        generator = torch.Generator().manual_seed(config.seed) if shuffle else None
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                    drop_last=train_drop_last if split == "train" else False,
                                    num_workers=num_workers, pin_memory=pin_memory,
                                    generator=generator)
    return loaders


def dataset_statistics(config: DataConfig) -> dict:
    splits = {}
    vocab_size = None
    for split in SPLITS:
        metadata = prepare_split(config, split)
        vocab_size = metadata["vocab_size"]
        count = metadata["token_count"]
        splits[split] = {"token_count": count, "sample_count": count // config.sequence_length,
                         "discarded_tail_tokens": count % config.sequence_length,
                         "cache_path": metadata["cache_path"]}
    return {"vocab_size": vocab_size, "sequence_length": config.sequence_length,
            "seed": config.seed, "splits": splits}
