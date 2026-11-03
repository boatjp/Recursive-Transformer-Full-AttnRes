import json
import hashlib
import os
import shutil
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import sentencepiece as sentencepiece
import torch

from data import (DataConfig, FixedBlockDataset, TokenStreamDataset, create_dataloaders,
                  dataset_statistics, prepare_split)
from recursive_transformer import ModelConfig, RecursiveTransformerLM


TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer/output-full/tokenizer.model"


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        shutil.copyfile(TOKENIZER, self.path / "tokenizer.model")
        self.texts = {
            "train": "日本語の文章です。\n\nこれは本文中の空行です。\nさらに続きます。" * 8,
            "validation": "検証用の文章です。\n二行目も含みます。" * 5,
            "test": "テスト用の文章です。\n文書はここまでです。" * 5,
        }
        for split, content in self.texts.items():
            (self.path / f"{split}.txt").write_text(content, encoding="utf-8")
        self.config = DataConfig(data_dir=self.path, sequence_length=8, seed=37)

    def test_tokenizer_and_all_splits(self):
        processor = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER))
        counts = {}
        for split in self.texts:
            metadata = prepare_split(self.config, split)
            self.assertEqual(metadata["vocab_size"], processor.vocab_size())
            self.assertEqual(metadata["bos_token_id"], processor.bos_id())
            self.assertEqual(metadata["eos_token_id"], processor.eos_id())
            self.assertEqual(metadata["pad_token_id"], processor.pad_id())
            self.assertGreater(metadata["token_count"], 8)
            counts[split] = metadata["token_count"]
        self.assertEqual(len(counts), 3)
        stats = dataset_statistics(self.config)
        self.assertEqual(stats["vocab_size"], processor.vocab_size())
        self.assertEqual(stats["sequence_length"], 8)
        self.assertEqual(stats["seed"], 37)
        for split, count in counts.items():
            self.assertEqual(stats["splits"][split]["token_count"], count)
            self.assertEqual(stats["splits"][split]["sample_count"], count // 8)
            self.assertEqual(stats["splits"][split]["discarded_tail_tokens"], count % 8)

    def test_blocks_are_contiguous_nonoverlapping_and_unshifted(self):
        metadata = prepare_split(self.config, "train")
        stream = TokenStreamDataset(Path(metadata["cache_path"]), metadata["token_count"])
        dataset = FixedBlockDataset(stream, 8)
        self.assertRaises(IndexError, stream.__getitem__, -1)
        self.assertRaises(IndexError, dataset.__getitem__, len(dataset))
        flattened = []
        for sample in dataset:
            self.assertEqual(sample["input_ids"].shape, (8,))
            torch.testing.assert_close(sample["input_ids"], sample["labels"])
            torch.testing.assert_close(sample["attention_mask"], torch.ones(8, dtype=torch.long))
            self.assertTrue(((sample["input_ids"] >= 0)
                             & (sample["input_ids"] < metadata["vocab_size"])).all())
            flattened.extend(sample["input_ids"].tolist())
        self.assertEqual(flattened, [stream[index] for index in range(len(flattened))])
        self.assertEqual(len(flattened), metadata["token_count"] - metadata["token_count"] % 8)
        with patch("data.struct.unpack_from", wraps=struct.unpack_from) as unpack:
            torch.testing.assert_close(dataset[0]["input_ids"], torch.tensor(stream.read_range(0, 8)))
            self.assertEqual(unpack.call_count, 2)
            self.assertEqual(unpack.call_args_list[0].args[0], "<8i")
        stream.close()

    def test_line_tokenization_matches_reference_with_blank_lines_and_crlf(self):
        source = self.path / "train.txt"
        source.write_bytes("日本語。\r\n\r\n本文の中に空行。\r\n  \r\n次の行。\n".encode("utf-8"))
        processor = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER))
        with source.open("r", encoding="utf-8") as handle:
            expected = [token_id for line in handle
                        for token_id in (processor.encode(line.rstrip("\n"), out_type=int)
                                         if line.strip() else [])]
        metadata = prepare_split(self.config, "train")
        stream = TokenStreamDataset(Path(metadata["cache_path"]), metadata["token_count"])
        self.assertEqual(list(stream.read_range(0, len(stream))), expected)
        self.assertEqual(metadata["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(metadata["tokenization_method"], "nonempty_lines_without_newlines_v1")
        stream.close()

    def test_bos_eos_once_and_blank_lines_are_not_boundaries(self):
        config = replace(self.config, add_bos=True, add_eos=True)
        metadata = prepare_split(config, "train")
        stream = TokenStreamDataset(Path(metadata["cache_path"]), metadata["token_count"])
        self.assertEqual(stream[0], metadata["bos_token_id"])
        self.assertEqual(stream[metadata["token_count"] - 1], metadata["eos_token_id"])
        eos_positions = [index for index in range(len(stream)) if stream[index] == metadata["eos_token_id"]]
        self.assertEqual(eos_positions, [len(stream) - 1])
        stream.close()
        default = prepare_split(self.config, "train")
        plain = TokenStreamDataset(Path(default["cache_path"]), default["token_count"])
        self.assertNotIn(default["eos_token_id"], [plain[index] for index in range(len(plain))])
        plain.close()

    def test_shuffle_reproducible_and_evaluation_ordered(self):
        first = create_dataloaders(self.config, batch_size=2)
        second = create_dataloaders(self.config, batch_size=2)
        def starts(loader):
            return [row[0].item() for batch in loader for row in batch["input_ids"]]
        self.assertEqual(starts(first["train"]), starts(second["train"]))
        for split in ("validation", "test"):
            expected = [first[split].dataset[index]["input_ids"][0].item()
                        for index in range(len(first[split].dataset))]
            self.assertEqual(starts(first[split]), expected)
            self.assertFalse(first[split].drop_last)
        self.assertTrue(first["train"].drop_last)
        self.assertEqual(first["train"].num_workers, 0)

    def test_workers_return_same_order(self):
        single = create_dataloaders(self.config, batch_size=3, train_shuffle=False)
        workers = create_dataloaders(self.config, batch_size=3, num_workers=2,
                                     train_shuffle=False, pin_memory=True)
        for split in ("train", "validation", "test"):
            expected = [batch["input_ids"] for batch in single[split]]
            observed = [batch["input_ids"] for batch in workers[split]]
            self.assertEqual(len(expected), len(observed))
            for left, right in zip(expected, observed):
                torch.testing.assert_close(left, right)

    def test_cache_reuse_and_invalidation(self):
        metadata = prepare_split(self.config, "train")
        with patch("data._load_tokenizer", side_effect=AssertionError("cache should be reused")):
            self.assertEqual(prepare_split(self.config, "train"), metadata)
        changed = self.path / "train.txt"
        changed.write_text(self.texts["train"] + "増えた文章。", encoding="utf-8")
        new_metadata = prepare_split(self.config, "train")
        self.assertNotEqual(metadata["source_sha256"], new_metadata["source_sha256"])
        self.assertNotEqual(metadata["token_count"], new_metadata["token_count"])
        metadata_path = self.config.resolved_cache_dir / "train.json"
        stale = {**new_metadata, "version": 1, "chunk_characters": 65536}
        metadata_path.write_text(json.dumps(stale), encoding="utf-8")
        with patch("data._load_tokenizer", wraps=__import__("data")._load_tokenizer) as loader:
            prepare_split(self.config, "train")
            self.assertEqual(loader.call_count, 1)
        with patch("data._load_tokenizer", wraps=__import__("data")._load_tokenizer) as loader:
            prepare_split(replace(self.config, add_bos=True), "train")
            self.assertEqual(loader.call_count, 1)
        tokenizer_path = self.path / "tokenizer.model"
        os.utime(tokenizer_path, None)
        with patch("data._load_tokenizer", wraps=__import__("data")._load_tokenizer) as loader:
            prepare_split(self.config, "train")
            self.assertEqual(loader.call_count, 1)
        saved = json.loads((self.config.resolved_cache_dir / "train.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["token_count"], new_metadata["token_count"])

    def test_model_forward_loss_compatibility(self):
        batch = next(iter(create_dataloaders(self.config, batch_size=2)["train"]))
        for kind in ("residual", "attnres"):
            config = ModelConfig.from_tokenizer(TOKENIZER, residual_type=kind, hidden_size=16,
                                                num_heads=2, num_steps=2,
                                                intermediate_size=32, embedding_rank=8)
            model = RecursiveTransformerLM(config)
            output = model(**batch)
            self.assertEqual(output.logits.shape, (2, 8, config.vocab_size))
            self.assertTrue(torch.isfinite(output.loss))

    def test_invalid_inputs_and_empty_data(self):
        self.assertRaises(ValueError, DataConfig, sequence_length=1)
        self.assertRaises(ValueError, prepare_split, self.config, "other")
        self.assertRaises(ValueError, create_dataloaders, self.config, 0)
        (self.path / "validation.txt").write_text("", encoding="utf-8")
        self.assertRaises(ValueError, prepare_split, self.config, "validation")
        (self.path / "validation.txt").unlink()
        self.assertRaises(FileNotFoundError, prepare_split, self.config, "validation")
        (self.path / "validation.txt").write_bytes(b"\xff")
        self.assertRaises(UnicodeDecodeError, prepare_split, self.config, "validation")


if __name__ == "__main__":
    unittest.main()
