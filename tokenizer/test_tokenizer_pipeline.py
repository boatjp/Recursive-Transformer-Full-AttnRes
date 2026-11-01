import unittest
from argparse import Namespace
from pathlib import Path

from sudachipy import dictionary, tokenizer

from tokenizer_pipeline import (
    SPLIT_RATIOS,
    choose_split,
    count_sudachi_words,
    deduplication_text,
    iter_sudachi_chunks,
    make_config,
)


class ByteLimitedTokenizer:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.inputs: list[str] = []

    def tokenize(self, text: str, mode: object) -> list[str]:
        del mode
        if len(text.encode("utf-8")) > self.max_bytes:
            raise ValueError("input exceeds tokenizer byte limit")
        self.inputs.append(text)
        return list(text)


class PipelineTests(unittest.TestCase):
    def test_split_is_deterministic(self) -> None:
        self.assertEqual(choose_split("jawiki/123", 42), choose_split("jawiki/123", 42))

    def test_split_ratios_sum_to_one(self) -> None:
        self.assertEqual(sum(SPLIT_RATIOS.values()), 1.0)

    def test_dedup_canonicalization(self) -> None:
        self.assertEqual(deduplication_text("  a\r\nb  \r\n"), "a\nb")

    def test_full_defaults(self) -> None:
        config = make_config(
            Namespace(
                mode="full",
                output_dir=Path("output"),
                seed=20251031,
                target_train_words=None,
                vocab_size=None,
                shuffle_buffer_size=None,
                min_document_words=50,
                low_frequency_threshold=5,
            )
        )
        self.assertEqual(config.target_train_words, 10_000_000)
        self.assertEqual(config.vocab_size, 16_384)
        self.assertEqual(config.character_coverage, 0.9995)
        self.assertTrue(config.byte_fallback)
        self.assertEqual(config.normalization_rule_name, "nmt_nfkc")

    def test_long_text_prefers_newline_and_sentence_boundaries(self) -> None:
        text = "あ" * 8 + "。" + "い" * 8 + "\n" + "う" * 8
        chunks = list(iter_sudachi_chunks(text, max_bytes=30))

        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 30 for chunk in chunks))
        self.assertEqual(chunks[0], "あ" * 8 + "。")
        self.assertEqual(chunks[1], "い" * 8 + "\n")

    def test_long_text_without_boundaries_is_split_safely(self) -> None:
        text = "漢" * 40
        chunks = list(iter_sudachi_chunks(text, max_bytes=31))

        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 31 for chunk in chunks))
        self.assertEqual([len(chunk) for chunk in chunks], [10, 10, 10, 10])

    def test_multibyte_text_is_counted_without_oversized_sudachi_input(self) -> None:
        text = ("日本語の長文です。" * 4) + "末尾"
        sudachi = ByteLimitedTokenizer(max_bytes=30)

        count = count_sudachi_words(sudachi, text, mode=object(), max_bytes=30)

        self.assertEqual(count, len(text))
        self.assertEqual("".join(sudachi.inputs), text)
        self.assertTrue(
            all(len(chunk.encode("utf-8")) <= 30 for chunk in sudachi.inputs)
        )

    def test_actual_sudachi_handles_oversized_unbroken_japanese_text(self) -> None:
        text = "日本語" * 7_000
        self.assertGreater(len(text.encode("utf-8")), 49_149)
        sudachi = dictionary.Dictionary().create()

        count = count_sudachi_words(
            sudachi, text, tokenizer.Tokenizer.SplitMode.C
        )

        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
