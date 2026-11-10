import hashlib
import io
import json
import math
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import sentencepiece as sentencepiece
import torch
from torch import nn

from data import DataConfig, prepare_split
from evaluate import evaluate_checkpoint, main as evaluation_main
from generate import choose_token, generate_token_ids, main as generation_main
from inference import load_model_checkpoint
from recursive_transformer import ModelConfig, RecursiveTransformerLM
from train import TrainConfig


TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer/output-full/tokenizer.model"


class FixedLogits(nn.Module):
    def __init__(self, vocab_size: int, preferred_id: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.preferred_id = preferred_id
        self.lengths = []

    def forward(self, input_ids):
        self.lengths.append(input_ids.shape[1])
        logits = torch.zeros((1, input_ids.shape[1], self.vocab_size), device=input_ids.device)
        logits[:, :, self.preferred_id] = 100
        return logits


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        shutil.copyfile(TOKENIZER, self.path / "tokenizer.model")
        for split in ("train", "validation", "test"):
            (self.path / f"{split}.txt").write_text("日本語の評価用文章です。\n" * 15, encoding="utf-8")
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(self.path / "tokenizer.model"))

    def checkpoint(self, kind="residual"):
        config = TrainConfig(data_dir=str(self.path), output_dir=str(self.path),
                             sequence_length=8, micro_batch_size=2, residual_type=kind,
                             model_overrides={"hidden_size": 16, "num_heads": 2, "num_steps": 2,
                                              "intermediate_size": 32, "embedding_rank": 8})
        model_config = ModelConfig.from_tokenizer(self.path / "tokenizer.model",
                                                   residual_type=kind, **config.model_overrides)
        model = RecursiveTransformerLM(model_config)
        data_config = DataConfig(data_dir=self.path, sequence_length=8)
        identity = {split: {name: prepare_split(data_config, split)[name] for name in (
            "source_sha256", "tokenizer_sha256", "token_count", "tokenization_method")}
            for split in ("train", "validation", "test")}
        path = self.path / "best.pt"
        torch.save({"config": asdict(config), "model": model.state_dict(),
                    "data_identity": identity}, path)
        return path, model

    def test_evaluate_both_models_and_no_parameter_updates(self):
        for kind in ("residual", "attnres"):
            path, model = self.checkpoint(kind)
            before = {name: value.clone() for name, value in model.state_dict().items()}
            result = evaluate_checkpoint(path, batch_size=2, device=torch.device("cpu"))
            self.assertEqual(result["residual_type"], kind)
            self.assertGreater(result["test_loss"], 0)
            self.assertAlmostEqual(result["test_perplexity"], math.exp(result["test_loss"]))
            self.assertEqual(result["precision"], "fp32")
            restored, _, _ = load_model_checkpoint(path, torch.device("cpu"))
            self.assertFalse(restored.training)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, before[name])

    def test_invalid_checkpoint_tokenizer_and_data(self):
        self.assertRaises(FileNotFoundError, load_model_checkpoint,
                          self.path / "missing.pt", torch.device("cpu"))
        path, _ = self.checkpoint()
        state = torch.load(path, weights_only=False)
        state["data_identity"]["test"]["source_sha256"] = "wrong"
        torch.save(state, path)
        self.assertRaisesRegex(ValueError, "test data identity", evaluate_checkpoint,
                               path, 2, torch.device("cpu"))
        state["data_identity"]["test"]["source_sha256"] = hashlib.sha256(
            (self.path / "test.txt").read_bytes()).hexdigest()
        state["data_identity"]["test"]["tokenizer_sha256"] = "wrong"
        torch.save(state, path)
        self.assertRaisesRegex(ValueError, "tokenizer identity", load_model_checkpoint,
                               path, torch.device("cpu"))
        path.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(ValueError, "Invalid or incompatible checkpoint"):
            load_model_checkpoint(path, torch.device("cpu"))

    def test_sampling_temperature_top_k_top_p_and_seed(self):
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(choose_token(logits, sampling=False), 3)
        self.assertEqual(choose_token(logits, top_k=1), 3)
        self.assertEqual(choose_token(logits, top_k=0, top_p=0.1), 3)
        first = torch.Generator().manual_seed(21)
        second = torch.Generator().manual_seed(21)
        self.assertEqual([choose_token(logits, top_k=0, generator=first) for _ in range(10)],
                         [choose_token(logits, top_k=0, generator=second) for _ in range(10)])
        original = torch.multinomial
        probabilities = []
        def capture(values, count, generator=None):
            probabilities.append(values.clone())
            return original(values, count, generator=generator)
        with patch("generate.torch.multinomial", side_effect=capture):
            choose_token(logits, temperature=0.5, top_k=0, top_p=1)
            choose_token(logits, temperature=2.0, top_k=0, top_p=1)
        self.assertGreater(probabilities[0][-1], probabilities[1][-1])
        self.assertRaises(ValueError, choose_token, logits, temperature=0)

    def test_eos_and_no_context_truncation(self):
        prompt = "日本語の文章"
        eos_model = FixedLogits(self.tokenizer.vocab_size(), self.tokenizer.eos_id())
        original = self.tokenizer.encode(prompt)
        self.assertEqual(generate_token_ids(eos_model, self.tokenizer, prompt,
                                            max_new_tokens=4, sampling=False), original)
        self.assertEqual(eos_model.lengths, [len(original)])
        token = self.tokenizer.encode("猫")[0]
        model = FixedLogits(self.tokenizer.vocab_size(), token)
        long_prompt = "猫" * 600
        prefix = self.tokenizer.encode(long_prompt)
        result = generate_token_ids(model, self.tokenizer, long_prompt,
                                    max_new_tokens=2, sampling=False)
        self.assertEqual(len(result), len(prefix) + 2)
        self.assertEqual(model.lengths, [len(prefix), len(prefix) + 1])
        self.assertGreater(len(prefix), 512)

    def test_cache_rejection_and_both_model_generation(self):
        path, _ = self.checkpoint()
        for kind in ("residual", "attnres"):
            if kind == "attnres":
                path, _ = self.checkpoint(kind)
            model, _, _ = load_model_checkpoint(path, torch.device("cpu"))
            tokens = generate_token_ids(model, self.tokenizer, "日本語", max_new_tokens=2,
                                        top_k=1, seed=7)
            self.assertGreaterEqual(len(tokens), len(self.tokenizer.encode("日本語")))
            self.assertIsInstance(self.tokenizer.decode(tokens), str)
            self.assertRaises(NotImplementedError, generate_token_ids,
                              model, self.tokenizer, "日本語", use_cache=True)

    def test_out_of_memory_reports_context_length(self):
        model = FixedLogits(self.tokenizer.vocab_size(), 4)
        with patch.object(model, "forward", side_effect=torch.OutOfMemoryError("simulated")):
            with self.assertRaisesRegex(RuntimeError, "context length"):
                generate_token_ids(model, self.tokenizer, "日本語", max_new_tokens=1)

    def test_command_line_evaluation_and_generation(self):
        path, _ = self.checkpoint()
        output = self.path / "metrics.json"
        with (patch("sys.argv", ["evaluate.py", "--checkpoint", str(path),
                                 "--output", str(output), "--batch-size", "2"]),
              redirect_stdout(io.StringIO())):
            evaluation_main()
        self.assertGreater(json.loads(output.read_text(encoding="utf-8"))["test_loss"], 0)
        with (patch("sys.argv", ["generate.py", "--checkpoint", str(path),
                                 "--prompt", "日本語", "--max-new-tokens", "1", "--greedy"]),
              redirect_stdout(io.StringIO()) as printed):
            generation_main()
        self.assertTrue(printed.getvalue().strip())
        with (patch("sys.argv", ["generate.py", "--prompt", "日本語", "--use-cache"]),
              redirect_stderr(io.StringIO()) as errors):
            with self.assertRaises(SystemExit):
                generation_main()
        self.assertIn("KV Cache is not implemented", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
