import json
import math
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from recursive_transformer import ModelConfig, RecursiveTransformerLM
from train import (TrainConfig, Trainer, WarmupCosineScheduler, evaluate,
                   optimizer_groups, valid_token_count)


class TinyDataset(Dataset):
    def __init__(self, size=7, masked=False):
        self.size = size
        self.masked = masked

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        tokens = torch.tensor([1, 4 + index % 3, 7 + index % 4, 9, 2])
        labels = tokens.clone()
        if self.masked and index % 2:
            labels[2:4] = -100
        return {"input_ids": tokens, "labels": labels,
                "attention_mask": torch.ones_like(tokens)}


class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.config = TrainConfig(output_dir=str(self.path), micro_batch_size=2,
                                  gradient_accumulation_steps=2, max_epochs=2,
                                  validation_interval=1, checkpoint_interval=1,
                                  sequence_length=5, precision="fp32")
        self.dataset = TinyDataset(masked=True)
        self.loaders = {"train": DataLoader(self.dataset, batch_size=2, drop_last=True),
                        "validation": DataLoader(TinyDataset(3), batch_size=2)}

    def trainer(self, kind="residual", config=None, seed=123):
        torch.manual_seed(seed)
        model = RecursiveTransformerLM(ModelConfig(vocab_size=31, hidden_size=16,
                                                    num_steps=2, num_heads=2,
                                                    intermediate_size=32, embedding_rank=8,
                                                    residual_type=kind))
        return Trainer(config or replace(self.config, residual_type=kind), loaders=self.loaders,
                       model=model, data_identity={"test_data": "fixed"}, device=torch.device("cpu"))

    def test_optimizer_groups_cover_each_parameter_once(self):
        for kind in ("residual", "attnres"):
            trainer = self.trainer(kind)
            groups = trainer.optimizer.param_groups
            all_ids = [id(parameter) for group in groups for parameter in group["params"]]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(set(all_ids), {id(parameter) for parameter in trainer.model.parameters()})
            self.assertEqual(groups[0]["weight_decay"], 0.1)
            self.assertEqual(groups[1]["weight_decay"], 0)
            no_decay = {id(parameter) for parameter in groups[1]["params"]}
            self.assertIn(id(trainer.model.attn_norms[0].weight), no_decay)
            self.assertIn(id(trainer.model.attn_norms[0].bias), no_decay)
            if kind == "attnres":
                self.assertIn(id(trainer.model.attnres.queries), no_decay)
            self.assertNotIn(id(trainer.model.embedding.weight), no_decay)
            self.assertNotIn(id(trainer.model.block.ffn.up.weight), no_decay)

    def test_schedule_warmup_cosine_and_update_frequency(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=0.001)
        scheduler = WarmupCosineScheduler(optimizer, 10, 0.001, 0.2, 0.1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0005)
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001)
        rates = [scheduler.rate(number) for number in range(3, 11)]
        self.assertTrue(all(left > right for left, right in zip(rates, rates[1:])))
        self.assertAlmostEqual(rates[-1], 0.0001)
        self.assertEqual(scheduler.completed_steps, 1)

    def test_accumulation_scales_by_valid_tokens_including_short_group(self):
        trainer = self.trainer()
        order = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(42)).tolist()[:6]
        batches = [self.loaders["train"].collate_fn([self.dataset[index] for index in order[start:start + 2]])
                   for start in (0, 2)]
        counts = [valid_token_count(batch) for batch in batches]
        with torch.no_grad():
            losses = [trainer.model(**batch).loss.item() for batch in batches]
        expected = sum(loss * count for loss, count in zip(losses, counts)) / sum(counts)
        with patch.object(trainer, "_log") as logger:
            trainer.run(max_steps=1)
        self.assertAlmostEqual(logger.call_args.args[0]["train_loss"], expected, places=6)
        self.assertEqual(trainer.global_step, 1)
        self.assertEqual(trainer.scheduler.completed_steps, 1)
        final_batch = self.loaders["train"].collate_fn([self.dataset[index] for index in order[4:6]])
        with torch.no_grad():
            final_loss = trainer.model(**final_batch).loss.item()
        with patch.object(trainer, "_log") as logger:
            trainer.run(max_steps=2)
        self.assertEqual(trainer.global_step, 2)
        self.assertEqual(trainer.epoch, 1)
        self.assertEqual(trainer.batch_cursor, 0)
        self.assertAlmostEqual(logger.call_args.args[0]["train_loss"], final_loss, places=6)

    def test_one_step_both_models_and_gradient_clipping(self):
        for kind in ("residual", "attnres"):
            trainer = self.trainer(kind)
            before = trainer.model.embedding.weight.detach().clone()
            norms = []
            original = torch.nn.utils.clip_grad_norm_
            def clip(parameters, max_norm, **kwargs):
                result = original(parameters, max_norm, **kwargs)
                norms.append((max_norm, float(result)))
                return result
            with patch("train.nn.utils.clip_grad_norm_", side_effect=clip):
                trainer.run(max_steps=1)
            self.assertEqual(trainer.global_step, 1)
            self.assertEqual(norms[0][0], 1.0)
            self.assertTrue(math.isfinite(norms[0][1]))
            self.assertGreater((before - trainer.model.embedding.weight).abs().sum().item(), 0)
            record = json.loads((trainer.output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(record["residual_type"], kind)
            self.assertTrue(math.isfinite(record["validation_loss"]))
            self.assertTrue(math.isfinite(record["validation_perplexity"]))
            self.assertTrue((trainer.output_dir / "best.pt").is_file())
            self.assertTrue((trainer.output_dir / "last.pt").is_file())

    def test_step_measurement_excludes_validation_and_checkpoint(self):
        trainer = self.trainer()
        events = []
        expected_tokens = sum(valid_token_count(batch) for batch in list(trainer._epoch_loader())[:2])
        with (patch.object(trainer, "_start_step_measurement", side_effect=lambda: events.append("start") or 10.0),
              patch.object(trainer, "_finish_step_measurement", side_effect=lambda started: events.append("finish") or (2.0, 123)),
              patch("train.evaluate", side_effect=lambda *args: events.append("validation") or (1.0, math.e)),
              patch.object(trainer, "save_checkpoint", side_effect=lambda *args: events.append("checkpoint")),
              patch.object(trainer, "_log") as logger):
            trainer.run(max_steps=1)
        self.assertEqual(events[:3], ["start", "finish", "validation"])
        self.assertIn("checkpoint", events)
        record = logger.call_args.args[0]
        self.assertEqual(record["gpu_peak_memory_bytes"], 123)
        self.assertAlmostEqual(record["tokens_per_second"], expected_tokens / 2)

    def test_cuda_step_measurement_resets_and_synchronizes(self):
        trainer = self.trainer()
        trainer.device = torch.device("cuda")
        events = []
        with (patch("train.torch.cuda.synchronize", side_effect=lambda device: events.append("synchronize")),
              patch("train.torch.cuda.reset_peak_memory_stats", side_effect=lambda device: events.append("reset")),
              patch("train.torch.cuda.max_memory_allocated", side_effect=lambda device: events.append("peak") or 456),
              patch("train.time.monotonic", side_effect=[10.0, 12.0])):
            started = trainer._start_step_measurement()
            elapsed, peak = trainer._finish_step_measurement(started)
        self.assertEqual(events, ["synchronize", "reset", "synchronize", "peak"])
        self.assertEqual((elapsed, peak), (2.0, 456))

    def test_cpu_step_measurement_skips_cuda(self):
        trainer = self.trainer()
        with (patch("train.torch.cuda.synchronize", side_effect=AssertionError("CUDA used")),
              patch("train.torch.cuda.reset_peak_memory_stats", side_effect=AssertionError("CUDA used")),
              patch("train.torch.cuda.max_memory_allocated", side_effect=AssertionError("CUDA used"))):
            elapsed, peak = trainer._finish_step_measurement(trainer._start_step_measurement())
        self.assertGreater(elapsed, 0)
        self.assertEqual(peak, 0)

    def test_validation_preserves_parameters_and_mode(self):
        trainer = self.trainer()
        before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
        trainer.model.train()
        loss, perplexity = evaluate(trainer.model, self.loaders["validation"], torch.device("cpu"))
        self.assertAlmostEqual(perplexity, math.exp(loss))
        self.assertTrue(trainer.model.training)
        for old, parameter in zip(before, trainer.model.parameters()):
            torch.testing.assert_close(old, parameter)

    def test_clip_norm_caps_global_gradient(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3.0, 4.0])
        original = torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.0)
        self.assertAlmostEqual(float(original), 5.0)
        self.assertAlmostEqual(float(parameter.grad.norm()), 1.0, places=6)

    def test_small_data_pipeline_one_step(self):
        source = Path(__file__).resolve().parents[1] / "tokenizer/output-full/tokenizer.model"
        shutil.copyfile(source, self.path / "tokenizer.model")
        for split in ("train", "validation", "test"):
            (self.path / f"{split}.txt").write_text("日本語の学習確認です。\n" * 18, encoding="utf-8")
        config = replace(self.config, data_dir=str(self.path),
                         model_overrides={"hidden_size": 16, "num_heads": 2,
                                          "num_steps": 2, "intermediate_size": 32,
                                          "embedding_rank": 8})
        trainer = Trainer(config, device=torch.device("cpu"))
        with patch.object(trainer, "_log"):
            trainer.run(max_steps=1)
        self.assertEqual(trainer.global_step, 1)
        self.assertTrue((self.path / "last.pt").is_file())

    def test_resume_matches_uninterrupted_parameters_lr_and_order(self):
        full = self.trainer(seed=50)
        full.run(max_steps=3)
        expected = {name: value.detach().clone() for name, value in full.model.state_dict().items()}
        expected_lr = full.optimizer.param_groups[0]["lr"]
        partial = self.trainer(seed=50)
        partial.run(max_steps=1)
        resumed = self.trainer(seed=999)
        resumed.load_checkpoint(self.path / "last.pt")
        self.assertEqual(resumed.global_step, 1)
        resumed.run(max_steps=3)
        self.assertEqual(resumed.global_step, 3)
        self.assertEqual(resumed.optimizer.param_groups[0]["lr"], expected_lr)
        for name, value in resumed.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
        self.assertEqual(resumed.scheduler.completed_steps, 3)
        state = torch.load(self.path / "last.pt", weights_only=False)
        self.assertEqual(state["shuffle"]["batch_cursor"], resumed.batch_cursor)
        self.assertEqual(state["global_step"], 3)
        self.assertIsNone(state["scaler"])
        self.assertRaises(ValueError, self.trainer(config=replace(self.config, seed=99)).load_checkpoint,
                          self.path / "last.pt")

    def test_seed_and_precision_fallback(self):
        first = self.trainer(seed=33)
        second = self.trainer(seed=33)
        for left, right in zip(first.model.parameters(), second.model.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        bf16 = self.trainer(config=replace(self.config, precision="bf16"))
        self.assertFalse(bf16.use_bf16)
        bf16.run(max_steps=1)
        self.assertEqual(bf16.global_step, 1)

    def test_nonfinite_loss_stops_before_update(self):
        trainer = self.trainer()
        original = trainer.model.forward
        def invalid(*args, **kwargs):
            result = original(*args, **kwargs)
            result.loss = result.loss * float("nan")
            return result
        with patch.object(trainer.model, "forward", side_effect=invalid):
            self.assertRaises(FloatingPointError, trainer.run, 1)
        self.assertEqual(trainer.global_step, 0)

    def test_config_json_and_invalid_values(self):
        path = self.path / "config.json"
        path.write_text(json.dumps({"residual_type": "attnres", "betas": [0.9, 0.95]}), encoding="utf-8")
        loaded = TrainConfig.from_json(path)
        self.assertEqual(loaded.betas, (0.9, 0.95))
        self.assertEqual(loaded.residual_type, "attnres")
        self.assertRaises(ValueError, TrainConfig, sequence_length=1)
        self.assertRaises(ValueError, TrainConfig, warmup_fraction=1.5)


if __name__ == "__main__":
    unittest.main()
