import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as functional

from recursive_transformer import (
    CausalAttention, FeedForward, FullAttentionResiduals, ModelConfig,
    RecursiveTransformerLM, set_seed,
)


TOKENIZER = Path(__file__).resolve().parents[1] / "tokenizer/output-full/tokenizer.model"


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        set_seed(123)
        self.config = ModelConfig(vocab_size=31, hidden_size=16, num_steps=3,
                                  num_heads=2, intermediate_size=40, embedding_rank=8,
                                  pad_token_id=0)
        self.tokens = torch.tensor([[1, 4, 7, 2], [1, 8, 3, 2]])

    def models(self):
        for residual_type in ("residual", "attnres"):
            yield RecursiveTransformerLM(replace(self.config, residual_type=residual_type))

    def assert_finite_backward(self, model, tokens):
        result = model(tokens, labels=tokens)
        self.assertTrue(torch.isfinite(result.logits).all())
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.block.ffn.up.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.block.attention.query.weight.grad.abs().sum().item(), 0)

    def test_shapes_and_backward(self):
        for model in self.models():
            with self.subTest(kind=model.config.residual_type):
                self.assertEqual(model(self.tokens).shape, (2, 4, 31))
                self.assert_finite_backward(model, self.tokens)

    def test_default_sizes_and_query_count(self):
        expected = {"attention": 1644800, "ffn": 13107200,
                    "embedding": 3268608, "lm_head": 3268608,
                    "step_norms_and_biases": 43008, "final_norm": 640}
        for kind, total in (("residual", 21332864), ("attnres", 21353984)):
            with self.subTest(kind=kind), torch.device("meta"):
                model = RecursiveTransformerLM(ModelConfig(residual_type=kind))
                counts = model.parameter_counts()
                for name, value in expected.items():
                    self.assertEqual(counts[name], value, name)
                self.assertEqual(counts["total"], total)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), total)
                if kind == "attnres":
                    self.assertEqual(model.attnres.queries.shape, (33, 640))
                    self.assertEqual(counts["attnres_queries"], 21120)
                print(kind, counts)

    def test_shared_weights_used_at_every_step(self):
        for model in self.models():
            self.assertEqual(sum(isinstance(module, CausalAttention) for module in model.modules()), 1)
            self.assertEqual(sum(isinstance(module, FeedForward) for module in model.modules()), 1)
            for module in (model.block.attention, model.block.ffn):
                calls = []
                handle = module.register_forward_pre_hook(
                    lambda current, inputs: calls.append(tuple(id(param) for param in current.parameters()))
                )
                model(self.tokens)
                handle.remove()
                self.assertEqual(len(calls), self.config.num_steps)
                self.assertTrue(all(params == calls[0] for params in calls))

    def test_step_norms_and_biases_are_independent(self):
        for model in self.models():
            groups = (model.attn_norms, model.ffn_norms, model.query_norms, model.key_norms)
            parameters = [parameter for group in groups for norm in group for parameter in norm.parameters()]
            self.assertEqual(len(parameters), len({parameter.data_ptr() for parameter in parameters}))
            with torch.no_grad():
                model.attn_norms[0].bias.fill_(2)
            self.assertEqual(model.attn_norms[1].bias.abs().sum().item(), 0)
            self.assertTrue(all(module.bias is None for module in model.modules()
                                if isinstance(module, torch.nn.Linear)))

    def test_depth_attention_numerical_oracle_and_token_isolation(self):
        config = replace(self.config, hidden_size=4, num_heads=1)
        aggregator = FullAttentionResiduals(config)
        with torch.no_grad():
            aggregator.queries[0].copy_(torch.tensor([1., 0., 0., 0.]))
        first = torch.tensor([[[1., 0., 0., 0.], [-1., 0., 0., 0.]]], requires_grad=True)
        second = -first
        mixed, weights = aggregator([first, second], 0)
        expected_weights = torch.tensor([2., -2.]).softmax(0)
        torch.testing.assert_close(weights[0, 0], expected_weights, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(weights[0, 1], expected_weights.flip(0), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(mixed[0, 0], first[0, 0] * expected_weights[0]
                                   + second[0, 0] * expected_weights[1])
        changed = first.detach().clone()
        changed[:, 1] = 20
        other, _ = aggregator([changed, second.detach()], 0)
        torch.testing.assert_close(other[:, 0], mixed[:, 0])
        mixed[:, 0].sum().backward()
        self.assertEqual(first.grad[:, 1].abs().sum().item(), 0)
        self.assertGreater(first.grad[:, 0].abs().sum().item(), 0)

    def test_full_history_depth_weights_and_no_additive_residual(self):
        model = RecursiveTransformerLM(replace(self.config, residual_type="attnres"))
        outputs = []
        handles = [module.register_forward_hook(lambda current, inputs, result: outputs.append(result))
                   for module in (model.input_projection, model.block.attention, model.block.ffn)]
        result = model(self.tokens, return_details=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(result.depth_weights), 2 * self.config.num_steps + 1)
        for depth, weights in enumerate(result.depth_weights, start=1):
            self.assertEqual(weights.shape, (2, 4, depth))
            torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))
        mean_history = torch.stack(outputs).mean(0)
        expected = model.lm_head(model.head_projection(model.final_norm(mean_history)))
        torch.testing.assert_close(result.logits, expected)

    def test_readout_uses_dedicated_query(self):
        model = RecursiveTransformerLM(replace(self.config, residual_type="attnres"))
        query_indices = []
        handle = model.attnres.register_forward_pre_hook(
            lambda current, inputs: query_indices.append(inputs[1] % len(current.queries))
        )
        result = model(self.tokens, labels=self.tokens)
        handle.remove()
        self.assertEqual(query_indices, list(range(2 * self.config.num_steps + 1)))
        result.loss.backward()
        self.assertGreater(model.attnres.queries.grad[-1].abs().sum().item(), 0)

    def test_future_tokens_cannot_change_prefix(self):
        changed = self.tokens.clone()
        changed[:, 2:] = torch.tensor([[9, 10], [11, 12]])
        for model in self.models():
            torch.testing.assert_close(model(self.tokens)[:, :2], model(changed)[:, :2],
                                       atol=1e-7, rtol=1e-5)

    def test_left_right_and_all_padding(self):
        tokens = self.tokens[:1]
        for model in self.models():
            original = model(tokens)
            left = functional.pad(tokens, (2, 0))
            right = functional.pad(tokens, (0, 2))
            torch.testing.assert_close(model(left)[:, 2:], original, atol=1e-7, rtol=1e-5)
            torch.testing.assert_close(model(right)[:, :4], original, atol=1e-7, rtol=1e-5)
            mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
            left[:, :2] = 20
            torch.testing.assert_close(model(left, attention_mask=mask)[:, 2:], original,
                                       atol=1e-7, rtol=1e-5)
            padded = torch.zeros_like(self.tokens)
            result = model(padded, labels=padded)
            self.assertTrue(torch.isfinite(result.logits).all())
            self.assertEqual(result.loss.item(), 0)
            result.loss.backward()
            self.assertTrue(all(torch.isfinite(param.grad).all() for param in model.parameters()))

    def test_shifted_loss_and_ignored_labels(self):
        tokens = torch.tensor([[0, 1, 4, 2, 0], [1, 7, 2, 0, 0]])
        labels = tokens.clone()
        labels[0, 2] = -100
        for model in self.models():
            result = model(tokens, labels=labels)
            prediction = torch.stack([result.logits[0, 2], result.logits[1, 0], result.logits[1, 1]])
            target = torch.tensor([2, 7, 2])
            torch.testing.assert_close(result.loss, functional.cross_entropy(prediction, target))
            ignored = model(tokens, labels=torch.full_like(tokens, -100))
            self.assertEqual(ignored.loss.item(), 0)
            singleton = model(tokens[:, :1], labels=tokens[:, :1])
            self.assertEqual(singleton.loss.item(), 0)

    def test_final_ffn_changes_logits_and_receives_gradient(self):
        for model in self.models():
            saved = []

            def capture(current, inputs, output):
                output.retain_grad()
                saved.append(output)

            handle = model.block.ffn.register_forward_hook(capture)
            baseline = model(self.tokens, labels=self.tokens)
            baseline.loss.backward()
            handle.remove()
            self.assertEqual(len(saved), self.config.num_steps)
            for output in saved:
                self.assertIsNotNone(output.grad)
                self.assertGreater(output.grad.abs().sum().item(), 0)
            calls = []

            def perturb_last(current, inputs, output):
                calls.append(True)
                if len(calls) == self.config.num_steps:
                    return output + torch.arange(output.shape[-1], device=output.device) * 0.1
                return output

            handle = model.block.ffn.register_forward_hook(perturb_last)
            changed = model(self.tokens)
            handle.remove()
            self.assertGreater((baseline.logits - changed).abs().max().item(), 1e-5)

    def test_embedding_and_lm_head_are_untied(self):
        for model in self.models():
            before = model.lm_head.weight.clone()
            self.assertNotEqual(model.embedding.weight.data_ptr(), model.lm_head.weight.data_ptr())
            self.assertNotEqual(model.input_projection.weight.data_ptr(), model.head_projection.weight.data_ptr())
            with torch.no_grad():
                model.embedding.weight.fill_(2)
            torch.testing.assert_close(model.lm_head.weight, before)

    def test_bfloat16_native_and_autocast_backward(self):
        for native in (False, True):
            for model in self.models():
                with self.subTest(native=native, kind=model.config.residual_type):
                    if native:
                        model = model.to(torch.bfloat16)
                    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not native):
                        result = model(self.tokens, labels=self.tokens, return_details=True)
                    self.assertEqual(result.logits.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(result.logits).all())
                    self.assertTrue(torch.isfinite(result.loss))
                    result.loss.backward()
                    self.assertTrue(all(torch.isfinite(param.grad).all() for param in model.parameters()))
                    for weights in result.depth_weights:
                        self.assertEqual(weights.dtype, torch.float32)

    def test_seed_reproducibility(self):
        for kind in ("residual", "attnres"):
            config = replace(self.config, residual_type=kind)
            set_seed(42)
            first = RecursiveTransformerLM(config)(self.tokens)
            set_seed(42)
            second = RecursiveTransformerLM(config)(self.tokens)
            torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_real_tokenizer_metadata(self):
        import sentencepiece as sentencepiece

        processor = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER))
        config = ModelConfig.from_tokenizer(TOKENIZER)
        self.assertEqual(config.vocab_size, processor.vocab_size())
        for field, method in (("pad_token_id", "pad_id"), ("bos_token_id", "bos_id"),
                              ("eos_token_id", "eos_id"), ("unk_token_id", "unk_id")):
            expected = getattr(processor, method)()
            self.assertEqual(getattr(config, field), expected if expected >= 0 else None)
        self.assertRaises(ValueError, ModelConfig.from_tokenizer, TOKENIZER, vocab_size=12)

    def test_production_models_forward_backward(self):
        for kind in ("residual", "attnres"):
            with self.subTest(kind=kind):
                model = RecursiveTransformerLM(ModelConfig.from_tokenizer(TOKENIZER, residual_type=kind))
                self.assert_finite_backward(model, torch.tensor([[1, 4, 7, 2]]))

    def test_invalid_configs_and_input_shapes(self):
        for changes in ({"num_steps": 0}, {"hidden_size": 15}, {"residual_type": "other"},
                        {"pad_token_id": 31}, {"norm_eps": float("nan")}):
            self.assertRaises(ValueError, replace, self.config, **changes)
        model = RecursiveTransformerLM(self.config)
        self.assertRaises(ValueError, model, torch.zeros(0, 2, dtype=torch.long))
        self.assertRaises(ValueError, model, self.tokens, attention_mask=torch.ones(2, 3))
        self.assertRaises(ValueError, model, self.tokens, labels=torch.ones(2, 3))
        self.assertRaises(ValueError, model, self.tokens, attention_mask=torch.full((2, 4), 2))


if __name__ == "__main__":
    unittest.main()
