import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig, MultiMax

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('none', 'attention', 'attention_lm_head')
torch.set_num_threads(1)


def make_model(variant='none', backend='manual'):
    torch.manual_seed(1337)
    with contextlib.redirect_stdout(io.StringIO()):
        return GPT(GPTConfig(block_size=8, vocab_size=32, n_layer=2, n_head=2,
                             n_embd=16, dropout=0.0, bias=False,
                             multimax=variant, attention_backend=backend))


class MultiMaxTests(unittest.TestCase):
    def test_formula_and_identity(self):
        module = MultiMax()
        x = torch.tensor([-3., -1., .2, 2., 4.], requires_grad=True)
        self.assertTrue(torch.equal(module(x), x))
        with torch.no_grad():
            module.ranges.copy_(torch.tensor([-.5, .5, -1.5, 1.5]))
            module.ts.copy_(torch.tensor([.1, -.2, .03, -.04]))
        expected = x + .1*torch.relu(-.5-x) - .2*torch.relu(x-.5) + .03*torch.relu(-1.5-x)**2 - .04*torch.relu(x-1.5)**2
        torch.testing.assert_close(module(x), expected)
        module(x).sum().backward()
        for p in [x, *module.parameters()]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(set(dict(module.named_parameters())), {'ranges', 'ts'})

    def test_counts_common_weights_and_rng(self):
        baseline = make_model()
        rng = torch.get_rng_state().clone()
        state = baseline.state_dict()
        for variant, extra, modules in [('attention', 16, 2), ('attention_lm_head', 24, 3)]:
            model = make_model(variant)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(model.get_num_params(False)-baseline.get_num_params(False), extra)
            self.assertEqual(sum(isinstance(m, MultiMax) for m in model.modules()), modules)
            for key, value in state.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
            self.assertIs(model.lm_head.weight, model.transformer.wte.weight)

    def test_backend_and_identity_equivalence(self):
        x = torch.arange(8).view(1, 8)
        baseline = make_model().eval()
        expected, expected_loss = baseline(x, x)
        auto = make_model(backend='auto').eval()
        self.assertEqual(auto.transformer.h[0].attn.flash, hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        actual, loss = auto(x, x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(loss, expected_loss)
        for variant in VARIANTS[1:]:
            model = make_model(variant).eval()
            self.assertFalse(model.transformer.h[0].attn.flash)
            for module in model.modules():
                if isinstance(module, MultiMax):
                    with torch.no_grad():
                        module.ts.zero_()
            actual, loss = model(x, x)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(loss, expected_loss, atol=0, rtol=0)

    def test_causality_gradients_and_inference(self):
        x = torch.arange(8).repeat(2, 1)
        y = (x+1) % 32
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = make_model(variant).eval()
                logits, loss = model(x, y)
                last, _ = model(x)
                torch.testing.assert_close(last, logits[:, [-1], :])
                changed = x.clone()
                changed[:, 4:] = 25
                alternative, _ = model(changed, y)
                torch.testing.assert_close(logits[:, :4], alternative[:, :4], atol=0, rtol=0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                with contextlib.redirect_stdout(io.StringIO()):
                    opt = model.configure_optimizers(.1, 1e-3, (.9, .95), 'cpu')
                included = [id(p) for g in opt.param_groups for p in g['params']]
                self.assertEqual(len(included), len(set(included)))
                self.assertEqual(set(included), {id(p) for p in model.parameters()})
                opt.step()
                self.assertEqual(model.generate(x[:, :2], max_new_tokens=2).shape, (2, 4))
                model.crop_block_size(4)
                # Upstream's CE .view() expects contiguous targets, as get_batch supplies.
                self.assertTrue(torch.isfinite(model(x[:, :4], y[:, :4].contiguous())[1]))

    def test_checkpoint_roundtrip_and_legacy_config(self):
        x = torch.arange(8).view(1, 8)
        for variant in VARIANTS:
            model = make_model(variant).eval()
            buffer = io.BytesIO()
            torch.save({'model_args': asdict(model.config), 'model': model.state_dict()}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, weights_only=True)
            with contextlib.redirect_stdout(io.StringIO()):
                restored = GPT(GPTConfig(**checkpoint['model_args'])).eval()
            restored.load_state_dict(checkpoint['model'], strict=True)
            torch.testing.assert_close(restored(x)[0], model(x)[0], atol=0, rtol=0)
        baseline = make_model(backend='auto')
        old_config = asdict(baseline.config)
        del old_config['multimax'], old_config['attention_backend']
        with contextlib.redirect_stdout(io.StringIO()):
            old = GPT(GPTConfig(**old_config))
        old.load_state_dict(baseline.state_dict(), strict=True)
        self.assertEqual(old.config.multimax, 'none')

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            GPTConfig(multimax='bias')
        with self.assertRaises(ValueError):
            GPTConfig(attention_backend='unknown')

    def test_matched_configs(self):
        common = {}
        exec((ROOT/'config/train_multimax_common.py').read_text(), common)
        common.pop('__builtins__')
        settings = []
        for suffix in ('baseline', 'attention', 'attention_lm_head'):
            cfg = dict(common)
            exec((ROOT/f'config/multimax_{suffix}.py').read_text(), cfg)
            settings.append(cfg)
        for cfg in settings:
            self.assertEqual(cfg['batch_size']*cfg['gradient_accumulation_steps']*cfg['block_size'], 327680)
            self.assertEqual(cfg['attention_backend'], 'manual')
        for k in common:
            self.assertTrue(all(cfg[k] == common[k] for cfg in settings), k)


class TrainingSmokeTests(unittest.TestCase):
    def test_train_and_resume_three_variants(self):
        with tempfile.TemporaryDirectory(prefix='nanogpt_smoke_') as tmp:
            root = Path(tmp)
            for name in ('model.py', 'train.py', 'configurator.py'):
                shutil.copyfile(ROOT/name, root/name)
            data = root/'data'/'fixture'
            data.mkdir(parents=True)
            tokens = (np.arange(256) % 32).astype(np.uint16)
            tokens.tofile(data/'train.bin')
            tokens[::-1].tofile(data/'val.bin')
            import pickle
            with (data/'meta.pkl').open('wb') as f:
                pickle.dump({'vocab_size': 32}, f)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            shared = [sys.executable, 'train.py', '--dataset=fixture', '--device=cpu', '--dtype=float32',
                      '--compile=False', '--n_layer=2', '--n_head=2', '--n_embd=16', '--block_size=8',
                      '--batch_size=2', '--gradient_accumulation_steps=2', '--eval_iters=1',
                      '--eval_interval=1', '--log_interval=1', '--warmup_iters=0', '--lr_decay_iters=4',
                      '--wandb_log=False', '--attention_backend=manual']
            for variant in VARIANTS:
                output = root/f'out-{variant}'
                args = shared + [f'--out_dir={output}', f'--multimax={variant}', '--max_iters=2']
                run = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=90)
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                ckpt = torch.load(output/'ckpt.pt', weights_only=True, map_location='cpu')
                self.assertEqual(ckpt['iter_num'], 2)
                self.assertEqual(ckpt['model_args']['multimax'], variant)
                self.assertTrue(ckpt['optimizer']['state'])
                # Defaults intentionally differ; checkpoint architecture must win.
                args = shared + [f'--out_dir={output}', '--multimax=none', '--attention_backend=auto',
                                 '--init_from=resume', '--max_iters=3']
                run = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=90)
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                ckpt = torch.load(output/'ckpt.pt', weights_only=True, map_location='cpu')
                self.assertEqual(ckpt['iter_num'], 3)
                self.assertEqual(ckpt['model_args']['multimax'], variant)
                self.assertEqual(ckpt['config']['multimax'], variant)
                self.assertEqual(ckpt['model_args']['attention_backend'], 'manual')


if __name__ == '__main__':
    unittest.main()
