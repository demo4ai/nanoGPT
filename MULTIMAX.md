# MultiMax in nanoGPT

This fork supports three variants:

| Variant | `multimax` | Insertion points | Extra parameters (12 layers) |
|---|---|---|---:|
| Baseline | `none` | None | 0 |
| Attention MultiMax | `attention` | Each scaled QK score, before mask and softmax | 96 |
| Attention + LM Head MultiMax | `attention_lm_head` | Attention scores and final vocabulary logits | 104 |

The LM Head transform is applied during both training and generation. “Output”
means vocabulary logits, **not** the attention output projection. Ordinary Linear
and LayerNorm bias is controlled separately by upstream's `bias` setting.

## Transform

```python
y = (x + ts[0] * relu(ranges[0] - x)
       + ts[1] * relu(x - ranges[1])
       + ts[2] * relu(ranges[2] - x)**2
       + ts[3] * relu(x - ranges[3])**2)
```

Four thresholds and four coefficients are shared over an input tensor, with an
independent module per attention layer and one optional LM Head module. This
segmented-ReLU transform is **not** `torch.nn.SELU`. Thresholds and coefficients
use truncated-normal initialization (`std=0.02`); the initial transform is not
exactly the identity. Setting `ts` to zero makes it the identity.

MultiMax is applied to finite attention scores **before** the causal `-inf` mask.
Do not apply the quadratic terms to masked infinities. Softmax/cross-entropy
retain PyTorch's internal stabilization; this does not guarantee protection
against overflow inside the polynomial transform itself.

Common GPT weights and the subsequent CPU RNG state match across variants at
the same seed: new parameter initialization runs after common initialization,
inside an isolated RNG context. This is a new controlled implementation, **not a
claim of bitwise reproduction of the historical training runs**.

## Run one experiment

Install the upstream dependencies (`torch`, `numpy`; `datasets`, `tiktoken`,
`tqdm` for OpenWebText preparation), and prepare the data as described in the
upstream README. These commands do not download data automatically:

```bash
bash run_multimax.sh baseline
bash run_multimax.sh attention
bash run_multimax.sh attention_lm_head
```

Run them separately, or explicitly select separate GPUs. Logs are timestamped
under `logs/`; checkpoints go into distinct `out-*` directories. Do not run two
jobs against the same output directory. No long training is launched by tests.

Optional two-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash run_multimax.sh attention
```

`PYTHON_BIN` selects the interpreter. All arms share
`config/train_multimax_common.py`: 12 layers, width 768, 12 heads, sequence 1024,
micro-batch 8, accumulation budget 40, and 327,680 tokens/update. DDP divides
accumulation by world size; world size must divide 40. CLI overrides apply last;
use identical overrides for every arm. CUDA dtype selection follows upstream.

The matched configs use explicit (`manual`) attention for **all three** arms.
This controls backend differences but uses more memory and is slower than fused
attention. Outside these configs, `attention_backend='auto'` preserves upstream
SDPA for baseline; MultiMax attention always uses the explicit score path.
Matching total tokens does not guarantee identical trajectories across GPU counts.

The example retains upstream's 600,000-step schedule and loop semantics: a final
checkpoint is evaluated/saved at index 600,000, followed by upstream's last
unsaved update. No stopping-logic change is made in this fork. Adapt the schedule
and budget together for shorter runs. Historical experiments had differing
batch configurations; do not describe those results as a matched-budget ablation.

