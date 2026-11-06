# Matched nanoGPT/GPT-2-small ablation budget (not a historical-run reproduction).
# Apply this file BEFORE exactly one of the three variant configs.
init_from = 'scratch'
dataset = 'openwebtext'
n_layer = 12
n_head = 12
n_embd = 768
bias = False
dropout = 0.0
seed = 1337
attention_backend = 'manual'
# 8 * 40 * 1024 = 327,680 tokens/update, independent of DDP world size.
# train.py divides accumulation by world size; it must divide 40 exactly.
batch_size = 8
gradient_accumulation_steps = 40
block_size = 1024
learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 2000
max_iters = 600000
lr_decay_iters = 600000
decay_lr = True
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
eval_interval = 2000
eval_iters = 200
log_interval = 10
always_save_checkpoint = True
eval_only = False
wandb_log = False
