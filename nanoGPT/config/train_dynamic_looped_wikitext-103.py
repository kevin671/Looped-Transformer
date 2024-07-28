# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

init_from = "dynamic_looped_gpt"

wandb_log = True
wandb_project = "perplexity"
wandb_run_name = "dynamic_looped_wikitext-103"

# model config
n_loops = 100
n_layer = 1
n_head = 12
n_embd = 768

out_dir = f"out/dynamic_looped_{n_loops}_wikitext-103"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 2
block_size = 1024
gradient_accumulation_steps = 10 * 4

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
