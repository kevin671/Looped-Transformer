# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
import uuid

init_from = "looped_gpt"

n_loops = 12  # 100
n_layer = 1
n_head = 16
n_embd = 2048
use_input_injection = True

unique_suffix = uuid.uuid4().hex[:4]  # ランダムな8文字の文字列を生成

wandb_log = True
wandb_project = "perplexity"
wandb_run_name = f"looped_{n_layer}_layer_{n_loops}_loop_wikitext-103_{unique_suffix}"

out_dir = f"out/looped_{n_layer}_layer_{n_loops}_loop_wikitext-103_{unique_suffix}"

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 6
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
