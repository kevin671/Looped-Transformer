init_from = "hyper_gpt"

n_layer = 100
n_head = 12
n_embd = 768

hyper_size = 256
n_z = 64

wandb_log = True
wandb_project = "perplexity"
wandb_run_name = f"hypergpt_{n_layer}_layer"

out_dir = f"out/hypergpt_{n_layer}_layer"

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

learning_rate = 1e-4
