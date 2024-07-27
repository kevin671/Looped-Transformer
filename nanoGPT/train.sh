#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=4
#PJM -g gg45
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc


export TRITON_CACHE_DIR="/work/gg45/g45004/Looped-Transformer/nanoGPT/tmp"
export WANDB_CONFIG_DIR="/work/gg45/g45004/Looped-Transformer/nanoGPT/tmp"
export WANDB_API_KEY="f1462e37dc61bbcaa335f10a8dd966bbaec5423a"

torchrun --standalone --nproc_per_node=4 train.py config/train_wikitext-103.py