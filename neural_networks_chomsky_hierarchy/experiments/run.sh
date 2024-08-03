#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gk36
#PJM -j
#PJM --fs /work

export PYTHONPATH=$(pwd)/..

export WANDB_DATA_DIR="/work/gg45/g45004/Looped-Transformer/neural_networks_chomsky_hierarchy/tmp"
export WANDB_CACHE_DIR="/work/gg45/g45004/Looped-Transformer/neural_networks_chomsky_hierarchy/tmp"
export WANDB_CONFIG_DIR="/work/gg45/g45004/Looped-Transformer/neural_networks_chomsky_hierarchy/tmp"
export WANDB_API_KEY="f1462e37dc61bbcaa335f10a8dd966bbaec5423a"

source /work/gg45/g45004/.bashrc

python experiments/run.py