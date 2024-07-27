#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -g gg45
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

python train.py --net hypernet_vit