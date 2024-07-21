#!/bin/sh
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -g gg45
#PJM -j
#PJM -m b
#PJM -m e
#PJM --fs /work

source /work/gg45/g45004/.bashrc

python function_approximation.py