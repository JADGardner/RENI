#! /bin/bash

export CUDA_VISIBLE_DEVICES=0
export WOLRD_SIZE=1

torchrun \
  --nproc_per_node ${WOLRD_SIZE} \
  --nnodes 1 \
  --node_rank 0 \
  --master_port 47775 \
  src/training/RENI_Training.py \
  --world-size ${WOLRD_SIZE}