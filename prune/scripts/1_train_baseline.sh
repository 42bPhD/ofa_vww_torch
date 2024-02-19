#!/usr/bin/env bash


source config.sh

python train.py --gpus ${GPUS} \
  --lr 1e-3 \
  --epochs 5 \
  --pretrained ${BASELINE_PATH} \
  --data_dir ${DATA_DIR} \
  --num_workers 8 \
  --batch_size 128 \
  --weight_decay 1e-4 \
  --momentum 0.9

