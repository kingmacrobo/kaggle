#!/usr/bin/env bash
BASE_DIR=/home/wanghongbo/workspace/kaggle/carvana_image_masking_challenge
python train.py --model_dir=${BASE_DIR}/checkpoints \
                --out_mask_dir=${BASE_DIR}/validate_out_mask \
                --train_list=${BASE_DIR}/data/train/imgs.txt \
                --test_list=${BASE_DIR}/data/train/imgs.txt \
                --train_mask_dir=${BASE_DIR}/data/train_masks \
                --train_mask_edge_dir=${BASE_DIR}/data/train_mask_edge \
                --debug_dir=${BASE_DIR}/debug
