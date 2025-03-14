#!/bin/sh
#SBATCH --job-name=molnet
#SBATCH --partition=a800
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/loss_%j.out
#SBATCH --error=./logs/loss_%j.err



python msformer_finetune.py \
  --device cuda \
  --batch_size 8 \
  --n_head 12 \
  --n_layer 12 \
  --n_embd 768 \
  --d_dropout 0.1 \
  --dropout 0.1 \
  --lr_start 3e-5 \
  --num_workers 16 \
  --max_epochs 50 \
  --num_feats 32 \
  --maxfrags 500 \
  --expname BACE \
  --dataset_name bace \
  --data_root "./datasets/molnet/bace/" \
  --attention_type full \
  --measure_name Class \
  --checkpoints_folder './checkpoints_bace' \
  --num_classes 2  
  
  
python msformer_finetune.py \
  --device cuda \
  --batch_size 8 \
  --n_head 12 \
  --n_layer 12 \
  --n_embd 768 \
  --d_dropout 0.1 \
  --dropout 0.1 \
  --lr_start 3e-5 \
  --num_workers 16 \
  --max_epochs 50 \
  --num_feats 32 \
  --maxfrags 500 \
  --expname BBBP \
  --dataset_name bbbp \
  --data_root "./datasets/molnet/bbbp/" \
  --attention_type full \
  --measure_name p_np \
  --checkpoints_folder './molnet/checkpoints_bbbp' \
  --num_classes 2  