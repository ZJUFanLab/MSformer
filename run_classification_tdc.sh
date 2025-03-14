#!/bin/sh
#SBATCH --job-name=tdc_hts
#SBATCH --partition=a800
#SBATCH --ntasks=16
#SBATCH --gres=gpu:2
#SBATCH --output=./logs/diceloss_%j.out
#SBATCH --error=./logs/diceloss_%j.err





python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name cav3_t-type_calcium_channels_butkiewicz \
        --data_root "./datasets/sampled_tdc/knn3/cav3_t-type_calcium_channels_butkiewicz/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_cav3_t-type_calcium_channels_butkiewicz'\
        --num_classes 2 \
        

python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name choline_transporter_butkiewicz \
        --data_root "./datasets/sampled_tdc/knn3/choline_transporter_butkiewicz/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_choline_transporter_butkiewicz'\
        --num_classes 2 \

python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name kcnq2_potassium_channel_butkiewicz \
        --data_root "./datasets/sampled_tdc/knn3/kcnq2_potassium_channel_butkiewicz/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_kcnq2_potassium_channel_butkiewicz'\
        --num_classes 2 \
        
python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name m1_muscarinic_receptor_agonists_butkiewicz \
        --data_root "./datasets/sampled_tdc/knn3/m1_muscarinic_receptor_agonists_butkiewicz/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_m1_muscarinic_receptor_agonists_butkiewicz'\
        --num_classes 2 \

python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name orexin1_receptor_butkiewicz \
        --data_root "./datasets/sampled_tdc/knn3/orexin1_receptor_butkiewicz/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_orexin1_receptor_butkiewicz'\
        --num_classes 2 \
        
python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name potassium_ion_channel_kir2 \
        --data_root "./datasets/sampled_tdc/knn3/potassium_ion_channel_kir2/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_potassium_ion_channel_kir2'\
        --num_classes 2 \
        
python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name sarscov2_3clpro_diamond \
        --data_root "./datasets/sampled_tdc/knn3/sarscov2_3clpro_diamond/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_sarscov2_3clpro_diamond'\
        --num_classes 2 \
        
       
python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name serine_threonine_kinase_33_butkiewicz \
        --data_root "./datasets/sampled_tdc/knn3/serine_threonine_kinase_33_butkiewicz/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_serine_threonine_kinase_33_butkiewicz'\
        --num_classes 2 \
        
python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name tyrosyl-dna_phosphodiesterase_butkiewicz \
        --data_root "./datasets/sampled_tdc/knn3/tyrosyl-dna_phosphodiesterase_butkiewicz/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/checkpoints_tyrosyl-dna_phosphodiesterase_butkiewicz'\
        --num_classes 2 \
        
        
python msformer_finetune.py \
        --device cuda \
        --batch_size 8  \
        --n_head 12\
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 16\
        --max_epochs 50 \
        --num_feats 32 \
        --maxfrags 1000 \
        --expname attention \
        --dataset_name hiv \
        --data_root "./datasets/sampled_tdc/knn3/hiv/" \
        --measure_name  labels \
        --checkpoints_folder './tdc/hts/hiv'\
        --num_classes 2 \