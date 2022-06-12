#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR=r'C:\Users\AHMED\Desktop\Bach\audio\audio'

# You need to modify this path to your workspace to store features and models
WORKSPACE=r'C:\Users\AHMED\Desktop\Bach'

# Hyper-parameters
GPU_ID=0
MODEL_TYPE='Cnn_9layers_AvgPooling'
BATCH_SIZE=32



############ Train and validate on development dataset ############

# Train & inference
for TAXONOMY_LEVEL in 'fine' 'coarse'
do
  # Train
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold=1 --batch_size=$BATCH_SIZE --cuda

  # Inference and evaluate
  CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --taxonomy_level=$TAXONOMY_LEVEL --model_type=$MODEL_TYPE --holdout_fold=1 --iteration=10000 --batch_size=$BATCH_SIZE --cuda
done

# Plot statistics
#python utils/plot_results.py --workspace=$WORKSPACE --taxonomy_level=fine



