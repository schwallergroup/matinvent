#!/bin/bash

# Please modify the following environment variables
RESULT_PATH="YOUR FOLDER PATH TO SAVE RESULTS"
MODEL_PATH="YOUR FOLDER PATH OF MODEL CHECKPOINT"  # e.g., exp_res/test/models/loop_0099
REFERENCE_PATH="YOUR FILE PATH OF REFERENCE ENERGY DATASET"  # reference_MP2020correction.gz from MatterGen (https://github.com/microsoft/mattergen)
# e.g., download it by the command: wget https://github.com/microsoft/mattergen/raw/main/data-release/alex-mp/reference_MP2020correction.gz
mkdir -p $RESULT_PATH

# 1. Unconditional/RL-finetuned MatterGen model to generate crystal structures
# Note: Please modify the batch_size and num_batches parameters
# based on your GPU memory and the number of samples you want to generate.
mattergen-generate \
    --output_path=$RESULT_PATH \
    --model_path=$MODEL_PATH \
    --batch_size=32 \
    --num_batches=32 \
    --record_trajectories=False \
    > $RESULT_PATH/generation.log 2>&1

# 2. evaluate the generated structure to obtain SUN ratio
mattergen-evaluate \
    --structures_path=$RESULT_PATH/generated_crystals.extxyz \
    --relax=True \
    --structure_matcher='disordered' \
    --potential_load_path="MatterSim-v1.0.0-5M.pth" \
    --reference_dataset_path=$REFERENCE_PATH \
    --save_as="$RESULT_PATH/metrics.json"
