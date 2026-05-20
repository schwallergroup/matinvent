#!/bin/bash

[ ! -d "exp_res" ] && mkdir -p exp_res
# export HYDRA_FULL_ERROR=1  # for debug

EXPNAME="lemat-genbench-alex-mp-20"

nohup python -u main.py \
    expname=${EXPNAME} \
    pipeline=mat_invent \
    model=mattergen \
    reward=ehull \
    logger=wandb \
    device=cuda:0 \
    > exp_res/${EXPNAME}.log 2>&1 &
