#!/bin/bash

[ ! -d "exp_res" ] && mkdir -p exp_res
# export HYDRA_FULL_ERROR=1  # for debug

EXPNAME="test"

nohup python -u main.py \
    expname=${EXPNAME} \
    pipeline=mat_invent \
    model=mattergen \
    reward=hhi \
    logger=wandb \
    device=cuda:0 \
    > exp_res/${EXPNAME}.log 2>&1 &
