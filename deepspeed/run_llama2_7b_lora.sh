#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ZERO_STAGE=$1
if [[ "$ZERO_STAGE" =~ ^[0-9]+$ ]]; then
    shift
else
    ZERO_STAGE=3
fi
OUTPUT=./output_llama2_7b_lora
mkdir -p $OUTPUT

deepspeed main.py \
   --data_split 2,4,4 \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 256 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --dtype bf16 \
   --zero_stage $ZERO_STAGE \
   --lora_dim 16 \
   --lora_module_name "model.layers." \
   --only_optimize_lora \
   --lora_learning_rate 3e-4 \
   --deepspeed \
   --output_dir $OUTPUT \
   "$@" \
   #&> $OUTPUT/training.log
