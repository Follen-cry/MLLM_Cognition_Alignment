#!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-90B-Vision-Instruct"

# LLaMA3.2-Vision Does not support flash-attnetion2.

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path path_to_cognition_training.json \
    --image_folder path_to_cognition_images \
    --freeze_img_projector False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --bf16 True \
    --fp16 False \
    --output_dir path_to_store_model_checkpoint \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 9e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy no \
    --dataloader_num_workers 4 \
    --soft_label_enable True \
    --soft_label_dist triangular \
    --soft_label_eta 0.08 \
    --soft_label_lambda 2.0 
