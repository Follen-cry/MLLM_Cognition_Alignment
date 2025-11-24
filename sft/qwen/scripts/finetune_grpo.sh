#!/bin/bash
# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
 MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME=/scratch/local/ssd/junlin/models/Qwen2.5-vl-7b-cognition_sft

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed src/train/train_grpo.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /homes/55/junlin/mllm_benchmark_project/Qwen2-VL-Finetune/data/cognition_training.json \
    --image_folder /homes/55/junlin/mllm_benchmark_project/Qwen2-VL-Finetune/data/cognition/cognition_images \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /scratch/network/ssd2/junlin/models/qwen2.5-vl-7b-cognition_unfreezellm_grpo \
    --num_train_epochs 1 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 256 \
    --max_prompt_length 512 \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((256 * 28 * 28)) \
    --learning_rate 1e-6 \
    --vision_lr 2e-7 \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy no \
    --dataloader_num_workers 4 \