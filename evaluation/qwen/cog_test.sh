#!/bin/bash

#MODEL="path to your sft model"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# This test the alignment of the base model:
python cog_test.py \
    --model-id "$MODEL" \
    --base-dir "path to test_msg_file folder" \
    --img-base "path to cognition_images folder" \
    --out-dir "./output/${MODEL}" \
    --sample-n 120  \
    --seed 19


