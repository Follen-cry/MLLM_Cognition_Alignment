#!/bin/bash

MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
#MODEL="path to your sft model"

python new_llama_cog.py \
    --model-id  "$MODEL" \
    --base-dir "path to test_msg_file folder" \
    --img-base "path to cognition_images folder" \
    --out-dir "./output/${MODEL}" \
    --sample-n 120  \
    --seed 19



