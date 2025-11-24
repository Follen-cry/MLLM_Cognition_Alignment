# #!/bin/bash

MODEL_NAME="google/gemma-3-4b-it"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /home/yuwon/workspace/Gemma3-Finetune/output/test_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /home/yuwon/workspace/Gemma3-Finetune/output/merge_test \
    --safe-serialization