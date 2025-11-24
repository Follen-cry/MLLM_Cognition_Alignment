# #!/bin/bash

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
# MODEL_NAME="meta-llama/Llama-3.2-90B-Vision-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path  /scratch/local/ssd/junlin/models/Llama/soft_label/Llama-3.2-11B-Vision-sft_3e-4_0.08_2.0 \
    --model-base $MODEL_NAME  \
    --save-model-path  /scratch/local/ssd/junlin/models/Llama-3.2-11B-Vision-sft_3e-4_0.08_2.0_full \
    --safe-serialization