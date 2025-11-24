#!/bin/bash

MODEL="google/gemma-3-12b-it"

#Base model evaluation
python gemma_cog.py \
--base-dir "path to test_msg_file folder" \
--img-base "path to cognition_images folder" \
--out-dir "./output/${MODEL}" \
--sample-n 120  \


python gemma_cog.py \
--lora-path "path to the sft lora model" \
--base-dir "path to test_msg_file folder" \
--img-base "path to cognition_images folder" \
--out-dir "./output/sft" \
--sample-n 120  \

