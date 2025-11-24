#!/bin/bash

python batch_gene_image.py \
  --prompts_json prompts/funniness.json \
  --out_dir ./img/funniness \
  --num_images 1 \


python batch_gene_image.py \
  --prompts_json prompts/aesthetics.json \
  --out_dir ./img/aesthetics \
  --num_images 1 \


python batch_gene_image.py \
  --prompts_json prompts/emotion.json \
  --out_dir ./img/emotion \
  --num_images 1 \


python batch_gene_image.py \
  --prompts_json prompts/memorability.json \
  --out_dir ./img/memorablity \
  --num_images 1 \

