# From Pixels to Feelings: Aligning MLLMs with Human Cognitive Perception

[cite_start]This repository contains the official code and data for **CogIP-Bench** (Cognition Image Property Benchmark) and the associated alignment methods described in the paper *"From Pixels to Feelings: Aligning MLLMs with Human Cognitive Perception of Images"*[cite: 1, 6].

[cite_start]While Multimodal Large Language Models (MLLMs) excel at identifying "what" is in an image, they often struggle to understand "how" an image feels to a human observer[cite: 4]. This project addresses that gap by evaluating and aligning models on subjective cognitive properties.

## 🧠 Project Overview

[cite_start]This framework focuses on four key dimensions of visual cognition[cite: 17, 145]:

1.  [cite_start]**Aesthetics:** Visual appeal, harmony, and artistic value[cite: 146].
2.  [cite_start]**Funniness:** Unexpected visual incongruity and humor[cite: 150].
3.  [cite_start]**Emotional Valence:** The positive or negative emotional tone evoked by the image[cite: 154].
4.  [cite_start]**Memorability:** How likely an image is to be remembered[cite: 161].

We provide tools for:
* [cite_start]**Benchmarking:** Evaluating MLLMs (Qwen, Llama, Gemma) against human judgment[cite: 177].
* [cite_start]**Alignment (SFT):** A training pipeline using **Soft-Label Loss** and a **"Describe-then-Predict"** strategy to teach models subjective cognition[cite: 197, 202].
* [cite_start]**Generation:** Leveraging the aligned backbone to guide image generation (via Qwen-Image) toward specific cognitive traits[cite: 30].

---

## 📂 Directory Structure

The repository is organized into four main modules matching the pipeline described in the paper.

```text
MLLM_Cognition_Alignment
├── data/                               # Dataset and Ground Truths
│   └── cognition/
│       ├── cognition_images/           # Raw image files
│       ├── cognition_scores/           # Ground truth scores (Aesthetics, Funniness, etc.)
│       ├── test_msg_file/              # Formatted evaluation files for different models
│       │   ├── Aesthetics/
│       │   ├── Emotional_Valence/
│       │   ├── Funniness/
│       │   └── Memorability/
[cite_start]│       ├── cognition_training.json     # SFT Dataset with "Describe-then-Predict" prompts [cite: 198]
[cite_start]│       └── training_grpo.json          # RL dataset for GRPO experiments 
│
[cite_start]├── evaluation/                         # Benchmarking Scripts [cite: 172]
│   ├── gemma/                          # Eval scripts for Gemma-3 variants
│   ├── llama/                          # Eval scripts for Llama-3.2-Vision
│   └── qwen/                           # Eval scripts for Qwen2/2.5-VL
│
[cite_start]├── qwen-image/                         # Downstream Application: Image Generation [cite: 232]
│   ├── prompts/                        # Prompts for generating cognition-aligned images
│   ├── batch_gene_image.py             # Inference script for image generation
│   └── run_batch.sh                    # Batch execution script
│
[cite_start]├── sft/                                # Supervised Fine-Tuning Pipeline [cite: 194]
│   ├── gemma/                          # Training code for Gemma
│   ├── llama/                          # Training code for Llama
│   └── qwen/                           # Training code for Qwen
│       ├── scripts/                    # Launch scripts (Deepspeed/Accelerate)
│       └── src/                        # Source code for Soft-Label Loss implementation
│
├── environment.yaml                    # Conda environment setup
└── requirements.txt                    # Python dependencies
```
---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/MLLM_Cognition_Alignment.git](https://github.com/your-username/MLLM_Cognition_Alignment.git)
    cd MLLM_Cognition_Alignment
    ```

2.  **Create the environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate cognition_align
    ```

---

## 🚀 Usage Guide

### 1. Data Preparation (`data/`)

The `data` folder contains the CogIP-Bench dataset components:

* **`cognition_training.json`**: Contains the training split (**3,200 examples**) formatted with the **"Describe-then-Predict"** prompts.
* **`training_grpo.json`**: Data used for the reinforcement learning (**Group Relative Policy Optimization**) ablation studies.
* **`test_msg_file/`**: Contains `.json` files pre-formatted for inference on the test split (**480 examples**).

### 2. Supervised Fine-Tuning (`sft/`)

We employ a custom SFT pipeline that uses **Soft-Label Loss** to handle the numerical nature of the scores. The code handles the conversion of regression targets into soft probability distributions over token space.

To train a model (e.g., Qwen2.5-VL), navigate to the relevant directory and run the script:

```bash
cd sft/qwen
bash scripts/finetune_lora.sh
```
### 3. Evaluation (`evaluation/`)

To benchmark a model's performance on the 4 cognitive dimensions:

1.  Navigate to the specific model folder (e.g., `evaluation/gemma`).
2.  Run the evaluation script which loads the model and iterates through the `test_msg_file`.

```bash
cd evaluation/gemma
bash cog_test.sh
```
> **Note:** Ensure you configure the path to `cognition_training.json` in the script.

### 4. Image Generation (`qwen-image/`)

This module demonstrates the **transferability of cognitive alignment**. It uses the SFT-aligned MLLM as the backbone for the Qwen-Image pipeline to generate images with specific emotional or aesthetic qualities.

```bash
cd qwen-image
bash run_batch.sh
```

## 📊 Methodology Highlights

* **Describe-then-Predict:** We force the model to first generate a **descriptive label** (e.g., "very high aesthetic") before predicting the float score. This leverages the LLM's reasoning capabilities.
* **Soft-Label Loss:** Standard Cross-Entropy treats numbers as independent tokens. We implement a **soft-label distribution (triangular function)** to preserve numerical relationships during training, ensuring the model is penalized proportionally to the distance from the ground truth score.