# From Pixels to Feelings: Aligning MLLMs with Human Cognitive Perception

This repository contains the official code and data for **CogIP-Bench** (Cognition Image Property Benchmark) and the associated alignment methods described in the paper *"From Pixels to Feelings: Aligning MLLMs with Human Cognitive Perception of Images"*.

While Multimodal Large Language Models (MLLMs) excel at identifying "what" is in an image, they often struggle to understand "how" an image feels to a human observer. This project addresses that gap by evaluating and aligning models on subjective cognitive properties.

## 🧠 Project Overview
<img width="1100" height="590" alt="8fbca76d6c9e84a46ebf8f4ae850c900" src="https://github.com/user-attachments/assets/f11b6e93-c95a-478a-a217-f6942f9b3e4b" />

This framework focuses on four key dimensions of visual cognition:

1. **Aesthetics:** Visual appeal, harmony, and artistic value.
2. **Funniness:** Unexpected visual incongruity and humor.
3. **Emotional Valence:** The positive or negative emotional tone evoked by the image.
4. **Memorability:** How likely an image is to be remembered.

We provide tools for:
* **Benchmarking:** Evaluating MLLMs (Qwen, Llama, Gemma) against human judgment.
* **Alignment (SFT):** A training pipeline using **Soft-Label Loss** and a **"Describe-then-Predict"** strategy to teach models subjective cognition.
* **Generation:** Leveraging the aligned backbone to guide image generation (via Qwen-Image) toward specific cognitive traits.

---

## 📂 Directory Structure

The repository is organized into four main modules matching the workflow described in the paper.

```text
MLLM_Cognition_Alignment
├── data/                               # Dataset and Ground Truths
│   └── cognition/
│       ├── cognition_images/           # Raw image files
│       ├── cognition_scores/           # Ground truth scores across 4 cognition traits
│       ├── test_msg_file/              # Evaluation message files for models
│       │   ├── Aesthetics/
│       │   ├── Emotional_Valence/
│       │   <img width="1598" height="730" alt="ee85fa2b9236553ff0f98179c3500f4b" src="https://github.com/user-attachments/assets/56b2e11c-c8cf-4da6-8cbb-f094eada77fc" />
├── Funniness/
│       │   └── Memorability/
│       ├── cognition_training.json     # SFT dataset with Describe-then-Predict prompts
│       └── training_grpo.json          # RL dataset for GRPO experiments
│
├── evaluation/                         # Benchmarking Scripts
│   ├── gemma/
│   ├── llama/
│   └── qwen/
│
├── qwen-image/                         # Downstream Application: Image Generation
│   ├── prompts/
│   ├── batch_gene_image.py
│   └── run_batch.sh
│
├── sft/                                # Supervised Fine-Tuning Pipeline
│   ├── gemma/
│   ├── llama/
│   └── qwen/
│       ├── scripts/
│       └── src/
│
├── environment.yaml                    # Base environment description
└── requirements.txt                    # Python dependencies

```
> **Note:** Installation instructions are module-specific.  
> Please navigate into each subfolder (e.g., `sft/qwen/`, `evaluation/gemma/`, `qwen-image/`) to find scripts and guidance relevant to that component.

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
<img width="1095" height="378" alt="ceb5fa301f693f8f599d11c67d15ca87" src="https://github.com/user-attachments/assets/17e94a43-fb22-4a44-99c0-0a7293d11381" />

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

[Model Weight](https://huggingface.co/foolen/qwen2.5-vl-7b-cognition-full-sft) is provided (HuggingFace).
```bash
cd qwen-image
bash run_batch.sh
```
<img width="1578" height="890" alt="16b5a48e2028643c278bf962015db2fc" src="https://github.com/user-attachments/assets/db0c4cd3-f184-4b74-8055-dc8c0244ef5c" />  

**Figure:** Qualitative comparison of images generated by the Qwen-Image pipeline using different LLM backbones (same prompt).  

For each pair:  
- *Left: Base model; right: SFT model.*  
- *SFT backbones show stronger cognitive cue alignment in generated images.*



## 📊 Methodology Highlights

* **Describe-then-Predict:** We force the model to first generate a **descriptive label** (e.g., "very high aesthetic") before predicting the float score. This leverages the LLM's reasoning capabilities.
* **Soft-Label Loss:** Standard Cross-Entropy treats numbers as independent tokens. We implement a **soft-label distribution (triangular function)** to preserve numerical relationships during training, ensuring the model is penalized proportionally to the distance from the ground truth score.
