# gemma3_eval_lora.py
# Gemma-3 multimodal evaluation w/ optional LoRA adapters.

from transformers import AutoModelForCausalLM, AutoProcessor, logging
from peft import PeftModel
from PIL import Image
from scipy.stats import spearmanr
import torch
import numpy as np
import json
import re
import os
from tqdm import tqdm
import random
import argparse


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_message_and_answer(message_text, answer_text, output_file):
    with open(output_file, "a") as f:
        f.write("=== Message ===\n")
        f.write(message_text.strip() + "\n")
        f.write("--- Answer ---\n")
        f.write(answer_text.strip() + "\n")
        f.write("===============\n\n")


def extract_pil_images(messages):
    """Collect PIL images from the chat messages (in order)."""
    imgs = []
    for turn in messages:
        content = turn.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "image" in item:
                    path = item["image"]
                    imgs.append(Image.open(path).convert("RGB"))
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-3-12b-it",
                        help="Gemma-3 VLM checkpoint (e.g., google/gemma-3-6b-it or 27b-it).")
    parser.add_argument("--lora-path", default=None,
                        help="Path to LoRA adapters. If given, the base model is wrapped with PeftModel.")
    parser.add_argument("--merge-lora", action="store_true",
                        help="Merge LoRA weights into the base model and unload adapters for faster inference.")
    parser.add_argument("--base-dir", default="/homes/55/junlin/mllm_benchmark_project/Qwen2-VL-Finetune/data/test_msg_file")
    parser.add_argument("--img-base", default="/homes/55/junlin/mllm_benchmark_project/Qwen2-VL-Finetune/data/cognition/cognition_images")
    parser.add_argument("--out-dir",  default="/scratch/local/ssd/junlin/output/cognition/sft")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Number of examples to sample per subtask (random without replacement). Use all if omitted.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    logging.set_verbosity_error()

    # ----- Load model + processor (Gemma-3 multimodal) -----
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    # Optional: attach LoRA adapters
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path, local_files_only=True)
        if args.merge_lora:
            # Merge LoRA into the base weights for faster generation
            model = model.merge_and_unload()

    # Ensure a pad token id exists for generation
    if getattr(model.config, "pad_token_id", None) is None:
        eos_id = getattr(processor, "eos_token_id", None) or processor.tokenizer.eos_token_id
        model.config.pad_token_id = eos_id

    model.eval()

    # ----- List subtasks -----
    subtasks = [
        f for f in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, f))
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    for subtask in subtasks:
        print(f"\n=== 🏷️ Evaluating subtask: {subtask} ===")
        task_path = os.path.join(args.base_dir, subtask)
        messages_file = os.path.join(task_path, f"{subtask}_messages.json")
        scores_file   = os.path.join(task_path, f"{subtask}_scores.json")

        with open(messages_file, "r") as f:
            messages_list = json.load(f)
        with open(scores_file, "r") as f:
            ref_scores = json.load(f)

        assert len(messages_list) == len(ref_scores), "Mismatch between messages and scores!"

        # ---- Sampling (per subtask) ----
        N = len(messages_list)
        if args.sample_n is not None and args.sample_n < N:
            idxs = sorted(random.sample(range(N), args.sample_n))
            messages_list = [messages_list[i] for i in idxs]
            ref_scores    = [ref_scores[i]    for i in idxs]
            print(f"Sampling {len(idxs)}/{N} items for subtask {subtask}.")
            with open(os.path.join(args.out_dir, f"sample_indices_{subtask}.json"), "w") as f:
                json.dump(idxs, f)
        else:
            print(f"Using all {N} items for subtask {subtask}.")

        # Output file (include sample size suffix if applicable)
        suffix = f"_n{len(messages_list)}" if args.sample_n else ""
        output_file = os.path.join(args.out_dir, f"sft2_results_{subtask}{suffix}.txt")

        # Clear old file before appending
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write("")

        model_scores = []
        valid_ref_scores = []

        for msg, true_score in tqdm(zip(messages_list, ref_scores), total=len(messages_list), desc="Processing"):
            # Fix up image path to absolute path and sanity-check it opens
            image_relative_path = msg[0]["content"][0]["image"]
            image_path = os.path.join(args.img_base, image_relative_path)
            msg[0]["content"][0]["image"] = image_path
            _ = Image.open(image_path).convert("RGB")

            # Build prompt via chat template; provide images separately
            messages = msg
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pil_images = extract_pil_images(messages)

            inputs = processor(
                text=[text],
                images=pil_images if len(pil_images) > 0 else None,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=100
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            # Save message and answer
            save_message_and_answer(text, answer, output_file)

            # Parse numbers from prediction and reference
            pred_matches = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', answer)
            pred_score = float(pred_matches[0]) if pred_matches else np.nan

            true_matches = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', true_score)
            true_score_f = float(true_matches[0]) if true_matches else np.nan

            model_scores.append(pred_score)
            valid_ref_scores.append(true_score_f)

        # ---- Metrics (Spearman + MAE/MSE) ----
        pairs = [(m, r) for m, r in zip(model_scores, valid_ref_scores) if not (np.isnan(m) or np.isnan(r))]
        if not pairs:
            print("❗ No valid scores parsed.")
            continue

        preds, golds = zip(*pairs)
        rho, p = spearmanr(preds, golds)
        rho = np.round(rho, 4)
        p   = np.round(p,   4)

        # ---- MAE / MSE ----
        preds_arr = np.asarray(preds, dtype=float)
        golds_arr = np.asarray(golds, dtype=float)
        mae = np.round(np.mean(np.abs(preds_arr - golds_arr)), 4)
        mse = np.round(np.mean((preds_arr - golds_arr) ** 2), 4)

        print(f"📊 Spearman correlation for {subtask}: {rho} (p={p})")
        print(f"📉 MAE for {subtask}: {mae}")
        print(f"📈 MSE for {subtask}: {mse}")

        results[subtask] = {"spearman": float(rho), "p_value": float(p), "mae": float(mae), "mse": float(mse)}

    # Final summary
    print("\n✅ Complete results:")
    if args.lora_path:
        print(args.lora_path)
    else:
        print(args.model_id)
    for subtask, metrics in results.items():
        print(f"{subtask}: Spearman={metrics['spearman']}, p-value={metrics['p_value']}, MAE={metrics['mae']}, MSE={metrics['mse']}")


if __name__ == "__main__":
    main()
