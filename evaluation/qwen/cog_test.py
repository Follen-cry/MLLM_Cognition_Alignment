from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, logging
from qwen_vl_utils import process_vision_info
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--base-dir", default="/homes/55/junlin/mllm_benchmark_project/Qwen2-VL-Finetune/data/test_msg_file")
    parser.add_argument("--img-base", default="/homes/55/junlin/mllm_benchmark_project/Qwen2-VL-Finetune/data/cognition/cognition_images")
    parser.add_argument("--out-dir",  default="/scratch/local/ssd/junlin/output/cognition/sft")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Number of examples to sample per subtask (random without replacement). Use all if omitted.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    logging.set_verbosity_error()

    # Load model + processor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    # List subtasks
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
            # Save sampled indices for reproducibility
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
            image_relative_path = msg[0]["content"][0]["image"]
            image_path = os.path.join(args.img_base, image_relative_path)
            msg[0]["content"][0]["image"] = image_path

            # ensure image exists/opens
            _ = Image.open(image_path).convert("RGB")
            messages = msg

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, 
                                               max_new_tokens=100,
                                               do_sample=False,
                                               temperature=None)
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

        # Metrics
        pairs = [(m, r) for m, r in zip(model_scores, valid_ref_scores) if not (np.isnan(m) or np.isnan(r))]
        if not pairs:
            print("❗ No valid scores parsed.")
            continue
        preds, golds = zip(*pairs)
        preds = np.array(preds, dtype=float)
        golds = np.array(golds, dtype=float)

        rho, p = spearmanr(preds, golds)
        mae = np.mean(np.abs(preds - golds))
        mse = np.mean((preds - golds) ** 2)

        rho = np.round(rho, 4)
        p   = np.round(p,   4)
        mae = np.round(mae, 4)
        mse = np.round(mse, 4)

        print(f"📊 {subtask}: Spearman={rho} (p={p}) | MAE={mae} | MSE={mse}")
        results[subtask] = {"spearman": rho, "p_value": p, "mae": mae, "mse": mse}

    # Final summary
    print("\n✅ Complete results:")
    print(args.model_id)
    for subtask, metrics in results.items():
        print(f"{subtask}: Spearman={metrics['spearman']}, p-value={metrics['p_value']}, MAE={metrics['mae']}, MSE={metrics['mse']}")

if __name__ == "__main__":
    main()
