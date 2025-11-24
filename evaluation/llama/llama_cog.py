import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq

from scipy.stats import spearmanr
import torch
import numpy as np
import json
import re
import os
from tqdm import tqdm
import random
import argparse

# NEW: import peft for LoRA application
from peft import PeftModel

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
    parser.add_argument("--base-model", default=None,
                        help="Path to full base model (required if --model-id is a LoRA adapter).")
    parser.add_argument("--base-dir", default="/homes/55/junlin/mllm_benchmark_project/Qwen2-VL-Finetune/data/test_msg_file")
    parser.add_argument("--img-base", default="/homes/55/junlin/mllm_benchmark_project/Qwen2-VL-Finetune/data/cognition/cognition_images")
    parser.add_argument("--out-dir",  default="/scratch/local/ssd/junlin/output/cognition/sft")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Number of examples to sample per subtask (random without replacement). Use all if omitted.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print("set seed:", args.seed)
    set_seed(args.seed)

    model_id = args.model_id
    # Load processor (try model_id first, fall back to base-model if provided)
    print("Loading model...")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception:
        if args.base_model is None:
            raise
        processor = AutoProcessor.from_pretrained(args.base_model)

    # Load model: try to load model_id (full model). If it fails (likely a LoRA-only folder),
    # require --base-model and apply the LoRA adapter using peft.
    if args.base_model:
        print("not full model, try applying lora...")
        # load base and apply LoRA adapter
        base = MllamaForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Applying lora adaptor")
        model = PeftModel.from_pretrained(base, model_id, is_trainable=False)

    else:
        print("loading full model....")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
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
        output_file = os.path.join(args.out_dir, f"sft_results_{subtask}{suffix}.txt")

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
            image = Image.open(image_path).convert("RGB")
            raw_text = msg[0]["content"][1]["text"]
            text = re.sub(r"\s*<image>\s*", " ", raw_text).strip()

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]}
            ]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    do_sample=False,    # or True with temperature/top_p if you like
                    temperature=None         
                )
            prompt_len = inputs["input_ids"].shape[1]              # <-- new
            gen_only = output_ids[:, prompt_len:]                  # <-- new
            answer = processor.batch_decode(
                gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0].strip() 

            # Save message and answer
            save_message_and_answer(text, answer, output_file)

            # Parse numbers from prediction and reference
            #pred_matches = re.findall(r"\d+\.?\d*", answer)
            pred_matches = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', answer)
            pred_score = float(pred_matches[0]) if pred_matches else np.nan
            if np.abs(pred_score) > 10:
                pred_score = np.nan  # invalid score

            true_matches = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', true_score)
            true_score_f = float(true_matches[0]) if true_matches else np.nan

            model_scores.append(pred_score)
            valid_ref_scores.append(true_score_f)

         # Metrics
        pairs = [(m, r) for m, r in zip(model_scores, valid_ref_scores) if (not np.isnan(m) and not np.isnan(r))]
        if not pairs:
            print("❗ No valid scores parsed.")
            continue

        preds, golds = zip(*pairs)
        preds = np.array(preds, dtype=np.float64)
        golds = np.array(golds, dtype=np.float64)

        rho, p = spearmanr(preds, golds)
        rho = np.round(rho, 4)
        p = np.round(p, 4)

        mae = float(np.mean(np.abs(preds - golds)))
        mse = float(np.mean((preds - golds) ** 2))
        mae = np.round(mae, 4)
        mse = np.round(mse, 4)


        print(f"📊 Spearman correlation for {subtask}: {rho} (p={p})")
        print(f"📉 MAE for {subtask}: {mae}")
        print(f"📈 MSE for {subtask}: {mse}")
        #print(f"\n model scores: {preds}")
        #print(f"\n reference scores: {golds}")

        results[subtask] = {
            "spearman": rho,
            "p_value": p,
            "mae": mae,
            "mse": mse
        }

    # Final summary
    print("\n✅ Complete results: with seed", args.seed)
    print(args.model_id)
    for subtask, metrics in results.items():
        print(f"{subtask}: Spearman={metrics['spearman']} (p={metrics['p_value']}), MAE={metrics['mae']}, MSE={metrics['mse']})")
if __name__ == "__main__":
    main()
