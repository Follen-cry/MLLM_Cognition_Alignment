# compare_text_encoders.py
import argparse
import json
import os
import re
from typing import Dict, List, Union

import torch
from diffusers import QwenImagePipeline
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from PIL import Image, ImageDraw, ImageFont
from peft import PeftModel  

MODEL_DIR = "/scratch/local/ssd/junlin/models/qwen-image"
#FINETUNED = "/scratch/local/ssd/junlin/models/soft_label/Qwen2.5-vl-7b-cognition_sft_lora_3e-5_0.15_2.0"
FINETUNED = "/scratch/local/ssd/junlin/models/Qwen2.5-vl-7b-cognition_sft"
DEFAULT_PROMPT = "a neon-lit alley in rain, cinematic reflections, detailed"


# ----------------------------- Args -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # Single prompt (fallback)
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)

    # NEW: prompt JSON file (list[str] or dict[str,str])
    ap.add_argument("--prompts_json", type=str, default=None,
                    help="Path to JSON containing prompts (list of strings OR dict name->prompt).")

    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--num_images", type=int, default=4,
                    help="Number of images to generate per prompt per encoder.")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)

    # Generation params exposed for convenience
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--true_cfg_scale", type=float, default=2.0)
    ap.add_argument("--negative_prompt", type=str, default="")

    # NEW: optionally save side-by-side comparisons
    ap.add_argument("--save_combined", action="store_true")
    ap.add_argument("--combined_font_size", type=int, default=12)
    ap.add_argument("--max_prompt_chars", type=int, default=200,
                    help="Truncate displayed prompt in combined image to this length.")

    # Device hints
    ap.add_argument("--device_map", type=str, default="balanced")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    # NEW: base model path (only needed if FINETUNED is a LoRA folder)
    ap.add_argument("--base_vl", type=str, default=None,
                    help="Path to the full base Qwen2.5-VL model (required if FINETUNED is a LoRA).")
    return ap.parse_args()


# -------------------------- Utilities --------------------------
def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_\-]+", "", text)
    return (text[:max_len]).strip("_") or "prompt"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_prompts(args) -> List[Dict[str, str]]:
    items = []
    if args.prompts_json:
        with open(args.prompts_json, "r") as f:
            data: Union[List[str], Dict[str, str]] = json.load(f)

        if isinstance(data, list):
            for i, p in enumerate(data):
                items.append({"name": f"p{i:03d}", "prompt": str(p)})
        elif isinstance(data, dict):
            for k, p in data.items():
                items.append({"name": slugify(k), "prompt": str(p)})
        else:
            raise ValueError("prompts_json must be a list[str] or dict[str,str].")
    else:
        items.append({"name": slugify(args.prompt), "prompt": args.prompt})
    return items

def pil_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def add_labels(img_left, img_right, prompt, label_left="Default", label_right="Finetuned",
               font_size=12, max_prompt_chars=200):
    font = pil_font(font_size)
    trunc_prompt = prompt if len(prompt) <= max_prompt_chars else prompt[:max_prompt_chars] + "…"

    combined_width = img_left.width + img_right.width
    text_height = font_size * 4
    combined_height = img_left.height + text_height
    combined = Image.new("RGB", (combined_width, combined_height), "white")

    combined.paste(img_left, (0, 0))
    combined.paste(img_right, (img_left.width, 0))

    draw = ImageDraw.Draw(combined)
    draw.text((10, img_left.height + 5), f"{label_left}", font=font, fill="black")
    draw.text((img_left.width + 10, img_right.height + 5), f"{label_right}", font=font, fill="black")
    draw.text((10, img_left.height + font_size + 10), f"Prompt: {trunc_prompt}", font=font, fill="black")
    return combined


# ------------------------ Image Generation ------------------------
@torch.inference_mode()
def generate_image(pipeline, prompt, seed, width, height, steps, true_cfg_scale, negative_prompt):
    gen = torch.Generator(device="cuda").manual_seed(int(seed))
    return pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        true_cfg_scale=float(true_cfg_scale),
        width=int(width),
        height=int(height),
        generator=gen,
    ).images[0]

def cast_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()
    dtype = cast_dtype(args.dtype)

    # I/O layout
    base_out = os.path.abspath(args.out_dir)
    out_default_dir = os.path.join(base_out, "default")
    out_finetuned_dir = os.path.join(base_out, "finetuned")
    out_combined_dir = os.path.join(base_out, "combined")
    ensure_dir(out_default_dir)
    ensure_dir(out_finetuned_dir)
    if args.save_combined:
        ensure_dir(out_combined_dir)

    prompts = load_prompts(args)
    print(f"[Init] {len(prompts)} prompts loaded.")

    # 1) Load default Qwen-Image pipeline
    print("[1/4] Loading default Qwen-Image pipeline...")
    pipe = QwenImagePipeline.from_pretrained(
        MODEL_DIR,
        dtype=dtype,                 # (kept as-is for minimal changes)
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    )
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

        # 2) Generate all DEFAULT images
    print("[2/4] Generating DEFAULT images...")
    for p_idx, item in enumerate(prompts):
        p_name, prompt_text = item["name"], item["prompt"]
        for i in range(args.num_images):
            seed = args.seed + i  # reproducible & comparable between encoders
            img = generate_image(
                pipe, prompt_text, seed,
                width=args.width, height=args.height,
                steps=args.steps, true_cfg_scale=args.true_cfg_scale,
                negative_prompt=args.negative_prompt
            )
            fname = f"{p_idx:03d}_{p_name}_seed{seed}_default.png"
            img.save(os.path.join(out_default_dir, fname))

    # 3) Swap to FINETUNED text encoder (tokenizer + text_backbone)
    print("[3/4] Swapping to FINETUNED text encoder...")

    # Try tokenizer from FINETUNED first; fall back to base if this is a LoRA repo
    try:
        tok_ft = AutoTokenizer.from_pretrained(
            FINETUNED, trust_remote_code=True, use_fast=False, local_files_only=True
        )
    except Exception:
        if not args.base_vl:
            raise
        tok_ft = AutoTokenizer.from_pretrained(
            args.base_vl, trust_remote_code=True, use_fast=False, local_files_only=True
        )
    if tok_ft.pad_token is None:
        tok_ft.pad_token = tok_ft.eos_token

    # Try to load FINETUNED as a full model; if that fails, assume it's a LoRA and apply to base
    try:
        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            FINETUNED,
            dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
    except OSError as e:
        if not args.base_vl:
            raise OSError(
                str(e)
                + "\nDetected a LoRA adapter. Please pass --base_vl pointing to the full base Qwen2.5-VL model."
            )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_vl,
            dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        vl_model = PeftModel.from_pretrained(base, FINETUNED, is_trainable=False).eval()

    # find the text backbone within the model
    text_backbone = (
        getattr(vl_model, "model", None)
        or getattr(vl_model, "language_model", None)
        or getattr(vl_model, "text_model", None)
        or vl_model
    )
    for p in text_backbone.parameters():
        p.requires_grad = False

    # attach tokenizer & encoder to pipeline
    pipe.tokenizer = tok_ft
    pipe.text_encoder = text_backbone

    # align device/dtype
    exec_device = getattr(pipe, "_execution_device", torch.device("cuda:0"))
    pipe_dtype = getattr(pipe.transformer, "dtype", dtype)
    pipe.text_encoder.to(exec_device, dtype=pipe_dtype)
    for p in pipe.text_encoder.parameters():
        if p.dtype != pipe_dtype:
            p.data = p.data.to(pipe_dtype)

    # 4) Generate all FINETUNED images (+ optional combined)
    print("[4/4] Generating FINETUNED images...")
    for p_idx, item in enumerate(prompts):
        p_name, prompt_text = item["name"], item["prompt"]
        for i in range(args.num_images):
            seed = args.seed + i
            img_ft = generate_image(
                pipe, prompt_text, seed,
                width=args.width, height=args.height,
                steps=args.steps, true_cfg_scale=args.true_cfg_scale,
                negative_prompt=args.negative_prompt
            )
            ft_name = f"{p_idx:03d}_{p_name}_seed{seed}_finetuned.png"
            ft_path = os.path.join(out_finetuned_dir, ft_name)
            img_ft.save(ft_path)

            if args.save_combined:
                def_name = f"{p_idx:03d}_{p_name}_seed{seed}_default.png"
                def_path = os.path.join(out_default_dir, def_name)
                if os.path.exists(def_path):
                    img_def = Image.open(def_path).convert("RGB")
                    combined = add_labels(
                        img_def, img_ft, prompt_text,
                        label_left="Default", label_right="Finetuned",
                        font_size=args.combined_font_size,
                        max_prompt_chars=args.max_prompt_chars
                    )
                    comb_name = f"{p_idx:03d}_{p_name}_seed{seed}_combined.png"
                    combined.save(os.path.join(out_combined_dir, comb_name))

    print(f"\nDone.\nDefault images : {out_default_dir}\nFinetuned images: {out_finetuned_dir}")
    if args.save_combined:
        print(f"Combined images: {out_combined_dir}")


if __name__ == "__main__":
    main()
