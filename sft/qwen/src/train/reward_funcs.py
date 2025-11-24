import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# ---------- Label rules (for reference / optional future use) ----------
LABEL_RULES = {
    "aesthetics": [
        (0.0,  3.5,  "very low"),
        (3.5,  5.0,  "low"),
        (5.0,  6.5,  "medium"),
        (6.5,  8.0,  "high"),
        (8.0, 10.1, "very high"),
    ],
    "funniness": [
        (0.0,  3.5,  "very low"),
        (3.5,  5.0,  "low"),
        (5.0,  6.5,  "medium"),
        (6.5,  8.0,  "high"),
        (8.0, 10.1, "very high"),
    ],
    "memorability": [
        (0.00, 0.35, "very low"),
        (0.35, 0.50, "low"),
        (0.50, 0.65, "medium"),
        (0.65, 0.80, "high"),
        (0.80, 1.00, "very high"),
    ],
    "emotional_valence": [
        (-3.0, -0.1, "negative"),
        ( 0.0,  0.1, "neutral"),
        ( 1.0,  3.1, "positive"),
    ],
}

# ---------- Regex helpers ----------
_ANSWER_RE = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_NUM_RE    = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _extract_number(text: str) -> Optional[float]:
    """Extract the last numeric value from text. Works for '<answer>low 4.348</answer>' or 'low 4.348'."""
    if not isinstance(text, str):
        return None
    m = _ANSWER_RE.search(text)
    candidate = m.group(1) if m else text
    nums = _NUM_RE.findall(candidate)
    return float(nums[-1]) if nums else None

def _to_text_list_from_completions(completions: List[Any]) -> List[str]:
    """
    completions[i] can be:
      - [ {'content': str, ...}, ... ]  (we take [0]['content'])
      - {'content': str, ...}
      - plain str
    Normalize to list[str].
    """
    out = []
    for c in completions:
        if isinstance(c, list) and c and isinstance(c[0], dict):
            out.append(c[0].get("content", ""))
        elif isinstance(c, dict):
            out.append(c.get("content", ""))
        else:
            out.append(str(c))
    return out

def _to_text_list_from_assistant(assistant: Any, n: int) -> List[str]:
    """assistant matches completions length or is broadcastable."""
    if assistant is None:
        return ["" for _ in range(n)]
    if isinstance(assistant, list):
        vals = []
        for a in assistant:
            if isinstance(a, dict):
                vals.append(a.get("content", ""))
            else:
                vals.append(str(a))
        return vals
    val = assistant.get("content", "") if isinstance(assistant, dict) else str(assistant)
    return [val for _ in range(n)]

# ---------- Task inference & scaling ----------
def _task_from_sample(sample: Dict[str, Any], prompt_text: str, gt: Optional[float]) -> str:

    # derive from image path
    img = sample.get("image") or sample.get("images") or ""
    if isinstance(img, list) and img:
        img = img[0]
    if isinstance(img, str) and "/" in img:
        prefix = img.split("/", 1)[0].strip().lower()
        if prefix in ("aesthetics", "funniness", "memorability", "emotional_valence"):
            return prefix
 

def _scale_for_task(task: str) -> float:
    """
    Normalization scale = (max - min) of the task domain.
      - aesthetics/funniness: 10
      - memorability: 1
      - emotional_valence: 6 (from -3..3)
    """
    t = task.lower()
    if t in ("aesthetics", "funniness"):
        return 10.0
    if t == "memorability":
        return 1.0
    if t == "emotional_valence":
        return 6.0
    return 10.0  # safe default

def score_closeness_reward(
    *,
    prompts: List[str],
    completions: List[Any],
    assistant: Optional[Any] = None,
    miss_penalty: float = -0.5,
    # You can override behavior via reward_kwargs in your launcher:
    #   --reward_kwargs '{"default_task":"aesthetics","debug":true}'
    default_task: str = "aesthetics",
    debug: bool = False,
    **kwargs,
) -> List[float]:
    """
    Reward = 1 - |pred - gt| / scale(task), clipped to [-1, 1].
    - scale(task): 10 (aesthetics/funniness), 1 (memorability), 6 (emotional_valence)
    - Attempts to read task/image from kwargs['samples'][i] if trainer provides them.
      Otherwise, infers task from prompt keywords or GT numeric range.
    - Answers like '<answer>low 4.348</answer>' are parsed; label text is ignored for scoring.
    """
    now = datetime.now().strftime("%d-%H-%M-%S-%f")
    pred_texts = _to_text_list_from_completions(completions)
    gt_texts   = _to_text_list_from_assistant(assistant, n=len(pred_texts))
    samples    = kwargs.get("samples") or kwargs.get("metas") or kwargs.get("batch")  # repo-dependent

    rewards: List[float] = []

    for i, (pred_txt, gt_txt) in enumerate(zip(pred_texts, gt_texts)):
        pred = _extract_number(pred_txt)
        gt   = _extract_number(gt_txt)

        # pull per-sample metadata if available
        sample_i: Dict[str, Any] = {}
        if isinstance(samples, list) and i < len(samples) and isinstance(samples[i], dict):
            sample_i = samples[i]

        # robust task inference
        task = _task_from_sample(sample_i, prompt_text=(prompts[i] if i < len(prompts) else ""), gt=gt) \
               or default_task
        scale = _scale_for_task(task)

        # compute reward
        if pred is None or gt is None:
            r = miss_penalty
        else:
            r = 1.0 - abs(pred - gt) / max(scale, 1e-6)
            r = max(min(r, 1.0), -1.0)

        rewards.append(r)

        if debug or os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "./reward_debug.log")
            try:
                with open(log_path, "a") as f:
                    f.write(f"[{now}] task={task} scale={scale} reward={r:.4f}\n")
                    if sample_i:
                        img = sample_i.get("image")
                        f.write(f"  image={img}\n")
                    f.write(f"  pred_txt={pred_txt}\n  gt_txt={gt_txt}\n")
            except Exception:
                pass

    return rewards