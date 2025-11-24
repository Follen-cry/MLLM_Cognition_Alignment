import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    is_peft_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)

from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

def _get_digit_ids(tokenizer):
    ids = []
    for d in range(10):
        # Prefer single-token digits; fallback to first id if split
        toks = tokenizer.encode(str(d), add_special_tokens=False)
        if len(toks) == 0:
            raise ValueError("Tokenizer returned empty encoding for digit")
        ids.append(toks[0])
    return torch.tensor(ids, dtype=torch.long)

def _triangular_psi(center_idx, num_digits=10, device="cpu"):
    # weights proportional to (max_dist - |k - center|)
    ks = torch.arange(num_digits, device=device)
    max_dist = num_digits - 1  # 9 for digits 0..9
    w = (max_dist - (ks - center_idx).abs()).clamp(min=0).float()
    w = w / w.sum()
    return w  # shape [10]


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class LLamaVTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(LLamaVTrainer, self).__init__(*args, **kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.projector_lr is not None:
                lr_mapper["multi_modal_projector"] = self.args.projector_lr
            if self.args.vision_lr is not None:
                lr_mapper["vision_model"] = self.args.vision_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(LLamaVTrainer, self)._save_checkpoint(model, trial)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        compute_loss with soft-labeling + positional weighting.

        - Soft-labeling enabled via self.args.soft_label_enable
        - Positional weighting: treats a digit at place p (units/tenths/hundredths) with weight
        proportional to numeric distance abs(k - center) * (10**p), so changes in more-significant
        places count more.
        """
        if not getattr(self.args, "soft_label_enable", False):
            return super().compute_loss(model, inputs, return_outputs)

        # --- tokenizer retrieval (prefer processing_class) ---
        tok = None
        if hasattr(self, "processing_class") and self.processing_class is not None:
            tok = getattr(self.processing_class, "tokenizer", None)
        if tok is None:
            tok = getattr(self, "tokenizer", None)
        if tok is None:
            raise ValueError(
                "Tokenizer not found. Pass the processor (with tokenizer) as `processing_class=` when creating the Trainer."
            )

        # --- cache digit ids (on model device) ---
        device = next(model.parameters()).device
        if not hasattr(self, "_digit_ids"):
            self._digit_ids = _get_digit_ids(tok).to(device)  # [10]

        labels = inputs.get("labels", None)
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs)

        # --- forward without labels ---
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, T, V]

        # shift as HF does
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]

        ignore_index = -100
        valid_mask = (labels != ignore_index)

        # ensure digit ids live on logits.device
        digit_ids = self._digit_ids.to(logits.device)  # [10]

        # vectorized detection of digit targets: labels shape [B,T], digit_ids [10]
        eq = labels.unsqueeze(-1) == digit_ids.view(1, 1, -1)  # [B, T, 10]
        is_digit = eq.any(dim=-1)  # [B, T]
        digit_idx = torch.where(
            is_digit,
            eq.float().argmax(dim=-1).to(labels.dtype),
            torch.full_like(labels, -1),
        )

        reg_mask = valid_mask & (~is_digit)
        num_mask = valid_mask & is_digit

        logp = F.log_softmax(logits, dim=-1)  # [B, T, V]

        # ---- Regular tokens: standard NLL ----
        dtype = logits.dtype
        reg_loss = torch.tensor(0.0, device=logits.device, dtype=dtype)
        N_r = reg_mask.sum()
        if N_r > 0:
            reg_loss_all = F.nll_loss(
                logp.transpose(1, 2),  # [B, V, T]
                labels,                 # [B, T]
                ignore_index=ignore_index,
                reduction="none",
            )  # [B, T]
            reg_loss = reg_loss_all[reg_mask].sum()

        # ---- Numeric tokens: positional-weighted soft CE on digit subspace ----
        num_loss = torch.tensor(0.0, device=logits.device, dtype=dtype)
        N_n = num_mask.sum()
        if N_n > 0:
            # indices of numeric positions, in order (N_n, 2) each row = (b_idx, t_idx)
            num_positions = num_mask.nonzero(as_tuple=False)  # tensor[[b,t], ...] on device
            # For indexing Python-level loops, convert to list of tuples (cheap because N_n small)
            num_indices = [(int(x[0].item()), int(x[1].item())) for x in num_positions]

            # log-probs only at digit token ids -> [B, T, 10]
            logp_digits_full = logp.index_select(dim=-1, index=digit_ids)  # [B, T, 10]
            # flatten selection -> [N_n, 10] in same order as num_indices
            logp_digits = torch.stack([logp_digits_full[b, t, :] for (b, t) in num_indices], dim=0)  # [N_n, 10]

            # settings
            et = float(getattr(self.args, "soft_label_eta", 0.08))
            dist_kind = getattr(self.args, "soft_label_dist", "triangular").lower()

            # precompute token id for decimal point (if present in tokenizer)
            try:
                dot_enc = tok.encode(".", add_special_tokens=False)
                dot_id = int(dot_enc[0]) if len(dot_enc) > 0 else None
            except Exception:
                dot_id = None

            # helper sets for quick checks
            digit_id_set = set(int(x.item()) for x in digit_ids)

            centers = digit_idx[num_mask]  # [N_n], dtype long

            # Build psi rows per numeric position (N_n x 10)
            psi_rows = []
            for i_pos, ((b_idx, t_idx), center) in enumerate(zip(num_indices, centers)):
                c = int(center.item())  # 0..9

                # find contiguous numeric span around (b_idx, t_idx) in labels[b_idx]
                row = labels[b_idx]  # [T]
                Tlen = row.shape[0]

                # expand left until non-digit and non-dot
                left = t_idx
                while left - 1 >= 0:
                    left_id = int(row[left - 1].item())
                    if (left_id in digit_id_set) or (dot_id is not None and left_id == dot_id):
                        left -= 1
                    else:
                        break

                # expand right until non-digit and non-dot
                right = t_idx
                while right + 1 < Tlen:
                    right_id = int(row[right + 1].item())
                    if (right_id in digit_id_set) or (dot_id is not None and right_id == dot_id):
                        right += 1
                    else:
                        break

                # locate decimal index in span (if any)
                decimal_idx = None
                if dot_id is not None:
                    for j in range(left, right + 1):
                        if int(row[j].item()) == dot_id:
                            decimal_idx = j
                            break
                if decimal_idx is None:
                    # treat as integer: decimal point sits just after the integer span
                    decimal_idx = right + 1

                # compute place p = decimal_idx - token_index - 1
                p = decimal_idx - t_idx - 1  # integer; negative -> fractional places

                # compute numeric distances for candidate digits k=0..9
                # numeric_dist_k = abs(k - c) * (10 ** p)
                # Use float32 for distances to avoid overflow in extreme p; cast to logits dtype later.
                k_vals = torch.arange(10, device=logits.device, dtype=torch.float32)
                center_val = float(c)
                # place_scale might be fractional or large; clamp p to reasonable range to avoid inf/underflow
                p_clamped = max(min(p, 9), -9)  # safe clamp
                place_scale = (10.0 ** float(p_clamped))
                numeric_dists = torch.abs(k_vals - center_val) * place_scale  # [10], float32

                # Triangular-like weighting over numeric distances
                max_dist = numeric_dists.max().item()
                if max_dist == 0.0:
                    w = torch.ones(10, device=logits.device, dtype=logits.dtype) / 10.0
                else:
                    w = (max_dist - numeric_dists).clamp(min=0.0)  # higher for closer numeric distance
                    # convert to logits dtype for consistency and normalize
                    w = w.to(dtype=logits.dtype, device=logits.device)
                    if w.sum() == 0:
                        w = torch.ones_like(w) / 10.0
                    else:
                        w = w / w.sum()

                psi_rows.append(w)

            psi = torch.stack(psi_rows, dim=0)  # [N_n, 10], dtype logits.dtype, device logits.device

            # build delta (one-hot) for centers
            delta = F.one_hot(centers.clamp(min=0), num_classes=10).float().to(dtype=logits.dtype, device=logits.device)  # [N_n,10]

            # choose mixing
            if dist_kind == "uniform":
                psi = torch.ones_like(psi) / 10.0  # override if uniform requested
                
            q_sl = (1.0 - et) * delta + et * psi  # [N_n,10]

            # compute soft CE over digit subspace: -sum(q * logp_digits)
            # logp_digits is already [N_n, 10]
            num_loss_vec = -(q_sl * logp_digits).sum(dim=-1)  # [N_n]
            num_loss = num_loss_vec.sum()

        lam = float(getattr(self.args, "soft_label_lambda", 2.0))
        denom = (N_r + N_n).clamp(min=1)
        total_loss = (reg_loss + lam * num_loss) / denom

        return (total_loss, outputs) if return_outputs else total_loss

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'vision_model' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
            
    #         elif 'img_projection' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    #     return super().training_step(model, inputs)