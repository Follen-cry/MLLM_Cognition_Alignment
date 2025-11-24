import transformers.models.gemma3.modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast
import torch
from typing import Optional, List, Union, Tuple
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss
)

def replace_gemma3_forward(use_liger=True):
    if use_liger:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward = gemma3_mixed_modality_forward_with_flce
    else:
        transformers.models.gemma3.modeling_gemma3.Gemma3ForConditionalGeneration.forward = gemma3_mixed_modality_forward

def gemma3_mixed_modality_forward_with_flce(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    is_training = token_type_ids is not None and labels is not None

    # Replace image id woth PAD if the image token if OOV, to avoid index-errors
    if input_ids is not None and self.config.image_token_index >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_index
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # Merge text and images

    if pixel_values is None:
        dummy_pixel_values = torch.zeros([1, 3, 896, 896]).to(device=inputs_embeds.device)
        dummy_image_features = self.get_image_features(dummy_pixel_values)
        inputs_embeds += dummy_image_features.mean() * 0

    else:
        image_features = self.get_image_features(pixel_values)

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

        if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            raise ValueError(
                f"Number of images does not match number of special image tokens in the input text. "
                f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                "tokens from image embeddings."
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # mask out pad-token-ids in labels for BC
    if labels is not None and self.pad_token_id in labels:
        logger.warning_once(
            "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
            "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
        )
        labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

    causal_mask = self._update_causal_mask(
        attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
    )
    outputs = self.language_model.model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **lm_kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    if self.training and (labels is not None):
        shift_hidden_states = hidden_states[..., :-1, :]
        shift_labels = labels[..., 1:]

        hidden_device = shift_hidden_states.device
        if attention_mask is not None:
            # we use the input attention mask to shift the hidden_states and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -shift_hidden_states.shape[1] :].to(hidden_device)
            shift_hidden_states = shift_hidden_states[shift_attention_mask.to(hidden_device) != 0].contiguous()
            shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
        else:
            shift_hidden_states = shift_hidden_states.contiguous()
            shift_labels = shift_labels.contiguous()

        # Flatten hidden state
        shift_hidden_states = shift_hidden_states.view(-1, self.config.text_config.hidden_size)
        shift_labels = shift_labels.view(-1).to(hidden_device)

        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(self.language_model.lm_head.weight, shift_hidden_states, shift_labels)
    else:
        logits = self.language_model.lm_head(hidden_states)
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Gemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )




def gemma3_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    is_training = token_type_ids is not None and labels is not None

    # Replace image id woth PAD if the image token if OOV, to avoid index-errors
    if input_ids is not None and self.config.image_token_index >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_index
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # Merge text and images

    if pixel_values is None:
        dummy_pixel_values = torch.zeros([1, 3, 896, 896]).to(device=inputs_embeds.device)
        dummy_image_features = self.get_image_features(dummy_pixel_values)
        inputs_embeds += dummy_image_features.mean() * 0

    else:
        image_features = self.get_image_features(pixel_values)

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

        if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            raise ValueError(
                f"Number of images does not match number of special image tokens in the input text. "
                f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                "tokens from image embeddings."
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # mask out pad-token-ids in labels for BC
    if labels is not None and self.pad_token_id in labels:
        logger.warning_once(
            "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
            "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
        )
        labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

    causal_mask = self._update_causal_mask(
        attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
    )
    outputs = self.language_model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **lm_kwargs,
    )

    logits = outputs[0]
    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        if attention_mask is not None:
            # we use the input attention mask to shift the logits and labels, because it is 2D.
            # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
            shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
            shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
        else:
            shift_logits = shift_logits.contiguous()
            shift_labels = shift_labels.contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()

        flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
        flat_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(flat_logits, flat_labels)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Gemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )