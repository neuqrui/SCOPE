# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import json
import logging
import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Alert when a micro-batch is close to the step-51 failure mode:
# student logprob of thousands negative, gap ~ -log π_S, low_var_kl exp near overflow.
_DIAG_STUDENT_MIN = -20.0
_DIAG_GAP_MAX = 20.0
_DIAG_KL_RAW_MAX = 40.0


def _masked_minmax(values: torch.Tensor, mask: torch.Tensor):
    selected = values[mask.bool()]
    if selected.numel() == 0:
        return None
    return float(selected.min().item()), float(selected.max().item()), float(selected.mean().item())


def _extreme_token_mask(
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_prob: torch.Tensor,
    teacher_log_probs: Optional[torch.Tensor] = None,
    ref_log_prob: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Valid response positions whose current logprob would blow up the update.

    Healthy sampled actions sit near logprob -1. A student logprob below -20
    (probability under 2e-9), a teacher gap above 20, or a raw KL above 40 is
    the step-51 failure mode. Those positions are dropped from the loss; the
    rest of the sequence is unchanged.
    """
    valid = response_mask.bool()
    bad = valid & ~torch.isfinite(log_prob)
    bad = bad | (valid & (log_prob < _DIAG_STUDENT_MIN))
    bad = bad | (valid & ~torch.isfinite(old_log_prob))
    if teacher_log_probs is not None:
        gap = teacher_log_probs - log_prob
        bad = bad | (valid & ~torch.isfinite(teacher_log_probs))
        bad = bad | (valid & torch.isfinite(gap) & (gap > _DIAG_GAP_MAX))
    if ref_log_prob is not None:
        kl_raw = ref_log_prob - log_prob
        bad = bad | (valid & ~torch.isfinite(ref_log_prob))
        bad = bad | (valid & torch.isfinite(kl_raw) & (kl_raw > _DIAG_KL_RAW_MAX))
    return bad


def _record_actor_diag(record: dict) -> None:
    """Append one JSON line. Means in the trainer log hide the token that blows up."""
    diag_dir = os.environ.get("ACTOR_DIAG_DIR", "/data/qirui/SDAR/logs/actor_diag")
    try:
        os.makedirs(diag_dir, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = os.path.join(diag_dir, f"rank{rank}.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[actor-diag] failed to write: {exc}", flush=True)


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        sdl_coef = data.meta_info.get("sdl_coef", self.config.get("sdl_loss_coef", 0.1))
        accumulate_only = bool(data.meta_info.get("accumulate_only", False))
        global_steps = data.meta_info.get("global_steps", -1)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if (
            self.config.get("use_sdl_loss", False)
            or self.config.get("use_sdar_loss", False)
            or self.config.get("use_opsd_kl_loss", False)
        ):
            select_keys.append("teacher_log_probs")
        if "opsd_token_mask" in data.batch.keys():
            select_keys.append("opsd_token_mask")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        dropped_token_total = 0
        dropped_sample_total = 0
        valid_token_total = 0
        valid_sample_total = 0
        num_optimizer_steps = max(1, self.config.ppo_epochs * len(dataloader))
        if accumulate_only:
            self.actor_optimizer.zero_grad()
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if not accumulate_only:
                    self.actor_optimizer.zero_grad()

                step_diag = {
                    "global_steps": global_steps,
                    "ppo_epoch": epoch,
                    "mini_batch": batch_idx,
                    "student_logprob_min": None,
                    "gap_max": None,
                    "kl_raw_max": None,
                    "nonfinite": False,
                    "alert": False,
                    "dropped_tokens": 0,
                    "dropped_samples": 0,
                    "valid_tokens": 0,
                    "valid_samples": 0,
                }

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    teacher_for_filter = data["teacher_log_probs"] if "teacher_log_probs" in data else None
                    ref_for_filter = data["ref_log_prob"] if self.config.use_kl_loss and "ref_log_prob" in data else None
                    bad_tokens = _extreme_token_mask(
                        log_prob=log_prob,
                        response_mask=response_mask,
                        old_log_prob=old_log_prob,
                        teacher_log_probs=teacher_for_filter,
                        ref_log_prob=ref_for_filter,
                    )
                    if entropy is not None:
                        bad_tokens = bad_tokens | (response_mask.bool() & ~torch.isfinite(entropy))
                        entropy = torch.where(bad_tokens, torch.zeros_like(entropy), entropy)
                    old_fill = torch.where(torch.isfinite(old_log_prob), old_log_prob.detach(), torch.zeros_like(log_prob))
                    safe_log_prob = torch.where(bad_tokens, old_fill, log_prob)
                    advantages_for_loss = torch.where(bad_tokens, torch.zeros_like(advantages), advantages)
                    train_mask = response_mask * (~bad_tokens).to(dtype=response_mask.dtype)
                    teacher_for_loss = None
                    if teacher_for_filter is not None:
                        teacher_for_loss = torch.where(
                            torch.isfinite(teacher_for_filter), teacher_for_filter, torch.zeros_like(log_prob)
                        )
                    ref_for_loss = None
                    if ref_for_filter is not None:
                        ref_for_loss = torch.where(torch.isfinite(ref_for_filter), ref_for_filter, torch.zeros_like(log_prob))
                    dropped_tokens = int(bad_tokens.sum().item())
                    dropped_samples = int(bad_tokens.any(dim=-1).sum().item())
                    valid_tokens = int(response_mask.bool().sum().item())
                    valid_samples = int(response_mask.bool().any(dim=-1).sum().item())
                    step_diag["dropped_tokens"] += dropped_tokens
                    step_diag["dropped_samples"] += dropped_samples
                    step_diag["valid_tokens"] += valid_tokens
                    step_diag["valid_samples"] += valid_samples
                    dropped_token_total += dropped_tokens
                    dropped_sample_total += dropped_samples
                    valid_token_total += valid_tokens
                    valid_sample_total += valid_samples

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    old_for_loss = torch.where(torch.isfinite(old_log_prob), old_log_prob, torch.zeros_like(old_log_prob))
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_for_loss,
                        log_prob=safe_log_prob,
                        advantages=advantages_for_loss,
                        response_mask=train_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    pg_loss_coef = self.config.get("pg_loss_coef", 1.0)
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=train_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss * pg_loss_coef - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss * pg_loss_coef

                    if self.config.use_kl_loss:
                        # compute kl loss
                        kld = kl_penalty(logprob=safe_log_prob, ref_logprob=ref_for_loss, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=train_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.get("use_sdl_loss", False):
                        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss
                        sdl_loss = compute_sdl_loss(
                            student_log_probs=safe_log_prob,
                            teacher_log_probs=teacher_for_loss,
                            old_log_probs=old_for_loss,
                            response_mask=train_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = policy_loss + sdl_loss * sdl_coef
                        metrics["actor/sdl_loss"] = sdl_loss.detach().item()
                        metrics["actor/sdl_coef"] = sdl_coef

                    if self.config.get("use_sdar_loss", False):
                        from verl.trainer.ppo.sdar_utils import compute_sdar_loss
                        sdar_loss, sdar_metrics = compute_sdar_loss(
                            student_log_probs=safe_log_prob,
                            teacher_log_probs=teacher_for_loss,
                            response_mask=train_mask,
                            gate_beta=self.config.get("sdar_gate_beta", 5.0),
                            loss_agg_mode=loss_agg_mode,
                        )
                        sdar_coef = self.config.get("sdar_loss_coef", 0.1)
                        policy_loss = policy_loss + sdar_loss * sdar_coef
                        metrics.update(sdar_metrics)
                        metrics["sdar/coef"] = sdar_coef

                    if "teacher_log_probs" in data:
                        from verl.trainer.ppo.opsd_kl_utils import compute_opsd_sampled_kl_loss

                        opsd_kl_objective = self.config.get("opsd_kl_objective", "sampled_kl")
                        opsd_kl_gate = self.config.get("opsd_kl_gate", "none")
                        opsd_kl_gate_beta = self.config.get("opsd_kl_gate_beta", 5.0)
                        opsd_kl_advantage_clip = self.config.get("opsd_kl_advantage_clip", 10.0)
                        if self.config.get("use_opsd_kl_loss", False):
                            opsd_response_mask = train_mask
                            if "opsd_token_mask" in data:
                                opsd_response_mask = train_mask * data["opsd_token_mask"][:, :response_length]
                            if opsd_response_mask.sum() > 0:
                                opsd_kl_loss, opsd_kl_metrics = compute_opsd_sampled_kl_loss(
                                    student_log_probs=safe_log_prob,
                                    teacher_log_probs=teacher_for_loss,
                                    response_mask=opsd_response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    objective=opsd_kl_objective,
                                    gate=opsd_kl_gate,
                                    gate_beta=opsd_kl_gate_beta,
                                    advantage_clip=opsd_kl_advantage_clip,
                                )
                            else:
                                opsd_kl_loss = safe_log_prob.new_zeros(())
                                opsd_kl_metrics = {
                                    "opsd_kl/loss": 0.0,
                                    "opsd_kl/teacher_gap_mean": 0.0,
                                    "opsd_kl/bounded_advantage_mean": 0.0,
                                    "opsd_kl/clip_ratio": 0.0,
                                }
                            opsd_kl_coef = self.config.get("opsd_kl_loss_coef", 0.1)
                            policy_loss = policy_loss + opsd_kl_loss * opsd_kl_coef
                            metrics.update(opsd_kl_metrics)
                            metrics["opsd_kl/coef"] = opsd_kl_coef
                            metrics["opsd_kl/gate_beta"] = float(opsd_kl_gate_beta)
                            metrics["opsd_kl/advantage_clip"] = float(opsd_kl_advantage_clip)
                        elif self.config.get("log_opsd_kl_reference", True):
                            with torch.no_grad():
                                _, opsd_kl_ref_metrics = compute_opsd_sampled_kl_loss(
                                    student_log_probs=safe_log_prob.detach(),
                                    teacher_log_probs=teacher_for_loss,
                                    response_mask=train_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    objective=opsd_kl_objective,
                                    gate=opsd_kl_gate,
                                    gate_beta=opsd_kl_gate_beta,
                                    advantage_clip=opsd_kl_advantage_clip,
                                )
                            metrics.update({f"opsd_kl_ref/{k.split('/', 1)[-1]}": v for k, v in opsd_kl_ref_metrics.items()})

                    finite_log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
                    policy_loss = policy_loss + finite_log_prob.sum() * 0
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    if accumulate_only:
                        loss = loss / num_optimizer_steps

                    with torch.no_grad():
                        student_stats = _masked_minmax(log_prob, response_mask)
                        gap_stats = None
                        kl_raw_max = None
                        if student_stats is not None:
                            step_diag["student_logprob_min"] = (
                                student_stats[0]
                                if step_diag["student_logprob_min"] is None
                                else min(step_diag["student_logprob_min"], student_stats[0])
                            )
                        if "teacher_log_probs" in data:
                            gap = data["teacher_log_probs"] - log_prob
                            gap_stats = _masked_minmax(gap, response_mask)
                            if gap_stats is not None:
                                step_diag["gap_max"] = (
                                    gap_stats[1]
                                    if step_diag["gap_max"] is None
                                    else max(step_diag["gap_max"], gap_stats[1])
                                )
                        if self.config.use_kl_loss and "ref_log_prob" in data:
                            kl_raw = data["ref_log_prob"] - log_prob
                            kl_stats = _masked_minmax(kl_raw, response_mask)
                            if kl_stats is not None:
                                kl_raw_max = kl_stats[1]
                                step_diag["kl_raw_max"] = (
                                    kl_raw_max
                                    if step_diag["kl_raw_max"] is None
                                    else max(step_diag["kl_raw_max"], kl_raw_max)
                                )
                        has_tokens = bool(response_mask.any().item())
                        nonfinite = not bool(torch.isfinite(log_prob[response_mask.bool()]).all()) if has_tokens else False
                        nonfinite = nonfinite or not bool(torch.isfinite(policy_loss).all())
                        step_diag["nonfinite"] = step_diag["nonfinite"] or nonfinite
                        alert = nonfinite
                        alert = alert or (student_stats is not None and student_stats[0] < _DIAG_STUDENT_MIN)
                        alert = alert or (gap_stats is not None and gap_stats[1] > _DIAG_GAP_MAX)
                        alert = alert or (kl_raw_max is not None and kl_raw_max > _DIAG_KL_RAW_MAX)
                        if alert and not step_diag["alert"]:
                            step_diag["alert"] = True
                            rank_lp = torch.nan_to_num(log_prob, nan=-1e8, posinf=0.0, neginf=-1e8)
                            # masked_fill can be non-contiguous; reshape, not view.
                            worst = (-rank_lp).masked_fill(~response_mask.bool(), -1e9).reshape(-1)
                            k = min(8, int(worst.numel()))
                            values, indices = torch.topk(worst, k)
                            flat_student = log_prob.reshape(-1)
                            flat_response = responses.reshape(-1)
                            worst_tokens = []
                            for value, index in zip(values.tolist(), indices.tolist()):
                                if value < -1e8:
                                    continue
                                item = {
                                    "index": int(index),
                                    "token_id": int(flat_response[index].item()),
                                    "student_logprob": float(flat_student[index].item()),
                                }
                                if "teacher_log_probs" in data:
                                    item["teacher_logprob"] = float(data["teacher_log_probs"].reshape(-1)[index].item())
                                if "ref_log_prob" in data:
                                    item["ref_logprob"] = float(data["ref_log_prob"].reshape(-1)[index].item())
                                if "old_log_probs" in data:
                                    item["old_logprob"] = float(data["old_log_probs"].reshape(-1)[index].item())
                                worst_tokens.append(item)
                            step_diag["worst_tokens"] = worst_tokens
                            print(
                                f"[actor-diag] step={global_steps} mini={batch_idx} "
                                f"student_min={None if student_stats is None else round(student_stats[0], 3)} "
                                f"gap_max={None if gap_stats is None else round(gap_stats[1], 3)} "
                                f"kl_raw_max={None if kl_raw_max is None else round(kl_raw_max, 3)} "
                                f"dropped_tokens={dropped_tokens}/{valid_tokens} "
                                f"dropped_samples={dropped_samples}/{valid_samples} nonfinite={nonfinite}",
                                flush=True,
                            )
                        diag_metrics = {}
                        if student_stats is not None:
                            diag_metrics["actor/student_logprob_min"] = student_stats[0]
                        if gap_stats is not None:
                            diag_metrics["actor/teacher_gap_max"] = gap_stats[1]
                        if kl_raw_max is not None:
                            diag_metrics["actor/kl_raw_max"] = kl_raw_max
                        if diag_metrics:
                            append_to_dict(metrics, diag_metrics)

                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                if not accumulate_only:
                    grad_norm = self._optimizer_step()
                    grad_norm_value = grad_norm.detach().item()
                    data = {"actor/grad_norm": grad_norm_value}
                    append_to_dict(metrics, data)
                    step_diag["grad_norm"] = grad_norm_value
                    step_diag["step_skipped"] = not bool(torch.isfinite(grad_norm).item())
                    _record_actor_diag(step_diag)
        if not accumulate_only:
            self.actor_optimizer.zero_grad()
        # One number per training step. all_reduce sums the FSDP shards; SwanLab keeps rank 0.
        drop_stats = torch.tensor(
            [dropped_token_total, dropped_sample_total, valid_token_total, valid_sample_total],
            dtype=torch.float64,
            device=torch.device(get_device_name(), get_device_id()),
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(drop_stats, op=torch.distributed.ReduceOp.SUM)
        dropped_token_sum, dropped_sample_sum, valid_token_sum, valid_sample_sum = (float(v) for v in drop_stats.tolist())
        metrics["actor/dropped_token_sum"] = dropped_token_sum
        metrics["actor/dropped_sample_sum"] = dropped_sample_sum
        metrics["actor/dropped_token_frac"] = dropped_token_sum / max(valid_token_sum, 1.0)
        metrics["actor/dropped_sample_frac"] = dropped_sample_sum / max(valid_sample_sum, 1.0)
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_opd_policy(self, data: DataProto):
        """Update the policy with fixed-coefficient sampled K3-IS distillation."""

        from verl.trainer.ppo.skillsd_utils import compute_sdl_loss

        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        opd_coef = float(self.config.get("opd_coef", 0.01))
        if opd_coef < 0:
            raise ValueError("opd_coef must be nonnegative")

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "teacher_log_probs",
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    if isinstance(micro_batch, DataProto):
                        micro_batch = {
                            **micro_batch.batch.to(get_torch_device().current_device()),
                            **micro_batch.non_tensor_batch,
                        }
                    else:
                        micro_batch = micro_batch.to(get_torch_device().current_device())

                    responses = micro_batch["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_batch["attention_mask"]
                    if multi_turn:
                        response_mask = micro_batch["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    _, log_prob = self._forward_micro_batch(
                        micro_batch=micro_batch,
                        temperature=temperature,
                        calculate_entropy=False,
                    )
                    opd_loss = compute_sdl_loss(
                        student_log_probs=log_prob,
                        teacher_log_probs=micro_batch["teacher_log_probs"],
                        old_log_probs=micro_batch["old_log_probs"],
                        response_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                    weighted_opd_loss = opd_loss * opd_coef
                    loss = weighted_opd_loss / self.gradient_accumulation
                    loss.backward()

                    append_to_dict(
                        metrics,
                        {
                            "opd/loss": opd_loss.detach().item(),
                            "opd/weighted_loss": weighted_opd_loss.detach().item(),
                            "opd/coef": opd_coef,
                        },
                    )

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_ana_policy(self, data: DataProto):
        """Policy gradient on analyzer tokens, separate for success and failure scores."""

        self.actor_module.train()
        if str(data.meta_info.get("ana_loss_mode", "pg")) == "sft":
            return self._update_ana_sft(data)
        temperature = data.meta_info["temperature"]
        coef = float(data.meta_info.get("ana_pg_coef", 0.01))
        alpha_pos = float(data.meta_info.get("ana_alpha_pos", 1.0))
        alpha_neg = float(data.meta_info.get("ana_alpha_neg", 1.0))
        micro_batch_size = max(1, int(data.meta_info.get("ana_micro_batch_size", 1)))
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "ana_is_success",
        ]
        batch = data.select(batch_keys=select_keys).batch
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == "vanilla":
            policy_loss_fn = compute_policy_loss
        elif loss_mode == "gspo":
            policy_loss_fn = compute_policy_loss_gspo
        else:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")
        clip_ratio = self.config.clip_ratio
        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)

        metrics = {}
        accumulate = bool(data.meta_info.get("accumulate_into_existing_grad", False))
        if not accumulate:
            self.actor_optimizer.zero_grad()
        weight = 1.0 / max(int(data.meta_info.get("ana_real_rows", batch.batch_size[0])), 1)
        pos_values = []
        neg_values = []
        offset = 0
        batch_size = batch.batch_size[0]
        while offset < batch_size:
            take = min(micro_batch_size, batch_size - offset)
            micro_batch = batch[offset : offset + take].to(get_torch_device().current_device())
            offset += take
            responses = micro_batch["responses"]
            micro_response_length = responses.size(1)
            response_mask = micro_batch["attention_mask"][:, -micro_response_length:]
            pos_mask = response_mask * micro_batch["ana_is_success"][:, :micro_response_length]
            neg_mask = response_mask * (1 - micro_batch["ana_is_success"][:, :micro_response_length])
            if pos_mask.sum() == 0 and neg_mask.sum() == 0:
                continue
            _entropy, log_prob = self._forward_micro_batch(
                micro_batch=micro_batch,
                temperature=temperature,
                calculate_entropy=False,
            )
            piece = None
            if pos_mask.sum() > 0:
                pos_loss, _, _, _ = policy_loss_fn(
                    old_log_prob=micro_batch["old_log_probs"],
                    log_prob=log_prob,
                    advantages=micro_batch["advantages"],
                    response_mask=pos_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode="token-mean",
                )
                piece = alpha_pos * pos_loss
                pos_values.append(pos_loss.detach())
            if neg_mask.sum() > 0:
                neg_loss, _, _, _ = policy_loss_fn(
                    old_log_prob=micro_batch["old_log_probs"],
                    log_prob=log_prob,
                    advantages=micro_batch["advantages"],
                    response_mask=neg_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode="token-mean",
                )
                neg_term = alpha_neg * neg_loss
                piece = neg_term if piece is None else piece + neg_term
                neg_values.append(neg_loss.detach())
            (coef * weight * piece).backward()

        grad_norm = self._optimizer_step()
        pos_mean = torch.stack(pos_values).mean() if pos_values else torch.zeros(())
        neg_mean = torch.stack(neg_values).mean() if neg_values else torch.zeros(())
        metrics = {
            "ana_pg/loss": float(coef * (alpha_pos * pos_mean + alpha_neg * neg_mean)),
            "ana_pg/pos_loss": float(pos_mean),
            "ana_pg/neg_loss": float(neg_mean),
            "ana_pg/coef": coef,
            "actor/grad_norm": grad_norm.detach().item(),
        }
        self.actor_optimizer.zero_grad()
        return metrics

    def _update_ana_sft(self, data: DataProto):
        """Token NLL of an external analysis. No advantage is computed."""
        temperature = data.meta_info["temperature"]
        coef = float(data.meta_info.get("ana_pg_coef", 0.01))
        micro_batch_size = max(1, int(data.meta_info.get("ana_micro_batch_size", 1)))
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        accumulate = bool(data.meta_info.get("accumulate_into_existing_grad", False))
        if not accumulate:
            self.actor_optimizer.zero_grad()
        weight = 1.0 / max(int(data.meta_info.get("ana_real_rows", batch.batch_size[0])), 1)
        nll_values = []
        offset = 0
        batch_size = batch.batch_size[0]
        while offset < batch_size:
            take = min(micro_batch_size, batch_size - offset)
            micro_batch = batch[offset : offset + take].to(get_torch_device().current_device())
            offset += take
            response_length = micro_batch["responses"].size(1)
            response_mask = micro_batch["attention_mask"][:, -response_length:]
            if response_mask.sum() == 0:
                continue
            _entropy, log_prob = self._forward_micro_batch(
                micro_batch=micro_batch,
                temperature=temperature,
                calculate_entropy=False,
            )
            nll = agg_loss(-log_prob, response_mask, loss_agg_mode="token-mean")
            (coef * weight * nll).backward()
            nll_values.append(nll.detach())
        grad_norm = self._optimizer_step()
        nll_mean = torch.stack(nll_values).mean() if nll_values else torch.zeros(())
        metrics = {
            "ana_sft/nll": float(nll_mean),
            "ana_sft/loss": float(coef * nll_mean),
            "ana_sft/coef": coef,
            "actor/grad_norm": grad_norm.detach().item(),
        }
        self.actor_optimizer.zero_grad()
        return metrics
