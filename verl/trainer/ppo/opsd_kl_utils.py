"""
Sampled-token OPSD. gap_t = stopgrad(log π_T - log π_S). Three bounds:

  sdar-gate: L = g_t * (log π_T - log π_S),  g_t = sigmoid(beta * gap_t)
             gradient w.r.t. log π_S is -g_t
  clip:      L = -clip(gap_t, -c, c) * log π_S
  tanh:      L = -(c * tanh(gap_t / c)) * log π_S

clip and tanh keep the sampled-token reverse-KL gradient and only bound the
advantage. sdar-gate is the SEED / compute_sdar_loss residual.
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss

SUPPORTED_OPSD_KL_OBJECTIVES = ("sampled_kl", "reverse_kl")
SUPPORTED_OPSD_KL_GATES = ("none", "sdar", "sdar-gate", "clip", "tanh")
_GATE_ALIASES = {
    "none": "none",
    "sdar": "sdar",
    "sdar-gate": "sdar",
    "sdar_gate": "sdar",
    "clip": "clip",
    "tanh": "tanh",
}


def compute_opsd_sampled_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    objective: str = "sampled_kl",
    gate: str = "none",
    gate_beta: float = 5.0,
    advantage_clip: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """
    Sampled-token OPSD with one of three advantage bounds.

    sdar / sdar-gate:
        L = agg( sigmoid(beta * gap) * (log π_T - log π_S) )
    clip:
        L = agg( -clip(gap, -c, c) * log π_S )
    tanh:
        L = agg( -(c * tanh(gap / c)) * log π_S )
    none:
        L = agg( log π_T - log π_S ), gradient w.r.t. log π_S is -1.

    gap = stopgrad(log π_T - log π_S). c is advantage_clip.
    """
    if objective not in SUPPORTED_OPSD_KL_OBJECTIVES:
        raise ValueError(
            f"Unsupported OPSD KL objective={objective!r}. "
            f"Supported: {SUPPORTED_OPSD_KL_OBJECTIVES}. "
            "forward_kl requires full vocab distributions and is not implemented."
        )
    gate_key = _GATE_ALIASES.get(str(gate).strip().lower())
    if gate_key is None:
        raise ValueError(
            f"Unsupported OPSD KL gate={gate!r}. "
            "Supported: sdar-gate (alias sdar), clip, tanh, none."
        )
    bound = float(advantage_clip)
    if gate_key in ("clip", "tanh") and bound <= 0:
        raise ValueError(f"advantage_clip must be positive for gate={gate_key}, got {bound}")

    teacher_log_probs = teacher_log_probs.detach()
    gap = (teacher_log_probs - student_log_probs.detach()).detach()
    if gate_key == "sdar":
        token_gate = torch.sigmoid(float(gate_beta) * gap).detach()
        bounded_advantage = token_gate
        loss_mat = token_gate * (teacher_log_probs - student_log_probs)
    elif gate_key == "clip":
        token_gate = torch.ones_like(gap)
        bounded_advantage = gap.clamp(-bound, bound)
        loss_mat = -bounded_advantage * student_log_probs
    elif gate_key == "tanh":
        token_gate = torch.ones_like(gap)
        bounded_advantage = bound * torch.tanh(gap / bound)
        loss_mat = -bounded_advantage * student_log_probs
    else:
        token_gate = torch.ones_like(gap)
        bounded_advantage = token_gate
        loss_mat = teacher_log_probs - student_log_probs
    loss = agg_loss(loss_mat=loss_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gap_mean = (gap * response_mask).sum() / mask_sum
        student_lp_mean = (student_log_probs.detach() * response_mask).sum() / mask_sum
        teacher_lp_mean = (teacher_log_probs * response_mask).sum() / mask_sum
        gate_mean = (token_gate * response_mask).sum() / mask_sum
        gate_active = ((token_gate > 0.5).float() * response_mask).sum() / mask_sum
        advantage_mean = (bounded_advantage * response_mask).sum() / mask_sum
        if gate_key in ("clip", "tanh"):
            clip_ratio = ((gap.abs() > bound).float() * response_mask).sum() / mask_sum
        else:
            clip_ratio = gap.new_zeros(())

    metrics = {
        "opsd_kl/teacher_gap_mean": gap_mean.item(),
        "opsd_kl/student_logprob_mean": student_lp_mean.item(),
        "opsd_kl/teacher_logprob_mean": teacher_lp_mean.item(),
        "opsd_kl/gate_mean": gate_mean.item(),
        "opsd_kl/gate_active_ratio": gate_active.item(),
        "opsd_kl/bounded_advantage_mean": advantage_mean.item(),
        "opsd_kl/clip_ratio": clip_ratio.item(),
        "opsd_kl/loss": loss.detach().item(),
    }
    return loss, metrics
