"""Score a student reply under the student prompt and the teacher prompt.

The objective matches the sdar-gate mode of sampled-token OPSD
(``compute_opsd_sampled_kl_loss`` with gate="sdar"), the same residual as SEED and SDAR:

    gap_t = log π_T(y_t | x, skill) - log π_S(y_t | x)
    g_t   = sigmoid(beta * gap_t)
    L_t   = g_t * gap_t

Think and action entropies are means over tokens inside those tags.
A positive advantage means the teacher assigns the student's token a higher probability.
"""

from __future__ import annotations

import math
import re


_TAG_RE = {
    "think": re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL),
    "action": re.compile(r"<action>(.*?)</action>", flags=re.IGNORECASE | re.DOTALL),
}


def _token_char_spans(tokenizer, response_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    spans = []
    cursor = ""
    for idx in range(len(response_ids)):
        nxt = tokenizer.decode(response_ids[: idx + 1], skip_special_tokens=False)
        spans.append((len(cursor), len(nxt)))
        cursor = nxt
    return cursor, spans


def _content_mask(text: str, spans: list[tuple[int, int]], pattern: re.Pattern) -> list[bool]:
    mask = [False] * len(spans)
    for match in pattern.finditer(text):
        start, end = match.span(1)
        for i, (tok_start, tok_end) in enumerate(spans):
            mid = (tok_start + tok_end) / 2
            if start <= mid < end:
                mask[i] = True
    return mask


def _mean(values: list[float], mask: list[bool] | None = None) -> float | None:
    if mask is None:
        chosen = list(values)
    else:
        chosen = [value for value, keep in zip(values, mask) if keep]
    if not chosen:
        return None
    return float(sum(chosen) / len(chosen))


def _segment_stats(
    *,
    student_logprob: list[float],
    teacher_logprob: list[float],
    student_entropy: list[float],
    teacher_entropy: list[float],
    mask: list[bool] | None,
    opsd_coef: float,
    gate_beta: float = 5.0,
) -> dict:
    if mask is None:
        indices = list(range(len(student_logprob)))
    else:
        indices = [i for i, keep in enumerate(mask) if keep]
    gaps = []
    student_lps = []
    teacher_lps = []
    losses = []
    for i in indices:
        advantage = teacher_logprob[i] - student_logprob[i]
        scaled = float(gate_beta) * advantage
        if scaled >= 0:
            gate = 1.0 / (1.0 + math.exp(-scaled))
        else:
            z = math.exp(scaled)
            gate = z / (1.0 + z)
        gaps.append(advantage)
        student_lps.append(student_logprob[i])
        teacher_lps.append(teacher_logprob[i])
        losses.append(gate * advantage)
    opsd_loss = None if not losses else float(sum(losses) / len(losses))
    opsd_loss_sum = None if not losses else float(sum(losses))
    return {
        "n_tokens": len(indices),
        "student_entropy": _mean(student_entropy, mask),
        "teacher_entropy": _mean(teacher_entropy, mask),
        "teacher_gap": None if not gaps else float(sum(gaps) / len(gaps)),
        "student_logprob": None if not student_lps else float(sum(student_lps) / len(student_lps)),
        "teacher_logprob": None if not teacher_lps else float(sum(teacher_lps) / len(teacher_lps)),
        "opsd_loss": opsd_loss,
        "opsd_loss_sum": opsd_loss_sum,
        "opsd_weighted_loss": None if opsd_loss is None else float(opsd_coef * opsd_loss),
        "opsd_weighted_loss_sum": None if opsd_loss_sum is None else float(opsd_coef * opsd_loss_sum),
    }


def annotate_episode(
    policy,
    episode: dict,
    *,
    opsd_coef: float = 0.01,
    logit_temperature: float = 1.0,
) -> None:
    """Fill per-step entropy and sampled-OPSD fields. Does not raise on one bad step."""
    episode["opsd_coef"] = float(opsd_coef)
    episode["logit_temperature"] = float(logit_temperature)
    episode["objective_name"] = "sampled_kl"
    for step in episode.get("steps", []):
        step["objective"] = _score_step(
            policy,
            step,
            opsd_coef=float(opsd_coef),
            logit_temperature=float(logit_temperature),
        )


def _score_step(policy, step: dict, *, opsd_coef: float, logit_temperature: float) -> dict:
    response_ids = step.get("student_response_ids") or []
    teacher_prompt = step.get("teacher_prompt")
    result = {"error": None, "distilled": bool(step.get("distill", True))}
    if step.get("distill") is False:
        result["distilled"] = False
        zero = _zero_objective()
        result["total"] = dict(zero)
        result["response"] = dict(zero)
        result["think"] = dict(zero)
        result["action"] = dict(zero)
        return result
    if not response_ids or not teacher_prompt:
        result["error"] = "missing student tokens or teacher prompt"
        return result
    try:
        student = policy.score_tokens(
            step["student_prompt"],
            response_ids,
            logit_temperature=logit_temperature,
        )
        teacher = policy.score_tokens(
            teacher_prompt,
            response_ids,
            logit_temperature=logit_temperature,
        )
    except Exception as exc:
        result["error"] = str(exc)
        return result

    text, spans = _token_char_spans(policy.tokenizer, response_ids)
    think_mask = _content_mask(text, spans, _TAG_RE["think"])
    action_mask = _content_mask(text, spans, _TAG_RE["action"])
    common = dict(
        student_logprob=student["logprob"],
        teacher_logprob=teacher["logprob"],
        student_entropy=student["entropy"],
        teacher_entropy=teacher["entropy"],
        opsd_coef=opsd_coef,
    )
    result["response"] = _segment_stats(mask=None, **common)
    result["think"] = _segment_stats(mask=think_mask, **common)
    result["action"] = _segment_stats(mask=action_mask, **common)
    result["total"] = _step_total(common, opsd_coef=opsd_coef)
    result["token_kl"], result["token_loss"], result["token_span"] = _token_objective(
        common,
        think_mask=think_mask,
        action_mask=action_mask,
        opsd_coef=opsd_coef,
    )
    result["teacher_own"] = _teacher_own_entropy(
        policy,
        step,
        logit_temperature=logit_temperature,
    )
    return result


def _zero_objective() -> dict:
    return {
        "n_tokens": 0,
        "opsd_loss": 0.0,
        "opsd_loss_sum": 0.0,
        "opsd_weighted_loss": 0.0,
        "opsd_weighted_loss_sum": 0.0,
        "teacher_gap": 0.0,
    }


def _step_total(common: dict, *, opsd_coef: float) -> dict:
    """One scalar for the whole step, matching training's token-mean aggregation.

    token_loss_t = -(log π_T - log π_S) * log π_S
    opsd_loss = mean_t(token_loss_t)
    opsd_weighted_loss = opsd_coef * opsd_loss
    """
    student_lp = common["student_logprob"]
    teacher_lp = common["teacher_logprob"]
    losses = []
    for s_lp, t_lp in zip(student_lp, teacher_lp):
        losses.append(-(t_lp - s_lp) * s_lp)
    if not losses:
        return {
            "n_tokens": 0,
            "opsd_loss": None,
            "opsd_loss_sum": None,
            "opsd_weighted_loss": None,
            "opsd_weighted_loss_sum": None,
        }
    loss_sum = float(sum(losses))
    loss_mean = loss_sum / len(losses)
    return {
        "n_tokens": len(losses),
        "opsd_loss": loss_mean,
        "opsd_loss_sum": loss_sum,
        "opsd_weighted_loss": float(opsd_coef * loss_mean),
        "opsd_weighted_loss_sum": float(opsd_coef * loss_sum),
    }


def _token_objective(
    common: dict,
    *,
    think_mask: list[bool],
    action_mask: list[bool],
    opsd_coef: float,
) -> tuple[list[float], list[float], list[str]]:
    """Per-token KL is log π_T − log π_S. token_loss keeps the signed training term."""
    kls = []
    losses = []
    spans = []
    for index, (student_lp, teacher_lp) in enumerate(zip(common["student_logprob"], common["teacher_logprob"])):
        gap = teacher_lp - student_lp
        kls.append(float(gap))
        losses.append(float(opsd_coef * (-gap * student_lp)))
        if index < len(think_mask) and think_mask[index]:
            spans.append("think")
        elif index < len(action_mask) and action_mask[index]:
            spans.append("action")
        else:
            spans.append("other")
    return kls, losses, spans


def _teacher_own_entropy(policy, step: dict, *, logit_temperature: float) -> dict:
    response_ids = step.get("teacher_response_ids") or []
    teacher_prompt = step.get("teacher_prompt")
    if not response_ids or not teacher_prompt:
        return {"error": "missing teacher response tokens"}
    try:
        scored = policy.score_tokens(
            teacher_prompt,
            response_ids,
            logit_temperature=logit_temperature,
        )
    except Exception as exc:
        return {"error": str(exc)}
    text, spans = _token_char_spans(policy.tokenizer, response_ids)
    think_mask = _content_mask(text, spans, _TAG_RE["think"])
    action_mask = _content_mask(text, spans, _TAG_RE["action"])
    return {
        "think": {
            "n_tokens": sum(think_mask),
            "entropy": _mean(scored["entropy"], think_mask),
        },
        "action": {
            "n_tokens": sum(action_mask),
            "entropy": _mean(scored["entropy"], action_mask),
        },
    }
