"""Dump agent rollout trajectories for offline analysis and human inspection."""

from __future__ import annotations

import html
import json
import os
import re
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch


def _to_python(x: Any) -> Any:
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x


def _safe_get(non_tensor: dict, key: str, idx: int, default=None):
    if key not in non_tensor:
        return default
    arr = non_tensor[key]
    try:
        return _to_python(arr[idx])
    except Exception:
        return default


def _serialize_available_actions(aa: Any) -> Any:
    if aa is None:
        return None
    if isinstance(aa, dict):
        out = {}
        for k, v in aa.items():
            if isinstance(v, (list, tuple)):
                # Keep clickables short for readability.
                seq = list(v)
                out[k] = seq if len(seq) <= 40 else seq[:40] + [f"...(+{len(seq) - 40})"]
            else:
                out[k] = _to_python(v)
        return out
    return _to_python(aa)


def _extract_task_query(prompt_text: str) -> Optional[str]:
    """Best-effort extraction of the overall task query from a model prompt."""
    text = str(prompt_text or "")
    patterns = [
        r"Your task is to:\s*(.+?)(?:\.\s*\n|\n)",
        r"Your task is:\s*(.+?)(?:\n)",
        r"Your question:\s*(.+?)(?:\n)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def build_episodes_from_batch(
    *,
    tokenizer,
    batch,
    global_step: int,
    split: str = "train",
    max_episodes: Optional[int] = None,
) -> list[dict]:
    """Rebuild episode-level trajectories from a flattened agent rollout batch."""
    n = len(batch.batch["prompts"])
    prompts = tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    responses = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist() if "token_level_scores" in batch.batch else [0.0] * n

    nt = batch.non_tensor_batch
    has_teacher = ("teacher_log_probs" in batch.batch) and ("old_log_probs" in batch.batch)
    teacher_lp = batch.batch["teacher_log_probs"] if has_teacher else None
    student_lp = batch.batch["old_log_probs"] if has_teacher else None
    response_mask = batch.batch.get("response_mask", None)
    by_traj: dict[str, list[dict]] = defaultdict(list)

    for i in range(n):
        traj_uid = str(_safe_get(nt, "traj_uid", i, f"row-{i}"))
        turn = int(_safe_get(nt, "turn_step", i, len(by_traj[traj_uid])))
        aa = _serialize_available_actions(_safe_get(nt, "available_actions", i, None))
        prompt_i = prompts[i]
        skill_text = _safe_get(nt, "skill_text", i, None)
        selected_skills = _safe_get(nt, "selected_skills", i, None)
        matched_tasks = _safe_get(nt, "matched_tasks", i, None)

        step_rec = {
            "turn": turn,
            "observation": _safe_get(nt, "anchor_obs", i, None),
            "prompt": prompt_i,
            "response": responses[i],
            "step_reward": float(_safe_get(nt, "rewards", i, 0.0) or 0.0),
            "token_score": float(scores[i]),
            "is_action_valid": bool(_safe_get(nt, "is_action_valid", i, True)),
            "available_actions": aa,
            "step_won": _safe_get(nt, "step_won", i, None),
            "step_task_score": _safe_get(nt, "step_task_score", i, None),
            "task_query": _extract_task_query(prompt_i),
            "selected_skills": selected_skills,
            "matched_tasks": matched_tasks,
            "skill_text": skill_text,
            "teacher_prompt": None,
            "opsd_token_advantages": None,
            "step_skill_record": _to_python(_safe_get(nt, "step_skill_record", i, None)),
        }

        if skill_text:
            step_rec["teacher_prompt"] = (
                f"[Privileged Skill Information]\n{skill_text}\n\n{prompt_i}"
            )

        if has_teacher and teacher_lp is not None and student_lp is not None:
            if response_mask is not None:
                mask_i = response_mask[i].bool()
            else:
                mask_i = teacher_lp[i] != 0
            valid = int(mask_i.sum().item()) if hasattr(mask_i, "sum") else 0
            if valid > 0:
                t = teacher_lp[i][mask_i].detach().float().cpu()
                s = student_lp[i][mask_i].detach().float().cpu()
                # A_t = stopgrad(log π_T - log π_S)
                a = (t - s).tolist()
                token_ids = batch.batch["responses"][i][mask_i].detach().cpu().tolist()
                tokens = [tokenizer.decode([tid]) for tid in token_ids]
                step_rec["opsd_token_advantages"] = [
                    {
                        "pos": pos,
                        "token_id": int(tid),
                        "token": tok,
                        "teacher_log_prob": float(tv),
                        "student_log_prob": float(sv),
                        "A_t": float(av),
                    }
                    for pos, (tid, tok, tv, sv, av) in enumerate(
                        zip(token_ids, tokens, t.tolist(), s.tolist(), a)
                    )
                ]

        by_traj[traj_uid].append(step_rec)

    episodes = []
    for traj_uid, steps in by_traj.items():
        steps = sorted(steps, key=lambda s: s["turn"])
        task_query = None
        step_skill_record = None
        for s in steps:
            if s.get("task_query") and task_query is None:
                task_query = s["task_query"]
            if step_skill_record is None and s.get("step_skill_record"):
                step_skill_record = s["step_skill_record"]
        ep = {
            "global_step": global_step,
            "split": split,
            "traj_uid": traj_uid,
            "num_turns": len(steps),
            "task_query": task_query,
            "step_skill_record": step_skill_record,
            "steps": steps,
        }
        episodes.append(ep)

    # Fill episode-level metrics by scanning batch rows again.
    traj_meta = {}
    for i in range(n):
        traj_uid = str(_safe_get(nt, "traj_uid", i, f"row-{i}"))
        if traj_uid in traj_meta:
            continue
        traj_meta[traj_uid] = {
            "uid": str(_safe_get(nt, "uid", i, "")),
            "episode_reward": float(_safe_get(nt, "episode_rewards", i, 0.0) or 0.0),
            "episode_length": float(_safe_get(nt, "episode_lengths", i, 0.0) or 0.0),
            "tool_callings": float(_safe_get(nt, "tool_callings", i, 0.0) or 0.0),
            "traj_won": _safe_get(nt, "traj_won", i, None),
            "traj_task_score": _safe_get(nt, "traj_task_score", i, None),
            "data_source": _safe_get(nt, "data_source", i, None),
            "gamefile": _safe_get(nt, "gamefile", i, None),
        }

    for ep in episodes:
        ep.update(traj_meta.get(ep["traj_uid"], {}))

    # Prefer failed / low-score episodes for analysis when subsampling.
    episodes.sort(key=lambda e: (float(e.get("traj_won") or 0.0), float(e.get("episode_reward") or 0.0), e["traj_uid"]))
    if max_episodes is not None and max_episodes > 0 and len(episodes) > max_episodes:
        # Keep half lowest reward and half highest for contrast.
        half = max(1, max_episodes // 2)
        selected = episodes[:half] + episodes[-half:]
        # Dedup while preserving order
        seen = set()
        uniq = []
        for e in selected:
            if e["traj_uid"] in seen:
                continue
            seen.add(e["traj_uid"])
            uniq.append(e)
        episodes = uniq[:max_episodes]

    return episodes


def _truncate_text(text: str, max_chars: int = 2500) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    head = max_chars // 5
    tail = max_chars - head
    return text[:head] + f"\n\n...[truncated {len(text) - max_chars} chars]...\n\n" + text[-tail:]


def _render_opsd_adv_table_md(rows: list[dict]) -> list[str]:
    lines = [
        "### OPSD token advantages `A_t = stopgrad(log π_T - log π_S)`",
        "",
        "| pos | token | teacher_logp | student_logp | A_t |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        tok = str(row["token"]).replace("|", "\\|").replace("\n", "\\n")
        lines.append(
            f"| {row['pos']} | `{tok}` | {row['teacher_log_prob']:.6f} | "
            f"{row['student_log_prob']:.6f} | {row['A_t']:.6f} |"
        )
    lines.append("")
    return lines


def render_episode_markdown(ep: dict) -> str:
    lines = []
    lines.append(f"# Episode `{ep.get('traj_uid', '')}`")
    lines.append("")
    lines.append(f"- global_step: **{ep.get('global_step')}**")
    lines.append(f"- split: `{ep.get('split')}`")
    lines.append(f"- episode_reward: **{ep.get('episode_reward')}**")
    lines.append(f"- episode_length: **{ep.get('episode_length')}**")
    lines.append(f"- traj_won: **{ep.get('traj_won')}**")
    lines.append(f"- traj_task_score: **{ep.get('traj_task_score')}**")
    lines.append(f"- tool_callings: {ep.get('tool_callings')}")
    lines.append(f"- data_source: `{ep.get('data_source')}`")
    if ep.get("gamefile"):
        lines.append(f"- gamefile: `{ep.get('gamefile')}`")
    lines.append("")
    if ep.get("task_query"):
        lines.append("## Task query")
        lines.append("")
        lines.append("```text")
        lines.append(str(ep.get("task_query")))
        lines.append("```")
        lines.append("")

    for step in ep.get("steps", []):
        lines.append(f"## Turn {step.get('turn')}")
        lines.append("")
        lines.append(
            f"- step_reward: `{step.get('step_reward')}` | valid_action: `{step.get('is_action_valid')}` "
            f"| step_won: `{step.get('step_won')}` | step_task_score: `{step.get('step_task_score')}`"
        )
        lines.append("")

        # 1) Observation
        obs = step.get("observation")
        if obs is not None:
            lines.append("### Observation")
            lines.append("```text")
            lines.append(str(obs).rstrip())
            lines.append("```")
            lines.append("")

        # 2) Available actions
        aa = step.get("available_actions")
        if aa is not None:
            lines.append("### Available actions")
            lines.append("```json")
            lines.append(json.dumps(aa, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

        # 3) Full student prompt
        lines.append("<details><summary>Full prompt (student / model input)</summary>")
        lines.append("")
        lines.append("```text")
        lines.append(_truncate_text(step.get("prompt", ""), 3500).rstrip())
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

        # 4) Model response
        lines.append("### Model response / action")
        lines.append("```text")
        lines.append(str(step.get("response", "")).rstrip())
        lines.append("```")
        lines.append("")

        # 5) Teacher / skill / A_t (optional)
        if step.get("selected_skills") is not None or step.get("skill_text"):
            lines.append("### Selected skills")
            lines.append(f"- selected_skills: `{step.get('selected_skills')}`")
            lines.append(f"- matched_tasks: `{step.get('matched_tasks')}`")
            lines.append("")
            lines.append("<details><summary>Selected skill text</summary>")
            lines.append("")
            lines.append("```text")
            lines.append(_truncate_text(step.get("skill_text") or "", 6000))
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        if step.get("teacher_prompt"):
            lines.append("<details><summary>Teacher full prompt</summary>")
            lines.append("")
            lines.append("```text")
            lines.append(_truncate_text(step.get("teacher_prompt"), 4500))
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")
        if step.get("opsd_token_advantages"):
            lines.extend(_render_opsd_adv_table_md(step["opsd_token_advantages"]))

    return "\n".join(lines)


def _render_step_skill_html(record: Optional[dict], esc) -> list[str]:
    if not isinstance(record, dict) or not record:
        return []
    mode = record.get("loss_mode")
    parts = ["<div class='ana'><h2>Analyzer</h2>"]
    judge_temperature = ""
    if record.get("judge_temperature") is not None:
        judge_temperature = f"judge_temperature=<code>{esc(record.get('judge_temperature'))}</code> | "
    parts.append(
        f"<div>loss_mode=<code>{esc(mode)}</code> | kind=<code>{esc(record.get('kind'))}</code> | "
        f"source=<code>{esc(record.get('analysis_source'))}</code> | "
        f"temperature=<code>{esc(record.get('temperature'))}</code> | "
        f"{judge_temperature}"
        f"max_tokens=<code>{esc(record.get('max_tokens'))}</code></div>"
    )
    if record.get("parse_error"):
        parts.append(f"<div class='bad'>parse_error: {esc(record.get('parse_error'))}</div>")
    parts.append(
        "<h3>step_skills</h3><pre>"
        + esc(json.dumps(record.get("step_skills") or [], ensure_ascii=False, indent=2))
        + "</pre>"
    )
    if record.get("kind") == "failure":
        parts.append(
            "<h3>successful reference actions</h3><pre>"
            + esc(json.dumps(record.get("reference_actions") or [], ensure_ascii=False, indent=2))
            + "</pre>"
        )
    parts.append(
        "<details><summary>Analysis prompt</summary><pre>"
        + esc(_truncate_text(record.get("analysis_prompt") or "", 8000))
        + "</pre></details>"
    )
    parts.append(
        "<h3>Analyzer reply</h3><pre>"
        + esc(record.get("analysis_output") or "")
        + "</pre>"
    )
    if mode == "pg":
        parts.append(
            f"<div>judge_score=<b>{esc(record.get('judge_score'))}</b> | "
            f"advantage=<b>{esc(record.get('advantage'))}</b></div>"
        )
        parts.append(
            "<h3>Judge reason</h3><pre>"
            + esc(record.get("judge_reason") or "")
            + "</pre>"
        )
        parts.append(
            "<details><summary>Judge prompt</summary><pre>"
            + esc(_truncate_text(record.get("judge_prompt") or "", 8000))
            + "</pre></details>"
        )
        parts.append(
            "<details><summary>Judge output</summary><pre>"
            + esc(record.get("judge_output") or "")
            + "</pre></details>"
        )
    elif mode == "sft":
        parts.append(
            f"<div>SFT target is the API analysis. model=<code>{esc(record.get('api_model'))}</code> "
            f"base_url=<code>{esc(record.get('api_base_url'))}</code>. No advantage.</div>"
        )
    parts.append("</div>")
    return parts


def render_episode_html(ep: dict) -> str:
    def esc(x):
        return html.escape(str(x))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Episode {}</title>".format(esc(ep.get("traj_uid", ""))),
        "<style>",
        "body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;line-height:1.45;color:#1f2328;}",
        ".meta{background:#f6f8fa;border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin-bottom:16px;}",
        ".task{background:#fff8c5;border:1px solid #d4a72c;border-radius:8px;padding:12px 14px;margin:16px 0;}",
        ".turn{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:16px 0;}",
        "pre{white-space:pre-wrap;background:#0d1117;color:#e6edf3;padding:12px;border-radius:8px;overflow:auto;}",
        "table{border-collapse:collapse;width:100%;font-size:13px;}",
        "th,td{border:1px solid #d0d7de;padding:4px 6px;vertical-align:top;}",
        "th{background:#f6f8fa;text-align:left;}",
        ".ok{color:#1a7f37}.bad{color:#cf222e}",
        "details{margin-top:8px}",
        "summary{cursor:pointer;color:#0969da}",
        ".action{border-left:4px solid #0969da;padding-left:10px;}",
        ".teacher{border-left:4px solid #bf3989;padding-left:10px;margin-top:12px;}",
        ".ana{background:#f6f8fa;border:1px solid #8250df;border-radius:8px;padding:12px 14px;margin:16px 0;}",
        "</style></head><body>",
        f"<h1>Episode <code>{esc(ep.get('traj_uid'))}</code></h1>",
        "<div class='meta'>",
        f"<div>global_step: <b>{esc(ep.get('global_step'))}</b> | split: <code>{esc(ep.get('split'))}</code></div>",
        f"<div>episode_reward: <b>{esc(ep.get('episode_reward'))}</b> | length: <b>{esc(ep.get('episode_length'))}</b></div>",
        f"<div>traj_won: <b>{esc(ep.get('traj_won'))}</b> | traj_task_score: <b>{esc(ep.get('traj_task_score'))}</b></div>",
        f"<div>tool_callings: {esc(ep.get('tool_callings'))} | data_source: <code>{esc(ep.get('data_source'))}</code></div>",
    ]
    if ep.get("gamefile"):
        parts.append(f"<div>gamefile: <code>{esc(ep.get('gamefile'))}</code></div>")
    parts.append("</div>")

    if ep.get("task_query"):
        parts.append("<div class='task'><h2>Task query</h2><pre>" + esc(ep.get("task_query")) + "</pre></div>")
    parts.extend(_render_step_skill_html(ep.get("step_skill_record"), esc))

    for step in ep.get("steps", []):
        valid = bool(step.get("is_action_valid", True))
        cls = "ok" if valid else "bad"
        parts.append(f"<div class='turn'><h2>Turn {esc(step.get('turn'))}</h2>")
        parts.append(
            f"<div>step_reward=<code>{esc(step.get('step_reward'))}</code> | "
            f"valid_action=<span class='{cls}'>{esc(valid)}</span> | "
            f"step_won=<code>{esc(step.get('step_won'))}</code> | "
            f"step_task_score=<code>{esc(step.get('step_task_score'))}</code></div>"
        )

        # 1) Observation
        if step.get("observation") is not None:
            parts.append("<h3>1. Observation</h3><pre>" + esc(step.get("observation", "")) + "</pre>")

        # 2) Available actions
        if step.get("available_actions") is not None:
            parts.append(
                "<h3>2. Available actions</h3><pre>"
                + esc(json.dumps(step.get("available_actions"), ensure_ascii=False, indent=2))
                + "</pre>"
            )

        # 3) Full student prompt
        parts.append(
            "<details open><summary><b>3. Full prompt (student / model input)</b></summary><pre>"
            + esc(_truncate_text(step.get("prompt", ""), 3500))
            + "</pre></details>"
        )

        # 4) Model response
        parts.append(
            "<div class='action'><h3>4. Model response / action</h3><pre>"
            + esc(step.get("response", ""))
            + "</pre></div>"
        )

        # 5) Teacher block
        if step.get("selected_skills") is not None or step.get("skill_text") or step.get("opsd_token_advantages"):
            parts.append("<div class='teacher'><h3>5. Teacher / OPSD</h3>")
            parts.append(f"<div>selected_skills: <code>{esc(step.get('selected_skills'))}</code></div>")
            parts.append(f"<div>matched_tasks: <code>{esc(step.get('matched_tasks'))}</code></div>")
            if step.get("skill_text"):
                parts.append(
                    "<details><summary>Selected skill text</summary><pre>"
                    + esc(_truncate_text(step.get("skill_text"), 6000))
                    + "</pre></details>"
                )
            if step.get("teacher_prompt"):
                parts.append(
                    "<details><summary>Teacher full prompt</summary><pre>"
                    + esc(_truncate_text(step.get("teacher_prompt"), 4500))
                    + "</pre></details>"
                )
            rows = step.get("opsd_token_advantages") or []
            if rows:
                parts.append(
                    "<h4>Token advantages "
                    "<code>A_t = stopgrad(log π_T(y_t|x,skill) - log π_S(y_t|x))</code></h4>"
                )
                parts.append(
                    "<table><thead><tr>"
                    "<th>pos</th><th>token</th><th>teacher_logp</th><th>student_logp</th><th>A_t</th>"
                    "</tr></thead><tbody>"
                )
                for row in rows:
                    parts.append(
                        "<tr>"
                        f"<td>{esc(row['pos'])}</td>"
                        f"<td><code>{esc(row['token'])}</code></td>"
                        f"<td>{row['teacher_log_prob']:.6f}</td>"
                        f"<td>{row['student_log_prob']:.6f}</td>"
                        f"<td><b>{row['A_t']:.6f}</b></td>"
                        "</tr>"
                    )
                parts.append("</tbody></table>")
            parts.append("</div>")

        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def dump_agent_trajectories(
    *,
    tokenizer,
    batch,
    dump_path: str,
    global_step: int,
    split: str = "train",
    max_episodes: int = 16,
    write_markdown: bool = True,
    write_html: bool = True,
) -> str:
    """
    Write:
      {dump_path}/step_{global_step}/episodes.jsonl
      {dump_path}/step_{global_step}/md/*.md
      {dump_path}/step_{global_step}/html/*.html
      {dump_path}/step_{global_step}/index.md
    """
    os.makedirs(dump_path, exist_ok=True)
    step_dir = os.path.join(dump_path, f"step_{global_step:06d}_{split}")
    os.makedirs(step_dir, exist_ok=True)
    md_dir = os.path.join(step_dir, "md")
    html_dir = os.path.join(step_dir, "html")
    if write_markdown:
        os.makedirs(md_dir, exist_ok=True)
    if write_html:
        os.makedirs(html_dir, exist_ok=True)

    episodes = build_episodes_from_batch(
        tokenizer=tokenizer,
        batch=batch,
        global_step=global_step,
        split=split,
        max_episodes=max_episodes,
    )

    jsonl_path = os.path.join(step_dir, "episodes.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    index_lines = [
        f"# Trajectories @ step {global_step} ({split})",
        "",
        f"- num_episodes_dumped: {len(episodes)}",
        f"- jsonl: `{jsonl_path}`",
        "",
        "| # | traj_uid | reward | length | won | task_score | md | html |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]

    for i, ep in enumerate(episodes):
        stem = f"{i:03d}_{ep.get('traj_uid', 'traj')[:8]}"
        md_rel = html_rel = ""
        if write_markdown:
            md_path = os.path.join(md_dir, f"{stem}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(render_episode_markdown(ep))
            md_rel = f"[md](md/{stem}.md)"
        if write_html:
            html_path = os.path.join(html_dir, f"{stem}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(render_episode_html(ep))
            html_rel = f"[html](html/{stem}.html)"
        index_lines.append(
            f"| {i} | `{ep.get('traj_uid')}` | {ep.get('episode_reward')} | {ep.get('episode_length')} | "
            f"{ep.get('traj_won')} | {ep.get('traj_task_score')} | {md_rel} | {html_rel} |"
        )

    index_path = os.path.join(step_dir, "index.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")

    print(f"[TrajectoryDump] Wrote {len(episodes)} episodes to {step_dir}")
    return step_dir


def dump_teacher_examples(
    *,
    tokenizer,
    batch,
    dump_path: str,
    global_step: int,
    split: str = "train",
    max_examples: int = 2,
    write_markdown: bool = True,
) -> Optional[str]:
    """
    Dump a few teacher-forward examples for offline analysis.

    Writes:
      {dump_path}/step_{global_step}_{split}/teacher_examples.jsonl
      {dump_path}/step_{global_step}_{split}/md/teacher_XXX.md
      {dump_path}/step_{global_step}_{split}/index.md
    """
    if "teacher_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        print("[TeacherDump] skip: teacher_log_probs/old_log_probs missing")
        return None

    os.makedirs(dump_path, exist_ok=True)
    step_dir = os.path.join(dump_path, f"step_{global_step:06d}_{split}")
    os.makedirs(step_dir, exist_ok=True)
    md_dir = os.path.join(step_dir, "md")
    if write_markdown:
        os.makedirs(md_dir, exist_ok=True)

    n = len(batch.batch["responses"])
    nt = batch.non_tensor_batch
    response_mask = batch.batch.get("response_mask", None)
    teacher_lp = batch.batch["teacher_log_probs"]
    student_lp = batch.batch["old_log_probs"]
    responses = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    prompts = tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)

    # Prefer contrastive picks: one low-reward / one high-reward when possible.
    rewards = []
    for i in range(n):
        rewards.append(float(_safe_get(nt, "episode_rewards", i, 0.0) or 0.0))
    order = sorted(range(n), key=lambda i: (rewards[i], i))
    if max_examples is None or max_examples <= 0 or len(order) <= max_examples:
        selected_idx = order
    else:
        half = max(1, max_examples // 2)
        selected_idx = order[:half] + order[-half:]
        # dedup preserve order
        seen = set()
        uniq = []
        for i in selected_idx:
            if i in seen:
                continue
            seen.add(i)
            uniq.append(i)
        selected_idx = uniq[:max_examples]

    examples = []
    for rank, i in enumerate(selected_idx):
        if response_mask is not None:
            mask_i = response_mask[i].bool()
        else:
            mask_i = teacher_lp[i] != 0
        valid = int(mask_i.sum().item()) if hasattr(mask_i, "sum") else 0
        if valid == 0:
            teacher_mean = student_mean = gap_mean = 0.0
            token_rows = []
            token_ids = []
            tokens = []
            teacher_lps = []
            student_lps = []
            teacher_probs = []
            student_probs = []
            gaps = []
        else:
            t = teacher_lp[i][mask_i].detach().float().cpu()
            s = student_lp[i][mask_i].detach().float().cpu()
            g = t - s
            teacher_mean = float(t.mean().item())
            student_mean = float(s.mean().item())
            gap_mean = float(g.mean().item())
            # Full sampled-token alignments (response tokens only).
            token_ids = batch.batch["responses"][i][mask_i].detach().cpu().tolist()
            teacher_lps = [float(x) for x in t.tolist()]
            student_lps = [float(x) for x in s.tolist()]
            teacher_probs = [float(x) for x in torch.exp(t).tolist()]
            student_probs = [float(x) for x in torch.exp(s).tolist()]
            gaps = [float(x) for x in g.tolist()]
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
            token_rows = [
                {
                    "pos": pos,
                    "token_id": int(tid),
                    "token": tok,
                    "teacher_log_prob": tlp,
                    "student_log_prob": slp,
                    "teacher_prob": tp,
                    "student_prob": sp,
                    "gap_log_prob": gap,
                }
                for pos, (tid, tok, tlp, slp, tp, sp, gap) in enumerate(
                    zip(token_ids, tokens, teacher_lps, student_lps, teacher_probs, student_probs, gaps)
                )
            ]

        ex = {
            "global_step": global_step,
            "split": split,
            "sample_idx": int(i),
            "traj_uid": str(_safe_get(nt, "traj_uid", i, "")),
            "turn_step": _safe_get(nt, "turn_step", i, None),
            "episode_reward": rewards[i],
            "episode_length": _safe_get(nt, "episode_lengths", i, None),
            "traj_won": _safe_get(nt, "traj_won", i, None),
            "traj_task_score": _safe_get(nt, "traj_task_score", i, None),
            "data_source": _safe_get(nt, "data_source", i, None),
            "observation": _safe_get(nt, "anchor_obs", i, None),
            "prompt": prompts[i],
            "response": responses[i],
            "selected_skills": _safe_get(nt, "selected_skills", i, None),
            "matched_tasks": _safe_get(nt, "matched_tasks", i, None),
            "skill_all": _safe_get(nt, "skill_all", i, None),
            "skill_text": _safe_get(nt, "skill_text", i, None),
            "teacher_logprob_mean": teacher_mean,
            "student_logprob_mean": student_mean,
            "teacher_student_gap_mean": gap_mean,
            "num_response_tokens": valid,
            # Full sampled-token series for offline analysis
            "token_ids": token_ids,
            "tokens": tokens,
            "teacher_log_probs": teacher_lps,
            "student_log_probs": student_lps,
            "teacher_probs": teacher_probs,
            "student_probs": student_probs,
            "gap_log_probs": gaps,
            "tokens_detail": token_rows,
        }
        examples.append(ex)

        if write_markdown:
            lines = [
                f"# Teacher example {rank} @ step {global_step}",
                "",
                f"- sample_idx: `{i}`",
                f"- traj_uid: `{ex['traj_uid']}`",
                f"- turn_step: `{ex['turn_step']}`",
                f"- episode_reward: **{ex['episode_reward']}**",
                f"- traj_won / task_score: `{ex['traj_won']}` / `{ex['traj_task_score']}`",
                f"- selected_skills: `{ex['selected_skills']}`",
                f"- matched_tasks: `{ex['matched_tasks']}`",
                f"- num_response_tokens: `{valid}`",
                f"- teacher_lp_mean: `{teacher_mean:.4f}` | student_lp_mean: `{student_mean:.4f}` | gap: `{gap_mean:.4f}`",
                "",
                "## Selected skill text",
                "```text",
                _truncate_text(ex.get("skill_text") or "", 6000),
                "```",
                "",
                "## Observation",
                "```text",
                str(ex.get("observation") or ""),
                "```",
                "",
                "## Model response / action",
                "```text",
                str(ex.get("response") or ""),
                "```",
                "",
                "## Sampled tokens: student vs teacher probabilities",
                "",
                "| pos | token | teacher_logp | student_logp | teacher_p | student_p | gap_logp |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
            for row in token_rows:
                tok = str(row["token"]).replace("|", "\\|").replace("\n", "\\n")
                lines.append(
                    f"| {row['pos']} | `{tok}` | {row['teacher_log_prob']:.6f} | {row['student_log_prob']:.6f} | "
                    f"{row['teacher_prob']:.6g} | {row['student_prob']:.6g} | {row['gap_log_prob']:.6f} |"
                )
            lines.extend(
                [
                    "",
                    "<details><summary>Full prompt</summary>",
                    "",
                    "```text",
                    _truncate_text(ex.get("prompt") or "", 3500),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )
            md_path = os.path.join(md_dir, f"teacher_{rank:03d}_{ex['traj_uid'][:8] or 'sample'}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    jsonl_path = os.path.join(step_dir, "teacher_examples.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    index_lines = [
        f"# Teacher examples @ step {global_step} ({split})",
        "",
        f"- num_examples: {len(examples)}",
        f"- jsonl: `{jsonl_path}`",
        "",
        "| # | traj_uid | reward | skills | gap_mean | md |",
        "|---|---|---:|---|---:|---|",
    ]
    for rank, ex in enumerate(examples):
        md_rel = f"[md](md/teacher_{rank:03d}_{(ex.get('traj_uid') or 'sample')[:8]}.md)" if write_markdown else ""
        index_lines.append(
            f"| {rank} | `{ex.get('traj_uid')}` | {ex.get('episode_reward')} | `{ex.get('selected_skills')}` | "
            f"{ex.get('teacher_student_gap_mean'):.4f} | {md_rel} |"
        )
    with open(os.path.join(step_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")

    print(f"[TeacherDump] Wrote {len(examples)} teacher examples to {step_dir}")
    return step_dir
