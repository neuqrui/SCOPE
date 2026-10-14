"""Write one debug episode to JSON and a readable HTML page."""

from __future__ import annotations

import html
import json
import os


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _num(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return _esc(value)


def _series_value(objective: dict | None, span: str) -> float | None:
    if not objective:
        return None
    stats = objective.get(span) or {}
    value = stats.get("teacher_gap")
    if value is None:
        return None
    return float(value)


def _trend_chart(episode: dict) -> str:
    steps = episode.get("steps") or []
    if not steps:
        return ""
    names = ("response", "think", "action")
    colors = {"response": "#0969da", "think": "#9a6700", "action": "#bf3989"}
    points = {name: [] for name in names}
    for index, step in enumerate(steps):
        objective = step.get("objective") or {}
        x = index
        for name in names:
            value = _series_value(objective, name)
            if value is not None:
                points[name].append((x, value))
    if not any(points.values()):
        return (
            "<div class='turn'><h2>KL 趋势</h2>"
            "<div>这一条轨迹还没有算出 KL。</div></div>"
        )

    width, height = 980, 280
    left, right, top, bottom = 64, 16, 16, 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [index for index in range(len(steps))]
    ys = [y for series in points.values() for _, y in series]
    y_min = min(0.0, min(ys))
    y_max = max(0.0, max(ys))
    if y_max - y_min < 1e-8:
        y_min -= 1.0
        y_max += 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def sx(index: float) -> float:
        if len(steps) == 1:
            return left + plot_w / 2
        return left + plot_w * index / (len(steps) - 1)

    def sy(value: float) -> float:
        return top + plot_h * (1 - (value - y_min) / (y_max - y_min))

    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='distillation trend'>",
        f"<line x1='{left}' y1='{sy(0)}' x2='{width - right}' y2='{sy(0)}' stroke='#d0d7de'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='#d0d7de'/>",
    ]
    for tick in (y_min + pad, 0.0, y_max - pad):
        parts.append(
            f"<text x='{left - 8}' y='{sy(tick) + 4}' text-anchor='end' font-size='11' fill='#656d76'>"
            f"{tick:.3g}</text>"
        )
    for index in xs:
        label = steps[index].get("turn", index)
        parts.append(
            f"<text x='{sx(index)}' y='{height - 12}' text-anchor='middle' font-size='11' fill='#656d76'>"
            f"{_esc(label)}</text>"
        )
    for name in names:
        series = points[name]
        run = []
        for x, y in series:
            if run and x != run[-1][0] + 1:
                poly = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in run)
                parts.append(f"<polyline fill='none' stroke='{colors[name]}' stroke-width='2' points='{poly}'/>")
                run = []
            run.append((x, y))
        if len(run) >= 2:
            poly = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in run)
            parts.append(f"<polyline fill='none' stroke='{colors[name]}' stroke-width='2' points='{poly}'/>")
        for x, y in series:
            parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3.5' fill='{colors[name]}'/>")
    parts.append("</svg>")
    legend = "".join(
        f"<span><i class='swatch' style='background:{colors[name]}'></i>{name}</span>"
        for name in names
    )
    return (
        "<div class='turn'><h2>KL 趋势</h2>"
        "<div>纵轴是 KL = mean(log π_T − log π_S)，没有乘负号，也没有乘 log π_S。"
        "正值表示 teacher 给这个 token 的概率高于学生。"
        "response 覆盖整段学生回复，think / action 只覆盖对应标签内的 token。"
        "未蒸馏的 step 记为 0。</div>"
        + "".join(parts)
        + f"<div class='legend'>{legend}</div></div>"
    )


def _append_token_run(parts, run, color, sx, sy) -> None:
    if len(run) >= 2:
        poly = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in run)
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='1.5' points='{poly}'/>")
        return
    x, y = run[0]
    parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='2.5' fill='{color}'/>")


def _token_chart(episode: dict) -> str:
    """One chart of every distilled reply. X is the token index within that reply's concatenation."""
    colors = {"think": "#9a6700", "action": "#bf3989", "other": "#0969da"}
    values = []
    spans = []
    boundaries = []
    cursor = 0
    for step in episode.get("steps") or []:
        objective = step.get("objective") or {}
        token_kl = objective.get("token_kl") or []
        token_span = objective.get("token_span") or []
        if not token_kl:
            continue
        boundaries.append((cursor, step.get("turn", cursor)))
        for index, value in enumerate(token_kl):
            values.append(float(value))
            spans.append(token_span[index] if index < len(token_span) else "other")
        cursor += len(token_kl)
    if not values:
        return (
            "<div class='turn'><h2>逐 token KL</h2>"
            "<div>没有进入蒸馏的回复，所以没有 token 级 KL。</div></div>"
        )

    width = max(980, 72 + len(values) * 3 + 16)
    height = 280
    left, right, top, bottom = 64, 16, 16, 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min = min(0.0, min(values))
    y_max = max(0.0, max(values))
    if y_max - y_min < 1e-8:
        y_min -= 1.0
        y_max += 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def sx(index: float) -> float:
        if len(values) == 1:
            return left + plot_w / 2
        return left + plot_w * index / (len(values) - 1)

    def sy(value: float) -> float:
        return top + plot_h * (1 - (value - y_min) / (y_max - y_min))

    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' role='img' aria-label='token distillation'>",
        f"<line x1='{left}' y1='{sy(0)}' x2='{width - right}' y2='{sy(0)}' stroke='#d0d7de'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='#d0d7de'/>",
    ]
    for tick in (y_min + pad, 0.0, y_max - pad):
        parts.append(
            f"<text x='{left - 8}' y='{sy(tick) + 4}' text-anchor='end' font-size='11' fill='#656d76'>"
            f"{tick:.3g}</text>"
        )
    for start, turn in boundaries:
        x = sx(start)
        parts.append(
            f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{height - bottom}' stroke='#d0d7de' stroke-dasharray='3 3'/>"
        )
        parts.append(
            f"<text x='{x + 4:.1f}' y='{height - 12}' font-size='11' fill='#656d76'>s{_esc(turn)}:{start}</text>"
        )
    run_span = spans[0]
    run = [(0, values[0])]
    for index in range(1, len(values)):
        if spans[index] != run_span:
            _append_token_run(parts, run, colors[run_span], sx, sy)
            run_span = spans[index]
            run = [(index, values[index])]
        else:
            run.append((index, values[index]))
    if run:
        _append_token_run(parts, run, colors[run_span], sx, sy)
    parts.append("</svg>")
    legend = "".join(
        f"<span><i class='swatch' style='background:{colors[name]}'></i>{name}</span>"
        for name in ("think", "action", "other")
    )
    return (
        "<div class='turn'><h2>逐 token KL</h2>"
        "<div>横坐标是学生回复 token 的序号，多步回复按顺序接在一起。"
        "竖线是每个 step 的起点，标注为 s步号:全局token序号。"
        "纵轴是该 token 的 log π_T − log π_S。"
        "颜色表示 token 落在 think、action，还是标签外。"
        "未蒸馏的 step 不出现在这张图里。</div>"
        "<div style='overflow-x:auto'>"
        + "".join(parts)
        + f"</div><div class='legend'>{legend}</div></div>"
    )


def _objective_table(objective: dict | None) -> str:
    if not objective:
        return ""
    if objective.get("distilled") is False:
        return (
            "<div><b>本步蒸馏目标: 0</b>（response / think / action 均未进入 OPSD）</div>"
        )
    if objective.get("error") and "think" not in objective:
        return f"<div class='bad'>objective error: {_esc(objective.get('error'))}</div>"
    rows = []
    for name in ("think", "action", "response"):
        stats = objective.get(name) or {}
        rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{_esc(stats.get('n_tokens'))}</td>"
            f"<td>{_num(stats.get('student_entropy'))}</td>"
            f"<td>{_num(stats.get('teacher_entropy'))}</td>"
            f"<td>{_num(stats.get('teacher_gap'))}</td>"
            f"<td>{_num(stats.get('student_logprob'))}</td>"
            f"<td>{_num(stats.get('teacher_logprob'))}</td>"
            f"<td>{_num(stats.get('opsd_loss'))}</td>"
            f"<td>{_num(stats.get('opsd_weighted_loss'))}</td>"
            "</tr>"
        )
    own = objective.get("teacher_own") or {}
    own_line = ""
    if own.get("error"):
        own_line = f"<div>teacher own entropy: {_esc(own.get('error'))}</div>"
    else:
        think = own.get("think") or {}
        action = own.get("action") or {}
        own_line = (
            "<div>teacher's own reply entropy: "
            f"think={_num(think.get('entropy'))} (n={_esc(think.get('n_tokens'))}), "
            f"action={_num(action.get('entropy'))} (n={_esc(action.get('n_tokens'))})</div>"
        )
    total = objective.get("total") or {}
    think = objective.get("think") or {}
    action = objective.get("action") or {}
    total_line = (
        "<div><b>本步蒸馏目标</b> response "
        f"opsd_weighted_loss=<b>{_num(total.get('opsd_weighted_loss'))}</b> "
        f"(token-mean) | think "
        f"<b>{_num(think.get('opsd_weighted_loss'))}</b> | action "
        f"<b>{_num(action.get('opsd_weighted_loss'))}</b></div>"
        "<div>token-sum: "
        f"response={_num(total.get('opsd_weighted_loss_sum'))}, "
        f"think={_num(think.get('opsd_weighted_loss_sum'))}, "
        f"action={_num(action.get('opsd_weighted_loss_sum'))}. "
        "加权值已乘 opsd_coef。</div>"
    )
    return (
        "<h3>Student reply objective</h3>"
        + total_line
        + "<div>Entropy and sampled OPSD are means over tokens inside the tag. "
        "Teacher columns score the student's tokens under the skill prompt. "
        "teacher_gap = A = log π_T − log π_S. "
        "opsd_loss = mean(−A · log π_S). "
        "opsd_weighted_loss = opsd_coef · opsd_loss.</div>"
        "<table><thead><tr>"
        "<th>span</th><th>n</th><th>student entropy</th><th>teacher entropy</th>"
        "<th>teacher_gap</th><th>student logprob</th><th>teacher logprob</th>"
        "<th>opsd_loss</th><th>opsd_weighted_loss</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + own_line
    )


def write_episode(episode: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = f"episode_{int(episode.get('episode_idx', 0)):03d}"
    json_path = os.path.join(out_dir, stem + ".json")
    html_path = os.path.join(out_dir, stem + ".html")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(episode, f, ensure_ascii=False, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(episode))
    return json_path, html_path


def render_html(episode: dict) -> str:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>debug episode { _esc(episode.get('episode_idx')) }</title>",
        "<style>",
        "body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.45;color:#1f2328;}",
        ".meta,.task{border-radius:8px;padding:12px 14px;margin:12px 0;}",
        ".meta{background:#f6f8fa;border:1px solid #d0d7de;}",
        ".task{background:#fff8c5;border:1px solid #d4a72c;}",
        ".turn{border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin:16px 0;}",
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;}",
        ".student{border-left:4px solid #0969da;padding-left:10px;}",
        ".teacher{border-left:4px solid #bf3989;padding-left:10px;}",
        "pre{white-space:pre-wrap;background:#0d1117;color:#e6edf3;padding:12px;border-radius:8px;overflow:auto;}",
        "details{margin-top:8px} summary{cursor:pointer;color:#0969da}",
        ".ok{color:#1a7f37}.bad{color:#cf222e}",
        ".legend{display:flex;gap:16px;margin:8px 0 0;font-size:14px}",
        ".swatch{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle}",
        "</style></head><body>",
        f"<h1>Episode { _esc(episode.get('episode_idx')) }</h1>",
        "<div class='meta'>",
        f"<div>teacher_mode: <b>{_esc(episode.get('teacher_mode'))}</b></div>",
        f"<div>objective: <b>{_esc(episode.get('objective_name'))}</b> | "
        f"logit_temperature: <b>{_esc(episode.get('logit_temperature'))}</b> | "
        f"opsd_coef: <b>{_esc(episode.get('opsd_coef'))}</b></div>"
        f"<div>history_length: <b>{_esc(episode.get('history_length'))}</b> "
        "(student prompt only). Analyzer, when used, sees every raw step observation.</div>",
        f"<div>reward: <b>{_esc(episode.get('episode_reward'))}</b> | "
        f"length: <b>{_esc(episode.get('episode_length'))}</b> | "
        f"won: <b>{_esc(episode.get('won'))}</b> | "
        f"task_score: <b>{_esc(episode.get('task_score'))}</b></div>",
        "</div>",
        "<div class='task'><h2>Task</h2><pre>" + _esc(episode.get("task")) + "</pre></div>",
        _trend_chart(episode),
        _token_chart(episode),
    ]
    if episode.get("episode_skill") or episode.get("analysis_prompt"):
        parts.append("<div class='turn'><h2>SEED analysis</h2>")
        parts.append(f"<div>summary: {_esc(episode.get('episode_summary'))}</div>")
        parts.append("<h3>episode_skill</h3><pre>" + _esc(episode.get("episode_skill")) + "</pre>")
        if episode.get("analysis_parse_error"):
            parts.append(f"<div class='bad'>parse_error: {_esc(episode.get('analysis_parse_error'))}</div>")
        parts.append("<details><summary>Analysis prompt (full trajectory)</summary><pre>")
        parts.append(_esc(episode.get("analysis_prompt")))
        parts.append("</pre></details>")
        parts.append("<details><summary>Analyzer raw output</summary><pre>")
        parts.append(_esc(episode.get("analysis_raw")))
        parts.append("</pre></details></div>")

    if episode.get("candidate_skills") is not None or episode.get("judge_prompt"):
        parts.append("<div class='turn'><h2>Skill judge</h2>")
        parts.append(f"<div>candidate_skills: <code>{_esc(episode.get('candidate_skills'))}</code></div>")
        if episode.get("domain_skill") is not None or episode.get("domain_keywords") is not None:
            parts.append(
                f"<div>domain_skill: <b>{_esc(episode.get('domain_skill'))}</b> | "
                f"domain_keywords: <code>{_esc(episode.get('domain_keywords'))}</code></div>"
            )
        if episode.get("judge_parse_error"):
            parts.append(f"<div class='bad'>parse_error: {_esc(episode.get('judge_parse_error'))}</div>")
        parts.append("<details><summary>Judge prompt</summary><pre>")
        parts.append(_esc(episode.get("judge_prompt")))
        parts.append("</pre></details>")
        parts.append("<details><summary>Judge raw output</summary><pre>")
        parts.append(_esc(episode.get("judge_raw")))
        parts.append("</pre></details></div>")

    for step in episode.get("steps", []):
        valid = bool(step.get("is_action_valid"))
        same = step.get("student_action") == step.get("teacher_action")
        parts.append(f"<div class='turn'><h2>Turn {_esc(step.get('turn'))}</h2>")
        parts.append(
            f"<div>reward=<code>{_esc(step.get('reward'))}</code> | "
            f"valid=<span class='{'ok' if valid else 'bad'}'>{_esc(valid)}</span> | "
            f"won=<code>{_esc(step.get('won'))}</code> | "
            f"task_score=<code>{_esc(step.get('task_score'))}</code> | "
            f"actions_match=<b>{_esc(same)}</b></div>"
        )
        parts.append(
            f"<div>student action: <code>{_esc(step.get('student_action'))}</code><br>"
            f"teacher action: <code>{_esc(step.get('teacher_action'))}</code></div>"
        )
        if step.get("page_type") is not None:
            parts.append(
                f"<div>page: <b>{_esc(step.get('page_type'))}</b> → {_esc(step.get('page_route'))} "
                f"rules=<code>{_esc(step.get('page_rules'))}</code> | "
                f"domain_skill: <b>{_esc(step.get('domain_skill'))}</b> "
                f"keywords=<code>{_esc(step.get('domain_keywords'))}</code></div>"
            )
        if step.get("distill") is not None or step.get("keyword_skills") is not None:
            parts.append(
                f"<div>distill: <b>{_esc(step.get('distill'))}</b> | "
                f"keyword_skills: <code>{_esc(step.get('keyword_skills'))}</code> | "
                f"violation_skills: <code>{_esc(step.get('selected_skills'))}</code></div>"
            )
            if step.get("violation_reason"):
                parts.append(f"<div>violation_reason: {_esc(step.get('violation_reason'))}</div>")
        parts.append(_objective_table(step.get("objective")))
        if step.get("selected_skills") is not None:
            parts.append(
                f"<div>selected_skills: <code>{_esc(step.get('selected_skills'))}</code> | "
                f"matched_tasks: <code>{_esc(step.get('matched_tasks'))}</code></div>"
            )
        parts.append("<h3>Raw observation (anchor, no history window)</h3><pre>")
        parts.append(_esc(step.get("anchor_obs")))
        parts.append("</pre>")
        parts.append("<div class='pair'>")
        parts.append("<div class='student'><h3>Student output</h3><pre>")
        parts.append(_esc(step.get("student_response")))
        parts.append("</pre></div>")
        parts.append("<div class='teacher'><h3>Teacher output</h3><pre>")
        parts.append(_esc(step.get("teacher_response")))
        parts.append("</pre></div></div>")
        parts.append("<details><summary>Student prompt</summary><pre>")
        parts.append(_esc(step.get("student_prompt")))
        parts.append("</pre></details>")
        parts.append("<details><summary>Teacher prompt</summary><pre>")
        parts.append(_esc(step.get("teacher_prompt")))
        parts.append("</pre></details>")
        if step.get("skill_text"):
            parts.append("<details><summary>Skill text</summary><pre>")
            parts.append(_esc(step.get("skill_text")))
            parts.append("</pre></details>")
        parts.append("</div>")
    parts.append("</body></html>")
    return "".join(parts)
