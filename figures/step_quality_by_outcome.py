"""Step-level correct vs error ratios for WebShop and ALFWorld SFT labels.

Error steps are the unique indices listed in step_skills[].target_steps.
Every other step present in the trajectory is counted as correct.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

CORRECT_COLOR = PALETTE["blue_main"]
ERROR_COLOR = PALETTE["red_2"]
CORRECT_TEXT = "white"
ERROR_TEXT = "#272727"

WEBSHOP_PATH = Path("./all_sft_data_v1.jsonl")
ALFWORLD_PATH = Path("work/alfworld_sft_data/train.jsonl")
OUT_PATH = Path("./figures/step_quality_by_outcome")

TASK_ORDER = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2", "WebShop"]
STEP_RE = re.compile(r"^Step (\d+)\s*$", re.M)


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.0
    use_tex: bool = False
    font_family: tuple[str, ...] = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "font.size": style.font_size,
            "axes.labelsize": style.font_size,
            "axes.titlesize": style.font_size + 1,
            "xtick.labelsize": style.font_size - 1,
            "ytick.labelsize": style.font_size,
            "legend.fontsize": style.font_size - 1,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": style.use_tex,
        }
    )


def finalize_figure(fig, out_path, formats=("png", "pdf"), dpi=300, pad=0.35):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = out_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        saved.append(path)
    plt.close(fig)
    return saved


def classify_alfworld(task: str) -> str:
    text = task.strip().lower()
    if text.startswith("look ") or text.startswith("examine "):
        return "Look"
    if text.startswith("clean "):
        return "Clean"
    if text.startswith("heat "):
        return "Heat"
    if text.startswith("cool "):
        return "Cool"
    if text.startswith("find two ") or text.startswith("put two "):
        return "Pick2"
    if text.startswith("put "):
        return "Pick"
    raise ValueError(f"Unrecognized ALFWorld task: {task}")


def parse_steps(user_text: str) -> int:
    indices = [int(value) for value in STEP_RE.findall(user_text)]
    if not indices or indices != list(range(indices[-1] + 1)):
        raise ValueError("Trajectory steps are missing or not contiguous")
    return indices[-1] + 1


def error_indices(assistant_text: str, n_steps: int) -> set[int]:
    payload = json.loads(assistant_text)
    targets: set[int] = set()
    for skill in payload.get("step_skills") or []:
        for step in skill.get("target_steps") or []:
            step = int(step)
            if step < 0 or step >= n_steps:
                raise ValueError(f"target step {step} outside 0..{n_steps - 1}")
            targets.add(step)
    return targets


def accumulate(path: Path, environment: str) -> dict[tuple[str, str], dict[str, int]]:
    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"trajectories": 0, "correct": 0, "error": 0}
    )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            meta = row["metadata"]
            role = meta["role"]
            if role not in {"success", "failure"}:
                raise ValueError(f"Unexpected role: {role}")
            category = "WebShop" if environment == "webshop" else classify_alfworld(meta["task_key"])
            messages = row["messages"]
            n_steps = parse_steps(messages[1]["content"])
            n_error = len(error_indices(messages[2]["content"], n_steps))
            bucket = stats[(role, category)]
            bucket["trajectories"] += 1
            bucket["error"] += n_error
            bucket["correct"] += n_steps - n_error
    return stats


def percent_pair(correct: int, error: int) -> tuple[float, float]:
    total = correct + error
    correct_pct = 100.0 * correct / total
    error_pct = 100.0 * error / total
    shown_correct = round(correct_pct, 1)
    shown_error = round(100.0 - shown_correct, 1)
    return shown_correct, shown_error


def draw_bracket(ax, y0: float, y1: float, label: str) -> None:
    x = -13.5
    ax.plot(
        [x + 3.2, x, x, x + 3.2],
        [y0, y0, y1, y1],
        color="#272727",
        linewidth=1.1,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(
        x - 1.6,
        (y0 + y1) / 2,
        label,
        ha="right",
        va="center",
        rotation=90,
        fontsize=14,
        color="#272727",
        clip_on=False,
    )


def draw_panel(ax, stats, role: str, title: str, invert: bool) -> None:
    y = np.arange(len(TASK_ORDER))
    correct_widths = []
    error_widths = []
    trajectory_counts = []
    for category in TASK_ORDER:
        bucket = stats[(role, category)]
        correct_pct, error_pct = percent_pair(bucket["correct"], bucket["error"])
        correct_widths.append(correct_pct)
        error_widths.append(error_pct)
        trajectory_counts.append(bucket["trajectories"])

    correct_bars = ax.barh(
        y,
        correct_widths,
        color=CORRECT_COLOR,
        edgecolor="black",
        linewidth=1.0,
        height=0.72,
        zorder=3,
    )
    ax.barh(
        y,
        error_widths,
        left=correct_widths,
        color=ERROR_COLOR,
        edgecolor="black",
        linewidth=1.0,
        height=0.72,
        hatch="//",
        zorder=3,
    )

    for bar, width in zip(correct_bars, correct_widths):
        ax.text(
            width / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            ha="center",
            va="center",
            color=CORRECT_TEXT,
            fontsize=12,
            zorder=4,
        )
    for index, (left, width) in enumerate(zip(correct_widths, error_widths)):
        ax.text(
            left + width / 2,
            index,
            f"{width:.1f}",
            ha="center",
            va="center",
            color=ERROR_TEXT,
            fontsize=12,
            zorder=4,
        )
    for index, count in enumerate(trajectory_counts):
        ax.text(
            103.2,
            index,
            f"n={count}",
            ha="left",
            va="center",
            fontsize=12,
            color="#272727",
            clip_on=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(TASK_ORDER)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Step ratio (%)")
    ax.set_title(title, pad=10)
    if invert:
        ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle=(0, (2, 2)), linewidth=0.8, color="#D5D5D5", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=3)
    draw_bracket(ax, -0.38, 5.38, "ALFWorld")
    draw_bracket(ax, 5.62, 6.38, "WebShop")


def main() -> None:
    stats = accumulate(WEBSHOP_PATH, "webshop")
    alfworld = accumulate(ALFWORLD_PATH, "alfworld")
    for key, value in alfworld.items():
        stats[key]["trajectories"] += value["trajectories"]
        stats[key]["correct"] += value["correct"]
        stats[key]["error"] += value["error"]

    summary = {}
    for role in ("success", "failure"):
        for category in TASK_ORDER:
            bucket = stats[(role, category)]
            correct_pct, error_pct = percent_pair(bucket["correct"], bucket["error"])
            summary[f"{role}/{category}"] = {
                "trajectories": bucket["trajectories"],
                "correct_steps": bucket["correct"],
                "error_steps": bucket["error"],
                "correct_pct": correct_pct,
                "error_pct": error_pct,
            }
    summary_path = OUT_PATH.with_name("step_quality_by_outcome_stats.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    success_n = sum(stats[("success", category)]["trajectories"] for category in TASK_ORDER)
    failure_n = sum(stats[("failure", category)]["trajectories"] for category in TASK_ORDER)

    apply_publication_style(FigureStyle(font_size=15, axes_linewidth=1.6))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6), sharey=True)
    draw_panel(axes[0], stats, "success", f"(a)  Successful trajectories (n={success_n})", invert=True)
    draw_panel(axes[1], stats, "failure", f"(b)  Failed trajectories (n={failure_n})", invert=False)
    axes[1].tick_params(labelleft=False)

    handles = [
        Patch(facecolor=CORRECT_COLOR, edgecolor="black", linewidth=1.0, label="Correct steps"),
        Patch(facecolor=ERROR_COLOR, edgecolor="black", linewidth=1.0, hatch="//", label="Error steps"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.52, -0.02),
        handlelength=2.4,
        handleheight=1.1,
        columnspacing=1.6,
        frameon=True,
        fancybox=False,
        edgecolor="#272727",
    )
    fig.tight_layout(rect=(0.04, 0.08, 0.96, 1.0), w_pad=3.2)
    saved = finalize_figure(fig, OUT_PATH, formats=("png", "pdf"), dpi=300, pad=0.18)
    for path in saved:
        print(path)


GLM3_PATHS = (
    Path("/tmp/sft_glm3/train.jsonl"),
    Path("/tmp/sft_glm3/val.jsonl"),
    Path("/tmp/sft_glm3/test.jsonl"),
)
GLM3_OUT = Path("./figures/step_quality_webshop_glm3")


def accumulate_glm3(paths: tuple[Path, ...]) -> tuple[dict[str, dict[str, int]], int]:
    stats = {
        "success": {"trajectories": 0, "correct": 0, "error": 0},
        "failure": {"trajectories": 0, "correct": 0, "error": 0},
    }
    dropped = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                role = row["metadata"]["role"]
                if role not in stats:
                    raise ValueError(f"Unexpected role: {role}")
                try:
                    n_steps = parse_steps(row["messages"][1]["content"])
                    n_error = len(error_indices(row["answer_content"], n_steps))
                except (json.JSONDecodeError, ValueError):
                    dropped += 1
                    continue
                bucket = stats[role]
                bucket["trajectories"] += 1
                bucket["error"] += n_error
                bucket["correct"] += n_steps - n_error
    return stats, dropped


def plot_glm3_webshop() -> None:
    stats, dropped = accumulate_glm3(GLM3_PATHS)
    summary = {"dropped_unparseable": dropped}
    for role, label in (("success", "Successful"), ("failure", "Failed")):
        bucket = stats[role]
        correct_pct, error_pct = percent_pair(bucket["correct"], bucket["error"])
        summary[role] = {
            "trajectories": bucket["trajectories"],
            "correct_steps": bucket["correct"],
            "error_steps": bucket["error"],
            "correct_pct": correct_pct,
            "error_pct": error_pct,
            "label": label,
        }
    GLM3_OUT.with_name("step_quality_webshop_glm3_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    labels = [
        f"Successful trajectories (n={stats['success']['trajectories']})",
        f"Failed trajectories (n={stats['failure']['trajectories']})",
    ]
    correct_widths = []
    error_widths = []
    for role in ("success", "failure"):
        correct_pct, error_pct = percent_pair(stats[role]["correct"], stats[role]["error"])
        correct_widths.append(correct_pct)
        error_widths.append(error_pct)

    apply_publication_style(FigureStyle(font_size=15, axes_linewidth=1.6))
    fig, ax = plt.subplots(figsize=(10.4, 3.15))
    y = np.arange(len(labels))
    ax.barh(y, correct_widths, color=CORRECT_COLOR, edgecolor="black", linewidth=1.0, height=0.62, zorder=3)
    ax.barh(
        y,
        error_widths,
        left=correct_widths,
        color=ERROR_COLOR,
        edgecolor="black",
        linewidth=1.0,
        height=0.62,
        hatch="//",
        zorder=3,
    )
    for index, width in enumerate(correct_widths):
        ax.text(width / 2, index, f"{width:.1f}", ha="center", va="center", color=CORRECT_TEXT, fontsize=13, zorder=4)
    for index, (left, width) in enumerate(zip(correct_widths, error_widths)):
        ax.text(
            left + width / 2,
            index,
            f"{width:.1f}",
            ha="center",
            va="center",
            color=ERROR_TEXT,
            fontsize=13,
            zorder=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Step ratio (%)")
    ax.set_title("WebShop step quality, GLM-3 with thinking", pad=10)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linestyle=(0, (2, 2)), linewidth=0.8, color="#D5D5D5", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4)
    handles = [
        Patch(facecolor=CORRECT_COLOR, edgecolor="black", linewidth=1.0, label="Correct steps"),
        Patch(facecolor=ERROR_COLOR, edgecolor="black", linewidth=1.0, hatch="//", label="Error steps"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#272727",
        handlelength=2.4,
        handleheight=1.05,
        columnspacing=1.6,
    )
    fig.tight_layout()
    saved = finalize_figure(fig, GLM3_OUT, formats=("png", "pdf"), dpi=300, pad=0.16)
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
    plot_glm3_webshop()
