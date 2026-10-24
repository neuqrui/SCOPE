"""Score an analyzer model against ALFWorld SFT step labels.

Each SFT line is messages [system, user, assistant]. The script feeds system
and user to the model, parses step_skills[].target_steps from both the
generation and the assistant label, and compares those error-step sets.

Skill text and how steps are grouped into skills are not scored. The same
error steps can be split differently and still count as the same prediction.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


_STEP_RE = re.compile(r"^Step (\d+)\s*$", re.M)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.S | re.I)


def _parse_args():
    parser = argparse.ArgumentParser(description="Compare predicted error steps with ALFWorld SFT labels")
    parser.add_argument("--jsonl", default="work/alfworld_sft_data/train.jsonl")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec name=path. Repeat for several models. An FSDP actor directory is merged in memory.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Number of trajectories. 0 means the whole file.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default="work/tmp/alfworld_step_overlap.jsonl")
    args = parser.parse_args()
    if not args.model:
        args.model = ["sft=work/ckpt/sft/alfworld_analyzer_qwen2.5_3b/global_step_261"]
    return args


def _model_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).name
        specs.append((name, Path(path)))
    names = [name for name, _path in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Model names must be unique, got {names}")
    return specs


def _fsdp_shards(model_dir: Path) -> list[Path]:
    shards = list(model_dir.glob("model_world_size_*_rank_*.pt"))
    return sorted(shards, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def _merge_fsdp_state(model_dir: Path) -> dict[str, torch.Tensor]:
    shards = _fsdp_shards(model_dir)
    if not shards:
        raise FileNotFoundError(f"No FSDP model shards in {model_dir}")
    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in shards]
    merged = {}
    for key in list(loaded[0].keys()):
        pieces = []
        placement = None
        full_shape = None
        for shard in loaded:
            tensor = shard.pop(key)
            if isinstance(tensor, DTensor):
                placement = tensor.placements[0]
                full_shape = tuple(tensor.shape)
                pieces.append(tensor._local_tensor.detach())
            else:
                pieces.append(tensor.detach())
        if placement is not None and type(placement).__name__ == "Shard":
            value = torch.cat(pieces, dim=placement.dim)
            if full_shape is not None and tuple(value.shape) != full_shape:
                raise RuntimeError(f"{key} merged to {tuple(value.shape)}, expected {full_shape}")
        else:
            value = pieces[0]
        merged[key] = value.to(torch.bfloat16).contiguous()
        del pieces
    del loaded
    gc.collect()
    return merged


def _load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if _fsdp_shards(model_dir):
        state = _merge_fsdp_state(model_dir)
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        del state
        gc.collect()
        bad_missing = [key for key in missing if key.endswith("weight") or key.endswith("bias")]
        if unexpected or bad_missing:
            raise RuntimeError(f"FSDP load mismatch missing={bad_missing} unexpected={unexpected}")
        model = model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
    model.eval()
    return tokenizer, model


def _answer_text(content: str) -> str:
    return _THINK_BLOCK.sub("", content).strip()


def _load_json_object(text: str) -> tuple[dict | None, str | None]:
    cleaned = _answer_text(text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None, "no json object"
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "json is not an object"
    return parsed, None


def _trajectory_length(user_text: str) -> int | None:
    indices = [int(value) for value in _STEP_RE.findall(user_text)]
    if not indices or indices != list(range(indices[-1] + 1)):
        return None
    return indices[-1] + 1


def _error_steps(text: str) -> tuple[set[int] | None, str | None]:
    parsed, error = _load_json_object(text)
    if parsed is None:
        return None, error
    skills = parsed.get("step_skills")
    if not isinstance(skills, list):
        return None, "step_skills is not a list"
    steps: set[int] = set()
    for item in skills:
        if not isinstance(item, dict):
            return None, "step skill is not an object"
        targets = item.get("target_steps")
        if not isinstance(targets, list):
            return None, "target_steps is not a list"
        for value in targets:
            if isinstance(value, bool) or not isinstance(value, int):
                return None, f"target step is not an int: {value!r}"
            steps.add(value)
    return steps, None


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _set_scores(gold: set[int], pred: set[int]) -> dict[str, float]:
    inter = len(gold & pred)
    union = len(gold | pred)
    precision = _safe_div(inter, len(pred))
    recall = _safe_div(inter, len(gold))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "jaccard": 1.0 if union == 0 else inter / union,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": float(gold == pred),
        "intersection": inter,
        "n_gold": len(gold),
        "n_pred": len(pred),
    }


def _load_rows(path: str, limit: int, seed: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                row = json.loads(line)
                row["_line"] = line_no
                rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _summarize(records: list[dict]) -> dict:
    scored = [row for row in records if row["scores"] is not None]
    inter = sum(row["scores"]["intersection"] for row in scored)
    n_gold = sum(row["scores"]["n_gold"] for row in scored)
    n_pred = sum(row["scores"]["n_pred"] for row in scored)
    micro_p = _safe_div(inter, n_pred)
    micro_r = _safe_div(inter, n_gold)
    micro_f1 = 0.0 if micro_p + micro_r == 0 else 2 * micro_p * micro_r / (micro_p + micro_r)
    return {
        "n": len(records),
        "parse_ok": len(scored),
        "parse_rate": len(scored) / len(records) if records else 0.0,
        "exact_match_rate": _mean([row["scores"]["exact"] for row in scored]),
        "macro_jaccard": _mean([row["scores"]["jaccard"] for row in scored]),
        "macro_precision": _mean([row["scores"]["precision"] for row in scored]),
        "macro_recall": _mean([row["scores"]["recall"] for row in scored]),
        "macro_f1": _mean([row["scores"]["f1"] for row in scored]),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "mean_gold_steps": _mean([row["scores"]["n_gold"] for row in scored]),
        "mean_pred_steps": _mean([row["scores"]["n_pred"] for row in scored]),
        "out_of_range_steps": sum(row["n_out_of_range"] for row in records),
    }


def _score_prediction(row: dict, prediction: str, truncated: bool) -> dict:
    n_steps = _trajectory_length(row["messages"][1]["content"])
    gold, gold_error = _error_steps(row["messages"][2]["content"])
    pred, pred_error = _error_steps(prediction)
    scores = None
    n_out_of_range = 0
    pred_in_range = None
    if gold is not None and pred is not None and n_steps is not None:
        pred_in_range = {step for step in pred if 0 <= step < n_steps}
        n_out_of_range = len(pred - pred_in_range)
        scores = _set_scores(gold, pred_in_range)
    return {
        "line": row["_line"],
        "task": (row.get("metadata") or {}).get("task_key"),
        "role": (row.get("metadata") or {}).get("role"),
        "n_steps": n_steps,
        "gold_steps": None if gold is None else sorted(gold),
        "pred_steps": None if pred_in_range is None else sorted(pred_in_range),
        "gold_error": gold_error,
        "pred_error": pred_error,
        "truncated": truncated,
        "n_out_of_range": n_out_of_range,
        "scores": scores,
        "prediction": prediction,
    }


def _generate(tokenizer, model, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[tuple[str, bool]]:
    results = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        for row_index in range(output.shape[0]):
            generated_ids = output[row_index, prompt_width:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append((text, int(generated_ids.shape[0]) >= max_new_tokens))
        print(f"generated {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return results


def _write_records(path: Path, records: list[dict], summary: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    args = _parse_args()
    specs = _model_specs(args.model)
    rows = _load_rows(args.jsonl, args.limit, args.seed)
    if not rows:
        raise SystemExit(f"No rows in {args.jsonl}")

    out_path = Path(args.out)
    multiple = len(specs) > 1
    summaries = {}
    for name, model_dir in specs:
        print(f"loading {name} from {model_dir}", flush=True)
        tokenizer, model = _load_model(model_dir)
        prompts = [
            tokenizer.apply_chat_template(row["messages"][:2], add_generation_prompt=True, tokenize=False)
            for row in rows
        ]
        generations = _generate(tokenizer, model, prompts, args.max_new_tokens, args.batch_size)
        records = [
            _score_prediction(row, prediction, truncated)
            for row, (prediction, truncated) in zip(rows, generations)
        ]
        summary = _summarize(records)
        summary["model"] = str(model_dir)
        summaries[name] = summary
        target = out_path.parent / out_path.stem / f"{name}.jsonl" if multiple else out_path
        _write_records(target, records, summary)
        print(json.dumps({name: summary}, indent=2), flush=True)
        print(f"wrote {target}", flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    if multiple:
        compare_path = out_path.parent / out_path.stem / "compare.json"
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"wrote {compare_path}", flush=True)


if __name__ == "__main__":
    main()
