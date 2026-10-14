"""Turn analyzer chat JSONL into SFT parquet.

Each line has messages [system, user, assistant]. The assistant text is the
training target, including {"step_skills": []}. Samples whose full chat
template exceeds --max-length, or whose assistant text is at least
--max-response-tokens, are dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare analyzer SFT data")
    parser.add_argument("--jsonl", required=True, help="JSONL with a messages field")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/data/qirui/SDAR/models/Qwen2.5-3B-Instruct"),
        help="Tokenizer used to match MultiTurnSFTDataset length accounting.",
    )
    parser.add_argument("--out-dir", default="/data/qirui/SDAR/sft/analyzer_qwen2.5_3b")
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-length",
        type=int,
        default=12288,
        help="Max full conversation tokens after the chat template. Must match the trainer.",
    )
    parser.add_argument("--val-jsonl", default=None, help="Use this file as val instead of splitting --jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="Train on <think>...</think> plus the JSON answer. Off keeps only the JSON.",
    )
    return parser.parse_args()


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.S | re.I)


def _answer_text(content: str) -> str:
    return _THINK_BLOCK.sub("", content).strip()


def _with_thinking(answer: str, thinking: str) -> str:
    thinking = thinking.strip()
    if not thinking:
        return answer
    return f"<think>\n{thinking}\n</think>\n{answer}"


def _messages(row: dict, line_no: int, include_thinking: bool) -> tuple[list[dict], str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"line {line_no} needs messages [system, user, assistant]")
    cleaned = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"line {line_no} has an empty {message.get('role')} message")
        cleaned.append({"role": message["role"], "content": content})
    answer = row.get("answer_content")
    if not isinstance(answer, str) or not answer.strip():
        answer = _answer_text(cleaned[2]["content"])
    thinking = row.get("thinking_content") if include_thinking else ""
    if not isinstance(thinking, str):
        thinking = ""
    if include_thinking and not thinking.strip() and "<think>" in cleaned[2]["content"].lower():
        assistant = cleaned[2]["content"]
    else:
        assistant = _with_thinking(answer, thinking if include_thinking else "")
    if not assistant.strip():
        raise ValueError(f"line {line_no} has an empty assistant message")
    cleaned[2] = {"role": "assistant", "content": assistant}
    return cleaned, answer


def _is_empty_skills(assistant: str) -> bool:
    try:
        parsed = json.loads(assistant)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("step_skills") == []


def _valid_answer(answer: str) -> bool:
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and isinstance(parsed.get("step_skills"), list)


def _load_records(path: str, include_thinking: bool) -> tuple[list[tuple[list[dict], str]], int]:
    records = []
    dropped_invalid = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            messages, answer = _messages(json.loads(line), line_no, include_thinking)
            if not _valid_answer(answer):
                dropped_invalid += 1
                continue
            records.append((messages, answer))
    return records, dropped_invalid


def main():
    args = _parse_args()
    if args.val_jsonl is None and not 0 < args.val_ratio < 1:
        raise ValueError(f"--val-ratio must be in (0, 1), got {args.val_ratio}")
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    records, dropped_invalid = _load_records(args.jsonl, args.include_thinking)
    if args.val_jsonl is None:
        if len(records) < 2:
            raise ValueError(f"Need at least 2 valid examples, got {len(records)}")
        groups = {"empty": [], "skill": []}
        for index, (_messages_row, answer) in enumerate(records):
            key = "empty" if _is_empty_skills(answer) else "skill"
            groups[key].append(index)
        rng = np.random.RandomState(args.seed)
        val_index = set()
        for indexes in groups.values():
            if len(indexes) < 2:
                continue
            order = np.array(indexes, dtype=int)
            rng.shuffle(order)
            n_val = int(round(len(order) * args.val_ratio))
            n_val = min(max(n_val, 1), len(order) - 1)
            val_index.update(int(i) for i in order[:n_val])
        labeled = [(record, index in val_index) for index, record in enumerate(records)]
    else:
        val_records, dropped_val_invalid = _load_records(args.val_jsonl, args.include_thinking)
        dropped_invalid += dropped_val_invalid
        if not records or not val_records:
            raise ValueError("Train and val each need at least one valid example")
        labeled = [(record, False) for record in records] + [(record, True) for record in val_records]
        records = records + val_records

    rows = {"train": [], "val": []}
    dropped_response = 0
    dropped_total = 0
    kept_empty = {"train": 0, "val": 0}
    for (messages, answer), is_val in labeled:
        assistant_ids = tokenizer(messages[2]["content"], add_special_tokens=False)["input_ids"]
        if len(assistant_ids) >= args.max_response_tokens:
            dropped_response += 1
            continue
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if len(full_ids) > args.max_length:
            dropped_total += 1
            continue
        split = "val" if is_val else "train"
        rows[split].append({"messages": messages})
        if _is_empty_skills(answer):
            kept_empty[split] += 1

    for split, split_rows in rows.items():
        if not split_rows:
            raise RuntimeError(f"{split} split is empty after length filtering")
        frame = pd.DataFrame.from_records(split_rows)
        path = os.path.join(args.out_dir, f"{split}.parquet")
        frame.to_parquet(path, index=False)
        print(f"{split}: {len(frame)} examples ({kept_empty[split]} empty) -> {path}", flush=True)

    summary = {
        "jsonl": args.jsonl,
        "model_path": args.model_path,
        "max_response_tokens": args.max_response_tokens,
        "max_length": args.max_length,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "include_thinking": args.include_thinking,
        "n_input": len(records),
        "n_train": len(rows["train"]),
        "n_val": len(rows["val"]),
        "n_empty_train": kept_empty["train"],
        "n_empty_val": kept_empty["val"],
        "n_dropped_invalid": dropped_invalid,
        "n_dropped_response": dropped_response,
        "n_dropped_total": dropped_total,
    }
    with open(os.path.join(args.out_dir, "prepare_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
