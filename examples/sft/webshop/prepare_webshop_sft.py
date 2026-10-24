"""Turn WebShop rejection-sampling trajectories into single-turn SFT parquet.

Each step is one example. The stored prompt already contains the history window
used at rollout, and the response is the completion sampled for that step.
Train and val are split by trajectory so steps from the same episode stay together.
Samples longer than ``--max-length`` after the SFT chat template are dropped.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare single-turn WebShop SFT data")
    parser.add_argument(
        "--success-path",
        default="work/reject_sample/webshop_grpo_3b_step150/success.jsonl",
    )
    parser.add_argument(
        "--model-path",
        default="work/ckpt/hf/grpo_qwen2.5_3b_step150",
        help="Tokenizer used to match verl SFTDataset length accounting.",
    )
    parser.add_argument("--out-dir", default="work/sft/webshop_grpo_3b_step150")
    parser.add_argument("--max-length", type=int, default=4608)
    parser.add_argument("--val-trajectories", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sequence_length(tokenizer, prompt: str, response: str) -> int:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    response_text = response + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    return len(prompt_ids) + len(response_ids)


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    trajectories = []
    with open(args.success_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    if len(trajectories) <= args.val_trajectories:
        raise ValueError(
            f"Need more than {args.val_trajectories} trajectories to hold out a val split, got {len(trajectories)}"
        )

    order = np.arange(len(trajectories))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(order)
    val_index = set(int(i) for i in order[: args.val_trajectories])

    rows = {"train": [], "val": []}
    dropped = 0
    for index, traj in enumerate(trajectories):
        split = "val" if index in val_index else "train"
        for step in traj["steps"]:
            prompt = step.get("prompt") or ""
            response = step.get("response") or ""
            if not prompt.strip() or not response.strip() or step.get("prompt_truncated"):
                dropped += 1
                continue
            if _sequence_length(tokenizer, prompt, response) > args.max_length:
                dropped += 1
                continue
            rows[split].append(
                {
                    "prompt": prompt,
                    "response": response,
                    "task_id": traj["task_id"],
                    "env_seed": int(traj["env_seed"]),
                    "goal_idx": int(traj["goal_idx"]),
                    "turn": int(step["turn"]),
                }
            )

    for split, records in rows.items():
        if not records:
            raise RuntimeError(f"{split} split is empty")
        frame = pd.DataFrame.from_records(records)
        path = os.path.join(args.out_dir, f"{split}.parquet")
        frame.to_parquet(path, index=False)
        print(f"{split}: {len(frame)} steps, {frame['task_id'].nunique()} trajectories -> {path}", flush=True)

    summary = {
        "success_path": args.success_path,
        "model_path": args.model_path,
        "max_length": args.max_length,
        "val_trajectories": args.val_trajectories,
        "seed": args.seed,
        "n_trajectories": len(trajectories),
        "n_train": len(rows["train"]),
        "n_val": len(rows["val"]),
        "n_dropped": dropped,
    }
    with open(os.path.join(args.out_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
