"""Turn ALFWorld analyzer chat JSONL into SFT parquet.

Each line has messages [system, user, assistant] plus metadata, the same
schema as all_sft_data_v1.jsonl / the WebShop analyzer SFT data. Defaults
point at the unzipped alfworld_sft_data; the filtering and split logic is
shared with examples/sft/analyzer/prepare_analyzer_sft.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_DEFAULT_JSONL = "work/alfworld_sft_data/train.jsonl"
_DEFAULT_OUT = "work/sft/alfworld_analyzer_qwen2.5_3b"
_DEFAULT_MODEL = "work/models/Qwen2.5-3B-Instruct"


def _load_analyzer_prepare():
    path = Path(__file__).resolve().parents[1] / "analyzer" / "prepare_analyzer_sft.py"
    spec = importlib.util.spec_from_file_location("prepare_analyzer_sft", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    prep = _load_analyzer_prepare()
    argv = list(sys.argv[1:])
    flags = {token for token in argv if token.startswith("--")}
    if "--jsonl" not in flags:
        argv = ["--jsonl", _DEFAULT_JSONL, *argv]
    if "--out-dir" not in flags:
        argv = ["--out-dir", _DEFAULT_OUT, *argv]
    if "--model-path" not in flags:
        argv = ["--model-path", _DEFAULT_MODEL, *argv]
    sys.argv = [sys.argv[0], *argv]
    prep.main()


if __name__ == "__main__":
    main()
