# SCOPE: Selective Critical-Step On-Policy Self-Distillation

SCOPE trains a long-horizon LLM agent with outcome-based GRPO plus corrective self-distillation on selected error steps. After a supervised warm-up, the same policy analyzes its own rollouts, writes a step-specific skill for each identified error, and distills that skill only at the selected steps. A bounded signed teacher–student coefficient keeps the auxiliary update stable. A success-relative pairwise length reward favors shorter successful trajectories.

At evaluation the agent uses the ordinary interaction history. It does not call the analyzer and does not receive hindsight skills.

This repository builds on the [SDAR](https://github.com/ZJU-REAL/SDAR) / verl-agent training stack. SCOPE's launchers are under `scope/`. The paper evaluates ALFWorld and WebShop.

## Method settings

These match the full SCOPE configuration.

| Symbol | Config | Value |
|---|---|---|
| Maximum skill entries \(K_{\max}\) | analyzer schema | 5 |
| Signed-gap slope \(\alpha\) | `OPSD_GATE_BETA` | 2.5 |
| Saturation bound \(c\) | `OPSD_ADV_CLIP` | 1 |
| Distillation weight \(\lambda\) | `algorithm.opsd_kl.opsd_coef` | 0.01 |
| Length temperature \(\tau_{\mathrm{len}}\) | `success_length_pairwise_tau` | 0.1 |
| Length bonus \(\lambda_{\mathrm{len}}\) | `success_length_pairwise_coef` | 5 |
| PPO clip \(\epsilon\) | `actor.clip_ratio` | 0.2 |
| Reference KL \(\beta_{\mathrm{ref}}\) | `actor.kl_loss_coef` | 0.01 |
| Entropy weight \(\lambda_H\) | `actor.entropy_coeff` | 0.001 |

The distillation loss on a selected token is

\[
-\,c\,\tanh\Big(\frac{\alpha}{c}\,\mathrm{sg}[\log\pi_T-\log\pi_S]\Big)\,\log\pi_S
\]

and the microbatch averages that term over selected response tokens only. Unselected steps are masked out. If a batch has no selected tokens, the auxiliary loss is zero. The online analyzer is not trained with a separate supervised loss (`ana_loss_mode=none`).

For a successful trajectory \(i\), the shaped reward is \(R_i+\lambda_{\mathrm{len}} b_i\). Failures keep \(R_i\). Inside one task group,

\[
p_{ij}=\sigma\big((\log T_j-\log T_i)/\tau_{\mathrm{len}}\big),\qquad
b_i=\frac{1}{|S|-1}\sum_{j\in S\setminus\{i\}}p_{ij}.
\]

A group with one success uses \(b_i=1/2\). Text ALFWorld and WebShop assign task reward 10 on success and 0 on failure, so a success scores \(10+5b_i\).

Failed trajectories receive at most three successful action sequences from the same on-policy group. The analyzer prompt is the Appendix C schema: `trajectory_summary` and up to five `step_skills`, each with `target_steps`, `error_reason`, and `skill`.

## Repository layout

| Path | Role |
|---|---|
| `scope/sample/` | Sample 180 training tasks, 8 rollouts each, per base model |
| `scope/sft/` | Supervised warm-up on that model's `train.jsonl` |
| `scope/alfworld/`, `scope/webshop/` | Online SCOPE training |
| `verl/trainer/ppo/step_skill.py` | Analyzer prompt, parser, and step selection |
| `verl/trainer/ppo/opsd_kl_utils.py` | Bounded signed OPSD loss |
| `agent_system/reward_manager/episode.py` | Pairwise length reward |

SFT and RL are separate. RL does not run SFT. Each task and each base model uses its own sample and its own checkpoint.

## Installation

Package versions follow the SDAR environment. Create two conda environments. Names are overridable in `.env` (see `.env.example`).

### `scope` (Python 3.12): ALFWorld

```bash
conda create -n scope python==3.12 -y
conda activate scope
pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
```

ALFWorld game files belong in `data/alfworld`. Point `ALFWORLD_DATA` at that directory. Do not rely on a fresh `alfworld-download` if you need a fixed train pool: the official download writes `~/.cache/alfworld`, and a different game list changes the 180-task sample.

### `scope-webshop` (Python 3.10)

WebShop requires Python <= 3.10.

```bash
conda create -n scope-webshop python==3.10 -y
conda activate scope-webshop
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d small
cd ../../../../..
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

`setup.sh -d small` downloads `items_shuffle_1000.json` and `items_ins_v2_1000.json`. If those files are already present, install the Python dependencies and spaCy models instead of running `setup.sh` again. spaCy / typer version warnings can be ignored.

Copy `.env.example` to `.env` when you need to override paths. Scripts load `.env` through `examples/_sdar_env.sh`.

Logging uses SwanLab (`trainer.logger=['console','swanlab']`). `SWANLAB_MODE=local` needs no API key.

## Data

`data/` is gitignored.

Sampling uses `RandomState(0)`. Keeping the source files unchanged keeps the 180-item draw unchanged.

| File | What it fixes |
|---|---|
| `data/alfworld` (`json_2.1.1`) | ALFWorld train games. The sampler takes 30 games from each of six task types. |
| WebShop `items_shuffle_1000.json`, `items_ins_v2_1000.json` | WebShop train goals. |

ALFWorld and WebShop RL also read `data/verl-agent/text/train.parquet` and `test.parquet`. Those files are batch placeholders. The RL scripts rewrite them from `train_data_size` and `val_data_size`. Task content comes from ALFWorld games and WebShop goals.

## Models and outputs

| Variable | Default |
|---|---|
| `SCOPE_BASE_3B` | `Qwen/Qwen2.5-3B-Instruct`, or `MODEL_PATH` |
| `SCOPE_BASE_7B` | `Qwen/Qwen2.5-7B-Instruct` |
| `SCOPE_BASE_QWEN3` | `Qwen/Qwen3-1.7B`, or `MODEL_PATH_QWEN3` |
| `SCOPE_SAMPLE_ROOT` | `work/rollouts/scope` |
| `SCOPE_SFT_ROOT` | `work/ckpt/sft/scope` |

Put a local model directory in `.env` if you do not want to download from Hugging Face. Qwen3-1.7B SFT defaults to 2 epochs. The other SFT scripts default to 3. Override with `SCOPE_SFT_EPOCHS`.

## 1. Sample

One base model, one output directory. Do not share a sample across model sizes.

```bash
SAMPLE_GPUS=2 bash scope/sample/run_webshop_3b.sh
```

`SAMPLE_GPUS` sets the visible devices. The unset case keeps the current `CUDA_VISIBLE_DEVICES`.

Default output: `$SCOPE_SAMPLE_ROOT/<task>_<model>/`.

- ALFWorld: 180 train games, 30 per task type, 8 rollouts each.
- WebShop: 180 train goals, 8 rollouts each.

The sampler does not write `train.jsonl`. Build that file in the same directory before SFT. Each line is one object whose `messages` field is `[system, user, assistant]`, using the analyzer schema in `verl/trainer/ppo/step_skill.py`. Offline labels can come from any annotator that follows that schema. The paper uses an external annotator only for this warm-up.

## 2. Supervised warm-up

```bash
SFT_CUDA_VISIBLE_DEVICES=0,1 bash scope/sft/run_alfworld_3b.sh
```

The script reads `$SCOPE_SAMPLE_ROOT/<task>_<model>/train.jsonl` and writes `$SCOPE_SFT_ROOT/<task>_<model>/global_step_*`. It exits if `train.jsonl` is missing.

## 3. Online training

```bash
bash scope/webshop/run_webshop_3b_step_skill.sh
```

If `SCOPE_RL_MODEL` is unset, RL loads the latest `global_step_*` directory that contains `config.json`. If it is set, that directory is used and must contain `config.json`. There is no fallback to the base model.

```bash
SCOPE_RL_MODEL=/path/to/global_step_N bash scope/alfworld/run_alfworld_3b_step_skill.sh
```

Serial runs, one model after another:

```bash
bash scope/alfworld/run_alfworld_serial.sh
bash scope/webshop/run_webshop_serial.sh
```

Training runs for 180 steps (`trainer.total_training_steps=180`) and writes a checkpoint at step 150 (`trainer.save_freq=150`). The paper's main ALFWorld and WebShop numbers use the step-150 checkpoint. Validation episodes are the ordinary policy: no skill prompt and no analyzer call. `val_data_size=128`.

### GPUs

| Task | Default GPUs | Rollout tensor parallel |
|---|---|---|
| ALFWorld | 8 | 2 |
| WebShop | 2 | 2 |

```bash
N_GPUS_PER_NODE=4 bash scope/alfworld/run_alfworld_3b_step_skill.sh
```

### Overrides

```bash
OPSD_GATE=clip bash scope/webshop/run_webshop_3b_step_skill.sh
OPSD_GATE=tanh OPSD_ADV_CLIP=10 bash scope/webshop/run_webshop_7b_step_skill.sh
LENGTH_REWARD=adv bash scope/alfworld/run_alfworld_3b_step_skill.sh
```

| Variable | Default | Values |
|---|---|---|
| `OPSD_GATE` | `tanh` | `tanh`, `clip`, `sdar` (alias `sdar-gate`) |
| `OPSD_GATE_BETA` | `2.5` | slope \(\alpha\) for `tanh` and `sdar` |
| `OPSD_ADV_CLIP` | `1` | bound \(c\) for `tanh` and `clip` |
| `LENGTH_REWARD` | `pairwise` | `pairwise`, `adv`, `bonus`, `none` |

`tanh` is the paper's signed coefficient. `clip` and `sdar` are ablations. `LENGTH_REWARD=pairwise` is the paper's efficiency term. `bonus` adds a fixed-horizon length bonus on successes. `adv` adds a length advantage after GRPO normalization and does not change the episode reward. `none` leaves the task reward unchanged.

The experiment name records the gate and the length reward, for example:

```text
scope_webshop_qwen2.5_3b_gatetanh_c1_length_pairwise_step_skill
```

## Training loop

1. Roll out a group of 8 trajectories from the current policy snapshot.
2. The same snapshot analyzes every trajectory. Successes are judged from their own observations, responses, and action-validity flags. Failures also see up to three successful action sequences from the same group.
3. Parsed skills are inserted into the teacher prompt only at their target steps. Teacher log-probabilities are computed from the snapshot and cached.
4. The student is updated with GRPO on the pairwise-shaped reward, plus \(\lambda\) times the masked tanh OPSD loss, plus the reference KL and entropy terms.
5. The updated policy is the actor and the analyzer on the next iteration.

Other script settings: invalid-action penalty coefficient 0.1, ALFWorld `max_steps=30` and `history_length=5`, learning rate `1e-6`, and validation temperature 0.4. The online analyzer completion limit in the training scripts is 1024 tokens. The 8192-token, temperature-0 setting in the paper is the offline annotator used to build SFT data, not the online policy.
