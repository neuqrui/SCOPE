<div align="center">

# SCOPE

### Selective Critical-Step On-Policy Self-Distillation<br>for Agentic Reinforcement Learning

**Learn from the right steps. Complete tasks with fewer interactions.**

[![Code](https://img.shields.io/badge/Code-GitHub-24292e?logo=github)](https://github.com/neuqrui/SCOPE)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-ALFWorld%20%7C%20WebShop-317F79)](#benchmarks-and-results)
[![Backbones](https://img.shields.io/badge/Backbones-Qwen2.5%20%7C%20Qwen3-796590)](#benchmarks-and-results)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

An agentic RL framework that localizes errors, generates step-specific hindsight skills,<br>and distills corrective guidance only at the selected decisions.

[Overview](#overview) · [Framework](#framework) · [Results](#benchmarks-and-results) · [Installation](#installation) · [Quick Start](#quick-start) · [Reproduction Guide](docs/reproduction.md) · [Citation](#citation)

</div>

## Overview

**Task success does not tell us which steps were correct.** A successful agent can recover from mistakes, while a failed trajectory can contain useful decisions. Applying one hindsight skill to an entire trajectory can also change teacher preferences at unrelated steps.

<p align="center">
  <img src="assets/motivation.png" alt="Outcome-step mismatch and off-target teacher signals from trajectory-level skills." width="100%">
</p>
<p align="center"><sub>In the paper's annotated WebShop trajectories, 29.8% of steps in successful episodes are erroneous, while 42.4% of steps in failed episodes are correct. Skill-unrelated steps account for 57.9% of nonzero sampled-token teacher–student gaps.</sub></p>

SCOPE combines outcome-based GRPO with selective on-policy self-distillation:

- **Self-bootstrapped error analysis.** After a supervised warm-up, the current policy analyzes both successful and failed rollouts and writes corrective skills for selected error steps.
- **Step-specific distillation.** Each skill guides only its target steps; other steps receive no auxiliary distillation signal.
- **Bounded signed updates.** A smooth, bounded teacher–student coefficient preserves both positive reinforcement and negative correction while limiting extreme updates.
- **Success-relative efficiency.** A pairwise length reward favors shorter successful trajectories within the same task group.

**At evaluation, SCOPE uses the ordinary interaction history, with no hindsight skills or analyzer calls.** External annotation is used only for the offline supervised warm-up.

## Framework

<p align="center">
  <img src="assets/framework.png" alt="SCOPE framework: analyzer initialization, self-bootstrapped error analysis, selective bounded signed distillation, and efficiency-aware GRPO." width="100%">
</p>
<p align="center"><sub>Supervised initialization teaches error localization and skill generation. Online training combines selective corrective supervision with task rewards and a preference for shorter successful interactions.</sub></p>

### 1. Initialize the analyzer

Collect a small pool of base-policy trajectories and annotate errors with an external annotator (GLM-5.2 in the paper). Each annotation contains a trajectory summary and up to five skill entries, with `target_steps`, `error_reason`, and `skill`. Supervised fine-tuning teaches the policy to produce this structured analysis.

### 2. Analyze on-policy experience

For each task, sample a group of eight trajectories. The same policy snapshot analyzes its own experience. Successful rollouts are inspected for avoidable mistakes; failed rollouts may also receive up to three successful action sequences from the same group. The analyzer can return no skills when the recorded evidence supports no clear error.

### 3. Distill only at selected steps

The teacher receives the skill assigned to the selected step. Teacher and student score the same sampled response tokens, while the student receives the ordinary history. Teacher scores are cached before student optimization; the snapshot stays fixed during that outer iteration.

For a selected response token, define the signed gap and bounded coefficient as

$$
\Delta = \log \pi_T - \log \pi_S,
\qquad
w(\Delta) = c\,\tanh\!\left(\frac{\alpha}{c}\,\operatorname{sg}[\Delta]\right).
$$

The auxiliary loss averages $-w(\Delta)\log\pi_S$ over selected response tokens only. Here $\operatorname{sg}$ stops gradients through the coefficient. Unselected steps are masked out, and an empty selection gives zero auxiliary loss. The default values are $\alpha=2.5$, $c=1$, and distillation weight $0.01$.

### 4. Reward efficient success

Within a task group, compare each successful trajectory's interaction length against the other successes:

$$
p_{ij}=\sigma\!\left(\frac{\log T_j-\log T_i}{\tau_{\mathrm{len}}}\right),
\qquad
b_i=\frac{1}{|S|-1}\sum_{j\in S\setminus\{i\}}p_{ij}.
$$

$S$ is the set of successful trajectories, $T_i$ counts environment interactions, and $\sigma$ is the sigmoid. Shorter successes receive a larger bonus. A group with one success uses $b_i=1/2$; failures keep their original task reward. The shaped reward for a success is $R_i+\lambda_{\mathrm{len}}b_i$, with $\tau_{\mathrm{len}}=0.1$ and $\lambda_{\mathrm{len}}=5$.

The student is updated with efficiency-aware GRPO, selective self-distillation, reference KL regularization, and an entropy term. The updated policy supplies the actor and analyzer for the next iteration.

## Benchmarks and Results

SCOPE is evaluated on **ALFWorld** (text-based household tasks) and **WebShop** (product search and purchase), across three Qwen backbones. ALFWorld reports success across 128 seen-split validation episodes; WebShop uses a fixed set of 128 tasks and reports normalized task score and success rate. All values below are percentages from the paper's main table; SCOPE uses the **step-150 checkpoint**.

### Main comparison

| Backbone | Method | ALFWorld success ↑ | WebShop score ↑ | WebShop success ↑ |
| :--- | :--- | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | GRPO | 75.0 | 79.8 | 63.3 |
| | SDAR | 84.4 | 85.0 | 68.0 |
| | GRSD | 86.7 | 86.9 | 76.5 |
| | SEED | 91.8 | 88.5 | 78.9 |
| | **SCOPE** | **97.7** | 87.9 | **85.2** |
| Qwen2.5-7B-Instruct | GRPO | 81.2 | 80.9 | 72.6 |
| | SDAR | 85.9 | 89.4 | 82.8 |
| | GRSD | 92.2 | 91.6 | 82.8 |
| | SEED | 96.1 | 89.7 | 78.1 |
| | **SCOPE** | **98.4** | 89.6 | **84.4** |
| Qwen3-1.7B | GRPO | 46.1 | 67.3 | 38.3 |
| | SDAR | 53.9 | 76.8 | 58.6 |
| | GRSD | **92.2** | 84.5 | 66.4 |
| | SEED | 92.0 | 87.1 | 77.3 |
| | **SCOPE** | **92.2** | 83.2 | **78.9** |

SCOPE achieves the highest WebShop success and the highest or tied highest ALFWorld success among the evaluated methods for each backbone. On Qwen2.5-3B, the gains over SEED are **5.9 percentage points on ALFWorld** and **6.3 points on WebShop**. WebShop score measures partial requirement matching separately from exact task success; SCOPE does not lead every score comparison.

<details>
<summary><b>All eight baselines on Qwen2.5-3B-Instruct</b></summary>

| Method | Guidance | ALFWorld success ↑ | WebShop score ↑ | WebShop success ↑ |
| :--- | :--- | ---: | ---: | ---: |
| Vanilla | None | 21.9 | 6.7 | 0.8 |
| Skill-Prompt* | Skill bank | 28.9 | 0.2 | 0.8 |
| GRPO | None | 75.0 | 79.8 | 63.3 |
| AgentOPSD | Skill bank | 84.4 | 90.4 | 69.5 |
| SDAR | Skill bank | 84.4 | 85.0 | 68.0 |
| OPID | Hybrid skills | 84.3 | 85.0 | 74.2 |
| GRSD | Trajectory-level skill | 86.7 | 86.9 | 76.5 |
| SEED | Trajectory-level skill | 91.8 | 88.5 | 78.9 |
| **SCOPE** | **Step-specific skills** | **97.7** | 87.9 | **85.2** |

*Skill-Prompt receives skills at evaluation. The other methods use their ordinary policy inputs.

</details>

### Generalization to unseen tasks

On ALFWorld Unseen, SCOPE reaches **89.6% success**, compared with **66.4% for GRPO**. Reported average turns decrease from **18.86 to 10.06**, a **46.7% reduction**.

<p align="center">
  <img src="assets/unseen-generalization.png" alt="ALFWorld Unseen: SCOPE improves success across all six task families and reduces reported average turns." width="760">
</p>

<details>
<summary><b>Ablations and interaction efficiency</b></summary>

ALFWorld ablations use Qwen2.5-3B at **step 180**, separately from the step-150 main comparison.

| Variant | ALFWorld success ↑ |
| :--- | ---: |
| **Full SCOPE** | **98.4** |
| Without SFT initialization | 92.2 |
| Without Step-Skill | 96.1 |
| Trajectory-level guidance | 96.1 |
| Skill-bank guidance | 95.3 |
| Without length reward | 93.0 |
| Absolute length bonus | 97.7 |

<p align="center"><img src="assets/ablation-success.png" alt="ALFWorld training dynamics comparing initialization, guidance, and length reward variants." width="100%"></p>
<p align="center"><sub>Paper ablation and parameter-sensitivity curves. Success axes start at 10% in panel A and 40% in panels B–D.</sub></p>

<p align="center"><img src="assets/response-steps.png" alt="Mean response steps during ALFWorld training; lower is better." width="620"></p>
<p align="center"><sub>Mean response steps on ALFWorld. The GRPO comparison uses seed 0.</sub></p>

</details>

## Installation

SCOPE builds on the [SDAR](https://github.com/ZJU-REAL/SDAR) / verl-agent training stack. Use separate environments for ALFWorld and WebShop, which require different Python and vLLM versions.

```bash
git clone https://github.com/neuqrui/SCOPE.git
cd SCOPE
cp .env.example .env
```

Edit `.env` to set local model and data paths. `SWANLAB_MODE=local` enables local logging without an API key.

<details>
<summary><b>ALFWorld environment · Python 3.12</b></summary>

```bash
conda create -n scope python==3.12 -y
conda activate scope
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
```

Place ALFWorld game files (`json_2.1.1`) under `data/alfworld` and set `ALFWORLD_DATA` accordingly. Preserve the same game pool to reproduce the 180-task sample; a fresh upstream download may change that pool.

</details>

<details>
<summary><b>WebShop environment · Python 3.10</b></summary>

```bash
conda create -n scope-webshop python==3.10 -y
conda activate scope-webshop
cd agent_system/environments/env_package/webshop/webshop
./setup.sh -d small
cd ../../../../..
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
pip install vllm==0.8.2
```

The setup downloads `items_shuffle_1000.json` and `items_ins_v2_1000.json`. If those files already exist, install the Python dependencies and spaCy models directly instead of downloading the data again.

</details>

Default online launchers use **8 GPUs for ALFWorld** and **2 GPUs for WebShop**, with rollout tensor parallelism of 2. `N_GPUS_PER_NODE` overrides the device count; memory needs depend on the backbone and batch settings.

## Quick Start

Run commands from the repository root after preparing the corresponding environment and data. Each task and backbone needs its **own rollout sample, annotated SFT data, and warm-up checkpoint**.

### 1. Collect base-policy rollouts

```bash
# ALFWorld · Qwen2.5-3B
SAMPLE_GPUS=0,1 bash scope/sample/run_alfworld_3b.sh

# WebShop · Qwen2.5-3B
SAMPLE_GPUS=0,1 bash scope/sample/run_webshop_3b.sh
```

Each sampler draws 180 training tasks with eight rollouts per task, using `RandomState(0)`. Outputs go to `work/rollouts/scope/<task>_<model>/` by default.

### 2. Prepare annotations and warm up

**Sampling does not create the annotated `train.jsonl`.** Annotate the collected trajectories using the analyzer schema in [`verl/trainer/ppo/step_skill.py`](verl/trainer/ppo/step_skill.py), then save one system–user–assistant conversation per line under the `messages` field:

```text
work/rollouts/scope/alfworld_qwen2.5_3b/train.jsonl
work/rollouts/scope/webshop_qwen2.5_3b/train.jsonl
```

The assistant target contains `trajectory_summary` and `step_skills`. Each skill has `target_steps`, `error_reason`, and `skill`. The paper uses GLM-5.2 for these offline labels; online training uses the policy itself as the analyzer.

```bash
SFT_CUDA_VISIBLE_DEVICES=0,1 bash scope/sft/run_alfworld_3b.sh
SFT_CUDA_VISIBLE_DEVICES=0,1 bash scope/sft/run_webshop_3b.sh
```

Warm-up checkpoints are written to `work/ckpt/sft/scope/<task>_<model>/global_step_*`. Qwen2.5 defaults to three epochs; Qwen3-1.7B defaults to two. Set `SCOPE_SFT_EPOCHS` to override.

### 3. Run online SCOPE training

```bash
bash scope/alfworld/run_alfworld_3b_step_skill.sh
bash scope/webshop/run_webshop_3b_step_skill.sh
```

RL automatically loads the latest warm-up checkpoint for the matching task and backbone. To choose a checkpoint explicitly:

```bash
SCOPE_RL_MODEL=/path/to/global_step_N \
  bash scope/alfworld/run_alfworld_3b_step_skill.sh
```

The checkpoint must contain `config.json`. The launchers require SFT weights and do not fall back to the base model. Training runs for 180 updates, saves at step 150, and validates every five updates. Main paper results use step 150; validation runs without skills or analyzer calls.

For Qwen2.5-7B or Qwen3-1.7B, use the matching `7b` or `qwen3_1.7b` launchers in each stage. Full model paths, data preparation notes, hyperparameters, ablation switches, and serial launchers are documented in the **[Reproduction Guide](docs/reproduction.md)**.

## Repository

```text
SCOPE/
├── scope/
│   ├── sample/                    Base-policy rollout collection
│   ├── sft/                       Supervised analyzer warm-up
│   ├── alfworld/                  ALFWorld online training
│   ├── webshop/                   WebShop online training
│   └── env.sh                     Model paths and method defaults
├── verl/trainer/ppo/
│   ├── step_skill.py              Analyzer schema and step selection
│   └── opsd_kl_utils.py            Bounded signed distillation
├── agent_system/reward_manager/
│   └── episode.py                 Success-relative pairwise reward
├── docs/reproduction.md           Detailed setup and training guide
└── assets/                        Paper figures used in this README
```

Base models default to `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, and `Qwen/Qwen3-1.7B`. Override them in `.env` with `SCOPE_BASE_3B`, `SCOPE_BASE_7B`, and `SCOPE_BASE_QWEN3`. Environment data and generated training outputs are prepared locally; `data/` is gitignored.

## Acknowledgements

SCOPE builds on [SDAR](https://github.com/ZJU-REAL/SDAR), [verl](https://github.com/volcengine/verl), and [verl-agent](https://github.com/langfengQ/verl-agent), with evaluations on [ALFWorld](https://github.com/alfworld/alfworld) and [WebShop](https://github.com/princeton-nlp/WebShop). We thank the authors and maintainers of these projects.

## License

This repository is released under the [Apache License 2.0](LICENSE). Refer to the upstream projects for their respective licenses and data terms.

## Citation

To reference this repository:

```bibtex
@misc{scope2026,
  title = {SCOPE: Selective Critical-Step On-Policy Self-Distillation for Agentic Reinforcement Learning},
  year  = {2026},
  url   = {https://github.com/neuqrui/SCOPE}
}
```
