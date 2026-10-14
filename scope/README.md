# SCOPE 训练流程

三个任务是 ALFWorld、WebShop、Search。每个任务三个模型：Qwen2.5-3B、Qwen2.5-7B、Qwen3-1.7B。RL 方法固定三点：`step_skill`、OPSD gate、length reward。其余训练超参对齐 `examples/sdar_trainer`。

SFT 和 RL 是分开的脚本。RL 不会自己去做 SFT。

## 先做什么

对每一个「任务 + 模型」按这个顺序。三个模型各采各的，不能共用一份数据。

1. 用这个基座模型采样。180 条训练数据，每条 8 次，得到该模型自己的离线轨迹。
2. 把这份采样做成 SFT jsonl，放到同一次采样的目录里，文件名固定为 `train.jsonl`。格式与 ALFWorld analyzer 相同：每行一个对象，`messages` 为 `[system, user, assistant]`。
3. SFT，默认 3 个 epoch。读的就是上一步这个模型的 `train.jsonl`。权重写到 `global_step_*`。
4. RL。自动读取该任务、该模型 SFT 目录里步数最大、且含 `config.json` 的 checkpoint。

九组数据彼此独立，例如 WebShop 3B 的 SFT 只用 `webshop_qwen2.5_3b/train.jsonl`，不会去读 7B 或 Qwen3-1.7B 的采样。`train.jsonl` 不存在时 SFT 脚本会退出。采样报告不会自动写成这份 jsonl。

## Conda 环境

安装命令和包版本跟仓库根目录 `README.md` 的 Installation 一节相同。本仓库脚本实际激活的环境名写在根目录 `.env`，由 `examples/_sdar_env.sh` 按 `TASK_KIND` 选择。不要照 README 里的 `verl-webshop`、`retriever` 另建一套，否则脚本找不到环境。

| 任务 | `TASK_KIND` | 环境名 | Python | 用来跑什么 |
|---|---|---|---|---|
| ALFWorld、Search 的采样 / SFT / RL | `ALFWORLD` 或 `SEARCH` | `sdar`（`CONDA_ENV`） | 3.12 | 学生模型 |
| WebShop 的采样 / SFT / RL | `WEBSHOP` | `sdar-webshop`（`CONDA_ENV_WEBSHOP`） | 3.10 | 学生模型。WebShop 要求 Python ≤ 3.10 |
| Search 检索服务 | `RETRIEVER` | `sdar-retriever`（`CONDA_ENV_RETRIEVER`） | 3.10 | 只跑 `examples/search/retriever/retrieval_launch.sh`。约占 6GB 显存 |

环境前缀默认是 `CONDA_ENV_BIN_ROOT=/data/qirui/envs`，也就是 `/data/qirui/envs/sdar` 这种目录。换机器时在 `.env` 里改 `CONDA_ENV_BIN_ROOT`。

`sdar`（ALFWorld 和 Search 训练）:

```bash
conda create -n sdar python==3.12 -y
conda activate sdar
pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

`sdar-webshop`:

```bash
conda create -n sdar-webshop python==3.10 -y
conda activate sdar-webshop
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d small
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```

`setup.sh -d small` 会用 gdown 覆盖 `items_shuffle_1000.json` 和 `items_ins_v2_1000.json`。已有这两份文件时不要跑完整的 `setup.sh`。新环境只装依赖：`pip install -r requirements.txt`、`conda install -c conda-forge openjdk=11`、两个 spacy 模型，然后把现有 json 放回 `webshop/data`。spacy / typer 的版本警告可以忽略。

`sdar-retriever`:

```bash
conda create -n sdar-retriever python=3.10 -y
conda activate sdar-retriever
conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

日志用 SwanLab，不用 WandB。Key 见文末。

## 数据

`data/` 在 `.gitignore` 里，`git clone` 不会带上这些文件。本机 `data/` 是指向 `/data/qirui/SDAR/data` 的符号链接。换机器时把这个目录拷过去，并让仓库里的 `data/` 指到它。不要重新下载会改变抽样的那几份文件。

采样脚本都用 `RandomState(0)`。下面三份文件的字节不变，再抽 180 条就是同一批。文件变了，抽出来的 180 条就会变。

| 必须原样保留 | 大约大小 | 决定什么 |
|---|---|---|
| `data/alfworld`，尤其 `json_2.1.1` | 2.4GB | ALFWorld 训练游戏池。采样在六类任务里各取 30 个 |
| `agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json` 和 `items_ins_v2_1000.json` | 4.3MB + 144KB | WebShop 训练 goal 池。`data/webshop/` 里有一份相同副本 |
| `data/searchR1_processed_direct/train.parquet` 和 `test.parquet` | 343MB + 71MB | Search 的训练题和验证题。采样只在 `train.parquet` 里抽 180 行 |

不要做的事：

- 不要跑 `alfworld-download`。官方包会写到 `~/.cache/alfworld`，游戏列表可能和现在的 `data/alfworld` 不一致。`~/.cache/alfworld` 现在是指向 `data/alfworld` 的符号链接。
- 不要对已经有 json 的目录再跑 `./setup.sh -d small` 或 `-d all`。
- 不要重新跑 `python examples/data_preprocess/preprocess_search_r1_dataset.py` 来覆盖 `data/searchR1_processed_direct`。

Search 检索语料不是题目，不决定抽到哪 180 道题。只有要真正跑 Search 环境时才需要，大约 74GB：

- `data/searchR1/e5_Flat.index`（61GB）
- `data/searchR1/wiki-18.jsonl`（14GB）

已有这两份就直接用。没有时才下载，并且必须下到 `data/searchR1`，不要下到别的 `~/data/searchR1`：

```bash
conda activate sdar-retriever
local_dir=data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

索引来自 `PeterJinGo/wiki-18-e5-index`，语料来自 `PeterJinGo/wiki-18-corpus`。服务入口是 `bash examples/search/retriever/retrieval_launch.sh`，读 `SEARCH_DATA_ROOT`。

ALFWorld 和 WebShop 的 RL 还读 `data/verl-agent/text/train.parquet`、`test.parquet`。这两个文件只有几 KB，是 batch 占位，不是任务内容。`scope/*/ _run.sh` 里的 `examples.data_preprocess.prepare` 会按 `train_data_size` 重写它们。任务内容来自 `data/alfworld` 和 WebShop json。

根目录 `.env` 里已经把路径指到仓库 `data/`：

```text
DATA_ROOT=/home/liuqirui/SDAR/data
ALFWORLD_DATA=/home/liuqirui/SDAR/data/alfworld
SEARCH_DATA_ROOT=/home/liuqirui/SDAR/data/searchR1
SEARCH_PROCESSED_ROOT=/home/liuqirui/SDAR/data/searchR1_processed_direct
SEARCH_URL=http://127.0.0.1:8000/retrieve
```

`examples/_sdar_env.sh` 在这些变量没设置时，默认也是 `$PROJECT_ROOT/data` 下的同名目录。

## 路径

`scope/env.sh` 会先加载仓库根目录 `.env`（经 `examples/_sdar_env.sh`），再加载 `scope/.env`。模型与数据默认值写在 `scope/env.sh`，`scope/.env` 可以覆盖。

| 变量 | 默认 |
|---|---|
| `SCOPE_BASE_3B` | `/data/qirui/SDAR/models/Qwen2.5-3B-Instruct` |
| `SCOPE_BASE_7B` | `Qwen/Qwen2.5-7B-Instruct` |
| `SCOPE_BASE_QWEN3` | `/data/qirui/SDAR/models/Qwen3-1.7B` |
| `SCOPE_SFT_ROOT` | `/data/qirui/SDAR/ckpt/sft/scope` |
| `SCOPE_SAMPLE_ROOT` | `/data/qirui/SDAR/ckpt/rollouts_val/scope` |

每个模型的 SFT jsonl 默认是 `$SCOPE_SAMPLE_ROOT/<任务>_<模型名>/train.jsonl`。单次换文件可以设 `SCOPE_SFT_JSONL`。

仓库根目录 `.env` 里的 `MODEL_PATH_QWEN3` 指向 Qwen3-4B。SCOPE 不用它。

本机没有本地的 Qwen2.5-7B-Instruct 目录，7B 与 SDAR 脚本一样使用 Hugging Face 名。

## 1. 采样

脚本在 `scope/sample/`。名字是 `run_<任务>_<尺寸>.sh`，尺寸为 `3b`、`7b`、`qwen3_1.7b`。

```bash
SAMPLE_GPUS=2 bash scope/sample/run_webshop_3b.sh
```

`SAMPLE_GPUS` 指定可见卡。不设时沿用当前环境里的 `CUDA_VISIBLE_DEVICES`。

输出目录按模型分开，默认是 `$SCOPE_SAMPLE_ROOT/<任务>_<模型名>/`，例如 `/data/qirui/SDAR/ckpt/rollouts_val/scope/webshop_qwen2.5_3b/`。这个目录只属于这一次采样所用的基座模型。

- ALFWorld：`games.json`，以及该目录下的轨迹报告。180 个训练游戏，每类 30 个，每个 8 次。
- WebShop：`goals.json` 和 `qwen2.5_3b.json`（文件名随模型变）。180 个训练 goal，每个 8 次。
- Search：`questions.json` 和 `<模型名>.json`。从训练集随机抽 180 条，每条 8 次。检索服务需要已经在 `SEARCH_URL` 上。

## 2. SFT

脚本在 `scope/sft/`。名字是 `run_<任务>_<尺寸>.sh`。

```bash
SFT_CUDA_VISIBLE_DEVICES=0,1 bash scope/sft/run_alfworld_3b.sh
```

不设 `SFT_CUDA_VISIBLE_DEVICES` 时使用当前的 `CUDA_VISIBLE_DEVICES`。进程数等于卡数。脚本读取 `$SCOPE_SAMPLE_ROOT/<任务>_<模型名>/train.jsonl`，也就是该基座模型自己的采样目录。

保存目录是 `$SCOPE_SFT_ROOT/<任务>_<模型名>/`，例如：

```text
/data/qirui/SDAR/ckpt/sft/scope/alfworld_qwen2.5_3b/global_step_87
```

`SCOPE_SFT_EPOCHS` 默认 3。

## 3. RL

入口在 `scope/<任务>/run_<任务>_<尺寸>_step_skill.sh`。三个入口都执行同目录的 `_run.sh`。三个任务默认都训 180 个 step（`trainer.total_training_steps=180`），`trainer.save_freq=150`，所以第 150 步会存 checkpoint。第 180 步是最后一步，trainer 还会再存一次。`total_epochs=180` 只是让 ALFWorld 和 WebShop 的 dataloader 够跑满这些 step；Search 会在第 180 步停，不会把整个 `train.parquet` 扫 180 遍。

```bash
bash scope/webshop/run_webshop_3b_step_skill.sh
```

模型路径由 `scope_latest_sft` 决定，定义在 `scope/env.sh`：

- 未设置 `SCOPE_RL_MODEL` 时，使用 `$SCOPE_SFT_ROOT/<任务>_<模型名>/` 下步数最大且含 `config.json` 的 `global_step_*`。
- 设置了 `SCOPE_RL_MODEL` 时，直接用这个目录，目录里必须有 `config.json`。

```bash
SCOPE_RL_MODEL=/data/qirui/SDAR/ckpt/sft/alfworld_analyzer_qwen2.5_3b/global_step_261 \
  bash scope/alfworld/run_alfworld_3b_step_skill.sh
```

目录不存在时脚本退出，不会退回基座模型。

一个任务的三个模型按 3B、7B、Qwen3-1.7B 串行：

```bash
bash scope/alfworld/run_alfworld_serial.sh
bash scope/webshop/run_webshop_serial.sh
bash scope/search/run_search_serial.sh
```

Search 的 serial 会在 `SEARCH_URL` 没有响应时启动 `examples/search/retriever/retrieval_launch.sh`，三个模型都结束后关掉它自己拉起的服务。

### GPU

卡数按任务，不按模型。可用 `N_GPUS_PER_NODE` 覆盖。

| 任务 | 默认 GPU | rollout 张量并行 |
|---|---|---|
| ALFWorld | 8 | 2 |
| WebShop | 2 | 2 |
| Search | 4 | 1 |

```bash
N_GPUS_PER_NODE=4 bash scope/alfworld/run_alfworld_3b_step_skill.sh
```

### Gate 和 length reward

只影响 RL。在启动命令前加环境变量，不用改脚本。`scope_init_method`（`scope/env.sh`）读取它们，传给 `algorithm.opsd_kl.gate`，并写进实验名。

```bash
OPSD_GATE=clip bash scope/webshop/run_webshop_3b_step_skill.sh
OPSD_GATE=tanh OPSD_ADV_CLIP=10 bash scope/webshop/run_webshop_7b_step_skill.sh
OPSD_GATE=sdar OPSD_GATE_BETA=5.0 bash scope/alfworld/run_alfworld_3b_step_skill.sh
LENGTH_REWARD=adv bash scope/search/run_search_3b_step_skill.sh
```

| 变量 | 默认 | 取值 |
|---|---|---|
| `OPSD_GATE` | `sdar` | `sdar`（或 `sdar-gate`）、`clip`、`tanh` |
| `OPSD_GATE_BETA` | `5.0` | 只用于 `sdar` |
| `OPSD_ADV_CLIP` | `10` | 只用于 `clip` 和 `tanh` |
| `LENGTH_REWARD` | `pairwise` | `pairwise`、`adv`、`bonus`、`none` |

默认实验名：

```text
scope_webshop_qwen2.5_3b_gatesdar_beta5.0_length_pairwise_step_skill
```

换成 clip 且 `OPSD_ADV_CLIP=10` 时，中间一段变为 `gateclip_c10`。不同 gate 因此落在不同 checkpoint 目录。

## SwanLab

Key 在仓库根目录 `.env`，变量名是 `SWANLAB_API_KEY`。同文件还有 `SWANLAB_MODE`（当前为 `cloud`）和 `SWANLAB_LOG_DIR`（当前为 `/data/qirui/SDAR/logs/swanlab`）。

`scope/env.sh` 通过 `examples/_sdar_env.sh` 加载这份 `.env`。`scope/.env` 里没有 SwanLab 配置。改 key 时改根目录 `.env`，或在启动前导出 `SWANLAB_API_KEY`。
