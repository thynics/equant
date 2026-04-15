# equant

`equant` 是一个面向开源 Qwen2 模型的 KV cache 量化实验仓库，重点是对比 `dynamic / quanto / kivi` 三类 cache 在时延和精度上的影响。

当前仓库聚焦这四件事：

- 包含一个可运行的 `KIVI` 基线实现
- 包含开源模型下载与本地准备脚本
- 包含开源评测数据集接入，支持 QA / 长上下文任务打分
- 能快速基于 `kivi` 对一个开源模型在测试集上跑出分数

仓库本身不追踪权重。`artifacts/models/` 和 `artifacts/model_code/` 已被 `.gitignore` 忽略；benchmark 会在本地权重缺失或下载不完整时自动触发一次下载，并在后续运行中直接复用。

## 目录

- `scripts/setup_env.sh`: 创建虚拟环境并安装依赖
- `scripts/prepare_model_assets.py`: 下载或续传一个开源模型到本地
- `scripts/prepare_qwen_assets.py`: 向后兼容的 Qwen 资产准备脚本
- `scripts/run_qa_eval.py`: 运行开源 QA 数据集评测并输出分数
- `scripts/run_kv_bench.py`: 运行 KV cache latency benchmark
- `scripts/run_long_context_eval.py`: 运行一个最小的官方 LongBench-E 子集
- `src/equant/benchmarks/kv_latency.py`: benchmark 主逻辑
- `src/equant/evals/qa_eval.py`: QA 评测主逻辑

## 默认资源位置

- 模型权重: `artifacts/models/<model-id-sanitized>`
- 模型代码快照: `artifacts/model_code/transformers_qwen2`
- 结果输出: `results/`

## 快速开始：下载模型并跑 QA 分数

默认推荐先用一个较小的开源 Qwen2 模型做 smoke test，例如 `Qwen/Qwen2.5-0.5B-Instruct`：

```bash
cd /Users/bytedance/equant
bash scripts/setup_env.sh
source .venv/bin/activate

python scripts/prepare_model_assets.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct

python scripts/run_qa_eval.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --model-dir /Users/bytedance/equant/artifacts/models/Qwen-Qwen2.5-0.5B-Instruct \
  --cache dynamic kivi-int2 quanto-int4 \
  --datasets boolq squad \
  --max-samples-per-dataset 20 \
  --use-chat-template
```

第一次运行时，如果本地模型缺失或只有半截下载，脚本会自动拉取或续传。评测结果会写到 `results/qa_eval_<timestamp>.jsonl`，终端会按 `dataset x cache_mode` 输出聚合分数。

当前内置了五个开源 QA 数据集：

- `boolq`: accuracy
- `squad`: F1 / exact match
- `coqa`: F1 / exact match
- `truthfulqa`: F1 / exact match against truthful reference answers
- `gsm8k`: final numeric answer accuracy

## 模型下载与部署

如果你只想先准备模型，不跑评测，可以直接执行：

```bash
python scripts/prepare_model_assets.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct
```

这个脚本会：

- 下载或续传 Hugging Face 模型快照
- 在 `artifacts/manifests/` 里记录资产清单
- 如果传入 `--vendor-code-dir`，会对 `Qwen2` 模型额外导出本地 `transformers` 源码快照，便于后续排查

## 最小长上下文打分

第一步先接官方 benchmark，而不是自造样例。当前脚本使用 `THUDM/LongBench` 里的 `LongBench-E` 子集，只跑两个 retrieval/synthetic 任务：

- `passage_count`
- `passage_retrieval_en`

这两个任务的 prompt 和 scoring 都按官方 LongBench 配置来，适合先观察 KV cache 量化在长上下文检索能力上的影响，同时不把初始接入做得太重。

```bash
python scripts/run_long_context_eval.py \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --model-dir /Users/bytedance/equant/artifacts/models/Qwen-Qwen2.5-0.5B-Instruct \
  --cache dynamic kivi-int2 quanto-int4 \
  --datasets passage_count passage_retrieval_en \
  --max-samples-per-dataset 20
```

默认只抽样每个数据集 20 条做第一轮实验；如果要跑完整子集，可以传 `--max-samples-per-dataset -1`。输出会按 `dataset x cache_mode` 和 LongBench-E 官方长度桶 `0-4k / 4-8k / 8k+` 汇总分数，并把逐样本结果写到 `results/longbench_e_<timestamp>.jsonl`。

## KIVI baseline

仓库现在支持 `kivi-int2` / `kivi-int4` 作为 baseline cache mode。这里复现的是 KIVI 的核心算法思路：

- key cache: per-channel quantization
- value cache: per-token quantization
- residual full-precision window

当前实现是纯 PyTorch 版本，目标是提供一个可比较的算法 baseline，便于先做时延/LongBench 对比。它没有移植 KIVI 官方 CUDA kernel，因此不能把当前实现的吞吐结果视为官方 KIVI kernel 的极限性能。

## KV Latency Benchmark

如果你要跑纯 latency benchmark，可以继续使用 `run_kv_bench.py`。真实时延实验更适合在 NVIDIA GPU 机器上对更大的模型跑，例如 `Qwen/Qwen2.5-14B`：

```bash
python scripts/prepare_model_assets.py \
  --model-id Qwen/Qwen2.5-14B \
  --model-dir /Users/bytedance/equant/artifacts/models/Qwen2.5-14B \
  --vendor-code-dir /Users/bytedance/equant/artifacts/model_code/transformers_qwen2 \
  --manifest-path /Users/bytedance/equant/artifacts/manifests/qwen2_14b_assets.json

python scripts/run_kv_bench.py \
  --model-id Qwen/Qwen2.5-14B \
  --model-dir /Users/bytedance/equant/artifacts/models/Qwen2.5-14B \
  --cache dynamic kivi-int2 quanto-int2 \
  --prompt-length 2048 \
  --max-new-tokens 128 \
  --trials 3
```

## Benchmark 指标

每个 trial 会输出并落盘这些指标:

- `prefill_latency_s`
- `decode_latency_s`
- `decode_step_mean_ms`
- `decode_step_p50_ms`
- `decode_step_p95_ms`
- `generated_tokens`
- `decode_tokens_per_s`
- `prompt_tokens_per_s`

结果文件默认写到 `results/kv_latency_<timestamp>.jsonl`。

如果你只想验证本地已有权重，不希望任何联网行为，可以加 `--no-download`。

## 当前机器限制

当前这台机器是 `macOS arm64`，且没有 `CUDA`/`MPS` 可用。它适合:

- 准备代码和权重
- 做 QA / LongBench 的小规模框架验证
- 在较小模型上 smoke test

它不适合直接跑 `Qwen2.5-14B` 的真实高性能时延实验。要得到有意义的 KV cache latency 数据，建议在有 NVIDIA GPU 的机器上复用这个目录继续跑。

## 参考

- Qwen2.5-14B 模型卡: <https://huggingface.co/Qwen/Qwen2.5-14B>
- Hugging Face KV cache 文档: <https://huggingface.co/docs/transformers/en/kv_cache>
