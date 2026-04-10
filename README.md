# equant

`equant` 是一个面向 Qwen 14B 的 KV cache 量化时延实验框架。当前默认模型是 `Qwen/Qwen2.5-14B`，原因是它是官方 14B 基座模型，且模型实现已经集成在较新的 `transformers` 中。

仓库本身不追踪权重。`artifacts/models/` 和 `artifacts/model_code/` 已被 `.gitignore` 忽略；benchmark 会在本地权重缺失或下载不完整时自动触发一次下载，并在后续运行中直接复用。

## 目录

- `scripts/setup_env.sh`: 创建虚拟环境并安装依赖
- `scripts/prepare_qwen_assets.py`: 可选的预热脚本；如果本地缺少权重，会下载并导出本地 Qwen2 `transformers` 源码快照
- `scripts/run_kv_bench.py`: 运行 KV cache latency benchmark
- `src/equant/benchmarks/kv_latency.py`: benchmark 主逻辑

## 默认资源位置

- 模型权重: `artifacts/models/Qwen2.5-14B`
- 模型代码快照: `artifacts/model_code/transformers_qwen2`
- 结果输出: `results/`

## 快速开始

```bash
cd /Users/bytedance/equant
bash scripts/setup_env.sh
source .venv/bin/activate

python scripts/run_kv_bench.py \
  --model-dir /Users/bytedance/equant/artifacts/models/Qwen2.5-14B \
  --cache dynamic quanto-int4 quanto-int2 \
  --prompt-length 2048 \
  --max-new-tokens 128 \
  --trials 3
```

第一次运行时，如果 `artifacts/models/Qwen2.5-14B` 缺失或只有半截下载，脚本会自动拉取或续传。你也可以先手动预热:

```bash
python scripts/prepare_qwen_assets.py
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
- 做框架验证
- 在较小模型上 smoke test

它不适合直接跑 `Qwen2.5-14B` 的真实高性能时延实验。要得到有意义的 KV cache latency 数据，建议在有 NVIDIA GPU 的机器上复用这个目录继续跑。

## 参考

- Qwen2.5-14B 模型卡: <https://huggingface.co/Qwen/Qwen2.5-14B>
- Hugging Face KV cache 文档: <https://huggingface.co/docs/transformers/en/kv_cache>
