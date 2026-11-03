# Qwen3.5-2B

## Overview

BF16 small-model recipe for a single S5000 GPU with MTP1 speculative decoding.

> [!TIP]
> Use TP1 and enable MTP1 speculative decoding for this checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 1x S5000 |
| Precision | BF16 |
| Tensor parallelism | TP1 |
| Speculative decoding | MTP1 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |
| Prefix caching | Off |

## Validation

| Item | Value |
|---|---|
| Checkpoint | `Qwen/Qwen3.5-2B` @ <revision> |
| vLLM-MUSA | v0.28.0 release image, v0.28.0-dev @ <commit> |
| MUSA driver | <driver version> |
| Hardware | 1x S5000 |
| Served request | 4K-input/1K-output, up to 128 concurrent |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export VLLM_DISABLE_COMPILE_CACHE=1
export SAFETENSORS_FAST_GPU=1

vllm serve /models/Qwen3.5-2B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --served-model-name Qwen3.5-2B \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
```

## Configuration notes

- Chunked prefill stays enabled for long prompts.
- `VLLM_DISABLE_COMPILE_CACHE=1` matches the validated QA launch script.
- Replace `/models/Qwen3.5-2B` if needed.

Return to the [Qwen recipe index](README.md).
