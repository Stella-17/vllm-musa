# Qwen3.8-27B-FP8

## Overview

FP8 dense-model recipe with three-token MTP speculative decoding.

> [!TIP]
> Use this TP2 profile when serving the FP8 checkpoint.

## At a glance

| Setting | Value |
|---|---|
| Hardware | 2x S5000 |
| Precision | FP8 |
| Tensor parallelism | TP2 |
| Speculative decoding | MTP3 |
| Maximum context | 8,192 tokens |
| Maximum sequences | 128 |

## Validation

| Item | Value |
|---|---|
| Checkpoint | `Qwen/Qwen3.8-27B-FP8` @ <revision> |
| vLLM-MUSA | v0.28.0 release image, v0.28.0-dev @ <commit> |
| MUSA driver | <driver version> |
| Hardware | 2x S5000 |
| Served request | 4K-input/1K-output, up to 128 concurrent |

## Launching the server

```bash
export VLLM_PLUGINS=musa,musa_custom_ops
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1

CAPTURE_SIZES="$(seq -s, 4 4 512)"

vllm serve /models/Qwen3.8-27B-FP8 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --served-model-name Qwen3.8-27B-FP8 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config "{\"mode\":\"NONE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

## Configuration notes

- This FP8 checkpoint was validated independently; see the BF16 page for the
  architecture note and the shared 27B dense workload rationale.
- The memory-utilization target drops to 0.80 for the FP8 checkpoint.
- MTP3 uses the FlashAttention backend for draft attention.
- Graph capture sizes account for the additional draft tokens.
- Replace `/models/Qwen3.8-27B-FP8` if needed.

Return to the [Qwen recipe index](README.md).
