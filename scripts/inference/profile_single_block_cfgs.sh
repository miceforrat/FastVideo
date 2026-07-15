#!/usr/bin/env bash
set -euo pipefail

SCRIPT="examples/inference/profiling/profile_single_ts_block_args.py"
LOG_DIR="logs/block_profile"
mkdir -p "${LOG_DIR}"

SEQLENS=(640 1024 1560 2304 3600)
CHUNK_SIZES=(1 2 3 4 5)
KV_MULTS=(7)
TEXTLENS=(256 512 1024)
BATCH_SIZES=(1 2 3 4)

# model_name:num_heads:head_dim:ffn_dim
MODELS=(
  "sfwan2.1-1.3B:12:128:8960"
  "sfwan2.1-a14b:40:128:13824"
)

for model_cfg in "${MODELS[@]}"; do
  IFS=":" read -r MODEL_NAME NUM_HEADS HEAD_DIM FFN_DIM <<< "${model_cfg}"

  for SEQLEN in "${SEQLENS[@]}"; do
    for CS in "${CHUNK_SIZES[@]}"; do
      for KV_MULT in "${KV_MULTS[@]}"; do
        KV_CACHE_FRAMES=$((CS * KV_MULT))

        for TEXTLEN in "${TEXTLENS[@]}"; do
          for BS in "${BATCH_SIZES[@]}"; do

            LOG_FILE="${LOG_DIR}/${MODEL_NAME}_seqlen${SEQLEN}_cs${CS}_kv${KV_CACHE_FRAMES}f_kvm${KV_MULT}_text${TEXTLEN}_bs${BS}_h${HEAD_DIM}x${NUM_HEADS}_ffn${FFN_DIM}.log"

            echo "Running: ${LOG_FILE}"

            python "${SCRIPT}" \
              --seqlen "${SEQLEN}" \
              --kv-cache-frames "${KV_CACHE_FRAMES}" \
              --chunk-size "${CS}" \
              --num-chunks 7 \
              --batch-size "${BS}" \
              --head-dim "${HEAD_DIM}" \
              --num-heads "${NUM_HEADS}" \
              --ffn-dim "${FFN_DIM}" \
              --text-len "${TEXTLEN}" \
              > "${LOG_FILE}" 2>&1

          done
        done
      done
    done
  done
done

echo "All profiling runs finished. Logs saved to ${LOG_DIR}"