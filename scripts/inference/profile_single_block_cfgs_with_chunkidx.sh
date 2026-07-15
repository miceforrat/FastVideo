#!/usr/bin/env bash
# set -euo pipefail

SCRIPT="examples/inference/profiling/profile_single_ts_block_args_with_chunkidx.py"
LOG_DIR="logs/block_profile_chunk_idx"
CSV_PATH="${LOG_DIR}/profile_results.csv"
FAIL_COUNT=0
mkdir -p "${LOG_DIR}"

SEQLENS=(640 1024 1560 2304 3600)
CHUNK_SIZES=(1 2 3 4 5)
CHUNK_IDXS=(0 1 2 3 4 5 6)
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
      for CHUNK_IDX in "${CHUNK_IDXS[@]}"; do
        for KV_MULT in "${KV_MULTS[@]}"; do
          KV_CACHE_FRAMES=$((CS * KV_MULT))

          for TEXTLEN in "${TEXTLENS[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do

              LOG_FILE="${LOG_DIR}/${MODEL_NAME}_seqlen${SEQLEN}_cs${CS}_chunkidx${CHUNK_IDX}_kv${KV_CACHE_FRAMES}f_kvm${KV_MULT}_text${TEXTLEN}_bs${BS}_h${HEAD_DIM}x${NUM_HEADS}_ffn${FFN_DIM}.log"

              echo "Running: ${LOG_FILE}"

              if python "${SCRIPT}" \
                --seqlen "${SEQLEN}" \
                --kv-cache-frames "${KV_CACHE_FRAMES}" \
                --chunk-size "${CS}" \
                --chunk-idx "${CHUNK_IDX}" \
                --batch-size "${BS}" \
                --head-dim "${HEAD_DIM}" \
                --num-heads "${NUM_HEADS}" \
                --ffn-dim "${FFN_DIM}" \
                --text-len "${TEXTLEN}" \
                --csv-path "${CSV_PATH}" \
                > "${LOG_FILE}" 2>&1
              then
                echo "[OK] ${LOG_FILE}"
              else
                echo "[FAILED] ${LOG_FILE}"
                FAIL_COUNT=$((FAIL_COUNT + 1))
              fi

            done
          done
        done
      done
    done
  done
done

echo "All profiling runs finished."
echo "Logs saved to ${LOG_DIR}"
echo "CSV saved to ${CSV_PATH}"
echo "Failed runs: ${FAIL_COUNT}"