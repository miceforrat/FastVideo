#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_ID="${GPU_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WARMUP_ITERS="${WARMUP_ITERS:-3}"
PROFILE_ITERS="${PROFILE_ITERS:-9}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/logs/chunk0_6_best_pm8_step2}"
RERUN_COMPLETED="${RERUN_COMPLETED:-0}"

# Fill in the best DiT SM count found by the coarse sweep for each chunk.
# Each chunk scans [best - 8, best + 8] with a step size of 2.
BEST_DIT_SMS=(
    "96"  # chunk 0 (cold VAE)
    "88"  # chunk 1
    "96"  # chunk 2
    "104"  # chunk 3
    "104"  # chunk 4
    "112"  # chunk 5
    "112"  # chunk 6
)

for chunk_idx in {0..6}; do
    best_sms="${BEST_DIT_SMS[${chunk_idx}]}"
    if [[ ! "${best_sms}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: set BEST_DIT_SMS[${chunk_idx}] before running." >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILE="${OUTPUT_DIR}/run_status.tsv"
printf 'chunk_idx\tbest_dit_sms\trequested_dit_sms\tvae_mode\tstatus\texit_code\tlog_file\n' \
    > "${SUMMARY_FILE}"

total_runs=0
completed_runs=0
skipped_runs=0
failed_runs=0

for chunk_idx in {0..6}; do
    best_sms="${BEST_DIT_SMS[${chunk_idx}]}"
    start_sms=$((best_sms - 8))
    end_sms=$((best_sms + 8))

    for dit_sms in $(seq "${start_sms}" 2 "${end_sms}"); do
        total_runs=$((total_runs + 1))
        log_file="${OUTPUT_DIR}/chunk${chunk_idx}_sm${dit_sms}.log"
        vae_args=()
        vae_mode="steady"

        if [[ "${chunk_idx}" -eq 0 ]]; then
            vae_args+=(--vae-first-chunk)
            vae_mode="cold_first_chunk"
        fi

        if [[ "${RERUN_COMPLETED}" != "1" ]] \
            && [[ -f "${log_file}" ]] \
            && grep -q '^parallel wall:' "${log_file}"; then
            echo "[skip] chunk=${chunk_idx} dit_sms=${dit_sms} vae=${vae_mode}"
            printf '%s\t%s\t%s\t%s\tskipped\t0\t%s\n' \
                "${chunk_idx}" "${best_sms}" "${dit_sms}" "${vae_mode}" \
                "${log_file}" >> "${SUMMARY_FILE}"
            skipped_runs=$((skipped_runs + 1))
            continue
        fi

        echo "============================================================"
        echo "chunk=${chunk_idx} best=${best_sms} dit_sms=${dit_sms} vae=${vae_mode}"
        echo "log=${log_file}"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" vae_dit_testbench_cuda_graph.py \
            --chunk-idx "${chunk_idx}" \
            --dit-sms "${dit_sms}" \
            --ignore-sm-coscheduling \
            --warmup-iters "${WARMUP_ITERS}" \
            --profile-iters "${PROFILE_ITERS}" \
            "${vae_args[@]}" \
            2>&1 | tee "${log_file}"

        exit_code=${PIPESTATUS[0]}

        if [[ "${exit_code}" -eq 0 ]] \
            && grep -q '^parallel wall:' "${log_file}"; then
            status="completed"
            completed_runs=$((completed_runs + 1))
        else
            status="failed"
            failed_runs=$((failed_runs + 1))
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${chunk_idx}" "${best_sms}" "${dit_sms}" "${vae_mode}" \
            "${status}" "${exit_code}" "${log_file}" \
            >> "${SUMMARY_FILE}"
    done
done

echo "============================================================"
echo "Fine sweep finished"
echo "total=${total_runs}"
echo "completed=${completed_runs}"
echo "skipped=${skipped_runs}"
echo "failed=${failed_runs}"
echo "status_file=${SUMMARY_FILE}"
echo "============================================================"

if [[ "${failed_runs}" -ne 0 ]]; then
    exit 1
fi
