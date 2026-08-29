#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_ID="${GPU_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WARMUP_ITERS="${WARMUP_ITERS:-3}"
PROFILE_ITERS="${PROFILE_ITERS:-5}"
PREVIOUS_REQUEST_LAST_CHUNK="${PREVIOUS_REQUEST_LAST_CHUNK:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/logs/real_chunk_pairing_best_pm8_step2}"
RERUN_COMPLETED="${RERUN_COMPLETED:-0}"

# Fill in the best DiT SM count from the 8-SM coarse sweep for each DiT
# chunk. Each chunk will scan [best - 8, best + 8] with a step size of 2.
BEST_DIT_SMS=(
    "80"  # DiT chunk 0, paired with previous-request VAE tail
    "104"  # DiT chunk 1, paired with VAE chunk 0 (cold)
    "96"  # DiT chunk 2, paired with VAE chunk 1
    "104"  # DiT chunk 3, paired with VAE chunk 2
    "104"  # DiT chunk 4, paired with VAE chunk 3
    "112"  # DiT chunk 5, paired with VAE chunk 4
    "112"  # DiT chunk 6, paired with VAE chunk 5
)

for dit_chunk_idx in 0 1 2 3 4 5 6; do
    best_sms="${BEST_DIT_SMS[${dit_chunk_idx}]}"
    if [[ ! "${best_sms}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: set BEST_DIT_SMS[${dit_chunk_idx}] before running." >&2
        exit 2
    fi

    if (( best_sms - 8 <= 0 )); then
        echo "ERROR: BEST_DIT_SMS[${dit_chunk_idx}]=${best_sms} gives an invalid lower bound." >&2
        exit 2
    fi

    if (( best_sms + 8 >= 170 )); then
        echo "ERROR: BEST_DIT_SMS[${dit_chunk_idx}]=${best_sms} gives an invalid upper bound." >&2
        exit 2
    fi
done

mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILE="${OUTPUT_DIR}/run_status.tsv"
printf 'dit_chunk_idx\tvae_chunk_idx\tbest_dit_sms\trequested_dit_sms\tvae_mode\tstatus\texit_code\tlog_file\n' \
    > "${SUMMARY_FILE}"

total_runs=0
completed_runs=0
skipped_runs=0
failed_runs=0

for dit_chunk_idx in 0 1 2 3 4 5 6; do
    best_sms="${BEST_DIT_SMS[${dit_chunk_idx}]}"
    start_sms=$((best_sms - 8))
    end_sms=$((best_sms + 8))

    if [[ "${dit_chunk_idx}" -eq 0 ]]; then
        vae_chunk_idx="${PREVIOUS_REQUEST_LAST_CHUNK}"
    else
        vae_chunk_idx=$((dit_chunk_idx - 1))
    fi

    if [[ "${vae_chunk_idx}" -eq 0 ]]; then
        vae_mode="cold"
    else
        vae_mode="steady"
    fi

    for dit_sms in $(seq "${start_sms}" 2 "${end_sms}"); do
        total_runs=$((total_runs + 1))
        log_file="${OUTPUT_DIR}/chunk${dit_chunk_idx}_sm${dit_sms}.log"

        if [[ "${RERUN_COMPLETED}" != "1" ]] \
            && [[ -f "${log_file}" ]] \
            && grep -q '^parallel wall:' "${log_file}"; then
            echo "[skip] DiT chunk=${dit_chunk_idx} VAE chunk=${vae_chunk_idx} DiT SM=${dit_sms}"
            printf '%s\t%s\t%s\t%s\t%s\tskipped\t0\t%s\n' \
                "${dit_chunk_idx}" "${vae_chunk_idx}" "${best_sms}" \
                "${dit_sms}" "${vae_mode}" "${log_file}" \
                >> "${SUMMARY_FILE}"
            skipped_runs=$((skipped_runs + 1))
            continue
        fi

        echo "============================================================"
        echo "[run ${total_runs}/63]"
        echo "DiT chunk=${dit_chunk_idx}"
        echo "VAE chunk=${vae_chunk_idx}"
        echo "VAE mode=${vae_mode}"
        echo "coarse best DiT SM=${best_sms}"
        echo "requested DiT SM=${dit_sms}"
        echo "log=${log_file}"
        echo "============================================================"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" vae_dit_real_chunk_colocation_testbench.py \
            --chunk-idx "${dit_chunk_idx}" \
            --previous-request-last-chunk "${PREVIOUS_REQUEST_LAST_CHUNK}" \
            --dit-sms "${dit_sms}" \
            --ignore-sm-coscheduling \
            --warmup-iters "${WARMUP_ITERS}" \
            --profile-iters "${PROFILE_ITERS}" \
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

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${dit_chunk_idx}" "${vae_chunk_idx}" "${best_sms}" \
            "${dit_sms}" "${vae_mode}" "${status}" "${exit_code}" \
            "${log_file}" >> "${SUMMARY_FILE}"
    done
done

echo "============================================================"
echo "Real chunk fine sweep finished"
echo "total=${total_runs}"
echo "completed=${completed_runs}"
echo "skipped=${skipped_runs}"
echo "failed=${failed_runs}"
echo "status_file=${SUMMARY_FILE}"
echo "============================================================"

if [[ "${failed_runs}" -ne 0 ]]; then
    exit 1
fi
