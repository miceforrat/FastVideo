#!/usr/bin/env bash

set -uo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "Usage: $0 LOG_DIR [OUTPUT_PREFIX]" >&2
    exit 2
fi

LOG_DIR="${1%/}"

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "Log directory does not exist: ${LOG_DIR}" >&2
    exit 2
fi

if [[ "$#" -eq 2 ]]; then
    OUTPUT_PREFIX="$2"
else
    LOG_PARENT="$(dirname -- "${LOG_DIR}")"
    LOG_BASENAME="$(basename -- "${LOG_DIR}")"
    OUTPUT_PREFIX="${LOG_PARENT}/${LOG_BASENAME}_results"
fi

CSV_FILE="${OUTPUT_PREFIX}.csv"
TSV_FILE="${OUTPUT_PREFIX}.tsv"

shopt -s nullglob
log_files=("${LOG_DIR}"/chunk*_sm*.log)
shopt -u nullglob

if [[ "${#log_files[@]}" -eq 0 ]]; then
    echo "No chunk*_sm*.log files found in ${LOG_DIR}" >&2
    exit 2
fi

rows=()
failed_logs=0

for log_file in "${log_files[@]}"; do
    base_name="$(basename -- "${log_file}")"

    if [[ ! "${base_name}" =~ ^chunk([0-9]+)_sm([0-9]+)\.log$ ]]; then
        echo "Skip unexpected filename: ${base_name}" >&2
        continue
    fi

    chunk_idx="${BASH_REMATCH[1]}"
    requested_dit_sms="${BASH_REMATCH[2]}"

    dit_sms="$(
        sed -n 's/.*actual_dit=\([0-9][0-9]*\).*/\1/p' \
            "${log_file}" | head -n 1
    )"
    vae_sms="$(
        sed -n 's/.*actual_vae=\([0-9][0-9]*\).*/\1/p' \
            "${log_file}" | head -n 1
    )"
    dit_alone="$(
        sed -n 's/^Green DiT eager alone: \([^ ]*\) ms$/\1/p' \
            "${log_file}" | head -n 1
    )"
    vae_graph="$(
        sed -n 's/^Green VAE graph alone: \([^ ]*\) ms$/\1/p' \
            "${log_file}" | head -n 1
    )"
    dit_colocated="$(
        sed -n 's/^DiT with VAE graph: \([^ ]*\) ms$/\1/p' \
            "${log_file}" | head -n 1
    )"
    vae_colocated="$(
        sed -n 's/^VAE graph colocated: \([^ ]*\) ms$/\1/p' \
            "${log_file}" | head -n 1
    )"
    full_sm_dit="$(
        sed -n 's/^Full-SM sequential DiT: \([^ ]*\) ms$/\1/p' \
            "${log_file}" | head -n 1
    )"
    full_sm_vae="$(
        sed -n 's/^Full-SM sequential VAE: \([^ ]*\) ms$/\1/p' \
            "${log_file}" | head -n 1
    )"
    full_sm_sequential="$(
        sed -n 's/^Full-SM sequential total: \([^ ]*\) ms$/\1/p' \
            "${log_file}" | head -n 1
    )"

    if [[ -z "${dit_sms}" || -z "${vae_sms}" \
        || -z "${dit_alone}" || -z "${vae_graph}" \
        || -z "${dit_colocated}" || -z "${vae_colocated}" ]]; then
        echo "Missing fields in ${log_file}" >&2
        failed_logs=$((failed_logs + 1))
        continue
    fi

    if [[ "${requested_dit_sms}" != "${dit_sms}" ]]; then
        echo "Notice: ${base_name} requested ${requested_dit_sms} SM, " \
            "actual split is ${dit_sms} SM" >&2
    fi

    rows+=(
        "${chunk_idx},${dit_sms},${vae_sms},${dit_alone},${vae_graph},${dit_colocated},${vae_colocated},${full_sm_dit},${full_sm_vae},${full_sm_sequential}"
    )
done

if [[ "${#rows[@]}" -eq 0 ]]; then
    echo "No complete log records were found" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${OUTPUT_PREFIX}")"

{
    printf '%s\n' \
        'chunk_idx,dit_sms,vae_sms,dit_alone_ms,vae_graph_ms,dit_colocated_ms,vae_colocated_ms,full_sm_dit_ms,full_sm_vae_ms,full_sm_sequential_ms'
    printf '%s\n' "${rows[@]}" | sort -t, -k1,1n -k2,2n
} > "${CSV_FILE}"

awk -F, 'BEGIN { OFS="\t" } { print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10 }' \
    "${CSV_FILE}" > "${TSV_FILE}"

echo "Parsed logs: ${#rows[@]}"
echo "Incomplete logs: ${failed_logs}"
echo "CSV: ${CSV_FILE}"
echo "TSV: ${TSV_FILE}"

if [[ "${failed_logs}" -ne 0 ]]; then
    exit 1
fi
