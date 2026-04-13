#!/bin/bash
set -e

SCRIPT="examples/inference/profiling/basic_self_forcing_causal_profiling.py"

LOG_DIR=logs
mkdir -p "$LOG_DIR"

for bs in 1 2; do
  for num_gpus in 8 4 2 1; do

    if [[ "$num_gpus" -eq 1 ]]; then
      log_file="$LOG_DIR/bs${bs}_gpus${num_gpus}_fsdp0.log"
      python "$SCRIPT" --num_gpus "$num_gpus" --bs "$bs" 2>&1 | tee "$log_file"
    else
      log_file="$LOG_DIR/bs${bs}_gpus${num_gpus}_fsdp0.log"
      python "$SCRIPT" --num_gpus "$num_gpus" --bs "$bs" 2>&1 | tee "$log_file"

      log_file="$LOG_DIR/bs${bs}_gpus${num_gpus}_fsdp1.log"
      python "$SCRIPT" --fsdp --num_gpus "$num_gpus" --bs "$bs" 2>&1 | tee "$log_file"
    fi
  done
done