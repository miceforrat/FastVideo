#!/bin/bash

set -e


CHUNKS=(0 1 2 3 4 5 6)

SMS_LIST=(
    17
    34
    51
    68
    85
    102
    119
    136
    153
    170
)


for chunk in "${CHUNKS[@]}"
do

    for sms in "${SMS_LIST[@]}"
    do

        echo "======================================"
        echo "Running:"
        echo "  chunk_idx = ${chunk}"
        echo "  num_sms   = ${sms}"
        echo "======================================"


        python examples/inference/profiling/profile_dit_chunk_denoising_and_gen_kv.py \
            --chunk-idx ${chunk} \
            --num-sms ${sms} \
            --warmup-iters 5 \
            --profile-iters 10


        echo ""
        echo "Finished chunk=${chunk}, sms=${sms}"
        echo ""

    done

done


echo "All experiments finished."