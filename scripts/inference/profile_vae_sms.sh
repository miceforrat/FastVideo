for sms in 17 34 51 68 85 102 119 136 153 170
do
    python examples/inference/profiling/profile_vae_one_chunk_green_ctx.py  \
        --num-sms $sms
done