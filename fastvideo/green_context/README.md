# FastVideo Green Context runtime

This package contains the Green Context runtime used by FastVideo pipelines.
The historical implementation under `green+pytorch_demo/` remains unchanged.

Build the extension inside the FastVideo container:

```bash
cd /FastVideo/fastvideo/green_context
/opt/venv/bin/python setup.py build_ext --inplace
```

Then verify that the extension can be imported without creating contexts:

```bash
cd /FastVideo
/opt/venv/bin/python -c \
  'from fastvideo.green_context import GreenContextPairPool; print(GreenContextPairPool)'
```

Creating a pool provisions GPU resources immediately:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python -c '
from fastvideo.green_context import GreenContextPairPool
pool = GreenContextPairPool([104, 112], ignore_sm_coscheduling=True)
for sms in pool.requested_dit_sm_counts:
    pair = pool[sms]
    print(sms, pair.actual_dit_sms, pair.actual_vae_sms, pair.total_sms)
pool.synchronize()
'
```

Keep the pool alive for at least as long as any returned external stream is in
use. Destroying its native Green Context objects detaches those streams.
