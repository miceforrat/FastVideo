# profile_qwen3_prefill.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-8B"
SEQ_LEN = 4680
WARMUP = 3

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).eval()

config = model.config

num_heads = getattr(config, "num_attention_heads", None)
num_kv_heads = getattr(config, "num_key_value_heads", None)
hidden_size = getattr(config, "hidden_size", None)
head_dim = getattr(config, "head_dim", None)

if head_dim is None and hidden_size is not None and num_heads is not None:
    head_dim = hidden_size // num_heads

print("Model loaded.")
print(f"hidden_size = {hidden_size}")
print(f"num_attention_heads = {num_heads}")
print(f"num_key_value_heads = {num_kv_heads}")
print(f"head_dim = {head_dim}")
print(f"seq_len = {SEQ_LEN}")
print(f"dtype = {next(model.parameters()).dtype}")
print(f"attn_impl = {getattr(config, '_attn_implementation', None)}")

vocab_size = config.vocab_size

input_ids = torch.randint(
    low=100,
    high=min(vocab_size - 1, 30000),
    size=(1, SEQ_LEN),
    device="cuda",
    dtype=torch.long,
)

attention_mask = torch.ones_like(input_ids)

for _ in range(WARMUP):
    with torch.inference_mode():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

torch.cuda.synchronize()
print("Warmup finished.")

torch.cuda.nvtx.range_push("qwen3_prefill")

with torch.inference_mode():
    _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("Finished.")