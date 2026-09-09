"""Generate independent cached and full-prefix Gemma 2 logits.

Requires torch==2.14.0, transformers==4.48.3 and safetensors.
Usage: python transformers_reference.py /private/tmp/eredu-gemma2
"""

import json
import pathlib
import sys

import torch
import transformers
from safetensors.torch import save_file
from transformers import Gemma2Config, Gemma2ForCausalLM, HybridCache


destination = pathlib.Path(sys.argv[1])
root = destination / "synthetic-transformers"
root.mkdir(parents=True, exist_ok=True)
config = Gemma2Config(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=64,
    max_position_embeddings=64,
    sliding_window=2,
    query_pre_attn_scalar=7,
    attn_logit_softcapping=0.3,
    final_logit_softcapping=0.7,
    hidden_activation="gelu_pytorch_tanh",
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
)
config._attn_implementation = "eager"
torch.manual_seed(20260909)
model = Gemma2ForCausalLM(config).float().eval()
with torch.no_grad():
    for parameter in model.parameters():
        parameter.copy_(torch.randn_like(parameter) * 0.07)
model.save_pretrained(root, safe_serialization=True)

# Reserve full-context storage and use absolute positions even after sliding
# layers evict old keys. Check the reference cache against fresh full prefixes.
tokens = [1, 3, 2, 4, 5, 6, 7]
cache = HybridCache(
    config=config, max_batch_size=1, max_cache_len=len(tokens),
    device="cpu", dtype=torch.float32,
)
outputs = []
full_prefix_errors = []
with torch.no_grad():
    for end in range(3, len(tokens) + 1):
        start = 0 if end == 3 else end - 1
        output = model(
            torch.tensor([tokens[start:end]]),
            attention_mask=torch.ones(1, end, dtype=torch.long),
            cache_position=torch.arange(start, end),
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        outputs.append(output.logits.float().clone())
        full = model(torch.tensor([tokens[:end]]), use_cache=False).logits[:, -1]
        error = float((output.logits[:, -1] - full).abs().max())
        assert error < 1e-5, error
        full_prefix_errors.append(error)
save_file(
    {"prefill_logits": outputs[0], "decode_logits": torch.cat(outputs[1:], dim=1)},
    str(destination / "independent.safetensors"),
)
report = {
    "torch": torch.__version__, "transformers": transformers.__version__,
    "seed": 20260909, "tokens": tokens,
    "greedy_ids": [int(x[0, -1].argmax()) for x in outputs],
    "cached_full_prefix_max_abs_errors": full_prefix_errors,
}
(root / "reference.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
