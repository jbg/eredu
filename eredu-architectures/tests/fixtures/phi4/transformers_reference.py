"""Generate independent Phi-4 SafeTensors and GGUF fixtures with cached logits.

Requires torch, transformers==4.48.3, safetensors and gguf==0.19.0.
Usage: python transformers_reference.py /private/tmp/eredu-phi4-validation
"""

import json
import pathlib
import sys

import gguf
import numpy as np
import torch
import transformers
from safetensors.torch import save_file
from transformers import Phi3Config, Phi3ForCausalLM

root = pathlib.Path(sys.argv[1])
artifact = root / "synthetic-transformers"
artifact.mkdir(parents=True, exist_ok=True)
config = Phi3Config(
    hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
    max_position_embeddings=64, original_max_position_embeddings=64,
    rope_theta=250000.0, rms_norm_eps=1e-5, sliding_window=None,
    tie_word_embeddings=False, bos_token_id=0, eos_token_id=63, pad_token_id=63,
)
config._attn_implementation = "eager"
torch.manual_seed(20261013)
model = Phi3ForCausalLM(config).float().eval()
with torch.no_grad():
    for name, parameter in model.named_parameters():
        values = torch.randn_like(parameter) * 0.07
        if "norm" in name:
            values += 1.0
        parameter.copy_(values)
model.save_pretrained(artifact, safe_serialization=True)

tokens = [1, 3, 2, 4, 5, 6, 7]
def reference_rows(model, tokens):
    cache = None
    outputs = []
    full_prefix_errors = []
    with torch.no_grad():
        for end in range(3, len(tokens) + 1):
            start = 0 if end == 3 else end - 1
            result = model(torch.tensor([tokens[start:end]]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            outputs.append(result.logits.float().clone())
            full = model(torch.tensor([tokens[:end]]), use_cache=False).logits[:, -1]
            error = float((result.logits[:, -1] - full).abs().max())
            assert error < 1e-5, error
            full_prefix_errors.append(error)
    return outputs, full_prefix_errors


outputs, full_prefix_errors = reference_rows(model, tokens)
save_file(
    {"prefill_logits": outputs[0], "decode_logits": torch.cat(outputs[1:], dim=1)},
    str(root / "synthetic-independent.safetensors"),
)

# Phi GGUF keeps Q/K/V component order and half-head rotary pairs unchanged.
names = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
for layer in range(config.num_hidden_layers):
    for source, target in {
        "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
        "self_attn.qkv_proj": "attn_qkv", "self_attn.o_proj": "attn_output",
        "mlp.gate_up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
    }.items():
        names[f"model.layers.{layer}.{source}.weight"] = f"blk.{layer}.{target}.weight"
for quantized in [False, True]:
    writer = gguf.GGUFWriter(root / ("synthetic-q8.gguf" if quantized else "synthetic.gguf"), "phi3")
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_dimension_count(config.hidden_size // config.num_attention_heads)
    writer.add_rope_freq_base(config.rope_theta)
    writer.add_sliding_window(0)
    dequantized = {}
    for name, value in model.state_dict().items():
        values = value.detach().numpy().astype(np.float32)
        if quantized and values.ndim == 2:
            encoded = gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(names[name], encoded, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            dequantized[name] = torch.from_numpy(gguf.quants.dequantize(encoded, gguf.GGMLQuantizationType.Q8_0))
        else:
            writer.add_tensor(names[name], values)
            dequantized[name] = value
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if quantized:
        model.load_state_dict(dequantized)
        model.save_pretrained(root / "synthetic-dequantized-transformers", safe_serialization=True)
        q8_outputs, _ = reference_rows(model, tokens)
        save_file(
            {"prefill_logits": q8_outputs[0], "decode_logits": torch.cat(q8_outputs[1:], dim=1)},
            str(root / "synthetic-q8-independent.safetensors"),
        )
report = {
    "torch": torch.__version__, "transformers": transformers.__version__,
    "seed": 20261013, "tokens": tokens,
    "greedy_ids": [int(value[0, -1].argmax()) for value in outputs],
    "cached_full_prefix_max_abs_errors": full_prefix_errors,
}
(root / "synthetic-reference.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
