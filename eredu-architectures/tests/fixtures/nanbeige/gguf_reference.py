"""Independent packed GGUF fixture and exact dequantized SafeTensors oracle.

Uses the publisher's Nanbeige metadata, physical block numbering and Q/K row
permutation. The float oracle reverses that permutation after dequantization.
Requires numpy and safetensors; run with --output DIR --bits 4 or 8.
"""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file


parser = argparse.ArgumentParser(
    description='Write packed GGUF and dequantized SafeTensors fixtures.'
)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--bits', type=int, choices=[4, 8], default=8)
args = parser.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=True)
bits = args.bits
config = {
    'model_type': 'nanbeige',
    'architectures': ['NanbeigeForCausalLM'],
    'hidden_size': 32,
    'num_hidden_layers': 2,
    'num_loops': 2,
    'intermediate_size': 64,
    'num_attention_heads': 8,
    'num_key_value_heads': 2,
    'head_dim': 8,
    'vocab_size': 64,
    'rms_norm_eps': 1e-05,
    'rope_theta': 70000000.0,
    'max_position_embeddings': 1024,
    'tie_word_embeddings': False,
}
shapes = {
    'model.embed_tokens.weight': (64, 32),
    'model.norm.weight': (32,),
    'lm_head.weight': (64, 32),
}
for l in range(2):
    p = f'model.layers.{l}.'
    block_shapes = {
        'input_layernorm.weight': (32,),
        'post_attention_layernorm.weight': (32,),
        'self_attn.q_proj.weight': (64, 32),
        'self_attn.k_proj.weight': (16, 32),
        'self_attn.v_proj.weight': (16, 32),
        'self_attn.o_proj.weight': (32, 64),
        'mlp.gate_proj.weight': (64, 32),
        'mlp.up_proj.weight': (64, 32),
        'mlp.down_proj.weight': (32, 64),
    }
    for k, shape in block_shapes.items():
        shapes[p + k] = shape
replace = {
    'model.embed_tokens': 'token_embd',
    'model.norm': 'output_norm',
    'lm_head': 'output',
    'model.layers.': 'blk.',
    'self_attn.q_proj': 'attn_q',
    'self_attn.k_proj': 'attn_k',
    'self_attn.v_proj': 'attn_v',
    'self_attn.o_proj': 'attn_output',
    'mlp.gate_proj': 'ffn_gate',
    'mlp.up_proj': 'ffn_up',
    'mlp.down_proj': 'ffn_down',
    'input_layernorm': 'attn_norm',
    'post_attention_layernorm': 'ffn_norm',
}
physical = {}
oracle = {}
tensors = []
for name, shape in shapes.items():
    values = [
        1.0 + i % 7 * 0.001 if 'norm' in name
        else ((i * 17 + len(name) * 7) % 101 - 50) * 0.003
        for i in range(np.prod(shape))
    ]
    a = np.array(values, dtype=np.float32).reshape(shape)
    physical[name] = a.copy()
    permuted = name.endswith(('q_proj.weight', 'k_proj.weight'))
    if permuted:
        a = a.reshape(-1, 2, 4, shape[1]).transpose(0, 2, 1, 3).reshape(shape)
    if a.ndim == 2:
        parts = []
        deq = []
        for block in a.reshape(-1, 32):
            bound = 127 if bits == 8 else 7
            d = np.float16(np.abs(block).max() / bound)
            q = np.clip(np.rint(block / np.float32(d)), -bound, bound).astype(np.int8)
            if bits == 8:
                packed = q.tobytes()
            else:
                nibbles = (q + 8).astype(np.uint8)
                packed = (nibbles[:16] | nibbles[16:] << 4).tobytes()
            parts.append(d.tobytes() + packed)
            deq.append(q.astype(np.float32) * np.float32(d))
        data = b''.join(parts)
        dtype = 8 if bits == 8 else 2
        b = np.concatenate(deq).reshape(shape)
    else:
        data = a.tobytes()
        dtype = 0
        b = a
    if permuted:
        b = b.reshape(-1, 4, 2, shape[1]).transpose(0, 2, 1, 3).reshape(shape)
    oracle[name] = b.copy()
    ggname = name
    for x, y in replace.items():
        ggname = ggname.replace(x, y)
    tensors.append((ggname, shape[::-1], dtype, data))
for label, weights in [('tiny-safe', physical), (f'tiny-q{bits}-oracle', oracle)]:
    p = root / label
    p.mkdir(exist_ok=True)
    (p / 'config.json').write_text(json.dumps(config))
    save_file(weights, str(p / 'model.safetensors'))
metadata = {
    'general.architecture': 'nanbeige',
    'nanbeige.embedding_length': 32,
    'nanbeige.block_count': 2,
    'nanbeige.num_loops': 2,
    'nanbeige.skip_loop_final_norm': False,
    'nanbeige.attention.head_count': 8,
    'nanbeige.attention.head_count_kv': 2,
    'nanbeige.attention.key_length': 8,
    'nanbeige.feed_forward_length': 64,
    'nanbeige.attention.layer_norm_rms_epsilon': 1e-05,
    'nanbeige.rope.freq_base': 70000000.0,
    'nanbeige.vocab_size': 64,
    'nanbeige.context_length': 1024,
}

def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


out = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
for k, v in metadata.items():
    out += string(k)
    if isinstance(v, str):
        out += struct.pack('<I', 8) + string(v)
    elif isinstance(v, bool):
        out += struct.pack('<I?', 7, v)
    elif isinstance(v, int):
        out += struct.pack('<II', 4, v)
    else:
        out += struct.pack('<If', 6, v)
offset = 0
data = bytearray()
for name, shape, dtype, payload in tensors:
    out += string(name) + struct.pack('<I', len(shape))
    out += struct.pack('<' + 'Q' * len(shape), *shape)
    out += struct.pack('<IQ', dtype, offset)
    data += payload
    data += b'\x00' * (-len(data) % 32)
    offset = len(data)
out += b'\x00' * (-len(out) % 32)
out += data
(root / f'tiny-q{bits}.gguf').write_bytes(out)
