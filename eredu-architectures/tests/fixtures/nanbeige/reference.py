"""Independent scalar oracle for the released dense Nanbeige loop equations.

Source: Nanbeige/Nanbeige4.2-3B at
0e137298720f7241e83b8aabecc4263dcc7d84b3/modeling_nanbeige.py.
Uses no Eredu code or native backend. Run with Python 3 to regenerate logits.json.
Weights match the deterministic named-parameter fixture in reference_numeric.rs.
This is a tiny synthetic checkpoint, not logits from the released 3B weights.
"""
import json
import math
from pathlib import Path
import struct


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def weight(name, size, norm=False):
    h = 2166136261
    for byte in name.encode():
        h = ((h ^ byte) * 16777619) & 0xffffffff
    values = []
    for i in range(size):
        x = (h + i * 747796405) & 0xffffffff
        shift = i % 23
        x = ((x << shift) | (x >> (32 - shift))) & 0xffffffff
        centered = x % 2001 - 1000
        values.append(f32(1 + f32(centered / 50000)) if norm else f32(centered / 25000))
    return values


def linear(xs, name, out):
    n = len(xs[0])
    w = weight(name + '.weight', out * n)
    return [[sum(x[k] * w[j*n+k] for k in range(n)) for j in range(out)] for x in xs]


def norm(xs, name):
    n = len(xs[0])
    w = weight(name + '.weight', n, True)
    return [[v * w[i] / math.sqrt(sum(t*t for t in x) / n + 1e-5)
             for i, v in enumerate(x)] for x in xs]


def add(xs, ys):
    return [[a+b for a, b in zip(x, y)] for x, y in zip(xs, ys)]


def rope(vector, position):
    # Non-interleaved split-half RoPE; head_dim=4, theta=70 million.
    result = vector.copy()
    for i in range(2):
        angle = position / (70000000 ** (2*i / 4))
        c, s = math.cos(angle), math.sin(angle)
        result[i] = vector[i]*c - vector[i+2]*s
        result[i+2] = vector[i+2]*c + vector[i]*s
    return result


def block(xs, layer):
    prefix = f'model.layers.{layer}'
    x = norm(xs, prefix + '.input_layernorm')
    q = linear(x, prefix + '.self_attn.q_proj', 16)
    k = linear(x, prefix + '.self_attn.k_proj', 8)
    v = linear(x, prefix + '.self_attn.v_proj', 8)
    q = [[rope(row[h:h+4], pos) for h in range(0, 16, 4)] for pos, row in enumerate(q)]
    k = [[rope(row[h:h+4], pos) for h in range(0, 8, 4)] for pos, row in enumerate(k)]
    attended = []
    for pos in range(len(xs)):
        row = []
        for head in range(4):
            kv = head // 2
            scores = [sum(a*b for a, b in zip(q[pos][head], k[t][kv])) / 2
                      for t in range(pos+1)]
            probs = [math.exp(s-max(scores)) for s in scores]
            z = sum(probs)
            row.extend(sum(p / z * v[t][kv*4+i] for t, p in enumerate(probs)) for i in range(4))
        attended.append(row)
    xs = add(xs, linear(attended, prefix + '.self_attn.o_proj', 8))
    x = norm(xs, prefix + '.post_attention_layernorm')
    gate = linear(x, prefix + '.mlp.gate_proj', 12)
    up = linear(x, prefix + '.mlp.up_proj', 12)
    activated = [[g / (1 + math.exp(-g)) * u for g, u in zip(gs, us)] for gs, us in zip(gate, up)]
    return add(xs, linear(activated, prefix + '.mlp.down_proj', 8))


def forward(tokens, loops, skip_norm):
    table = weight('model.embed_tokens.weight', 17*8)
    xs = [table[t*8:(t+1)*8] for t in tokens]
    for repeat in range(loops):
        for layer in range(2):
            xs = block(xs, layer)
        if not skip_norm or repeat == loops-1:
            xs = norm(xs, 'model.norm')
    return linear(xs, 'lm_head', 17)


if __name__ == '__main__':
    result = {f'{loops}_{skip}': forward([1, 3, 2, 4, 5], loops, skip)
              for loops in [1, 2, 3] for skip in [False, True]}
    Path(__file__).with_name('logits.json').write_text(json.dumps(result, indent=2) + '\n')
