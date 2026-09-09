"""Independent scalar Gemma 2 oracle: four norms, capped GQA, GELU and tied head.
Synthetic nonzero named weights; no Eredu/native runtime imports.
Regenerate: python3 eredu-architectures/tests/fixtures/gemma2/reference.py
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
    w = weight(name + '.weight', 8, False)
    return [[x * (1 + w[i]) / math.sqrt(sum(y*y for y in row)/8 + 1e-5)
             for i, x in enumerate(row)] for row in xs]


def rope(v, position):
    out = v[:]
    for i in range(2):
        angle = position / (10000 ** (2*i/4))
        c, s = math.cos(angle), math.sin(angle)
        out[i] = v[i]*c-v[i+2]*s
        out[i+2] = v[i+2]*c+v[i]*s
    return out


def add(a,b):
    return [[x+y for x,y in zip(xs,ys)] for xs,ys in zip(a,b)]


def forward(tokens, scalar=7.0, cap=0.3, output_cap=0.7):
    table = weight('model.embed_tokens.weight', 17*8)
    xs = [[x*math.sqrt(8) for x in table[t*8:(t+1)*8]] for t in tokens]
    for layer in range(4):
        root = f'model.layers.{layer}'
        x = norm(xs, root+'.input_layernorm')
        q,k,v = [linear(x, root+'.self_attn.'+field, width) for field,width in
                 [('q_proj',8),('k_proj',4),('v_proj',4)]]
        q = [[rope(row[h:h+4],pos) for h in range(0,8,4)] for pos,row in enumerate(q)]
        k = [rope(row,pos) for pos,row in enumerate(k)]
        attended=[]
        for pos in range(len(xs)):
            row=[]
            start=max(0,pos-1) if layer%2==0 else 0
            for head in range(2):
                scores=[sum(a*b for a,b in zip(q[pos][head],k[t]))/math.sqrt(scalar)
                        for t in range(start,pos+1)]
                if cap is not None: scores=[cap*math.tanh(s/cap) for s in scores]
                probs=[math.exp(s-max(scores)) for s in scores]
                z=sum(probs)
                row.extend(sum(probs[t-start]/z*v[t][i] for t in range(start,pos+1)) for i in range(4))
            attended.append(row)
        xs=add(xs,norm(linear(attended,root+'.self_attn.o_proj',8),root+'.post_attention_layernorm'))
        x=norm(xs,root+'.pre_feedforward_layernorm')
        gate,up=[linear(x,root+'.mlp.'+field,12) for field in ['gate_proj','up_proj']]
        activated=[[0.5*g*(1+math.tanh(math.sqrt(2/math.pi)*(g+0.044715*g**3)))*u
                    for g,u in zip(gs,us)] for gs,us in zip(gate,up)]
        xs=add(xs,norm(linear(activated,root+'.mlp.down_proj',8),root+'.post_feedforward_layernorm'))
    xs=norm(xs,'model.norm')
    logits=[[sum(x[i]*table[t*8+i] for i in range(8)) for t in range(17)] for x in xs]
    return [[output_cap*math.tanh(v/output_cap) if output_cap else v for v in row] for row in logits]


if __name__ == '__main__':
    Path(__file__).with_name('logits.json').write_text(json.dumps(forward([1,3,2,4,5,6,7]),indent=2)+'\n')
