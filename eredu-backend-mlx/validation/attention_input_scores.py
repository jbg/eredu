"""Independent eager attention fixture with explicit BF16 rounding boundaries."""
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(1)
scale = 1 / math.sqrt(128)
def tensor(shape, phase):
    count = 1
    for size in shape:
        count *= size
    return (torch.sin(torch.arange(count, dtype=torch.float32) * .37 + phase) * 2.5).reshape(shape).bfloat16()

q, k, v = tensor((1,4,3,4), .1), tensor((1,2,5,4), .7), tensor((1,2,5,6), 1.2)
mask = torch.arange(5)[None,:] <= (torch.arange(3) + 2)[:,None]
scores = (q @ k.repeat_interleave(2,dim=1).transpose(-1,-2)) * scale
scores = scores.masked_fill(~mask, float('-inf'))
probabilities = torch.softmax(scores.float(), dim=-1).bfloat16()
result = probabilities @ v.repeat_interleave(2,dim=1)
fixture = {name: dict(shape=list(value.shape), values=value.float().flatten().tolist())
           for name,value in [('queries',q),('keys',k),('values',v),('output',result)]}
fixture['torch'] = torch.__version__
fixture['scale'] = scale
q, k, v = tensor((1,4,33,4), .1), tensor((1,2,257,4), .7), tensor((1,2,257,4), 1.2)
# Broadcast key masking deliberately permits future keys: this exercises the
# contiguous bounded path's explicit-mask semantics independently of causality.
mask = torch.arange(257) % 7 != 0
scores = (q @ k.repeat_interleave(2,dim=1).transpose(-1,-2)) * scale
scores = scores.masked_fill(~mask, float('-inf'))
result = torch.softmax(scores.float(),dim=-1).bfloat16() @ v.repeat_interleave(2,dim=1)
fixture['bounded'] = {name: dict(shape=list(value.shape),values=value.float().flatten().tolist())
                      for name,value in [('queries',q),('keys',k),('values',v),('output',result)]}
Path(__file__).with_suffix('.json').write_text(json.dumps(fixture, indent=2)+'\n')
