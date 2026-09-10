"""Independent BF16 weighted RMS reductions, including rounding-boundary inputs."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

torch.set_num_threads(1)
tensors = {}
for width in [32, 768, 1280, 2560]:
    x = (torch.sin(torch.arange(17 * width, dtype=torch.float32) * .173 + .73) * 3.1).reshape(17, width).bfloat16()
    weight = (torch.cos(torch.arange(width, dtype=torch.float32) * .127 + .23) * .19 + .8).bfloat16()
    work = x.float()
    variance = work.square().mean(-1, keepdim=True)
    inverse = torch.rsqrt(variance + 1e-6)
    tensors[f'{width}.input'] = x
    tensors[f'{width}.weight'] = weight
    tensors[f'{width}.variance'] = variance
    tensors[f'{width}.inverse'] = inverse
    tensors[f'{width}.output'] = ((work * inverse) * weight.float()).bfloat16()
path = Path(__file__).with_suffix('.safetensors')
save_file(tensors, path)
path.with_suffix('.json').write_text(json.dumps({
    'torch': torch.__version__, 'torch_git': torch.version.git_version,
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'operation': 'weighted RMS normalization with FP32 reduction and BF16 output',
}, indent=2) + '\n')
