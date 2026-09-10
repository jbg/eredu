"""Independent CPU BF16 matrix products across row counts and memory layouts."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

torch.set_num_threads(1)
tensors = {}
for width in [7, 24, 32, 128, 768, 2560]:
    a = (torch.sin(torch.arange(3 * width, dtype=torch.float32) * .193 + .37) * 1.3).reshape(3, width).bfloat16()
    w = (torch.cos(torch.arange(19 * width, dtype=torch.float32) * .117 + .91) * .17).reshape(19, width).bfloat16()
    tensors[f"{width}.input"] = a
    tensors[f"{width}.weight"] = w
    for rows in [1, 2, 3]:
        tensors[f"{width}.output.{rows}"] = a[:rows] @ w.T
path = Path(__file__).with_suffix('.safetensors')
save_file(tensors, path)
path.with_suffix('.json').write_text(json.dumps({
    'torch': torch.__version__, 'torch_git': torch.version.git_version,
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'operation': 'BF16 CPU matrix multiplication, FP32 accumulation',
}, indent=2) + '\n')
