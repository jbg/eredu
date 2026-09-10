"""Independent FP32 sigmoid scores and normalized BF16 mixture coefficients."""
from pathlib import Path
import hashlib
import json
import torch
from safetensors.torch import save_file

torch.set_num_threads(1)
x = (torch.arange(6400, dtype=torch.float32) / 256 - 12.5).bfloat16().float().reshape(64, 100)
scores = x.sigmoid()
ids = (torch.arange(8)[None, :] * 11 + torch.arange(64)[:, None] * 3) % 100
selected = scores.gather(1, ids)
coefficients = (selected / selected.sum(-1, keepdim=True) * 2.5).bfloat16()
p = Path(__file__).with_suffix('.safetensors')
save_file({'input': x, 'scores': scores, 'ids': ids.int(), 'coefficients': coefficients}, p)
p.with_suffix('.json').write_text(json.dumps({
    'torch': torch.__version__, 'torch_git': torch.version.git_version,
    'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
    'operation': 'FP32 sigmoid; selected sum normalization; scale 2.5; BF16 coefficient rounding',
}, indent=2) + '\n')
