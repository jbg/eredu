"""Independent selected BF16 projections with cancellation and repeated groups."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

torch.set_num_threads(1)
torch.manual_seed(937)
ids = torch.tensor([2, 0, 4, 2, 1, 0, 3, 4], dtype=torch.int32)
tensors = {'ids': ids}
for width in [128, 768, 2560]:
    inputs = torch.randn(8, width).bfloat16()
    weights = (torch.randn(5, 64, width) * .13).bfloat16()
    # Four strided accumulators lose the unit term; the reference row-dot
    # reduction preserves it. Repeated group 0 must use the bank's row layout.
    inputs[1].fill_(1)
    inputs[5].fill_(1)
    weights[0, 0].zero_()
    weights[0, 0, 0] = 2 ** 24
    weights[0, 0, 4] = 1
    weights[0, 0, 32] = -(2 ** 24)
    outputs = torch.stack([inputs[i] @ weights[int(group)].T for i, group in enumerate(ids)])
    tensors.update({f'{width}.input': inputs, f'{width}.weight': weights, f'{width}.output': outputs})
path = Path(__file__).with_suffix('.safetensors')
save_file(tensors, path)
path.with_suffix('.json').write_text(json.dumps({
    'torch': torch.__version__, 'torch_git': torch.version.git_version,
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'operation': 'selected BF16 row projections; FP32 accumulation; seed 937',
}, indent=2) + '\n')
