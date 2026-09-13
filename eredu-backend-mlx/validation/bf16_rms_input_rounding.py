"""Extract the small RMS rounding regression from the pinned independent run.

First run component_sparse_reference.py with --all-positions. Pass its JSON
output here; no native backend is used to construct the expected result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument('reference', type=Path)
args = parser.parse_args()
reference = json.loads(args.reference.read_text())
assert reference['provenance']['repository'] == 'LiquidAI/LFM2-8B-A1B'
assert reference['provenance']['revision'] == 'c1c44ff9fc00db3ebf4516970563f5f383d23670'
checkpoint = Path(reference['provenance']['path'])
config = json.loads((checkpoint / 'config.json').read_text())
layer = config['num_dense_layers'] + 1
assert layer == 3 and config['norm_eps'] == 1e-5
step = reference['trials']['deletion'][1]
value = torch.tensor(step['boundaries']['mixer_residual'], dtype=torch.bfloat16).reshape(1, 1, 2048)
expected = torch.tensor(step['boundaries']['routed_input'], dtype=torch.bfloat16).reshape_as(value)
index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
name = f'model.layers.{layer}.ffn_norm.weight'
with (checkpoint / index[name]).open('rb') as source:
    assert hashlib.file_digest(source, 'sha256').hexdigest() == reference['provenance']['weights_sha256'][index[name]]
with safe_open(checkpoint / index[name], framework='pt') as source:
    weight = source.get_tensor(name)
work = value.float()
actual = (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-5)).bfloat16() * weight
assert torch.equal(actual, expected)
path = Path(__file__).with_suffix('.safetensors')
half_value, half_weight = value.half(), weight.half()
half_work = half_value.float()
half_output = (half_work * torch.rsqrt(half_work.square().mean(-1, keepdim=True) + 1e-5)).half() * half_weight
save_file({'input': value, 'weight': weight, 'output': expected,
           'f16_input': half_value, 'f16_weight': half_weight, 'f16_output': half_output}, path,
          metadata={'source': 'LiquidAI/LFM2-8B-A1B',
                    'revision': 'c1c44ff9fc00db3ebf4516970563f5f383d23670',
                    'boundary': 'layer 3 ffn_norm, deletion cached decode prediction 1',
                    'epsilon': '1e-05',
                    'equation': '(x.float() * rsqrt(mean(x.float() ** 2) + eps)).bfloat16() * weight',
                    'torch': reference['versions']['torch']})
path.with_suffix('.json').write_text(json.dumps({
    'source': 'LiquidAI/LFM2-8B-A1B',
    'revision': 'c1c44ff9fc00db3ebf4516970563f5f383d23670',
    'prefix_ids': reference['prefix_ids'],
    'prediction': 1,
    'boundary': 'layer 3 feed-forward RMS normalization',
    'epsilon': 1e-5,
    'torch': reference['versions']['torch'],
    'transformers': reference['versions']['transformers'],
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'equation': '(input.float() * torch.rsqrt(input.float().square().mean(-1, keepdim=True) + 1e-5)).bfloat16() * weight',
    'native_failure_before_fix': '19 differing coordinates on Metal despite exact residual input',
    'f16_variant': 'Inputs and gains converted to F16; independent float32 RMS, F16 normalized value and product',
}, indent=2) + '\n')
