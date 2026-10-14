"""Independently check released BF16 adapter rounding and native score distributions.

Usage: python nemotron_adapter_precision.py CHECKPOINT REFERENCE.safetensors
Native component files are emitted by the ignored normal-loader reference test.
FP32 source comparisons have separate stricter tolerances.
"""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from nemotron_reference_provenance import verify_release
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('checkpoint',type=Path)
parser.add_argument('reference',type=Path)
args=parser.parse_args()
root=args.checkpoint
release=verify_release(root,adapter=True)
config=json.loads((root/'linear_spec_lora/adapter_config.json').read_text())
scale=config['lora_alpha']/config['r']
torch.set_num_threads(8)
ref=load_file(str(args.reference))
weight=None
for member in release['weight_files']:
    with safe_open(root/member['rfilename'],framework='pt',device='cpu') as f:
        if 'encoder.layers.0.self_attn.o_proj.weight' in f.keys():
            weight=f.get_tensor('encoder.layers.0.self_attn.o_proj.weight').double()
            break
if weight is None: raise ValueError('released base projection is missing')
with safe_open(root/'linear_spec_lora/adapter_model.safetensors',framework='pt',device='cpu') as f:
    a=f.get_tensor('base_model.model.encoder.layers.0.self_attn.o_proj.lora_A.weight').double()
    b=f.get_tensor('base_model.model.encoder.layers.0.self_attn.o_proj.lora_B.weight').double()
for step in range(4):
    for mode in ['causal','block','draft']:
        label=f'{mode}.{step}'
        path=root/f'normal-loader-{label}-bf16-components.safetensors'
        if not path.exists(): raise FileNotFoundError(path)
        native=load_file(str(path))
        x=native[f'{label}.component.0.attention.write_input'].double()
        assert torch.equal(x,ref[f'{label}.component.0.attention.write_input'].double())
        base=(x@weight.T).to(torch.bfloat16).double()
        exact=base
        # FP64 products independently evaluate the published PEFT equation.
        # Round each FP32 adapter projection before the next operator.
        if mode=='draft':
            down=(x@a.T).float().double()
            delta=((down@b.T).float()*scale).double()
            exact=base+delta
        for name,actual in [('native',native[f'{label}.component.0.attention.write']),('torch',ref[f'{label}.component.0.attention.write'])]:
            actual=actual.to(torch.bfloat16)
            direction=torch.full_like(actual,float('inf'))
            spacing=(torch.nextafter(actual,direction).double()-actual.double()).abs()
            distance=(actual.double()-exact).abs()
            # F32 reduction sensitivity is evaluated using the absolute product sum.
            eps=torch.finfo(torch.float32).eps
            error=eps*x.shape[-1]/(1-eps*x.shape[-1])*(x.abs()@weight.abs().T)
            if mode=='draft':
                down_error=eps*x.shape[-1]/(1-eps*x.shape[-1])*(x.abs()@a.abs().T)
                error+=scale*(down_error@b.abs().T+eps*a.shape[0]/(1-eps*a.shape[0])*(down.abs()@b.abs().T))
            valid=distance<=spacing*0.50001+error
            print(label,name,'max_distance',distance.max().item(),'max_spacing',spacing.max().item(),'valid',bool(valid.all()),'outside_half_ulp',int((distance>spacing*.50001).sum()),flush=True)
            assert valid.all()
        actual=native[f'{label}.logits'].double()
        expected=ref[f'{label}.logits'].double()
        log_actual=actual.log_softmax(-1); log_expected=expected.log_softmax(-1)
        kl=(log_expected.exp()*(log_expected-log_actual)).sum(-1)
        matches=(actual.argmax(-1)==expected.argmax(-1)).sum()
        print(label,'mean_KL',kl.mean().item(),'max_KL',kl.max().item(),'greedy_matches',int(matches),'rows',actual.shape[-2],flush=True)

        assert kl.max() <= (0.005 if mode=='draft' else 0.002)
        assert int(matches) == actual.shape[-2]
