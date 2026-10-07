#!/usr/bin/env python3
"""Original pinned codec PRVQ oracle with nonzero synthetic fixed means."""
import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REVISION = '097dfe9e2f55baf653b83035868bdc89849f1b47'
SOURCE = 'nemo/collections/speechlm2/modules/ear_tts_vae_codec.py'


def tensor(value):
    return {'shape': list(value.shape), 'values': value.detach().reshape(-1).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.speech_checkout / SOURCE
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected_hash = '840906c66696fb38966645e11a768229d2c53059bb2a823cb09f8f2460a810ce'
    assert digest == expected_hash, source
    names = {'PreTrainedEMAVariance', 'PreTrainedProbabilisticVQ'}
    nodes = [node for node in ast.parse(source.read_bytes()).body
             if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in nodes} == names
    scope = dict(torch=torch, Tensor=Tensor, nn=nn, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    torch.set_num_threads(1)
    torch.manual_seed(0)  # Every randomly initialized mean is overwritten.
    model = scope['PreTrainedProbabilisticVQ'](channels=4, num_mixtures=5, depth=3).eval()
    with torch.no_grad():
        for depth, table in enumerate(model.mus_list):
            values = [.47 * math.sin(i * .73 + depth * 1.19) + .18 * math.cos(i * .31)
                      for i in range(table.numel())]
            table.copy_(torch.tensor(values).reshape(table.shape))
            model._variance_list[depth].variance.fill_(.13 / (depth + 1))
        # Exact duplicates exercise first-argmin behavior at every depth.
        for table in model.mus_list:
            table[1].copy_(table[0])
    latent = torch.tensor([.83, -.16, .51, -.27, -.39, .71, -.52, .12,
                           .29, .37, -.68, .94, -.19, -.47, .23, .61]).reshape(2, 2, 4)
    latent[0, 0] = model.mus_list[0][0]

    def run(active, input):
        retained = {}
        def record(frame, event, arg):
            if event == 'return' and frame.f_code is active.encode.__func__.__code__:
                retained['residual'] = frame.f_locals['r'].clone()
        sys.setprofile(record)
        try:
            codes, quantized = active.encode(input, return_z_q=True)
        finally:
            sys.setprofile(None)
        # The pre-trained inference class initializes decode accumulation as
        # F32 even for a Float64 model. Adding the first mean promotes it.
        decoded = active.decode(codes)
        return dict(codes=torch.stack(codes, -1), quantized=quantized,
                    residual=retained['residual'], decoded=decoded,
                    prefix_decoded=active.decode(codes[:2]))

    f32 = run(model, latent)
    params = {'tts_model.audio_codec.prvq.' + name: tensor(value)
              for name, value in model.state_dict().items()}
    f64 = run(model.double(), latent.double())
    assert torch.equal(f32['codes'], f64['codes'])
    max_delta = max((f32[key].double() - f64[key]).abs().max().item()
                    for key in f32 if key != 'codes')
    # Fixed independently of Eredu output; conservative float32 arithmetic bound.
    assert max_delta < 1e-6
    result = dict(provenance=dict(revision=REVISION, source=SOURCE, sha256=digest,
                                 pytorch=torch.__version__, execution='CPU; original class ASTs'),
                  geometry=dict(depth=3, cardinality=5, channels=4),
                  parameters=params, input=tensor(latent),
                  **{key: tensor(value) for key, value in f32.items()},
                  tolerance=dict(absolute=1e-6, relative=1e-5,
                                 f32_vs_f64_max_absolute=max_delta,
                                 decisions_unchanged=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {args.output}; f32/f64 max absolute delta {max_delta:.9g}')


if __name__ == '__main__':
    main()
