#!/usr/bin/env python3
"""Hash-verified NVIDIA tiny FastConformer/subsampling CPU oracle.

Executes original class/method ASTs from the pinned Speech checkout, with no
Eredu imports. Adapter/access mixins are inert; training dropout and autocast
are disabled. This is synthetic component evidence, not released-weight parity.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

REVISION = '097dfe9e2f55baf653b83035868bdc89849f1b47'
ROOT = 'nemo/collections/asr/'
FILES = {
    'parts/submodules/causal_convs.py': 'c202f682212471cd80efd25bd201cd9a34126170ccc5c86112a2febfffbc10dd',
    'parts/submodules/conformer_modules.py': 'cd42348cfd98fe16c2f85b1fa901401951e5288ba0f13a0faeeb2e8b43ba7462',
    'parts/submodules/subsampling.py': '82cb1b263b6c6e9ae49e247b758ed2d62d72ad1244fed5ad89526b344f6c74cb',
    'parts/submodules/multi_head_attention.py': 'f06bbe3b76cb27401bf3d3c393b2bdd82dd583631160cef01d56ef1d38b11813',
    'modules/conformer_encoder.py': '54fffd4ad6ff9369913f20b82809948f7b96bb86bd083fd547252d4ab6cb742a',
    'parts/utils/activations.py': 'e89ad3cc5114fa025b031348c36faff2f6b8ef566a62b105c59b4f1177f4ff03',
}
SCHEDULE_SOURCE = 'nemo/collections/speechlm2/inference/model_wrappers/perception_cache.py'
SCHEDULE_HASH = '96308c2057296a291ef517b413638400669cd8d94807cf135e6dc7021c72b9d9'


class AdapterMixin:
    def is_adapter_available(self):
        return False


class AccessMixin:
    def is_access_enabled(self, _guid):
        return False


def load_nodes(checkout, namespace, relative, names, parent=None):
    source = (checkout / ROOT / relative).read_bytes()
    assert hashlib.sha256(source).hexdigest() == FILES[relative], relative
    nodes = ast.parse(source).body
    if parent:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == parent).body
    nodes = [node for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(checkout / ROOT / relative), 'exec'), namespace)


def tensor(value):
    return {'shape': list(value.shape), 'values': value.detach().cpu().reshape(-1).tolist()}


def initialize(module):
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(module.named_parameters()):
            values = [0.23 * math.sin((index + 1) * .61 + ordinal * .91)
                      for index in range(parameter.numel())]
            # Keep normalization scales near one, with nonzero learned biases.
            if parameter.ndim == 1 and name.endswith('weight'):
                values = [1.0 + value for value in values]
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))


def parameters(module, prefix):
    return {prefix + name: tensor(value) for name, value in module.named_parameters()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert hashlib.sha256((args.checkout / SCHEDULE_SOURCE).read_bytes()).hexdigest() == SCHEDULE_HASH
    torch.set_num_threads(1)
    namespace = dict(torch=torch, nn=nn, F=F, math=math, List=List, Tuple=Tuple,
                     Union=Union, LayerNorm=nn.LayerNorm, INF_VAL=10000.0,
                     avoid_float16_autocast_context=contextlib.nullcontext,
                     AttentionAdapterModuleMixin=AdapterMixin, AccessMixin=AccessMixin,
                     activation_registry={})
    load_nodes(args.checkout, namespace, 'parts/utils/activations.py', ['Swish'])
    load_nodes(args.checkout, namespace, 'parts/submodules/causal_convs.py', ['CausalConv1D', 'CausalConv2D'])
    load_nodes(args.checkout, namespace, 'parts/submodules/subsampling.py', ['ConvSubsampling', 'calc_length'])
    load_nodes(args.checkout, namespace, 'parts/submodules/multi_head_attention.py',
               ['MultiHeadAttention', 'RelPositionMultiHeadAttention', 'PositionalEncoding', 'RelPositionalEncoding'])
    load_nodes(args.checkout, namespace, 'parts/submodules/conformer_modules.py',
               ['ConformerLayer', 'ConformerConvolution', 'ConformerFeedForward'])
    load_nodes(args.checkout, namespace, 'modules/conformer_encoder.py', ['_create_masks'], 'ConformerEncoder')
    mask_owner = SimpleNamespace(self_attention_model='rel_pos', att_context_style='chunked_limited')
    create_masks = namespace['_create_masks']
    subsampler = namespace['ConvSubsampling'](
        subsampling='dw_striding', subsampling_factor=8, feat_in=8, feat_out=4,
        conv_channels=2, subsampling_conv_chunking_factor=-1, activation=nn.ReLU(), is_causal=True).eval()
    layer = namespace['ConformerLayer'](
        d_model=4, d_ff=8, n_heads=2, conv_kernel_size=3, conv_norm_type='layer_norm',
        conv_context_size=[2, 0], dropout=0.0, dropout_att=0.0, att_context_size=[3, 0],
        use_bias=False, use_pytorch_sdpa=False).eval()
    layer.self_attn.cache_drop_size = 0
    layer.conv.depthwise_conv.cache_drop_size = 0
    position = namespace['RelPositionalEncoding'](d_model=4, dropout_rate=0.0, xscale=None).eval()
    position.extend_pe(64, 'cpu', torch.float32)
    initialize(subsampler)
    initialize(layer)
    projection = nn.Linear(4, 6).eval()
    initialize(projection)
    feature = torch.tensor([math.sin(i * .17) * .7 + math.cos(i * .07) * .2 for i in range(41 * 8)]).reshape(1, 41, 8)
    hidden = torch.tensor([math.sin(i * .31 + .1) * .8 + math.cos(i * .13) * .3 for i in range(6 * 4)]).reshape(1, 6, 4)

    def subsample_run(model, source):
        records = []
        with torch.inference_mode():
            full, full_length = model(source, torch.tensor([41]))
            for step in range(6):
                if step == 0:
                    window, drop = source[:, :1], 0
                else:
                    start = 1 + (step - 1) * 8
                    cache_start = max(0, start - 9)
                    prefix = source[:, cache_start:start]
                    if prefix.shape[1] < 9:
                        prefix = torch.cat([source.new_zeros(1, 9 - prefix.shape[1], 8), prefix], dim=1)
                    window, drop = torch.cat([prefix, source[:, start:start + 8]], dim=1), 2
                out, length = model(window, torch.tensor([window.shape[1]]))
                records.append({'input': tensor(window), 'raw_output': tensor(out), 'drop': drop,
                                'output': tensor(out[:, drop:]), 'raw_length': int(length[0])})
        assembled = torch.cat([torch.tensor(item['output']['values'], dtype=source.dtype).reshape(item['output']['shape']) for item in records], dim=1)
        return full, full_length, records, float((full - assembled).abs().max())

    def conformer_run(model, source, pos, counts=(1, 1, 2, 1, 1)):
        records = []
        channel = source.new_zeros(1, 3, 4)
        temporal = source.new_zeros(1, 4, 2)
        valid = 0
        cursor = 0
        with torch.inference_mode():
            full_input, full_pos = pos(source)
            full_pad, full_mask = create_masks(mask_owner, [3, 0], torch.tensor([6]), 6, None, 'cpu')
            full_output = model(full_input, att_mask=full_mask, pos_emb=full_pos, pad_mask=full_pad)
            for count in counts:
                value = source[:, cursor:cursor + count]
                x, positions = pos(value, cache_len=3)
                pad, mask = create_masks(mask_owner, [3, 0], torch.tensor([count + 3]), count + 3,
                                         torch.tensor([3 - valid]), 'cpu')
                pad, mask = pad[:, 3:], mask[:, 3:]
                output, channel, temporal = model(x, att_mask=mask, pos_emb=positions, pad_mask=pad,
                                                  cache_last_channel=channel, cache_last_time=temporal)
                valid = min(3, valid + count)
                records.append({'input': tensor(value), 'output': tensor(output), 'positions': tensor(positions),
                                'attention_mask': tensor(mask), 'pad_mask': tensor(pad),
                                'channel_cache': tensor(channel), 'time_cache': tensor(temporal), 'valid_cache': valid})
                cursor += count
        assembled = torch.cat([torch.tensor(item['output']['values'], dtype=source.dtype).reshape(item['output']['shape']) for item in records], dim=1)
        return full_output, records, float((full_output - assembled).abs().max())

    sub_full, sub_len, sub_steps, sub_segment_delta = subsample_run(subsampler, feature)
    layer_full, layer_steps, layer_segment_delta = conformer_run(layer, hidden, position)
    coupled_encoder, coupled_steps, coupled_segment_delta = conformer_run(
        layer, sub_full, position, counts=(1,) * 6)
    with torch.inference_mode():
        coupled_projected = projection(coupled_encoder)
        for record in coupled_steps:
            record['encoder_output'] = record.pop('output')
            record['projected_output'] = tensor(projection(torch.tensor(
                record['encoder_output']['values']).reshape(record['encoder_output']['shape'])))
    sub_double, _, _, _ = subsample_run(copy.deepcopy(subsampler).double(), feature.double())
    layer_double, double_steps, _ = conformer_run(copy.deepcopy(layer).double(), hidden.double(), copy.deepcopy(position).double())
    coupled_double, _, _ = conformer_run(copy.deepcopy(layer).double(), sub_double,
                                        copy.deepcopy(position).double(), counts=(1,) * 6)
    with torch.inference_mode():
        projected_double = copy.deepcopy(projection).double()(coupled_double)
    precision_delta = max(float((sub_full.double() - sub_double).abs().max()),
                          float((layer_full.double() - layer_double).abs().max()),
                          float((coupled_encoder.double() - coupled_double).abs().max()),
                          float((coupled_projected.double() - projected_double).abs().max()))
    for low, high in zip(layer_steps, double_steps):
        for key in ['output', 'channel_cache', 'time_cache']:
            precision_delta = max(precision_delta, max(abs(a - b) for a, b in zip(low[key]['values'], high[key]['values'])))
    assert sub_segment_delta < 2e-6, sub_segment_delta
    assert layer_segment_delta < 2e-6, layer_segment_delta
    assert coupled_segment_delta < 2e-6, coupled_segment_delta
    assert precision_delta < 3e-6, precision_delta
    fixture = {
        'provenance': {'repository': 'https://github.com/NVIDIA-NeMo/Speech', 'revision': REVISION,
                       'source_sha256': {ROOT + key: value for key, value in FILES.items()},
                       'streaming_schedule_source': {'path': SCHEDULE_SOURCE, 'sha256': SCHEDULE_HASH},
                       'torch_version': torch.__version__, 'dtype': 'float32',
                       'oracle': 'original extracted NVIDIA classes/methods with inert adapter/access mixins',
                       'scope': 'synthetic tiny component weights; no released-weight/full-stack parity'},
        'tolerance': {'absolute': 3e-6, 'relative': 3e-5, 'f32_vs_f64_max_absolute': precision_delta,
                      'basis': 'same float32 parameters/inputs widened to float64 before comparison with Eredu'},
        'subsampling': {'geometry': {'features': 8, 'channels': 2, 'width': 4, 'factor': 8},
                        'parameters': parameters(subsampler, 'stt_model.perception.encoder.pre_encode.'),
                        'input': tensor(feature), 'output': tensor(sub_full), 'length': int(sub_len[0]),
                        'steps': sub_steps, 'segmented_vs_full_max_absolute': sub_segment_delta},
        'conformer': {'geometry': {'width': 4, 'feed_forward': 8, 'heads': 2, 'kernel': 3,
                                   'left_context': 3, 'right_context': 0, 'norm_epsilon': 1e-5},
                      'parameters': parameters(layer, 'stt_model.perception.encoder.layers.0.'),
                      'input': tensor(hidden), 'output': tensor(layer_full), 'steps': layer_steps,
                      'segmented_vs_full_max_absolute': layer_segment_delta},
        'perception': {'output_width': 6,
                       'parameters': parameters(projection, 'stt_model.perception.proj.'),
                       'encoder_output': tensor(coupled_encoder),
                       'projected_output': tensor(coupled_projected),
                       'steps': coupled_steps,
                       'segmented_vs_full_max_absolute': coupled_segment_delta},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + '\n')


if __name__ == '__main__':
    main()
