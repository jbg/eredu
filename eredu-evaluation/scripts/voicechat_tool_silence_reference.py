#!/usr/bin/env python3
"""Pinned NVIDIA full-perception tool-silence oracle, with compact learned weights.

Uses original NVIDIA preprocessing/Conformer code and wrapper cache selection.
The finite synthetic filter bank is shared with consumers: zero PCM has zero
power for every frequency, so filter coefficients cannot change this fixture.
No librosa, NeMo installation, Eredu imports, or released weights are required.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

import voicechat_perception_reference as reference

FEATURE_SOURCE = 'nemo/collections/asr/parts/preprocessing/features.py'
FEATURE_SHA256 = 'db0ba4b9f3592ed35d4e6e70e095c9d3e12d07df0810d0d183f26e9fe1f7a0ca'
WRAPPER_SOURCE = 'nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py'
WRAPPER_SHA256 = 'b50aeab4b36285cf041330b315e35a01788e1edb639380bb6105446787cc6a1a'


def source(path, expected):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, path
    return ast.parse(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    namespace = dict(torch=torch, nn=nn, F=F, math=math, logging=logging,
                     List=List, Tuple=Tuple, Union=Union, LayerNorm=nn.LayerNorm,
                     INF_VAL=10000.0, CONSTANT=1e-5, SAMPLE_RATE=16000,
                     avoid_float16_autocast_context=contextlib.nullcontext,
                     AttentionAdapterModuleMixin=reference.AdapterMixin,
                     AccessMixin=reference.AccessMixin, activation_registry={})
    for relative, names, parent in [
        ('parts/utils/activations.py', ['Swish'], None),
        ('parts/submodules/causal_convs.py', ['CausalConv1D', 'CausalConv2D'], None),
        ('parts/submodules/subsampling.py', ['ConvSubsampling', 'calc_length'], None),
        ('parts/submodules/multi_head_attention.py', ['MultiHeadAttention', 'RelPositionMultiHeadAttention',
                                                    'PositionalEncoding', 'RelPositionalEncoding'], None),
        ('parts/submodules/conformer_modules.py', ['ConformerLayer', 'ConformerConvolution',
                                                 'ConformerFeedForward'], None),
        ('modules/conformer_encoder.py', ['_create_masks'], 'ConformerEncoder'),
    ]:
        reference.load_nodes(args.checkout, namespace, relative, names, parent)
    features = source(args.checkout / FEATURE_SOURCE, FEATURE_SHA256)
    selected = [node for node in features.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in {'FilterbankFeatures', 'normalize_batch'}]
    exec(compile(ast.Module(body=selected, type_ignores=[]), FEATURE_SOURCE, 'exec'), namespace)
    # Configure the original module directly, avoiding the optional librosa
    # constructor dependency. Both buffers are exported exactly for consumers.
    processor = namespace['FilterbankFeatures'].__new__(namespace['FilterbankFeatures'])
    nn.Module.__init__(processor)
    for key, value in dict(win_length=400, hop_length=160, n_fft=512,
                           stft_pad_amount=None, exact_pad=False, normalize='NA',
                           log=True, dither=0, frame_splicing=1, nfilt=128,
                           preemph=.97, pad_to=0, pad_value=0, mag_power=2.0,
                           use_grads=False, nb_augmentation_prob=0,
                           log_zero_guard_type='add', log_zero_guard_value=2**-24).items():
        setattr(processor, key, value)
    processor.register_buffer('window', torch.hann_window(400, periodic=False))
    processor.register_buffer('fb', torch.eye(128, 257).unsqueeze(0))
    processor.eval()
    subsampler = namespace['ConvSubsampling'](
        subsampling='dw_striding', subsampling_factor=8, feat_in=128, feat_out=4,
        conv_channels=2, subsampling_conv_chunking_factor=-1,
        activation=nn.ReLU(), is_causal=True).eval()
    layer = namespace['ConformerLayer'](
        d_model=4, d_ff=8, n_heads=2, conv_kernel_size=3, conv_norm_type='layer_norm',
        conv_context_size=[2, 0], dropout=0.0, dropout_att=0.0, att_context_size=[3, 0],
        use_bias=False, use_pytorch_sdpa=False).eval()
    position = namespace['RelPositionalEncoding'](d_model=4, dropout_rate=0.0, xscale=None).eval()
    position.extend_pe(128, 'cpu', torch.float32)
    projection = nn.Linear(4, 6).eval()
    for module in [subsampler, layer, projection]:
        reference.initialize(module)
    mask_owner = SimpleNamespace(self_attention_model='rel_pos', att_context_style='chunked_limited')

    def execute(preprocessor, sub, block, pos, proj, dtype):
        with torch.inference_mode():
            mel, mel_length = preprocessor(torch.zeros(1, 16000, dtype=dtype), torch.tensor([16000]))
            hidden, length = sub(mel.transpose(1, 2), mel_length)
            values, positions = pos(hidden)
            pad, mask = namespace['_create_masks'](mask_owner, [3, 0], length,
                                                   hidden.shape[1], None, 'cpu')
            encoded = block(values, att_mask=mask, pos_emb=positions, pad_mask=pad)
            projected = proj(encoded)
        return mel, mel_length, encoded, projected, length

    mel, mel_length, encoded, projected, length = execute(processor, subsampler, layer, position, projection, torch.float32)
    high = execute(copy.deepcopy(processor).double(), copy.deepcopy(subsampler).double(),
                   copy.deepcopy(layer).double(), copy.deepcopy(position).double(),
                   copy.deepcopy(projection).double(), torch.float64)
    precision = max(float((low.double() - higher).abs().max())
                    for low, higher in [(mel, high[0]), (encoded, high[2]), (projected, high[3])])
    assert mel.shape == (1, 128, 101) and mel_length.tolist() == [101]
    assert projected.shape == (1, 14, 6) and length.tolist() == [14], (projected.shape, length.tolist())
    assert precision < 3e-6, precision
    wrapper = source(args.wrapper, WRAPPER_SHA256)
    method = next(node for owner in wrapper.body if isinstance(owner, ast.ClassDef)
                  for node in owner.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_create_silence_embedding_cache')
    exec(compile(ast.Module(body=[method], type_ignores=[]), WRAPPER_SOURCE, 'exec'), namespace)
    calls = []

    def perception(input_signal, input_signal_length, return_encoder_emb):
        assert torch.equal(input_signal, torch.zeros(1, 16000))
        assert input_signal_length.tolist() == [16000] and return_encoder_emb
        calls.append(True)
        return projected, length, encoded

    embeddings = {}
    for name, dtype in [('embedding', torch.float32), ('bf16_embedding', torch.bfloat16)]:
        owner = SimpleNamespace(device='cpu', dtype=dtype,
                                model=SimpleNamespace(stt_model=SimpleNamespace(perception=perception)))
        namespace['_create_silence_embedding_cache'](owner)
        # Wrapper's [1,H] cache is shaped as one batch/cadence for execution.
        embeddings[name] = reference.tensor(owner._silence_embedding_cache.reshape(1, 1, 6).float())
    assert len(calls) == 2
    assert any(abs(value) > .01 for value in embeddings['embedding']['values'])
    fixture = {
        'provenance': {'repository': 'https://github.com/NVIDIA-NeMo/Speech', 'revision': reference.REVISION,
                       'source_sha256': {reference.ROOT + key: value for key, value in reference.FILES.items()} |
                                        {FEATURE_SOURCE: FEATURE_SHA256, WRAPPER_SOURCE: WRAPPER_SHA256},
                       'torch_version': torch.__version__, 'dtype': 'float32',
                       'oracle': 'original NVIDIA preprocessing, subsampling, Conformer, and wrapper cache-selection ASTs',
                       'scope': 'synthetic learned weights and one-hot mel filters; zero PCM makes filter coefficients irrelevant; no released-weight parity',
                       'bf16_scope': 'wrapper final embedding cast only; learned computation remains float32'},
        'tolerance': {'absolute': 3e-6, 'relative': 3e-5, 'f32_vs_f64_max_absolute': precision,
                      'basis': 'same float32 buffers/parameters widened to float64 before comparison with Eredu'},
        'preprocessing': {'sample_rate': 16000, 'samples': 16000,
                          'window': reference.tensor(processor.window), 'filters': reference.tensor(processor.fb),
                          'mel': reference.tensor(mel.transpose(1, 2)), 'mel_length': int(mel_length[0])},
        'subsampling': {'geometry': {'features': 128, 'channels': 2, 'width': 4, 'factor': 8},
                        'parameters': reference.parameters(subsampler, 'stt_model.perception.encoder.pre_encode.')},
        'conformer': {'geometry': {'width': 4, 'feed_forward': 8, 'heads': 2, 'kernel': 3,
                                   'left_context': 3, 'right_context': 0, 'norm_epsilon': 1e-5},
                      'parameters': reference.parameters(layer, 'stt_model.perception.encoder.layers.0.')},
        'perception': {'output_width': 6,
                       'parameters': reference.parameters(projection, 'stt_model.perception.proj.'),
                       'encoder_output': reference.tensor(encoded), 'projected_output': reference.tensor(projected),
                       **embeddings},
    }
    args.output.write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
    print(f'Wrote {args.output}; F32/F64 max absolute error={precision:.9g}')


if __name__ == '__main__':
    main()
