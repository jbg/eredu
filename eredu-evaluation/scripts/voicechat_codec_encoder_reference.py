#!/usr/bin/env python3
"""Pinned original EAR STFT, Wav2Latent and PRVQ encoder numerical oracle.

Synthetic nonzero parameters; includes an all-zero carrier and a nonzero signal.
The encoder reference has no cache path. This fixture establishes finite-input
component equations and exact code choices, not released-weight parity.
"""
import argparse
import copy
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Concatenate

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from voicechat_mog_reference import extract, tensor
from voicechat_codec_reference import REVISION, SOURCE, DIGEST


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.speech_checkout / SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == DIGEST
    namespace = dict(torch=torch, Tensor=Tensor, nn=nn, F=F, math=math,
                     Callable=Callable, Concatenate=Concatenate)
    extract(source, ['zero_module', 'CausalConv1dCache', 'LayerNormNd', 'ConvNeXt1d',
                     'spectrogram', 'Wav2Latent', 'PreTrainedEMAVariance', 'PreTrainedProbabilisticVQ'], namespace)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cases = {}
    maximum_error = 0.
    for name, geometry, batch, frames in [
        ('compact', dict(latent_size=4, n_fft=8, hop_length=2, base_hidden_size=2,
            channel_mult=[1, 2], rates=[2, 3], num_blocks=2, kernel_size=3, groups=1), 2, 4),
        ('released_rates', dict(latent_size=4, n_fft=16, hop_length=4, base_hidden_size=1,
            channel_mult=[1, 2, 4], rates=[7, 7, 9], num_blocks=3, kernel_size=3, groups=1), 1, 2),
    ]:
        model = namespace['Wav2Latent'](**geometry).eval()
        quantizer = namespace['PreTrainedProbabilisticVQ'](channels=4, num_mixtures=5, depth=3).eval()
        with torch.no_grad():
            for ordinal, (key, parameter) in enumerate(model.named_parameters()):
                values = [.24 * math.sin(i * .41 + ordinal * .73 + .2) + .09 * math.cos(i * .19)
                          for i in range(parameter.numel())]
                if key.endswith('norm.weight'):
                    values = [1 + value for value in values]
                parameter.copy_(torch.tensor(values).reshape(parameter.shape))
            for depth, means in enumerate(quantizer.mus_list):
                means.copy_(torch.tensor([.31 * math.sin(i * .53 + depth * .61 + .2)
                                         for i in range(means.numel())]).reshape(means.shape))
                quantizer._variance_list[depth].variance.fill_(.13 / (depth + 1))
        ratio = geometry['hop_length'] * math.prod(geometry['rates'])
        pcm = torch.tensor([.67 * math.sin(i * .37 + .2) - .23 * math.cos(i * .17)
                            for i in range(batch * frames * ratio)]).reshape(batch, 1, frames * ratio)
        if batch == 2:
            pcm[0] = 0.
        spec = namespace['spectrogram'](pcm.squeeze(1), geometry['n_fft'], geometry['hop_length'], geometry['n_fft'])
        packed = torch.cat([spec.real, spec.imag], dim=1)

        def run(active, q):
            stages = []
            hooks = [layer.register_forward_hook(lambda module, args, output: stages.append(tensor(output)))
                     for layer in active.layers]
            # Original spectrogram explicitly returns F32. Widen only learned
            # arithmetic after this fixed source boundary for precision evidence.
            cast = active.layers[0].register_forward_pre_hook(
                lambda module, args: (args[0].to(next(module.parameters()).dtype),))
            try:
                with torch.inference_mode():
                    latent = active(pcm)
                    codes, quantized = q.encode(latent, return_z_q=True)
                    return dict(stages=stages, latent=tensor(latent), codes=tensor(torch.stack(codes, -1)),
                                quantized=tensor(quantized), decoded=tensor(q.decode(codes)))
            finally:
                cast.remove()
                for hook in hooks:
                    hook.remove()
        narrow = run(model, quantizer)
        wide = run(copy.deepcopy(model).double(), copy.deepcopy(quantizer).double())
        assert narrow['codes'] == wide['codes']
        for a, b in zip(narrow['stages'] + [narrow['latent'], narrow['quantized'], narrow['decoded']],
                        wide['stages'] + [wide['latent'], wide['quantized'], wide['decoded']]):
            maximum_error = max(maximum_error, *[abs(x - y) for x, y in zip(a['values'], b['values'])])
        parameters = {'tts_model.audio_codec.encoder.' + key: tensor(value)
                      for key, value in model.state_dict().items()}
        parameters.update({'tts_model.audio_codec.prvq.' + key: tensor(value)
                           for key, value in quantizer.state_dict().items()})
        # Full intermediate arrays localize compact numerical failures. The
        # rate case keeps final outputs without repeating its long hidden stack.
        if name != 'compact':
            narrow.pop('stages')
        cases[name] = dict(geometry=geometry, quantizer_geometry=dict(depth=3, cardinality=5, channels=4),
            parameters=parameters, pcm=tensor(pcm), spectrum=dict(real=tensor(spec.real), imag=tensor(spec.imag)),
            packed=tensor(packed), samples_per_code=ratio, **narrow)
    # An independent F64 STFT execution measures the original Float32 Hann/STFT error.
    pcm = torch.tensor([.2, -.7, .4, .3, -.9, .8, .1, -.5, .6, -.2, .7, -.4]).reshape(1, -1)
    direct = []
    spectral_error = 0.
    for fft_size, hop in [(8, 3), (9, 2)]:
        actual = namespace['spectrogram'](pcm, fft_size, hop, fft_size)
        left = (fft_size - hop) // 2
        padded = F.pad(pcm.double(), (left, fft_size - hop - left))
        window = torch.hann_window(fft_size, dtype=torch.float32).double()
        wide = torch.stft(padded, fft_size, hop_length=hop, win_length=fft_size, window=window,
                         center=False, normalized=False, onesided=True, return_complex=True)
        spectral_error = max(spectral_error, float((actual.to(torch.complex128) - wide).abs().max()))
        direct.append(dict(n_fft=fft_size, hop_length=hop, pcm=tensor(pcm.unsqueeze(1)),
                           real=tensor(actual.real), imag=tensor(actual.imag)))
    assert maximum_error < 2e-5 and spectral_error < 2e-6
    document = dict(provenance=dict(speech_revision=REVISION, source=SOURCE, source_sha256=DIGEST,
        torch_version=torch.__version__, scope='Original AST STFT/Wav2Latent/PRVQ with synthetic tiny nonzero parameters; no encoder cache or released-weight parity.'),
        comparison=dict(absolute_tolerance=2e-5, relative_tolerance=5e-5,
            fixed_float32_spectral_float64_neural_max_absolute_error=maximum_error,
            float32_vs_float64_stft_original_float32_window_max_absolute_error=spectral_error,
            codes='Exact equality across both learned-arithmetic precision runs.'), encoders=cases, spectral=direct)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(json.dumps(document['comparison']))


if __name__ == '__main__':
    main()
