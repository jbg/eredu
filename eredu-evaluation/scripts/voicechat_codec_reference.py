#!/usr/bin/env python3
"""Original NVIDIA EAR neural decoder and streaming ISTFT component oracle.

Synthetic nonzero parameters; original Latent2Wav/ConvNeXt/cache/ISTFT ASTs.
Includes compact and released-rate geometries without released model weights.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Concatenate

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from voicechat_mog_reference import extract, tensor

REVISION = '097dfe9e2f55baf653b83035868bdc89849f1b47'
SOURCE = 'nemo/collections/speechlm2/modules/ear_tts_vae_codec.py'
DIGEST = '840906c66696fb38966645e11a768229d2c53059bb2a823cb09f8f2460a810ce'


def complex_tensor(value):
    return dict(real=tensor(value.real), imag=tensor(value.imag))


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
                     'spec_to_wav', 'Latent2Wav', 'PreTrainedEMAVariance', 'PreTrainedProbabilisticVQ'], namespace)
    torch.set_num_threads(1)
    torch.manual_seed(0)

    def run(model, latent, chunks=None, constrain=True, flush_last=True):
        records = []
        cache = None if chunks is None else namespace['CausalConv1dCache']()
        chunks = [latent.shape[1]] if chunks is None else chunks
        offset = 0
        current = {}
        def projected(module, inputs, output):
            current['raw'] = tensor(output)
        hook = model.layers[-1].register_forward_hook(projected)
        def profile(frame, event, arg):
            if event != 'return':
                return
            if frame.f_code is namespace['spec_to_wav'].__code__:
                locals = frame.f_locals
                if model.n_fft == 8:
                    current['istft_input'] = complex_tensor(locals['spec'])
                if latent.shape[1] <= 6 and model.n_fft == 8:
                    current['windowed_ifft'] = tensor(locals['ifft'])
                    current['envelope'] = tensor(locals['window_envelope'])
            elif frame.f_code is namespace['Latent2Wav'].forward.__code__:
                locals = frame.f_locals
                if model.n_fft == 8:
                    current['magnitude'] = tensor(locals['mag'])
                    current['phase'] = tensor(locals['ph'])
        try:
            with torch.inference_mode():
                for index, size in enumerate(chunks):
                    current = dict(offset=offset, input=tensor(latent[:, offset:offset + size]),
                                   flush=cache is not None and flush_last and index == len(chunks) - 1)
                    sys.setprofile(profile)
                    try:
                        output = model(latent[:, offset:offset + size], cache=cache,
                                       flush=current['flush'], constrain_value_range=constrain)
                    finally:
                        sys.setprofile(None)
                    current['waveform'] = tensor(output)
                    current['cache'] = {} if cache is None else {
                        str(key): complex_tensor(value) if value.is_complex() else tensor(value)
                        for key, value in cache.cache.items()}
                    records.append(current)
                    offset += size
        finally:
            hook.remove()
        return records

    def setup(geometry):
        model = namespace['Latent2Wav'](**geometry).eval()
        with torch.no_grad():
            for ordinal, (name, parameter) in enumerate(model.named_parameters()):
                values = [.24 * math.sin(i * .41 + ordinal * .73 + .2) + .09 * math.cos(i * .19)
                          for i in range(parameter.numel())]
                if name.endswith('norm.weight'):
                    values = [1 + value for value in values]
                parameter.copy_(torch.tensor(values).reshape(parameter.shape))
        return model

    geometries = {
        'compact': dict(latent_size=4, n_fft=8, hop_length=2, base_hidden_size=2,
                        channel_mult=[2, 1], rates=[3, 2], num_blocks=2, kernel_size=3, groups=1),
        'released_rates': dict(latent_size=4, n_fft=16, hop_length=4, base_hidden_size=1,
                               channel_mult=[4, 2, 1], rates=[9, 7, 7], num_blocks=3, kernel_size=3, groups=1),
    }
    cases = {}
    maximum_neural_error = 0.
    maximum_waveform_error = 0.
    for name, geometry in geometries.items():
        model = setup(geometry)
        frames = 6 if name == 'compact' else 4
        chunks = [1, 2, 1, 2] if name == 'compact' else [1, 2, 1]
        latent = torch.tensor([.67 * math.sin(i * .37 + .2) - .23 * math.cos(i * .17)
                               for i in range(frames * 4)]).reshape(1, frames, 4)
        full = run(model, latent)
        streaming = run(model, latent, chunks)
        wide = run(copy.deepcopy(model).double(), latent.double())
        for a, b in zip(full[0]['raw']['values'], wide[0]['raw']['values']):
            maximum_neural_error = max(maximum_neural_error, abs(a - b))
        for a, b in zip(full[0]['waveform']['values'], wide[0]['waveform']['values']):
            maximum_waveform_error = max(maximum_waveform_error, abs(a - b))
        raw_full = torch.tensor(full[0]['raw']['values']).reshape(full[0]['raw']['shape'])
        raw_chunks = torch.cat([torch.tensor(x['raw']['values']).reshape(x['raw']['shape']) for x in streaming], dim=-1)
        neural_difference = float((raw_full - raw_chunks).abs().max())
        assert neural_difference < 2e-5
        delay = math.ceil(((geometry['n_fft'] - geometry['hop_length']) // 2) / geometry['hop_length']) * geometry['hop_length']
        joined = [v for record in streaming for v in record['waveform']['values']]
        offline = full[0]['waveform']['values']
        assert len(joined) == len(offline) + delay
        # Startup zeros create the released delay. Offline edge normalization
        # differs; report its discrepancy instead of claiming identical edges.
        aligned_error = max(abs(a - b) for a, b in zip(joined[delay:], offline))
        interior = range(geometry['n_fft'], len(offline) - geometry['n_fft'])
        interior_error = max(abs(joined[delay + i] - offline[i]) for i in interior)
        assert interior_error < 2e-5
        assert not streaming[-1]['cache']
        cases[name] = dict(geometry=geometry, parameters={
            'tts_model.audio_codec.decoder.' + key: tensor(value) for key, value in model.state_dict().items()},
            full=full, streaming=streaming, neural_whole_vs_chunks_max_absolute_error=neural_difference,
            startup_delay_samples=delay, aligned_offline_max_absolute_error=aligned_error,
            aligned_offline_interior_max_absolute_error=interior_error)

    # An independent code-to-waveform path executes original PRVQ decoding
    # before the same original neural and spectral decoder.
    quantizer = namespace['PreTrainedProbabilisticVQ'](channels=4, num_mixtures=5, depth=3).eval()
    with torch.no_grad():
        for depth, means in enumerate(quantizer.mus_list):
            means.copy_(torch.tensor([.31 * math.sin(i * .53 + depth * .61 + .2)
                                     for i in range(means.numel())]).reshape(means.shape))
            quantizer._variance_list[depth].variance.fill_(.13 / (depth + 1))
    codes = torch.tensor([(i * 3 + i // 3) % 5 for i in range(6 * 3)]).reshape(1, 6, 3)
    with torch.inference_mode():
        latent = quantizer.decode([codes[..., i] for i in range(3)])
    model = setup(geometries['compact'])
    full = run(model, latent)
    stream = run(model, latent, [1, 2, 1, 2])
    unflushed = run(model, latent, [1, 2, 1, 2], flush_last=False)
    final_prefix = unflushed[-1]['waveform']['values']
    final_flushed = stream[-1]['waveform']['values']
    assert final_flushed[:len(final_prefix)] == final_prefix
    composition = dict(geometry=geometries['compact'], quantizer_geometry=dict(depth=3, cardinality=5, channels=4),
        parameters={**{'tts_model.audio_codec.decoder.' + key: tensor(value) for key, value in model.state_dict().items()},
                    **{'tts_model.audio_codec.prvq.' + key: tensor(value) for key, value in quantizer.state_dict().items()}},
        codes=tensor(codes), latent=tensor(latent), full=full, streaming=stream,
        streaming_unflushed=unflushed,
        finish_tail=dict(shape=[1, 1, len(final_flushed)-len(final_prefix)], values=final_flushed[len(final_prefix):]))

    # Direct inverse-transform cases independently exercise window constraints,
    # real-only DC/Nyquist bins and overlap-envelope reconstruction.
    nfft, hop, frames = 8, 2, 5
    real = torch.tensor([1.9 * math.sin(i * .43 + .2) for i in range(5 * frames)]).reshape(1, 5, frames)
    imag = torch.tensor([1.7 * math.cos(i * .31 + .4) for i in range(5 * frames)]).reshape(1, 5, frames)
    imag[:, [0, -1], :] = 0
    spectrum = torch.complex(real, imag)
    spectral = dict(n_fft=nfft, hop_length=hop, spectrum=complex_tensor(spectrum), cases=[])
    max_spectral_error = 0.
    for constrain in [False, True]:
        output = namespace['spec_to_wav'](spectrum, nfft, hop, nfft, constrain_value_range=constrain)
        wide = namespace['spec_to_wav'](spectrum.to(torch.complex128), nfft, hop, nfft, constrain_value_range=constrain)
        max_spectral_error = max(max_spectral_error, float((output.double() - wide).abs().max()))
        spectral['cases'].append(dict(constrain=constrain, waveform=tensor(output)))
    assert spectral['cases'][0]['waveform']['values'] != spectral['cases'][1]['waveform']['values']
    assert maximum_neural_error < 2e-5 and maximum_waveform_error < 2e-5 and max_spectral_error < 2e-5
    document = dict(provenance=dict(speech_revision=REVISION, source=SOURCE, source_sha256=DIGEST,
        torch_version=torch.__version__, scope='Original ASTs with tiny nonzero parameters; released upsampling/FFT rate geometry separately exercised; no released-weight or full codec encoder parity.'),
        comparison=dict(absolute_tolerance=2e-5, relative_tolerance=5e-5,
                        float32_vs_float64_neural_max_absolute_error=maximum_neural_error,
                        widened_neural_float32_spectral_waveform_max_absolute_error=maximum_waveform_error,
                        float32_vs_float64_ifft_with_original_float32_window_max_absolute_error=max_spectral_error),
        decoders=cases, spectral=spectral, composition=composition)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(json.dumps(document['comparison']))
    for name, case in cases.items():
        print(name, {key: value for key, value in case.items() if 'error' in key or key == 'startup_delay_samples'})


if __name__ == '__main__':
    main()
