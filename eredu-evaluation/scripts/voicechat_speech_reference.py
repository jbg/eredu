#!/usr/bin/env python3
"""Pinned NVIDIA codec decoding of pinned TTS decisions and EOS substitutions.

Uses existing independent TTS codes/means and compact nonzero codec parameters;
this component composition has no released weights or agent policy execution.
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
from voicechat_codec_reference import DIGEST, REVISION, SOURCE
from voicechat_mog_reference import extract, tensor

INPUTS = {
    'tts-transition-nvidia.json': 'a5feead59cfe6f62146d07898f08ae8c5a5fb18ce58610a362e2201891f46a1c',
    'codec-nvidia.json': '3d4fcc743b61f16d45bda4014221e973ff3ea8ee34944a9c77fec5c13b719fb1',
}


def load_tensor(value):
    return torch.tensor(value['values']).reshape(value['shape'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.speech_checkout / SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == DIGEST
    inputs = Path(__file__).resolve().parent.parent / 'fixtures' / 'voicechat'
    for name, digest in INPUTS.items():
        assert hashlib.sha256((inputs / name).read_bytes()).hexdigest() == digest
    tts = json.loads((inputs / 'tts-transition-nvidia.json').read_text())
    codec = json.loads((inputs / 'codec-nvidia.json').read_text())
    namespace = dict(torch=torch, Tensor=Tensor, nn=nn, F=F, math=math,
                     Callable=Callable, Concatenate=Concatenate)
    extract(source, ['zero_module', 'CausalConv1dCache', 'LayerNormNd', 'ConvNeXt1d',
                     'spec_to_wav', 'Latent2Wav', 'PreTrainedEMAVariance',
                     'PreTrainedProbabilisticVQ'], namespace)
    torch.set_num_threads(1)
    case = codec['decoders']['compact']
    model = namespace['Latent2Wav'](**case['geometry']).eval()
    model.load_state_dict({k.removeprefix('tts_model.audio_codec.decoder.'): load_tensor(v)
                           for k, v in case['parameters'].items()})
    quantizer = namespace['PreTrainedProbabilisticVQ'](channels=4, num_mixtures=5, depth=31).eval()
    means = load_tensor(tts['parameters']['tts_model.tts_model.rvq_embs'])
    with torch.no_grad():
        for depth, parameter in enumerate(quantizer.mus_list):
            parameter.copy_(means[depth])
            quantizer._variance_list[depth].variance.fill_(1.)
    first = load_tensor(tts['calls'][1]['codes'])
    second = load_tensor(tts['calls'][2]['codes'])
    silence = load_tensor(tts['silence']).reshape(1, 1, 31)

    def run(decoder, rvq, sequence, flush_last=False):
        cache = namespace['CausalConv1dCache']()
        outputs = []
        with torch.inference_mode():
            for index, codes in enumerate(sequence):
                latent = rvq.decode([codes[..., depth] for depth in range(31)])
                output = decoder(latent, cache=cache, constrain_value_range=True,
                                 flush=flush_last and index == len(sequence)-1)
                outputs.append(output)
        return outputs

    cases = {}
    error = 0.
    for name, final in [('main_eos', second), ('tool_eos', silence)]:
        sequence = [first, final]
        outputs = run(model, quantizer, sequence)
        wide = run(copy.deepcopy(model).double(), copy.deepcopy(quantizer).double(), sequence)
        for f32, f64 in zip(outputs, wide):
            error = max(error, float((f32.double()-f64).abs().max()))
        flushed = run(model, quantizer, sequence, flush_last=True)
        length = outputs[-1].shape[-1]
        assert torch.equal(flushed[-1][..., :length], outputs[-1])
        cases[name] = dict(frames=[tensor(x) for x in outputs],
                           tail=tensor(flushed[-1][..., length:]))
    # Fixed before comparing Eredu: same original codec F32/F64 evidence policy
    # as codec-nvidia.json, with this composition's measured precision bound.
    assert error < 2e-5
    document = dict(provenance=dict(speech_revision=REVISION, source=SOURCE,
        source_sha256=DIGEST, input_sha256=INPUTS, torch_version=torch.__version__,
        scope='Pinned TTS decisions and means through original codec ASTs with compact nonzero parameters; no released-weight validation.'),
        comparison=dict(absolute_tolerance=2e-5, relative_tolerance=5e-5,
            widened_neural_float32_spectral_waveform_max_absolute_error=error), cases=cases)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
    print(json.dumps(document['comparison']))


if __name__ == '__main__':
    main()
