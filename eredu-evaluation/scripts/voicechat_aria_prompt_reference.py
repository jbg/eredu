#!/usr/bin/env python3
"""Pinned fixed-Aria initialization policy with an explicit supplied-code boundary.

Executes original DuplexEARTTS initialization, retained-buffer cloning and batch
expansion methods. A supplied-code codec records the silent waveform requested
by the original method. This fixture does not execute waveform encoding or load
released weights; the codec-encoder fixture covers the neural encoder separately.
"""
import argparse
import hashlib
import json
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from voicechat_mog_reference import extract, tensor, PRECISION, REVISION

MODEL = 'nemo/collections/speechlm2/models/duplex_ear_tts.py'
HASHES = {
    MODEL: '63c2dc41bab12a844d1f301a09e673c6d49ce6c406932068334b1e8afea2390c',
    PRECISION: 'a8f7ecac9d4962efe9ab0d4fe2b7a06b9c735d9fa5e54a52e3635182728c1c5c',
}


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    for path, digest in HASHES.items():
        assert hashlib.sha256((args.speech_checkout / path).read_bytes()).hexdigest() == digest, path
    torch.set_num_threads(1)
    namespace = dict(torch=torch, F=F, logging=logging, contextmanager=contextmanager)
    extract(args.speech_checkout / PRECISION, ['fp32_precision'], namespace)
    extract(args.speech_checkout / MODEL, ['ensures_target_precision'], namespace)
    extract(args.speech_checkout / MODEL,
            ['set_init_inputs', 'get_init_inputs', 'get_audio_prompt_latent'], namespace, 'DuplexEARTTS')

    class SuppliedCodes:
        def __init__(self, codes):
            self.codes = codes
        def encode(self, audio, lengths):
            self.carrier = dict(shape=list(audio.shape), lengths=tensor(lengths),
                                nonzero_samples=int(torch.count_nonzero(audio)))
            return self.codes.clone(), None

    class Boundary:
        pass
    for name in ['set_init_inputs', 'get_init_inputs', 'get_audio_prompt_latent']:
        setattr(Boundary, name, namespace[name])

    def run(name, sample_rate, samples_per_frame, duration, depth, cardinality):
        model = Boundary()
        model.device = torch.device('cpu')
        model.target_sample_rate = sample_rate
        model.target_samples_per_frame = samples_per_frame
        model.data_cfg = SimpleNamespace(audio_prompt_duration=duration)
        model.cfg = Config(use_system_prompt=False, tts_config=SimpleNamespace(context_hidden_size=None))
        model.tokenizer = SimpleNamespace(bos=1, eos=2)
        model.text_pad_id = 12
        model.speech_pad_id = cardinality
        model.audio_codec_run_dtype = torch.float32
        frames = int(duration * sample_rate // samples_per_frame)
        supplied = torch.tensor([(7 * i + i // depth) % cardinality for i in range((frames + 1) * depth)],
                                dtype=torch.int64).reshape(1, frames + 1, depth)
        model.audio_codec = SuppliedCodes(supplied)
        speaker = torch.arange(frames * 4, dtype=torch.float32).reshape(1, frames, 4) * .03125 - .5
        model.audio_prompt_latents = {'Aria': speaker}
        initial = model.set_init_inputs(system_prompt='This is ignored by the released TTS policy.', speaker_name='Aria')
        # The retained cache owns clones, even when the returned original is modified.
        initial['code'].fill_(-77)
        batches = []
        for batch in [1, 3]:
            prepared = model.get_init_inputs(B=batch)
            batches.append(dict(batch=batch, tensors={k: None if v is None else tensor(v)
                                                      for k, v in prepared.items()}))
        return dict(name=name, sample_rate=sample_rate, samples_per_frame=samples_per_frame,
                    prompt_duration=duration, prompt_frames=frames, codebooks=depth,
                    cardinality=cardinality, text_pad_id=12, text_eos_id=2, speech_pad_id=cardinality,
                    supplied_codes=tensor(supplied), supplied_aria=tensor(speaker),
                    carrier=model.audio_codec.carrier, batches=batches)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='In CPU autocast')
        cases = [run('compact', 100, 8, .4, 3, 5),
                 run('released_frame_geometry', 22050, 1764, 3.0, 31, 1024)]
    document = dict(provenance=dict(speech_revision=REVISION, source_sha256=HASHES,
        torch_version=torch.__version__, scope='Original fixed-Aria initialization and retained-buffer batch expansion; supplied codec codes and tiny Aria values. No waveform encoding or released-weight parity.'),
        comparison=dict(integer_and_boolean='exact', supplied_aria='exact'), cases=cases)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(f'{args.output}: original initialization and batch expansion; exact discrete comparison')


if __name__ == '__main__':
    main()
