#!/usr/bin/env python3
"""Pinned LiquidCodec component reference using the released neural codec weights.

Full publisher encode/decode produces trained latents and spectral predictions.
Native comparison currently covers spectral analysis, scalar projection/codes and
spectral streaming only; it does not establish native full-codec parity.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    source = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.is_relative_to(checkout):
        parser.error('reference payloads must live outside the repository')
    manifest = json.loads((checkout / 'validation/fixtures/lfm3/liquid-sources.json').read_text())
    if digest(source / manifest['weight_file']) != manifest['weights_sha256']:
        parser.error('released codec weight hash differs')
    for name, expected in manifest['source_sha256'].items():
        if digest(source / name) != expected:
            parser.error(f'pinned source hash differs: {name}')
    output.mkdir(parents=True, exist_ok=True)
    derived = output / 'f32-component'
    derived.mkdir(exist_ok=True)
    shutil.copyfile(source / 'audio_tokenizer/config.json', derived / 'config.json')
    config_cls = get_class_from_dynamic_module('configuration_liquid_codec.LiquidCodecConfig', source, local_files_only=True)
    codec_cls = get_class_from_dynamic_module('modeling_liquid_codec.LiquidCodecModel', source, local_files_only=True)
    cache_cls = get_class_from_dynamic_module('modeling_liquid_codec.LiquidCodecStreamingCache', source, local_files_only=True)
    config = config_cls.from_pretrained(source / 'audio_tokenizer', local_files_only=True)
    config._attn_implementation = 'eager'
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    codec = codec_cls(config).float().eval()
    with safe_open(source / manifest['weight_file'], framework='pt') as checkpoint:
        weights = {name: checkpoint.get_tensor(name).float().contiguous() for name in checkpoint.keys()}
    codec.load_state_dict(weights, strict=True)
    save_file({name: value for name, value in weights.items() if name.startswith('quantizer.')}, derived / 'model.safetensors')
    del weights
    tensors = {}
    cases = []
    with torch.inference_mode():
        for name, samples in [('short', 2305), ('window', 1920 * 9 + 117)]:
            time = torch.arange(samples, dtype=torch.float32) / 24000.0
            pcm = 0.2 * torch.sin(2 * torch.pi * 173.0 * time) + 0.07 * torch.cos(2 * torch.pi * 1093.0 * time)
            pcm = pcm.unsqueeze(0)
            padded = torch.nn.functional.pad(pcm, (0, -samples % config.frame_size))
            features = codec.stft(padded)
            latents = codec.to_quantizer(codec.encoder(codec.frontend(features).mT)).mT
            codes = codec.quantizer.encode(latents)
            decoded = codec.quantizer.decode(codes)
            embeddings = codec.from_quantizer(decoded.mT)
            upsampled = torch.nn.functional.interpolate(embeddings.mT, embeddings.shape[1] * config.stft_frame_stack,
                                                        mode='nearest-exact').mT
            hidden = codec.decoder.lfm(inputs_embeds=upsampled, use_cache=False).last_hidden_state
            log_magnitude, angle = codec.decoder.lin(hidden).mT.contiguous().chunk(2, 1)
            raw, envelope = codec.decoder.istft(torch.polar(log_magnitude.exp(), angle))
            full = codec.decode(codes).audio_values
            cache = cache_cls(config)
            streamed = []
            lengths = [1, codes.shape[-1] - 1]
            start = 0
            for count in lengths:
                streamed.append(codec.decode(codes[..., start:start + count], streaming_cache=cache,
                                              finalize=False).audio_values)
                start += count
            streamed.append(codec.decode(codes[..., :0], streaming_cache=cache, finalize=True).audio_values)
            tensors.update({f'{name}.pcm': pcm.unsqueeze(1), f'{name}.features': features,
                            f'{name}.latents': latents, f'{name}.codes': codes.to(torch.int32),
                            f'{name}.decoded': decoded, f'{name}.log_magnitude': log_magnitude,
                            f'{name}.angle': angle, f'{name}.raw_audio': raw.unsqueeze(-1),
                            f'{name}.raw_envelope': envelope.reshape(1, -1, 1),
                            f'{name}.full_pcm': full.unsqueeze(1),
                            f'{name}.streamed_pcm': torch.cat(streamed, -1).unsqueeze(1)})
            cases.append({'id': name, 'samples': samples, 'codec_frames': codes.shape[-1],
                          'code_chunks': lengths, 'spectral_chunks': [n * config.stft_frame_stack for n in lengths],
                          'streamed_samples': [v.shape[-1] for v in streamed],
                          'reference_streaming_maximum_error': (full - torch.cat(streamed, -1)).abs().max().item()})
    save_file({name: value.contiguous() for name, value in tensors.items()}, output / 'reference.safetensors')
    report = {'repository': manifest['repository'], 'revision': manifest['revision'],
              'source_sha256': manifest['weights_sha256'], 'derived_sha256': digest(derived / 'model.safetensors'),
              'dtype': 'float32', 'attention': 'eager', 'maximum_absolute_tolerance': manifest['maximum_absolute_tolerance'],
              'cases': cases, 'package_versions': {name: importlib.metadata.version(name)
                  for name in ('torch', 'transformers', 'safetensors', 'huggingface_hub')}}
    (output / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
