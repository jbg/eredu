#!/usr/bin/env python3
"""Pinned LFM3 Audio mixed prefill, cached language steps, and one speech frame."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
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
    root = args.checkpoint.resolve()
    out = args.output.resolve()
    checkout = Path(__file__).resolve().parents[1]
    if out.is_relative_to(checkout):
        parser.error('reference payloads must live outside the checkout')
    fixtures = checkout / 'validation/fixtures/lfm3'
    sources = json.loads((fixtures / 'sources.json').read_text())
    audio = next(source for source in sources['sources']
                 if source['repository'] == 'LiquidAI/LFM3-Audio-2B')
    liquid = json.loads((fixtures / 'liquid-sources.json').read_text())
    for name, expected in (audio['source_sha256'] | liquid['source_sha256']).items():
        if digest(root / name) != expected:
            parser.error(f'pinned source hash differs: {name}')
    model_hash = digest(root / 'model.safetensors')
    codec_hash = digest(root / liquid['weight_file'])
    if model_hash != audio['weights_sha256'] or codec_hash != liquid['weights_sha256']:
        parser.error('released model or codec weight hash differs')
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    def cls(module, name):
        return get_class_from_dynamic_module(module + '.' + name, root, local_files_only=True)
    model = cls('modeling_lfm3_audio', 'Lfm3AudioForConditionalGeneration').from_pretrained(
        root, local_files_only=True, dtype=torch.float32, attn_implementation='eager'
    ).eval()
    extractor = cls('feature_extraction_lfm3_audio', 'Lfm3AudioFeatureExtractor').from_pretrained(root / 'preprocessor_config.json')
    cfg = model.config
    codec_cls = cls('modeling_liquid_codec', 'LiquidCodecModel')
    codec = codec_cls.from_pretrained(root / 'audio_tokenizer', local_files_only=True,
                                      dtype=torch.float32, attn_implementation='eager').eval()
    pcm = 0.11 * torch.randn(2151, generator=torch.Generator().manual_seed(671))
    prepared = extractor(pcm, sampling_rate=16000, return_tensors='pt')
    features = prepared.input_features
    length = int(prepared.feature_attention_mask.sum())
    tokens = [cfg.text_config.bos_token_id, 5, cfg.audio_token_id, cfg.audio_token_id, 13,
              cfg.audio_out_token_id, 21, 34]
    assert (length+7)//8 == 2
    codes = torch.tensor([[27,52,81,101,131,271,314,612,814,1439,1512,1718]])
    decode = [55, cfg.audio_out_token_id, 89, 144, 233, cfg.audio_out_token_id, 377, 610]
    trace = {'pcm': pcm, 'features': features, 'codes': codes.to(torch.int32)}
    with torch.inference_mode():
        result = model.model(
            input_ids=torch.tensor([tokens]), input_features=features,
            feature_attention_mask=prepared.feature_attention_mask, audio_out_codes=codes,
            use_cache=True,
        )
        hidden = result.last_hidden_state
        cache = result.past_key_values
        trace['prefill.hidden'] = hidden
        trace['prefill.logits'] = model.lm_head(hidden)
        for index, token in enumerate(decode):
            kwargs = {}
            if token == cfg.audio_out_token_id:
                kwargs['audio_out_codes'] = codes if index == 1 else torch.full_like(codes,cfg.audio_eos_code)
            result = model.model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True, **kwargs)
            cache = result.past_key_values
            trace[f'decode.{index}.hidden'] = result.last_hidden_state
            trace[f'decode.{index}.logits'] = model.lm_head(result.last_hidden_state)
        state = result.last_hidden_state[:, -1]
        generated = torch.zeros_like(codes)
        mask = torch.ones_like(codes, dtype=torch.bool)
        order = [7, 1, 10, 0, 5, 3, 2, 6, 8, 11, 4, 9]
        start = 0
        for index, count in enumerate(cfg.maskgit_schedule):
            group = order[start:start + count]
            depth = model.model.depth_decoder(state, generated, mask)
            trace[f'depth.{index}'] = depth
            generated[:, group] = depth[:, group, :cfg.audio_eos_code].argmax(-1)
            mask[:, group] = False
            start += count
        trace['generated'] = generated.to(torch.int32)
        trace['speech.pcm'] = codec.decode(generated.unsqueeze(-1)).audio_values.unsqueeze(1)
        result = model.model(input_ids=torch.tensor([[cfg.audio_out_token_id]]),
                             past_key_values=cache, use_cache=True, audio_out_codes=generated)
        trace['speech.hidden'] = result.last_hidden_state
        trace['speech.logits'] = model.lm_head(result.last_hidden_state)
    save_file({k:v.contiguous() for k,v in trace.items()}, out/'reference.safetensors')
    metadata = {
        'prompt': tokens, 'decode': decode, 'order': order,
        'model_sha256': model_hash, 'codec_sha256': codec_hash,
        'maximum_absolute_tolerance': 5e-4,
    }
    (out / 'reference.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(tokens, decode, flush=True)


if __name__ == '__main__':
    main()
