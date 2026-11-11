#!/usr/bin/env python3
"""Pinned LFM3 Audio mixed prefill, cached language steps, and one speech frame."""
import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def gguf_depth(root, projector, out, fixtures):
    import gguf

    manifest = json.loads((fixtures / 'audio-gguf-sources.json').read_text())
    expected = manifest['variants']['Q4_K_M']['projector']['publisher_sha256']
    if digest(projector) != expected:
        raise ValueError('projector hash differs from the pinned Q4_K_M artifact')

    def load_class(module, name):
        return get_class_from_dynamic_module(f'{module}.{name}', root, local_files_only=True)

    config = load_class('configuration_lfm3_audio', 'Lfm3AudioConfig').from_pretrained(
        root, local_files_only=True
    )
    config.depth_config._attn_implementation = 'eager'
    model = load_class('modeling_lfm3_audio', 'Lfm3AudioDepthDecoder')(config).eval()
    names = {
        'ln1': 'operator_norm', 'ln2': 'ffn_norm',
        'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj',
        'attn_v': 'self_attn.v_proj', 'attn_out': 'self_attn.o_proj',
        'attn_q_norm': 'self_attn.q_norm', 'attn_k_norm': 'self_attn.k_norm',
        'attn_gate': 'self_attn.gate_proj',
        'ffn_gate': 'feed_forward.w1', 'ffn_down': 'feed_forward.w2', 'ffn_up': 'feed_forward.w3',
    }
    weights = {}
    for tensor in gguf.GGUFReader(projector).tensors:
        name = tensor.name.removeprefix('a.gen.code.')
        if name == tensor.name or name == 'out_embd.weight':
            continue
        value = torch.from_numpy(gguf.dequantize(tensor.data, tensor.tensor_type).copy())
        if name.startswith('blk.'):
            _, layer, field, _ = name.split('.')
            weights[f'model.layers.{layer}.{names[field]}.weight'] = value
        elif name in ('embd.weight', 'head_norm.weight'):
            slot = 'embedding' if name == 'embd.weight' else 'embedding_norm'
            weights.update({f'depth_embeddings.{i}.{slot}.weight': row for i, row in enumerate(value)})
        else:
            key = {
                'mask_embd': 'mask_embedding', 'proj_in.bias': 'depth_linear.bias',
                'proj_in.weight': 'depth_linear.weight', 'output_norm.weight': 'model.embedding_norm.weight',
            }[name]
            weights[key] = value
    model.load_state_dict(weights, strict=True)
    hidden = (torch.arange(config.text_config.hidden_size) % 67 - 33).float()[None] / 32
    codes = torch.zeros((1, 12), dtype=torch.long)
    masked = torch.ones_like(codes, dtype=torch.bool)
    order = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
    with torch.inference_mode(), (out / 'depth-scores.f32').open('wb') as trace:
        for pass_index in range(3):
            logits = model(hidden, codes, masked)
            logits.numpy().astype('<f4', copy=False).tofile(trace)
            slots = order[4 * pass_index:4 * (pass_index + 1)]
            codes[:, slots] = logits[:, slots, :config.audio_eos_code].argmax(-1)
            masked[:, slots] = False


def live_asr(model, extractor, root, wav_path, out, provenance):
    with wave.open(str(wav_path), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError('LiveASR fixture requires mono 16-kHz signed 16-bit WAV')
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype('<f4') / 32768
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
    prompt = tokenizer(
        '<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n',
        add_special_tokens=False, return_tensors='pt',
    ).input_ids
    suppress = tokenizer.convert_tokens_to_ids(['<|mixed_end|>', '<|endoftext|>'])
    stop = tokenizer.convert_tokens_to_ids(['<|endoftext|>', '<|im_start|>'])
    cadence = 8 * model.config.interleaved_n_audio
    blocks = []
    state = None
    features = extractor(pcm, sampling_rate=16000, return_tensors='pt').input_features
    split = [7] + list(range(cadence, features.shape[-1], cadence))
    if len(split) < 3:
        raise ValueError('LiveASR fixture must contain more than two full audio blocks')
    start = 0
    for index, end in enumerate([*split, features.shape[-1]]):
        final = end == features.shape[-1]
        restart = index == 2
        result = model.transcribe_live(
            features[..., start:end], state, input_ids=prompt if state is None or restart else None,
            is_final=final, max_new_tokens=96 if final else 48,
            suppress_token_ids=suppress, stop_token_ids=stop,
        )
        state = result.state
        blocks.append({
            'frames': end - start, 'final': final, 'restart': restart,
            'audio_tokens': result.num_audio_tokens, 'closed': result.closed,
            'tokens': result.sequences[0].tolist(),
        })
        start = end
    pcm.tofile(out / 'live-pcm.f32')
    (out / 'live-reference.json').write_text(json.dumps({
        **provenance,
        'wav_sha256': digest(wav_path), 'blocks': blocks,
    }, indent=2) + '\n')
    print(tokenizer.decode([token for block in blocks for token in block['tokens']],
                           skip_special_tokens=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--live-asr-wav', type=Path, help='compare the publisher LiveASR protocol')
    parser.add_argument('--gguf-depth', type=Path, help='decode the pinned GGUF depth weights into the publisher model')
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
    if args.gguf_depth:
        out.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(4)
        gguf_depth(root, args.gguf_depth, out, fixtures)
        return
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
    if args.live_asr_wav:
        live_asr(model, extractor, root, args.live_asr_wav, out, {
            'model_sha256': model_hash, 'codec_sha256': codec_hash,
        })
        return
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
