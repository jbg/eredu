#!/usr/bin/env python3
"""Pinned EAR-TTS ingress oracle with disclosed backbone/character stand-ins.

Executes original forward, conditioning, depthsum and gated fusion methods.
A backbone recorder returns its supplied embeddings; a subword stand-in
returns a supplied table. No Gemma, tokenizer or character-model parity claim.
"""
import argparse
import copy
import hashlib
import json
import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from voicechat_mog_reference import extract, tensor, HASHES, MODEL, PRECISION, REVISION


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    for path, digest in HASHES.items():
        assert hashlib.sha256((args.speech_checkout / path).read_bytes()).hexdigest() == digest
    torch.set_num_threads(1)
    torch.manual_seed(0)
    namespace = dict(torch=torch, Tensor=Tensor, nn=nn, F=F, Any=Any, Cache=Any,
                     contextmanager=contextmanager, RVQEARTTSOutput=SimpleNamespace)
    extract(args.speech_checkout / PRECISION, ['fp32_precision'], namespace)
    extract(args.speech_checkout / MODEL, ['RMSNorm', 'GatedProjectedSumRMSNorm'], namespace)
    extract(args.speech_checkout / MODEL, ['forward', '_prepare_conditioning', 'depthsum_embedding'],
            namespace, 'RVQEARTTSModel')

    class SuppliedSubwords(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('table', torch.tensor([.29 * math.sin(i * .39 + .2) for i in range(20)]).reshape(5, 4))
        def forward(self, ids, mask):
            return F.embedding(ids, self.table) * mask.unsqueeze(-1)

    class Recorder(nn.Module):
        def forward(self, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache):
            self.record = dict(inputs=tensor(inputs_embeds),
                               attention_mask=None if attention_mask is None else tensor(attention_mask),
                               position_ids=None if position_ids is None else tensor(position_ids),
                               use_cache=use_cache)
            return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=past_key_values)

    class Boundary(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 4
            self.config = Config(use_audio_prompt_frozen_projection=True, use_gated_fusion_for_text_audio=True)
            self.bos_emb = nn.Parameter(torch.empty(4))
            self.null_emb = nn.Parameter(torch.empty(4))
            self.embed_code = nn.Linear(4, 4, bias=False)
            self.embed_context = None
            self.embed_subword = SuppliedSubwords()
            self.gated_fusion_audio_text = namespace['GatedProjectedSumRMSNorm'](4, 4, 4, 31)
            self.backbone = Recorder()
            self.register_buffer('rvq_embs', torch.tensor([.14 * math.sin(i * .43 + .1)
                                                         for i in range(31 * 5 * 4)]).reshape(31, 5, 4))
            self.register_buffer('audio_prompt_projection_W', torch.tensor([.3 * math.sin(i * .31 + .8)
                                                                           for i in range(16)]).reshape(4, 4))
    for name in ['forward', '_prepare_conditioning', 'depthsum_embedding']:
        setattr(Boundary, name, namespace[name])
    model = Boundary().eval()
    with torch.no_grad():
        for ordinal, (_, parameter) in enumerate(model.named_parameters()):
            values = [.27 * math.sin(i * .47 + ordinal * .83 + .3) for i in range(parameter.numel())]
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))
    parameters = {'tts_model.tts_model.' + key: tensor(value) for key, value in model.state_dict().items()
                  if not key.startswith('embed_subword.')}
    code = torch.tensor([(i * 3 + i // 31) % 5 for i in range(5 * 31)]).reshape(1, 5, 31)
    code[:, 3, 5:9] = 5  # Ordinary PAD entries contribute exactly zero.
    audio_mask = torch.tensor([[False, True, True, False, True]])
    subwords = torch.tensor([[0, 1, 2, 3, 4]])
    subword_mask = torch.tensor([[True, False, True, True, True]])
    speaker = torch.tensor([.62 * math.sin(i * .31 + .4) - .13 * math.cos(i * .47)
                            for i in range(5 * 4)]).reshape(1, 5, 4)

    def run(active, name, provided_speaker=True, recurrent=False, fully_masked=False):
        selected = code[:, -2:] if recurrent else code
        ids = subwords[:, -2:] if recurrent else subwords
        mask = subword_mask[:, -2:] if recurrent else subword_mask
        if fully_masked:
            mask = torch.zeros_like(mask)
        dtype = active.bos_emb.dtype
        audio = None if recurrent else audio_mask
        prompt = speaker.to(dtype) if provided_speaker and not recurrent else None
        positions = torch.arange(selected.shape[1]).unsqueeze(0)
        attention = torch.ones_like(positions)
        fusion_inputs = {}
        def before_fusion(module, values):
            fusion_inputs.update(audio=tensor(values[0]), conditioning=tensor(values[1]))
        hook = active.gated_fusion_audio_text.register_forward_pre_hook(before_fusion)
        try:
            with torch.inference_mode():
                output = active(code=selected, audio_mask=audio, subword_ids=ids, subword_mask=mask,
                                attention_mask=attention, position_ids=positions, training=False,
                                guidance_enabled=True, use_cache=True, audio_prompt_latent=prompt)
        finally:
            hook.remove()
        return dict(name=name, code=tensor(selected), audio_mask=None if audio is None else tensor(audio),
                    subword_ids=tensor(ids), subword_mask=tensor(mask),
                    supplied_subwords=tensor(active.embed_subword(ids, mask)),
                    supplied_speaker=None if prompt is None else tensor(prompt),
                    audio_embedding=fusion_inputs['audio'], conditioning=fusion_inputs['conditioning'],
                    backbone=active.backbone.record, output=tensor(output.hidden_states),
                    retained_speaker=None if output.audio_prompt_latent is None else tensor(output.audio_prompt_latent))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='In CPU autocast')
        cases = [run(model, 'retained_speaker'), run(model, 'computed_speaker', False),
                 run(model, 'recurrent', recurrent=True), run(model, 'fully_masked', fully_masked=True)]
        wider = run(copy.deepcopy(model).double(), 'retained_speaker')
    error = max(abs(a - b) for a, b in zip(cases[0]['output']['values'], wider['output']['values']))
    assert error < 2e-5
    document = dict(provenance=dict(speech_revision=REVISION, source_sha256=HASHES, torch_version=torch.__version__,
        scope='Original EAR-TTS forward/conditioning/fusion with supplied subword-table and backbone recorder; no character/Gemma model or released-weight parity.'),
        dimensions=dict(hidden=4, latent=4, codebooks=31, cardinality=5), parameters=parameters,
        comparison=dict(absolute_tolerance=2e-5, relative_tolerance=5e-5,
                        mixed_float64_linear_float32_rms_max_absolute_error=error), cases=cases)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(f'{args.output}: wider arithmetic output difference {error:.9g}')


if __name__ == '__main__':
    main()
