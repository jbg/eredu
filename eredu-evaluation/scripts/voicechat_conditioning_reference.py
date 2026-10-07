#!/usr/bin/env python3
"""Original EAR-TTS character static conditioning and gated-fusion oracle.

The character transformer is a disclosed fixed affine stub. A separate fixture
validates the real T5Gemma encoder; this fixture isolates lookup/pooling/scatter.
"""
import argparse
import copy
import hashlib
import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicechat_mog_reference import extract, tensor, MODEL, PRECISION, HASHES, REVISION
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    for source, digest in HASHES.items():
        assert hashlib.sha256((args.speech_checkout / source).read_bytes()).hexdigest() == digest
    scope = dict(torch=torch, Tensor=Tensor, nn=nn, F=F, contextmanager=contextmanager,
                 AutoTokenizer=object, DictConfig=object,
                 logging=SimpleNamespace(info=lambda *args: None))
    extract(args.speech_checkout / PRECISION, ['fp32_precision'], scope)
    extract(args.speech_checkout / MODEL,
            ['sequence_mask', 'build_vocabs', 'SubwordFlagEmbedding', 'BOSEOSEmbedding',
             'CharAwareSubwordEncoder', 'RMSNorm', 'GatedProjectedSumRMSNorm'], scope)
    torch.set_num_threads(1)
    torch.manual_seed(0)

    class Tokenizer:
        def __init__(self):
            self.tokens = ['<bos>', 'a', '<live-eos>', 'Ġab', 'b', 'Ġ', '▁', 'x', '!', '<', '>', 's',
                           '<eos>', 'Ωab', '??', 'xx', '<tail>']
            self.vocab = {token: index for index, token in enumerate(self.tokens)}
            self.tokenizer = self
            self.vocab_size = len(self.tokens)
            self.bos_token, self.eos_token = '<bos>', '<eos>'
        def get_vocab(self):
            return self.vocab
        def ids_to_tokens(self, index):
            return self.tokens[index]

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.record = {}
        def forward(self, inputs_embeds, attention_mask):
            hidden = self.linear(inputs_embeds)
            self.record = dict(embedded=inputs_embeds, attention_mask=attention_mask, hidden=hidden)
            return SimpleNamespace(last_hidden_state=hidden)

    tokenizer = Tokenizer()
    char = scope['CharAwareSubwordEncoder'].__new__(scope['CharAwareSubwordEncoder'])
    nn.Module.__init__(char)
    char.subword_id_to_char_ids, char.char_vocab, char.subword_padding_idx = scope['build_vocabs'](tokenizer)
    char.char_padding_idx = len(char.char_vocab)
    char.hidden_size = 4
    char.use_subword_flag_emb = char.use_bos_eos_emb = True
    char.embed_tokens = nn.Embedding(char.char_padding_idx + 1, 4, padding_idx=char.char_padding_idx)
    char.proj_embedding = nn.Linear(4, 4, bias=False)
    char.subword_flag_emb = scope['SubwordFlagEmbedding'](tokenizer, 4)
    char.bos_eos_emb = scope['BOSEOSEmbedding'](tokenizer, 4)
    char.backbone = Stub()
    fusion = scope['GatedProjectedSumRMSNorm'](4, 4, 4, 31)  # Original fourth positional argument.
    with torch.no_grad():
        for ordinal, (_, parameter) in enumerate(list(char.named_parameters()) + list(fusion.named_parameters())):
            values = [.31 * math.sin(i * .47 + ordinal * .81) + .07 * math.cos(i * .23)
                      for i in range(parameter.numel())]
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))
    tokens = torch.tensor([[0, 12, 2, 13], [17, 999, 15, 3]])
    mask = torch.tensor([[True, True, False, True], [True, True, False, True]])
    char_ids, char_lengths = char.prepare_inputs(tokens, mask)
    audio = torch.tensor([.8 * math.sin(i * .39) - .13 for i in range(32)]).reshape(2, 4, 4)

    def run(character, combine, audio):
        output = character(tokens, mask)
        return dict(character_output=output, fusion_output=combine(audio, output),
                    **character.backbone.record)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run(char, fusion, audio)
        wide = run(copy.deepcopy(char).double(), copy.deepcopy(fusion).double(), audio.double())
    delta = max((result[key].double() - wide[key].double()).abs().max().item()
                for key in result if key != 'attention_mask')
    assert delta < 2e-6
    parameters = {'tts_model.tts_model.embed_subword.' + name: tensor(value)
                  for name, value in char.named_parameters() if not name.startswith('backbone.')}
    parameters.update({'tts_model.tts_model.gated_fusion_audio_text.' + name: tensor(value)
                       for name, value in fusion.named_parameters()})
    payload = dict(provenance=dict(revision=REVISION, hashes=HASHES, pytorch=torch.__version__,
                                  backbone='Fixed nonzero affine stub; not T5Gemma validation'),
                   vocabulary=tokenizer.tokens, geometry=dict(characters=len(char.char_vocab), width=4),
                   parameters=parameters, tokens=tensor(tokens), selected=tensor(mask),
                   character_ids=tensor(char_ids), character_lengths=tensor(char_lengths),
                   continuation_flags=tensor(char.subword_flag_emb.is_continuation),
                   continuation_pad=char.subword_flag_emb.pad_id,
                   special_flags=tensor(char.bos_eos_emb.special_flags), special_pad=char.bos_eos_emb.pad_id,
                   audio=tensor(audio), **{key: tensor(value) for key, value in result.items()},
                   tolerance=dict(absolute=3e-6, relative=3e-5, f32_vs_widened_max_absolute=delta,
                                  widening='Original RMSNorm retains explicit Float32 conversion'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(f'Wrote {args.output}; maximum widening delta {delta:.9g}')


if __name__ == '__main__':
    main()
