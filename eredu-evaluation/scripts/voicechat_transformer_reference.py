#!/usr/bin/env python3
"""Actual Transformers 4.56.0 Gemma3/T5Gemma tiny nonzero numerical oracle.

Run with the pinned package directory on PYTHONPATH. No downloaded weights or
Eredu implementation is imported. Supplied embeddings bypass token lookup.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from transformers import Gemma3TextConfig, Gemma3TextModel, T5GemmaConfig, T5GemmaEncoderModel
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.t5gemma import modeling_t5gemma


SOURCE_HASHES = {'models/gemma3/configuration_gemma3.py': '7d0e83a2bfb99ce0612da5d12abb8296489cea12b8089de87b53cc372b8940d7', 'models/gemma3/modeling_gemma3.py': 'b4d552742a507de1b91770c347a79a9f35c535445f3c6b669274b6edc6dfebc9', 'models/t5gemma/configuration_t5gemma.py': '55cbd61b5750012b7301996cbe3619ea68507de23629132049394b2b4a0309ea', 'models/t5gemma/modeling_t5gemma.py': 'dfa7e6df7670c617fb5304dccadac81a7e6d77575e0ce4fe5b9eac338bc957f1', 'cache_utils.py': '9192c53314b5d6b31bd8ef91e56e559035961a764301fc82e6e7f64365405836', 'masking_utils.py': '5d883a0191406510008219e75b4c9c35ca36a534507286879525a0586f0eb9f8', 'modeling_rope_utils.py': '4365a972bf6ac45729c33a4eb96850e1306221f36977c5dc9b207cb186dd93f9', 'activations.py': '7b83cb02127676a22e6af6855aa7bd6baaa5db9bb5afcdaa843a271552f27773'}

def tensor(value):
    return {'shape': list(value.shape), 'values': value.detach().reshape(-1).tolist()}


def initialize(model):
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(model.named_parameters()):
            values = [.23 * math.sin(i * .41 + ordinal * .67 + .2) for i in range(parameter.numel())]
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert transformers.__version__ == '4.56.0'
    package = Path(transformers.__file__).parent
    for path, digest in SOURCE_HASHES.items():
        assert hashlib.sha256((package / path).read_bytes()).hexdigest() == digest, path
    torch.set_num_threads(1)
    torch.manual_seed(0)
    gemma_config = Gemma3TextConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        sliding_window=3, layer_types=['sliding_attention', 'full_attention'],
        attention_dropout=0., use_cache=True)
    gemma_config._attn_implementation = 'eager'
    gemma = Gemma3TextModel(gemma_config).eval()
    t5_config = T5GemmaConfig(encoder=dict(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        sliding_window=3, layer_types=['sliding_attention', 'full_attention'],
        attention_dropout=0., use_cache=False), is_encoder_decoder=False, vocab_size=16)
    t5_config._attn_implementation = 'eager'
    t5 = T5GemmaEncoderModel(t5_config).eval()
    initialize(gemma)
    initialize(t5)
    causal_input = torch.tensor([.37 * math.sin(i * .29 + .1) + .13 * math.cos(i * .53)
                                for i in range(6 * 8)]).reshape(1, 6, 8)
    char_input = torch.tensor([.31 * math.sin(i * .37 + .6) - .17 * math.cos(i * .23)
                              for i in range(2 * 5 * 8)]).reshape(2, 5, 8)
    char_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])

    def run(model, source, value, lengths, attention_mask=None):
        batches = []
        cache = None
        offset = 0
        original = source.eager_attention_forward
        trace = []

        def attention(module, query, key, val, mask, *args, **kwargs):
            result = original(module, query, key, val, mask, *args, **kwargs)
            trace.append(dict(layer=module.layer_idx, query=tensor(query), key=tensor(key), value=tensor(val),
                              mask=None if mask is None else tensor(mask),
                              scale=float(kwargs['scaling']),
                              probabilities=tensor(result[1]), attention_output=tensor(result[0])))
            return result

        with torch.inference_mode(), patch.object(source, 'eager_attention_forward', attention):
            for length in lengths:
                trace = []
                supplied = value[:, offset:offset + length]
                kwargs = dict(inputs_embeds=supplied)
                if source is modeling_gemma3:
                    kwargs.update(past_key_values=cache, use_cache=True)
                else:
                    kwargs['attention_mask'] = attention_mask
                result = model(**kwargs)
                cache = getattr(result, 'past_key_values', None)
                states = []
                if cache is not None:
                    for layer in cache.layers:
                        states.append(dict(kind=type(layer).__name__, keys=tensor(layer.keys),
                                           values=tensor(layer.values), seen=int(layer.get_seq_length())))
                batches.append(dict(offset=offset, input=tensor(supplied), output=tensor(result.last_hidden_state),
                                    attention=trace, cache=states))
                offset += length
        return batches

    full = run(gemma, modeling_gemma3, causal_input, [6])
    chunks = run(gemma, modeling_gemma3, causal_input, [2, 1, 2, 1])
    chars = run(t5, modeling_t5gemma, char_input, [5], char_mask)
    wide_full = run(copy.deepcopy(gemma).double(), modeling_gemma3, causal_input.double(), [6])
    wide_chars = run(copy.deepcopy(t5).double(), modeling_t5gemma, char_input.double(), [5], char_mask)
    full_values = full[0]['output']['values']
    chunk_values = [v for chunk in chunks for v in chunk['output']['values']]
    cached_difference = max(abs(a - b) for a, b in zip(full_values, chunk_values))
    wide_difference = max(abs(a - b) for x, y in [(full[0], wide_full[0]), (chars[0], wide_chars[0])]
                          for a, b in zip(x['output']['values'], y['output']['values']))
    assert cached_difference < 2e-5
    assert wide_difference < 2e-5
    document = {
        'provenance': {'transformers_version': transformers.__version__, 'torch_version': torch.__version__,
                       'source_sha256': SOURCE_HASHES,
                       'scope': 'Actual HF modules; synthetic tiny weights and supplied embeddings; no released-weight or complete EAR-TTS parity.'},
        'comparison': {'absolute_tolerance': 2e-5, 'relative_tolerance': 5e-5,
                       'whole_vs_cached_max_absolute_error': cached_difference,
                       'mixed_float64_linear_float32_norm_rope_max_absolute_error': wide_difference},
        'gemma3': {'config': gemma_config.to_dict(),
                   'parameters': {'tts_model.tts_model.backbone.' + key: tensor(value)
                                  for key, value in gemma.state_dict().items() if not key.startswith('embed_tokens.')},
                   'full': full, 'chunks': chunks},
        't5gemma': {'config': t5_config.to_dict(),
                    'parameters': {'tts_model.tts_model.embed_subword.backbone.' + key: tensor(value)
                                   for key, value in t5.state_dict().items() if 'embed_tokens.' not in key},
                    'attention_mask': tensor(char_mask), 'batches': chars},
    }
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    print(f'{args.output}: cached difference {cached_difference:.9g}, wider arithmetic {wide_difference:.9g}')


if __name__ == '__main__':
    main()
