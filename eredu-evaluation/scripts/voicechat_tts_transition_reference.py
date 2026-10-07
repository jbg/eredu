#!/usr/bin/env python3
"""Original cached EAR-TTS composition with actual pinned HF Gemma3 blocks.

Default mode supplies subword embeddings at an explicit character-encoder boundary;
--character-encoder runs original character conditioning with actual T5Gemma.
All generation methods, MoG equations, RVQ decisions and EOS input replacement
execute original NVIDIA methods; random uniforms/Gaussians are recorded inputs.
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
from unittest.mock import patch

import torch
import transformers
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import Gemma3TextConfig, Gemma3TextModel, T5GemmaConfig, T5GemmaEncoderModel
from voicechat_ingress_reference import Config
from voicechat_mog_reference import extract, tensor, HASHES, MODEL, PRECISION, REVISION, WARPERS_HASH
from voicechat_transformer_reference import SOURCE_HASHES

DUPLEX = 'nemo/collections/speechlm2/models/duplex_ear_tts.py'


class Output(dict):
    __getattr__ = dict.__getitem__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--character-encoder', action='store_true',
                        help='Execute original character mapping/flags with the real T5Gemma encoder.')
    args = parser.parse_args()
    assert transformers.__version__ == '4.56.0'
    package = Path(transformers.__file__).parent
    hashes = dict(HASHES)
    hashes[DUPLEX] = '63c2dc41bab12a844d1f301a09e673c6d49ce6c406932068334b1e8afea2390c'
    for path, digest in hashes.items():
        assert hashlib.sha256((args.speech_checkout / path).read_bytes()).hexdigest() == digest, path
    for path, digest in SOURCE_HASHES.items():
        assert hashlib.sha256((package / path).read_bytes()).hexdigest() == digest, path
    logits_source = package / 'generation/logits_process.py'
    assert hashlib.sha256(logits_source.read_bytes()).hexdigest() == WARPERS_HASH
    torch.set_num_threads(1)
    torch.manual_seed(0)
    namespace = dict(torch=torch, Tensor=Tensor, nn=nn, F=F, math=math, Any=Any, Cache=Any,
                     contextmanager=contextmanager, RVQEARTTSOutput=Output,
                     add_start_docstrings=lambda *args: lambda function: function,
                     LOGITS_PROCESSOR_INPUTS_DOCSTRING='')
    extract(logits_source, ['LogitsProcessor', 'TopPLogitsWarper', 'TopKLogitsWarper'], namespace)
    extract(args.speech_checkout / PRECISION, ['fp32_precision'], namespace)
    extract(args.speech_checkout / MODEL,
            ['RMSNorm', 'MLP', 'MLPLayer', 'MoGHead', 'depthsum_encoding_step',
             'gumbel_like', 'get_masking_rate', 'batch_matmul_pytorch', 'GatedProjectedSumRMSNorm'], namespace)
    namespace['batch_matmul'] = namespace['batch_matmul_pytorch']
    methods = ['forward', '_prepare_conditioning', 'depthsum_embedding', 'generate_step']
    extract(args.speech_checkout / MODEL, methods, namespace, 'RVQEARTTSModel')
    extract(args.speech_checkout / DUPLEX, ['infer_codes_one_step'], namespace, 'DuplexEARTTS')
    extract(args.speech_checkout / DUPLEX, ['replace_control_speech_codes'], namespace)
    geometry = Gemma3TextConfig(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        sliding_window=3, layer_types=['sliding_attention', 'full_attention'],
        attention_dropout=0., use_cache=True)
    geometry._attn_implementation = 'eager'

    class Tokenizer:
        tokens = ['<bos>', 'a', '<live-eos>', 'Ġab', 'b', 'Ġ', '▁', 'x', '!', '<', '>', 's',
                  '<eos>', 'Ωab', '??', 'xx', '<tail>']
        bos_token, eos_token = '<bos>', '<eos>'
        vocab_size = len(tokens)
        def __init__(self):
            self.tokenizer = self
            self.vocab = self.get_vocab()
        def get_vocab(self):
            return {token: index for index, token in enumerate(self.tokens)}
        def ids_to_tokens(self, index):
            return self.tokens[index]

    if args.character_encoder:
        namespace.update(AutoTokenizer=object, DictConfig=object,
                         logging=SimpleNamespace(info=lambda *args: None))
        extract(args.speech_checkout / MODEL,
                ['sequence_mask', 'build_vocabs', 'SubwordFlagEmbedding', 'BOSEOSEmbedding',
                 'CharAwareSubwordEncoder'], namespace)
    character_config = T5GemmaConfig(encoder=dict(vocab_size=16, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        sliding_window=16, layer_types=['sliding_attention', 'full_attention'],
        attention_dropout=0., use_cache=False), is_encoder_decoder=False, vocab_size=16)
    character_config._attn_implementation = 'eager'

    def character_encoder():
        tokenizer = Tokenizer()
        char = namespace['CharAwareSubwordEncoder'].__new__(namespace['CharAwareSubwordEncoder'])
        nn.Module.__init__(char)
        char.subword_id_to_char_ids, char.char_vocab, char.subword_padding_idx = namespace['build_vocabs'](tokenizer)
        char.char_padding_idx = len(char.char_vocab)
        char.hidden_size = 8
        char.use_subword_flag_emb = char.use_bos_eos_emb = True
        char.embed_tokens = nn.Embedding(char.char_padding_idx + 1, 8, padding_idx=char.char_padding_idx)
        char.proj_embedding = nn.Linear(8, 8, bias=False)
        char.subword_flag_emb = namespace['SubwordFlagEmbedding'](tokenizer, 8)
        char.bos_eos_emb = namespace['BOSEOSEmbedding'](tokenizer, 8)
        char.backbone = T5GemmaEncoderModel(character_config)
        return char

    class SuppliedSubwords(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('table', torch.tensor([.29 * math.sin(i * .39 + .2)
                                                        for i in range(16 * 8)]).reshape(16, 8))
        def forward(self, ids, mask):
            return F.embedding(ids, self.table) * mask.unsqueeze(-1)

    class TinyGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 8
            self.config = Config(num_quantizers=31, codebook_size=5, exponent=3., random_target_masking=False,
                disable_eos_prediction=True, use_audio_prompt_frozen_projection=True,
                use_gated_fusion_for_text_audio=True, context_hidden_size=None)
            self.bos_emb = nn.Parameter(torch.empty(8))
            self.null_emb = nn.Parameter(torch.empty(8))
            self.embed_code = nn.Linear(4, 8, bias=False)
            self.embed_context = None
            self.embed_subword = character_encoder() if args.character_encoder else SuppliedSubwords()
            self.gated_fusion_audio_text = namespace['GatedProjectedSumRMSNorm'](8, 8, 8, 31)
            self.backbone = Gemma3TextModel(geometry)
            self.mog_head = namespace['MoGHead'](hidden_size=8, intermediate_size=12, out_size=4,
                num_layers=3, num_predictions=4, low_rank=2, min_log_std=-4., eps=1e-6)
            self.register_buffer('rvq_embs', torch.tensor([.19 * math.sin(i * .39 + .7) + .11 * math.cos(i * .17)
                                                        for i in range(31 * 5 * 4)]).reshape(31, 5, 4))
            self.register_buffer('audio_prompt_projection_W', torch.tensor([.3 * math.sin(i * .31 + .8)
                                                                           for i in range(64)]).reshape(8, 8))
    for method in methods:
        setattr(TinyGenerator, method, namespace[method])
    model = TinyGenerator().eval()
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(model.named_parameters()):
            values = [.19 * math.sin((i + 1) * .47 + ordinal * .83) for i in range(parameter.numel())]
            if name == 'mog_head.proj_logits.weight':
                values = [v * 10 for v in values]
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))
    speaker = torch.tensor([.62 * math.sin(i * .31 + .4) - .13 * math.cos(i * .47)
                            for i in range(5 * 8)]).reshape(1, 5, 8)
    initial = torch.tensor([(i * 3 + i // 31) % 5 for i in range(5 * 31)]).reshape(1, 5, 31)
    initial[:, 2, 5:9] = 5
    silence = torch.tensor([(i * 2 + 1) % 5 for i in range(31)])
    parameters = {'tts_model.tts_model.' + key: tensor(value) for key, value in model.state_dict().items()
                  if (args.character_encoder or not key.startswith('embed_subword.'))
                  and not key.startswith('backbone.embed_tokens.')}
    parameters['tts_model.audio_prompt_latents.Aria'] = tensor(speaker)
    policy = dict(num_iter=8, guidance_scale=.2, top_p_or_k=.95, noise_scale=.001)

    def run(active):
        calls = []
        stages = []
        current = {}
        original_infer = namespace['MoGHead'].infer
        original_batch = namespace['batch_matmul_pytorch']
        original_encode = namespace['depthsum_encoding_step']

        def before_model(module, values, kwargs):
            current.update(input_codes=tensor(kwargs['code']),
                subword_ids=tensor(kwargs['subword_ids']), subword_mask=tensor(kwargs['subword_mask']),
                supplied_subwords=tensor(active.embed_subword(kwargs['subword_ids'], kwargs['subword_mask'])))
        def after_model(module, values, output):
            current.update(hidden=tensor(output.hidden_states), cache=[dict(kind=type(layer).__name__,
                keys=tensor(layer.keys), values=tensor(layer.values), seen=int(layer.get_seq_length()))
                for layer in output.past_key_values.layers])
        def before_fusion(module, values):
            current.update(audio_embedding=tensor(values[0]), conditioning=tensor(values[1]))
        def after_fusion(module, values, output):
            current['fused'] = tensor(output)
        def infer(head, value, guidance_scale=0., top_p_or_k=1.):
            stage = dict(slot=[3, 4, 5, 6, 7][len(stages)], input=tensor(value))
            stages.append(stage)
            hidden = head.mlp_stack(value)
            cond, uncond = hidden.chunk(2)
            stage['logits'] = tensor(head.proj_logits(cond + guidance_scale * (cond - uncond)))
            mu, logs = original_infer(head, value, guidance_scale, top_p_or_k)
            stage.update(mu=tensor(mu), logs=tensor(logs))
            return mu, logs
        def batch(x, w, indices, *args, **kwargs):
            stages[-1]['mixture_indices'] = tensor(indices)
            return original_batch(x, w, indices, *args, **kwargs)
        def uniform(value):
            step = len(calls)
            stage = len(stages) - 1
            draw = torch.tensor([.04 + .91 * ((i * 7 + stage * 3 + step * 5 + 2) % 17) / 17
                                 for i in range(value.numel())], dtype=torch.float32).to(value).reshape(value.shape)
            stages[-1]['uniform_draw'] = tensor(draw)
            return draw
        def gaussian(value):
            stage = len(stages) - 1
            draw = torch.tensor([math.sin(i * .73 + stage * .61 + len(calls) * .37 + .4) * 1.17
                                 for i in range(value.numel())], dtype=torch.float32).to(value).reshape(value.shape)
            stages[-1]['gaussian_draw'] = tensor(draw)
            return draw
        def encode(embs, residual, codes, start=0, count=72):
            result = original_encode(embs, residual, codes, start, count)
            stages[-1].update(latent=tensor(residual), codes=tensor(result))
            return result
        class Wrapper:
            tts_model = active
            cfg = Config(tts_config=active.config)
            codec_silence_tokens = silence
            text_eos_id = 2
            infer_codes_one_step = namespace['infer_codes_one_step']
        hooks = [active.register_forward_pre_hook(before_model, with_kwargs=True),
                 active.register_forward_hook(after_model),
                 active.gated_fusion_audio_text.register_forward_pre_hook(before_fusion),
                 active.gated_fusion_audio_text.register_forward_hook(after_fusion)]
        namespace['batch_matmul'] = batch
        namespace['depthsum_encoding_step'] = encode
        try:
            with torch.inference_mode(), patch.object(namespace['MoGHead'], 'infer', infer), \
                    patch.object(torch, 'rand_like', uniform), patch.object(torch, 'randn_like', gaussian):
                output = active(code=initial, audio_mask=torch.tensor([[False, False, False, False, True]]),
                    subword_ids=torch.tensor([[0, 12, 1, 3, 2]]), subword_mask=torch.tensor([[True, True, True, False, True]]),
                    audio_prompt_latent=speaker.to(active.bos_emb.dtype), use_cache=True, guidance_enabled=True)
                current.update(name='prefill', audio_mask=tensor(torch.tensor([[False, False, False, False, True]])),
                               feedback=tensor(initial[:, -1:]))
                calls.append(current)
                current = {}
                code = initial[:, -1:]
                cache = output.past_key_values
                decode_tokens = [(3, True), (2, True), (12, True)] if args.character_encoder else [(3, True), (2, True), (4, False)]
                for token, mask in decode_tokens:
                    stages = []
                    previous = code
                    code, cache = Wrapper().infer_codes_one_step(torch.tensor([[token]]), torch.tensor([[1]]),
                        torch.tensor([[mask]]), previous, cache, generation_config=policy)
                    current.update(name='decode', previous_codes=tensor(previous), codes=tensor(code), refinement=stages)
                    if args.character_encoder:
                        # Main streaming output keeps the generated waveform but
                        # replaces the following feedback after an EOS token.
                        code = silence.reshape(1, 1, 31) if token == 2 else code
                        current['feedback'] = tensor(code)
                    calls.append(current)
                    current = {}
        finally:
            for hook in hooks:
                hook.remove()
            namespace['batch_matmul'] = original_batch
            namespace['depthsum_encoding_step'] = original_encode
        return calls

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='In CPU autocast')
        calls = run(model)
        wide = run(copy.deepcopy(model).double())
    max_error = 0.
    for narrow, wider in zip(calls, wide):
        for field in ['supplied_subwords', 'audio_embedding', 'conditioning', 'fused', 'hidden']:
            max_error = max(max_error, *[abs(a - b) for a, b in zip(narrow[field]['values'], wider[field]['values'])])
        for field in ['keys', 'values']:
            for a, b in zip(narrow['cache'], wider['cache']):
                max_error = max(max_error, *[abs(x - y) for x, y in zip(a[field]['values'], b[field]['values'])])
        for a, b in zip(narrow.get('refinement', []), wider.get('refinement', [])):
            assert a['mixture_indices'] == b['mixture_indices']
            assert a['codes'] == b['codes']
            for field in ['logits', 'mu', 'logs', 'latent']:
                max_error = max(max_error, *[abs(x - y) for x, y in zip(a[field]['values'], b[field]['values'])])
    # The composed character/TTS Float32-versus-widened oracle measures 6.43e-5;
    # this bound is fixed before comparing any Eredu implementation output.
    absolute_tolerance = 1e-4 if args.character_encoder else 5e-5
    assert max_error < absolute_tolerance, f'independent precision delta: {max_error:.9g}'
    controls = torch.tensor([5, 6, 7])
    control_input = initial.clone()
    control_input[:, 1, 0:3] = controls
    replaced = namespace['replace_control_speech_codes'](control_input, controls, silence)
    result = dict(provenance=dict(speech_revision=REVISION, source_sha256=hashes,
        transformers_version=transformers.__version__, transformers_source_sha256=SOURCE_HASHES,
        transformers_logits_process_sha256=WARPERS_HASH, torch_version=torch.__version__,
        scope='Original NVIDIA cached TTS forward/generation and Duplex step; actual HF Gemma3; supplied subword embedding boundary; synthetic tiny parameters, not released weights or complete VoiceChat.'),
        dimensions=dict(hidden=8, latent=4, intermediate=12, low_rank=2, mixtures=4, mlp_layers=3,
                        codebooks=31, cardinality=5, speaker_frames=5, min_log_std=-4., eps=1e-6),
        transformer_config=geometry.to_dict(), parameters=parameters, policy=policy, silence=tensor(silence),
        comparison=dict(absolute_tolerance=absolute_tolerance, relative_tolerance=5e-5,
            mixed_float64_linear_float32_norm_max_absolute_error=max_error,
            decisions='Exact mixture and RVQ decisions across both precision runs; explicit F32-rounded draws.'),
        calls=calls, control_replacement=dict(input=tensor(control_input), controls=tensor(controls), output=tensor(replaced)))
    if args.character_encoder:
        result['provenance']['scope'] = ('Original NVIDIA character mapping/flags and cached TTS forward/generation; '
            'actual HF T5Gemma and Gemma3; synthetic tiny parameters, not released weights or complete VoiceChat.')
        char = model.embed_subword
        result['character'] = dict(config=character_config.encoder.to_dict(), vocabulary=Tokenizer.tokens,
            characters=len(char.char_vocab), continuation_flags=tensor(char.subword_flag_emb.is_continuation),
            continuation_pad=char.subword_flag_emb.pad_id, special_flags=tensor(char.bos_eos_emb.special_flags),
            special_pad=char.bos_eos_emb.pad_id)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'{args.output}: cached prefill + three steps; wider arithmetic max error {max_error:.9g}')


if __name__ == '__main__':
    main()
