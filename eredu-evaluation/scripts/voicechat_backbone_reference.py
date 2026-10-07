#!/usr/bin/env python3
"""Pinned NVIDIA fusion/final-normalization/dual-head boundary oracle.

Executes original AddFusion, NemotronHRMSNorm, DuplexSTTModel.forward and prompt
preparation methods. The llm stand-in returns the final-normalized supplied
residual, without executing transformer/Mamba blocks. Source hashes and that
limitation are explicit in the fixture. No Eredu code is imported.
"""
import argparse
import ast
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

SPEECH_REVISION = '097dfe9e2f55baf653b83035868bdc89849f1b47'
BASE_REVISION = '6533e8de2c68e4536bf7c411d7a3ce5734111476'
SOURCES = {
    'nemo/collections/speechlm2/parts/fusion.py': 'fc8ed0fcd0c31364df93e6bda37bda4d6fea320fc124b044c2ffd97adbe8ea37',
    'nemo/collections/speechlm2/models/duplex_stt_model.py': '415e1e01584bab96d9c059d5b0fe9162adaf19677ef43e565c3afad6e073e74b',
    'nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py':
        'b50aeab4b36285cf041330b315e35a01788e1edb639380bb6105446787cc6a1a',
}
MODEL_HASH = '9c831864760ab5d62787040fbd05039e6dfec4b6239d1fcc96ec635efc114734'


def select(path, namespace, names, parent=None):
    nodes = ast.parse(path.read_bytes()).body
    if parent:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == parent).body
    selected = [node for node in nodes if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def tensor(value):
    return {'shape': list(value.shape), 'values': value.detach().reshape(-1).tolist()}


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('base_modeling', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    for path, expected in SOURCES.items():
        assert hashlib.sha256((args.speech_checkout / path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256(args.base_modeling.read_bytes()).hexdigest() == MODEL_HASH
    torch.set_num_threads(1)
    namespace = dict(torch=torch, nn=nn, Tensor=Tensor, Optional=Optional, Tuple=Tuple, logging=logging)
    select(args.speech_checkout / list(SOURCES)[0], namespace, ['FusionModule', 'AddFusion'])
    select(args.base_modeling, namespace, ['NemotronHRMSNorm'])
    forward = select(args.speech_checkout / list(SOURCES)[1], namespace, ['forward'], 'DuplexSTTModel')['forward']
    wrapper_path = args.speech_checkout / list(SOURCES)[2]
    wrapper_class = next(node.name for node in ast.parse(wrapper_path.read_bytes()).body
                         if isinstance(node, ast.ClassDef) and any(
                             isinstance(method, ast.FunctionDef) and method.name == '_prepare_system_prompt_embeddings'
                             for method in node.body))
    wrapper_methods = select(wrapper_path, namespace, ['_prepare_system_prompt_embeddings', '_get_bos_embedding'], wrapper_class)

    class SuppliedResidualBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_f = namespace['NemotronHRMSNorm'](4, eps=1e-5)
            self.calls = 0
            self.residual = None

        def forward(self, inputs_embeds, return_dict, use_cache, cache_params=None):
            self.calls += 1
            assert inputs_embeds.shape == self.residual.shape
            return {'last_hidden_state': self.norm_f(self.residual), 'cache_params': cache_params}

    class Boundary(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(24, 4)
            self.llm = SuppliedResidualBackbone()
            self.lm_head = nn.Linear(4, 24, bias=False)
            self.function_head = nn.Linear(4, 24, bias=False)
            self.cfg = Config(pretrained_llm='NVIDIA-Nemotron-Nano-9B-v2', cache_key='cache_params')
            self.predict_user_text = False
            self.use_function_head = True
            self.text_pad_id = 12
            self._agent_boosts_printed = True
    Boundary.forward = forward
    model = Boundary().eval()
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(model.named_parameters()):
            values = [.43 * math.sin((index + 1) * .57 + ordinal * 1.31) for index in range(parameter.numel())]
            if name == 'llm.norm_f.weight':
                values = [1.0 + value for value in values]
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))
    fusion = namespace['AddFusion'](4, agent_text_weight=1.0, user_audio_weight=1.0,
                                    user_text_weight=1.0, function_weight=2.0).eval()
    all_parameters = {'stt_model.' + name: tensor(value) for name, value in model.named_parameters()}
    agents = torch.tensor([[12, 7, 3]])
    functions = torch.tensor([[12, 20, 21]])
    audio = torch.tensor([.003913, -.003898, .7311, -.2723, .13, -.29, .41, -.53,
                          -.2227, .3719, -.6713, .8821]).reshape(1, 3, 4)
    residual = torch.tensor([math.sin(i * .37 + .2) * .8 + math.cos(i * .19) * .31
                             for i in range(12)]).reshape(1, 3, 4)

    def run(active, dtype):
        with torch.inference_mode():
            agent = active.embed_tokens(agents)
            function = active.embed_tokens(functions)
            # The streaming wrapper casts perception output before fusion.
            supplied_audio = audio.to(dtype)
            fused = fusion(agent, supplied_audio, function_embeds=function)
            active.llm.residual = residual.to(dtype)
            active.llm.calls = 0
            result = active(fused)
            normalized = active.llm.norm_f(active.llm.residual)
            return {'dtype': str(dtype).split('.')[-1], 'agent_tokens': agents.tolist(),
                    'function_tokens': functions.tolist(), 'audio_input_f32': tensor(audio),
                    'audio_cast': tensor(supplied_audio), 'fused_input': tensor(fused),
                    'residual_hidden': tensor(active.llm.residual), 'normalized_hidden': tensor(normalized),
                    'text_logits': tensor(result['text_logits']), 'function_logits': tensor(result['function_logits']),
                    'function_argmax': result['function_logits'].argmax(-1).tolist(),
                    'backbone_invocations': active.llm.calls}

    f32 = run(model, torch.float32)
    bf16_model = copy.deepcopy(model).bfloat16()
    bf16 = run(bf16_model, torch.bfloat16)
    with torch.inference_mode():
        wrong_boundary = fusion(bf16_model.embed_tokens(agents).float(), audio,
                                function_embeds=bf16_model.embed_tokens(functions).float()).bfloat16()
    bf16['cast_after_fusion_max_difference'] = max(abs(a - b) for a, b in zip(
        bf16['fused_input']['values'], tensor(wrong_boundary)['values']))
    assert bf16['cast_after_fusion_max_difference'] > 0
    bf16_readout_ulps = 0.0
    with torch.inference_mode():
        normalized = bf16_model.llm.norm_f(bf16_model.llm.residual)
        for name, head in [('text_logits', bf16_model.lm_head), ('function_logits', bf16_model.function_head)]:
            widened = nn.functional.linear(normalized.double(), head.weight.double()).bfloat16()
            for actual, expected in zip(bf16[name]['values'], tensor(widened)['values']):
                ulp = 2.0 ** max(-133, math.floor(math.log2(abs(expected))) - 7) if expected else 2.0 ** -133
                bf16_readout_ulps = max(bf16_readout_ulps, abs(actual - expected) / ulp)
    assert bf16_readout_ulps <= 1
    wide = run(copy.deepcopy(model).double(), torch.float64)
    maximum = max(abs(a - b) for key in ['fused_input', 'normalized_hidden', 'text_logits', 'function_logits']
                  for a, b in zip(f32[key]['values'], wide[key]['values']))
    assert maximum < 2e-6

    class Wrapper:
        pass
    for name, method in wrapper_methods.items():
        setattr(Wrapper, name, method)
    wrapper = Wrapper()
    wrapper.model = SimpleNamespace(stt_model=model)
    wrapper.dtype = torch.float32
    wrapper.device = 'cpu'
    wrapper._has_asr_head = False
    wrapper.fusion_module = fusion
    # Token IDs are explicit fixture inputs; no tokenizer fidelity is asserted.
    wrapper.tokenizer = SimpleNamespace(bos_id=1, eos_id=2, text_to_ids=lambda _: [5, 7],
                                        ids_to_text=lambda _: 'tiny supplied tokenizer result')
    prompts = []
    for repeats in [1, 2]:
        wrapper.model_cfg = {'system_prompt_repeat_n': repeats}
        with torch.inference_mode():
            output, length = wrapper._prepare_system_prompt_embeddings('hello')
        prompts.append({'prompt_tokens': [1, 5, 7, 2] * repeats, 'repeat_n': repeats,
                        'feedback_token': 12, 'length': length, 'fused_input': tensor(output)})
    with torch.inference_mode():
        first_feedback = wrapper._get_bos_embedding()
    fixture = {
        'provenance': {'speech_repository': 'https://github.com/NVIDIA-NeMo/Speech', 'speech_revision': SPEECH_REVISION,
                       'base_repository': 'nvidia/NVIDIA-Nemotron-Nano-9B-v2', 'base_revision': BASE_REVISION,
                       'speech_source_sha256': SOURCES, 'base_modeling_sha256': MODEL_HASH,
                       'torch_version': torch.__version__,
                       'scope': 'original fusion, final RMSNorm, dual-head forward and prompt methods; llm stand-in supplies residual; no backbone-block/cache or released-weight parity'},
        'geometry': {'width': 4, 'vocab_size': 24, 'norm_epsilon': 1e-5, 'pad_id': 12, 'bos_id': 1, 'eos_id': 2},
        'fusion_weights': {'agent_text': 1.0, 'user_audio': 1.0, 'function': 2.0},
        'parameters': all_parameters,
        'tolerance': {'absolute': 2e-6, 'relative': 2e-5, 'f32_vs_widened_max_absolute': maximum,
                      'basis': 'same f32 inputs and weights widened; original RMSNorm intentionally accumulates f32; thresholds fixed before Eredu comparison'},
        'f32': f32, 'bf16': bf16, 'prompts': prompts,
        'first_agent_feedback': {'token': 12, 'embedding': tensor(first_feedback)},
        'bf16_comparison': {'fusion': 'exact BF16-rounded input/fusion values',
                            'readout': 'at most one BF16 ULP plus f32 tolerance for reduction-order differences',
                            'readout_vs_f64_accumulator_max_ulps': bf16_readout_ulps},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + '\n')


if __name__ == '__main__':
    main()
