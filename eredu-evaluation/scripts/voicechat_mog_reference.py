#!/usr/bin/env python3
"""Original pinned EAR-TTS MoG and iterative RVQ oracle, with recorded random draws.

Tiny learned dimensions; actual released 31-codebook/eight-slot schedule. Does
not execute Gemma, character conditioning, audio codec or released model weights.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REVISION = '097dfe9e2f55baf653b83035868bdc89849f1b47'
MODEL = 'nemo/collections/speechlm2/modules/ear_tts_model.py'
PRECISION = 'nemo/collections/speechlm2/parts/precision.py'
HASHES = {
    MODEL: '787971ac20b3498085aab0e3a228cd5a9aa13b9218b7f2d66f80c5afd513c9c0',
    PRECISION: 'a8f7ecac9d4962efe9ab0d4fe2b7a06b9c735d9fa5e54a52e3635182728c1c5c',
}
WARPERS_HASH = '90e5f20ab9354798e14cb78af2bb0e70c94f32b4c515e02789040d22e0c723ce'


def extract(path, names, namespace, parent=None):
    tree = ast.parse(path.read_bytes())
    if parent:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == parent)
        candidates = tree.body
    else:
        candidates = ast.walk(tree)
    nodes = [node for node in candidates if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    # Names are dependency ordered by their original file position.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def tensor(value):
    values = value.detach().reshape(-1).tolist()
    # JSON has no infinity; filtered logits retain their exact mask separately.
    values = [v if not isinstance(v, float) or math.isfinite(v) else None for v in values]
    return {'shape': list(value.shape), 'values': values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('speech_checkout', type=Path)
    parser.add_argument('transformers_logits_process', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    for path, digest in HASHES.items():
        assert hashlib.sha256((args.speech_checkout / path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256(args.transformers_logits_process.read_bytes()).hexdigest() == WARPERS_HASH
    torch.set_num_threads(1)
    torch.manual_seed(0)  # All initialized parameters are overwritten below.
    namespace = dict(torch=torch, Tensor=Tensor, nn=nn, F=F, math=math, contextmanager=contextmanager,
                     add_start_docstrings=lambda *args: lambda function: function,
                     LOGITS_PROCESSOR_INPUTS_DOCSTRING='')
    extract(args.transformers_logits_process, ['LogitsProcessor', 'TopPLogitsWarper', 'TopKLogitsWarper'], namespace)
    extract(args.speech_checkout / PRECISION, ['fp32_precision'], namespace)
    extract(args.speech_checkout / MODEL,
            ['RMSNorm', 'MLP', 'MLPLayer', 'MoGHead', 'depthsum_encoding_step',
             'gumbel_like', 'get_masking_rate', 'batch_matmul_pytorch'], namespace)
    namespace['batch_matmul'] = namespace['batch_matmul_pytorch']
    extract(args.speech_checkout / MODEL, ['depthsum_embedding', 'generate_step'], namespace, 'RVQEARTTSModel')

    class TinyGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_quantizers=31, codebook_size=5, exponent=3.0,
                                          random_target_masking=False, disable_eos_prediction=True)
            self.embed_code = nn.Linear(4, 4, bias=False)
            self.mog_head = namespace['MoGHead'](hidden_size=4, intermediate_size=8, out_size=4,
                                                num_layers=3, num_predictions=4, low_rank=2,
                                                min_log_std=-4.0, eps=1e-6)
            values = [.19 * math.sin(i * .39 + .7) + .11 * math.cos(i * .17) for i in range(31 * 5 * 4)]
            self.register_buffer('rvq_embs', torch.tensor(values).reshape(31, 5, 4))
    TinyGenerator.generate_step = namespace['generate_step']
    TinyGenerator.depthsum_embedding = namespace['depthsum_embedding']
    model = TinyGenerator().eval()
    with torch.no_grad():
        for ordinal, (name, parameter) in enumerate(model.named_parameters()):
            values = [.19 * math.sin((i + 1) * .47 + ordinal * .83) for i in range(parameter.numel())]
            if name == 'mog_head.proj_logits.weight':
                values = [v * 30 for v in values]  # Top-p genuinely removes mixture candidates.
            if name == 'mog_head.proj_logs.weight':
                values = [-6.1, 4.3, -3.2, 2.7]  # Exercises the released -4 lower clamp.
            parameter.copy_(torch.tensor(values).reshape(parameter.shape))
    params = {'tts_model.tts_model.' + name: tensor(value) for name, value in model.state_dict().items()}
    hidden = torch.tensor([.31, -.57, .83, -.19, -.42, .16, .73, -.61]).reshape(2, 1, 4)
    policy = dict(num_iter=8, guidance_scale=.2, top_p_or_k=.95, noise_scale=.001)
    rates = torch.linspace(0, 1, 9)[:-1].unsqueeze(-1)
    masks = torch.ceil(namespace['get_masking_rate'](rates, 3.) * 31).long()
    counts = masks - F.pad(masks[1:], [0, 0, 0, 1])

    def run(active, supplied_hidden, options):
        steps = []
        original_infer = namespace['MoGHead'].infer
        original_encode = namespace['depthsum_encoding_step']
        original_batch = namespace['batch_matmul_pytorch']
        original_gumbel = namespace['gumbel_like']
        warpers = {name: namespace[name] for name in ['TopPLogitsWarper', 'TopKLogitsWarper']}
        active_slots = [i for i, count in enumerate(counts.reshape(-1).tolist()) if count]

        def infer(head, x, guidance_scale=0., top_p_or_k=1.):
            stage = {'slot': active_slots[len(steps)], 'input': tensor(x),
                     'guidance_scale': guidance_scale, 'top_p_or_k': top_p_or_k}
            steps.append(stage)
            # Expose the original predictor's unguided forward outputs, and
            # guided projections for diagnosing inference independently.
            raw_logits, raw_mus, raw_residual, raw_logs = head(x)
            stage.update(predictor_logits=tensor(raw_logits), predictor_mus=tensor(raw_mus),
                         predictor_residual=tensor(raw_residual), predictor_logs=tensor(raw_logs))
            transformed = head.mlp_stack(x)
            stage['mlp_output'] = tensor(transformed)
            if guidance_scale > 0:
                cond, uncond = transformed.chunk(2)
                transformed = cond + guidance_scale * (cond - uncond)
            stage.update(guided_hidden=tensor(transformed), logits=tensor(head.proj_logits(transformed)),
                         mus=tensor(head.proj_mus(transformed).view(*transformed.shape[:-1], 4, 2)),
                         residual=tensor(head.proj_else(transformed)),
                         unclamped_logs=tensor(head.proj_logs(transformed)))
            mu, logs = original_infer(head, x, guidance_scale, top_p_or_k)
            stage.update(mu=tensor(mu), logs=tensor(logs))
            return mu, logs

        def batch(x, w, y, *args, **kwargs):
            steps[-1]['mixture_indices'] = tensor(y)
            result = original_batch(x, w, y, *args, **kwargs)
            field = 'expanded_selected_mean' if 'selected_mean' in steps[-1] else 'selected_mean'
            steps[-1][field] = tensor(result)
            return result

        def gumbel(value):
            result = original_gumbel(value)
            steps[-1]['gumbel_draw'] = tensor(result)
            return result

        def uniform_like(value):
            index = len(steps) - 1
            draws = torch.tensor([.04 + .91 * ((i * 7 + index * 3 + 2) % 17) / 17
                                  for i in range(value.numel())], dtype=torch.float32).to(value)
            draws = draws.reshape(value.shape)
            steps[-1]['uniform_draw'] = tensor(draws)
            return draws

        def gaussian_like(value):
            index = len(steps) - 1
            draws = torch.tensor([math.sin(i * .73 + index * .61 + .4) * 1.17
                                  for i in range(value.numel())], dtype=torch.float32).to(value)
            draws = draws.reshape(value.shape)
            steps[-1]['gaussian_draw'] = tensor(draws)
            return draws

        def encode(embs, residual, code, start=0, count=72):
            stage = steps[-1]
            stage.update(latent=tensor(residual), depth_start=start, depth_count=count, codes_before=tensor(code))
            result = original_encode(embs, residual, code, start, count)
            stage.update(codes_after=tensor(result), summed_feedback=tensor(active.depthsum_embedding(result)))
            return result

        def recording_warper(base):
            class Warper(base):
                def __call__(self, input_ids, scores):
                    filtered = super().__call__(input_ids, scores)
                    steps[-1].update(filtered_logits=tensor(filtered),
                                     retained=tensor(torch.isfinite(filtered)),
                                     probabilities=tensor(F.softmax(filtered, dim=-1)))
                    return filtered
            return Warper

        namespace['batch_matmul'] = batch
        namespace['gumbel_like'] = gumbel
        namespace['depthsum_encoding_step'] = encode
        for name, original in warpers.items():
            namespace[name] = recording_warper(original)
        try:
            with torch.inference_mode(), patch.object(namespace['MoGHead'], 'infer', infer), \
                    patch.object(torch, 'rand_like', uniform_like), patch.object(torch, 'randn_like', gaussian_like):
                code, lm_logits, eos = active.generate_step(supplied_hidden, **options)
                feedback = active.depthsum_embedding(code)
        finally:
            namespace['batch_matmul'] = original_batch
            namespace['gumbel_like'] = original_gumbel
            namespace['depthsum_encoding_step'] = original_encode
            namespace.update(warpers)
        return dict(policy=options, hidden=tensor(supplied_hidden), steps=steps, codes=tensor(code),
                    summed_feedback=tensor(feedback), lm_logits=None if lm_logits is None else tensor(lm_logits),
                    eos=None if eos is None else tensor(eos))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='In CPU autocast')
        released = run(model, hidden, policy)
        # Supplied draws are quantized once to F32, then identical in both runs.
        wide = run(copy.deepcopy(model).double(), hidden.double(), policy)
        alternatives = run(model, hidden[:1], dict(num_iter=8, guidance_scale=None, top_p_or_k=2, noise_scale=.23))
    max_error = 0.
    for f32, f64 in zip(released['steps'], wide['steps']):
        assert f32['mixture_indices'] == f64['mixture_indices']
        assert f32['codes_after'] == f64['codes_after']
        for field in ['logits', 'mus', 'mu', 'logs', 'latent', 'summed_feedback']:
            errors = [abs(a - b) for a, b in zip(f32[field]['values'], f64[field]['values'])]
            max_error = max(max_error, *errors)
    assert max_error < 2e-5
    assert any(not all(step['retained']['values']) for step in released['steps'])
    assert any(-4.0 in step['logs']['values'] for step in released['steps'])
    assert len(released['steps']) == 5

    # Focused depthsum contract: duplicate entries choose the first argmin;
    # suffix quantization receives a fresh residual and ignores prefix feedback.
    embeddings = torch.tensor([[[.2, -.3], [.2, -.3], [-.4, .7], [.6, .1]],
                               [[.5, -.1], [-.2, .4], [.3, .8], [-.6, -.2]],
                               [[.1, .2], [-.3, -.5], [.7, -.4], [-.8, .6]]])
    residual = torch.tensor([[[.2, -.3], [.53, .24]], [[-.31, .65], [.63, -.43]]])
    codes = torch.full((2, 2, 3), 4, dtype=torch.long)
    full = namespace['depthsum_encoding_step'](embeddings, residual.clone(), codes.clone(), 0, 3)
    prefix = codes.clone()
    prefix[..., 0] = torch.tensor([[2, 1], [3, 0]])
    suffix = namespace['depthsum_encoding_step'](embeddings, residual.clone(), prefix.clone(), 1, 2)
    tiny = SimpleNamespace(rvq_embs=embeddings)
    depthsum = namespace['depthsum_embedding']
    cases = dict(embeddings=tensor(embeddings), residual=tensor(residual), initial_codes=tensor(codes),
                 full_codes=tensor(full), full_sum=tensor(depthsum(tiny, full)),
                 suffix_initial_codes=tensor(prefix), suffix_codes=tensor(suffix),
                 suffix_sum=tensor(depthsum(tiny, suffix)), sentinel_sum=tensor(depthsum(tiny, codes)))
    output = {
        'provenance': {'speech_revision': REVISION, 'source_sha256': HASHES,
                       'transformers_version': '4.56.0', 'transformers_logits_process_sha256': WARPERS_HASH,
                       'torch_version': torch.__version__, 'device': 'cpu',
                       'scope': 'Original AST methods with tiny synthetic nonzero parameters and supplied draws; no Gemma, character encoder, codec or released-weight parity.',
                       'extractions': ['RMSNorm', 'MLP', 'MLPLayer', 'MoGHead', 'gumbel_like',
                                       'get_masking_rate', 'batch_matmul_pytorch', 'depthsum_encoding_step',
                                       'RVQEARTTSModel.generate_step', 'RVQEARTTSModel.depthsum_embedding',
                                       'fp32_precision', 'TopPLogitsWarper', 'TopKLogitsWarper']},
        'dimensions': {'hidden': 4, 'intermediate': 8, 'latent': 4, 'low_rank': 2,
                       'mixtures': 4, 'mlp_layers': 3, 'codebooks': 31, 'codebook_size': 5, 'eps': 1e-6, 'min_log_std': -4.},
        'comparison': {'absolute_tolerance': 2e-5, 'relative_tolerance': 5e-5,
                       'mixed_float64_linear_rvq_float32_rms_max_absolute_error': max_error,
                       'draws': 'Explicit uniform and Gaussian values quantized to Float32 before either run.',
                       'decisions': 'Exact mixture and RVQ code equality across both precision runs.'},
        'parameters': params, 'mask_counts': masks.reshape(-1).tolist(), 'depth_counts': counts.reshape(-1).tolist(),
        'released_policy': released, 'top_k_without_guidance': alternatives, 'depthsum_cases': cases,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(f'{args.output}: {len(released["steps"])} active iterations; mixed-precision max error {max_error:.9g}')


if __name__ == '__main__':
    main()
