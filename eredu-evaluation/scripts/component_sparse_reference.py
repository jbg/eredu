#!/usr/bin/env python3
"""Independent Transformers oracle for the pinned LFM2 sparse checkpoint.

The observer interposes at actual linear inputs, including the eager experts'
down projections. Masks change those inputs before the original linear executes.
No Eredu implementation is imported. Research policy stays in this consumer.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer
from component_sparse_score_reference import score_context, finish_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('provenance', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--trials', nargs='+', default=['baseline', 'deletion', 'keep_only', 'overlay', 'restored'])
    parser.add_argument('--all-layers', action='store_true', help='Capture every component boundary for diagnostics')
    parser.add_argument('--all-positions', action='store_true', help='Include earlier prefill positions in diagnostic captures')
    parser.add_argument('--write-reconstruction', action='store_true', help='Independently reconstruct final-prefill writes and signed component sums in float64')
    parser.add_argument('--score-reconstruction', action='store_true', help='Capture the complete additive readout and reconstruct fixed target scores and margins')
    parser.add_argument('--convolution-fixture', type=Path, help='Save the first deletion decode convolution window for native diagnosis')
    parser.add_argument('--projection-fixture', type=Path, help='Save deletion convolution projection inputs and outputs for native diagnosis')
    args = parser.parse_args()
    if args.score_reconstruction:
        args.all_layers = True
        args.write_reconstruction = True
    assert args.trials[0] == 'baseline'
    provenance = json.loads(args.provenance.read_text())
    checkpoint = Path(provenance['path'])
    for name, digest in provenance['weights_sha256'].items():
        with (checkpoint / name).open('rb') as source:
            assert hashlib.file_digest(source, 'sha256').hexdigest() == digest
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager', experts_implementation='eager')
    model.eval()
    assert model.config.model_type == 'lfm2_moe'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prefix = tokenizer.encode('The capital of France is', add_special_tokens=False)
    layers = model.model.layers
    first_attention = next(i for i, block in enumerate(layers) if hasattr(block, 'self_attn'))
    first_sparse = next(i for i, block in enumerate(layers) if hasattr(block.feed_forward, 'experts'))
    last_sparse = max(i for i, block in enumerate(layers) if hasattr(block.feed_forward, 'experts'))
    selected = [('dense', 0), ('attention', first_attention), ('routed', first_sparse), ('routed', last_sparse)]
    interventions = [dict(kind=kind, layer=layer) for kind, layer in selected if layer != last_sparse]
    intervention_groups = {(item['kind'], item['layer']) for item in interventions}
    if args.all_layers:
        selected = [(kind, i) for i, block in enumerate(layers)
                    for kind in (['attention'] if hasattr(block, 'self_attn') else []) +
                    ['routed' if hasattr(block.feed_forward, 'experts') else 'dense']]
    captured = [dict(kind=kind, layer=layer) for kind, layer in selected]
    result = dict(provenance=provenance, seed=17, dtype='bfloat16', attention='eager',
                  versions={p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'safetensors']},
                  prefix_ids=prefix, captured=captured, interventions=interventions,
                  component_indices=[1, 3], capture_all_positions=args.all_positions,
                  capture_head_input=True, convolution_input_layer=first_sparse + 1,
                  capture_writes=True,
                  capture_full_readout=args.score_reconstruction,
                  diagnostic_boundaries={
                      'block_input': f'model.layers.{first_sparse}.feed_forward.residual',
                      'mixer_residual': f'model.layers.{first_sparse + 1}.mixer.residual',
                      'routed_input': f'model.layers.{first_sparse + 1}.feed_forward.input',
                  },
                  trials={}, queries=[])
    print(json.dumps({'loaded': str(checkpoint), 'prefix': prefix,
                      'expert_bias_dtype': str(layers[first_sparse].feed_forward.expert_bias.dtype)}), flush=True)

    def trial(mode):
        steps = []
        active_bank = [None]
        handles = []
        original_linear = functional.linear
        original_convolution = functional.conv1d
        projection_fixture = {}

        def change(value, token_rows, kind, layer):
            if len(steps) != 1 or mode not in ('deletion', 'keep_only') or (kind, layer) not in intervention_groups:
                return value
            value = value.clone()
            for row in token_rows:
                if mode == 'deletion':
                    value[row, [1, 3]] = 0
                else:
                    kept = value[row, [1, 3]].clone()
                    value[row] = 0
                    value[row, [1, 3]] = kept
            return value

        def dense_hook(kind, layer):
            def hook(module, inputs):
                value = inputs[0]
                before = (value if args.all_positions else value[0, -1]).detach().float().flatten().tolist()
                flattened = value.reshape(-1, value.shape[-1])
                effective = change(flattened, [flattened.shape[0] - 1], kind, layer).reshape_as(value)
                steps[-1]['captures'][f'{kind}:{layer}'] = {
                    'original': before, 'effective': (effective if args.all_positions else effective[0, -1]).detach().float().flatten().tolist()}
                return (effective, *inputs[1:])
            return hook

        def bank_before(layer):
            def hook(module, inputs):
                hidden, indices, weights = inputs
                active_bank[0] = (layer, module, indices, weights)
                steps[-1]['captures'][f'routed:{layer}'] = []
            return hook

        def bank_after(module, inputs, output):
            active_bank[0] = None

        def linear(value, weight, bias=None):
            active = active_bank[0]
            if active is None:
                return original_linear(value, weight, bias)
            layer, module, indices, weights = active
            bank = module.down_proj
            assert bank.is_contiguous()
            offset = weight.data_ptr() - bank.data_ptr()
            stride = bank.stride(0) * bank.element_size()
            if tuple(weight.shape) != tuple(bank.shape[1:]) or not 0 <= offset < bank.numel() * bank.element_size():
                return original_linear(value, weight, bias)
            assert offset % stride == 0
            expert = offset // stride
            slots, tokens = torch.where(indices.T == expert)
            rows = torch.where(tokens == indices.shape[0] - 1)[0].tolist()
            effective = change(value, rows, 'routed', layer)
            for row in (range(len(tokens)) if args.all_positions else rows):
                slot = int(slots[row])
                steps[-1]['captures'][f'routed:{layer}'].append({
                    'token': int(tokens[row]), 'slot': slot, 'expert': expert,
                    'coefficient': float(weights[tokens[row], slot]),
                    'original': value[row].detach().float().tolist(),
                    'effective': effective[row].detach().float().tolist()})
            return original_linear(effective, weight, bias)

        def head_input(module, inputs):
            steps[-1]['head_input'] = inputs[0][0, -1].detach().float().tolist()

        def convolution_input(module, inputs, keywords):
            value = inputs[0] if inputs else keywords['hidden_states']
            value = value if args.all_positions else value[0, -1]
            steps[-1]['convolution_input'] = value.detach().float().flatten().tolist()

        def write_hook(kind, layer):
            def hook(module, inputs, output):
                value = output if args.all_positions else output[0, -1]
                steps[-1].setdefault('writes', {})[f'{kind}:{layer}'] = value.detach().float().flatten().tolist()
            return hook

        def convolution(value, weight, *positional, **keywords):
            output = original_convolution(value, weight, *positional, **keywords)
            if (args.convolution_fixture and mode == 'deletion' and len(steps) == 2
                    and weight.data_ptr() == layers[first_sparse + 1].conv.conv.weight.data_ptr()):
                from safetensors.torch import save_file
                save_file({
                    'input': value[:, :, -weight.shape[-1]:].transpose(1, 2).contiguous(),
                    'weight': weight.transpose(1, 2).contiguous(),
                    'expected': output[:, :, -1:].transpose(1, 2).contiguous(),
                }, args.convolution_fixture,
                    metadata={'mode': mode, 'prediction': '1', 'layer': str(first_sparse + 1)})
            return output

        handles.append(model.lm_head.register_forward_pre_hook(head_input))
        handles.append(layers[first_sparse + 1].conv.register_forward_pre_hook(convolution_input, with_kwargs=True))
        handles.append(layers[first_sparse + 1].conv.register_forward_hook(write_hook('convolution', first_sparse + 1)))
        if args.score_reconstruction:
            def readout_input(module, inputs):
                steps[-1].setdefault('readout', {})['residual'] = inputs[0][0, -1].detach().float().tolist()
            def readout_embedding(module, inputs, output):
                steps[-1].setdefault('readout', {})['embedding'] = output[0, -1].detach().float().tolist()
            handles.append(model.model.embed_tokens.register_forward_hook(readout_embedding))
            handles.append(model.model.embedding_norm.register_forward_pre_hook(readout_input))
            for layer, block in enumerate(layers):
                if hasattr(block, 'conv') and layer != first_sparse + 1:
                    handles.append(block.conv.register_forward_hook(write_hook('convolution', layer)))
        def boundary_hook(key):
            def hook(module, inputs):
                value = inputs[0] if args.all_positions else inputs[0][0, -1]
                steps[-1].setdefault('boundaries', {})[key] = value.detach().float().flatten().tolist()
            return hook
        handles.append(layers[first_sparse + 1].register_forward_pre_hook(boundary_hook('block_input')))
        handles.append(layers[first_sparse + 1].ffn_norm.register_forward_pre_hook(boundary_hook('mixer_residual')))
        handles.append(layers[first_sparse + 1].feed_forward.register_forward_pre_hook(boundary_hook('routed_input')))
        if args.projection_fixture and mode == 'deletion':
            def projection_hook(kind):
                def hook(module, inputs, output):
                    prediction = len(steps) - 1
                    projection_fixture[f'{kind}.input.{prediction}'] = inputs[0].detach().contiguous()
                    projection_fixture[f'{kind}.expected.{prediction}'] = output.detach().contiguous()
                    projection_fixture[f'{kind}.weight'] = module.weight.detach().contiguous()
                return hook
            for kind in ['in', 'out']:
                handles.append(getattr(layers[first_sparse + 1].conv, kind + '_proj').register_forward_hook(projection_hook(kind)))
        for kind, layer in selected:
            block = layers[layer]
            if kind == 'routed':
                handles.append(block.feed_forward.experts.register_forward_pre_hook(bank_before(layer)))
                handles.append(block.feed_forward.experts.register_forward_hook(bank_after))
                handles.append(block.feed_forward.register_forward_hook(write_hook(kind, layer)))
            else:
                projection = block.feed_forward.w2 if kind == 'dense' else block.self_attn.out_proj
                handles.append(projection.register_forward_pre_hook(dense_hook(kind, layer)))
                handles.append(projection.register_forward_hook(write_hook(kind, layer)))
        functional.linear = linear
        functional.conv1d = convolution
        try:
            with torch.inference_mode():
                inputs = torch.tensor([prefix])
                cache = None
                for prediction in range(4):
                    steps.append({'prediction': prediction, 'captures': {}})
                    output = model(input_ids=inputs, past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    logits = output.logits[0, -1].float()
                    token = int(logits.argmax())
                    steps[-1].update(token=token, logits=logits.tolist())
                    for key, value in steps[-1]['captures'].items():
                        if key.startswith('routed:'):
                            value.sort(key=lambda row: (row['token'], row['slot']))
                            positions = inputs.shape[1] if args.all_positions else 1
                            assert len(value) == model.config.num_experts_per_tok * positions
                    inputs = torch.tensor([[token]])
        finally:
            functional.linear = original_linear
            functional.conv1d = original_convolution
            for handle in handles:
                handle.remove()
        if projection_fixture:
            from safetensors.torch import save_file
            save_file(projection_fixture, args.projection_fixture,
                      metadata={'mode': mode, 'layer': str(first_sparse + 1)})
        return steps

    def reconstruct(step):
        reports = {}
        token = len(prefix) - 1 if step['prediction'] == 0 else 0
        context = score_context(model, step, result['score_targets'][step['prediction']]) if args.score_reconstruction else None
        with torch.inference_mode():
            for kind, layer in selected:
                block = layers[layer]
                key = f'{kind}:{layer}'
                if kind == 'routed':
                    routes = [(row['expert'], block.feed_forward.experts.down_proj[row['expert']],
                               row['coefficient'], row['effective'])
                              for row in step['captures'][key] if row['token'] == token]
                    routes.sort(key=lambda row: row[0])
                else:
                    weight = block.feed_forward.w2.weight if kind == 'dense' else block.self_attn.out_proj.weight
                    routes = [(0, weight, 1.0, step['captures'][key]['effective'][-weight.shape[-1]:])]
                width = routes[0][1].shape[0]
                directions = [torch.tensor([(i % 7 - 3) / 8 for i in range(width)], dtype=torch.float64)]
                if context is not None:
                    directions.extend(context['directions'].unbind())
                physical = torch.zeros(width, dtype=torch.bfloat16)
                component_terms, projected_terms = [[] for _ in directions], [[] for _ in directions]
                for expert, weight, coefficient, units in routes:
                    weight = weight.double()
                    units = torch.tensor(units, dtype=torch.float64)
                    write = weight @ units
                    for index, direction in enumerate(directions):
                        columns = direction @ weight
                        component_terms[index].extend((units * columns * coefficient).tolist())
                        projected_terms[index].extend((write * direction * coefficient).tolist())
                    physical += write.bfloat16() * torch.tensor(coefficient, dtype=torch.bfloat16)
                observed = torch.tensor(step['writes'][key][-width:], dtype=torch.float64)
                error = (physical.double() - observed).abs()
                signed_components, signed_projected = math.fsum(component_terms[0]), math.fsum(projected_terms[0])
                signed_observed = float(observed @ directions[0])
                reports[key] = dict(
                    prediction=step['prediction'], token=token, components=len(component_terms[0]),
                    positive_components=sum(value > 0 for value in component_terms[0]),
                    negative_components=sum(value < 0 for value in component_terms[0]),
                    physical_write_exact=bool(torch.equal(physical.double(), observed)),
                    different_write_coordinates=int(torch.count_nonzero(error)),
                    write_max_abs=float(error.max()), write_rms=float(error.square().mean().sqrt()),
                    signed_component_sum=signed_components, signed_projected_write=signed_projected,
                    projection_associativity_error=abs(signed_components - signed_projected),
                    signed_observed_write=signed_observed,
                    physical_rounding_shift=signed_observed - signed_components,
                    score_directions=[dict(signed_component_sum=math.fsum(component_terms[index]),
                                           signed_projected_write=math.fsum(projected_terms[index]),
                                           signed_observed_write=float(observed @ directions[index]),
                                           projection_associativity_error=abs(math.fsum(component_terms[index]) - math.fsum(projected_terms[index])))
                                      for index in range(1, len(directions))])
        report = {'groups': reports, 'projection_precision': 'float64 independent reference'}
        if context is not None:
            report['score_reconstruction'] = finish_scores(model, step, reports, context)
        return report

    baseline = trial('baseline')
    # Prove instrumentation leaves every ordinary cached prediction unchanged.
    with torch.inference_mode():
        inputs, cache = torch.tensor([prefix]), None
        for step in baseline:
            ordinary = model(input_ids=inputs, past_key_values=cache, use_cache=True)
            assert ordinary.logits[0, -1].float().tolist() == step['logits']
            cache = ordinary.past_key_values
            inputs = torch.tensor([[step['token']]])
    result['trials']['baseline'] = baseline
    if args.score_reconstruction:
        result['score_targets'] = [[step['token'], max((token for token in range(len(step['logits'])) if token != step['token']),
                                                     key=lambda token: (step['logits'][token], -token))] for step in baseline]
        result['score_reconstruction'] = {'baseline': [reconstruct(step) for step in baseline]}
        result['write_reconstruction'] = {'baseline': result['score_reconstruction']['baseline'][0]}
    elif args.write_reconstruction:
        result['write_reconstruction'] = {'baseline': reconstruct(baseline[0])}
    expert = next(row['expert'] for row in baseline[0]['captures'][f'routed:{first_sparse}']
                  if row['token'] == len(prefix) - 1 and row['slot'] == 0)
    selections = []
    for kind, layer, roles in [
        ('dense', 0, ['gate', 'value', 'write']),
        ('attention', first_attention, ['query', 'key', 'value', 'write']),
        ('routed', first_sparse, ['gate', 'value', 'write']),
    ]:
        block = layers[layer]
        for role in roles:
            if kind == 'dense':
                weight = getattr(block.feed_forward, {'gate': 'w1', 'value': 'w3', 'write': 'w2'}[role]).weight
                view = weight[:, 1] if role == 'write' else weight[1]
            elif kind == 'attention':
                weight = getattr(block.self_attn, {'query': 'q_proj', 'key': 'k_proj', 'value': 'v_proj', 'write': 'out_proj'}[role]).weight
                view = weight[:, 1] if role == 'write' else weight[1]
            else:
                bank = block.feed_forward.experts
                view = (bank.down_proj[expert, :, 1] if role == 'write' else
                        bank.gate_up_proj[expert, 1 + (model.config.moe_intermediate_size if role == 'value' else 0)])
            before = view.detach().clone()
            delta = torch.tensor([((i + len(selections)) % 5 - 2) / 256 for i in range(view.numel())])
            after = (before.float() + delta).to(view.dtype)
            selector = dict(kind=kind, layer=layer, role=role, unit=1)
            if kind == 'routed':
                selector['expert'] = expert
            coefficients = torch.tensor([(i % 7 - 3) / 8 for i in range(view.numel())], dtype=torch.float64)
            result['queries'].append(dict(selector=selector, original=before.float().tolist(),
                                          replacement=after.float().tolist(),
                                          projection=float(before.double() @ coefficients)))
            selections.append((view, before, after))
    for mode in args.trials[1:]:
        if mode == 'overlay':
            with torch.no_grad():
                for view, before, after in selections:
                    view.copy_(after)
        elif mode == 'restored':
            with torch.no_grad():
                for view, before, after in selections:
                    view.copy_(before)
        result['trials'][mode] = trial(mode)
        if args.score_reconstruction:
            result['score_reconstruction'][mode] = [reconstruct(step) for step in result['trials'][mode]]
            result['write_reconstruction'][mode] = result['score_reconstruction'][mode][0]
        elif args.write_reconstruction:
            result['write_reconstruction'][mode] = reconstruct(result['trials'][mode][0])
        if mode == 'restored':
            assert result['trials'][mode] == baseline
        print(json.dumps({'trial': mode, 'generated': [step['token'] for step in result['trials'][mode]]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + '\n')
    print(json.dumps({'output': str(args.output), 'trials': list(result['trials']),
                      'baseline_generated': [step['token'] for step in baseline]}), flush=True)


if __name__ == '__main__':
    main()
