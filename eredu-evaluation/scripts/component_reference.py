#!/usr/bin/env python3
"""Independent CPU reference for pinned dense and mixed-state component fixtures.

This consumer uses Transformers internals only as an independent oracle; Eredu
consumers must use public discovery, capture and parameter-operation contracts.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('provenance', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--f32-checkpoint', type=Path,
                        help='export an exact BF16/F16-to-F32 derivative for native F32 parity')
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text())
    checkpoint = Path(provenance['path'])
    with (checkpoint / 'model.safetensors').open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == provenance['weights_sha256']
    if args.f32_checkpoint:
        destination = args.f32_checkpoint.resolve()
        assert destination != checkpoint.resolve()
        if destination.is_relative_to(Path(__file__).resolve().parents[2]):
            parser.error('derived checkpoint must live outside the repository')
        destination.mkdir(parents=True, exist_ok=True)
        for sidecar in checkpoint.iterdir():
            if sidecar.is_file() and sidecar.suffix in {'.json', '.jinja'}:
                shutil.copyfile(sidecar, destination / sidecar.name)
        tensors = load_file(str(checkpoint / 'model.safetensors'))
        save_file({name: value.float() if value.is_floating_point() else value
                   for name, value in tensors.items()},
                  str(destination / 'model.safetensors'), metadata={'format': 'pt'})
        del tensors
        with (destination / 'model.safetensors').open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        provenance = {'repository': provenance['repository'], 'revision': provenance['revision'],
                      'source': provenance, 'path': str(destination),
                      'weights_sha256': digest,
                      'conversion': 'exact widening of stored floating values to F32; no quantization or parameter edits'}
        checkpoint = destination
    torch.manual_seed(17)
    torch.set_num_threads(2)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                               dtype=torch.float32, attn_implementation='eager')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    ids = tokenizer.encode('The capital of France is', add_special_tokens=False)
    tokens = torch.tensor([ids], dtype=torch.long)
    mixed = model.config.model_type == 'lfm2'
    first_attention = next(i for i, block in enumerate(model.model.layers) if hasattr(block, 'self_attn'))
    selected = sorted({0, first_attention, len(model.model.layers) - 1})
    intervention_layers = sorted({0, first_attention})
    ffn_fields = ('feed_forward.w1', 'feed_forward.w3', 'feed_forward.w2') if mixed else (
        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    attention_output = 'out_proj' if mixed else 'o_proj'

    def ffn(block):
        return block.feed_forward if mixed else block.mlp

    def down(block):
        return ffn(block).w2 if mixed else ffn(block).down_proj

    parameter_edits = []
    for ordinal, (layer, suffix, column) in enumerate([
        (first_attention, 'self_attn.q_proj.weight', False),
        (first_attention, 'self_attn.k_proj.weight', False),
        (first_attention, 'self_attn.v_proj.weight', False),
        (first_attention, 'self_attn.' + attention_output + '.weight', True),
        (0, ffn_fields[0] + '.weight', False), (0, ffn_fields[1] + '.weight', False),
        (0, ffn_fields[2] + '.weight', True),
    ]):
        name = f'model.layers.{layer}.' + suffix
        weight = model.get_parameter(name)
        shape = [weight.shape[0], 1] if column else [1, weight.shape[1]]
        count = shape[0] * shape[1]
        deltas = torch.tensor([((index + ordinal) % 5) * 0.002 - 0.003
                               for index in range(count)], dtype=torch.float32)
        parameter_edits.append({'id': f'edit-{ordinal}', 'parameter': name,
            'parameter_shape': list(weight.shape), 'dtype': 'float32',
            'region': {'starts': [0, 1] if column else [1, 0], 'shape': shape},
            'update': {'kind': 'add', 'values': deltas.tolist()}})

    def trial(keep_only, edit_parameters=False, delete=False):
        captures = [{}]
        handles = []
        originals = {}
        if edit_parameters:
            with torch.no_grad():
                for edit in parameter_edits:
                    weight = model.get_parameter(edit['parameter'])
                    originals[edit['parameter']] = weight.detach().clone()
                    row, col = edit['region']['starts']
                    rows, cols = edit['region']['shape']
                    weight[row:row+rows, col:col+cols] += torch.tensor(
                        edit['update']['values'], dtype=weight.dtype).reshape(rows, cols)

        def capture_input(path, masked):
            def hook(module, inputs):
                value = inputs[0]
                captures[-1][path] = value[0, -1].detach().double().tolist()
                if masked and len(captures) == 1:
                    effective = value.clone()
                    if delete:
                        effective[0, -1, [1, 3]] = 0.0
                    else:
                        effective[0, -1, :] = 0.0
                        effective[0, -1, [1, 3]] = value[0, -1, [1, 3]]
                else:
                    effective = value
                captures[-1][path + '.effective'] = effective[0, -1].detach().double().tolist()
                return (effective, *inputs[1:])
            return hook

        for layer in selected:
            block = model.model.layers[layer]
            modules = [('feed_forward.units', down(block))]
            if hasattr(block, 'self_attn'):
                modules.append(('attention.channels', getattr(block.self_attn, attention_output)))
            for suffix, module in modules:
                handles.append(module.register_forward_pre_hook(
                    capture_input(f'model.layers.{layer}.{suffix}', keep_only and layer in intervention_layers)))
        try:
            with torch.inference_mode():
                output = model(input_ids=tokens, use_cache=True)
                logits = output.logits[0, -1].double()
                ordered = logits.argsort(descending=True)
                target = int(ordered[0])
                cache = output.past_key_values
                generated = [target]
                decode = []
                for _ in range(3):
                    captures.append({})
                    next_output = model(input_ids=torch.tensor([[generated[-1]]]), past_key_values=cache, use_cache=True)
                    cache = next_output.past_key_values
                    row = next_output.logits[0, -1].double()
                    generated.append(int(row.argmax()))
                    decode.append({'token': generated[-1], 'score': float(row.max())})
                return {'prefill': {'target': target, 'target_score': float(logits[target]),
                        'competitor': int(ordered[1]), 'competitor_score': float(logits[ordered[1]]),
                        'target_log_probability': float(logits[target] - logits.logsumexp(0)),
                        'logits': logits.tolist()}, 'generated': generated,
                        'decode': decode, 'captures': captures}
        finally:
            for handle in handles:
                handle.remove()
            with torch.no_grad():
                for name, original in originals.items():
                    model.get_parameter(name).copy_(original)

    # Separate evidence for prefill and each cached prediction. The keep-only
    # intervention applies only to the last prompt row of prediction zero.
    baseline = trial(False)
    keep_only = trial(True)
    deletion = trial(True, delete=True)
    overlay = trial(False, True)
    restored = trial(False)
    assert restored == baseline, 'reference parameter restoration must recover every captured value'

    # Illustrative single-component association. Only trigger/non-trigger inputs fit
    # the read row; the third prefix is held out. This is plumbing evidence, not an
    # efficacy claim or an implementation of the paper's calibration/search procedure.
    prompts = [('trigger', 'The capital of France is'),
               ('non_trigger', 'The capital of Germany is'),
               ('held_out', 'A short poem about the sea:')]
    last = len(model.model.layers) - 1
    unit = 1
    def association_case(prefix):
        captured = {}
        def before_mlp(module, inputs):
            captured['input'] = inputs[0][0, -1].detach().double().tolist()
        def before_down(module, inputs):
            captured['units'] = inputs[0][0, -1].detach().double().tolist()
        handles = [ffn(model.model.layers[last]).register_forward_pre_hook(before_mlp),
                   down(model.model.layers[last]).register_forward_pre_hook(before_down)]
        try:
            with torch.inference_mode():
                output = model(input_ids=torch.tensor([prefix]), use_cache=True)
                captured['logits'] = output.logits[0, -1].double().tolist()
                captured['winner'] = int(output.logits[0, -1].argmax())
            return captured
        finally:
            for handle in handles:
                handle.remove()
    cases = [{'kind': kind, 'text': text,
              'prefix_ids': tokenizer.encode(text, add_special_tokens=False)} for kind, text in prompts]
    for case in cases:
        case['baseline'] = association_case(case['prefix_ids'])
    inputs = torch.tensor([case['baseline']['input'] for case in cases[:2]], dtype=torch.float64)
    coefficients = torch.linalg.solve(inputs @ inputs.T, torch.tensor([1.0, 0.0], dtype=torch.float64))
    read = (inputs.T @ coefficients).float()
    desired = tokenizer.encode(' London', add_special_tokens=False)
    assert len(desired) == 1
    target = desired[0]
    head = model.get_output_embeddings().weight
    direction = head[target].detach().double() - head[cases[0]['baseline']['winner']].detach().double()
    write = (16.0 * direction / direction.norm()).float()
    association_edits = []
    for suffix, column, values in [(ffn_fields[0] + '.weight', False, read),
                                   (ffn_fields[1] + '.weight', False, read),
                                   (ffn_fields[2] + '.weight', True, write)]:
        name = f'model.layers.{last}.{suffix}'
        weight = model.get_parameter(name)
        association_edits.append({'id': suffix, 'parameter': name,
            'parameter_shape': list(weight.shape), 'dtype': 'float32',
            'region': {'starts': [0, unit] if column else [unit, 0],
                       'shape': [weight.shape[0], 1] if column else [1, weight.shape[1]]},
            'update': {'kind': 'replace', 'values': values.tolist()}})
    originals = {}
    with torch.no_grad():
        for edit in association_edits:
            weight = model.get_parameter(edit['parameter'])
            originals[edit['parameter']] = weight.detach().clone()
            row, col = edit['region']['starts']
            rows, cols = edit['region']['shape']
            weight[row:row+rows, col:col+cols] = torch.tensor(edit['update']['values']).reshape(rows, cols)
    for case in cases:
        case['edited'] = association_case(case['prefix_ids'])
    with torch.no_grad():
        for name, value in originals.items():
            model.get_parameter(name).copy_(value)
    for case in cases:
        assert association_case(case['prefix_ids']) == case['baseline']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'provenance': provenance, 'seed': 17,
        'versions': {p: importlib.metadata.version(p) for p in ['torch','transformers','numpy','safetensors']},
        'dtype': 'float32', 'attention': 'eager', 'prefix_ids': ids,
        'captured_prediction_indices': [0,1,2,3], 'captured_layers': selected,
        'intervention_layers': intervention_layers, 'component_indices_kept': [1,3],
        'parameter_edits': parameter_edits, 'overlay': overlay,
        'association': {'layer': last, 'component': unit, 'target': target,
                        'parameter_edits': association_edits, 'cases': cases,
                        'method': 'Two-prefix minimum-norm read fit, tied gate/value row, normalized token-difference write; third prefix held out'},
        'baseline': baseline, 'keep_only': keep_only, 'deletion': deletion}, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'prefix_ids': ids,
                      'baseline_generated': baseline['generated'], 'keep_only_generated': keep_only['generated'],
                      'overlay_generated': overlay['generated'], 'deletion_generated': deletion['generated']}))


if __name__ == '__main__':
    main()
