#!/usr/bin/env python3
"""Export independent full-prefill and cached-decode logits for native conformance."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('provenance', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--f32-checkpoint', type=Path, required=True)
    parser.add_argument('--tokens', default='1,3,2,5,8,13,21,34,55')
    parser.add_argument('--decode', default='5,8,13')
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text())
    source = Path(provenance['path'])
    checkpoint = args.f32_checkpoint.resolve()
    if checkpoint.is_relative_to(source.resolve()) or checkpoint.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error('derived checkpoint must be outside the source checkpoint and repository')
    checkpoint.mkdir(parents=True, exist_ok=True)
    expected = provenance['weights_sha256']
    if isinstance(expected, str):
        expected = {'model.safetensors': expected}
    derived = {}
    for name, digest in expected.items():
        with (source / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        tensors = load_file(str(source / name))
        save_file({key: value.float() if value.is_floating_point() else value
                   for key, value in tensors.items()}, str(checkpoint / name),
                  metadata={'format': 'pt'})
        with (checkpoint / name).open('rb') as stream:
            derived[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    for sidecar in source.iterdir():
        if sidecar.suffix in ('.json', '.jinja'):
            shutil.copyfile(sidecar, checkpoint / sidecar.name)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                                dtype=torch.float32, attn_implementation='eager').eval()
    prompt = [int(value) for value in args.tokens.split(',')]
    decode = [int(value) for value in args.decode.split(',')]
    logits = []
    cache = None
    with torch.inference_mode():
        for tokens in [prompt, *[[token] for token in decode]]:
            output = model(input_ids=torch.tensor([tokens]), past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            logits.append(output.logits[0, -1].float().tolist())
    args.output.write_text(json.dumps({
        'provenance': provenance, 'model_path': str(checkpoint),
        'derived_weights_sha256': derived, 'prompt': prompt,
        'decode_tokens': decode, 'logits': logits,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'dtype': 'float32', 'attention': 'eager',
    }) + '\n')
    print(f'Wrote {len(logits)} full-vocabulary predictions to {args.output}')


if __name__ == '__main__':
    main()
