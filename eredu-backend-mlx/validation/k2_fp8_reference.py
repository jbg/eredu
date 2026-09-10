"""Publisher FP32 model with independently expanded FP8 weights and activation hooks.

Usage: k2_fp8_reference.py PINNED_PUBLISHER_DIRECTORY EXPORTED_FIXTURE OUTPUT_JSON
The native fixture exporter supplies checkpoint inputs, never reference outputs.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('publisher', type=Path)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config_data = json.loads((args.artifact / 'config.json').read_text())
    config_data.pop('quantization_config')
    prototype = transformers.AutoConfig.from_pretrained(
        args.publisher, trust_remote_code=True, local_files_only=True)
    config = type(prototype)(**config_data)
    config._attn_implementation = 'eager'
    model_class = get_class_from_dynamic_module(
        prototype.auto_map['AutoModelForCausalLM'], str(args.publisher), local_files_only=True)
    model = model_class(config).float().eval()
    tensors = load_file(args.artifact / 'model.safetensors')
    weights = {}
    quantized = []
    for name, tensor in tensors.items():
        if name.endswith('_scale_inv'):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale = tensors[name.removesuffix('.weight') + '.weight_scale_inv']
            expanded = scale.repeat_interleave(128, -2).repeat_interleave(128, -1)
            weights[name] = tensor.float() * expanded[:tensor.shape[0], :tensor.shape[1]]
            quantized.append(name.removesuffix('.weight'))
        else:
            weights[name] = tensor
    model.load_state_dict(weights, strict=True)

    def quantize(module, inputs):
        x = inputs[0]
        rows = x.float().reshape(-1, x.shape[-1])
        parts = []
        for start in range(0, rows.shape[-1], 128):
            block = rows[:, start:start + 128]
            scale = block.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
            parts.append((block / scale).to(torch.float8_e4m3fn).float() * scale)
        return (torch.cat(parts, -1).reshape(x.shape).to(x.dtype), *inputs[1:])

    for name, module in model.named_modules():
        if name in quantized:
            assert isinstance(module, torch.nn.Linear), name
            module.register_forward_pre_hook(quantize)
    outputs = []
    cache = None
    token_inputs = [[1, 3, 2], [4], [5], [6], [7]]
    with torch.inference_mode():
        for ids in token_inputs:
            output = model(torch.tensor([ids]), past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            outputs.append(output.logits[:, -1].flatten().tolist())
    report = {
        'torch': torch.__version__,
        'torch_git': torch.version.git_version,
        'transformers': transformers.__version__,
        'python': platform.python_version(),
        'publisher_revision': '5d624db156710b3dc30fcca0a5cc25cb2898e110',
        'publisher_source_sha256': hashlib.sha256(
            (args.publisher / 'modeling_k2_horizon.py').read_bytes()).hexdigest(),
        'config': config_data,
        'artifact_sha256': hashlib.sha256(
            (args.artifact / 'model.safetensors').read_bytes()).hexdigest(),
        'activation_quantization': (
            'per-row groups of 128; max(abs(x),1e-4)/448; '
            'E4M3FN round-to-nearest-even; dequantize before FP32 product'),
        'quantized_modules': sorted(quantized),
        'inputs': token_inputs,
        'logits': outputs,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
