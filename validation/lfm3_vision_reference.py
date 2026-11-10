#!/usr/bin/env python3
"""Pinned released LFM3 vision-component reference; outputs live outside the tree.

This runs publisher classes, including their image/video patchifiers. It measures
the encoder and projector without claiming conditional language-model parity.
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    source = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.is_relative_to(checkout):
        parser.error('reference payloads must live outside the repository')
    manifest = json.loads((checkout / 'validation/fixtures/lfm3/vision-sources.json').read_text())
    if digest(source / 'model.safetensors') != manifest['weights_sha256']:
        parser.error('released checkpoint digest differs from the pinned source')
    for name, expected in manifest['source_sha256'].items():
        if digest(source / name) != expected:
            parser.error(f'pinned source digest differs: {name}')
    output.mkdir(parents=True, exist_ok=True)
    derived = output / 'f32-component'
    derived.mkdir(exist_ok=True)
    for name in manifest['source_sha256']:
        shutil.copyfile(source / name, derived / name)
    config_cls = get_class_from_dynamic_module('configuration_lfm3_vl.Lfm3VlConfig', source, local_files_only=True)
    config = config_cls.from_pretrained(source, local_files_only=True)
    config.vision_config._attn_implementation = 'eager'
    vision_cls = get_class_from_dynamic_module('modeling_lfm3_vl.Lfm3VlVisionModel', source, local_files_only=True)
    projector_cls = get_class_from_dynamic_module('modeling_lfm3_vl.Lfm3VlMultiModalProjector', source, local_files_only=True)
    image_cls = get_class_from_dynamic_module('image_processing_lfm3_vl.Lfm3VlImageProcessor', source, local_files_only=True)
    video_cls = get_class_from_dynamic_module('video_processing_lfm3_vl.Lfm3VlVideoProcessor', source, local_files_only=True)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    vision = vision_cls(config.vision_config).float().eval()
    projector = projector_cls(config).float().eval()
    with safe_open(source / 'model.safetensors', framework='pt') as checkpoint:
        weights = {name: checkpoint.get_tensor(name).float().contiguous() for name in checkpoint.keys()
                   if name.startswith(('model.vision_tower.', 'model.multi_modal_projector.'))}
    vision.load_state_dict({k.removeprefix('model.vision_tower.'): v for k, v in weights.items()
                            if k.startswith('model.vision_tower.')}, strict=True)
    projector.load_state_dict({k.removeprefix('model.multi_modal_projector.'): v for k, v in weights.items()
                               if k.startswith('model.multi_modal_projector.')}, strict=True)
    save_file(weights, derived / 'model.safetensors')
    del weights
    tensors = {}
    cases = []
    # Each RGB value varies across channel, row, column and temporal position.
    # The spatial grids are rectangular, and one batch mixes frame shapes.
    for name, shapes in [('image', [(1, 64, 96)]), ('video_odd', [(3, 64, 96)]),
                         ('mixed_images', [(1, 64, 64), (1, 96, 64)])]:
        patches, grids = [], []
        for item, (frames, height, width) in enumerate(shapes):
            rgb = ((torch.arange(frames * 3 * height * width).reshape(frames, 3, height, width)
                    * 17 + item * 31) % 256).to(torch.uint8)
            tensors[f'{name}.rgb.{item}'] = rgb
            normalized = (rgb.float() * (1.0 / 255.0) - 0.5) / 0.5
            if frames == 1:
                values, h, w = image_cls.patchify(None, normalized, config.vision_config.patch_size,
                    config.vision_config.spatial_merge_size, config.vision_config.temporal_patch_size)
                t = 1
            else:
                values, t, h, w = video_cls.patchify(None, normalized.unsqueeze(0),
                    config.vision_config.patch_size, config.vision_config.spatial_merge_size,
                    config.vision_config.temporal_patch_size)
            patches.append(values.reshape(-1, values.shape[-1]))
            grids.append((t, h, w))
        pixels = torch.cat(patches)
        grid = torch.tensor(grids, dtype=torch.int64)
        with torch.inference_mode():
            hidden = vision(pixels, grid_thw=grid).last_hidden_state
            projected = projector(hidden)
        tensors[f'{name}.patches'] = pixels.contiguous()
        tensors[f'{name}.hidden'] = hidden.contiguous()
        tensors[f'{name}.projected'] = projected.contiguous()
        cases.append({'id': name, 'grid': grids, 'frames': shapes})
    save_file(tensors, output / 'reference.safetensors')
    report = {'repository': manifest['repository'], 'revision': manifest['revision'],
              'source_sha256': manifest['weights_sha256'], 'derived_sha256': digest(derived / 'model.safetensors'),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'attention': 'eager', 'dtype': 'float32', 'resizing': False, 'cases': cases,
              'maximum_absolute_tolerance': manifest['maximum_absolute_tolerance']}
    report['package_versions'] = {name: importlib.metadata.version(name)
                                 for name in ('torch', 'transformers', 'torchvision', 'pillow',
                                              'safetensors', 'huggingface_hub')}
    (output / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
