#!/usr/bin/env python3
"""Pinned released LFM3 vision-component reference; outputs live outside the tree.

This runs publisher classes, including their image/video patchifiers and an
optional mixed image/video conditional-generation case.
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
    parser.add_argument('--conditional', action='store_true', help='record mixed media prefill and eight cached steps')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--raw-prefill-chunk', type=int,
                        help='prefill the raw conditional case in cached chunks of this size')
    args = parser.parse_args()
    if args.raw_prefill_chunk is not None and (not args.conditional or args.raw_prefill_chunk <= 0):
        parser.error('--raw-prefill-chunk requires --conditional and a positive size')
    if args.dtype == 'bfloat16' and not args.conditional:
        parser.error('native BF16 replay requires --conditional')
    dtype = getattr(torch, args.dtype)
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
    config_cls = get_class_from_dynamic_module('configuration_lfm3_vl.Lfm3VlConfig', source, local_files_only=True)
    config = config_cls.from_pretrained(source, local_files_only=True)
    config.vision_config._attn_implementation = 'eager'
    vision_cls = get_class_from_dynamic_module('modeling_lfm3_vl.Lfm3VlVisionModel', source, local_files_only=True)
    projector_cls = get_class_from_dynamic_module('modeling_lfm3_vl.Lfm3VlMultiModalProjector', source, local_files_only=True)
    image_cls = get_class_from_dynamic_module('image_processing_lfm3_vl.Lfm3VlImageProcessor', source, local_files_only=True)
    video_cls = get_class_from_dynamic_module('video_processing_lfm3_vl.Lfm3VlVideoProcessor', source, local_files_only=True)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    model = None
    if args.conditional:
        model_cls = get_class_from_dynamic_module(
            'modeling_lfm3_vl.Lfm3VlForConditionalGeneration', source, local_files_only=True)
        model = model_cls.from_pretrained(
            source, local_files_only=True, dtype=dtype, attn_implementation='eager').eval()

    derived_sha256 = None
    if dtype == torch.bfloat16:
        vision = model.model.vision_tower
        projector = model.model.multi_modal_projector
    else:
        derived.mkdir(exist_ok=True)
        for name in manifest['source_sha256']:
            shutil.copyfile(source / name, derived / name)
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
        derived_sha256 = digest(derived / 'model.safetensors')
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
    def record_generation(model, name, ids, image_patches, image_grid, video_patches, video_grid,
                          chunk=None):
        generated = []
        with torch.inference_mode():
            tokens = torch.tensor([ids])
            result = model(
                input_ids=tokens,
                pixel_values=image_patches, image_grid_thw=image_grid,
                pixel_values_videos=video_patches, video_grid_thw=video_grid,
                use_cache=True)
            if name == 'raw':
                tensors['raw.image.projected'] = result.image_hidden_states.contiguous()
                tensors['raw.video.projected'] = result.video_hidden_states.contiguous()
            if chunk is None:
                prefill = result.logits
            else:
                # Reuse the publisher's projected rows and its own placeholder
                # admission for the prompt embeddings. The publisher decoder
                # then executes each prefix with the preceding prefix's cache.
                embeds = model.get_input_embeddings()(tokens)
                image = result.image_hidden_states
                video = result.video_hidden_states
                image_mask, video_mask = model.model.get_placeholder_mask(
                    tokens, inputs_embeds=embeds, image_features=image, video_features=video)
                embeds = embeds.masked_scatter(image_mask, image).masked_scatter(video_mask, video)
                result = None
                rows = []
                for prefix in embeds.split(chunk, dim=1):
                    result = model(inputs_embeds=prefix, use_cache=True,
                                   past_key_values=None if result is None else result.past_key_values)
                    rows.append(result.logits)
                prefill = torch.cat(rows, dim=1)
            tensors[f'{name}.prefill'] = prefill.contiguous()
            for step in range(8):
                token = result.logits[0, -1].argmax().item()
                generated.append(token)
                result = model(input_ids=torch.tensor([[token]]),
                               past_key_values=result.past_key_values, use_cache=True)
                tensors[f'{name}.decode.{step}'] = result.logits.contiguous()
        return {'prompt': ids, 'generated': generated, 'prefill_chunk': chunk}

    conditional = None
    if model is not None:
        image_grid = torch.tensor(cases[0]['grid'], dtype=torch.int64)
        video_grid = torch.tensor(cases[1]['grid'], dtype=torch.int64)
        image_count = tensors['image.projected'].shape[0]
        video_count = tensors['video_odd.projected'].shape[0]
        video_frame_positions = video_count // video_grid[0, 0].item()
        image = [config.vision_start_token_id, *([config.image_token_id] * image_count),
                 config.vision_end_token_id]
        frame = [config.vision_start_token_id, *([config.video_token_id] * video_frame_positions),
                 config.vision_end_token_id]
        ids = [config.bos_token_id, 5, *image, 8, *frame, 13, *frame, 21, 34]
        conditional = record_generation(
            model, 'conditional', ids, tensors['image.patches'], image_grid,
            tensors['video_odd.patches'], video_grid)

        from transformers import AutoTokenizer
        from transformers.video_utils import VideoMetadata
        settings = json.loads((source / 'processor_config.json').read_text())
        image_processor = image_cls(**settings['image_processor'])
        video_processor = video_cls(**settings['video_processor'])
        processor_cls = get_class_from_dynamic_module(
            'processing_lfm3_vl.Lfm3VlProcessor', source, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        processor = processor_cls(image_processor=image_processor, video_processor=video_processor, tokenizer=tokenizer)
        rgb_image = tensors['image.rgb.0'][0]
        rgb_video = ((torch.arange(7 * 3 * 33 * 63).reshape(7, 3, 33, 63) * 17) % 256).byte()
        metadata = VideoMetadata(total_num_frames=7, fps=6.0)
        sampled = video_processor.sample_frames(metadata)
        metadata.frames_indices = sampled.tolist()
        image_result = image_processor.preprocess(rgb_image, return_tensors='pt')
        video_result = video_processor.preprocess(rgb_video[sampled], do_sample_frames=False, return_tensors='pt')
        video_result['video_metadata'] = [metadata]
        image_text = processor.vision_start_token + processor.replace_image_token(image_result, 0) + processor.vision_end_token
        video_text = processor.replace_video_token(video_result, 0)
        tokens = [config.bos_token_id, 5, *tokenizer.encode(image_text, add_special_tokens=False), 8,
                  *tokenizer.encode(video_text, add_special_tokens=False), 21, 34]
        tensors['raw.video.rgb'] = rgb_video.contiguous()
        for key, result, pixels, grid in [
            ('image', image_result, 'pixel_values', 'image_grid_thw'),
            ('video', video_result, 'pixel_values_videos', 'video_grid_thw'),
        ]:
            tensors[f'raw.{key}.patches'] = result[pixels].reshape(-1, result[pixels].shape[-1]).contiguous()
            tensors[f'raw.{key}.grid'] = result[grid].to(torch.int32).contiguous()
        raw_processor = record_generation(
            model, 'raw', tokens, tensors['raw.image.patches'], image_result['image_grid_thw'],
            tensors['raw.video.patches'], video_result['video_grid_thw'],
            chunk=args.raw_prefill_chunk)
        raw_processor['video_fps'] = 6.0
    save_file(tensors, output / 'reference.safetensors')
    report = {'repository': manifest['repository'], 'revision': manifest['revision'],
              'source_sha256': manifest['weights_sha256'], 'derived_sha256': derived_sha256,
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'attention': 'eager', 'dtype': args.dtype, 'resizing': False, 'cases': cases,
              'maximum_absolute_tolerance': manifest['maximum_absolute_tolerance']}
    if conditional is not None:
        report['conditional'] = conditional
        report['raw_processor'] = raw_processor
    report['package_versions'] = {name: importlib.metadata.version(name)
                                 for name in ('torch', 'transformers', 'torchvision', 'pillow',
                                              'safetensors', 'huggingface_hub')}
    (output / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
