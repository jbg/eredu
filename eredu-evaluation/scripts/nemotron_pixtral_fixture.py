"""Generate a small nonzero fixture using the published Pixtral and VLM projector.

Run with Transformers 4.57.1. SOURCE is the pinned VLM-8B Python source directory.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
import re
import numpy as np
from PIL import Image
import torch
import transformers
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
from nemotron_reference_provenance import verify_release


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify_release(args.source,require_weights=False)
    assert transformers.__version__ == "4.57.1"
    package = types.ModuleType("pinned_nemotron_vlm")
    package.__path__ = [str(args.source.resolve())]
    sys.modules[package.__name__] = package
    source = importlib.import_module("pinned_nemotron_vlm.modeling_ministral")
    config_type = importlib.import_module(
        "pinned_nemotron_vlm.configuration_nemotron_labs_diffusion_vlm"
    ).NemotronLabsDiffusionVLMConfig
    config_json = json.loads((args.source / "config.json").read_text())
    config_json.update(hidden_size=6, intermediate_size=12, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=32,
                       mask_token_id=0)
    config_json["vision_config"].update(hidden_size=8, intermediate_size=12,
                                      num_hidden_layers=2, num_attention_heads=2,
                                      head_dim=4, patch_size=2, image_size=16)
    config = config_type(**config_json)
    vision_config = PixtralVisionConfig(**config_json["vision_config"])
    vision_config._attn_implementation = "eager"
    config.vision_config = vision_config
    tower = PixtralVisionModel(vision_config).eval()
    projector = source.Ministral3MultiModalProjector(config).eval()
    # Deterministic explicit weights, independent of random initialization order.
    parameters = {}
    for root, module in [("model.vision_tower", tower), ("model.multi_modal_projector", projector)]:
        with torch.no_grad():
            for index, (name, value) in enumerate(module.named_parameters()):
                values = torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
                values = (values * 0.17 + index * 0.31).sin() * 0.15
                if value.ndim == 1:
                    values += 1.0
                value.copy_(values)
                parameters[root + "." + name] = value.detach().clone()
    pixels = torch.sin(torch.arange(2*3*8*8).reshape(2,3,8,8) * 0.07)
    sizes = torch.tensor([[4,8],[8,4]])
    patches = []
    for image, (h,w) in zip(pixels, sizes):
        patches.append(torch.nn.functional.unfold(image[:,:h,:w].unsqueeze(0),kernel_size=2,stride=2)[0].T)
    outputs = {}
    tower.ln_pre.register_forward_hook(lambda m,i,o: outputs.update(embedding=o.detach()))
    for index, layer in enumerate(tower.transformer.layers):
        layer.register_forward_hook(lambda m,i,o,index=index: outputs.update({f"layer.{index}": o[0].detach()}))
    with torch.inference_mode():
        encoded = tower(pixel_values=pixels, image_sizes=sizes).last_hidden_state
        outputs["projected"] = projector(encoded.squeeze(0),sizes).unsqueeze(0)
    def encode(value):
        return {"shape": list(value.shape), "data": value.flatten().tolist()}
    processing = importlib.import_module("pinned_nemotron_vlm.image_processing")
    processor_cases = []
    for width,height,channels in [(3,5,3),(43,31,3),(32,5,3),(5,3,4)]:
        pixels = (np.arange(width*height*channels).reshape(height,width,channels)*17 % 251).astype(np.uint8)
        w,h,normalized = processing.encode_image(Image.fromarray(pixels),image_patch_size=2,max_image_size=16,spatial_merge_size=2)
        markers = processing.build_image_token_str(w,h)
        identities = {processing.IMG_START_TOKEN:18,processing.IMG_PAD_TOKEN:19,
                      processing.IMG_BREAK_TOKEN:20,processing.IMG_END_TOKEN:21}
        tokens = [identities[token] for token in re.findall(r"<\|image_[a-z]+\|>",markers)]
        packed = torch.nn.functional.unfold(torch.from_numpy(normalized.copy()).unsqueeze(0),kernel_size=2,stride=2)[0].T
        processor_cases.append({"width":width,"height":height,"channels":channels,"pixels":pixels.flatten().tolist(),
                                "grid":[h*2,w*2],"tokens":tokens,"patches":encode(packed)})
    args.output.write_text(json.dumps({
        "source_revision": "adca93d16471c1e07d594ae444d23e1876f6b365",
        "transformers": transformers.__version__, "torch": torch.__version__,
        "config": config_json, "grids": [[2,4],[4,2]],
        "patches": encode(torch.cat(patches)),
        "parameters": {key: encode(value) for key,value in parameters.items()},
        "outputs": {key: encode(value) for key,value in outputs.items()},
        "processor":processor_cases,
    }, separators=(",",":")) + "\n")


if __name__ == "__main__":
    main()
