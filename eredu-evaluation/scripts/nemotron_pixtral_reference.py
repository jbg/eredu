"""Released VLM-8B Pixtral and projector reference using the pinned implementation."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import numpy as np
from PIL import Image
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
from nemotron_reference_provenance import verify_release, verify_compressed


def decode_vlm_gguf(directory, configuration, release, preserve_source_precision=False):
    import gguf
    from nemotron_vlm_gguf_ancestry import source_name
    pin = json.loads((Path(__file__).resolve().parents[2] / "eredu-architectures/tests/fixtures/nemotron_labs_diffusion/vlm-gguf-conversion.json").read_text())
    assert pin["base_repository"] == release["repo"] and pin["base_revision"] == release["revision"]
    assert hashlib.sha256((Path(gguf.__file__).resolve().parent / "quants.py").read_bytes()).hexdigest() == pin["reference_quants_sha256"]
    weights = {}
    for member in pin["weight_files"]:
        path = directory / member["rfilename"]
        assert path.stat().st_size == member["size"]
        assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == member["lfs"]["sha256"]
        vision = member["rfilename"] == "mmproj-F16.gguf"
        reader = gguf.GGUFReader(str(path))
        for tensor in reader.tensors:
            if tensor.name == "v.token_embd.img_break":
                continue
            key = source_name(tensor.name, vision)
            shape = tuple(int(x) for x in reversed(tensor.shape))
            decoded = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(shape).astype(np.float32, copy=False)
            if key.endswith(("q_proj.weight", "k_proj.weight")):
                heads = configuration["vision_config"]["num_attention_heads"] if vision else configuration["num_attention_heads"] if key.endswith("q_proj.weight") else configuration["num_key_value_heads"]
                decoded = decoded.reshape(heads, shape[0] // heads // 2, 2, *shape[1:]).swapaxes(1, 2).reshape(shape)
            assert key not in weights
            # Vision pair-coordinate recipes retain FP32 outputs; other
            # dense F16 matrices retain their admitted source precision.
            if preserve_source_precision and tensor.tensor_type == gguf.GGMLQuantizationType.F16 and not (vision and key.endswith(("q_proj.weight", "k_proj.weight"))):
                decoded = decoded.astype(np.float16)
            weights[key] = torch.from_numpy(np.array(decoded, copy=True, order="C"))
    return weights, pin


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--text-context",action="store_true",help="compare image preparation with the shared causal and diffusion text cache")
    parser.add_argument("--compute-dtype",choices=["f32","bf16","mixed"],default="f32",help="mixed preserves admitted dense precision and evaluates promoted products")
    parser.add_argument("--tensor-fp8",type=Path,help="independently decode a pinned scalar-scaled W8A16 artifact")
    parser.add_argument("--gguf",type=Path,help="independently decode pinned VLM language and vision/projector GGUF companions")
    args = parser.parse_args()
    release = verify_release(args.checkpoint)
    if args.gguf and args.tensor_fp8:
        parser.error("select one complete quantized source representation")
    if args.gguf and args.compute_dtype == "bf16":
        parser.error("bounded quantized reference decoding uses F32 arithmetic")
    if args.compute_dtype == "mixed" and not (args.tensor_fp8 or args.gguf):
        parser.error("mixed arithmetic requires a scalar-scaled FP8 or F16/F32 GGUF artifact")
    if args.tensor_fp8 and args.compute_dtype == "bf16":
        parser.error("bounded FP8 projection decoding uses F32 products")
    mixed = args.compute_dtype == "mixed"
    compute_dtype = torch.float32 if args.compute_dtype == "f32" else torch.bfloat16
    weights_directory = args.checkpoint
    weight_files = release["weight_files"]
    if args.tensor_fp8:
        conversion = verify_compressed(args.tensor_fp8)
        assert conversion["base_repository"] == release["repo"] and conversion["base_revision"] == release["revision"]
        quantization = json.loads((args.tensor_fp8 / "config.json").read_text())["quantization_config"]
        assert quantization["quant_algo"] == "FP8_W8A16"
        assert all(g["input_activations"] is None for g in quantization["config_groups"].values())
        weights_directory = args.tensor_fp8
        weight_files = conversion["weight_files"]
    assert transformers.__version__ == "4.57.1"
    package = types.ModuleType("pinned_nemotron_vlm")
    package.__path__ = [str(args.checkpoint.resolve())]
    sys.modules[package.__name__] = package
    published = importlib.import_module("pinned_nemotron_vlm.modeling_ministral")
    processing = importlib.import_module("pinned_nemotron_vlm.image_processing")
    config_type = importlib.import_module("pinned_nemotron_vlm.configuration_nemotron_labs_diffusion_vlm").NemotronLabsDiffusionVLMConfig
    configuration = json.loads((args.checkpoint / "config.json").read_text())
    config = config_type(**configuration)
    config.vision_config = PixtralVisionConfig(**configuration["vision_config"])
    config.vision_config._attn_implementation = "eager"
    with torch.device("meta"):
        tower = PixtralVisionModel(config.vision_config)
        projector = published.Ministral3MultiModalProjector(config)
    tower_weights, projector_weights = {}, {}
    gguf_weights = {}
    if args.gguf:
        gguf_weights, conversion = decode_vlm_gguf(args.gguf, configuration, release, mixed)
        tower_weights = {key.removeprefix("encoder.vision_tower."): value for key, value in gguf_weights.items() if key.startswith("encoder.vision_tower.")}
        projector_weights = {key.removeprefix("encoder.multi_modal_projector."): value for key, value in gguf_weights.items() if key.startswith("encoder.multi_modal_projector.")}
    for artifact in ([] if args.gguf else weight_files):
        with safe_open(weights_directory / artifact["rfilename"],framework="pt",device="cpu") as weights:
            for key in weights.keys():
                if key.startswith("encoder.vision_tower."):
                    value = weights.get_tensor(key)
                    tower_weights[key.removeprefix("encoder.vision_tower.")] = value if mixed else value.float()
                elif key.startswith("encoder.multi_modal_projector."):
                    value = weights.get_tensor(key)
                    projector_weights[key.removeprefix("encoder.multi_modal_projector.")] = value if mixed else value.float()
    tower.load_state_dict(tower_weights,strict=True,assign=True)
    projector.load_state_dict(projector_weights,strict=True,assign=True)
    del tower_weights, projector_weights
    if not mixed:
        tower.to(dtype=compute_dtype);projector.to(dtype=compute_dtype)
    def promote_linear_products(root):
        for module in root.modules():
            if isinstance(module, torch.nn.Linear):
                def promoted_linear(inputs, module=module):
                    dtype = torch.promote_types(inputs.dtype, module.weight.dtype)
                    bias = None if module.bias is None else module.bias.to(dtype)
                    return torch.nn.functional.linear(inputs.to(dtype), module.weight.to(dtype), bias)
                module.forward = promoted_linear
    if mixed:
        promote_linear_products(tower)
        promote_linear_products(projector)
    tower.patch_positional_embedding = type(tower.patch_positional_embedding)(config.vision_config,device="cpu")
    tower.eval();projector.eval()
    torch.set_num_threads(8)
    records, images, patches, sizes = {}, [], [], []
    for index,(width,height) in enumerate([(43,31),(23,55)]):
        raw = (np.arange(width*height*3).reshape(height,width,3)*17 % 251).astype(np.uint8)
        w,h,normalized = processing.encode_image(Image.fromarray(raw))
        pixels = torch.from_numpy(normalized.copy())
        images.append(pixels)
        sizes.append([pixels.shape[1],pixels.shape[2]])
        patches.append(torch.nn.functional.unfold(pixels.unsqueeze(0),kernel_size=14,stride=14)[0].T)
        records[f"vision.rgb.{index}"] = torch.from_numpy(raw.copy())
        records[f"vision.rgb_size.{index}"] = torch.tensor([height,width],dtype=torch.int32)
    padded = torch.zeros(len(images),3,max(p.shape[1] for p in images),max(p.shape[2] for p in images))
    for index,pixels in enumerate(images):
        padded[index,:,:pixels.shape[1],:pixels.shape[2]] = pixels
    padded = padded.to(tower.patch_conv.weight.dtype if mixed else compute_dtype)
    records["vision.patches"] = torch.cat(patches).contiguous()
    records["vision.grids"] = (torch.tensor(sizes,dtype=torch.int32)//14).contiguous()
    tower.ln_pre.register_forward_hook(lambda m,i,o:records.update({"vision.patch_projection":i[0].detach().contiguous(),"vision.embedding":o.detach().contiguous()}))
    for index,layer in enumerate(tower.transformer.layers):
        layer.register_forward_hook(lambda m,i,o,index=index:records.update({f"vision.layer.{index}":o[0].detach().contiguous()}))
    with torch.inference_mode():
        encoded = tower(pixel_values=padded,image_sizes=torch.tensor(sizes)).last_hidden_state
        records["vision.projected"] = projector(encoded.squeeze(0),torch.tensor(sizes)).unsqueeze(0).contiguous()
    if args.text_context:
        from tokenizers import Tokenizer
        model_type=importlib.import_module("pinned_nemotron_vlm.modeling_nemotron_labs_diffusion_vlm").NemotronLabsDiffusionVLMModel
        config._attn_implementation="eager"
        with torch.device("meta"):
            model=model_type(config)
        text_weights={key: value for key, value in gguf_weights.items() if not key.startswith(("encoder.vision_tower.", "encoder.multi_modal_projector."))}
        for artifact in ([] if args.gguf else weight_files):
            with safe_open(weights_directory/artifact["rfilename"],framework="pt",device="cpu") as weights:
                for key in weights.keys():
                    if key.endswith(".weight_scale"):
                        continue
                    if not key.startswith(("encoder.vision_tower.","encoder.multi_modal_projector.")):
                        value = weights.get_tensor(key)
                        if value.dtype == torch.float8_e4m3fn:
                            scale = weights.get_tensor(key.removesuffix(".weight") + ".weight_scale")
                            assert scale.dtype == torch.float32 and scale.shape == torch.Size([])
                            assert torch.isfinite(scale) and scale > 0
                            value = value.float() * scale
                        text_weights[key]=value if mixed else value.float()
        missing,unexpected=model.load_state_dict(text_weights,strict=False,assign=True)
        assert not unexpected and missing and all(key.startswith(("encoder.vision_tower.","encoder.multi_modal_projector.")) for key in missing)
        del text_weights
        model.encoder.vision_tower=tower
        model.encoder.multi_modal_projector=projector
        if not mixed:
            model.to(dtype=compute_dtype)
        else:
            # F32 converted projections and BF16 dense parameters retain their
            # distinct source precision. Independently evaluate promoted linear
            # products rather than recasting the entire model to one dtype.
            promote_linear_products(model)
        tower.patch_positional_embedding = type(tower.patch_positional_embedding)(config.vision_config,device="cpu")
        model.encoder.rotary_emb=type(model.encoder.rotary_emb)(model.encoder.config,device="cpu")
        if mixed:
            # Bounded FP8 projection conversion selects F32 query/cache
            # arithmetic. Tables follow that selected precision even when the
            # initial learned embedding remains BF16.
            published_rotary = model.encoder.rotary_emb.forward
            def converted_rotary(inputs, position_ids):
                return published_rotary(inputs.float(), position_ids)
            model.encoder.rotary_emb.forward = converted_rotary
        model.eval()
        tokenizer=Tokenizer.from_file(str(args.checkpoint/"tokenizer.json"))
        markers=[tokenizer.encode(processing.build_image_token_str(w//28,h//28),add_special_tokens=False).ids for h,w in sizes]
        prompt=[1]+markers[0]+[4,5]+markers[1]+[7]
        records["context.tokens"]=torch.tensor([prompt],dtype=torch.int32)
        active=""
        for index,layer in enumerate(model.encoder.layers):
            def capture(module,inputs,value,index=index):
                records[f"{active}.layer.{index}"]=(value[0] if isinstance(value,tuple) else value).detach().contiguous()
            layer.register_forward_hook(capture)
            if index == 0:
                for name, component in [("attention.input", layer.input_layernorm),
                                        ("attention.query", layer.self_attn.q_proj),
                                        ("attention.key", layer.self_attn.k_proj),
                                        ("attention.value", layer.self_attn.v_proj),
                                        ("attention.write", layer.self_attn.o_proj),
                                        ("feed_forward.input", layer.post_attention_layernorm),
                                        ("feed_forward.write", layer.mlp)]:
                    def boundary(module, inputs, value, name=name):
                        records[f"{active}.component.0.{name}"] = value.detach().clone().contiguous()
                        records[f"{active}.component.0.{name}.input"] = inputs[0].detach().clone().contiguous()
                    component.register_forward_hook(boundary)
        cache=None
        with torch.inference_mode():
            for step,ids in enumerate([prompt,[17],[31],[421]]):
                active=f"causal.{step}"
                for layer in model.encoder.layers:
                    layer.self_attn.diffusion_lm=False
                if step==0:
                    embeddings=model._embed_with_vision(torch.tensor([ids]),padded,torch.tensor(sizes))
                    encoded=model.encoder(inputs_embeds=embeddings,past_key_values=cache,use_cache=True,use_causal_mask=True)
                else:
                    encoded=model.encoder(input_ids=torch.tensor([ids]),past_key_values=cache,use_cache=True,use_causal_mask=True)
                cache=encoded.past_key_values
                records[active+".logits"]=model.diffusion_head(encoded.last_hidden_state).contiguous()
                frontier=cache.get_seq_length()
                old=[(layer.keys.clone(),layer.values.clone()) for layer in cache.layers]
                canvas=[config.mask_token_id]*config.block_size
                canvas[0]=101;canvas[-1]=211
                records[f"block.{step}.tokens"]=torch.tensor([canvas],dtype=torch.int32)
                active=f"block.{step}"
                for layer in model.encoder.layers:
                    layer.self_attn.diffusion_lm=True
                refined=model.encoder(input_ids=torch.tensor([canvas]),past_key_values=cache,use_cache=False,use_causal_mask=False)
                records[active+".logits"]=model.diffusion_head(refined.last_hidden_state).contiguous()
                assert cache.get_seq_length()==frontier
                for layer,(keys,values) in zip(cache.layers,old):
                    assert torch.equal(layer.keys,keys) and torch.equal(layer.values,values)
                print(f"VLM step={step} frontier={frontier} transient_positions={len(canvas)}",flush=True)
    precision = ("published F16 matrices and F32 vectors retained; FP32 pair-coordinate recipes and promoted products" if mixed else "published F16/F32 GGUF companions independently decoded to FP32") if args.gguf else "scalar-scaled E4M3 decoded to FP32; dense BF16 retained; promoted products" if mixed else "scalar-scaled E4M3 and dense BF16 decoded to FP32" if args.tensor_fp8 else f"released BF16 weights; {args.compute_dtype.upper()} arithmetic"
    save_file(records,str(args.output),metadata={"revision":release["revision"],"torch":torch.__version__,"transformers":transformers.__version__,"precision":precision,"compute_dtype":args.compute_dtype, "weight_revision":conversion["revision"] if args.tensor_fp8 or args.gguf else release["revision"]})
    print(f"released vision patches={records['vision.patches'].shape[0]} projected={records['vision.projected'].shape}",flush=True)


if __name__ == "__main__":
    main()
