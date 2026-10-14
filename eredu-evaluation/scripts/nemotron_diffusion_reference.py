"""Pinned NVIDIA text implementation: causal cache and transient-block reference.

Usage: python nemotron_diffusion_reference.py CHECKPOINT OUTPUT.safetensors
The checkpoint directory contains the published Python files and weight artifact.
BF16 released values are decoded to FP32 in both comparison implementations.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
import hashlib

import torch
from safetensors.torch import load_file, save_file
from nemotron_reference_provenance import verify_release


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draft-adapter", action="store_true")
    parser.add_argument("--trace-components", action="store_true", help="retain independent normalization and projection boundaries for numerical diagnosis")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager", help="published attention implementation selected for independent arithmetic")
    parser.add_argument("--compute-dtype", choices=["f32", "bf16"], default="f32", help="independent arithmetic precision; source identities stay pinned")
    parser.add_argument("--mlx-affine",type=Path,help="independently decode the pinned MLX affine artifact")
    parser.add_argument("--mlx-embedding-rounding", choices=["f32", "bf16_cpu"], default="f32", help="BF16 CPU affine lookup rounds multiplication and bias addition before F32 activation conversion")
    parser.add_argument("--gguf",type=Path,help="independently decode a pinned GGUF artifact using the pinned primary reader")
    args = parser.parse_args()
    if args.mlx_embedding_rounding != "f32" and (not args.mlx_affine or args.compute_dtype != "f32"):
        parser.error("MLX CPU embedding rounding requires an affine source and F32 activations")
    verify_release(args.checkpoint,adapter=args.draft_adapter)
    package = types.ModuleType("pinned_nemotron")
    package.__path__ = [str(args.checkpoint.resolve())]
    sys.modules[package.__name__] = package
    config_type = importlib.import_module(
        "pinned_nemotron.configuration_nemotron_labs_diffusion"
    ).NemotronLabsDiffusionConfig
    model_type = importlib.import_module(
        "pinned_nemotron.modeling_nemotron_labs_diffusion"
    ).NemotronLabsDiffusionModel
    config = config_type(**json.loads((args.checkpoint / "config.json").read_text()))
    config._attn_implementation = args.attention
    with torch.device("meta"):
        model = model_type(config)
    if args.mlx_affine and args.gguf:
        parser.error("select one quantized source representation")
    if args.gguf:
        import gguf
        import numpy as np
        records=json.loads((Path(__file__).resolve().parents[2]/"eredu-architectures/tests/fixtures/nemotron_labs_diffusion/conversions.json").read_text())
        record=next(item for item in records if item.get("format")=="gguf" and item["weight_files"][0]["rfilename"]==args.gguf.name)
        artifact=record["weight_files"][0]
        assert args.gguf.stat().st_size==artifact["size"]
        assert hashlib.file_digest(args.gguf.open("rb"),"sha256").hexdigest()==artifact["lfs"]["sha256"]
        base_manifest=json.loads((args.checkpoint/"manifest.json").read_text())
        assert base_manifest["id"]==record["base_repository"] and base_manifest["sha"]==record["base_revision"]
        library=Path(gguf.__file__).resolve().parent
        assert hashlib.sha256((library/"quants.py").read_bytes()).hexdigest()==record["reference_quants_sha256"]
        reader=gguf.GGUFReader(str(args.gguf))
        fields={"attn_q":"self_attn.q_proj","attn_k":"self_attn.k_proj","attn_v":"self_attn.v_proj","attn_output":"self_attn.o_proj","attn_norm":"input_layernorm","ffn_norm":"post_attention_layernorm","ffn_gate":"mlp.gate_proj","ffn_up":"mlp.up_proj","ffn_down":"mlp.down_proj"}
        weights={}
        for tensor in reader.tensors:
            name=tensor.name
            if name=="rope_freqs.weight":continue
            if name.startswith("blk."):
                _,index,field,suffix=name.split(".")
                assert suffix=="weight" and field in fields
                target=f"encoder.layers.{index}.{fields[field]}.weight"
            else:
                target={"token_embd.weight":"encoder.embed_tokens.weight","output_norm.weight":"encoder.norm.weight","output.weight":"diffusion_head.weight"}[name]
            decoded=gguf.dequantize(tensor.data,tensor.tensor_type).reshape(tuple(reversed(tensor.shape))).astype(np.float32,copy=False)
            if record["rotary_pairing"]=="adjacent_pairs" and target.endswith(("self_attn.q_proj.weight","self_attn.k_proj.weight")):
                heads=config.num_attention_heads if target.endswith("q_proj.weight") else config.num_key_value_heads
                decoded=decoded.reshape(heads,config.head_dim//2,2,config.hidden_size).transpose(0,2,1,3).reshape(decoded.shape)
            weights[target]=torch.from_numpy(np.ascontiguousarray(decoded))
        print(f"independently decoded pinned GGUF {record['repo']}@{record['revision']}",flush=True)
    elif args.mlx_affine:
        records = json.loads((Path(__file__).resolve().parents[2] / "eredu-architectures/tests/fixtures/nemotron_labs_diffusion/conversions.json").read_text())
        record = next(item for item in records if item["repo"] == "mlx-community/Nemotron-Labs-Diffusion-3B-4bit")
        manifest = json.loads((args.mlx_affine / "manifest.json").read_text())
        assert manifest["sha"] == record["revision"] and manifest["id"] == record["repo"]
        file = args.mlx_affine / "model.safetensors"
        artifact = record["weight_files"][0]
        assert file.stat().st_size == artifact["size"]
        assert hashlib.file_digest(file.open("rb"),"sha256").hexdigest() == artifact["lfs"]["sha256"]
        packed = load_file(str(file))
        weights = {}
        import numpy as np
        for name,value in packed.items():
            if name.endswith((".scales",".biases")):
                continue
            if name.startswith("language_model.model."):
                target = "encoder." + name.removeprefix("language_model.model.")
            elif name.startswith("language_model.diffusion_head."):
                target = "diffusion_head." + name.removeprefix("language_model.diffusion_head.")
            else:
                raise ValueError(f"unexpected quantized parameter {name}")
            if value.dtype == torch.uint32:
                prefix = name.removesuffix(".weight")
                scales = packed[prefix + ".scales"].float().numpy()
                biases = packed[prefix + ".biases"].float().numpy()
                words = value.numpy()
                codes = ((words[...,None] >> (np.arange(8,dtype=np.uint32)*4)) & 15).reshape(words.shape[0],-1).astype(np.float32)
                decoded = codes*np.repeat(scales,64,axis=1)+np.repeat(biases,64,axis=1)
                if target == "encoder.embed_tokens.weight" and args.mlx_embedding_rounding == "bf16_cpu":
                    # MLX's CPU affine dequantizer applies distinct multiply and
                    # add operators in companion precision, then converts the
                    # selected lookup rows to F32. Projection kernels promote
                    # companions with their F32 activation input instead.
                    assert packed[prefix + ".scales"].dtype == torch.bfloat16
                    assert packed[prefix + ".biases"].dtype == torch.bfloat16
                    product = (torch.from_numpy(codes).bfloat16()
                               * torch.from_numpy(np.repeat(scales,64,axis=1)).bfloat16())
                    decoded = (product + torch.from_numpy(np.repeat(biases,64,axis=1)).bfloat16()).float().numpy()
                value = torch.from_numpy(decoded)
            weights[target] = value.float()
        del packed
        print("independently decoded pinned MLX UINT4 affine values",flush=True)
    else:
        weights = load_file(str(args.checkpoint / "model.safetensors"))
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model = model.to(dtype=torch.float32 if args.compute_dtype == "f32" else torch.bfloat16).eval()
    # Meta construction also creates non-persistent rotary buffers. Rebuild them
    # using the published constructor rather than deriving frequencies here.
    rotary_type = type(model.encoder.rotary_emb)
    model.encoder.rotary_emb = rotary_type(model.encoder.config, device="cpu")
    torch.set_num_threads(8)
    output = {}
    active = ""
    handles = []
    if args.draft_adapter:
        directory = args.checkpoint / "linear_spec_lora"
        adapter_config = json.loads((directory / "adapter_config.json").read_text())
        assert adapter_config["target_modules"] == ["o_proj"]
        assert not adapter_config["use_dora"] and not adapter_config["use_rslora"]
        adapter_weights = load_file(str(directory / "adapter_model.safetensors"))
        scale = adapter_config["lora_alpha"] / adapter_config["r"]
        for index, layer in enumerate(model.encoder.layers):
            prefix = f"base_model.model.encoder.layers.{index}.self_attn.o_proj"
            a, b = adapter_weights[prefix + ".lora_A.weight"], adapter_weights[prefix + ".lora_B.weight"]
            def delta(module, inputs, output, a=a, b=b):
                if active.startswith("draft."):
                    delta_input = inputs[0].to(a.dtype)
                    delta = torch.nn.functional.linear(torch.nn.functional.linear(delta_input,a),b)*scale
                    return (output + delta).to(output.dtype)
                return output
            handles.append(layer.self_attn.o_proj.register_forward_hook(delta))
    for index, layer in enumerate(model.encoder.layers):
        def capture(module, inputs, value, index=index):
            output[f"{active}.layer.{index}"] = value.detach().contiguous()
        handles.append(layer.register_forward_hook(capture))
        if args.trace_components:
            for name, module in [
                ("attention.input", layer.input_layernorm),
                ("attention.write", layer.self_attn.o_proj),
                ("feed_forward.input", layer.post_attention_layernorm),
                ("feed_forward.write", layer.mlp),
            ]:
                def component(module, inputs, value, index=index, name=name):
                    output[f"{active}.component.{index}.{name}"] = value.detach().contiguous()
                    if name == "attention.write":
                        output[f"{active}.component.{index}.attention.write_input"] = inputs[0].detach().contiguous()
                handles.append(module.register_forward_hook(component))
    cache = None
    cases = ([1, 100, 200], [17], [31], [421])
    with torch.inference_mode():
        for step, tokens in enumerate(cases):
            active = f"causal.{step}"
            for layer in model.encoder.layers:
                layer.self_attn.diffusion_lm = False
            encoded = model.encoder(
                input_ids=torch.tensor([tokens]), past_key_values=cache,
                use_cache=True, use_causal_mask=True,
            )
            cache = encoded.past_key_values
            output[active + ".logits"] = model.diffusion_head(encoded.last_hidden_state).contiguous()
            frontier = cache.get_seq_length()
            old = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
            # Keep the released block geometry even though this reference commits
            # only a few causal tokens between independent refinement invocations.
            canvas = [config.mask_token_id] * config.block_size
            canvas[0] = 101
            canvas[-1] = 211
            output[f"block.{step}.tokens"] = torch.tensor([canvas], dtype=torch.int32)
            active = f"block.{step}"
            for layer in model.encoder.layers:
                layer.self_attn.diffusion_lm = True
            refined = model.encoder(
                input_ids=torch.tensor([canvas]), past_key_values=cache,
                use_cache=False, use_causal_mask=False,
            )
            output[active + ".logits"] = model.diffusion_head(refined.last_hidden_state).contiguous()
            if args.draft_adapter:
                active = f"draft.{step}"
                drafted = model.encoder(input_ids=torch.tensor([canvas]), past_key_values=cache,
                                        use_cache=False, use_causal_mask=False)
                output[active + ".logits"] = model.diffusion_head(drafted.last_hidden_state).contiguous()
            assert cache.get_seq_length() == frontier
            for layer, (keys, values) in zip(cache.layers, old):
                assert torch.equal(layer.keys, keys) and torch.equal(layer.values, values)
            print(f"step={step} frontier={frontier} transient_positions={len(canvas)}", flush=True)
    for handle in handles:
        handle.remove()
    save_file(output, str(args.output), metadata={
        "weight_precision": (
            "pinned GGUF encodings independently decoded to FP32" if args.gguf else
            "MLX affine UINT4 and BF16 companions independently decoded to FP32" if args.mlx_affine else
            "released BF16 decoded to FP32"
        ),
        "compute_dtype": args.compute_dtype,
        "mlx_embedding_rounding": args.mlx_embedding_rounding if args.mlx_affine else "not_applicable",
        "attention_implementation": args.attention,
        "torch": torch.__version__,
        "transformers": importlib.import_module("transformers").__version__,
    })


if __name__ == "__main__":
    main()
