"""Pinned, independent context/refinement fixtures for released DiffusionGemma."""

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import time

import torch
from safetensors.torch import save_file
from transformers import DiffusionGemmaForBlockDiffusion

MODEL_REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
REFERENCE_REVISION = "55c3cc9f03cd4fbf2d27d382dcc8fcc544631f8e"
CASES = [([2, 100, 200], [101, 211]), ([17], [421, 653]), ([31, 29], [1019, 2047])]


def load_quantized_text(model, path):
    """Independent GGML dequantization into the reference library's shared weights."""
    import gguf
    digest = "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5"
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != digest:
            raise ValueError("GGUF does not match the pinned Q4_K_M release")
    names = {
        "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
        "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
        "attn_norm": "input_layernorm", "post_attention_norm": "post_attention_layernorm",
        "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
        "post_ffw_norm_1": "post_feedforward_layernorm_1", "pre_ffw_norm_2": "pre_feedforward_layernorm_2",
        "post_ffw_norm_2": "post_feedforward_layernorm_2", "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
        "ffn_gate_inp": "router.proj", "ffn_gate_up_exps": "experts.gate_up_proj",
        "ffn_down_exps": "experts.down_proj",
    }
    special = {"layer_output_scale.weight": "layer_scalar", "ffn_gate_inp.scale": "router.scale",
               "ffn_down_exps.scale": "router.per_expert_scale"}
    state = model.state_dict()
    count = 0
    for tensor in gguf.GGUFReader(str(path)).tensors:
        name = tensor.name
        if name == "rope_freqs.weight":
            continue
        if name.startswith("blk."):
            _, layer, suffix = name.split(".", 2)
            if suffix == "enc_layer_output_scale.weight":
                target = f"model.encoder.language_model.layers.{layer}.layer_scalar"
            else:
                field, companion = suffix.rsplit(".", 1)
                suffix = special.get(suffix, names.get(field, field) + "." + companion)
                if suffix.startswith("experts."):
                    suffix = suffix.removesuffix(".weight")
                target = f"model.decoder.layers.{layer}.{suffix}"
        elif name.startswith("self_cond_"):
            field = name.removeprefix("self_cond_").removesuffix(".weight")
            target = "model.decoder.self_conditioning." + ("pre_norm" if field == "pre_norm" else field + "_proj") + ".weight"
        else:
            target = {"token_embd.weight": "model.decoder.embed_tokens.weight",
                      "output_norm.weight": "model.decoder.norm.weight",
                      "output.weight": "lm_head.weight"}[name]
        value = gguf.dequantize(tensor.data, tensor.tensor_type)
        if tuple(value.shape) != tuple(state[target].shape):
            raise ValueError(f"{name}: {value.shape} differs from {target}: {state[target].shape}")
        state[target].copy_(torch.from_numpy(value.copy()))
        count += 1
    print(f"independently dequantized {count} GGUF parameters", flush=True)
    return {"repository": "unsloth/diffusiongemma-26B-A4B-it-GGUF",
            "revision": "f4183a2c7a354128d02545752303c4354d165bf0", "sha256": digest,
            "decoder": "gguf", "decoder_version": importlib.metadata.version("gguf")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--vision", action="store_true", help="also capture a released image invocation")
    parser.add_argument("--vision-only", action="store_true", help="capture image execution without text invocations")
    parser.add_argument("--vision-grid", type=int, nargs=2, default=[3, 3], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gguf", type=pathlib.Path, help="replace shared text with independently dequantized pinned GGUF weights")
    args = parser.parse_args()
    if any(d <= 0 or d % 3 for d in args.vision_grid):
        parser.error("vision grid dimensions must be positive multiples of three")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    manifest = json.loads((args.checkpoint / "source_manifest.json").read_text())
    if manifest.get("id") != "google/diffusiongemma-26B-A4B-it" or manifest.get("sha") != MODEL_REVISION:
        parser.error("checkpoint source manifest does not identify the pinned release")
    provenance = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if provenance is None or REFERENCE_REVISION not in json.loads(provenance).get("url", ""):
        parser.error("install Transformers from the pinned reference revision")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    dtype = torch.float32 if args.gguf else torch.bfloat16
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, dtype=dtype, attn_implementation="eager",
        experts_implementation="eager", local_files_only=True,
    ).eval()
    with torch.no_grad():
        quantized_source = load_quantized_text(model, args.gguf) if args.gguf else None
    records = {}
    active_step = 0

    def record(label):
        def hook(_module, _inputs, output):
            if active_step == 0:
                value = output[0] if isinstance(output, tuple) else output
                records[label] = value.float().contiguous()
        return hook

    encoder = model.model.encoder.language_model
    encoder.embed_tokens.register_forward_hook(record("embedding"))
    for index, layer in enumerate(encoder.layers):
        layer.register_forward_hook(record(f"layer.{index}"))
    def decoder_layer(index):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            records[f"decoder.{active_step}.layer.{index}"] = value.float().contiguous()
        return hook

    for index, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(decoder_layer(index))
    cache, prior = None, None
    with torch.inference_mode():
        for step, (tokens, canvas) in enumerate([] if args.vision_only else CASES):
            active_step = step
            encoded = model.model.encoder(
                input_ids=torch.tensor([tokens]), past_key_values=cache, use_cache=True
            )
            cache = encoded.past_key_values
            records[f"encode.{step}"] = encoded.last_hidden_state.float().contiguous()
            decoded = model(
                past_key_values=cache, decoder_input_ids=torch.tensor([canvas]),
                self_conditioning_logits=prior,
            )
            records[f"logits.{step}"] = decoded.logits.float().contiguous()
            prior = (decoded.logits / 0.8).to(dtype)
            records[f"prior.{step}"] = prior.float().contiguous()
            print(f"step={step} context={cache.get_seq_length()} argmax={decoded.logits.argmax(-1).tolist()}", flush=True)
    if args.vision or args.vision_only:
        vision = model.model.encoder.vision_tower
        def record_vision(label):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                records[label] = value.float().contiguous()
            return hook
        vision.patch_embedder.register_forward_hook(record_vision("vision.embedding"))
        for index, layer in enumerate(vision.encoder.layers):
            layer.register_forward_hook(record_vision(f"vision.layer.{index}"))
        width = 3 * model.config.vision_config.patch_size ** 2
        height, columns = args.vision_grid
        count = height * columns
        patches = (torch.arange(count*width, dtype=torch.float32).reshape(1,count,width)*0.013).sin().mul(0.2).to(torch.bfloat16)
        positions = torch.tensor([[[column,row] for row in range(height) for column in range(columns)]])
        records["vision.grid"] = torch.tensor([height, columns], dtype=torch.int32)
        records["vision.patches"] = patches.float()
        records["vision.positions"] = positions.to(torch.int32)
        with torch.inference_mode():
            output = model.model.encoder.get_image_features(pixel_values=patches, image_position_ids=positions)
        records["vision.pooled"] = output.last_hidden_state.float().contiguous()
        records["vision.projected"] = output.pooler_output.float().contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(records, str(args.output))
    args.output.with_suffix(".json").write_text(json.dumps({
        "model_repository": "google/diffusiongemma-26B-A4B-it",
        "model_revision": MODEL_REVISION, "reference_revision": REFERENCE_REVISION,
        "cases": [] if args.vision_only else CASES, "torch": torch.__version__, "dtype": "F32" if args.gguf else "BF16",
        "quantized_source": quantized_source,
        "self_conditioning_temperature": 0.8, "seconds": time.monotonic() - started,
        "attention_implementation": "eager", "experts_implementation": "eager",
        "vision_grid": args.vision_grid if args.vision or args.vision_only else None,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
