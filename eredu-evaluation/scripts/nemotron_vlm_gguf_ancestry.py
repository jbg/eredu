"""Independently compare pinned VLM GGUF tensors with the released source.

Comparisons retain at most one projection matrix, or a bounded slice of the
embedding/readout. Vision companion row-break metadata is checked separately
from the model's parameter catalog.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

from nemotron_reference_provenance import verify_release


def source_name(name, vision):
    if vision:
        direct = {
            "v.token_embd.img_break": "encoder.embed_tokens.weight",
            "v.patch_embd.weight": "encoder.vision_tower.patch_conv.weight",
            "v.pre_ln.weight": "encoder.vision_tower.ln_pre.weight",
            "mm.1.weight": "encoder.multi_modal_projector.linear_1.weight",
            "mm.2.weight": "encoder.multi_modal_projector.linear_2.weight",
            "mm.input_norm.weight": "encoder.multi_modal_projector.norm.weight",
            "mm.patch_merger.weight": "encoder.multi_modal_projector.patch_merger.merging_layer.weight",
        }
        if name in direct:
            return direct[name]
        prefix, block, index, field, suffix = name.split(".")
        assert prefix == "v" and block == "blk" and suffix == "weight"
        fields = {
            "attn_q": "attention.q_proj", "attn_k": "attention.k_proj",
            "attn_v": "attention.v_proj", "attn_out": "attention.o_proj",
            "ln1": "attention_norm", "ln2": "ffn_norm",
            "ffn_gate": "feed_forward.gate_proj", "ffn_up": "feed_forward.up_proj",
            "ffn_down": "feed_forward.down_proj",
        }
        return f"encoder.vision_tower.transformer.layers.{index}.{fields[field]}.weight"
    if name.startswith("blk."):
        block, index, field, suffix = name.split(".")
        assert block == "blk" and suffix == "weight"
        fields = {
            "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
            "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
            "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
        }
        return f"encoder.layers.{index}.{fields[field]}.weight"
    return {
        "token_embd.weight": "encoder.embed_tokens.weight",
        "output_norm.weight": "encoder.norm.weight",
        "output.weight": "diffusion_head.weight",
    }[name]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("conversion", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vision-only", action="store_true")
    args = parser.parse_args()
    pin = json.loads((Path(__file__).resolve().parents[2] / "eredu-architectures/tests/fixtures/nemotron_labs_diffusion/vlm-gguf-conversion.json").read_text())
    release = verify_release(args.checkpoint)
    assert release["repo"] == pin["base_repository"] and release["revision"] == pin["base_revision"]
    assert hashlib.sha256((Path(gguf.__file__).resolve().parent / "quants.py").read_bytes()).hexdigest() == pin["reference_quants_sha256"]
    config = json.loads((args.checkpoint / "config.json").read_text())
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    results = []
    with ExitStack() as stack:
        stores = {name: stack.enter_context(safe_open(args.checkpoint / name, framework="pt", device="cpu")) for name in set(index.values())}
        for member in pin["weight_files"]:
            vision = member["rfilename"] == "mmproj-F16.gguf"
            if args.vision_only and not vision:
                continue
            path = args.conversion / member["rfilename"]
            assert path.stat().st_size == member["size"]
            assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == member["lfs"]["sha256"]
            reader = gguf.GGUFReader(str(path))
            assert len(reader.tensors) == (223 if vision else 309)
            for tensor in reader.tensors:
                key = source_name(tensor.name, vision)
                source = stores[index[key]].get_slice(key)
                shape = tuple(source.get_shape())
                stored_shape = tuple(int(x) for x in reversed(tensor.shape))
                dtype = {gguf.GGMLQuantizationType.F32: np.float32, gguf.GGMLQuantizationType.F16: np.float16}[tensor.tensor_type]
                actual = tensor.data.reshape(stored_shape).view(np.uint8).reshape(-1)
                if tensor.name == "v.token_embd.img_break":
                    values = source[20].float().numpy()
                    assert stored_shape == values.shape
                    differing = int(np.count_nonzero(values.astype(dtype).view(np.uint8) != actual))
                else:
                    assert stored_shape == shape
                    paired = key.endswith(("q_proj.weight", "k_proj.weight"))
                    values = None
                    if paired:
                        heads = config["vision_config"]["num_attention_heads"] if vision else config["num_attention_heads"] if key.endswith("q_proj.weight") else config["num_key_value_heads"]
                        # Published conversion changes half-head coordinates to
                        # adjacent pairs. This comparison applies the producer
                        # transformation directly to the released values.
                        values = source[:].float().numpy().reshape(heads, 2, shape[0] // heads // 2, *shape[1:]).swapaxes(1, 2).reshape(shape)
                    row_elements = int(np.prod(shape[1:]))
                    rows_per_chunk = max(1, (8 << 20) // (row_elements * 4))
                    row_bytes = row_elements * np.dtype(dtype).itemsize
                    differing = 0
                    for start in range(0, shape[0], rows_per_chunk):
                        end = min(start + rows_per_chunk, shape[0])
                        chunk = values[start:end] if values is not None else source[start:end].float().numpy()
                        expected = np.ascontiguousarray(chunk.astype(dtype)).view(np.uint8).reshape(-1)
                        differing += int(np.count_nonzero(expected != actual[start * row_bytes:end * row_bytes]))
                results.append({"member": member["rfilename"], "tensor": tensor.name, "source": key, "bytes": int(actual.size), "differing_bytes": differing})
                print(f"{tensor.name}: differing_bytes={differing}", flush=True)
    result = {
        "repository": pin["repo"], "revision": pin["revision"],
        "official_repository": release["repo"], "official_revision": release["revision"],
        "reference_revision": pin["reference_revision"],
        "compared_bytes": sum(item["bytes"] for item in results),
        "differing_bytes": sum(item["differing_bytes"] for item in results), "tensors": results,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"compared_bytes={result['compared_bytes']} differing_bytes={result['differing_bytes']}", flush=True)
    if result["differing_bytes"]:
        raise SystemExit("released-source conversion differs")


if __name__ == "__main__":
    main()
