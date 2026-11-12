"""Build a local Eredu replay fixture from the same released GGUF pair in llama.cpp."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from safetensors.numpy import load_file, save_file

REFERENCE_REVISION = "7c2653acc749258dc62067da62d07024d971ccd2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("gguf_dir", type=Path)
    parser.add_argument("publisher_fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backbone", default="Q4_K_M", help="encoding in the source manifest")
    parser.add_argument("--projector", default="Q8_0", help="encoding in the source manifest")
    parser.add_argument("--single-row", action="store_true",
                        help="evaluate decoder prefill one row at a time in llama.cpp")
    parser.add_argument("--projector-cpu", action="store_true",
                        help="run the llama.cpp media projector on CPU")
    args = parser.parse_args()
    manifest_path = Path(__file__).parent / "fixtures/lfm3/vision-gguf-sources.json"
    manifest = json.loads(manifest_path.read_text())

    inputs = []
    digests = {}
    for role, encoding in [
        ("backbone", args.backbone),
        ("projector", args.projector),
    ]:
        source = manifest[f"{role}s"].get(encoding)
        if source is None:
            parser.error(f"no pinned {role} encoding {encoding!r}")
        path = args.gguf_dir / source["file"]
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if path.stat().st_size != source["bytes"] or digest != source["publisher_sha256"]:
            raise ValueError(f"{path} differs from the pinned GGUF source")
        inputs.append(path)
        digests[role] = digest

    metadata = json.loads((args.publisher_fixture / "reference.json").read_text())
    if metadata["dtype"] != "float32":
        raise ValueError("the pre-sized inputs require the publisher F32 fixture")
    publisher = load_file(args.publisher_fixture / "reference.safetensors")
    cases = tuple(f"{case['id']}." for case in metadata["cases"])
    tensors = {name: tensor for name, tensor in publisher.items() if name.startswith(cases)}
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.output / "llama-rows.f32"
    command = [args.runner.resolve(), *inputs, trace]
    if args.single_row:
        command.append("--single-row")
    if args.projector_cpu:
        command.append("--projector-cpu")
    with (args.output / "llama.log").open("w") as log:
        result = subprocess.run(
            command, check=True, text=True, stdout=subprocess.PIPE, stderr=log,
        )
    ids = [int(line) for line in result.stdout.splitlines()]
    if len(ids) != 8:
        raise ValueError("llama.cpp did not emit eight cached token IDs")
    metadata["conditional"]["generated"] = ids
    metadata["gguf_reference"] = {
        "llama_cpp_revision": REFERENCE_REVISION,
        "decoder_microbatch": 1 if args.single_row else 32,
        "projector_device": "cpu" if args.projector_cpu else "metal",
        **digests,
    }
    del metadata["raw_processor"]
    rows = np.fromfile(trace, dtype="<f4")
    offset = 0

    def take(shape):
        nonlocal offset
        end = offset + int(np.prod(shape))
        result = rows[offset:end].reshape(shape)
        offset = end
        return result

    for case, shape in [("image", (6, 2048)), ("video_odd", (12, 2048))]:
        tensors[f"{case}.projected"] = take(shape)
    tensors["conditional.prefill"] = take((1, len(metadata["conditional"]["prompt"]), 128000))
    for step in range(len(ids)):
        tensors[f"conditional.decode.{step}"] = take((1, 1, 128000))
    if offset != rows.size or not np.isfinite(rows).all():
        raise ValueError("unexpected llama.cpp score payload")
    save_file(tensors, args.output / "reference.safetensors")
    (args.output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
