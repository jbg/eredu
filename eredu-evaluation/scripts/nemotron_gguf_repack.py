"""Repack a pinned GGUF into shards with the pinned primary writer.

Metadata, encodings and tensor bytes remain exact; only physical tensor order,
offsets and split membership change. Outputs stay outside tracked source.
"""
import argparse
import hashlib
import json
from pathlib import Path

import gguf


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-bytes", type=int, default=1_500_000_000)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repository) or args.maximum_bytes <= 0:
        parser.error("select positive shard geometry and an output outside tracked source")
    records = json.loads((repository / "eredu-architectures/tests/fixtures/nemotron_labs_diffusion/conversions.json").read_text())
    records.append(json.loads((repository / "eredu-architectures/tests/fixtures/nemotron_labs_diffusion/vlm-gguf-conversion.json").read_text()))
    candidates = [(record, member) for record in records for member in record["weight_files"] if member["rfilename"] == args.artifact.name]
    assert len(candidates) == 1
    pin, member = candidates[0]
    assert args.artifact.stat().st_size == member["size"]
    assert hashlib.file_digest(args.artifact.open("rb"), "sha256").hexdigest() == member["lfs"]["sha256"]
    writer_path = Path(gguf.__file__).resolve().parent / "gguf_writer.py"
    assert hashlib.sha256(writer_path.read_bytes()).hexdigest() == "3f0f759149dbd9b1d860cc18667f6b3a4efd4bc25fb5449c07665482073363d8"
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("*.gguf")):
        raise SystemExit("output already contains GGUF artifacts")
    reader = gguf.GGUFReader(str(args.artifact))
    writer = gguf.GGUFWriter(args.output / args.artifact.name, reader.fields["general.architecture"].contents(), split_max_size=args.maximum_bytes)
    assert reader.alignment == 32
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in {"split.no", "split.count", "split.tensors.count"}:
            continue
        writer.add_key_value(key, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    for tensor in reversed(reader.tensors):
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    result = {
        "original_repository": pin["repo"], "original_revision": pin["revision"],
        "original_sha256": member["lfs"]["sha256"], "reference_revision": pin["reference_revision"],
        "writer_sha256": hashlib.sha256(writer_path.read_bytes()).hexdigest(),
        "repack_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maximum_bytes": args.maximum_bytes,
        "members": [{"name": file.name, "size": file.stat().st_size, "sha256": hashlib.file_digest(file.open("rb"), "sha256").hexdigest()} for file in sorted(args.output.glob("*.gguf"))],
    }
    (args.output / "repack-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
