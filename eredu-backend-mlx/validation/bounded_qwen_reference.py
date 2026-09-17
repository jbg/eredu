"""Independent released-checkpoint score reference (MLX-LM 0.31.3 / MLX 0.32.2).

Reads exact prompt/decode IDs; teacher-forced decode keeps every comparison on
identical inputs. The checkpoint must carry the verified eredu-provenance.json
manifest. This script needs no network and never executes checkpoint Python code.
"""
import argparse
import hashlib
import importlib.metadata
import json
import resource
from pathlib import Path


def verify(checkpoint):
    manifest = json.loads((checkpoint / "eredu-provenance.json").read_text())
    if manifest["repository"] != "Qwen/Qwen3.5-0.8B" or manifest["revision"] != "2fc06364715b967f1860aea9cf38778875588b17":
        raise ValueError("Expected the official Qwen3.5-0.8B repository")
    weight = next(entry for entry in manifest["files"] if entry["name"] == "model.safetensors-00001-of-00001.safetensors")
    if weight["digest"] != "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696":
        raise ValueError("Manifest does not describe the pinned released weights")
    for entry in manifest["files"]:
        path = checkpoint / entry["name"]
        if path.parent != checkpoint or path.stat().st_size != entry["size"]:
            raise ValueError(f"Invalid checkpoint file: {path}")
        digest = hashlib.sha256() if entry["digest_algorithm"] == "sha256" else hashlib.sha1()
        if entry["digest_algorithm"] == "git-blob-sha1":
            digest.update(f"blob {entry['size']}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["digest"]:
            raise ValueError(f"Checkpoint digest mismatch: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    provenance = verify(args.checkpoint)
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    data = json.loads(args.input.read_text())
    model, _ = load(str(args.checkpoint))
    cache = model.make_cache()
    mx.reset_peak_memory()
    passes = []
    for step, tokens in enumerate([data["prompt_ids"]] + [[n] for n in data["decode_ids"]]):
        logits = model(mx.array([tokens]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        values = np.array(logits)
        if not np.isfinite(values).all():
            raise ValueError("Reference produced non-finite scores")
        passes.append({"step": step, "shape": [1, values.size], "scores": values.tolist(),
                       "argmax": int(values.argmax()), "mlx_peak_bytes": mx.get_peak_memory()})
        del logits, values
    report = {"provenance": provenance, "versions": {name: importlib.metadata.version(name)
              for name in ("mlx", "mlx-lm", "transformers", "numpy")},
              "prompt_ids": data["prompt_ids"], "decode_ids": data["decode_ids"], "passes": passes,
              "process_maxrss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    passed = True
    if args.compare:
        native = json.loads(args.compare.read_text())
        if any(native[key] != data[key] for key in ("prompt_ids", "decode_ids")) or len(native["passes"]) != len(passes):
            raise ValueError("Native report used different prompt/decode inputs")
        comparisons = []
        for actual, expected in zip(native["passes"], passes):
            a, b = np.array(actual["scores"]), np.array(expected["scores"])
            if actual["shape"] != expected["shape"] or a.shape != b.shape:
                raise ValueError("Native/reference score geometry differs")
            delta = a - b
            close = bool(np.allclose(a, b, atol=args.atol, rtol=args.rtol))
            passed &= close and actual["argmax"] == expected["argmax"]
            comparisons.append({"step": expected["step"], "max_abs": float(np.abs(delta).max()),
                                "rmse": float(np.sqrt(np.mean(delta**2))), "allclose": close,
                                "argmax_equal": actual["argmax"] == expected["argmax"]})
        report["comparison"] = {"native": str(args.compare), "atol": args.atol, "rtol": args.rtol,
                                "passed": passed, "passes": comparisons}
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "argmax": [p["argmax"] for p in passes],
                      "comparison": report.get("comparison")}, indent=2), flush=True)
    if not passed:
        raise AssertionError("Released-checkpoint scores exceed the declared tolerance")


if __name__ == "__main__":
    main()
