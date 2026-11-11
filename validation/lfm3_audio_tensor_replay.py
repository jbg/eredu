"""Compare CPU MLX Ring Audio ranks against the serial released-payload runner."""

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time


def run_cohort(command, output, tensor, pipeline, residency='resident', reject=False):
    world = tensor * pipeline
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(world)]
        for listener in sockets:
            listener.bind(("127.0.0.1", 0))
        hosts = [[f"127.0.0.1:{listener.getsockname()[1]}"] for listener in sockets]

    hostfile = (output / "ring-hosts.json").resolve()
    hostfile.write_text(json.dumps(hosts))
    workers = []
    with ExitStack() as stack:
        try:
            for rank in range(world):
                trace = output / f"rank-{rank}.json"
                log = output / f"rank-{rank}.log"
                options = ["--tensor-cohort", "--pipeline-stages", str(pipeline)]
                for component in ("language", "encoder", "depth", "codec"):
                    options.extend((f"--{component}-residency", residency))
                if reject:
                    options.extend(("--reject-first-advance", "--tensor-seed", str(41 + rank)))
                worker = subprocess.Popen(
                    [*command(trace), *options],
                    env={**os.environ, "MLX_HOSTFILE": str(hostfile), "MLX_RANK": str(rank)},
                    stdout=stack.enter_context(log.open("w")),
                    stderr=subprocess.STDOUT,
                )
                workers.append((worker, log))
            deadline = time.monotonic() + 480
            while True:
                statuses = [(worker.poll(), log) for worker, log in workers]
                for status, log in statuses:
                    if status not in (None, 0):
                        raise RuntimeError(f"Audio rank exited {status}. See {log}")
                if all(status == 0 for status, _ in statuses):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Audio rank cohort timed out")
                time.sleep(0.25)
        finally:
            for worker, _ in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "assets", "manifest", "backbone", "projector", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--topology", choices=["tp2", "pp2", "tp2-pp2"], action="append")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--residency", choices=["resident", "host", "disk"], default="resident")
    parser.add_argument("--reject-divergent-seeds", action="store_true")
    args = parser.parse_args()
    if args.reject_divergent_seeds and (args.text or args.topology):
        parser.error('--reject-divergent-seeds uses the combined speech topology')
    args.output.mkdir(parents=True, exist_ok=True)

    def command(trace):
        return [
            str(args.runner.resolve()), str(args.assets), str(args.manifest), str(trace),
            "--device", "cpu", "--tensor-replay",
            "--gguf-backbone", str(args.backbone), "--gguf-projector", str(args.projector),
            *(["--tensor-text"] if args.text else []),
        ]

    if args.reject_divergent_seeds:
        run_cohort(command, args.output, 2, 2, args.residency, reject=True)
        results = [json.loads((args.output / f"rank-{rank}.json").read_text()) for rank in range(4)]
        if any(result != {"rejected": True, "committed_slots": 0} for result in results):
            raise RuntimeError("A rank committed a divergent depth decision")
        print(json.dumps({"ranks": 4, "rejected": True, "committed_slots": 0}), flush=True)
        return
    serial_path = args.output / "serial.json"
    subprocess.run(command(serial_path), check=True, timeout=480)
    serial = json.loads(serial_path.read_text())
    for name in args.topology or ["tp2", "pp2", "tp2-pp2"]:
        tensor, pipeline = {"tp2": (2, 1), "pp2": (1, 2), "tp2-pp2": (2, 2)}[name]
        cohort = args.output / name
        run_cohort(command, cohort, tensor, pipeline, args.residency)
        compare(serial, cohort, tensor, pipeline)


def compare(serial, cohort, tensor, pipeline):
    ranks = [
        json.loads((cohort / f"rank-{rank}.json").read_text())
        for rank in range(tensor * pipeline)
    ]
    if any(rank != ranks[0] for rank in ranks):
        raise RuntimeError("Audio ranks disagree")
    if any(ranks[0][key] != serial[key] for key in ("tokens", "codes")):
        raise RuntimeError("Distributed and serial Audio token or codebook IDs disagree")
    reference = serial["pcm"]
    output = ranks[0]["pcm"]
    if not all(math.isfinite(sample) for sample in reference + output):
        raise RuntimeError("Audio replay produced nonfinite PCM")
    errors = [a - b for a, b in zip(output, reference, strict=True)]
    max_absolute = max(map(abs, errors), default=0.0)
    energy = sum(x * x for x in reference)
    relative_l2 = math.sqrt(sum(e * e for e in errors) / energy) if energy else max_absolute
    if max_absolute > 5e-4 or relative_l2 > 1e-4:
        raise RuntimeError(f"Audio distributed PCM differs from serial: {max_absolute}, {relative_l2}")
    print(json.dumps({
        "tensor": tensor,
        "pipeline": pipeline,
        "ranks": len(ranks),
        "matched_tokens": len(serial["tokens"]),
        "matched_frames": len(serial["codes"]),
        "matched_samples": len(reference),
        "max_absolute_pcm": max_absolute,
        "relative_l2_pcm": relative_l2,
    }), flush=True)


if __name__ == "__main__":
    main()
