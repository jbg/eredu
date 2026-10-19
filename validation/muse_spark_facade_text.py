"""Compare literal facade text with independent decoding of pinned Spark BPE bytes.

Only base-vocabulary output IDs are accepted: protocol headers and special-token
rendering need a separate semantic reference. This checks incremental UTF-8 state
without importing the inference engine or its streaming decoder.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

RANK_SHA256 = "d0bdbaf59b0762c8c807617e2d8ea51420eb1b1de266df2495be755c8e0ed6ed"


def compare(model, run):
    raw = (model / "l4_200k_base").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RANK_SHA256
    ranks = {int(rank): base64.b64decode(token, validate=True)
             for token, rank in (line.split() for line in raw.splitlines())}
    assert set(ranks) == set(range(200000))
    results = []
    for directory in sorted(run.glob("rank-*")):
        if not directory.is_dir():
            continue
        report_path = directory / "comparison-evidence.json"
        report_raw = report_path.read_bytes()
        report = json.loads(report_raw)
        assert report["comparison_verified"] is True
        ids = report["ordinary_ids"]
        assert ids == report["assisted_ids"] and all(token in ranks for token in ids)
        expected = b"".join(ranks[token] for token in ids).decode("utf-8", errors="replace")
        trace = run / f"{directory.name}.log"
        trace_raw = trace.read_bytes()
        prefix = f"assisted rank={report['global_rank']} event="
        deltas = [json.loads(match.group(1)) for line in trace_raw.decode("utf-8").splitlines()
                  if line.startswith(prefix)
                  and (match := re.search(r"event=TextDelta\((.*)\)$", line))]
        actual = "".join(deltas)
        assert actual == expected, (directory.name, actual, expected)
        results.append({"rank": directory.name, "text": actual,
                        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
                        "trace_sha256": hashlib.sha256(trace_raw).hexdigest(), "verified": True})
    assert results
    return {"rank_sha256": RANK_SHA256,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "method": "concatenate published base-token byte strings, then decode UTF-8 with replacement; compare assisted semantic deltas",
            "scope": "literal base-vocabulary output; independent of inference and streaming state", "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.model, args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
