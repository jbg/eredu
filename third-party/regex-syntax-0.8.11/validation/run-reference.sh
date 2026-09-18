#!/usr/bin/env bash
set -euo pipefail

# Pass the pristine regex-syntax 0.8.11 source directory and an artifact directory.
reference_source=${1:?pristine source directory required}
artifact_dir=${2:?artifact directory required}
syntax_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
syntax_compiler=${RUSTC:-$(rustup which rustc)}
python3 - "$reference_source" "$syntax_root/../parser-upstream.json" <<'PYVERIFY'
import hashlib
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
provenance = json.loads(Path(sys.argv[2]).read_text())
package = next(p for p in provenance["packages"]
               if p["name"] == "regex-syntax" and p["version"] == "0.8.11")
for relative, expected in package["files"].items():
    actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"pristine reference hash mismatch: {relative}")
print(f"Verified {len(package['files'])} pristine upstream files.")
PYVERIFY
mkdir -p -- "$artifact_dir"

features=(--cfg 'feature="std"' --cfg 'feature="unicode"'
    --cfg 'feature="unicode-age"' --cfg 'feature="unicode-bool"'
    --cfg 'feature="unicode-case"' --cfg 'feature="unicode-gencat"'
    --cfg 'feature="unicode-perl"' --cfg 'feature="unicode-script"'
    --cfg 'feature="unicode-segment"')

"$syntax_compiler" --edition=2021 --crate-type lib --crate-name regex_syntax_reference \
    "${features[@]}" "$reference_source/src/lib.rs" \
    -o "$artifact_dir/libregex_syntax_reference.rlib"
"$syntax_compiler" --edition=2021 --crate-type lib --crate-name regex_syntax \
    "${features[@]}" "$syntax_root/src/lib.rs" \
    -o "$artifact_dir/libregex_syntax_funding.rlib"
"$syntax_compiler" --edition=2021 "$syntax_root/validation/reference.rs" \
    --extern "regex_syntax=$artifact_dir/libregex_syntax_funding.rlib" \
    --extern "regex_syntax_reference=$artifact_dir/libregex_syntax_reference.rlib" \
    -o "$artifact_dir/regex-syntax-reference-probe"
"$artifact_dir/regex-syntax-reference-probe"
