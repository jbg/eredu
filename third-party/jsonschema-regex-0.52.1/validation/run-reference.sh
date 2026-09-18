#!/usr/bin/env bash
set -euo pipefail
reference_source=${1:?pristine jsonschema-regex 0.52.1 source directory required}
artifact_dir=${2:?artifact directory required}
dependency_dir=${3:?Cargo dependency directory required}
syntax_library=${4:?exact local regex-syntax rlib required}
regex_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
regex_compiler=${RUSTC:-$(rustup which rustc)}
python3 - "$reference_source" "$regex_root/../parser-upstream.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
source = Path(sys.argv[1])
package = next(p for p in json.loads(Path(sys.argv[2]).read_text())["packages"] if p["name"] == "jsonschema-regex" and p["version"] == "0.52.1")
for relative, expected in package["files"].items():
    if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"pristine reference hash mismatch: {relative}")
print(f"Verified {len(package['files'])} pristine upstream files.")
PY
mkdir -p -- "$artifact_dir"
common=(--edition=2021 --crate-type lib -L "dependency=$dependency_dir" --extern "regex_syntax=$syntax_library")
"$regex_compiler" "${common[@]}" --crate-name jsonschema_regex_reference "$reference_source/src/lib.rs" -o "$artifact_dir/libjsonschema_regex_reference.rlib"
"$regex_compiler" "${common[@]}" --crate-name jsonschema_regex "$regex_root/src/lib.rs" -o "$artifact_dir/libjsonschema_regex.rlib"
"$regex_compiler" --edition=2021 "$regex_root/validation/reference.rs" -L "dependency=$dependency_dir" \
    --extern "jsonschema_regex=$artifact_dir/libjsonschema_regex.rlib" \
    --extern "jsonschema_regex_reference=$artifact_dir/libjsonschema_regex_reference.rlib" -o "$artifact_dir/ecma-reference-probe"
"$artifact_dir/ecma-reference-probe"
