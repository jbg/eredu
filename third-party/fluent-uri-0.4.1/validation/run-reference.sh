#!/usr/bin/env bash
set -euo pipefail
reference_source=${1:?pristine fluent-uri 0.4.1 source directory required}
artifact_dir=${2:?artifact directory required}
dependency_dir=${3:?Cargo dependency directory containing borrow-or-share and ref-cast rlibs required}
uri_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
uri_compiler=${RUSTC:-$(rustup which rustc)}
python3 - "$reference_source" "$uri_root/../parser-upstream.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
source = Path(sys.argv[1])
package = next(p for p in json.loads(Path(sys.argv[2]).read_text())["packages"] if p["name"] == "fluent-uri" and p["version"] == "0.4.1")
for relative, expected in package["files"].items():
    if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"pristine reference hash mismatch: {relative}")
print(f"Verified {len(package['files'])} pristine upstream files.")
PY
mkdir -p -- "$artifact_dir"
borrow_libraries=("$dependency_dir"/libborrow_or_share-*.rlib)
cast_libraries=("$dependency_dir"/libref_cast-*.rlib)
common=(--edition=2021 --crate-type lib --cfg 'feature="alloc"' --cfg 'feature="std"' --cfg 'feature="impl-error"'
    -L "dependency=$dependency_dir" --extern "borrow_or_share=${borrow_libraries[0]}" --extern "ref_cast=${cast_libraries[0]}")
"$uri_compiler" "${common[@]}" --crate-name fluent_uri_reference "$reference_source/src/lib.rs" -o "$artifact_dir/libfluent_uri_reference.rlib"
"$uri_compiler" "${common[@]}" --crate-name fluent_uri "$uri_root/src/lib.rs" -o "$artifact_dir/libfluent_uri.rlib"
"$uri_compiler" --edition=2021 "$uri_root/validation/reference.rs" -L "dependency=$dependency_dir" \
    --extern "fluent_uri=$artifact_dir/libfluent_uri.rlib" --extern "fluent_uri_reference=$artifact_dir/libfluent_uri_reference.rlib" \
    -o "$artifact_dir/uri-reference-probe"
"$artifact_dir/uri-reference-probe"
