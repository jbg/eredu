#!/usr/bin/env bash
set -euo pipefail
reference_source=${1:?pristine fancy-regex 0.19.0 source directory required}
artifact_dir=${2:?artifact directory required}
dependency_dir=${3:?Cargo dependency directory required}
automata_library=${4:?exact local regex-automata rlib required}
syntax_library=${5:?exact local regex-syntax rlib required}
bitset_library=${6:?exact bit-set rlib required}
hashbrown_library=${7:?exact local hashbrown rlib required}
regex_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
regex_compiler=${RUSTC:-$(rustup which rustc)}
python3 - "$reference_source" "$regex_root/../parser-upstream.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
source = Path(sys.argv[1])
package = next(p for p in json.loads(Path(sys.argv[2]).read_text())["packages"] if p["name"] == "fancy-regex" and p["version"] == "0.19.0")
for relative, expected in package["files"].items():
    if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"pristine reference hash mismatch: {relative}")
print(f"Verified {len(package['files'])} pristine upstream files.")
PY
mkdir -p -- "$artifact_dir"
common=(-C embed-bitcode=yes -C "opt-level=${REFERENCE_OPT_LEVEL:-0}" --edition=2018 --crate-type lib --cfg 'feature="std"' --cfg 'feature="unicode"' --cfg 'feature="perf"' --cfg 'feature="variable-lookbehinds"' -L "dependency=$dependency_dir" --extern "regex_automata=$automata_library" --extern "regex_syntax=$syntax_library" --extern "bit_set=$bitset_library")
"$regex_compiler" "${common[@]}" --crate-name fancy_regex_reference "$reference_source/src/lib.rs" -o "$artifact_dir/libfancy_regex_reference.rlib"
"$regex_compiler" "${common[@]}" --extern "hashbrown=$hashbrown_library" --crate-name fancy_regex "$regex_root/src/lib.rs" -o "$artifact_dir/libfancy_regex.rlib"
"$regex_compiler" -C "opt-level=${REFERENCE_OPT_LEVEL:-0}" -C "lto=${REFERENCE_LTO:-off}" --edition=2021 "$regex_root/validation/reference.rs" -L "dependency=$dependency_dir" \
 --extern "fancy_regex=$artifact_dir/libfancy_regex.rlib" --extern "fancy_regex_reference=$artifact_dir/libfancy_regex_reference.rlib" -o "$artifact_dir/fancy-reference-probe"
"$artifact_dir/fancy-reference-probe"

if [[ ${REFERENCE_BENCHMARK:-0} == 1 ]]; then
    "$regex_compiler" -C "opt-level=${REFERENCE_OPT_LEVEL:-0}" -C "lto=${REFERENCE_LTO:-off}" --edition=2021 "$regex_root/validation/performance.rs" -L "dependency=$dependency_dir" \
     --extern "fancy_regex=$artifact_dir/libfancy_regex.rlib" --extern "fancy_regex_reference=$artifact_dir/libfancy_regex_reference.rlib" -o "$artifact_dir/fancy-performance-probe"
    "$artifact_dir/fancy-performance-probe"
fi
