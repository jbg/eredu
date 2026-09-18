#!/usr/bin/env bash
set -euo pipefail
reference_source=${1:?pristine registry source directory required}
artifact_dir=${2:?artifact directory required}
automata_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
python3 - "$reference_source" "$automata_root" "$artifact_dir" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
reference, root, artifacts = map(Path, sys.argv[1:])
provenance = json.loads((root.parent / 'regex-automata-upstream.json').read_text())
for name, expected in provenance['files'].items():
    if hashlib.sha256((reference / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'pristine source mismatch: {name}')
print(f"Verified {len(provenance['files'])} pristine upstream files.")
(artifacts / 'src').mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / 'validation/onepass-reference.rs', artifacts / 'src/main.rs')
features = '["std", "syntax", "dfa-onepass", "unicode"]'
(artifacts / 'Cargo.toml').write_text(f'''[package]
name = "onepass-reference-check"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
selected = {{ package = "regex-automata", path = {json.dumps(str(root))}, default-features = false, features = {features} }}
reference = {{ package = "regex-automata", version = "=0.4.18", default-features = false, features = {features} }}
[patch.crates-io]
regex-syntax = {{ path = {json.dumps(str(root.parent / 'regex-syntax-0.8.11'))} }}
''')
PY
cargo run --offline --release --manifest-path "$artifact_dir/Cargo.toml"
