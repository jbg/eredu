#!/usr/bin/env bash
set -euo pipefail
reference_source=${1:?pristine fancy-regex 0.19.0 source directory required}
artifact_dir=${2:?artifact directory required}
regex_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
python3 - "$reference_source" "$regex_root" "$artifact_dir" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
reference, root, artifacts = map(Path, sys.argv[1:])
package = next(p for p in json.loads((root.parent / 'parser-upstream.json').read_text())['packages'] if p['name'] == 'fancy-regex' and p['version'] == '0.19.0')
for relative, expected in package['files'].items():
    if hashlib.sha256((reference / relative).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'pristine source mismatch: {relative}')
print(f"Verified {len(package['files'])} pristine upstream files.")
(artifacts / 'src/bin').mkdir(parents=True, exist_ok=True)
shutil.copyfile(root / 'validation/reference.rs', artifacts / 'src/bin/parity.rs')
shutil.copyfile(root / 'validation/performance.rs', artifacts / 'src/bin/performance.rs')
(artifacts / 'Cargo.toml').write_text(f'''[package]
name = "fancy-reference-check"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
fancy_regex = {{ package = "fancy-regex", path = {json.dumps(str(root))} }}
fancy_regex_reference = {{ package = "fancy-regex", version = "=0.19.0" }}
reference_automata = {{ package = "regex-automata", version = "=0.4.18" }}
reference_syntax = {{ package = "regex-syntax", version = "=0.8.11" }}
reference_aho = {{ package = "aho-corasick", version = "=1.1.5" }}
reference_memchr = {{ package = "memchr", version = "=2.8.3" }}
reference_bitset = {{ package = "bit-set", version = "=0.8.0" }}
reference_bitvec = {{ package = "bit-vec", version = "=0.8.0" }}
[profile.release]
lto = "thin"
''')
PY
cargo run --offline --release --manifest-path "$artifact_dir/Cargo.toml" --bin parity
cargo run --offline --release --manifest-path "$artifact_dir/Cargo.toml" --bin performance
