#!/usr/bin/env python3
"""Download and verify pinned public component-validation checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

FIXTURES = {
    'smollm2': ('HuggingFaceTB/SmolLM2-135M-Instruct',
                '12fd25f77366fa6b3b4b768ec3050bf629380bac',
                '5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c'),
    'lfm2': ('LiquidAI/LFM2-350M',
             'f37d3f5c8c5484bc01dad379a595cf4c68c4e70e',
             '387638dc889ff1a1395c3c2ab9605211e4c7e16f2d375361dd4e423b909a254e'),
    'lfm2-moe': ('LiquidAI/LFM2-8B-A1B',
                 'c1c44ff9fc00db3ebf4516970563f5f383d23670', {
                     'model-00001-of-00004.safetensors':
                         '927feafae7d99f40046cd365e8780fe2c36c72ba207ce3024ba54bf884599600',
                     'model-00002-of-00004.safetensors':
                         'e41140e9841cf338ed537fad0e0d0f16fbeebe0784719188ef7f2999b531d97e',
                     'model-00003-of-00004.safetensors':
                         '3e3b6236812186ae90f27a3d074f802d5251eab54840b8eca5eea8b730b7a789',
                     'model-00004-of-00004.safetensors':
                         '4f3cbb6a6a853785186b259fc7b23a5706cca8e53691ab3036a539596ffe38ea',
                 }),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    parser.add_argument('--model', choices=sorted(FIXTURES), default='smollm2')
    args = parser.parse_args()
    repository, revision, sha256 = FIXTURES[args.model]
    root = args.destination.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error('checkpoint validation artifacts must live outside the repository')
    root.mkdir(parents=True, exist_ok=True)
    metadata = HfApi().model_info(repository, revision=revision, files_metadata=True)
    assert metadata.sha == revision
    expected = {'model.safetensors': sha256} if isinstance(sha256, str) else sha256
    weights = {item.rfilename: item for item in metadata.siblings
               if item.rfilename.endswith('.safetensors')}
    assert weights.keys() == expected.keys()
    for name, digest in expected.items():
        assert weights[name].lfs.sha256 == digest
    checkpoint = Path(snapshot_download(repository, revision=revision,
        cache_dir=str(root / 'hf-cache'), allow_patterns=[
            'config.json', 'generation_config.json', *expected,
            'model.safetensors.index.json',
            'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
            'chat_template.jinja']))
    digests = {}
    for name, expected_digest in expected.items():
        with (checkpoint / name).open('rb') as source:
            digests[name] = hashlib.file_digest(source, 'sha256').hexdigest()
        assert digests[name] == expected_digest
    if len(expected) > 1:
        index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
        assert set(index['weight_map'].values()) == set(expected)
    # Preserve the existing single-file provenance consumed by dense references.
    verified = digests['model.safetensors'] if isinstance(sha256, str) else digests
    provenance = {'repository': repository, 'revision': revision, 'path': str(checkpoint),
                  'weights_sha256': verified, 'remote_lfs_sha256': sha256}
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
