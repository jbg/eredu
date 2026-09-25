#!/usr/bin/env python3
"""Validate integrated mixed GEMM against the same binary's cast-GEMM reference.

Requires release projection_baseline and projection_parity examples. All GPU runs
are serial. Ordinary timing is isolated from logits/capture and kernel tracing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

from projection_baseline import (
    CHECKPOINT_REVISION, CHECKPOINT_SHA256, command_output, sha256, summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-dir', type=Path, default=Path('target/release/examples'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--disable-tf32', action='store_true')
    args = parser.parse_args()
    if args.samples < 1 or sha256(args.model) != CHECKPOINT_SHA256:
        parser.error('positive sample count and pinned checkpoint required')
    args.output.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MLX_', 'EREDU_LFM2_'))}
    if args.disable_tf32:
        env['MLX_ENABLE_TF32'] = '0'
    manifest = {
        'checkpoint_revision': CHECKPOINT_REVISION, 'checkpoint_sha256': CHECKPOINT_SHA256,
        'hardware': command_output(['sysctl', '-n', 'machdep.cpu.brand_string']),
        'platform': platform.platform(), 'tf32_disabled': args.disable_tf32,
        'git_revision': command_output(['git', 'rev-parse', 'HEAD']),
        'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD', '--'])).hexdigest(),
        'binary_sha256': {name: sha256(args.binary_dir / name) for name in ['projection_baseline', 'projection_parity']},
        'runner_sha256': sha256(__file__),
        'source_sha256': {name: sha256(name) for name in [
            'eredu/examples/projection_parity.rs', 'eredu/examples/projection_baseline.rs',
            'safemlx-sys/src/mlx-c/mlx/c/mixed_gemm.cpp',
            'safemlx-sys/src/mlx-c/patches/mlx-metal-mixed-storage-gemm.patch']},
        'samples': args.samples, 'runs': [], 'cases': [],
        'notes': ['Same-build reference disables only multi-row mixed GEMM with a thread-affine diagnostic guard.',
                  'No tracing or capture during ordinary measurements; allocator cache zero; loading excluded.',
                  'First request and five reset/warm requests are separate; peak is total MLX active bytes, not RSS.',
                  'Logits are compared as full F32 bit arrays; SHA-256 is only the serialized evidence.',
                  'Self-drafting pinned checkpoint exercises two-proposal verification and cached decode.'],
    }

    def save():
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def run(name, binary, parameters, trace=False):
        print(name, flush=True)
        output = (args.output / (name + '.json')).resolve()
        log = args.output / (name + '.log')
        command = [str((args.binary_dir / binary).resolve()), str(args.model.absolute()), str(output), *parameters]
        started = time.monotonic()
        with log.open('w') as stream:
            result = subprocess.run(command, env={**env, **({'MLX_METAL_LOG_KERNEL_SELECTION': '1'} if trace else {})},
                                    stdout=stream, stderr=subprocess.STDOUT)
        manifest['runs'].append({'name': name, 'command': command, 'exit_code': result.returncode,
                                 'elapsed_seconds': time.monotonic() - started, 'log_sha256': sha256(log)})
        save()
        if result.returncode:
            raise SystemExit(f'failed {name}: {log}')
        return json.loads(output.read_text()), log.read_text()

    for policy in ['268435456', 'unlimited']:
        for positions in [128, 2000]:
            label = f'{policy}-{positions}'
            parity, _ = run(label + '-parity', 'projection_parity', [policy, str(positions)])
            baseline = {}
            for selection in ['reference', 'mixed']:
                baseline[selection], _ = run(label + '-' + selection, 'projection_baseline',
                    ['baseline', policy, str(positions), str(args.samples), selection])
            for request in baseline['mixed']['ordinary']:
                report = request['retention_after']
                assert report['status'] == 'available'
                assert all(group['usage']['value']['retained_payload_bytes'] == 0 for group in report['value']), 'eligible model retained a promoted weight'
            expected = baseline['reference']['ordinary'][0]['tokens']
            assert all(r['tokens'] == expected for b in baseline.values() for r in b['ordinary'])
            capture, _ = run(label + '-capture', 'projection_baseline',
                            ['capture', policy, str(positions), str(args.samples), 'mixed'])
            traced, trace = run(label + '-trace', 'projection_baseline',
                               ['capture', policy, str(positions), '1', 'mixed'], trace=True)
            assert traced['classes'] == capture['classes']
            case = summarize(capture, baseline['mixed'], trace)
            assert sum(c['invocations'] for c in case['classes']) == 93
            assert all(c['observed_path'] == 'mixed_gemm' for c in case['classes'])
            for replay in case['replays']:
                phases = {p['phase']: p for p in replay['phases']}
                mixed = phases['eredu_dispatch']
                native = phases['preconverted_gemm']
                normalized = {k.replace('_storage_bfloat16', '').replace('_storage_float16', '') for k in mixed['kernel_selections']}
                assert normalized == set(native['kernel_selections']), 'native arithmetic dispatch changed'
                assert mixed['peak_growth_bytes'] == native['peak_growth_bytes'], 'unexpected allocation beyond native workspace'
            case['parity'] = parity
            case['reference_ordinary'] = baseline['reference']['ordinary']
            case['summary'] = {selection: {
                'cold_first_token_ms': b['ordinary'][0]['first_token_ms'],
                'cold_prefill_peak_bytes': b['ordinary'][0]['prefill_active_peak_bytes'],
                'warm_first_token_ms': statistics.median(r['first_token_ms'] for r in b['ordinary'][1:]),
                'warm_prefill_peak_bytes': max(r['prefill_active_peak_bytes'] for r in b['ordinary'][1:]),
            } for selection, b in baseline.items()}
            manifest['cases'].append(case)
            save()
            print(json.dumps(case['summary']), flush=True)


if __name__ == '__main__':
    main()
