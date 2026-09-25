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
    CHECKPOINT_REVISION, CHECKPOINT_SHA256, command_output, sha256, summarize, compare_reference_case,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-dir', type=Path, default=Path('target/release/examples'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--disable-tf32', action='store_true')
    parser.add_argument('--forecasts', action='store_true', help='validate cold, loaded and continuation forecasts')
    parser.add_argument('--reference-manifest', type=Path, help='require historical projection bit fingerprints')
    args = parser.parse_args()
    if args.samples < 1 or sha256(args.model) != CHECKPOINT_SHA256:
        parser.error('positive sample count and pinned checkpoint required')
    reference = json.loads(args.reference_manifest.read_text()) if args.reference_manifest else None
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
        'binary_sha256': {name: sha256(args.binary_dir / name) for name in (['projection_baseline', 'projection_parity'] + (['projection_forecast'] if args.forecasts else []))},
        'runner_sha256': sha256(__file__),
        'reference_manifest_sha256': sha256(args.reference_manifest) if reference else None,
        'source_sha256': {name: sha256(name) for name in [
            'eredu/examples/projection_parity.rs', 'eredu/examples/projection_baseline.rs',
            'eredu/examples/projection_forecast.rs', 'eredu-runtime/src/projection_memory.rs',
            'eredu-runtime/src/workspace_resources.rs',
            'eredu-architectures/src/lfm2/block.rs',
            'eredu-backend-mlx/src/backend/nn/mixed_projection.rs',
            'eredu-backend-mlx/src/backend/runtime/residency/manager.rs',
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
            forecast = None
            if args.forecasts:
                forecast, _ = run(label + '-forecast', 'projection_forecast', [policy, str(positions)])
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
            if reference:
                previous = next(c for c in reference['cases'] if c['positions'] == positions and c['retention_policy'] == policy)
                compare_reference_case(case, previous)
                case['matches_reference_manifest'] = True
            if forecast:
                topology = forecast['loaded']['request']['domains'][0]['executions'][0]['execution_topology']
                assert forecast['native_binding_count'] == 93
                assert len(topology['projection_input_normalizations']) == 93
                gains = set(topology['f32_rms_normalization_gains'])
                assert all(sources and set(sources) <= gains for sources in topology['projection_input_normalizations'].values())
                def peak(doc):
                    return doc['estimate']['domains'][0]['generation_peak']['upper_bytes']
                case['forecast'] = {
                    'cold_overall_upper': forecast['cold']['estimate']['domains'][0]['overall_peak']['upper_bytes'],
                    'loaded_generation_upper': peak(forecast['loaded']),
                    'without_projection_facts_upper': forecast['unrefined_estimate']['domains'][0]['generation_peak']['upper_bytes'],
                    'observed_prefill_active_growth': forecast['observed_prefill_active_growth'],
                    'loaded_additional_upper': forecast['loaded']['estimate']['domains'][0]['additional_generation_peak']['upper_bytes'],
                    'native_binding_count': forecast['native_binding_count'],
                    'workspace_resources': forecast['workspace_resources'],
                    'projection_storage': topology['projection_storage'],
                    'projection_input_normalizations': topology['projection_input_normalizations'],
                    'f32_rms_normalization_gains': topology['f32_rms_normalization_gains'],
                    'unsupported_2001_preserves_prefill_bound': forecast['unsupported_2001_preserves_prefill_bound'],
                    'forecast_allocates_native_storage': forecast['forecast_allocates_native_storage'],
                    'ordinary_continuation_upper': peak(forecast['continuation']),
                    'controlled_continuation_upper': peak(forecast['controlled']),
                    'speculative_upper': peak(forecast['speculative']),
                    'speculative_continuation_upper': peak(forecast['speculative_continuation']),
                }
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
