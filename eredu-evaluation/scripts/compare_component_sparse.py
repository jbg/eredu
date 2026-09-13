#!/usr/bin/env python3
"""Compare public sparse-probe evidence with the independent reference.

Report-only mode diagnoses numerical and route differences; it is not acceptance.
Route slots remain in both artifacts. Unique top-k experts are joined by expert
identity for cross-implementation numerical comparison, never by packed position.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('native', type=Path)
    parser.add_argument('--atol', type=float, default=0.001)
    parser.add_argument('--rtol', type=float, default=0.001)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--require-controlled', action='store_true')
    parser.add_argument('--require-write-reconstruction', action='store_true')
    parser.add_argument('--require-score-reconstruction', action='store_true')
    parser.add_argument('--bf16-logit-ulps', type=int, default=0,
                        help='Allow this many BF16 output rounding steps only with exact measured head inputs')
    args = parser.parse_args()
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error('tolerances must be finite and nonnegative')
    if args.bf16_logit_ulps < 0:
        parser.error('BF16 rounding steps must be nonnegative')
    reference, native = [json.loads(path.read_text()) for path in (args.reference, args.native)]
    assert reference['provenance'] == native['provenance']
    metrics, failures = {}, []

    def compare(label, actual, expected, atol=None, rtol=None):
        actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            failures.append({'label': label, 'geometry_or_nonfinite': True})
            return
        error = np.abs(actual - expected)
        outside = error > (args.atol if atol is None else atol) + (args.rtol if rtol is None else rtol) * np.abs(expected)
        bf16 = {}
        if label.endswith('/logits') and args.bf16_logit_ulps:
            bits = [np.asarray(v, dtype=np.float32).view(np.uint32) for v in (actual, expected)]
            representable = ((bits[0] | bits[1]) & 0xffff) == 0
            def ordered(value):
                value = value >> 16
                return np.where(value & 0x8000, 0x8000 - (value & 0x7fff), 0x8000 + value).astype(np.int64)
            ulps = np.abs(ordered(bits[0]) - ordered(bits[1]))
            rounded = representable & (ulps <= args.bf16_logit_ulps)
            bf16 = {'bf16_max_ulps': int(ulps.max(initial=0)) if representable.all() else None,
                    'accepted_by_bf16_rounding': int((outside & rounded).sum())}
            outside &= ~rounded
        bad = int(outside.sum())
        metrics[label] = {'count': actual.size, 'max_abs': float(error.max(initial=0)),
                          'rms': float(np.sqrt(np.mean(error ** 2))), 'outside_tolerance': bad,
                          'reference_max_abs': float(np.abs(expected).max(initial=0)), **bf16}
        if bad:
            failures.append({'label': label, 'outside_tolerance': bad})

    if native['trials'].keys() != reference['trials'].keys():
        failures.append({'missing_trials': sorted(reference['trials'].keys() - native['trials'].keys()),
                         'extra_trials': sorted(native['trials'].keys() - reference['trials'].keys())})
    for trial, expected_steps in reference['trials'].items():
        if args.require_controlled:
            controlled = native.get('controlled', {}).get(trial, {})
            checks = ['ordinary_equal', 'cached_restore_equal', 'sibling_equal', 'parent_after_sibling_equal']
            if any(controlled.get(check) is not True for check in checks) or controlled.get('predictions_checked') != 14:
                failures.append({'trial': trial, 'missing_controlled_evidence': True})
        actual_steps = native['trials'].get(trial, [])
        if len(actual_steps) != len(expected_steps):
            failures.append({'trial': trial, 'prediction_count': [len(actual_steps), len(expected_steps)]})
        same_context = True
        for actual, expected in zip(actual_steps, expected_steps):
            prefix = f"{trial}/{expected['prediction']}"
            assert actual['prediction'] == expected['prediction']
            if not same_context:
                failures.append({'label': prefix, 'different_cached_prefix': True})
                continue
            if args.bf16_logit_ulps:
                if 'head_input' not in expected or 'head_input' not in actual:
                    failures.append({'label': prefix, 'missing_head_input_for_rounding_check': True})
                elif actual['head_input'] != expected['head_input']:
                    failures.append({'label': prefix, 'different_head_input': True})
            if 'head_input' in expected and 'head_input' in actual:
                compare(prefix + '/head_input', actual['head_input'], expected['head_input'])
            for auxiliary in ('convolution_input', 'routed_write'):
                if auxiliary in expected:
                    if auxiliary not in actual:
                        failures.append({'label': prefix, 'missing_auxiliary': auxiliary})
                    else:
                        compare(prefix + '/' + auxiliary, actual[auxiliary], expected[auxiliary])
            for key, values in expected.get('writes', {}).items():
                if key not in actual.get('writes', {}):
                    failures.append({'label': prefix + '/writes/' + key, 'missing_write': True})
                else:
                    compare(prefix + '/writes/' + key, actual['writes'][key], values)
            for category in ('boundaries', 'readout'):
                for key, values in expected.get(category, {}).items():
                    if key not in actual.get(category, {}):
                        failures.append({'label': prefix + '/' + category + '/' + key, 'missing_boundary': True})
                    else:
                        compare(prefix + '/' + category + '/' + key, actual[category][key], values)
            compare(prefix + '/logits', actual['logits'], expected['logits'])
            if actual['token'] != expected['token']:
                failures.append({'label': prefix, 'winner': [actual['token'], expected['token']]})
                same_context = False
            for key, expected_values in expected['captures'].items():
                actual_values = actual['captures'].get(key)
                if actual_values is None:
                    failures.append({'label': prefix + '/' + key, 'missing_capture': True})
                    continue
                if not key.startswith('routed:'):
                    for mode in ('original', 'effective'):
                        compare(f'{prefix}/{key}/{mode}', actual_values[mode], expected_values[mode])
                    continue
                default_token = len(reference['prefix_ids']) - 1 if expected['prediction'] == 0 else 0
                row_identity = lambda row: (row.get('token', default_token), row['expert'])
                expected_rows = {row_identity(row): row for row in expected_values}
                assert len(expected_rows) == len(expected_values), 'fixture requires unique experts per token'
                for mode in ('original', 'effective'):
                    actual_rows = {row_identity(row): row for row in actual_values[mode]}
                    assert len(actual_rows) == len(actual_values[mode])
                    if actual_rows.keys() != expected_rows.keys():
                        failures.append({'label': f'{prefix}/{key}/{mode}',
                                         'experts': [sorted(actual_rows), sorted(expected_rows)]})
                    for expert in actual_rows.keys() & expected_rows.keys():
                        left, right = actual_rows[expert], expected_rows[expert]
                        compare(f'{prefix}/{key}/{mode}/{expert}', left['values'], right[mode])
                        compare(f'{prefix}/{key}/{mode}/{expert}/coefficient', [left['coefficient']], [right['coefficient']])
    reconstruction = {'groups': 0, 'components': 0, 'exact_physical_writes': 0,
                      'max_write_abs': 0., 'max_projection_associativity_error': 0.,
                      'signed_projection_atol': 1e-6, 'signed_projection_rtol': 1e-6}
    if args.require_write_reconstruction or 'write_reconstruction' in reference:
        for trial, steps in reference['trials'].items():
            expected_groups = reference.get('write_reconstruction', {}).get(trial, {}).get('groups', {})
            actual_groups = native.get('write_reconstruction', {}).get(trial, {}).get('groups', {})
            declared = {f"{item['kind']}:{item['layer']}" for item in reference['captured']}
            if set(expected_groups) != declared or set(actual_groups) != declared:
                failures.append({'trial': trial, 'missing_write_reconstruction': True})
            for key in expected_groups.keys() & actual_groups.keys():
                expected, actual = expected_groups[key], actual_groups[key]
                label = f'{trial}/write_reconstruction/{key}'
                if any(actual.get(field) != expected.get(field) for field in ('prediction', 'token', 'components')):
                    failures.append({'label': label, 'reconstruction_identity_or_geometry': True})
                for field in ('signed_component_sum', 'signed_projected_write', 'signed_observed_write'):
                    compare(label + '/' + field, [actual.get(field)], [expected.get(field)], 1e-6, 1e-6)
                error, associativity = actual.get('write_max_abs'), actual.get('projection_associativity_error')
                if error is None or associativity is None or not np.isfinite([error, associativity]).all() or min(error, associativity) < 0:
                    failures.append({'label': label, 'invalid_reconstruction_error': True})
                    continue
                observed = steps[0]['writes'][key]
                width = len(observed) // (len(reference['prefix_ids']) if reference.get('capture_all_positions') else 1)
                limit = args.atol + args.rtol * max(abs(value) for value in observed[-width:])
                if error > limit:
                    failures.append({'label': label, 'physical_write_error': error, 'limit': limit})
                if associativity > 1e-6 + 1e-6 * abs(expected['signed_component_sum']):
                    failures.append({'label': label, 'projection_associativity_error': associativity})
                reconstruction['groups'] += 1
                reconstruction['components'] += actual['components']
                reconstruction['exact_physical_writes'] += bool(actual.get('physical_write_exact'))
                reconstruction['max_write_abs'] = max(reconstruction['max_write_abs'], error)
                reconstruction['max_projection_associativity_error'] = max(reconstruction['max_projection_associativity_error'], associativity)
    score_summary = {'predictions': 0, 'scores': 0, 'component_groups': 0, 'components': 0,
                     'max_absolute_score_error': 0., 'max_output_rounding_fraction': 0.,
                     'max_propagated_projection_bound': 0.}
    if args.require_score_reconstruction or 'score_reconstruction' in reference:
        declared = {f"{item['kind']}:{item['layer']}" for item in reference['captured']}
        def output_step(value):
            return math.ldexp(1., max(-126, math.frexp(abs(value))[1] - 1) - 7) if value else math.ldexp(1., -133)
        for trial, expected_steps in reference['trials'].items():
            expected_reports = reference.get('score_reconstruction', {}).get(trial, [])
            actual_reports = native.get('score_reconstruction', {}).get(trial, [])
            if len(expected_reports) != len(expected_steps) or len(actual_reports) != len(expected_steps):
                failures.append({'trial': trial, 'missing_score_reconstruction': True})
            for prediction, (expected_report, actual_report) in enumerate(zip(expected_reports, actual_reports)):
                prefix = f'{trial}/{prediction}/score_reconstruction'
                expected_groups, actual_groups = expected_report.get('groups', {}), actual_report.get('groups', {})
                if set(expected_groups) != declared or set(actual_groups) != declared:
                    failures.append({'label': prefix, 'missing_score_components': True})
                for key in expected_groups.keys() & actual_groups.keys():
                    a, e = actual_groups[key], expected_groups[key]
                    if any(a.get(field) != e.get(field) for field in ('prediction', 'token', 'components')):
                        failures.append({'label': prefix + '/' + key, 'score_component_geometry': True})
                    ad, ed = a.get('score_directions', []), e.get('score_directions', [])
                    if len(ad) != 2 or len(ed) != 2:
                        failures.append({'label': prefix + '/' + key, 'missing_score_directions': True})
                    for mode, (left, right) in enumerate(zip(ad, ed)):
                        for field in ('signed_component_sum', 'signed_projected_write', 'signed_observed_write'):
                            compare(f'{prefix}/{key}/{mode}/{field}', [left.get(field)], [right.get(field)], 1e-6, 1e-6)
                    observed = expected_steps[prediction]['writes'][key]
                    width = len(observed) // (len(reference['prefix_ids']) if prediction == 0 and reference.get('capture_all_positions') else 1)
                    error = a.get('write_max_abs', float('nan'))
                    limit = args.atol + args.rtol * max(abs(v) for v in observed[-width:])
                    if not np.isfinite(error) or error < 0 or error > limit:
                        failures.append({'label': prefix + '/' + key, 'physical_write_error': error, 'limit': limit})
                    score_summary['component_groups'] += 1
                    score_summary['components'] += a.get('components', 0)
                actual = actual_report.get('score_reconstruction', {})
                expected = expected_report.get('score_reconstruction', {})
                if (actual.get('residual_replay_exact') is not True or expected.get('residual_replay_exact') is not True
                        or any(actual.get(field) != expected.get(field) for field in ('prediction', 'residual_writes', 'other_write_count'))):
                    failures.append({'label': prefix, 'residual_accounting_failed': True})
                actual_scores, expected_scores = actual.get('scores', []), expected.get('scores', [])
                if len(actual_scores) != 2 or len(expected_scores) != 2:
                    failures.append({'label': prefix, 'missing_target_or_margin': True})
                for mode, (a, e) in enumerate(zip(actual_scores, expected_scores)):
                    label = f'{prefix}/{mode}'
                    if any(a.get(field) != e.get(field) for field in ('mode', 'target', 'alternative')):
                        failures.append({'label': label, 'score_identity': True})
                        continue
                    fields = ('components', 'operator_rounding', 'embedding', 'other_writes', 'residual_rounding',
                              'direction_rounding', 'normalization_rounding', 'normalization_offset')
                    # Aggregate components inherit the sum of their projection
                    # bounds. A small rounding remainder is a difference of two
                    # large sums, so its tolerance must not scale with that
                    # remainder's magnitude alone.
                    component_bound = math.fsum(1e-6 + 1e-6 * abs(group['score_directions'][mode]['signed_component_sum'])
                                                for group in expected_groups.values())
                    operator_bound = component_bound + math.fsum(1e-6 + 1e-6 * abs(group['score_directions'][mode]['signed_observed_write'])
                                                                 for group in expected_groups.values())
                    for field in fields:
                        if field in ('components', 'operator_rounding'):
                            compare(label + '/' + field, [a.get(field)], [e.get(field)],
                                    component_bound if field == 'components' else operator_bound, 0.)
                        else:
                            compare(label + '/' + field, [a.get(field)], [e.get(field)], 1e-6, 1e-6)
                    score_summary['max_propagated_projection_bound'] = max(score_summary['max_propagated_projection_bound'], operator_bound)
                    for field in ('reconstructed', 'affine_score'):
                        compare(label + '/' + field, [a.get(field)], [e.get(field)], 1e-9, 1e-9)
                    numbers = [a.get(field) for field in (*fields, 'reconstructed', 'affine_score', 'score', 'absolute_error', 'head_rounding_bound')]
                    if any(v is None for v in numbers) or not np.isfinite(numbers).all():
                        failures.append({'label': label, 'nonfinite_score_accounting': True})
                        continue
                    logits = native['trials'][trial][prediction]['logits']
                    target, alternative = logits[a['target']], logits[a['alternative']]
                    score = target - (alternative if mode else 0.)
                    bound = output_step(target) + (output_step(alternative) if mode else 0.)
                    error = abs(score - a['reconstructed'])
                    if (a['score'] != score or a['head_rounding_bound'] != bound or a['absolute_error'] != error
                            or abs(math.fsum(a[field] for field in fields) - a['reconstructed']) > 1e-9 * (1 + abs(a['affine_score']))
                            or abs(a['reconstructed'] - a['affine_score']) > 1e-9 * (1 + abs(a['affine_score']))
                            or error > bound + 1e-9):
                        failures.append({'label': label, 'score_accounting_or_rounding_failed': True})
                    score_summary['scores'] += 1
                    score_summary['max_absolute_score_error'] = max(score_summary['max_absolute_score_error'], error)
                    score_summary['max_output_rounding_fraction'] = max(score_summary['max_output_rounding_fraction'], error / bound)
                score_summary['predictions'] += 1
    report = {'accepted': not failures and not args.report_only, 'report_only': args.report_only,
              'controlled_required': args.require_controlled,
              'write_reconstruction_required': args.require_write_reconstruction,
              'write_reconstruction': reconstruction,
              'score_reconstruction_required': args.require_score_reconstruction,
              'score_reconstruction': score_summary,
              'bf16_logit_ulps': args.bf16_logit_ulps,
              'atol': args.atol, 'rtol': args.rtol,
              'compared_values': sum(m['count'] for m in metrics.values()),
              'failures': failures, 'metrics': metrics}
    print(json.dumps(report, indent=2))
    if failures and not args.report_only:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
