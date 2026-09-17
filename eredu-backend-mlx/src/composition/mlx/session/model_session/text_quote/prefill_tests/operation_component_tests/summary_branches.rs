//! Shorter actual Summary branches retain the full cold numerical allowance.
use super::*;
use crate::backend::array_copy::{CaptureCompletion, SummaryProgram};
use eredu_core::capture::CaptureSummary;

fn oracle(values: &[f32]) -> CaptureSummary {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(f64::from)
        .collect();
    CaptureSummary {
        elements: values.len() as u64,
        finite: finite.len() as u64,
        non_finite: (values.len() - finite.len()) as u64,
        nan: values.iter().filter(|v| v.is_nan()).count() as u64,
        positive_infinity: values.iter().filter(|v| **v == f32::INFINITY).count() as u64,
        negative_infinity: values.iter().filter(|v| **v == f32::NEG_INFINITY).count() as u64,
        min: finite.iter().copied().reduce(f64::min),
        max: finite.iter().copied().reduce(f64::max),
        mean: (!finite.is_empty()).then(|| finite.iter().sum::<f64>() / finite.len() as f64),
        rms: (!finite.is_empty())
            .then(|| (finite.iter().map(|v| v * v).sum::<f64>() / finite.len() as f64).sqrt()),
    }
}
fn compare(actual: &CaptureSummary, expected: &CaptureSummary) {
    assert_eq!(
        (
            actual.elements,
            actual.finite,
            actual.non_finite,
            actual.nan,
            actual.positive_infinity,
            actual.negative_infinity
        ),
        (
            expected.elements,
            expected.finite,
            expected.non_finite,
            expected.nan,
            expected.positive_infinity,
            expected.negative_infinity
        )
    );
    for (a, b) in [
        (actual.min, expected.min),
        (actual.max, expected.max),
        (actual.mean, expected.mean),
        (actual.rms, expected.rms),
    ] {
        match (a, b) {
            (Some(a), Some(b)) => assert!((a - b).abs() <= 2e-5 + 2e-5 * b.abs(), "{a} versus {b}"),
            (None, None) => (),
            _ => panic!("finite-only aggregates differ"),
        }
    }
}
#[test]
fn original_summary_zero_nonfinite_and_mixed_branches_retire_under_full_recipe() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    let stream = prepared.stream.clone();
    for case in 0..3 {
        let length = if case == 2 { 3073 } else { 2051 };
        let values: Vec<f32> = (0..length)
            .map(|i| match case {
                0 => 0.0,
                1 => [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][i % 3],
                _ if i < 1024 => 0.0,
                _ if i < 2048 => [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][i % 3],
                _ => [1.5, -2.25, 0.0][i % 3],
            })
            .collect();
        let expected = oracle(&values);
        let input = Array::from_slice(&values, &[length as i32]);
        input.evaluated().unwrap();
        let program = SummaryProgram::new(length as i32).unwrap();
        let population = program.population().unwrap();
        let mut ordinary_roots = Vec::new();
        let ordinary = program
            .execute(
                &input,
                &stream,
                CaptureCompletion::Ordinary,
                &mut |value| {
                    ordinary_roots.push(value.clone());
                    Ok(())
                },
                &mut |_| {},
            )
            .unwrap();
        compare(&ordinary, &expected);
        let actual_root_count = ordinary_roots.len();
        assert!(
            actual_root_count < population.retained_outputs,
            "actual branch omits the cold suffix"
        );
        drop(ordinary_roots);
        let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
        let metadata = WorkspaceTensor::existing(
            context
                .layout(&[length as i32], WorkspaceDtype::Float32)
                .unwrap(),
            &context,
        )
        .unwrap();
        context.begin_span();
        let mut retained = Vec::new();
        program.trace(&metadata, &context, &mut retained).unwrap();
        assert_eq!(retained.len(), population.retained_outputs);
        let plan = OriginalComponentTestPlan::from_summary_report(
            context.report(&[metadata]).unwrap(),
            program,
        );
        assert_eq!(
            plan.completion.nested_completions,
            population.scalar_completions
        );
        let quote = plan.quote();
        POINTWISE_CONTROLS.with(|slot| assert!(slot.replace(Some(plan.control_bytes())).is_none()));
        let controls_reset = PointwiseControlsReset;
        let baseline = prepared.pool.used_bytes().unwrap();
        let unquoted = prepared.pool.unquoted_owner_count().unwrap();
        let actual = with_prepared_original_operation_controls(
            None,
            &mut prepared,
            |_, observer, _, destination| {
                assert!(destination.is_none());
                OperationEvent::validate_traversal_leaf(&input, observer).unwrap();
                let completion = CaptureCompletion::Original(observer);
                let mut roots = Vec::with_capacity(population.retained_outputs);
                let mut graph =
                    OperationEvent::prepare_resident_graph(plan.completion.graph, observer)
                        .unwrap();
                graph
                    .configure_nested_completions(
                        &plan.completion.nested_traversal().unwrap(),
                        plan.completion.nested_completions,
                    )
                    .unwrap();
                let mut reads = 0;
                let result = program
                    .execute(
                        &input,
                        &stream,
                        completion,
                        &mut |value| {
                            assert!(roots.len() < roots.capacity());
                            roots.push(completion.clone_array(value)?);
                            Ok(())
                        },
                        &mut |n| {
                            assert_eq!(n, 1);
                            reads += 1;
                        },
                    )
                    .unwrap();
                assert_eq!(roots.len(), actual_root_count);
                assert!(
                    reads < population.scalar_completions,
                    "actual branch leaves unused accepted frontiers"
                );
                // Prove every actual retained root is complete before retiring any.
                completion.validate_identity().unwrap();
                for root in &roots {
                    observer.validate_completed_array(root).unwrap();
                }
                safemlx::try_with_submission_retirement(|| drop(roots)).unwrap();
                drop(graph);
                assert!(!observer.status().failed());
                result
            },
        );
        compare(&actual, &expected);
        assert!(quote.was_quoted());
        drop((quote, controls_reset, input));
        crate::backend::submission_recovery::wait_for_retirement(|| {
            disk::reclaim();
            prepared.pool.used_bytes().unwrap() == baseline
                && prepared.pool.unquoted_owner_count().unwrap() == unquoted
        });
    }
    drop(stream);
    prepared.finish();
}
