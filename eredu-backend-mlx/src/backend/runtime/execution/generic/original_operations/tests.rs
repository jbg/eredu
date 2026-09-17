use super::*;

fn geometry(input: u64, chunk: u64, outputs: u64) -> eredu_core::InferenceGeometry {
    eredu_core::InferenceGeometry {
        batch_size: 1,
        input_positions: input,
        cached_positions: 0,
        max_output_tokens: outputs,
        prefill_chunk_positions: chunk.min(input),
        output: eredu_core::OutputDemand::LastPosition,
    }
}

#[test]
fn selected_unit_population_matches_each_actual_span_and_cached_forward() {
    for units in [1, 3, 9] {
        for input in [1, 2, 5, 17] {
            for chunk in [1, 2, 7, 32] {
                for outputs in [1, 2, 5] {
                    let g = geometry(input, chunk, outputs);
                    let mut visited = 0;
                    let mut start = 0;
                    while start < input {
                        for _ in 0..units {
                            visited += 1;
                        }
                        start = (start + chunk).min(input);
                    }
                    for _ in 1..outputs {
                        for _ in 0..units {
                            visited += 1;
                        }
                    }
                    let actual = operation_counts(units, g).unwrap();
                    assert_eq!(actual.units, visited);
                    let prefill = PrefillControlPlan::new(g, true).unwrap();
                    let prefill_roles = (0..prefill.scope_count())
                        .map(|n| prefill.role(n).unwrap())
                        .count();
                    let prediction_roles = usize::try_from(outputs).unwrap() * 5;
                    assert_eq!(actual.scopes, prefill_roles + prediction_roles);
                }
            }
        }
    }
}

#[test]
fn physical_unit_destinations_validate_allocation_extents_before_preparation() {
    let factory = storage::unit_factory::<()>(None);
    // No factory invocation, native scope, source or permission is needed to
    // inspect the actual concrete slot layout.
    let one = storage::unit_layout(1, &factory).unwrap();
    let three = storage::unit_layout(3, &factory).unwrap();
    let each = PreparedUnit::<()>::control_bytes().unwrap()
        + u64::try_from(size_of::<Option<PreparedUnit<()>>>()).unwrap();
    assert_eq!(three - one, 2 * each);
    assert!(storage::unit_layout(usize::MAX, &factory).is_none());
    assert!(MlxLayerwisePolicy::<()>::original_operation_baseline_control_bytes().unwrap() > 0);
}

#[test]
fn invalid_geometry_and_representable_scalar_product_have_distinct_refusals() {
    let invalid = operation_counts(3, geometry(5, 0, 4)).unwrap_err();
    assert!(matches!(
        invalid,
        Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
    ));
    let overflow = operation_counts(usize::MAX, geometry(5, 2, 4)).unwrap_err();
    assert!(matches!(
        overflow,
        Error::PrefillControl(WorkingMemoryError::Overflow)
    ));
    let short = operation_counts(3, geometry(5, 2, 1)).unwrap();
    assert_eq!(short.units, 9); // No fictitious cached decode when M=1.
}
