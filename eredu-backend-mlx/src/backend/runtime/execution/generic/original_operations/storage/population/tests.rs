use super::*;
fn geometry(input: u64, chunk: u64, outputs: u64) -> eredu_core::InferenceGeometry {
    eredu_core::InferenceGeometry {
        batch_size: 1,
        input_positions: input,
        cached_positions: 0,
        max_output_tokens: outputs,
        prefill_chunk_positions: chunk,
        output: eredu_core::OutputDemand::LastPosition,
    }
}
fn row(
    requested: usize,
    units: usize,
    bindings: usize,
    sources: usize,
    submissions: usize,
) -> WindowPopulation {
    WindowPopulation {
        requested,
        units,
        physical_bindings: bindings,
        bindings,
        recipe_pending: sources,
        recipe_materializations: submissions,
        ..WindowPopulation::default()
    }
}

#[test]
fn selected_retry_population_covers_independent_attempt_trace_for_all_actual_forwards() {
    let windows = [row(2, 3, 5, 9, 4), row(1, 2, 2, 2, 0)];
    let actual = ResidencyPopulation::from_windows(7, &windows, geometry(5, 2, 3)).unwrap();
    let mut transfers = 0;
    let mut observations = 0;
    let mut pending = 0;
    let mut weight = 0;
    let mut forwards = 0;
    let mut position = 0;
    while position < 5 {
        forwards += 1;
        position = (position + 2).min(5);
    }
    for _ in 1..3 {
        forwards += 1;
    }
    for _ in 0..forwards {
        for source in windows {
            transfers += 1; // Unconditional warm-owner reservation.
            transfers += 1; // One shared missing canonical closure batch.
            observations += 1;
            for _ in 0..2 {
                // Both whole-unit attempts.
                for _ in 0..2 {
                    // Both per-binding attempts, internal recipe retries already counted.
                    for _ in 0..source.recipe_pending {
                        pending += 1;
                    }
                    for _ in 0..source.recipe_materializations {
                        weight += 1;
                    }
                }
                for _ in 0..source.physical_bindings {
                    weight += 1;
                }
            }
            for _ in 0..source.units {
                weight += 1;
            }
        }
    }
    assert_eq!(
        (
            actual.transfers,
            actual.observations,
            actual.pending,
            actual.weight
        ),
        (transfers, observations, pending, weight)
    );
    assert_eq!((actual.host, actual.materialization), (0, 0));
}

#[test]
fn uniform_payload_uses_maximum_owner_shape_while_request_banks_sum_real_windows() {
    let mut a = row(2, 4, 7, 11, 3);
    a.aliases = 5;
    a.bindings = 12;
    a.unit_id_bytes = 19;
    let b = row(3, 3, 2, 4, 0);
    let one = ResidencyPopulation::from_windows(9, &[a, b], geometry(1, 1, 1)).unwrap();
    let many = ResidencyPopulation::from_windows(9, &[a, b], geometry(9, 3, 4)).unwrap();
    assert_eq!(one.unprepared.transfer, many.unprepared.transfer);
    assert_eq!(many.transfers, one.transfers * 6);
    let shape = one.unprepared.transfer;
    assert_eq!(
        (
            shape.leases,
            shape.unit_ids,
            shape.pending_sources,
            shape.output_arrays
        ),
        (3, 4, 44, 12)
    );
    assert_eq!(
        (
            shape.retained_host,
            shape.retained_events,
            shape.alias_assignments
        ),
        (8, 14, 5)
    );
    assert_eq!(one.controller_units, 9); // Exact scratch capacity, independent of reached count.
}

#[test]
fn source_count_and_request_product_overflow_refuse_before_slot_factories() {
    let source = row(1, 1, 1, usize::MAX, 0);
    assert!(ResidencyPopulation::from_windows(1, &[source], geometry(1, 1, 1)).is_err());
    let source = row(1, 1, 1, 1, 0);
    assert!(ResidencyPopulation::from_windows(1, &[source], geometry(1, 1, u64::MAX)).is_err());
    let empty = ResidencyPopulation::from_windows(0, &[], geometry(1, 1, 1)).unwrap();
    assert_eq!((empty.transfers, empty.pending, empty.weight), (0, 0, 0));
}
