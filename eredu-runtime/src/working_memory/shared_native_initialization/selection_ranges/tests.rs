use super::*;
use eredu_checkpoint::{recipe::EncodedRecipeMappingPlan, store::TensorSelection};

fn required<P: SharedNativeInitializer>(plan: &P) -> Option<u64> {
    let result = WorkingMemoryPool::shared_native_initialization_required_bytes(plan);
    if std::env::var_os("EREDU_REQUIRE_SHARED_INPUT_INITIALIZATION_QUALIFICATION").is_some() {
        assert!(result.is_ok(), "{result:?}");
    }
    match result {
        Ok(bytes) => Some(bytes),
        Err(WorkingMemoryError::UnknownBound) => None,
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn selected_ranges_and_mapping_share_a_pool_and_retire_inputs_independently() {
    for refusal in [None, Some("ranges"), Some("mapping")] {
        let key = String::from("雪");
        let shape = [2, 6];
        let selection = TensorSelection::Indices {
            axis: 1,
            indices: vec![5, 0, 5],
        };
        let mut initial = [0; 2];
        let mut replacement = [];
        // Quote against the actual immutable geometry, before the admitted run.
        let plan = SelectionReadDestinationPlan::for_encoded(
            &key,
            8,
            &shape,
            12,
            &selection,
            &mut initial,
            &mut replacement,
        )
        .unwrap();
        let Some(range_bytes) = required(&plan) else {
            return;
        };
        let ranges = plan.build().unwrap();
        let source_plan = EncodedRecipeMappingPlan::source(20..32).unwrap();
        let source_bytes = required(&source_plan).unwrap();
        let source = source_plan.build().unwrap();
        let mapping_bytes =
            required(&EncodedRecipeMappingPlan::selected(&source, ranges.ranges()).unwrap())
                .unwrap();
        drop((source, ranges));
        let total = source_bytes + range_bytes + mapping_bytes;
        let limit = match refusal {
            Some("ranges") => source_bytes + range_bytes - 1,
            Some("mapping") => total - 1,
            None => total,
            _ => unreachable!(),
        };
        let pool = WorkingMemoryPool::new(limit, 0).unwrap();
        let source = pool
            .initialize_shared_native(EncodedRecipeMappingPlan::source(20..32).unwrap())
            .unwrap();
        let plan = SelectionReadDestinationPlan::for_encoded(
            &key,
            8,
            &shape,
            12,
            &selection,
            &mut initial,
            &mut replacement,
        )
        .unwrap();
        let result = pool.initialize_shared_native(plan);
        if refusal == Some("ranges") {
            let failure = result.unwrap_err();
            assert!(matches!(
                failure.accounting_failure(),
                Some(WorkingMemoryError::BudgetExceeded { .. })
            ));
            assert!(failure.rejected_plan().is_some());
            assert!(failure.constructor_failure().is_none());
            assert_eq!(pool.used_bytes().unwrap(), source_bytes);
            drop(failure);
            drop(source);
        } else {
            let ranges = result.unwrap();
            drop((key, selection));
            assert_eq!(
                ranges.output().ranges(),
                [5..6, 0..1, 5..6, 11..12, 6..7, 11..12]
            );
            assert!(ranges.output().physically_bounded());
            assert_eq!(pool.used_bytes().unwrap(), source_bytes + range_bytes);
            let result = pool.initialize_shared_native(
                EncodedRecipeMappingPlan::selected(source.output(), ranges.output().ranges())
                    .unwrap(),
            );
            if refusal == Some("mapping") {
                let failure = result.unwrap_err();
                assert!(matches!(
                    failure.accounting_failure(),
                    Some(WorkingMemoryError::BudgetExceeded { .. })
                ));
                assert!(failure.rejected_plan().is_some());
                assert_eq!(pool.used_bytes().unwrap(), source_bytes + range_bytes);
                drop(failure);
                drop((source, ranges));
            } else {
                let mapping = result.unwrap();
                assert_eq!(pool.used_bytes().unwrap(), total);
                drop((source, ranges));
                assert_eq!(pool.used_bytes().unwrap(), mapping_bytes);
                assert_eq!(mapping.output().byte_len(), 6);
                assert_eq!(
                    mapping.output().ranges().collect::<Vec<_>>(),
                    [
                        (25..26, 0..1),
                        (20..21, 1..2),
                        (25..26, 2..3),
                        (31..32, 3..4),
                        (26..27, 4..5),
                        (31..32, 5..6)
                    ]
                );
                drop(mapping);
            }
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn packed_ranges_outlive_selection_and_shape_scratch() {
    let mut initial = [0; 2];
    let mut replacement = [];
    let selection = TensorSelection::Indices {
        axis: 0,
        indices: vec![2, 0, 2],
    };
    let plan = SelectionReadDestinationPlan::for_encoded(
        "packed",
        4,
        &[3, 4],
        6,
        &selection,
        &mut initial,
        &mut replacement,
    )
    .unwrap();
    let Some(bytes) = required(&plan) else { return };
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let ranges = pool.initialize_shared_native(plan).unwrap();
    drop(selection);
    initial.fill(99);
    assert_eq!(ranges.output().ranges(), [4..6, 0..2, 4..6]);
    assert!(ranges.output().physically_bounded());
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(ranges);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
