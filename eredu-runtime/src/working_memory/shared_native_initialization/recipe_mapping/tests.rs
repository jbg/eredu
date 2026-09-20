use super::*;

fn required(plan: &EncodedRecipeMappingPlan<'_>) -> Option<u64> {
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
fn exact_mapping_chain_retains_only_live_outputs_after_children_retire() {
    let source_plan = || EncodedRecipeMappingPlan::source(20..28).unwrap();
    let Some(source_bytes) = required(&source_plan()) else {
        return;
    };
    let source = source_plan().build().unwrap();
    let selection = [4..8, 0..2, 0..2];
    let selection_plan = EncodedRecipeMappingPlan::selected(&source, &selection).unwrap();
    let selected_bytes = required(&selection_plan).unwrap();
    drop(source);
    let pool = WorkingMemoryPool::new(source_bytes + selected_bytes, 0).unwrap();
    let source = pool.initialize_shared_native(source_plan()).unwrap();
    let selected = pool
        .initialize_shared_native(
            EncodedRecipeMappingPlan::selected(source.output(), &selection).unwrap(),
        )
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), source_bytes + selected_bytes);
    drop(source);
    assert_eq!(pool.used_bytes().unwrap(), selected_bytes);
    assert_eq!(selected.output().byte_len(), 8);
    assert_eq!(
        selected.output().ranges().collect::<Vec<_>>(),
        [(24..28, 0..4), (20..22, 4..6), (20..22, 6..8)]
    );
    selected.validate_pool(&pool).unwrap();
    let foreign = WorkingMemoryPool::new(source_bytes + selected_bytes, 0).unwrap();
    assert!(matches!(
        selected.validate_pool(&foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    drop(selected);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn interleaving_admits_one_output_and_preserves_children_on_short_budget() {
    for short in [false, true] {
        let plans = [
            EncodedRecipeMappingPlan::source(0..6).unwrap(),
            EncodedRecipeMappingPlan::source(100..104).unwrap(),
        ];
        let Some(first_bytes) = required(&plans[0]) else {
            return;
        };
        let second_bytes = required(&plans[1]).unwrap();
        let [first, second] = plans.map(|plan| plan.build().unwrap());
        let children = [&first, &second];
        let output_bytes =
            required(&EncodedRecipeMappingPlan::interleaved(&children, &[3, 2], 2).unwrap())
                .unwrap();
        drop((first, second));
        let total = first_bytes + second_bytes + output_bytes;
        let pool = WorkingMemoryPool::new(total - u64::from(short), 0).unwrap();
        let first = pool
            .initialize_shared_native(EncodedRecipeMappingPlan::source(0..6).unwrap())
            .unwrap();
        let second = pool
            .initialize_shared_native(EncodedRecipeMappingPlan::source(100..104).unwrap())
            .unwrap();
        let children = [first.output(), second.output()];
        let result = pool.initialize_shared_native(
            EncodedRecipeMappingPlan::interleaved(&children, &[3, 2], 2).unwrap(),
        );
        if short {
            let failure = result.unwrap_err();
            assert!(matches!(
                failure.accounting_failure(),
                Some(WorkingMemoryError::BudgetExceeded { .. })
            ));
            assert!(failure.rejected_plan().is_some());
            assert!(failure.constructor_failure().is_none());
            assert_eq!(pool.used_bytes().unwrap(), first_bytes + second_bytes);
            assert_eq!(first.output().ranges().collect::<Vec<_>>(), [(0..6, 0..6)]);
            assert_eq!(
                second.output().ranges().collect::<Vec<_>>(),
                [(100..104, 0..4)]
            );
            drop(failure);
            drop((first, second));
        } else {
            let mapped = result.unwrap();
            assert_eq!(pool.used_bytes().unwrap(), total);
            drop((first, second));
            assert_eq!(pool.used_bytes().unwrap(), output_bytes);
            assert_eq!(mapped.output().byte_len(), 10);
            assert_eq!(
                mapped.output().ranges().collect::<Vec<_>>(),
                [
                    (0..3, 0..3),
                    (100..102, 3..5),
                    (3..6, 5..8),
                    (102..104, 8..10)
                ]
            );
            drop(mapped);
        }
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn empty_mapping_has_no_ranges_and_still_accounts_for_constructor_controls() {
    let plan = EncodedRecipeMappingPlan::source(7..7).unwrap();
    let Some(bytes) = required(&plan) else { return };
    assert!(bytes > 0);
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let output = pool.initialize_shared_native(plan).unwrap();
    assert_eq!(output.output().ranges().len(), 0);
    assert_eq!(output.output().byte_len(), 0);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(output);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
