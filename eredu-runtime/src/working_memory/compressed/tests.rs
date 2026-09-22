use super::*;
use eredu_nn::workspace::*;
use std::{cell::Cell, rc::Rc};

#[derive(Debug)]
struct Facts {
    remaining: Rc<Cell<Option<usize>>>,
    missing_updates: bool,
    missing_copies: bool,
}
impl WorkspaceMechanisms for Facts {
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "test mechanism has no disjoint host allocations".into(),
        }))
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if let Some(left) = self.remaining.get() {
            if left == 0 {
                return Err(Error::backend("injected storage pricing failure"));
            }
            self.remaining.set(Some(left - 1));
        }
        if (self.missing_updates && matches!(op.kind, WorkspaceOperationKind::SliceUpdate { .. }))
            || (self.missing_copies && matches!(op.kind, WorkspaceOperationKind::DeepCopy))
        {
            return Ok(None);
        }
        let alias = matches!(
            op.kind,
            WorkspaceOperationKind::Index { .. } | WorkspaceOperationKind::StaticSlice { .. }
        );
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|o| {
                    if alias {
                        Ok(WorkspaceOutputStorage::AliasInput(0))
                    } else {
                        o.bytes().map(WorkspaceOutputStorage::Allocate)
                    }
                })
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "test mechanism: logical output allocations and aliasing indices".into(),
        }))
    }
}
fn context(missing_updates: bool) -> (WorkspaceContext, Rc<Cell<Option<usize>>>) {
    let remaining = Rc::new(Cell::new(None));
    (
        WorkspaceContext::new(Facts {
            remaining: remaining.clone(),
            missing_updates,
            missing_copies: false,
        }),
        remaining,
    )
}
// Imported states are already resident inputs, not newly constructed parameters.
fn existing_f32(shape: &[i32], context: &WorkspaceContext) -> Result<WorkspaceTensor, Error> {
    WorkspaceTensor::existing(
        WorkspaceLayout::new(shape, WorkspaceDtype::Float32)?,
        context,
    )
}
fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}
fn cache(step: u32, context: &WorkspaceContext) -> WorkspaceCompressedCache {
    WorkspaceCompressedCache::new(nz(2), nz(8), nz(4), nz(step), context).unwrap()
}
fn append(
    cache: &mut WorkspaceCompressedCache,
    count: i32,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    let state = CompressedAttentionState {
        latent: WorkspaceTensor::full_f32(0.2, &[2, count, 8], context)?,
        rotary: WorkspaceTensor::full_f32(-0.3, &[2, count, 4], context)?,
    };
    let view = cache.append(state, context)?;
    assert!(matches!(view, CompressedAttentionView::Resident(_)));
    Ok(())
}
fn report(cache: &WorkspaceCompressedCache, context: &WorkspaceContext) -> WorkspaceTraceReport {
    context
        .report(&cache.retained_values().cloned().collect::<Vec<_>>())
        .unwrap()
}

#[test]
fn capacity_padding_growth_and_logical_views_preserve_the_full_storage_charge() {
    for (step, counts) in [(4, vec![2, 1, 4, 1, 1]), (256, vec![2, 253, 2, 7])] {
        let (context, _) = context(false);
        let mut cache = cache(step, &context);
        let mut position = 0;
        for count in counts {
            context.begin_span();
            let previous_capacity = cache.capacity();
            append(&mut cache, count, &context).unwrap();
            position += count;
            let expected_capacity = (position as u32).div_ceil(step) * step;
            assert_eq!(
                (cache.offset(), cache.capacity()),
                (position, expected_capacity as i32)
            );
            assert_eq!(
                cache.logical.as_ref().unwrap().latent.shape(),
                [2, position, 8]
            );
            assert_eq!(
                cache.storage.as_ref().unwrap().latent.shape(),
                [2, expected_capacity as i32, 8]
            );
            let report = report(&cache, &context);
            assert_eq!(
                report.retained_bytes,
                Some(2 * expected_capacity as u64 * (8 + 4) * 4)
            );
            let updates = report
                .operations
                .iter()
                .filter(|op| matches!(op.kind, WorkspaceOperationKind::SliceUpdate { .. }))
                .count();
            assert_eq!(updates, 2);
            if previous_capacity > 0 && cache.capacity() > previous_capacity {
                assert_eq!(
                    report
                        .operations
                        .iter()
                        .filter(|op| matches!(op.kind, WorkspaceOperationKind::Concatenate))
                        .count(),
                    2
                );
            }
        }
    }
}

#[test]
fn exact_first_capacity_aliases_inputs_without_padding_or_update_and_reset_keeps_step() {
    let (context, _) = context(false);
    let mut cache = cache(4, &context);
    append(&mut cache, 4, &context).unwrap();
    let report = report(&cache, &context);
    assert_eq!(report.operations.len(), 4); // two inputs and two logical views
    assert_eq!(report.retained_bytes, Some(2 * 4 * (8 + 4) * 4));
    cache.reset().unwrap();
    assert_eq!(
        (cache.offset(), cache.capacity(), cache.capacity_step()),
        (0, 0, 4)
    );
    assert!(cache.retained_values().next().is_none());
    assert_eq!(context.report(&[]).unwrap().total_bytes, report.total_bytes);
}

#[test]
fn restore_counts_distinct_backing_and_logical_copies_without_refunding_the_discarded_branch() {
    let (context, _) = context(false);
    let mut cache = cache(4, &context);
    append(&mut cache, 2, &context).unwrap();
    let checkpoint = cache.checkpoint();
    context.begin_span();
    append(&mut cache, 3, &context).unwrap();
    let branch = report(&cache, &context).total_bytes.unwrap();
    cache.restore(&checkpoint, &context).unwrap();
    let restored = report(&cache, &context);
    assert_eq!((cache.offset(), cache.capacity()), (2, 4));
    assert_eq!(restored.retained_bytes, Some(2 * (4 + 2) * (8 + 4) * 4));
    assert_eq!(
        restored.total_bytes,
        Some(branch + restored.retained_bytes.unwrap())
    );
    assert_eq!(
        restored
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::DeepCopy))
            .count(),
        4
    );
    context.begin_span();
    append(&mut cache, 1, &context).unwrap();
    assert_eq!(
        report(&cache, &context).retained_bytes,
        Some(2 * 4 * (8 + 4) * 4)
    );
}

#[test]
fn isolated_snapshot_compacts_storage_and_continuation_rounds_from_its_actual_capacity() {
    let (context, _) = context(false);
    let mut source = cache(256, &context);
    append(&mut source, 3, &context).unwrap();
    context.begin_span();
    let mut snapshot = source.isolated_snapshot(&context).unwrap();
    let compact = report(&snapshot, &context);
    assert_eq!((snapshot.offset(), snapshot.capacity()), (3, 3));
    assert_eq!(compact.retained_bytes, Some(2 * 3 * (8 + 4) * 4));
    assert_eq!(
        compact.total_bytes,
        Some(2 * compact.retained_bytes.unwrap())
    );
    assert_eq!(
        compact
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::Contiguous))
            .count(),
        2
    );
    context.begin_span();
    append(&mut snapshot, 1, &context).unwrap();
    let continued = report(&snapshot, &context);
    assert_eq!(snapshot.capacity(), 256);
    let padding = continued
        .operations
        .iter()
        .filter(|op| matches!(op.kind, WorkspaceOperationKind::Concatenate))
        .map(|op| op.inputs[1].shape()[1])
        .collect::<Vec<_>>();
    assert_eq!(padding, [253, 253]);
    assert_eq!((source.offset(), source.capacity()), (3, 256));
}

#[test]
fn failed_append_and_restore_preserve_prior_state_and_all_preceding_charges() {
    let (context, remaining) = context(false);
    let mut cache = cache(4, &context);
    append(&mut cache, 2, &context).unwrap();
    let checkpoint = cache.checkpoint();
    context.begin_span();
    remaining.set(Some(3)); // input creation and first slice update succeed
    assert!(append(&mut cache, 1, &context).is_err());
    assert_eq!((cache.offset(), cache.capacity()), (2, 4));
    let failed = report(&cache, &context);
    assert!(failed.total_bytes.unwrap() > 0);
    assert_eq!(failed.retained_bytes, Some(0));
    remaining.set(Some(2)); // both backing copies succeed; logical copy fails
    assert!(cache.restore(&checkpoint, &context).is_err());
    assert_eq!((cache.offset(), cache.capacity()), (2, 4));
    assert!(report(&cache, &context).total_bytes.unwrap() > failed.total_bytes.unwrap());
    remaining.set(None);
    append(&mut cache, 1, &context).unwrap();
    assert_eq!(cache.offset(), 3);
}

#[test]
fn wrong_geometry_context_and_capacity_overflow_fail_before_storage_work() {
    let (context, _) = context(false);
    let (foreign, _) = self::context(false);
    let mut cache = cache(4, &context);
    let state = CompressedAttentionState {
        latent: existing_f32(&[2, 3, 8], &context).unwrap(),
        rotary: existing_f32(&[2, 3, 4], &context).unwrap(),
    };
    assert!(cache.append(state.clone(), &foreign).is_err());
    let mut bad = state.clone();
    bad.latent = existing_f32(&[1, 3, 8], &context).unwrap();
    assert!(cache.append(bad, &context).is_err());
    cache.position = i32::MAX - 2;
    assert!(cache.append(state.clone(), &context).is_err());
    cache.position = i32::MAX - 4;
    assert!(cache.append(state, &context).is_err()); // capacity rounding overflows
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn missing_native_update_fact_keeps_an_otherwise_traceable_cache_unknown() {
    let (context, _) = context(true);
    let mut cache = cache(256, &context);
    append(&mut cache, 2, &context).unwrap();
    let report = report(&cache, &context);
    assert_eq!(report.total_bytes, None);
    assert_eq!(report.retained_bytes, None);
    assert_eq!(report.unpriced_operations.len(), 2);
}

#[test]
fn imported_capacity_and_independent_views_remain_charged_during_continuation() {
    for independent_views in [false, true] {
        let (context, _) = context(false);
        let latent_storage = WorkspaceExistingStorage::new(Some(4096), &context);
        let rotary_storage = WorkspaceExistingStorage::new(Some(2048), &context);
        let latent_view = if independent_views {
            WorkspaceExistingStorage::new(Some(192), &context)
        } else {
            latent_storage.clone()
        };
        let rotary_view = if independent_views {
            WorkspaceExistingStorage::new(Some(96), &context)
        } else {
            rotary_storage.clone()
        };
        let pair = |count, latent, rotary| CompressedAttentionState {
            latent: WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[2, count, 8], WorkspaceDtype::Float32).unwrap(),
                latent,
                &context,
            )
            .unwrap(),
            rotary: WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[2, count, 4], WorkspaceDtype::Float32).unwrap(),
                rotary,
                &context,
            )
            .unwrap(),
        };
        let mut cache = cache(4, &context)
            .with_existing_state(
                pair(7, &latent_storage, &rotary_storage),
                pair(2, &latent_view, &rotary_view),
            )
            .unwrap();
        assert_eq!((cache.offset(), cache.capacity()), (2, 7));
        let retained = 6144 + if independent_views { 288 } else { 0 };
        context.begin_state_span(cache.retained_arrays()).unwrap();
        let initial = report(&cache, &context);
        assert!(initial.operations.is_empty());
        assert_eq!(
            initial.state.as_ref().unwrap().retained_bytes,
            Some(retained)
        );
        append(&mut cache, 6, &context).unwrap();
        assert_eq!((cache.offset(), cache.capacity()), (8, 8));
        let continued = report(&cache, &context);
        assert_eq!(
            continued.state.as_ref().unwrap().displaced_bytes,
            Some(retained)
        );
        assert_eq!(
            continued.inference_transient_bytes(),
            continued.transient_bytes.map(|new| new + retained)
        );
        assert_eq!(continued.state.as_ref().unwrap().retained_bytes, Some(768));
    }
}

#[test]
fn imported_state_validates_geometry_and_context_without_equations() {
    let (context, _) = context(false);
    let (foreign, _) = self::context(false);
    let pair = |count, batch, context: &WorkspaceContext| CompressedAttentionState {
        latent: existing_f32(&[batch, count, 8], context).unwrap(),
        rotary: existing_f32(&[batch, count, 4], context).unwrap(),
    };
    for (storage, logical) in [
        (pair(4, 2, &foreign), pair(2, 2, &context)),
        (pair(4, 1, &context), pair(2, 2, &context)),
        (pair(4, 2, &context), pair(5, 2, &context)),
        (pair(4, 2, &context), pair(2, 3, &context)),
    ] {
        assert!(
            cache(4, &context)
                .with_existing_state(storage, logical)
                .is_err()
        );
    }
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn transaction_checkpoint_and_rollback_price_all_compressed_copies_beyond_successful_state() {
    use crate::working_memory::WorkspaceResidentLayerState;

    for independent_views in [false, true] {
        let (context, _) = context(false);
        let latent_storage = WorkspaceExistingStorage::new(Some(4096), &context);
        let rotary_storage = WorkspaceExistingStorage::new(Some(2048), &context);
        let latent_view = if independent_views {
            WorkspaceExistingStorage::new(Some(192), &context)
        } else {
            latent_storage.clone()
        };
        let rotary_view = if independent_views {
            WorkspaceExistingStorage::new(Some(96), &context)
        } else {
            rotary_storage.clone()
        };
        let pair = |count, latent, rotary| CompressedAttentionState {
            latent: WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[2, count, 8], WorkspaceDtype::Float32).unwrap(),
                latent,
                &context,
            )
            .unwrap(),
            rotary: WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[2, count, 4], WorkspaceDtype::Float32).unwrap(),
                rotary,
                &context,
            )
            .unwrap(),
        };
        let mut source = WorkspaceResidentLayerState::Compressed(
            cache(4, &context)
                .with_existing_state(
                    pair(7, &latent_storage, &rotary_storage),
                    pair(2, &latent_view, &rotary_view),
                )
                .unwrap(),
        );
        context.begin_state_span(source.retained_values()).unwrap();
        let checkpoint = source.checkpoint_for_transaction(&context).unwrap();
        let rollback = checkpoint.checkpoint_for_transaction(&context).unwrap();
        let copies = 2 * 2 * (7 + 2) * (8 + 4) * 4;
        let initial = context
            .report(&source.retained_values().cloned().collect::<Vec<_>>())
            .unwrap();
        let existing = 6144 + if independent_views { 288 } else { 0 };
        assert_eq!(
            initial.state.as_ref().unwrap().retained_bytes,
            Some(existing)
        );
        assert_eq!(initial.state.as_ref().unwrap().displaced_bytes, Some(0));
        assert_eq!(initial.inference_transient_bytes(), Some(copies));
        assert_eq!(initial.retained_bytes, Some(0));
        assert_eq!(
            initial
                .operations
                .iter()
                .filter(|op| matches!(op.kind, WorkspaceOperationKind::DeepCopy))
                .count(),
            8
        );
        // Even aliases of the original capacity buffers become distinct copied
        // logical views. Retaining each copy twice still charges each buffer once.
        let all_roots = source
            .retained_values()
            .chain(checkpoint.retained_values())
            .chain(rollback.retained_values())
            .chain(checkpoint.retained_values())
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            context
                .report(&all_roots)
                .unwrap()
                .state
                .unwrap()
                .retained_bytes,
            Some(existing + copies)
        );
        if let WorkspaceResidentLayerState::Compressed(source) = &mut source {
            append(source, 6, &context).unwrap();
            assert_eq!((source.offset(), source.capacity()), (8, 8));
        }
        for copy in [&checkpoint, &rollback] {
            let WorkspaceResidentLayerState::Compressed(copy) = copy else {
                unreachable!()
            };
            assert_eq!((copy.offset(), copy.capacity()), (2, 7));
        }
        let continued = context
            .report(&source.retained_values().cloned().collect::<Vec<_>>())
            .unwrap();
        assert_eq!(continued.state.as_ref().unwrap().retained_bytes, Some(768));
        assert_eq!(
            continued.state.as_ref().unwrap().displaced_bytes,
            Some(existing)
        );
        assert!(continued.inference_transient_bytes().unwrap() >= existing + copies);
    }
}

#[test]
fn transaction_copy_failures_and_missing_bounds_preserve_the_cold_source() {
    use crate::working_memory::WorkspaceResidentLayerState;

    let remaining = Rc::new(Cell::new(None));
    let context = WorkspaceContext::new(Facts {
        remaining: remaining.clone(),
        missing_updates: false,
        missing_copies: true,
    });
    let mut cache = cache(4, &context);
    append(&mut cache, 2, &context).unwrap();
    let source = WorkspaceResidentLayerState::Compressed(cache);
    context.begin_state_span(source.retained_values()).unwrap();
    remaining.set(Some(2));
    assert!(source.checkpoint_for_transaction(&context).is_err());
    let WorkspaceResidentLayerState::Compressed(original) = &source else {
        unreachable!()
    };
    assert_eq!((original.offset(), original.capacity()), (2, 4));
    remaining.set(None);
    let checkpoint = source.checkpoint_for_transaction(&context).unwrap();
    let _rollback = checkpoint.checkpoint_for_transaction(&context).unwrap();
    let report = context
        .report(&source.retained_values().cloned().collect::<Vec<_>>())
        .unwrap();
    assert_eq!(report.inference_transient_bytes(), None);
    assert_eq!(report.unpriced_operations.len(), 10);
    assert_eq!(report.state.unwrap().retained_bytes, Some(384));
    let (foreign, _) = self::context(false);
    assert!(source.checkpoint_for_transaction(&foreign).is_err());
    assert!(foreign.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn ordinary_fixed_and_pooling_transactions_share_and_deduplicate_existing_storage() {
    use crate::working_memory::{
        WorkspaceConcatStateFactory, WorkspacePoolingStateFactory, WorkspaceResidentLayerState,
    };
    use eredu_core::{AttentionPolicy, cache::*};

    let (context, _) = context(false);
    let storage = WorkspaceExistingStorage::new(Some(4096), &context);
    let view = |shape: &[i32]| {
        WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
            &storage,
            &context,
        )
        .unwrap()
    };
    let ordinary = WorkspaceConcatStateFactory::new(nz(2), &context)
        .unwrap()
        .project_layer(
            0,
            &LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap(),
            3,
            Some(view(&[2, 1, 3, 8])),
            Some(view(&[2, 1, 3, 8])),
            [],
        )
        .unwrap();
    let fixed_role = StateTensorRole::Convolution { slot: 0 };
    let fixed_policy = StateTensorPolicy::new(
        fixed_role,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(3).unwrap(),
            StateTensorDimension::fixed(8).unwrap(),
        ],
        StateTensorDtype::Floating,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    let fixed = WorkspaceConcatStateFactory::new(nz(2), &context)
        .unwrap()
        .project_layer(
            0,
            &LayerCachePolicy::FixedState {
                tensors: vec![fixed_policy],
            },
            3,
            None,
            None,
            [(fixed_role, Some(view(&[2, 3, 8])))],
        )
        .unwrap();
    let declarations = [
        PoolingStateComponent::PendingValues,
        PoolingStateComponent::PendingGates,
        PoolingStateComponent::Pooled,
    ]
    .into_iter()
    .map(|component| {
        let pooled = component == PoolingStateComponent::Pooled;
        let declaration = StateTensorPolicy::new_with_residency(
            StateTensorRole::Pooling {
                stream: 0,
                component,
            },
            vec![
                StateTensorDimension::Batch,
                if pooled {
                    StateTensorDimension::PrefixTokensDiv(nz(4))
                } else {
                    StateTensorDimension::PrefixTokensRem(nz(4))
                },
                StateTensorDimension::fixed(8).unwrap(),
            ],
            StateTensorDtype::Floating,
            if pooled {
                StateResidencyClass::SealablePaged
            } else {
                StateResidencyClass::AlwaysDeviceMutable
            },
        )
        .unwrap();
        if pooled {
            declaration.when_prefix_at_least(nz(4))
        } else {
            declaration.when_prefix_remainder_nonzero(nz(4))
        }
    })
    .collect::<Vec<_>>();
    let components = declarations
        .iter()
        .map(|declaration| {
            (
                declaration.role,
                declaration
                    .is_required_for(3)
                    .then(|| view(&declaration.resolved_shape(2, 3).unwrap())),
            )
        })
        .collect::<Vec<_>>();
    let pooling_policy = LayerCachePolicy::key_only_with_fixed_state(
        AttentionPolicy::sliding(5).unwrap(),
        1,
        8,
        declarations,
    )
    .unwrap();
    let pooling = WorkspacePoolingStateFactory::new(nz(2), &context)
        .unwrap()
        .project_layer(
            0,
            &pooling_policy,
            3,
            Some(view(&[2, 1, 3, 8])),
            Some(view(&[2, 1, 3, 1])),
            components,
        )
        .unwrap();
    let states = [
        WorkspaceResidentLayerState::Ordinary(ordinary),
        WorkspaceResidentLayerState::Ordinary(fixed),
        WorkspaceResidentLayerState::Pooling(pooling),
    ];
    context
        .begin_state_span(states.iter().flat_map(|state| state.retained_values()))
        .unwrap();
    let checkpoints = states
        .iter()
        .map(|state| state.checkpoint_for_transaction(&context).unwrap())
        .collect::<Vec<_>>();
    let rollback = checkpoints
        .iter()
        .map(|state| state.checkpoint_for_transaction(&context).unwrap())
        .collect::<Vec<_>>();
    let roots = states
        .iter()
        .chain(&checkpoints)
        .chain(&rollback)
        .flat_map(|state| state.retained_values().cloned())
        .collect::<Vec<_>>();
    let report = context.report(&roots).unwrap();
    assert!(report.operations.is_empty());
    assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(4096));
    assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
    assert_eq!(report.inference_transient_bytes(), Some(0));
    let (foreign, _) = self::context(false);
    for state in &states {
        assert!(state.checkpoint_for_transaction(&foreign).is_err());
    }
}

#[test]
fn empty_transaction_checkpoints_still_require_the_selected_context() {
    use crate::working_memory::{
        WorkspaceConcatStateFactory, WorkspacePoolingStateFactory, WorkspaceResidentLayerState,
    };
    use eredu_core::{AttentionPolicy, cache::LayerCachePolicy};

    let (context, _) = context(false);
    let (foreign, _) = self::context(false);
    let ordinary = WorkspaceConcatStateFactory::new(nz(2), &context)
        .unwrap()
        .project_layer(
            0,
            &LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap(),
            0,
            None,
            None,
            [],
        )
        .unwrap();
    let pooling = WorkspacePoolingStateFactory::new(nz(2), &context)
        .unwrap()
        .project_layer(
            0,
            &LayerCachePolicy::key_only(AttentionPolicy::sliding(5).unwrap(), 1, 8).unwrap(),
            0,
            None,
            None,
            [],
        )
        .unwrap();
    for state in [
        WorkspaceResidentLayerState::Ordinary(ordinary),
        WorkspaceResidentLayerState::Compressed(cache(4, &context)),
        WorkspaceResidentLayerState::Pooling(pooling),
    ] {
        assert!(state.retained_values().next().is_none());
        assert!(state.checkpoint_for_transaction(&foreign).is_err());
        assert!(
            state
                .checkpoint_for_transaction(&context)
                .unwrap()
                .retained_values()
                .next()
                .is_none()
        );
    }
    assert!(context.report(&[]).unwrap().operations.is_empty());
    assert!(foreign.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn present_empty_logical_state_compacts_and_grows_from_zero_without_losing_owners() {
    for capacity in [0, 4] {
        let (context, _) = context(false);
        let pair = |positions| CompressedAttentionState {
            latent: existing_f32(&[2, positions, 8], &context).unwrap(),
            rotary: existing_f32(&[2, positions, 4], &context).unwrap(),
        };
        let source = cache(4, &context)
            .with_existing_state(pair(capacity), pair(0))
            .unwrap();
        assert_eq!((source.offset(), source.capacity()), (0, capacity));
        context.begin_state_span(source.retained_arrays()).unwrap();
        let mut copied = source.isolated_snapshot(&context).unwrap();
        assert_eq!((copied.offset(), copied.capacity()), (0, 0));
        assert_eq!(copied.retained_arrays().count(), 4);
        let copied_report = report(&copied, &context);
        assert_eq!(copied_report.operations.len(), 4);
        assert!(copied_report.total_bytes.is_some());
        context.begin_state_span(copied.retained_arrays()).unwrap();
        append(&mut copied, 3, &context).unwrap();
        assert_eq!((copied.offset(), copied.capacity()), (3, 4));
        assert_eq!((source.offset(), source.capacity()), (0, capacity));
        assert!(report(&copied, &context).total_bytes.is_some());
    }
}
