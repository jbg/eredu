use super::*;
use crate::working_memory::{
    WorkspacePagedAppendState, WorkspacePagedBlock, WorkspacePagedGeometry,
};

#[test]
fn paged_stored_host_effects_keep_occurrence_and_partial_spending_without_device_aliases() {
    use crate::working_memory::WorkspacePagedLayerState;
    use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions, BlockwiseAttentionSpec};
    let (context, fail_after) = context();
    let (keys, values) = values(3, &context);
    let geometry = WorkspacePagedGeometry {
        block_size: 3,
        dimensions: [2, 2, 8],
        offset: 3,
        tail_start: 3,
        window: None,
        prefix_tokens: 0,
        key_only: false,
        retain_discarded: false,
    };
    let mut state = WorkspacePagedAppendState::project(
        geometry,
        [WorkspacePagedBlock::new(0, 3, [keys, values], true)].into_iter(),
        None,
        &context,
    )
    .unwrap();
    let types = [WorkspaceFloatingType::Float32; 2];
    context.begin_span();
    fail_after.set(Some(1));
    assert!(state.store_block_host(0, types, &context).is_err());
    assert!(!state.blocks()[0].requires_host_transfer());
    assert_eq!(context.operation_count(), 1);
    fail_after.set(None);
    state.store_block_host(0, types, &context).unwrap();
    assert!(state.blocks()[0].requires_host_transfer());
    assert!(state.blocks()[0].tensor_values().is_none());
    assert!(state.store_block_host(0, types, &context).is_err());
    let (foreign, _) = super::context();
    assert!(state.store_block_host(0, types, &foreign).is_err());
    assert_eq!(context.operation_count(), 3);
    assert_eq!(foreign.operation_count(), 0);
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap();
    let saved = WorkspacePagedLayerState::project(
        0,
        &policy,
        state.copy_metadata(&context).unwrap(),
        std::iter::empty(),
        &context,
    )
    .unwrap();
    assert_eq!(saved.retained_values().count(), 0);
    let queries = WorkspaceTensor::unloaded_f32(&[2, 2, 1, 8], &context).unwrap();
    let spec = || BlockwiseAttentionSpec {
        queries: &queries,
        scale: 0.5,
        mask: None,
        query_start: 2,
        context_end: 3,
        sliding_window: None,
        prefix_tokens: 0,
        sinks: None,
    };
    let options = BlockwiseAttentionOptions {
        arithmetic: AttentionArithmetic::InputScores,
        softcap: None,
    };
    assert!(
        state
            .scan_attention(
                spec(),
                options,
                |_, _, context| Err(
                    context.metadata_error(format_args!("after completed stored Host loads"))
                ),
                &context
            )
            .is_err()
    );
    assert!(!state.blocks()[0].requires_host_transfer());
    assert!(state.blocks()[0].tensor_values().is_some());
    let output = state
        .scan_attention(spec(), options, |_, _, _| Ok(None), &context)
        .unwrap();
    let report = context.report(&[output]).unwrap();
    let stores: Vec<_> = report
        .operations
        .iter()
        .filter_map(|operation| {
            if let WorkspaceOperationKind::HostStoreFloating(identity, _) = &operation.kind {
                assert!(operation.outputs.is_empty());
                Some(identity)
            } else {
                None
            }
        })
        .collect();
    let loads: Vec<_> = report
        .operations
        .iter()
        .filter_map(|operation| {
            if let WorkspaceOperationKind::HostLoadStoredFloating(identity, _) = &operation.kind {
                assert!(operation.inputs.is_empty());
                Some(identity)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(stores.len(), 3);
    assert_eq!(loads.len(), 2);
    assert!(stores[1].same_store(loads[0]));
    assert!(stores[2].same_store(loads[1]));
    assert!(!stores[0].same_store(loads[0]));
    assert_eq!(saved.retained_values().count(), 0);
    assert_eq!(saved.position(), 3);
}

#[test]
fn paged_append_keeps_ordered_blocks_and_restores_failed_tail_without_refund() {
    let (context, fail_after) = context();
    let pair = |n| {
        let (k, v) = values(n, &context);
        [k, v]
    };
    let geometry = WorkspacePagedGeometry {
        block_size: 3,
        dimensions: [2, 2, 8],
        offset: 5,
        tail_start: 3,
        window: Some(4),
        prefix_tokens: 3,
        key_only: false,
        retain_discarded: false,
    };
    let block = WorkspacePagedBlock::new(0, 3, pair(3), true);
    let mut state =
        WorkspacePagedAppendState::project(geometry, [block].into_iter(), Some(pair(2)), &context)
            .unwrap();
    let input = pair(8);
    context.begin_span();
    // The first full tail has been published/sealed before this later failure.
    fail_after.set(Some(5));
    assert!(
        state
            .append_normalized(input.clone(), false, &context)
            .is_err()
    );
    assert_eq!(state.geometry(), geometry);
    assert_eq!(
        state.blocks().iter().map(|b| b.range()).collect::<Vec<_>>(),
        [0..3]
    );
    assert_eq!(state.tail().unwrap()[0].shape()[2], 2);
    let spent = context.report(&[]).unwrap().total_bytes.unwrap();
    assert!(spent > 0);
    fail_after.set(None);
    state.append_normalized(input, false, &context).unwrap();
    assert_eq!(state.geometry().offset, 13);
    assert_eq!(state.geometry().tail_start, 12);
    assert_eq!(
        state.blocks().iter().map(|b| b.range()).collect::<Vec<_>>(),
        [0..3, 9..12]
    );
    assert_eq!(state.tail().unwrap()[0].shape()[2], 1);
    assert!(context.report(&[]).unwrap().total_bytes.unwrap() > spent);
    let (other, _) = super::context();
    let (k, v) = values(1, &other);
    assert!(state.append_normalized([k, v], true, &other).is_err());
    assert_eq!(state.geometry().offset, 13);
}

#[test]
fn paged_scan_preserves_prefix_partial_windows_two_pass_bias_and_failed_history() {
    use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions, BlockwiseAttentionSpec};
    let (context, _) = context();
    let pair = |n| {
        let (k, v) = values(n, &context);
        [k, v]
    };
    let geometry = WorkspacePagedGeometry {
        block_size: 3,
        dimensions: [2, 2, 8],
        offset: 13,
        tail_start: 12,
        window: Some(4),
        prefix_tokens: 3,
        key_only: false,
        retain_discarded: false,
    };
    let blocks =
        [0, 3, 6, 9].map(|start| WorkspacePagedBlock::new(start, start + 3, pair(3), start == 0));
    let mut state =
        WorkspacePagedAppendState::project(geometry, blocks.into_iter(), Some(pair(1)), &context)
            .unwrap();
    let queries = pair(3)[0].clone();
    let spec = || BlockwiseAttentionSpec {
        queries: &queries,
        scale: 0.5,
        mask: None,
        query_start: 10,
        context_end: 13,
        sliding_window: Some(4),
        prefix_tokens: 3,
        sinks: None,
    };
    let options = BlockwiseAttentionOptions {
        arithmetic: AttentionArithmetic::InputScores,
        softcap: Some(1.75),
    };
    let mut visited = Vec::new();
    let failure = state.scan_attention(
        spec(),
        options,
        |start, end, ctx| {
            visited.push(start..end);
            if visited.len() == 5 {
                return Err(ctx.metadata_error(format_args!("injected second-pass bias failure")));
            }
            Ok(Some(WorkspaceTensor::unloaded_f32(
                &[3, (end - start) as i32],
                ctx,
            )?))
        },
        &context,
    );
    assert!(failure.is_err());
    assert_eq!(visited, [0..3, 6..9, 9..12, 12..13, 0..3]);
    assert_eq!(
        state.blocks().iter().map(|b| b.range()).collect::<Vec<_>>(),
        [0..3, 3..6, 6..9, 9..12]
    );
    let spent = context.report(&[]).unwrap().total_bytes.unwrap();
    visited.clear();
    let output = state
        .scan_attention(
            spec(),
            options,
            |start, end, ctx| {
                visited.push(start..end);
                Ok(Some(WorkspaceTensor::unloaded_f32(
                    &[3, (end - start) as i32],
                    ctx,
                )?))
            },
            &context,
        )
        .unwrap();
    assert_eq!(output.shape(), [2, 2, 3, 8]);
    assert_eq!(
        visited,
        [0..3, 6..9, 9..12, 12..13, 0..3, 6..9, 9..12, 12..13]
    );
    assert_eq!(
        state.blocks().iter().map(|b| b.range()).collect::<Vec<_>>(),
        [0..3, 9..12]
    );
    assert!(context.report(&[output]).unwrap().total_bytes.unwrap() > spent);
    let mut wrong = spec();
    wrong.query_start = 9;
    assert!(
        state
            .scan_attention(
                wrong,
                options,
                |_, _, _| panic!("invalid source must precede bias callback"),
                &context
            )
            .is_err()
    );
}

#[test]
fn paged_layer_uses_shared_equations_fixed_slots_and_independent_metadata_checkpoints() {
    use crate::working_memory::{WorkspacePagedLayerState, WorkspaceResidentLayerState};
    use eredu_nn::AttentionArithmetic;
    let (context, _) = context();
    let geometry = WorkspacePagedGeometry {
        block_size: 3,
        dimensions: [2, 2, 8],
        offset: 5,
        tail_start: 3,
        window: Some(4),
        prefix_tokens: 3,
        key_only: false,
        retain_discarded: false,
    };
    let policy = LayerCachePolicy::KeyValueWithFixedState {
        attention: AttentionPolicy::sliding(4).unwrap(),
        num_key_value_heads: NonZeroU32::new(2).unwrap(),
        head_dim: NonZeroU32::new(8).unwrap(),
        tensors: vec![fixed()],
    };
    let (k, v) = values(3, &context);
    let (tailk, tailv) = values(2, &context);
    let paged = WorkspacePagedAppendState::project(
        geometry,
        [WorkspacePagedBlock::new(0, 3, [k, v], true)].into_iter(),
        Some([tailk, tailv]),
        &context,
    )
    .unwrap();
    let fixed_value = WorkspaceTensor::unloaded_f32(&[2, 3, 8], &context)
        .unwrap()
        .square(&context)
        .unwrap();
    let mut current = WorkspaceResidentLayerState::Paged(
        WorkspacePagedLayerState::project(
            0,
            &policy,
            paged,
            [(StateTensorRole::Convolution { slot: 3 }, Some(fixed_value))],
            &context,
        )
        .unwrap(),
    );
    assert!(current.uses_blockwise_attention());
    assert_eq!(current.retained_values().count(), 5);
    let mut saved = current.checkpoint_for_transaction(&context).unwrap();
    *current.convolution_state(3).unwrap() = None;
    assert!(saved.convolution_state(3).unwrap().is_some());
    let (k, v) = values(3, &context);
    let (k, v) = current.update_for_attention(k, v, &context).unwrap();
    assert_eq!(k.shape(), [2, 2, 3, 8]);
    let output = current
        .attention(
            AttentionRequest {
                queries: k.clone(),
                keys: k,
                values: v,
                scale: 0.5,
                mask: None,
                sinks: None,
                arithmetic: AttentionArithmetic::InputScores,
                softcap: Some(1.75),
            },
            &context,
        )
        .unwrap();
    assert_eq!(output.shape(), [2, 2, 3, 8]);
    assert_eq!(current.position(), 8);
    assert_eq!(saved.position(), 5);
    let spent = context.report(&[output]).unwrap().total_bytes.unwrap();
    current.reset().unwrap();
    assert_eq!(current.position(), 0);
    assert_eq!(current.retained_values().count(), 0);
    assert_eq!(saved.position(), 5);
    assert_eq!(saved.retained_values().count(), 5);
    assert!(context.report(&[]).unwrap().total_bytes.unwrap() >= spent);
    let (foreign, _) = super::context();
    assert!(saved.checkpoint_for_transaction(&foreign).is_err());
}

#[test]
fn isolated_paged_copy_preserves_intervals_fixed_values_and_failed_source_without_refund() {
    use crate::working_memory::WorkspacePagedLayerState;
    let (context, fail_after) = context();
    let geometry = WorkspacePagedGeometry {
        block_size: 3,
        dimensions: [2, 2, 8],
        offset: 5,
        tail_start: 3,
        window: Some(4),
        prefix_tokens: 3,
        key_only: false,
        retain_discarded: false,
    };
    let policy = LayerCachePolicy::KeyValueWithFixedState {
        attention: AttentionPolicy::sliding(4).unwrap(),
        num_key_value_heads: NonZeroU32::new(2).unwrap(),
        head_dim: NonZeroU32::new(8).unwrap(),
        tensors: vec![fixed()],
    };
    let (k, v) = values(3, &context);
    let (tk, tv) = values(2, &context);
    let paged = WorkspacePagedAppendState::project(
        geometry,
        [WorkspacePagedBlock::new(0, 3, [k, v], true)].into_iter(),
        Some([tk, tv]),
        &context,
    )
    .unwrap();
    let fixed_value = WorkspaceTensor::unloaded_f32(&[2, 3, 8], &context)
        .unwrap()
        .square(&context)
        .unwrap();
    let original = WorkspacePagedLayerState::project(
        0,
        &policy,
        paged,
        [(StateTensorRole::Convolution { slot: 3 }, Some(fixed_value))],
        &context,
    )
    .unwrap();
    let source: Vec<_> = original.retained_values().cloned().collect();
    context.begin_state_span(&source).unwrap();
    fail_after.set(Some(3));
    assert!(original.copy_isolated(&context).is_err());
    assert_eq!(original.geometry(), geometry);
    // The report deduplicates the actual storage identities, including entry
    // storage from before this span. Equal layouts alone would miss replacement.
    for (current, saved) in original.retained_values().zip(&source) {
        let saved_storage = context.report(&[saved.clone()]).unwrap().closing_storage;
        assert_eq!(saved_storage.maximum_allocations, 1);
        assert_eq!(
            context
                .report(&[current.clone(), saved.clone()])
                .unwrap()
                .closing_storage,
            saved_storage
        );
    }
    let spent = context.report(&source).unwrap().total_bytes.unwrap();
    fail_after.set(None);
    let copied = original.copy_isolated(&context).unwrap();
    assert_eq!(copied.geometry(), geometry);
    assert_eq!(copied.retained_values().count(), 5);
    for (copy, original) in copied.retained_values().zip(&source) {
        assert_eq!(copy.layout(), original.layout());
        let original_storage = context.report(&[original.clone()]).unwrap().closing_storage;
        let copied_storage = context.report(&[copy.clone()]).unwrap().closing_storage;
        assert_eq!(copied_storage, original_storage);
        let union = context
            .report(&[copy.clone(), original.clone()])
            .unwrap()
            .closing_storage;
        assert_eq!(union.maximum_allocations, 2);
        assert_eq!(union.bytes, original_storage.bytes.unwrap().checked_mul(2));
    }
    let roots: Vec<_> = source
        .iter()
        .cloned()
        .chain(copied.retained_values().cloned())
        .collect();
    assert!(context.report(&roots).unwrap().total_bytes.unwrap() > spent);
    let (foreign, _) = super::context();
    assert!(original.copy_isolated(&foreign).is_err());
    drop(original);
    assert_eq!(copied.geometry(), geometry);
    assert_eq!(copied.retained_values().count(), 5);
}

#[test]
fn paged_host_transfer_is_independent_once_only_and_keeps_completed_failure_prefix() {
    use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions, BlockwiseAttentionSpec};
    let (context, _) = context();
    let (keys, values) = values(3, &context);
    let sources = [keys.clone(), values.clone()];
    let block = WorkspacePagedBlock::host(
        0,
        3,
        [keys, values],
        [
            WorkspaceFloatingType::Float16,
            WorkspaceFloatingType::Bfloat16,
        ],
    );
    let geometry = WorkspacePagedGeometry {
        block_size: 3,
        dimensions: [2, 2, 8],
        offset: 3,
        tail_start: 3,
        window: None,
        prefix_tokens: 0,
        key_only: false,
        retain_discarded: false,
    };
    let mut state =
        WorkspacePagedAppendState::project(geometry, [block].into_iter(), None, &context).unwrap();
    let queries = WorkspaceTensor::unloaded_f32(&[2, 2, 1, 8], &context).unwrap();
    let spec = || BlockwiseAttentionSpec {
        queries: &queries,
        scale: 0.5,
        mask: None,
        query_start: 2,
        context_end: 3,
        sliding_window: None,
        prefix_tokens: 0,
        sinks: None,
    };
    let options = BlockwiseAttentionOptions {
        arithmetic: AttentionArithmetic::InputScores,
        softcap: Some(1.75),
    };
    context.begin_span();
    // These are completed host copies before the actual attention callback.
    assert!(
        state
            .scan_attention(
                spec(),
                options,
                |_, _, context| {
                    Err(context.metadata_error(format_args!("failure after completed promotion")))
                },
                &context
            )
            .is_err()
    );
    assert!(!state.blocks()[0].requires_host_transfer());
    assert_eq!(state.geometry(), geometry);
    let report = context.report(&sources).unwrap();
    let transfers: Vec<_> = report
        .operations
        .iter()
        .filter_map(|operation| {
            if let WorkspaceOperationKind::HostTransferFloating(dtype) = operation.kind {
                Some(dtype)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        transfers,
        [
            WorkspaceFloatingType::Float16,
            WorkspaceFloatingType::Bfloat16
        ]
    );
    let spent = report.total_bytes.unwrap();
    for (source, output) in sources.iter().zip(state.blocks()[0].values().unwrap()) {
        let one = context.report(&[source.clone()]).unwrap().closing_storage;
        let both = context
            .report(&[source.clone(), output.clone()])
            .unwrap()
            .closing_storage;
        assert_eq!(both.maximum_allocations, one.maximum_allocations + 1);
    }
    let output = state
        .scan_attention(spec(), options, |_, _, _| Ok(None), &context)
        .unwrap();
    let after = context.report(&[output]).unwrap();
    assert_eq!(
        after
            .operations
            .iter()
            .filter(|operation| matches!(
                operation.kind,
                WorkspaceOperationKind::HostTransferFloating(_)
            ))
            .count(),
        2
    );
    assert!(after.total_bytes.unwrap() > spent);
    let (other, _) = super::context();
    let before = other.operation_count();
    assert!(
        sources[0]
            .transfer_host_floating(WorkspaceFloatingType::Float32, &other)
            .is_err()
    );
    assert_eq!(other.operation_count(), before);
}

#[test]
fn conditional_host_itinerary_preserves_source_identity_each_selected_pass_and_failed_prefix() {
    use crate::working_memory::WorkspacePagedHostTrace;
    use eredu_nn::workspace::WorkspaceRepresentation;
    use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions, BlockwiseAttentionSpec};
    let (context, _) = context();
    let make = |dtype| {
        WorkspaceTensor::existing(
            context
                .layout(&[2, 2, 3, 8], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
            &context,
        )
        .unwrap()
    };
    let types = [
        WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Bfloat16,
    ];
    let trace = WorkspacePagedHostTrace::new(&context).unwrap();
    let mut state = WorkspacePagedAppendState::project(
        WorkspacePagedGeometry {
            block_size: 3,
            dimensions: [2, 2, 8],
            offset: 6,
            tail_start: 6,
            window: None,
            prefix_tokens: 0,
            key_only: false,
            retain_discarded: false,
        },
        [
            WorkspacePagedBlock::host(0, 3, [make(types[0]), make(types[1])], types),
            WorkspacePagedBlock::new(3, 6, [make(types[0]), make(types[1])], true),
        ]
        .into_iter(),
        None,
        &context,
    )
    .unwrap()
    .with_host_transfers(trace.clone())
    .unwrap();
    let query = WorkspaceTensor::unloaded_f32(&[2, 2, 1, 8], &context).unwrap();
    let spec = || BlockwiseAttentionSpec {
        queries: &query,
        scale: 0.5,
        mask: None,
        query_start: 5,
        context_end: 6,
        sliding_window: None,
        prefix_tokens: 0,
        sinks: None,
    };
    let options = BlockwiseAttentionOptions {
        arithmetic: AttentionArithmetic::InputScores,
        softcap: Some(1.75),
    };
    context.begin_span();
    for _ in 0..2 {
        state
            .scan_attention(spec(), options, |_, _, _| Ok(None), &context)
            .unwrap();
    }
    let count = context.operation_count();
    assert!(
        state
            .scan_attention(
                spec(),
                options,
                |_, _, context| Err(
                    context.metadata_error(format_args!("after conditional load completion"))
                ),
                &context
            )
            .is_err()
    );
    assert!(context.operation_count() > count);
    trace
        .with_entries(|entries| {
            assert_eq!(entries.len(), 2);
            assert!(!entries[0].requires_store());
            assert!(entries[1].requires_store());
            assert_eq!(entries[0].range(), 0..3);
            assert_eq!(entries[1].range(), 3..6);
            for entry in entries {
                assert_eq!(entry.first_invocation(), (5, 6));
                assert_eq!(entry.floating_type(0), Some(types[0]));
                assert_eq!(entry.floating_type(1), Some(types[1]));
            }
        })
        .unwrap();
    trace
        .with_loads(|loads| {
            assert_eq!(loads.len(), 9);
            for scan in loads[..8].chunks_exact(4) {
                assert_eq!(
                    scan.iter()
                        .map(|load| (load.entry, load.pass))
                        .collect::<Vec<_>>(),
                    [(0, 0), (1, 0), (0, 1), (1, 1)]
                );
                assert!(
                    scan.iter()
                        .all(|load| load.query_start == 5 && load.context_end == 6)
                );
            }
            assert_eq!((loads[8].entry, loads[8].pass), (0, 0));
        })
        .unwrap();
    let report = context.report(&[]).unwrap();
    assert_eq!(
        report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::HostStoreFloating(..)))
            .count(),
        6
    );
    assert_eq!(
        report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::HostLoadStoredFloating(..)))
            .count(),
        8
    );
    assert_eq!(
        report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::HostTransferFloating(_)))
            .count(),
        10
    );
    let stores = report
        .operations
        .iter()
        .filter_map(|op| match &op.kind {
            WorkspaceOperationKind::HostStoreFloating(identity, _) => Some(identity),
            _ => None,
        })
        .collect::<Vec<_>>();
    // Both later attempts, including the failed scan prefix, pay the same
    // possible stores. These are aliases of two destinations, not six sources.
    for pair in stores.chunks_exact(2) {
        assert!(pair[0].same_store(stores[0]));
        assert!(pair[1].same_store(stores[1]));
    }
    trace
        .with_entries(|entries| {
            let values = entries[1].stored().unwrap();
            let (foreign, _) = super::context();
            let count = foreign.operation_count();
            assert!(values[0].declare_deferred_store(&foreign).is_err());
            assert_eq!(foreign.operation_count(), count);
        })
        .unwrap();
    drop(state);
    trace
        .with_entries(|entries| assert_eq!(entries.len(), 2))
        .unwrap();
    trace
        .with_loads(|loads| assert_eq!(loads.len(), 9))
        .unwrap();
}

#[test]
fn resumed_paged_copy_retains_immutable_host_roots_and_isolates_tail_without_refund() {
    use crate::working_memory::WorkspacePagedLayerState;
    let (context, fail_after) = context();
    let trace = crate::working_memory::WorkspacePagedHostTrace::new(&context).unwrap();
    let (keys, values) = values(3, &context);
    let (tail_keys, tail_values) = super::values(2, &context);
    let geometry = WorkspacePagedGeometry {
        block_size: 3,
        dimensions: [2, 2, 8],
        offset: 5,
        tail_start: 3,
        window: None,
        prefix_tokens: 0,
        key_only: false,
        retain_discarded: false,
    };
    let paged = WorkspacePagedAppendState::project(
        geometry,
        [WorkspacePagedBlock::host(
            0,
            3,
            [keys, values],
            [WorkspaceFloatingType::Float32; 2],
        )]
        .into_iter(),
        Some([tail_keys, tail_values]),
        &context,
    )
    .unwrap()
    .with_host_transfers(trace.clone())
    .unwrap();
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap();
    let original =
        WorkspacePagedLayerState::project(0, &policy, paged, std::iter::empty(), &context).unwrap();
    let source: Vec<_> = original.retained_values().cloned().collect();
    context.begin_state_span(&source).unwrap();
    assert!(original.copy_isolated(&context).is_err());
    fail_after.set(Some(1));
    assert!(original.copy_for_resume(&context).is_err());
    let spent = context.report(&source).unwrap().total_bytes.unwrap();
    fail_after.set(None);
    let copied = original.copy_for_resume(&context).unwrap();
    trace
        .with_entries(|entries| assert!(entries.is_empty()))
        .unwrap();
    trace.with_loads(|loads| assert!(loads.is_empty())).unwrap();
    assert_eq!(copied.geometry(), geometry);
    assert_eq!(copied.retained_values().count(), 4);
    for (index, (copied, original)) in copied.retained_values().zip(&source).enumerate() {
        let together = context
            .report(&[copied.clone(), original.clone()])
            .unwrap()
            .closing_storage;
        assert_eq!(together.maximum_allocations, if index < 2 { 1 } else { 2 });
    }
    let roots: Vec<_> = source
        .iter()
        .cloned()
        .chain(copied.retained_values().cloned())
        .collect();
    assert!(context.report(&roots).unwrap().total_bytes.unwrap() > spent);
    let (foreign, _) = super::context();
    assert!(original.copy_for_resume(&foreign).is_err());
    drop((original, source, roots));
    assert_eq!(copied.geometry(), geometry);
    assert_eq!(copied.retained_values().count(), 4);
}

#[test]
fn resumed_host_copy_shares_only_unstarted_itinerary_and_refuses_started_alias() {
    use crate::working_memory::WorkspacePagedHostTrace;
    use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions, BlockwiseAttentionSpec};
    let (context, _) = context();
    let (keys, values) = values(3, &context);
    let trace = WorkspacePagedHostTrace::new(&context).unwrap();
    let original = WorkspacePagedAppendState::project(
        WorkspacePagedGeometry {
            block_size: 3,
            dimensions: [2, 2, 8],
            offset: 3,
            tail_start: 3,
            window: None,
            prefix_tokens: 0,
            key_only: false,
            retain_discarded: false,
        },
        [WorkspacePagedBlock::host(
            0,
            3,
            [keys, values],
            [WorkspaceFloatingType::Float32; 2],
        )]
        .into_iter(),
        None,
        &context,
    )
    .unwrap()
    .with_host_transfers(trace.clone())
    .unwrap();
    let mut resumed = original.copy_for_resume(&context).unwrap();
    let query = WorkspaceTensor::unloaded_f32(&[2, 2, 1, 8], &context).unwrap();
    context.begin_span();
    resumed
        .scan_attention(
            BlockwiseAttentionSpec {
                queries: &query,
                scale: 0.5,
                mask: None,
                query_start: 2,
                context_end: 3,
                sliding_window: None,
                prefix_tokens: 0,
                sinks: None,
            },
            BlockwiseAttentionOptions {
                arithmetic: AttentionArithmetic::Fused,
                softcap: None,
            },
            |_, _, _| Ok(None),
            &context,
        )
        .unwrap();
    trace
        .with_entries(|entries| {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].range(), 0..3);
            assert!(!entries[0].requires_store());
        })
        .unwrap();
    trace
        .with_loads(|loads| assert_eq!(loads.len(), 1))
        .unwrap();
    let operations = context.operation_count();
    assert!(original.copy_for_resume(&context).is_err());
    assert!(resumed.copy_for_resume(&context).is_err());
    assert_eq!(context.operation_count(), operations);
    drop((original, resumed));
    trace
        .with_entries(|entries| assert_eq!(entries.len(), 1))
        .unwrap();
}

#[test]
fn paged_read_sources_keep_opaque_geometry_and_use_shared_loads_across_scan_passes() {
    use crate::working_memory::WorkspacePagedHostTrace;
    use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions, BlockwiseAttentionSpec};
    let (context, _) = context();
    let read = [
        context
            .declare_host_read_value(&[2, 2, 3, 8], WorkspaceFloatingType::Float16)
            .unwrap(),
        context
            .declare_host_read_value(&[2, 2, 3, 8], WorkspaceFloatingType::Float16)
            .unwrap(),
    ];
    assert_eq!(context.operation_count(), 0);
    let (foreign, _) = super::context();
    assert!(read[0].load(&foreign).is_err());
    assert_eq!(foreign.operation_count(), 0);
    let geometry = WorkspacePagedGeometry {
        block_size: 3,
        dimensions: [2, 2, 8],
        offset: 3,
        tail_start: 3,
        window: None,
        prefix_tokens: 0,
        key_only: false,
        retain_discarded: false,
    };
    assert!(
        WorkspacePagedAppendState::project(
            geometry,
            [WorkspacePagedBlock::read_source(0, 3, read.clone())].into_iter(),
            None,
            &foreign,
        )
        .is_err()
    );
    let trace = WorkspacePagedHostTrace::new(&context).unwrap();
    let mut state = WorkspacePagedAppendState::project(
        geometry,
        [WorkspacePagedBlock::read_source(0, 3, read)].into_iter(),
        None,
        &context,
    )
    .unwrap()
    .with_host_transfers(trace.clone())
    .unwrap();
    assert!(state.blocks()[0].values().is_none());
    assert!(state.blocks()[0].tensor_values().is_none());
    assert!(state.blocks()[0].requires_host_transfer());
    let saved = state.copy_for_resume(&context).unwrap();
    assert!(saved.blocks()[0].tensor_values().is_none());
    let queries = WorkspaceTensor::unloaded_f32(&[2, 2, 1, 8], &context).unwrap();
    let output = state
        .scan_attention(
            BlockwiseAttentionSpec {
                queries: &queries,
                scale: 0.5,
                mask: None,
                query_start: 2,
                context_end: 3,
                sliding_window: None,
                prefix_tokens: 0,
                sinks: None,
            },
            BlockwiseAttentionOptions {
                arithmetic: AttentionArithmetic::InputScores,
                softcap: None,
            },
            |_, _, _| Ok(None),
            &context,
        )
        .unwrap();
    assert!(state.blocks()[0].tensor_values().is_some());
    trace
        .with_entries(|entries| {
            assert_eq!(entries.len(), 1);
            assert!(!entries[0].requires_store());
            assert_eq!(
                entries[0].floating_type(0),
                Some(WorkspaceFloatingType::Float16)
            );
            assert_eq!(entries[0].shape(1), Some([2, 2, 3, 8].as_slice()));
        })
        .unwrap();
    trace
        .with_loads(|loads| assert_eq!(loads.iter().map(|v| v.pass).collect::<Vec<_>>(), [0, 1]))
        .unwrap();
    let report = context.report(&[output]).unwrap();
    assert!(
        !report
            .operations
            .iter()
            .any(|op| matches!(op.kind, WorkspaceOperationKind::HostStoreFloating(..)))
    );
    assert_eq!(
        report
            .operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::HostLoadStoredFloating(..)))
            .count(),
        4
    );
    assert!(
        state.copy_for_resume(&context).is_err(),
        "an active read itinerary cannot become a saved source"
    );
}
