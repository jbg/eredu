use super::*;
use crate::LayerRuntimeState;
use eredu_core::{AttentionPolicy, LayerSchedule, cache::*};

#[derive(Debug)]
struct Facts;
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|layout| {
                    if matches!(
                        op.kind,
                        WorkspaceOperationKind::Index { .. } | WorkspaceOperationKind::StaticSlice { .. } | WorkspaceOperationKind::View(_) | WorkspaceOperationKind::Transpose(_)
                    ) {
                        Ok(WorkspaceOutputStorage::AliasInput(0))
                    } else {
                        layout.bytes().map(WorkspaceOutputStorage::Allocate)
                    }
                })
                .collect::<Result<_, _>>()?,
            scratch_bytes: 0,
            assumptions: "fixture exact allocations and view aliases".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixture has no host payload workspace".into(),
        }))
    }
}

fn policy(streams: usize, window: u32) -> LayerCachePolicy {
    let mut tensors = Vec::new();
    for stream in 0..streams {
        let ratio = NonZeroU32::new(if stream == 0 { 4 } else { 6 }).unwrap();
        for component in COMPONENTS {
            if streams == 1
                && matches!(
                    component,
                    PoolingStateComponent::OverlapValues | PoolingStateComponent::OverlapGates
                )
            {
                continue;
            }
            let extent = match component {
                PoolingStateComponent::PendingValues | PoolingStateComponent::PendingGates => {
                    StateTensorDimension::PrefixTokensRem(ratio)
                }
                PoolingStateComponent::Pooled => StateTensorDimension::PrefixTokensDiv(ratio),
                _ => StateTensorDimension::Fixed(ratio),
            };
            let declaration = StateTensorPolicy::new_with_residency(
                StateTensorRole::Pooling {
                    stream: stream as u32,
                    component,
                },
                vec![
                    StateTensorDimension::Batch,
                    extent,
                    StateTensorDimension::fixed(8).unwrap(),
                ],
                StateTensorDtype::Floating,
                if component == PoolingStateComponent::Pooled {
                    StateResidencyClass::SealablePaged
                } else {
                    StateResidencyClass::AlwaysDeviceMutable
                },
            )
            .unwrap();
            tensors.push(
                if matches!(
                    component,
                    PoolingStateComponent::PendingValues | PoolingStateComponent::PendingGates
                ) {
                    declaration.when_prefix_remainder_nonzero(ratio)
                } else {
                    declaration.when_prefix_at_least(ratio)
                },
            );
        }
    }
    if tensors.is_empty() {
        LayerCachePolicy::key_only(AttentionPolicy::sliding(window).unwrap(), 1, 8).unwrap()
    } else {
        LayerCachePolicy::key_only_with_fixed_state(
            AttentionPolicy::sliding(window).unwrap(),
            1,
            8,
            tensors,
        )
        .unwrap()
    }
}
fn layout(policy: LayerCachePolicy) -> StateLayout {
    StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap()
}
fn step(cache: &mut WorkspacePoolingLayerState, tokens: i32, context: &WorkspaceContext) {
    let position = cache.offset();
    let values = WorkspaceTensor::full_f32(0.75, &[2, tokens, 8], context).unwrap();
    let gates = WorkspaceTensor::full_f32(-0.25, &[2, tokens, 8], context).unwrap();
    let local = cache.append_local(values.clone(), context).unwrap();
    assert_eq!(
        local.shape(),
        [2, position.min(cache.local.window() - 1) + tokens, 8]
    );
    assert_eq!(
        cache.local_mask(tokens, position, context).unwrap().shape(),
        [tokens, local.shape()[1]]
    );
    for stream in 0..cache.streams.len() as u32 {
        let ratio = cache.pooling_ratio(stream).unwrap();
        let ready = cache
            .accumulate_pooling_windows(stream, values.clone(), gates.clone(), position, context)
            .unwrap();
        let complete = ready.values.shape()[1];
        assert_eq!(complete, ((position % ratio + tokens) / ratio) * ratio);
        if complete > 0 && cache.stream(stream).unwrap().declarations[3].is_some() {
            let tail = |value: WorkspaceTensor| {
                value
                    .index(
                        &[
                            Index::Full,
                            Index::Range(complete - ratio, complete),
                            Index::Full,
                        ],
                        context,
                    )
                    .unwrap()
            };
            cache
                .replace_pooling_overlap(stream, tail(ready.values), tail(ready.gates))
                .unwrap();
        }
        let pooled = WorkspaceTensor::full_f32(1.25, &[2, complete / ratio, 8], context).unwrap();
        let history = cache.append_pooled(stream, pooled, context).unwrap();
        assert_eq!(history.shape(), [2, (position + tokens) / ratio, 8]);
        let mask = cache
            .pooling_mask(stream, tokens, position, context)
            .unwrap();
        assert_eq!(mask.is_some(), tokens != 1 && history.shape()[1] > 0);
    }
}

#[test]
fn pooling_workspace_tracks_partial_overlap_and_local_sentinel_through_uneven_chunks() {
    for streams in 0..=2 {
        for window in [1, 5] {
            let context = WorkspaceContext::new(Facts);
            let mut state =
                WorkspacePoolingStateFactory::new(NonZeroU32::new(2).unwrap(), &context)
                    .unwrap()
                    .realize(&layout(policy(streams, window)))
                    .unwrap();
            let cache = state.layer(0).unwrap();
            let mut position = 0;
            for count in [3, 4, 1, 5, 1, 1] {
                context.begin_state_span(cache.retained_values()).unwrap();
                step(cache, count, &context);
                position += count;
                assert_eq!(cache.offset(), position);
                let roots = cache.retained_values().cloned().collect::<Vec<_>>();
                let report = context.report(&roots).unwrap();
                assert!(report.inference_transient_bytes().is_some());
                assert!(report.state.as_ref().unwrap().retained_bytes.is_some());
                let retained_local = cache.local.values().collect::<Vec<_>>();
                assert_eq!(retained_local.len(), if window == 1 { 0 } else { 2 });
                if window != 1 {
                    assert_eq!(
                        retained_local[1].shape(),
                        [2, 1, (window as i32 - 1).min(position), 1]
                    );
                }
            }
            let saved = cache.checkpoint().unwrap();
            step(cache, 5, &context);
            cache.restore(&saved, &context).unwrap();
            assert_eq!(cache.offset(), position);
            step(cache, 2, &context);
            cache.clear().unwrap();
            assert_eq!(cache.offset(), 0);
            assert_eq!(cache.retained_values().count(), 0);
        }
    }
}

#[test]
fn pooling_projection_preserves_full_shared_backing_and_rejects_incomplete_geometry() {
    let context = WorkspaceContext::new(Facts);
    let factory = WorkspacePoolingStateFactory::new(NonZeroU32::new(2).unwrap(), &context).unwrap();
    let policy = policy(2, 5);
    let storage = WorkspaceExistingStorage::new(Some(4096), &context);
    let view = |shape: &[i32]| {
        WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
            &storage,
            &context,
        )
        .unwrap()
    };
    let keys = view(&[2, 1, 3, 8]);
    let sentinel = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[2, 1, 3, 1], WorkspaceDtype::Float32).unwrap(),
        &context,
    )
    .unwrap();
    let components = policy
        .fixed_state()
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
    let mut projected = factory
        .project_layer(
            0,
            &policy,
            3,
            Some(keys.clone()),
            Some(sentinel.clone()),
            components.clone(),
        )
        .unwrap();
    context
        .begin_state_span(projected.retained_values())
        .unwrap();
    let report = context
        .report(&projected.retained_values().cloned().collect::<Vec<_>>())
        .unwrap();
    assert_eq!(
        report.state.as_ref().unwrap().retained_bytes,
        Some(4096 + 24)
    );
    assert_eq!(report.tensor_buffers.total_bytes, Some(0));
    assert!(
        factory
            .project_layer(0, &policy, 3, Some(keys.clone()), None, components.clone())
            .is_err()
    );
    assert!(
        factory
            .project_layer(
                0,
                &policy,
                3,
                Some(keys.clone()),
                Some(sentinel.clone()),
                components.iter().skip(1).cloned()
            )
            .is_err()
    );
    assert!(
        factory
            .project_layer(0, &policy, 4, Some(keys), Some(sentinel), components)
            .is_err()
    );
    step(&mut projected, 4, &context);
    assert_eq!(projected.offset(), 7);
    let foreign = WorkspaceContext::new(Facts);
    assert!(projected.local_mask(1, 7, &foreign).is_err());
}

#[test]
fn pooling_restore_rejects_different_local_geometry_before_changing_state() {
    let context = WorkspaceContext::new(Facts);
    let factory = WorkspacePoolingStateFactory::new(NonZeroU32::new(2).unwrap(), &context).unwrap();
    let mut original = factory.create_layer(0, &policy(0, 5)).unwrap();
    step(&mut original, 3, &context);
    let incompatible =
        LayerCachePolicy::key_only(AttentionPolicy::sliding(5).unwrap(), 1, 16).unwrap();
    let other = factory.create_layer(0, &incompatible).unwrap();
    assert!(
        original
            .restore(&other.checkpoint().unwrap(), &context)
            .is_err()
    );
    assert_eq!(original.offset(), 3);
    let multiple_heads =
        LayerCachePolicy::key_only(AttentionPolicy::sliding(5).unwrap(), 2, 8).unwrap();
    assert!(crate::state::pooling_attention_geometry(0, &multiple_heads).is_err());
}
