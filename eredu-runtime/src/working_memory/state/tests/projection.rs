use super::*;

fn existing(
    shape: &[i32],
    dtype: WorkspaceDtype,
    bytes: Option<u64>,
    context: &WorkspaceContext,
) -> WorkspaceTensor {
    WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(shape, dtype).unwrap(),
        &WorkspaceExistingStorage::new(bytes, context),
        context,
    )
    .unwrap()
}

#[test]
fn projected_attention_keeps_physical_capacity_and_continues_at_the_real_frontier() {
    for window in [None, Some(4), Some(1)] {
        for key_only in [false, true] {
            let (context, _) = context();
            let factory =
                WorkspaceConcatStateFactory::new(NonZeroU32::new(2).unwrap(), &context).unwrap();
            let attention = window.map_or(AttentionPolicy::Full, |n| {
                AttentionPolicy::sliding(n).unwrap()
            });
            let policy = if key_only {
                LayerCachePolicy::key_only(attention, 2, 8).unwrap()
            } else {
                LayerCachePolicy::key_value(attention, 2, 8).unwrap()
            };
            let count = window.map_or(7, |n| 7.min(n - 1)) as i32;
            let keys = (count > 0).then(|| {
                existing(
                    &[2, 2, count, 8],
                    WorkspaceDtype::Float32,
                    Some(4096),
                    &context,
                )
            });
            // Keys and values may be aliases of the same backing allocation.
            let values = if key_only { None } else { keys.clone() };
            let mut layer = factory
                .project_layer(0, &policy, 7, keys, values, [])
                .unwrap();
            context.begin_state_span(layer.retained_values()).unwrap();
            let before = context.report(&roots(&layer)).unwrap();
            assert!(before.operations.is_empty());
            assert_eq!(
                before.state.unwrap().retained_bytes,
                Some(if count == 0 { 0 } else { 4096 })
            );
            let (keys, values) = self::super::values(2, &context);
            let (visible, _) = layer.update_for_attention(keys, values, &context).unwrap();
            assert_eq!(visible.shape(), [2, 2, count + 2, 8]);
            assert_eq!(layer.position(), 9);
            let after = context.report(&roots(&layer)).unwrap();
            assert_eq!(
                after.state.unwrap().displaced_bytes,
                Some(if count == 0 { 0 } else { 4096 })
            );
        }
    }
}

#[test]
fn projected_fixed_slots_keep_absence_and_unknown_capacity_without_replay() {
    let (context, _) = context();
    let factory = WorkspaceConcatStateFactory::new(NonZeroU32::new(2).unwrap(), &context).unwrap();
    let policy = LayerCachePolicy::FixedState {
        tensors: vec![fixed()],
    };
    let role = StateTensorRole::Convolution { slot: 3 };
    let mut empty = factory
        .project_layer(0, &policy, 7, None, None, [(role, None)])
        .unwrap();
    assert!(empty.convolution_state(3).unwrap().is_none());
    assert_eq!(empty.position(), 7);
    let value = existing(&[2, 3, 8], WorkspaceDtype::Float32, None, &context);
    let layer = factory
        .project_layer(0, &policy, 7, None, None, [(role, Some(value))])
        .unwrap();
    context.begin_state_span(layer.retained_values()).unwrap();
    let report = context.report(&roots(&layer)).unwrap();
    assert!(report.operations.is_empty());
    assert_eq!(report.inference_transient_bytes(), None);
}

#[test]
fn projected_state_rejects_foreign_geometry_roles_and_dtype_before_equations() {
    let (context, _) = context();
    let (foreign, _) = super::context();
    let factory = WorkspaceConcatStateFactory::new(NonZeroU32::new(2).unwrap(), &context).unwrap();
    let policy = LayerCachePolicy::FixedState {
        tensors: vec![fixed()],
    };
    let role = StateTensorRole::Convolution { slot: 3 };
    for slots in [
        vec![],
        vec![(role, None), (role, None)],
        vec![(StateTensorRole::Recurrent, None)],
        vec![(
            role,
            Some(existing(
                &[1, 3, 8],
                WorkspaceDtype::Float32,
                Some(1000),
                &context,
            )),
        )],
        vec![(
            role,
            Some(existing(
                &[2, 3, 8],
                WorkspaceDtype::Int32,
                Some(1000),
                &context,
            )),
        )],
        vec![(
            role,
            Some(existing(
                &[2, 3, 8],
                WorkspaceDtype::Float32,
                Some(1000),
                &foreign,
            )),
        )],
    ] {
        assert!(factory
            .project_layer(0, &policy, 7, None, None, slots)
            .is_err());
    }
    let attention = LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap();
    let wrong = existing(&[2, 2, 6, 8], WorkspaceDtype::Float32, Some(1000), &context);
    assert!(factory
        .project_layer(0, &attention, 7, Some(wrong.clone()), Some(wrong), [])
        .is_err());
    assert!(context.report(&[]).unwrap().operations.is_empty());
}
