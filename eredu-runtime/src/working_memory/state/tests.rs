use super::*;
use crate::{LayerRuntimeState, ResettableRuntimeState, RuntimeState};
use eredu_core::{AttentionPolicy, LayerSchedule, cache::*};
use eredu_nn::workspace::*;
use std::{cell::Cell, rc::Rc};

mod paged;
mod projection;

#[derive(Debug)]
struct Facts {
    fail_after: Rc<Cell<Option<usize>>>,
}
impl WorkspaceMechanisms for Facts {
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "fixture has no disjoint host workspace".into(),
        }))
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if let Some(remaining) = self.fail_after.get() {
            if remaining == 0 {
                return Err(Error::backend("injected allocation fact failure"));
            }
            self.fail_after.set(Some(remaining - 1));
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
            assumptions: "fixture logical allocation and aliasing slice".into(),
        }))
    }
}
fn context() -> (WorkspaceContext, Rc<Cell<Option<usize>>>) {
    let fail_after = Rc::new(Cell::new(None));
    (
        WorkspaceContext::new(Facts {
            fail_after: fail_after.clone(),
        }),
        fail_after,
    )
}
fn layout(policy: LayerCachePolicy) -> StateLayout {
    StateLayout::new(LayerSchedule::new(1, vec![policy]).unwrap()).unwrap()
}
fn make(
    policy: LayerCachePolicy,
    context: &WorkspaceContext,
) -> DeviceState<WorkspaceBackend, WorkspaceConcatLayerState> {
    WorkspaceConcatStateFactory::new(NonZeroU32::new(2).unwrap(), context)
        .unwrap()
        .realize(&layout(policy))
        .unwrap()
}
fn values(count: i32, context: &WorkspaceContext) -> (WorkspaceTensor, WorkspaceTensor) {
    let x = WorkspaceTensor::unloaded_f32(&[2, 2, count, 8], context).unwrap();
    (x.square(context).unwrap(), x.tanh(context).unwrap())
}
fn roots(layer: &WorkspaceConcatLayerState) -> Vec<WorkspaceTensor> {
    layer.retained_values().cloned().collect()
}

#[test]
fn sliding_views_keep_complete_append_backing_and_key_only_shares_storage() {
    for key_only in [false, true] {
        for window in [1, 4] {
            let (context, _) = context();
            let attention = AttentionPolicy::sliding(window).unwrap();
            let policy = if key_only {
                LayerCachePolicy::key_only(attention, 2, 8).unwrap()
            } else {
                LayerCachePolicy::key_value(attention, 2, 8).unwrap()
            };
            let mut state = make(policy, &context);
            let layer = state.layer(0).unwrap();
            let (keys, values) = values(7, &context);
            let (visible_keys, visible_values) =
                layer.update_for_attention(keys, values, &context).unwrap();
            assert_eq!(visible_keys.shape(), [2, 2, 7, 8]);
            assert_eq!(visible_values.shape(), visible_keys.shape());
            assert_eq!(layer.offset(), 7);
            let report = context.report(&roots(layer)).unwrap();
            let buffers = if key_only { 1 } else { 2 };
            if window == 1 {
                assert!(roots(layer).is_empty());
                assert_eq!(report.retained_bytes, Some(0));
            } else {
                assert!(roots(layer).iter().all(|v| v.shape() == [2, 2, 3, 8]));
                assert_eq!(roots(layer).len(), buffers as usize);
                assert_eq!(report.retained_bytes, Some(buffers * 2 * 2 * 7 * 8 * 4));
                assert!(
                    report.retained_bytes.unwrap()
                        > roots(layer)
                            .iter()
                            .map(|v| v.layout().bytes().unwrap())
                            .sum::<u64>()
                );
            }
            context.begin_span();
            let (keys, values) = self::values(2, &context);
            let (keys, _) = layer.update_for_attention(keys, values, &context).unwrap();
            assert_eq!(keys.shape()[2], if window == 1 { 2 } else { 5 });
            assert_eq!(layer.offset(), 9);
            let report = context.report(&roots(layer)).unwrap();
            assert_eq!(
                report.retained_bytes,
                Some(if window == 1 {
                    0
                } else {
                    buffers * 2 * 2 * 5 * 8 * 4
                })
            );
        }
    }
}

#[test]
fn failed_append_preserves_frontier_and_roots_without_refunding_prior_work() {
    let (context, fail_after) = context();
    let mut state = make(
        LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap(),
        &context,
    );
    let layer = state.layer(0).unwrap();
    let (k, v) = values(2, &context);
    layer.update_for_attention(k, v, &context).unwrap();
    context.begin_span();
    let (k, v) = values(3, &context);
    let baseline = context.report(&roots(layer)).unwrap().total_bytes.unwrap();
    fail_after.set(Some(1)); // K concatenation succeeds, V concatenation fails.
    assert!(
        layer
            .update_for_attention(k.clone(), v.clone(), &context)
            .is_err()
    );
    assert_eq!(layer.offset(), 2);
    assert!(roots(layer).iter().all(|v| v.shape()[2] == 2));
    assert_eq!(
        context.report(&roots(layer)).unwrap().retained_bytes,
        Some(0)
    );
    assert!(context.report(&roots(layer)).unwrap().total_bytes.unwrap() > baseline);
    fail_after.set(None);
    layer.update_for_attention(k, v, &context).unwrap();
    assert_eq!(layer.offset(), 5);
    let report = context.report(&roots(layer)).unwrap();
    assert_eq!(report.retained_bytes, Some(2 * 2 * 2 * 5 * 8 * 4));
    assert!(report.transient_bytes.unwrap() >= baseline + 2 * 2 * 5 * 8 * 4);
}

#[test]
fn geometry_and_foreign_context_fail_before_state_or_ledger_changes() {
    let (context, _) = context();
    let (foreign, _) = self::context();
    let mut state = make(
        LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap(),
        &context,
    );
    let layer = state.layer(0).unwrap();
    let (k, v) = values(2, &context);
    let before = context.report(&[]).unwrap().total_bytes;
    // Malformed inputs are imported metadata; rejection must add no work.
    let bad = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[1, 2, 2, 8], WorkspaceDtype::Float32).unwrap(),
        &context,
    )
    .unwrap();
    assert!(
        layer
            .update_for_attention(bad, v.clone(), &context)
            .is_err()
    );
    assert!(
        layer
            .update_for_attention(k.clone(), v.clone(), &foreign)
            .is_err()
    );
    let bad = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[2, 2, 2, 8], WorkspaceDtype::Float32).unwrap(),
        &foreign,
    )
    .unwrap();
    assert!(
        layer
            .update_for_attention(bad, v.clone(), &context)
            .is_err()
    );
    assert_eq!(layer.offset(), 0);
    assert!(roots(layer).is_empty());
    assert_eq!(context.report(&[]).unwrap().total_bytes, before);
    layer.position = i32::MAX;
    assert!(layer.update_for_attention(k, v, &context).is_err());
    assert_eq!(layer.offset(), i32::MAX);
}

fn fixed() -> StateTensorPolicy {
    StateTensorPolicy::new(
        StateTensorRole::Convolution { slot: 3 },
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(3).unwrap(),
            StateTensorDimension::fixed(8).unwrap(),
        ],
        StateTensorDtype::Floating,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap()
}
#[test]
fn fixed_slots_and_segment_reset_preserve_policy_and_do_not_refund_storage() {
    let (context, _) = context();
    let mut state = make(
        LayerCachePolicy::FixedState {
            tensors: vec![fixed()],
        },
        &context,
    );
    let segment = state.layout().segments()[0].id().clone();
    let layer = state.layer(0).unwrap();
    assert!(layer.convolution_state(2).is_err());
    let x = WorkspaceTensor::unloaded_f32(&[2, 3, 8], &context)
        .unwrap()
        .square(&context)
        .unwrap();
    *layer.convolution_state(3).unwrap() = Some(x);
    layer.advance_fixed(7).unwrap();
    assert!(layer.advance_fixed(0).is_err());
    assert_eq!(layer.position(), 7);
    let before = context.report(&roots(layer)).unwrap();
    let mut snapshot = state.clone();
    state.reset_segment(&segment).unwrap();
    assert_eq!(state.layout(), snapshot.layout());
    assert_eq!(state.layer(0).unwrap().position(), 0);
    assert!(state.layer(0).unwrap().retained_values().next().is_none());
    assert!(
        state
            .layer(0)
            .unwrap()
            .convolution_state(3)
            .unwrap()
            .is_none()
    );
    assert_eq!(context.report(&[]).unwrap().total_bytes, before.total_bytes);
    assert_eq!(snapshot.layer(0).unwrap().position(), 7);
}

#[test]
fn attention_plus_components_advances_once_and_wrong_cache_mechanism_is_typed() {
    let (context, _) = context();
    let mut state = make(
        LayerCachePolicy::KeyValueWithFixedState {
            attention: AttentionPolicy::Full,
            num_key_value_heads: NonZeroU32::new(2).unwrap(),
            head_dim: NonZeroU32::new(8).unwrap(),
            tensors: vec![fixed()],
        },
        &context,
    );
    let layer = state.layer(0).unwrap();
    assert!(layer.advance_fixed(1).is_err());
    let (k, v) = values(3, &context);
    layer.update_for_attention(k, v, &context).unwrap();
    assert_eq!(layer.position(), 3);
    assert!(layer.convolution_state(3).unwrap().is_none());
    let compressed = layout(LayerCachePolicy::CompressedLatentRotary {
        attention: AttentionPolicy::Full,
        latent_dim: NonZeroU32::new(8).unwrap(),
        rotary_dim: NonZeroU32::new(4).unwrap(),
    });
    let result = WorkspaceConcatStateFactory::new(NonZeroU32::new(2).unwrap(), &context)
        .unwrap()
        .realize(&compressed);
    assert!(matches!(
        result,
        Err(StateError::InvalidLayer { layer: 0, .. })
    ));
}

#[test]
fn auxiliary_convolution_errors_retain_roles_after_workspace_owners_drop() {
    use crate::working_memory::WorkspaceResidentLayerState;
    use std::error::Error as _;
    let (context, fail_after) = context();
    // A missing slot needs no tensor operation or backend mechanism fact.
    fail_after.set(Some(0));
    let factory = WorkspaceConcatStateFactory::new(NonZeroU32::new(1).unwrap(), &context).unwrap();
    let mut ordinary = factory.create_layer(0, &LayerCachePolicy::NoState).unwrap();
    let mut resident = WorkspaceResidentLayerState::Ordinary(ordinary.clone());
    let mut errors = Vec::new();
    for slot in 0..4 {
        errors.push((slot, ordinary.convolution_state(slot).unwrap_err()));
        errors.push((slot, resident.convolution_state(slot).unwrap_err()));
    }
    assert_eq!(ordinary.position(), 0);
    assert_eq!(resident.position(), 0);
    drop((ordinary, resident, factory, context));
    for (slot, error) in errors {
        let cloned = error.clone();
        drop(error);
        let role = StateTensorRole::Convolution { slot };
        assert!(matches!(
            cloned.source().and_then(|cause| cause.downcast_ref::<StateError>()),
            Some(StateError::UnknownComponent { role: actual }) if *actual == role
        ));
        assert_eq!(
            cloned.to_string(),
            StateError::UnknownComponent { role }.to_string()
        );
    }
}

#[derive(Debug)]
struct MetadataOnlyFacts;
impl WorkspaceMechanisms for MetadataOnlyFacts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        unreachable!("this fixture exercises state metadata without tensor execution")
    }
}
impl WorkspaceFactMechanisms for MetadataOnlyFacts {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        unreachable!()
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        unreachable!()
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        unreachable!()
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        unreachable!()
    }
}

#[test]
fn fixed_checkpoint_copy_is_charged_before_mutation_and_preserves_saved_slots() {
    fn make(context: &WorkspaceContext) -> WorkspaceConcatLayerState {
        let factory =
            WorkspaceConcatStateFactory::new(NonZeroU32::new(2).unwrap(), context).unwrap();
        let mut layer = factory
            .create_layer(
                0,
                &LayerCachePolicy::FixedState {
                    tensors: vec![fixed()],
                },
            )
            .unwrap();
        let value = WorkspaceTensor::existing(
            context.layout(&[2, 3, 8], WorkspaceDtype::Float32).unwrap(),
            context,
        )
        .unwrap();
        *layer
            .fixed_component(StateTensorRole::Convolution { slot: 3 })
            .unwrap() = Some(value);
        layer.advance_fixed(7).unwrap();
        layer
    }
    let recording = WorkspaceContext::new_recording_facts(MetadataOnlyFacts);
    let mut live = make(&recording);
    let construction = recording.metadata_census().unwrap();
    let saved = live.clone();
    assert_eq!(recording.metadata_census().unwrap(), construction);
    *live
        .fixed_component(StateTensorRole::Convolution { slot: 3 })
        .unwrap() = None;
    assert!(
        saved
            .fixed
            .get(&StateTensorRole::Convolution { slot: 3 })
            .unwrap()
            .is_some()
    );
    assert!(
        live.fixed
            .get(&StateTensorRole::Convolution { slot: 3 })
            .unwrap()
            .is_none()
    );
    assert!(recording.metadata_census().unwrap().context_bytes() > construction.context_bytes());

    let bounded = WorkspaceContext::new_with_metadata_budget(MetadataOnlyFacts, construction);
    let mut live = make(&bounded);
    let saved = live.clone();
    let before = bounded.metadata_census().unwrap();
    let failure = live.reset().unwrap_err();
    let StateError::WorkspaceConstruction(cause) = failure else {
        panic!("unexpected state error");
    };
    use std::error::Error as _;
    assert!(matches!(
        cause
            .source()
            .and_then(|source| source.downcast_ref::<WorkspaceMetadataError>()),
        Some(WorkspaceMetadataError::Capacity { .. })
    ));
    assert_eq!(live.position(), 7);
    assert!(
        live.fixed
            .get(&StateTensorRole::Convolution { slot: 3 })
            .unwrap()
            .is_some()
    );
    assert!(
        saved
            .fixed
            .get(&StateTensorRole::Convolution { slot: 3 })
            .unwrap()
            .is_some()
    );
    assert_eq!(bounded.metadata_census().unwrap(), before);
}
