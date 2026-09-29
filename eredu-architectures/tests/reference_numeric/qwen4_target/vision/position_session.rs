//! Media position state through selected sessions, rollback and control-state forks.
use super::*;
use eredu_architectures::qwen4_exp::position::PositionError;
use eredu_core::cache::StateTensorRole;
use eredu_runtime::replicated_session::ReplicatedTextSnapshotMechanisms;
use eredu_runtime::*;

// The ordinary numeric backend clones all fixed, recurrent and append-stream values.
impl<A> ReplicatedTextSnapshotMechanisms<A, NumericBackend> for NumericReplicatedMechanisms
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
{
    fn estimate_snapshot_state(
        &self,
        state: &State,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let bytes = state
            .as_ref()
            .iter()
            .flat_map(RuntimeLayerState::retained_values)
            .try_fold(std::mem::size_of::<State>() as u64, |sum, t| {
                sum.checked_add(std::mem::size_of::<NumericTensor>() as u64)?
                    .checked_add(t.data.len() as u64 * 4)?
                    .checked_add(t.exact_i32.as_ref().map_or(0, |v| v.len() as u64 * 4))
            })?;
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }
    fn copy_snapshot_state(&mut self, state: &State, _: &NumericContext) -> Result<State, Error> {
        Ok(state.clone())
    }
}
struct FailAfterIngress;
impl ActivationObserver<NumericTensor, Error> for FailAfterIngress {
    fn observe(&mut self, path: &str, _: &NumericTensor) -> Result<(), Error> {
        if path == "model.layers.0.mixer.write" {
            Err(Error::backend("injected post-ingress failure"))
        } else {
            Ok(())
        }
    }
}
fn delta(state: &State) -> Option<i32> {
    state.as_ref()[0].fixed[&StateTensorRole::PositionDelta]
        .as_ref()
        .map(|t| {
            assert_eq!(t.element_type(), Some(eredu_nn::TensorElementType::I32));
            assert_eq!(t.shape(), [1]);
            t.to_i32_vec(&NumericContext::default()).unwrap()[0]
        })
}
fn position_error<'a>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a eredu_architectures::qwen4_exp::position::PositionError> {
    loop {
        if let Some(error) = error.downcast_ref() {
            return Some(error);
        }
        error = error.source()?;
    }
}
#[test]
fn selected_media_target_sessions_restore_fork_and_reset_persistent_rotary_position() {
    use eredu_nn::EmbeddingOperator;
    let (_dir, target, ingress) = setup();
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let inspector = Inspector(0.into());
    let input = prompt(&inspector);
    let identity = input
        .cache_identity("qwen4-media-position-fixture-v1")
        .unwrap();
    let prepared = ingress
        .admission_config()
        .admit(&input, &inspector)
        .and_then(|admitted| admitted.prepare(&ctx))
        .unwrap();
    let mut tower = tower(&ingress, &ctx);
    let output = prepared
        .with_vision_input(|v| tower.forward(v.unwrap(), &ctx))
        .unwrap()
        .embeddings;
    let mut embedding =
        TargetModel::<NumericBackend>::new(target.bound_spec().unwrap(), &ctx).unwrap();
    let parameters =
        <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(
            &mut embedding,
        );
    parameters.visit_parameters_mut(&mut TargetBind {
        ordinary: target.static_parameters(),
        experts: vec![],
        ctx: &ctx,
    });
    let media = prepared
        .assemble(
            0..prepared.token_ids().dim(1),
            Some(&output),
            |ids| parameters.embeddings.forward(ids, &ctx),
            &ctx,
        )
        .unwrap();
    let mut reference: Option<Vec<NumericTensor>> = None;
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let plan = super::super::super::gguf::plan(&target);
        let mechanisms = super::super::super::cold::capabilities(plan.requirements(), None);
        let request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
            ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
                .with_session(eredu_core::SessionCapabilities::new(false, true, true))
                .with_exact_completion(true),
            WeightResidency::with_layers(residency),
        )
        .unwrap();
        let selected = plan.select(&request, &mechanisms, None).unwrap();
        let (handoff, source) = selected.prepare::<NumericBackend, State>(&ctx).unwrap();
        let mut mechanisms = if residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(source)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(source)
        };
        mechanisms.fixture_state_factory = Some(super::super::super::gguf::state_from_layout);
        macro_rules! exercise {
            ($session:expr) => {{
                let mut session = $session;
                let initial = session.checkpoint(&ctx).unwrap();
                assert_eq!(delta(&initial), None);
                // Failure after lexical ingress must undo both its histories and delta.
                let failed = media.with_target_input(|input| {
                    session.prefill_input_with_observer(input, &ctx, &mut FailAfterIngress)
                });
                assert!(failed.is_err());
                super::super::super::session::assert_checkpoint(
                    &session.checkpoint(&ctx).unwrap(),
                    &initial,
                );
                let mut outputs = Vec::new();
                for range in [0..1, 1..4, 4..9, 9..18] {
                    let part = media.slice(range, &ctx).unwrap();
                    let output = part
                        .with_target_input(|input| {
                            session.prefill_input_with_cache_identity(input, identity.clone(), &ctx)
                        })
                        .unwrap();
                    outputs.push(output);
                    assert_eq!(delta(&session.checkpoint(&ctx).unwrap()), Some(-2));
                }
                let token = NumericTensor::from_i32_slice(&[3], &[1, 1], &ctx).unwrap();
                let saved = session.checkpoint(&ctx).unwrap();
                let next = session.decode(&token, &ctx).unwrap();
                session.rollback(saved.clone(), &ctx).unwrap();
                assert_tensor_exact(
                    &session.decode(&token, &ctx).unwrap(),
                    &next,
                    "media checkpoint replay",
                );
                session.rollback(saved.clone(), &ctx).unwrap();
                // A reused control snapshot remains immutable while either child advances.
                let frozen = session.capture_control_state(&ctx).unwrap();
                let mut child = session.copy_control_state(&frozen, &ctx).unwrap();
                let first = session.decode(&token, &ctx).unwrap();
                session.exchange_control_state(&mut child, &ctx).unwrap();
                assert_tensor_exact(
                    &session.decode(&token, &ctx).unwrap(),
                    &first,
                    "media control fork",
                );
                let mut restored = session.copy_control_state(&frozen, &ctx).unwrap();
                session.exchange_control_state(&mut restored, &ctx).unwrap();
                for step in 0..16 {
                    let token = NumericTensor::from_i32_slice(&[step % 12], &[1, 1], &ctx).unwrap();
                    outputs.push(session.decode(&token, &ctx).unwrap());
                }
                let before = session.checkpoint(&ctx).unwrap();
                // A changed delta fails before execution and the shared transaction restores.
                let changed = TargetInput {
                    position_delta: Some(0),
                    ids: Some(OriginalTokenIds::Tensor(&token)),
                    batch: 1,
                    tokens: 1,
                    embeddings: None,
                    visible: None,
                    rotary: None,
                };
                let error = session.decode_input(changed, &ctx).unwrap_err();
                assert!(matches!(
                    position_error(&error),
                    Some(eredu_architectures::qwen4_exp::position::PositionError::Changed)
                ));
                super::super::super::session::assert_checkpoint(
                    &session.checkpoint(&ctx).unwrap(),
                    &before,
                );
                for (malformed, expected) in [
                    (None, PositionError::Missing),
                    (
                        Some(NumericTensor::new([1], vec![-2.])),
                        PositionError::Geometry,
                    ),
                    (
                        Some(NumericTensor::from_i32_slice(&[-2], &[1, 1], &ctx).unwrap()),
                        PositionError::Geometry,
                    ),
                    (
                        Some(NumericTensor::from_i32_slice(&[i32::MIN], &[1], &ctx).unwrap()),
                        PositionError::Overflow,
                    ),
                    (
                        Some(NumericTensor::from_i32_slice(&[i32::MAX], &[1], &ctx).unwrap()),
                        PositionError::Overflow,
                    ),
                ] {
                    let mut corrupt = before.clone();
                    *corrupt
                        .layer(0)
                        .unwrap()
                        .fixed_component(StateTensorRole::PositionDelta)
                        .unwrap() = malformed;
                    session.rollback(corrupt.clone(), &ctx).unwrap();
                    let error = session.decode(&token, &ctx).unwrap_err();
                    assert_eq!(
                        std::mem::discriminant(position_error(&error).unwrap()),
                        std::mem::discriminant(&expected),
                    );
                    super::super::super::session::assert_checkpoint(
                        &session.checkpoint(&ctx).unwrap(),
                        &corrupt,
                    );
                    session.rollback(before.clone(), &ctx).unwrap();
                }
                session.reset(&ctx).unwrap();
                assert_eq!(delta(&session.checkpoint(&ctx).unwrap()), None);
                session.prefill(&token, None, &ctx).unwrap();
                assert_eq!(delta(&session.checkpoint(&ctx).unwrap()), Some(0));
                Ok::<_, String>(outputs)
            }};
        }
        let outputs =
            eredu_architectures::prepared_execution::construct_selected_routed_session(
                handoff,
                mechanisms,
                &ctx,
                |_, _, rows| {
                    let rows = rows.unwrap();
                    let providers = rows
                        .prepared()
                        .bind(
                            rows.prepared()
                                .entries()
                                .iter()
                                .map(|(id, e)| {
                                    (
                                        id.clone(),
                                        super::super::super::row_bank::SourceRows::new(e),
                                    )
                                })
                                .collect(),
                        )
                        .unwrap();
                    Ok::<_, String>((
                        BTreeMap::<
                            RoutedBankId,
                            (NumericGroupedBankMechanism, NumericIndexedMovement),
                        >::new(),
                        Some(providers),
                    ))
                },
                (),
                |_, s, _| exercise!(s),
                |_, s, _| exercise!(s),
            )
            .unwrap();
        if let Some(reference) = &reference {
            for (a, b) in outputs.iter().zip(reference) {
                assert_tensor_close(a, b, "media selected residency parity");
            }
        } else {
            reference = Some(outputs);
        }
    }
}
