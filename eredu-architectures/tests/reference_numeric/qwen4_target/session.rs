//! Exact text input through the existing selected session and transaction driver.
use super::*;
use eredu_nn::NeuralOperatorCapabilities;
use eredu_runtime::*;

type Session = ReplicatedTextSession<
    TargetModel<NumericBackend>,
    NumericBackend,
    NumericReplicatedMechanisms,
    RoutedReplicatedTextExecution<ParameterProviders<ResidentExpertProvider, Rows>>,
>;

fn setup(
    ctx: &NumericContext,
    architecture: TargetModel<NumericBackend>,
    prompt_cache: bool,
) -> (
    TargetModel<NumericBackend>,
    ReplicatedTextRequirements,
    BackendMechanismCapabilities,
    String,
) {
    let graph = <TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::execution_graph(&architecture).unwrap();
    let layout = specification().state_layout().unwrap();
    let description = architecture.parameter_description(ctx).unwrap();
    let partition = PartitionState::new(layout.clone(), 0).unwrap();
    let identity = architecture
        .state_identity(&partition, Default::default())
        .unwrap()
        .prompt_cache_identity(&layout)
        .unwrap()
        .architecture_fingerprint()
        .to_owned();
    let source =
        eredu_checkpoint::SourceTensorEncoding::Safetensors(eredu_checkpoint::StoredDtype::F32);
    let parameters = description
        .groups()
        .iter()
        .flat_map(|group| {
            let owner = match group.owner() {
                ParameterGroupOwner::StaticRole(role) => {
                    ReplicatedTextParameterOwner::StaticRole(role.clone())
                }
                ParameterGroupOwner::ExecutionUnit {
                    group, global_unit, ..
                } => ReplicatedTextParameterOwner::ExecutionUnit {
                    group: group.as_str().into(),
                    unit: *global_unit,
                },
                _ => panic!("unexpected fixture owner"),
            };
            group
                .members()
                .iter()
                .map(move |member| (member, owner.clone()))
        })
        .map(|(member, owner)| {
            let name = member.target();
            let shape = member.global_shape().to_vec();
            ReplicatedTextParameterRequirement::new(
                name,
                vec![name.into()],
                vec![ReplicatedTextPhysicalSource::new(
                    name,
                    name,
                    "fixture.safetensors",
                    name,
                    source.clone(),
                    shape.iter().product::<usize>() as u64 * 4,
                )
                .unwrap()],
                vec![],
                Some(source.clone()),
                Some(shape.clone()),
                shape,
                eredu_checkpoint::LinearFormat::Dense,
                if name.contains(".mlp.experts.") {
                    ReplicatedTextParameterRole::LinearWeight
                } else {
                    ReplicatedTextParameterRole::Other
                },
                owner,
                ReplicatedTextParameterPresence::Required,
                ParameterTransformConstraint::None,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let lowerings = parameters
        .iter()
        .map(|p| {
            WeightLoweringCapability::new(
                p.lowering_descriptor(eredu_checkpoint::LinearFormat::Dense)
                    .unwrap(),
                WeightLoweringKind::Direct,
            )
        })
        .collect();
    let units = ExecutionUnitLayout::new(&graph, [layout.len()]).unwrap();
    let requirements = ReplicatedTextRequirements::new(
        identity.clone(),
        NeuralOperatorCapabilities::NONE,
        graph,
        units,
        vec![<TargetModel<NumericBackend> as LayeredArchitecture<
            NumericBackend,
            State,
        >>::group_transport(&architecture, 0)],
        layout.clone(),
        ReplicatedTextStateAccess::AttentionWithStreams,
        parameters,
    )
    .unwrap()
    .with_floating_state_source(eredu_core::checkpoint::TensorDtype::F32)
    .with_chunked_prefill(true)
    .with_append_streams(stream_bindings(&specification()))
    .unwrap();
    let components: Vec<_> = layout
        .layers()
        .iter()
        .enumerate()
        .flat_map(|(layer, policy)| {
            policy.components().into_iter().map(move |component| {
                StateComponentMechanism::new(
                    layer,
                    component,
                    Some(StateComponentPlacement::Device),
                    None,
                )
            })
        })
        .collect();
    let capabilities = eredu_core::SessionCapabilities::new(false, true, true);
    let mechanisms = BackendMechanismCapabilities::new(
        NeuralOperatorCapabilities::NONE,
        lowerings,
        vec![WeightResidencyMechanism::Resident],
        StateMechanismCapabilities::new(components)
            .with_floating_state_dtype(
                eredu_core::checkpoint::TensorDtype::F32,
                StateStorageDtype::F32,
            )
            .with_transactions(true, true)
            .with_reset(true)
            .with_prompt_cache(prompt_cache)
            .with_observation_retention(true),
    )
    .with_session(capabilities)
    .with_prompt_cache(prompt_cache)
    .with_exact_completion(true)
    .with_chunked_prefill(true);
    (architecture, requirements, mechanisms, identity)
}
pub(super) fn session(ctx: &NumericContext) -> Session {
    session_with_bound(bind_spec(specification()), ctx, false)
}
fn session_with_bound(bound: BoundTargetSpec, ctx: &NumericContext, prompt_cache: bool) -> Session {
    let architecture = TargetModel::new(bound, ctx).unwrap();
    let (architecture, requirements, mechanisms, identity) = setup(ctx, architecture, prompt_cache);
    let capabilities = eredu_core::SessionCapabilities::new(false, true, true);
    let selected = select_replicated_text_realization(
        &requirements,
        &ReplicatedTextSelectionRequest::new(
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
        )
        .with_session(capabilities)
        .with_prompt_cache(prompt_cache)
        .with_exact_completion(true),
        &mechanisms,
    )
    .unwrap();
    let contract = prepare_replicated_text_contract::<_, NumericBackend, State>(
        &architecture,
        None,
        selected,
        &identity,
        ctx,
    )
    .unwrap();
    fn initialize(metadata: ParameterMetadata, value: &mut NumericTensor) {
        Parameters::default().visit_mut(metadata, value);
    }
    fn make_state(layout: &StateLayout) -> Result<State, Error> {
        let spec = specification();
        assert_eq!(*layout, spec.state_layout().unwrap());
        Ok(state(&spec))
    }
    construct_replicated_text_session_with_execution(
        architecture,
        None,
        contract,
        NumericReplicatedMechanisms {
            fixture_parameter_init: Some(initialize),
            fixture_state_factory: Some(make_state),
            ..Default::default()
        },
        RoutedReplicatedTextExecution::new(ParameterProviders {
            grouped: ResidentExpertProvider,
            rows: Rows::default(),
        }),
        ctx,
    )
    .unwrap()
}

pub(super) fn assert_checkpoint(actual: &State, expected: &State) {
    assert_state_exact(
        actual,
        expected,
        actual.layout().len(),
        "target session checkpoint",
    );
    for (actual, expected) in actual.as_ref().iter().zip(expected.as_ref()) {
        for (a, b) in RuntimeLayerState::retained_values(actual)
            .zip(RuntimeLayerState::retained_values(expected))
        {
            assert_eq!(a.dtype, b.dtype);
            assert_eq!(a.exact_i32, b.exact_i32);
        }
        assert_eq!(actual.streams.len(), expected.streams.len());
        for ((slot, lane, a), (other_slot, other_lane, b)) in
            actual.streams.iter().zip(&expected.streams)
        {
            assert_eq!((slot, lane, a.len()), (other_slot, other_lane, b.len()));
        }
    }
}

#[test]
fn qwen4_shared_session_exact_text_chunk_decode_and_transaction_restore() {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let mut whole = session(&ctx);
    let mut chunks = session(&ctx);
    let ids = [3, 4, 7, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let tokens = NumericTensor::from_i32_slice(&ids, &[1, 19], &ctx).unwrap();
    let visible = NumericTensor::new(
        [1, 19],
        (0..19)
            .map(|i| if i < 2 || i == 8 { 0. } else { 1. })
            .collect(),
    );
    let expected = whole.prefill(&tokens, Some(&visible), &ctx).unwrap();
    let mut actual = None;
    let mut start = 0;
    for length in [1, 2, 4, 3, 9] {
        actual = Some(
            chunks
                .prefill(
                    &NumericTensor::from_i32_slice(
                        &ids[start..start + length],
                        &[1, length as i32],
                        &ctx,
                    )
                    .unwrap(),
                    Some(&NumericTensor::new(
                        [1, length as i32],
                        visible.data[start..start + length].to_vec(),
                    )),
                    &ctx,
                )
                .unwrap(),
        );
        start += length;
    }
    assert_tensor_close(&actual.unwrap(), &expected, "exact text session chunks");
    for i in 0..16 {
        let token = NumericTensor::from_i32_slice(&[i % 29], &[1, 1], &ctx).unwrap();
        assert_tensor_close(
            &chunks.decode(&token, &ctx).unwrap(),
            &whole.decode(&token, &ctx).unwrap(),
            "cached text session",
        );
    }
    let saved = chunks.checkpoint(&ctx).unwrap();
    let before = chunks.execution_strategy().provider().rows.calls.len();
    let token = NumericTensor::from_i32_slice(&[17], &[1, 1], &ctx).unwrap();
    let mut fail = Observe {
        paths: vec![],
        fail: true,
    };
    assert!(chunks
        .decode_with_observer(&token, &ctx, &mut fail)
        .is_err());
    assert!(chunks.execution_strategy().provider().rows.calls.len() > before);
    assert_checkpoint(&chunks.checkpoint(&ctx).unwrap(), &saved);
    let after_failure = chunks.execution_strategy().provider().rows.calls.len();
    let actual = chunks.decode(&token, &ctx).unwrap();
    let after_success = chunks.checkpoint(&ctx).unwrap();
    chunks.rollback(saved.clone(), &ctx).unwrap();
    assert_checkpoint(&chunks.checkpoint(&ctx).unwrap(), &saved);
    assert!(chunks.execution_strategy().provider().rows.calls.len() > after_failure);
    assert_tensor_exact(
        &chunks.decode(&token, &ctx).unwrap(),
        &actual,
        "restored text session",
    );
    assert_checkpoint(&chunks.checkpoint(&ctx).unwrap(), &after_success);
    let mut fork = session(&ctx);
    fork.rollback(saved, &ctx).unwrap();
    assert_tensor_exact(
        &fork.decode(&token, &ctx).unwrap(),
        &actual,
        "forked text session",
    );
    assert_checkpoint(&fork.checkpoint(&ctx).unwrap(), &after_success);
    let bad = NumericTensor::new([1, 1], vec![17.]);
    let before = chunks.execution_strategy().provider().rows.calls.len();
    assert!(chunks.decode(&bad, &ctx).is_err());
    assert_eq!(
        chunks.execution_strategy().provider().rows.calls.len(),
        before
    );
    assert_checkpoint(&chunks.checkpoint(&ctx).unwrap(), &after_success);
}

#[path = "session_rows.rs"]
mod selected_rows;

#[test]
fn qwen4_same_geometry_different_hashes_reject_prompt_cache_restore_before_io() {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let geometry = specification();
    let first_hash = fixture_hash();
    let second_hash = eredu_architectures::qwen4_exp::ngram::NGramHashSpec::new(
        32,
        7,
        3,
        1,
        vec![23703573157769, 9007199254740993, 8052911324071],
        vec![11, 13],
        vec![0, 11],
        24,
    )
    .unwrap();
    let ids = [3u64, 4, 6, 9];
    assert_ne!(
        first_hash
            .select(Some(&ids), 1, 4, &[7, 7], 8)
            .unwrap()
            .rows,
        second_hash
            .select(Some(&ids), 1, 4, &[7, 7], 8)
            .unwrap()
            .rows
    );
    let first = BoundTargetSpec::new(geometry.clone(), BTreeMap::from([(1, first_hash)])).unwrap();
    let second =
        BoundTargetSpec::new(geometry.clone(), BTreeMap::from([(1, second_hash)])).unwrap();
    assert_eq!(
        first.geometry().geometry_fingerprint(),
        second.geometry().geometry_fingerprint()
    );
    assert_eq!(
        first.geometry().state_layout().unwrap(),
        second.geometry().state_layout().unwrap()
    );
    assert_ne!(first.state_fingerprint(), second.state_fingerprint());
    let layout = geometry.state_layout().unwrap();
    let partition = PartitionState::new(layout.clone(), 0).unwrap();
    let first_model = TargetModel::<NumericBackend>::new(first.clone(), &ctx).unwrap();
    let second_model = TargetModel::<NumericBackend>::new(second.clone(), &ctx).unwrap();
    let first_identity = first_model
        .state_identity(&partition, Default::default())
        .unwrap()
        .prompt_cache_identity(&layout)
        .unwrap();
    let second_identity = second_model
        .state_identity(&partition, Default::default())
        .unwrap()
        .prompt_cache_identity(&layout)
        .unwrap();
    assert_ne!(first_identity, second_identity);
    let descriptor = eredu_core::cache::PromptCacheDescriptor::from_model_identity(
        first_identity,
        "same-claimed-checkpoint",
        "same-token-prefix",
        1,
    )
    .unwrap();
    let mut first_session = session_with_bound(first, &ctx, true);
    let mut second_session = session_with_bound(second, &ctx, true);
    let tokens = NumericTensor::from_i32_slice(&[3, 4, 6, 9], &[1, 4], &ctx).unwrap();
    first_session.prefill(&tokens, None, &ctx).unwrap();
    second_session.prefill(&tokens, None, &ctx).unwrap();
    assert_ne!(
        first_session.execution_strategy().provider().rows.calls,
        second_session.execution_strategy().provider().rows.calls
    );
    let before = second_session.checkpoint(&ctx).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("never-created-cache");
    let error = second_session
        .load_prompt_cache(&missing, &descriptor, &[3, 4, 6, 9], &ctx)
        .unwrap_err();
    assert!(
        matches!(
            &error,
            ReplicatedTextSessionError::PromptCache(
                eredu_core::cache::PromptCacheError::Incompatible(_)
            )
        ),
        "unexpected restore rejection: {error:?}"
    );
    assert!(!missing.exists());
    assert_checkpoint(&second_session.checkpoint(&ctx).unwrap(), &before);
}
