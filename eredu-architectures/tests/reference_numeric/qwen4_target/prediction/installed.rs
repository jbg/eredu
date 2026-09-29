//! Exact selected state reaches construction and the common prediction executor.
use super::super::super::row_bank::SourceRows;
use super::executor::{Lane, Materializer, SourceBinding};
use super::*;
use eredu_architectures::prediction_extension::*;
use eredu_architectures::qwen4_exp::prepared::SelectedTargetExecution;
use eredu_architectures::routed_text::PlannedResidentBank;
use eredu_runtime::*;

type Provider = RoutedBankProviders<PlannedResidentBank>;
pub(super) type Executor = MaterializedQwen4Prediction<NumericBackend, Materializer, Provider>;
pub(super) type Session = ReplicatedTextSession<
    Target,
    NumericBackend,
    NumericReplicatedMechanisms,
    RoutedReplicatedTextExecution<
        ParameterProviders<Provider, Option<RowLookupProviders<BoundedRowLookup<SourceRows>>>>,
    >,
>;
pub(super) struct Invoker<'a> {
    pub(super) session: &'a mut Session,
    pub(super) ctx: &'a NumericContext,
}
impl PredictionOperationInvoker<Target, NumericBackend, State> for Invoker<'_> {
    type Error = Error;
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, Error>
    where
        O: PredictionTargetOperation<Target, NumericBackend, State>,
    {
        self.session
            .apply_prediction_target_operation(operation, self.ctx)
            .map_err(Error::backend_source)
    }
    fn invalid(message: String) -> Error {
        Error::backend(message)
    }
}

pub(super) type Target = TargetModel<NumericBackend>;

pub(super) fn realize(selected: &SelectedStateRealization) -> Result<Lane, String> {
    assert_eq!(selected.policy(), &CacheResidencyPolicy::Device);
    DeviceState::create(selected.layout().clone(), |ordinal, policy| {
        let mut layer = NumericHybridLayerState::new(policy);
        for binding in selected
            .append_streams()
            .iter()
            .filter(|b| b.layer == ordinal)
        {
            for lane in 0..binding.lanes {
                layer.streams.push((
                    binding.spec.slot,
                    lane,
                    ResidentAppendStream::new(
                        binding.spec.clone(),
                        binding.limits,
                        binding.payload_bytes,
                        binding.scratch_bytes,
                        binding.catalog_bytes,
                    )
                    .map_err(|e| e.to_string())?,
                ));
            }
        }
        Ok::<_, String>(layer)
    })
    .map(Lane)
    .map_err(|e| e.to_string())
}

pub(super) fn bind_target(
    target_handoff: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<Target>,
    source: SharedCheckpointSource,
    residency: LayerWeightResidency,
    auxiliary_banks: &[RoutedBankId],
    ctx: &NumericContext,
) -> (Session, Option<Provider>) {
    let rows = target_handoff.row_lookups().unwrap().prepared();
    let rows = rows
        .bind(
            rows.entries()
                .iter()
                .map(|(id, entry)| (id.clone(), SourceRows::new(entry)))
                .collect(),
        )
        .unwrap();
    let mut mechanisms = if residency.is_fully_resident() {
        NumericReplicatedMechanisms::with_bound_checkpoint(source)
    } else {
        NumericReplicatedMechanisms::with_bounded_checkpoint(source)
    };
    mechanisms.fixture_state_factory = Some(super::super::gguf::state_from_layout);
    target_handoff
        .construct_resident_session::<NumericBackend, _, _>(
            mechanisms,
            Some(rows),
            auxiliary_banks,
            ctx,
        )
        .unwrap()
}

/// Joint providers retain every declared parameter role, but no table/control authority.
pub(super) fn assert_joint_sources(
    selected: &SelectedTargetExecution,
    artifact: &SharedCheckpointSource,
    target_source: &SharedCheckpointSource,
    source: &SharedCheckpointSource,
) {
    let requirements = selected.realization().text().requirements();
    let primary = requirements
        .parameters()
        .iter()
        .flat_map(|parameter| parameter.sources().iter().cloned())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        target_source
            .source_keys()
            .into_iter()
            .collect::<BTreeSet<_>>(),
        primary
    );
    for key in artifact
        .source_keys()
        .into_iter()
        .filter(|key| !primary.contains(key))
    {
        assert!(target_source.source_metadata(&key).is_err());
        assert!(target_source.acquire_lease(key.as_str().into()).is_err());
    }
    for key in &primary {
        assert_eq!(
            target_source.source_metadata(key).unwrap(),
            artifact.source_metadata(key).unwrap()
        );
        assert_eq!(
            target_source.source_provenance(key).unwrap(),
            artifact.source_provenance(key).unwrap()
        );
    }
    let expected = requirements
        .parameters()
        .iter()
        .chain(requirements.auxiliary_parameters())
        .flat_map(|parameter| parameter.sources().iter().cloned())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        source.source_keys().into_iter().collect::<BTreeSet<_>>(),
        expected
    );
    for key in &expected {
        assert_eq!(
            source.source_metadata(key).unwrap(),
            artifact.source_metadata(key).unwrap()
        );
        assert_eq!(
            source.source_provenance(key).unwrap(),
            artifact.source_provenance(key).unwrap()
        );
    }
    for key in artifact
        .source_keys()
        .into_iter()
        .filter(|key| !expected.contains(key))
    {
        assert!(source.source_metadata(&key).is_err());
        assert!(source.acquire_lease(key.as_str().into()).is_err());
    }
}

pub(super) fn check(
    selected: &SelectedTargetExecution,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
) {
    let spec = selected.prediction_spec().unwrap();
    let owner = prepared.prediction(spec.limits).unwrap();
    let mut binding = SourceBinding {
        source: prepared.artifact().clone(),
        context: ctx,
        residency: selected.realization().text().residency(),
        roles: vec![],
    };
    let before = binding.source.source_diagnostics().unwrap().physical_reads;
    let failed = selected
        .prepare_prediction_weights::<NumericBackend>(ctx)
        .unwrap()
        .materialize_executor::<Materializer, _>(&mut binding, ResidentExpertProvider, |_, _| {
            Err("state allocation failed".into())
        });
    assert!(matches!(failed, Err(PredictionConstructionError::State(_))));
    let failed = selected
        .prepare_prediction_weights::<NumericBackend>(ctx)
        .unwrap()
        .materialize_executor::<Materializer, _>(&mut binding, ResidentExpertProvider, |_, _| {
            Ok(Lane(state(prepared.spec())))
        });
    assert!(matches!(
        failed,
        Err(PredictionConstructionError::Contract(_))
    ));
    assert!(binding.roles.is_empty());
    assert_eq!(
        binding.source.source_diagnostics().unwrap().physical_reads,
        before
    );
    let joint = selected
        .clone()
        .prepare_prediction_execution::<NumericBackend, State>(ctx)
        .unwrap();
    let auxiliary_banks = joint.prediction_banks();
    assert_joint_sources(
        selected,
        prepared.artifact(),
        joint.target_source(),
        joint.provider_source(),
    );
    let (target_handoff, prediction_handoff, source, provider_source) = joint.into_parts();
    binding.source = provider_source;
    assert_eq!(
        binding.source.source_diagnostics().unwrap().physical_reads,
        before
    );
    let bank = spec.units[0].feed_forward.feed_forward.bank;
    let (mut session, provider) = bind_target(
        target_handoff,
        source,
        binding.residency,
        &auxiliary_banks,
        ctx,
    );
    let provider = provider.unwrap();
    assert_eq!(provider.banks().keys().copied().collect::<Vec<_>>(), [bank]);
    assert!(!session
        .execution_strategy()
        .provider()
        .grouped
        .banks()
        .contains_key(&bank));
    let prompt = NumericTensor::token_ids(&[3, 4, 7]);
    let (logits, capture) = session
        .prefill_prediction_target(&prompt, None, ctx)
        .unwrap();
    assert_eq!(capture.shape, [1, 3, 2, 2]);
    let (ordinary, source) = selected
        .clone()
        .prepare::<NumericBackend, State>(ctx)
        .unwrap();
    let expected_primary = selected
        .realization()
        .text()
        .requirements()
        .parameters()
        .iter()
        .flat_map(|parameter| parameter.sources().iter().cloned())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        source.source_keys().into_iter().collect::<BTreeSet<_>>(),
        expected_primary
    );
    for parameter in selected
        .realization()
        .text()
        .requirements()
        .auxiliary_parameters()
    {
        for key in parameter
            .sources()
            .iter()
            .filter(|key| !expected_primary.contains(*key))
        {
            assert!(source.source_metadata(key).is_err());
            assert!(source.acquire_lease(key.as_str().into()).is_err());
        }
    }
    let (mut baseline, no_auxiliary) = bind_target(ordinary, source, binding.residency, &[], ctx);
    assert!(no_auxiliary.is_none());
    let (baseline_logits, baseline_capture) = baseline
        .prefill_prediction_target(&prompt, None, ctx)
        .unwrap();
    assert_tensor_exact(&logits, &baseline_logits, "paired target prefill");
    assert_tensor_exact(
        &capture,
        &baseline_capture,
        "paired target residual capture",
    );
    assert!(session
        .execution_strategy()
        .provider()
        .rows
        .as_ref()
        .unwrap()
        .providers()
        .values()
        .all(|p| p.bank().completed() > 0));
    let mut executor = prediction_handoff
        .materialize_executor::<Materializer, _>(&mut binding, provider, |_, state| {
            assert_eq!(Some(state), selected.prediction_state());
            realize(state)
        })
        .unwrap();
    assert_eq!(binding.roles.len(), spec.units.len() + 1);
    assert_eq!(
        executor
            .provider()
            .banks()
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        auxiliary_banks
    );

    let mut lane = <Executor as MaterializedPredictionExecutor<
        Target,
        NumericBackend,
        Materializer,
    >>::new_state(&executor);
    let mut replay = lane.clone();
    let mut target = Target::new(prepared.bound_spec().unwrap(), ctx).unwrap();
    <Target as LayeredArchitecture<NumericBackend, State>>::static_modules_mut(&mut target)
        .visit_parameters_mut(&mut Parameters::default());
    let untouched = session.checkpoint(ctx).unwrap();
    let mut expected_shared = shared(&owner, ctx);
    let mut expected_units: Vec<_> = (0..spec.units.len())
        .map(|depth| unit(&owner, depth, ctx))
        .collect();
    let mut expected_state = prediction_state(&spec);
    let tokens = NumericTensor::token_ids(&(0..35).map(|i| i % 31 + 1).collect::<Vec<_>>());
    let captures = NumericTensor::new(
        [1, 35, 2, 2],
        (0..140)
            .map(|i| ((i * 13 % 41) as f32 - 20.) / 17.)
            .collect(),
    );
    for range in [0..3, 3..8, 8..19]
        .into_iter()
        .chain((19..35).map(|i| i..i + 1))
    {
        let hidden = captures.axis_slice(1, range.start, range.end);
        let token = tokens.axis_slice(1, range.start, range.end);
        let embedding = target.prediction_embedding(&token, ctx).unwrap();
        for (depth, unit) in expected_units.iter_mut().enumerate() {
            let expected = unit
                .forward(
                    &mut expected_shared,
                    PredictionInput {
                        embeddings: &embedding,
                        residual: &hidden,
                        visible: None,
                        rotary: None,
                    },
                    expected_state.layer(depth).unwrap(),
                    &mut ResidentExpertProvider,
                    ctx,
                    &mut ComponentInstrumentation::disabled(),
                )
                .unwrap();
            let logits = target
                .prediction_logits(
                    &expected.hidden,
                    ctx,
                    &mut ComponentInstrumentation::disabled(),
                )
                .unwrap();
            let (actual, capture) = executor
                .logits::<State, _>(
                    &mut Invoker {
                        session: &mut session,
                        ctx,
                    },
                    &hidden,
                    &token,
                    depth,
                    &mut lane,
                )
                .unwrap();
            assert_tensor_exact(&actual, &logits, "selected executor logits");
            assert_tensor_exact(&capture, &expected.capture, "selected executor capture");
        }
        executor
            .advance_observed::<State, _>(
                &mut Invoker {
                    session: &mut session,
                    ctx,
                },
                &hidden,
                &token,
                &mut replay,
                None,
            )
            .unwrap();
    }
    // The admitted second lane is unused here; compare every populated component
    // and prove target state has not been used as prediction storage.
    assert_state_exact(
        &lane.0,
        &replay.0,
        spec.units.len(),
        "selected committed replay",
    );
    assert_state_exact(
        &session.checkpoint(ctx).unwrap(),
        &untouched,
        prepared.spec().units.len(),
        "selected prediction isolates target",
    );
    let saved = lane.clone();
    let token = NumericTensor::token_ids(&[5]);
    let hidden = captures.axis_slice(1, 0, 1);
    let mut invoker = Invoker {
        session: &mut session,
        ctx,
    };
    let first = executor
        .logits::<State, _>(&mut invoker, &hidden, &token, 0, &mut lane)
        .unwrap();
    lane = saved;
    let replay = executor
        .logits::<State, _>(&mut invoker, &hidden, &token, 0, &mut lane)
        .unwrap();
    assert_tensor_exact(&first.0, &replay.0, "selected lane rollback logits");
    assert_tensor_exact(&first.1, &replay.1, "selected lane rollback capture");
}

/// A fresh pair from exactly one retained selection and provider collection.
pub(super) fn paired(
    selected: &SelectedTargetExecution,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
) -> (Session, Executor) {
    let joint = selected
        .clone()
        .prepare_prediction_execution::<NumericBackend, State>(ctx)
        .unwrap();
    let banks = joint.prediction_banks();
    assert_joint_sources(
        selected,
        prepared.artifact(),
        joint.target_source(),
        joint.provider_source(),
    );
    let (target, prediction, source, provider_source) = joint.into_parts();
    let residency = selected.realization().text().residency();
    let (session, provider) = bind_target(target, source, residency, &banks, ctx);
    let mut binding = SourceBinding {
        source: provider_source,
        context: ctx,
        residency,
        roles: vec![],
    };
    let extension = prediction
        .materialize_executor::<Materializer, _>(&mut binding, provider.unwrap(), |_, state| {
            realize(state)
        })
        .unwrap();
    (session, extension)
}

pub(super) fn advance(
    extension: &mut Executor,
    session: &mut Session,
    hidden: &NumericTensor,
    tokens: &NumericTensor,
    lane: &mut Lane,
    ctx: &NumericContext,
) {
    extension
        .advance_observed::<State, _>(&mut Invoker { session, ctx }, hidden, tokens, lane, None)
        .unwrap();
}
