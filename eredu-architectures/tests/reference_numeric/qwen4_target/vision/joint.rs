//! Joint selected graph against separately executed tower and target equations.
use super::super::super::{cold, gguf, session};
use super::*;
use eredu_architectures::qwen4_exp::conditional::{ConditionalInput, ConditionalModel};
use eredu_runtime::*;

fn processor_request() -> eredu_runtime::ProcessorSelectionRequest {
    eredu_runtime::ProcessorSelectionRequest::new([M::Text, M::Image, M::Video])
        .with_projected_embeddings(true)
}
fn processor_facts() -> eredu_runtime::MediaPrimitiveCapabilities {
    use eredu_runtime::ProcessorPrimitive as P;
    eredu_runtime::MediaPrimitiveCapabilities::new(
        [M::Text, M::Image, M::Video],
        [M::Text, M::Image, M::Video],
        [M::Text, M::Image, M::Video],
        [
            P::TensorU32,
            P::TensorF32,
            P::TensorI32,
            P::RgbNormalize,
            P::RgbResizeBicubic,
            P::VideoSampling,
        ],
        i32::MAX as u64,
    )
}

fn reference(
    target: &PreparedTarget,
    ingress: &MediaIngress,
    prepared: &eredu_architectures::qwen4_exp::media::PreparedMediaInput<NumericTensor>,
    ctx: &NumericContext,
) -> (Vec<NumericTensor>, State) {
    use eredu_nn::EmbeddingOperator;
    let mut tower = tower(ingress, ctx);
    let projected = prepared
        .with_vision_input(|v| tower.forward(v.unwrap(), ctx))
        .unwrap()
        .embeddings;
    let mut model = TargetModel::<NumericBackend>::new(target.bound_spec().unwrap(), ctx).unwrap();
    let modules=<TargetModel<NumericBackend> as LayeredArchitecture<NumericBackend,State>>::static_modules_mut(&mut model);
    modules.visit_parameters_mut(&mut TargetBind {
        ordinary: target.static_parameters(),
        experts: vec![],
        ctx,
    });
    let units = (0..target.spec().units.len())
        .map(|i| {
            let mut unit = model.construct_unit(i, ctx).unwrap();
            let experts = if matches!(target.spec().units[i], UnitSpec::Decoder { .. }) {
                (0..3).map(|e| target.expert(i, e).unwrap()).collect()
            } else {
                vec![]
            };
            unit.visit_parameters_mut(&mut TargetBind {
                ordinary: target.unit(i).unwrap(),
                experts,
                ctx,
            });
            unit
        })
        .collect();
    let rows = target
        .row_lookups(
            RowLookupLimits {
                requests: 128,
                rows_per_acquisition: 2,
                acquisition_bytes: 128,
                host_bytes: 65536,
                output_bytes: 32768,
            },
            ResidencyPolicy::Cacheable,
        )
        .unwrap();
    let rows = rows
        .bind(
            rows.entries()
                .iter()
                .map(|(id, e)| (id.clone(), crate::row_bank::SourceRows::new(e)))
                .collect(),
        )
        .unwrap();
    let mut provider = ParameterProviders {
        grouped: ResidentExpertProvider,
        rows,
    };
    let mut runtime = LayerwiseRuntime::new(model, ResidentUnitWindow::new(units));
    let mut state = gguf::state_from_layout(&target.spec().state_layout().unwrap()).unwrap();
    let mut prefill = Vec::new();
    let length = prepared.token_ids().dim(1);
    for start in (0..length).step_by(13) {
        let chunk = prepared
            .assemble(
                start..(start + 13).min(length),
                Some(&projected),
                |ids| {
                    <TargetModel<NumericBackend> as LayeredArchitecture<
                            NumericBackend,
                            State,
                        >>::static_modules_mut(runtime.architecture_mut())
                        .embeddings
                        .forward(ids, ctx)
                },
                ctx,
            )
            .unwrap();
        let logits = chunk
            .with_target_input(|input| {
                runtime.forward_with_provider_and_observer_and_context(
                    input,
                    &mut state,
                    ExpertPass::Prefill,
                    &mut provider,
                    ctx,
                    &mut Observe::default(),
                )
            })
            .unwrap()
            .0;
        prefill.push(logits);
    }
    let first = NumericTensor::concatenate(&prefill, 1, ctx)
        .unwrap()
        .index(
            &[Index::Full, Index::Range(length - 1, length), Index::Full],
            ctx,
        )
        .unwrap();
    let mut output = vec![first];
    for step in 0..16 {
        let token = NumericTensor::from_i32_slice(&[step % 12], &[1, 1], ctx).unwrap();
        let input = <TargetModel<NumericBackend> as ReplicatedTextArchitecture<
            NumericBackend,
            State,
        >>::text_input(&token, None);
        output.push(
            runtime
                .forward_with_provider_and_observer_and_context(
                    input,
                    &mut state,
                    ExpertPass::Decode,
                    &mut provider,
                    ctx,
                    &mut Observe::default(),
                )
                .unwrap()
                .0,
        );
    }
    (output, state)
}
fn vision_acquisitions() -> usize {
    REFERENCE_STAGE_EVIDENCE.with(|e| {
        e.borrow()
            .payload_reads
            .iter()
            .filter(|read| read.task.starts_with("model.visual."))
            .count()
    })
}
struct FailAt(&'static str);
fn assert_media_error(mut error: &(dyn std::error::Error + 'static)) {
    loop {
        if let Some(error) = error.downcast_ref::<MediaInputError>() {
            assert!(matches!(error, MediaInputError::Geometry(_)));
            return;
        }
        error = error.source().expect("typed media error source");
    }
}
impl ActivationObserver<NumericTensor, Error> for FailAt {
    fn observe(&mut self, path: &str, _: &NumericTensor) -> Result<(), Error> {
        if path == self.0 {
            Err(Error::backend("injected joint graph failure"))
        } else {
            Ok(())
        }
    }
}
#[test]
fn joint_vision_target_selected_sessions_match_separate_equations_and_preserve_lifecycle() {
    let (_dir, target, ingress) = setup();
    run_joint(&target, &ingress);
    processor::run_processor(&target, &ingress);
}
#[test]
fn joint_vision_target_safetensors_keeps_retained_roles_and_bounded_execution() {
    let (dir, target, text_st) = gguf::fixtures(false);
    let path = dir.path().join("vision.gguf");
    write(&path, false, false, 32, false);
    let projector = gguf_vision(
        &target,
        &eredu_gguf::Checkpoint::open(&path).unwrap(),
        media(),
    )
    .unwrap();
    let (_vision_dir, vision_source) = transforms::fixture(&values());
    let mut config = eredu_architectures::qwen4_exp::config::Config::from_gguf(
        &eredu_gguf::Checkpoint::open(dir.path().join("target.gguf")).unwrap(),
    )
    .unwrap();
    config.ngram.source = eredu_architectures::qwen4_exp::config::NGramSourceLayout::Safetensors {
        vocabulary_base: 5,
        vocabulary_alignment: 1,
        shards: 1,
        seed: 1,
    };
    config.vision = Some(projector.config().clone());
    config.media = Some(media());
    let composite = Arc::new(
        eredu_checkpoint::store::CompositeCheckpointSource::new([
            text_st.artifact().clone(),
            vision_source,
        ])
        .unwrap(),
    );
    let target = PreparedTarget::safetensors(
        composite,
        config,
        eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
            &serde_json::json!({}),
        )
        .unwrap(),
        target.spec().limits,
    )
    .unwrap();
    let ingress = target.media_ingress(target.vision().unwrap()).unwrap();
    run_joint(&target, &ingress);
    processor::run_processor(&target, &ingress);
}
fn cold_joint(
    target: &PreparedTarget,
    ingress: &MediaIngress,
    residency: LayerWeightResidency,
) -> eredu_architectures::qwen4_exp::prepared::ConditionalHeaderExecutionPlan {
    use eredu_architectures::qwen4_exp::{
        config::NGramSourceLayout,
        prepared::{GgufTargetPlan, SafetensorsTargetPlan, TargetLoadError},
    };
    let request = super::super::super::load_policy::request(residency).with_media_execution(
        MediaLoadRequest::Required(MediaExecutionPolicy::new(processor_request()).unwrap()),
    );
    let disabled = request
        .clone()
        .with_media_execution(MediaLoadRequest::Disabled);
    match target.spec().configuration().ngram.source {
        NGramSourceLayout::Safetensors { .. } => {
            let (_, physical) = super::super::header::headers(target.artifact().as_ref());
            let header = SafetensorsTargetPlan::prepare(
                target.artifact().as_ref(),
                target.spec().configuration().clone(),
                eredu_architectures::qwen4_exp::checkpoint::schema::SafetensorsEncoding::from_json(
                    &serde_json::json!({}),
                )
                .unwrap(),
            )
            .unwrap();
            let vision = header.vision_plan(physical.clone()).unwrap();
            let physical = physical
                .into_iter()
                .filter(|(key, _)| header.resolution().source_keys().contains(key))
                .collect::<BTreeMap<_, _>>();
            assert!(matches!(
                header.execution_plan_for_load(
                    &request,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support,
                    physical.clone()
                ),
                Err(TargetLoadError::MediaPreparationRequired)
            ));
            assert!(matches!(
                header.conditional_execution_plan_for_load(
                    &disabled,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support,
                    vision.clone(),
                    physical.clone()
                ),
                Err(TargetLoadError::MissingMediaPolicy)
            ));
            header
                .conditional_execution_plan_for_load(
                    &request,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support,
                    vision,
                    physical,
                )
                .unwrap()
        }
        NGramSourceLayout::Gguf { .. } => {
            let target_path = target
                .artifact()
                .source_provenance("model.embed_tokens.weight")
                .unwrap()
                .backing_shard
                .unwrap();
            let header =
                GgufTargetPlan::prepare(&eredu_gguf::Checkpoint::open(target_path).unwrap())
                    .unwrap();
            let projector_path = ingress
                .vision()
                .artifact()
                .source_provenance("model.visual.pos_embed.weight")
                .unwrap()
                .backing_shard
                .unwrap();
            let vision = GgufVisionPlan::prepare(
                target.spec().configuration(),
                &eredu_gguf::Checkpoint::open(projector_path).unwrap(),
            )
            .and_then(|plan| plan.bind_media_tokens(media(), None))
            .unwrap();
            assert!(matches!(
                header.execution_plan_for_load(
                    &request,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support
                ),
                Err(TargetLoadError::MediaPreparationRequired)
            ));
            assert!(matches!(
                header.conditional_execution_plan_for_load(
                    &disabled,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support,
                    vision.clone()
                ),
                Err(TargetLoadError::MissingMediaPolicy)
            ));
            header
                .conditional_execution_plan_for_load(
                    &request,
                    eredu_nn::TensorElementType::F32,
                    &cold::Support,
                    vision,
                )
                .unwrap()
        }
    }
}
struct JointCompositeVisitor<'a> {
    context: &'a NumericContext,
    input: &'a PreparedModelInput<NumericTensor>,
    expected: &'a [NumericTensor],
    started: bool,
}
impl eredu_architectures::replicated_text::CompositeTextArchitectureVisitor<NumericBackend, State>
    for JointCompositeVisitor<'_>
{
    type Output = ();
    type Error = String;
    fn construction_started(&mut self) {
        self.started = true;
    }
    fn visit<A>(
        self,
        _: eredu_architectures::replicated_text::PreparedCompositeTextArchitecture<
            A,
            A::AdmissionConfig,
        >,
        _: SharedCheckpointSource,
    ) -> Result<(), String>
    where
        A: eredu_architectures::composite_execution::CompositeArchitecture<
                NumericBackend,
                State,
                Error = Error,
            > + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        Err("conditional target requires routed construction".into())
    }
    fn visit_routed<A>(
        self,
        prepared: eredu_architectures::replicated_text::PreparedRoutedCompositeTextArchitecture<
            A,
            A::AdmissionConfig,
        >,
        source: SharedCheckpointSource,
    ) -> Result<(), String>
    where
        A: eredu_architectures::composite_execution::CompositeArchitecture<
                NumericBackend,
                State,
                Error = Error,
            > + RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
    {
        use eredu_architectures::composite_execution::PreparedCompositeInput;
        assert!(self.started);
        assert_eq!(
            prepared.requirements().processor(),
            prepared.processor().requirements()
        );
        assert!(
            prepared
                .capability_estimate()
                .capabilities()
                .modalities
                .image
        );
        let residency = prepared.routed().text().selected().residency();
        let mut mechanisms = if residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(source)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(source)
        };
        mechanisms.fixture_state_factory = Some(gguf::state_from_layout);
        macro_rules! exercise {
            ($session:expr, $facts:expr) => {{
                let mut session = $session;
                let (_, _, admission) = $facts.into_parts();
                let inspector = Inspector(0.into());
                let admitted = A::admit_prepared_input(&admission, self.input, &inspector)
                    .map_err(|e| e.to_string())?;
                let paired = PreparedCompositeInput::new(self.input, &admitted)?;
                let first = session
                    .prefill_input(paired, self.context)
                    .map_err(|e| e.to_string())?;
                assert_tensor_close(
                    &first,
                    &self.expected[0],
                    "generic composite visitor prefill",
                );
                let checkpoint = session
                    .checkpoint(self.context)
                    .map_err(|e| e.to_string())?;
                for step in 0..16 {
                    let source = input(vec![text(&[step % 12])], &inspector);
                    let admitted = A::admit_prepared_input(&admission, &source, &inspector)
                        .map_err(|e| e.to_string())?;
                    let paired = PreparedCompositeInput::new(&source, &admitted)?;
                    let result = session
                        .decode_input(paired, self.context)
                        .map_err(|e| e.to_string())?;
                    assert_tensor_close(
                        &result,
                        &self.expected[step as usize + 1],
                        "generic composite visitor cached decode",
                    );
                }
                session
                    .rollback(checkpoint, self.context)
                    .map_err(|e| e.to_string())?;
                let source = input(vec![text(&[0])], &inspector);
                let admitted = A::admit_prepared_input(&admission, &source, &inspector)
                    .map_err(|e| e.to_string())?;
                let result = session
                    .decode_input(
                        PreparedCompositeInput::new(&source, &admitted)?,
                        self.context,
                    )
                    .map_err(|e| e.to_string())?;
                assert_tensor_close(
                    &result,
                    &self.expected[1],
                    "generic composite visitor rollback",
                );
                Ok::<_, String>(())
            }};
        }
        eredu_architectures::prepared_execution::construct_selected_routed_composite_session(
            prepared,
            mechanisms,
            self.context,
            |_, _, rows| {
                let rows = rows.expect("selected row provider");
                let providers = rows
                    .prepared()
                    .bind(
                        rows.prepared()
                            .entries()
                            .iter()
                            .map(|(id, entry)| {
                                (id.clone(), crate::row_bank::SourceRows::new(entry))
                            })
                            .collect(),
                    )
                    .unwrap();
                Ok::<_, String>(
                    (
                        BTreeMap::<
                            RoutedBankId,
                            (NumericGroupedBankMechanism, NumericIndexedMovement),
                        >::new(),
                        Some(providers),
                    ),
                )
            },
            (),
            |_, session, facts| exercise!(session, facts),
            |_, session, facts| exercise!(session, facts),
        )
        .map_err(|e| e.to_string())
    }
}

fn run_joint(target: &PreparedTarget, ingress: &MediaIngress) {
    let ctx = NumericContext {
        bind_checkpoint_values: true,
        ..Default::default()
    };
    let inspector = Inspector(0.into());
    let prompt_input = prompt(&inspector);
    let prepared = ingress
        .admission_config()
        .admit(&prompt_input, &inspector)
        .and_then(|admitted| admitted.prepare(&ctx))
        .unwrap();
    let expected = reference(&target, &ingress, &prepared, &ctx);
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let before = target
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads;
        let vision_before = ingress
            .vision()
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads;
        let bound_plan = gguf::plan(&target)
            .with_vision(ingress.vision().clone())
            .unwrap();
        let plan = cold_joint(target, ingress, residency);
        assert_eq!(
            plan.requirements(),
            bound_plan.requirements(),
            "source-free and retained joint contracts"
        );
        let graph = plan.requirements().text().execution_graph();
        assert_eq!(
            graph.groups()[0].id(),
            eredu_architectures::decoder::TARGET_EXECUTION_GROUP
        );
        assert_eq!(graph.execution_order(), [1, 0]);
        assert_eq!(plan.requirements().text().execution_units().len(), 5);
        let facts = cold::capabilities(plan.requirements(), None);
        let request = plan.load_selection_request().unwrap().clone();
        assert_eq!(plan.load_processor_request(), Some(&processor_request()));

        let changed = ProcessorSelectionRequest::new([M::Image]);
        assert!(matches!(plan.clone().select(&request, &facts, None, &changed, &processor_facts()), Err(eredu_architectures::qwen4_exp::prepared::TargetSelectionError::ProcessorRequestMismatch)));
        let selected = plan
            .select(
                &request,
                &facts,
                None,
                &processor_request(),
                &processor_facts(),
            )
            .unwrap();
        assert!(
            selected
                .header_plan()
                .capability_estimate()
                .capabilities()
                .modalities
                .image
        );
        assert!(
            selected
                .header_plan()
                .capability_estimate()
                .capabilities()
                .modalities
                .video
        );
        let workspace = selected
            .parameter_materialization_workspace(
                &super::super::super::super::prepared_adapter::NumericPreparationProvider {
                    addressable: true,
                },
            )
            .unwrap();
        assert!(workspace.ordinary_recipe_peak_bytes >= 32 * 768 * 4);
        assert_eq!(
            target
                .artifact()
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            before
        );
        assert_eq!(
            ingress
                .vision()
                .artifact()
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            vision_before
        );
        let missing_vision: SharedCheckpointSource = Arc::new(
            eredu_checkpoint::store::RestrictedCheckpointSource::including(
                ingress.vision().artifact().clone(),
                "missing projector role",
                std::collections::BTreeSet::new(),
            )
            .unwrap(),
        );
        assert!(selected
            .clone()
            .bind(target.artifact().clone(), missing_vision, None)
            .is_err());
        assert_eq!(
            target
                .artifact()
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            before,
            "a rejected projector must not acquire target integer controls"
        );
        let selected = selected
            .bind(
                target.artifact().clone(),
                ingress.vision().artifact().clone(),
                None,
            )
            .unwrap();
        let control_reads = usize::from(matches!(
            target.spec().configuration().ngram.source,
            eredu_architectures::qwen4_exp::config::NGramSourceLayout::Safetensors { .. }
        )) as u64
            * 3;
        let after_binding = target
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads;
        assert_eq!(after_binding, before + control_reads);
        let vision_after_binding = ingress
            .vision()
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads;
        reset_reference_stage_evidence("qwen4-joint");
        let (handoff, source) = selected
            .clone()
            .prepare::<NumericBackend, State>(&ctx)
            .unwrap();
        assert_eq!(
            target
                .artifact()
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            after_binding
        );
        assert_eq!(
            ingress
                .vision()
                .artifact()
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            vision_after_binding
        );
        let mut mechanisms = if residency.is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(source)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(source)
        };
        mechanisms.fixture_state_factory = Some(gguf::state_from_layout);
        macro_rules! exercise {
            ($session:expr) => {{
                let mut session = $session;
                let empty = session.checkpoint(&ctx).unwrap();
                let oversized = input(vec![text(&[3; 33])], &inspector);
                let oversized = ingress
                    .admission_config()
                    .admit(&oversized, &inspector)
                    .and_then(|admitted| admitted.prepare(&ctx))
                    .unwrap();
                let error = session
                    .prefill_input(ConditionalInput::Media(&oversized), &ctx)
                    .unwrap_err();
                assert_media_error(&error);
                session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &empty);
                for path in ["model.visual.blocks.1.output", "model.layers.0.mixer.write"] {
                    assert!(session
                        .prefill_input_with_observer(
                            ConditionalInput::Media(&prepared),
                            &ctx,
                            &mut FailAt(path)
                        )
                        .is_err());
                    session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &empty);
                }
                let first = session
                    .prefill_input(ConditionalInput::Media(&prepared), &ctx)
                    .unwrap();
                assert_tensor_close(&first, &expected.0[0], "joint prefill");
                // A restricted SafeTensors role may share aggregate source counters
                // with target weights; attribute acquisition by canonical task identity.
                let vision_reads = vision_acquisitions();
                let checkpoint = session.checkpoint(&ctx).unwrap();
                let error = session
                    .prefill_input(ConditionalInput::Media(&prepared), &ctx)
                    .unwrap_err();
                assert_media_error(&error);
                session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &checkpoint);
                let frozen = session.capture_control_state(&ctx).unwrap();
                let mut fork = session.copy_control_state(&frozen, &ctx).unwrap();
                let token = NumericTensor::from_i32_slice(&[3], &[1, 1], &ctx).unwrap();
                let a = session.decode(&token, &ctx).unwrap();
                session.exchange_control_state(&mut fork, &ctx).unwrap();
                assert_tensor_exact(&a, &session.decode(&token, &ctx).unwrap(), "joint fork");
                session.rollback(checkpoint, &ctx).unwrap();
                let mut no_vision = FailAt("model.visual.blocks.1.output");
                for step in 0..16 {
                    let token = NumericTensor::from_i32_slice(&[step % 12], &[1, 1], &ctx).unwrap();
                    let value = session
                        .decode_input_with_observer(
                            <ConditionalModel<NumericBackend> as ReplicatedTextArchitecture<
                                NumericBackend,
                                State,
                            >>::text_input(&token, None),
                            &ctx,
                            &mut no_vision,
                        )
                        .unwrap();
                    assert_tensor_close(
                        &value,
                        &expected.0[step as usize + 1],
                        "joint cached decode",
                    );
                }
                session::assert_checkpoint(&session.checkpoint(&ctx).unwrap(), &expected.1);
                session.reset(&ctx).unwrap();
                let tokens = NumericTensor::from_i32_slice(&[3, 4, 5], &[1, 3], &ctx).unwrap();
                session
                    .prefill_input_with_observer(
                        <ConditionalModel<NumericBackend> as ReplicatedTextArchitecture<
                            NumericBackend,
                            State,
                        >>::text_input(&tokens, None),
                        &ctx,
                        &mut no_vision,
                    )
                    .unwrap();
                session.reset(&ctx).unwrap();
                let projected = input(
                    vec![PreparedInputPart::new(
                        M::Text,
                        P::Embeddings(NumericTensor::new([1, 3, 32], vec![0.125; 96])),
                        [(K::OriginalTokenIds, tokens.clone())],
                    )
                    .unwrap()],
                    &inspector,
                );
                let projected = ingress
                    .admission_config()
                    .admit(&projected, &inspector)
                    .and_then(|admitted| admitted.prepare(&ctx))
                    .unwrap();
                session
                    .prefill_input_with_observer(
                        ConditionalInput::Media(&projected),
                        &ctx,
                        &mut no_vision,
                    )
                    .unwrap();
                assert_eq!(
                    vision_acquisitions(),
                    vision_reads,
                    "decode and projected requests do not reacquire vision payloads"
                );
                Ok::<_, String>(())
            }};
        }
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
                            .map(|(id, e)| (id.clone(), crate::row_bank::SourceRows::new(e)))
                            .collect(),
                    )
                    .unwrap();
                Ok::<_, String>(
                    (
                        BTreeMap::<
                            RoutedBankId,
                            (NumericGroupedBankMechanism, NumericIndexedMovement),
                        >::new(),
                        Some(providers),
                    ),
                )
            },
            (),
            |_, s, _| exercise!(s),
            |_, s, _| exercise!(s),
        )
        .unwrap();
        REFERENCE_STAGE_EVIDENCE.with(|e| {
            let e = e.borrow();
            if !residency.is_fully_resident() {
                assert_eq!(e.peak_bound_units, 1);
                assert_eq!(&e.bounded_unit_acquisitions[..2], &[3, 4]);
                assert!(e
                    .payload_reads
                    .iter()
                    .filter(|r| r.task.starts_with("model.visual.blocks."))
                    .all(|r| r.physically_bounded));
            }
        });
        selected
            .visit_composite::<NumericBackend, State, _>(
                &ctx,
                JointCompositeVisitor {
                    context: &ctx,
                    input: &prompt_input,
                    expected: &expected.0,
                    started: false,
                },
            )
            .unwrap();
    }
}

#[path = "processor.rs"]
mod processor;

#[test]
fn conditional_history_rejection_precedes_request_tensors_and_media_embeddings() {
    use eredu_architectures::composite_execution::{CompositeArchitecture, PreparedCompositeInput};
    use eredu_architectures::qwen4_exp::input::RequestError;
    let (directory, _, original_ingress) = setup();
    let mut limits = eredu_evaluation::qwen4_exp::limits();
    limits.history_tokens = 32;
    let header = eredu_architectures::qwen4_exp::prepared::GgufTargetPlan::prepare(
        &eredu_gguf::Checkpoint::open(directory.path().join("target.gguf")).unwrap(),
    )
    .unwrap();
    let source = super::super::super::qwen4_gguf::open_text_source(header.text_plan());
    let target = header.bind(source, limits).unwrap();
    let ingress = target
        .media_ingress(original_ingress.vision().clone())
        .unwrap();
    let ctx = NumericContext::default();
    let inspector = Inspector(0.into());
    let source = input(
        vec![
            text(&vec![3; 30]),
            PreparedInputPart::new(
                M::Image,
                P::Tensor(pixels(16)),
                [(K::PatchGrid, grid(1, 4, 4))],
            )
            .unwrap(),
        ],
        &inspector,
    );
    let before = ctx.integer_tensor_constructions.get();
    assert!(
        ingress
            .admission_config()
            .admit(&source, &inspector)
            .is_err(),
        "34 positions exceed complete history during host admission"
    );
    assert_eq!(before, ctx.integer_tensor_constructions.get());
    let mut model = ConditionalModel::<NumericBackend>::new(
        target.bound_spec().unwrap(),
        ingress.clone(),
        ingress.vision().config().clone(),
        &ctx,
    )
    .unwrap();
    let mut state = gguf::state_from_layout(&target.spec().state_layout().unwrap()).unwrap();
    for index in 0..target.spec().units.len() {
        // Only the authoritative prefix is inspected before rejection. Retained
        // state remains deliberately untouched by a rejected invocation.
        let lane = state.layer(index).unwrap();
        lane.fixed_offset = 31;
        if let Some(attention) = &mut lane.attention {
            attention.offset = 31;
        }
    }
    let saved = state.clone();
    let counts = || {
        (
            ctx.integer_tensor_constructions.get(),
            ctx.embedding_forward_calls.get(),
        )
    };
    let before = counts();
    let assert_history = |mut error: &(dyn std::error::Error + 'static), required| loop {
        if let Some(RequestError::History {
            required: actual,
            limit,
        }) = error.downcast_ref::<RequestError>()
        {
            assert_eq!((*actual, *limit), (required, 32));
            break;
        }
        error = error.source().expect("typed history rejection");
    };
    // Both ordinary token continuation and raw image/video composite admission
    // reject before preparing native request tensors or starting the encoder.
    for source in [input(vec![text(&[3, 4])], &inspector), prompt(&inspector)] {
        let proof = ingress
            .admission_config()
            .admit(&source, &inspector)
            .unwrap()
            .into_composite();
        let required = 31 + proof.decoder_positions();
        let error = model
            .begin_composite_forward(
                PreparedCompositeInput::new(&source, &proof).unwrap(),
                &mut state,
                &ctx,
            )
            .err()
            .expect("composite request exceeds cumulative history");
        assert_history(&error, required);
        assert_eq!(
            counts(),
            before,
            "no native request preparation or encoder entry"
        );
        session::assert_checkpoint(&state, &saved);
    }
}

#[path = "request_extent.rs"]
mod request_extent;

#[test]
fn conditional_prediction_headers_select_independent_state_without_payload_reads() {
    use eredu_architectures::qwen4_exp::{
        checkpoint::schema::SafetensorsEncoding, config::Config, mtp::PredictionLimits,
        prepared::SafetensorsTargetPlan,
    };
    use eredu_checkpoint::store::SafetensorsWeightStore;
    use eredu_evaluation::qwen4_exp::{
        add_prediction_weights, add_vision_weights, PreparedFixtures,
    };
    let directory = tempfile::tempdir().unwrap();
    let fixture = PreparedFixtures::write(directory.path()).unwrap();
    add_prediction_weights(&fixture.safetensors_path).unwrap();
    add_vision_weights(&fixture.safetensors_path, 2);
    let json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture.safetensors_path.join("config.json")).unwrap(),
    )
    .unwrap();
    let source: SharedCheckpointSource =
        Arc::new(SafetensorsWeightStore::open(&fixture.safetensors_path).unwrap());
    let config = Config::from_json(&json).unwrap();
    let header = SafetensorsTargetPlan::prepare(
        source.as_ref(),
        config,
        SafetensorsEncoding::from_json(&json).unwrap(),
    )
    .unwrap();
    let (_, physical) = super::super::header::headers(source.as_ref());
    let vision = header.vision_plan(physical.clone()).unwrap();
    let policy = eredu_evaluation::qwen4_exp::bounded_policy();
    let request = NormalizedLoadRequest::default()
        .with_bounded_execution(policy)
        .with_drafting(DraftingLoadRequest::Disabled);
    let target_limits = eredu_architectures::qwen4_exp::target::TargetLimits::from_load_request(
        &request,
        eredu_nn::TensorElementType::F32,
    )
    .unwrap();
    let limits = PredictionLimits {
        qsa: target_limits.qsa,
        tile_blocks: target_limits.tile_blocks,
        selection_workspace: target_limits.selection_workspace,
        element: target_limits.element,
    };
    let prediction = header.prediction_spec(limits).unwrap();
    let streams = prediction
        .units
        .iter()
        .flat_map(|unit| {
            let eredu_architectures::qwen4_exp::target::MixerSpec::Indexed(attention) = &unit.mixer
            else {
                return Vec::new();
            };
            attention
                .state
                .streams()
                .into_iter()
                .map(|spec| AppendStreamBinding {
                    layer: unit.depth,
                    lanes: limits.qsa.batch as u32,
                    spec,
                    limits: policy.append().limits(),
                    payload_bytes: policy.append().payload_bytes(),
                    scratch_bytes: policy.append().scratch_bytes(),
                    catalog_bytes: policy.append().catalog_bytes(),
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let plan = header
        .execution_plan_for_load(
            &request,
            eredu_nn::TensorElementType::F32,
            &cold::Support,
            physical,
        )
        .unwrap()
        .with_prediction(limits, streams)
        .unwrap()
        .with_vision(vision)
        .unwrap();
    let facts = cold::capabilities(plan.requirements(), None);
    let selection = plan.load_selection_request().unwrap().clone();
    let prediction_facts = synthesize_state_capabilities(
        plan.prediction_state_requirements().unwrap(),
        &CacheResidencyPolicy::Device,
        &NumericMechanismSupport {
            persistent_session: true,
            ..Default::default()
        },
    );
    assert!(plan
        .clone()
        .select(
            &selection,
            &facts,
            None,
            &processor_request(),
            &processor_facts()
        )
        .is_err());
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    let selected = plan
        .select(
            &selection,
            &facts,
            Some(&prediction_facts),
            &processor_request(),
            &processor_facts(),
        )
        .unwrap();
    assert!(selected.prediction_state().is_some());
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    assert_eq!(selected.prediction_spec().unwrap().units.len(), 1);
    let bound = selected.bind(source.clone(), source, None).unwrap();
    assert!(bound.prediction_state().is_some());
}

#[path = "partitioned.rs"]
mod partitioned;

#[path = "ordinary_partition.rs"]
mod ordinary_partition;
