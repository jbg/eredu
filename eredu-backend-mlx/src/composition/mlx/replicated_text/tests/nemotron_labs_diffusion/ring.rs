//! Explicit local Ring validation of loaded model, residency and mode controls.
use super::*;
use eredu_core::{
    TextAdvancement,
    decoding::{DecodingMode, SpeculationOptionsRequest},
    execution_control::GenerationControlHandle,
};
#[derive(Clone)]
struct RingFilter(usize);
impl TokenFilterController for RingFilter {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
        Ok(eredu_core::TokenFilter::allowed((0..self.0).map(|id| id != 0).collect()).unwrap())
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        assert_ne!(token, 0);
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}
impl eredu_runtime::execution_control::SnapshotTokenController for RingFilter {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        Some(std::mem::size_of::<Self>() as u64)
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(Self(self.0))
    }
}

impl eredu_core::SpeculativeTokenFilterController for RingFilter {
    fn filter_at(&self, _: &[u32]) -> Result<eredu_core::TokenFilter, Self::Error> {
        Ok(eredu_core::TokenFilter::allowed((0..self.0).map(|id| id != 0).collect()).unwrap())
    }
    fn prefix_is_complete(&self, _: &[u32]) -> Result<bool, Self::Error> {
        Ok(false)
    }
    fn control_snapshot_bytes(&self) -> Option<u64> {
        Some(std::mem::size_of::<Self>() as u64)
    }
}

fn request_table_handoffs<'a>(
    runtime: &mut ModelRuntime<MlxBackend<'a>>,
    root_commands: bool,
    vision: bool,
    vocabulary: usize,
    adapter: bool,
) -> Vec<u32> {
    use crate::composition::mlx::speculative::{
        MlxSpeculativeSampling, SpeculativeExecutionStreams,
    };
    use eredu_core::{
        GenerationRequestConfig, GenerationRequestTable, SpeculativeCallbackPublisher,
        SpeculativeOutputRuntime, SpeculativeRandomness, SpeculativeSemanticConstraint,
    };
    use eredu_runtime::ConstrainedSampler;
    runtime.reset().unwrap();
    let activations = if std::env::var_os("EREDU_NEMOTRON_RING_ACTIVATIONS").is_some() {
        use eredu_core::capture::*;
        use eredu_core::speculative::*;
        let discovery = <MlxBackend<'_> as eredu_core::SpeculativeGenerationBackend>::speculative_activation_discovery(runtime).unwrap();
        let allowance = CaptureUsage {
            captures: 8,
            retained_bytes: (if vision { 192 } else { 64 }) << 20,
            host_bytes: (if vision { 192 } else { 64 }) << 20,
            encoded_bytes: 4 << 20,
        };
        Some(
            SpeculativeActivationPlan {
                schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                captures: CapturePlan {
                    schema_version: CAPTURE_SCHEMA_VERSION,
                    provenance: CaptureProvenance::ContentExact,
                    selections: {
                        let mut selections = vec![CaptureSelection {
                            id: "shared-text-layer".into(),
                            path: "model.layers.0.feed_forward.residual".into(),
                            schedule: Default::default(),
                            transform: CaptureTransform::FullTensor,
                            slices: vec![],
                        }];
                        if vision {
                            for (id, path) in [
                                (
                                    "shared-vision-residual",
                                    "model.vision_tower.transformer.layers.0.feed_forward.residual",
                                ),
                                (
                                    "shared-projector",
                                    "model.multi_modal_projector.linear_2.output",
                                ),
                            ] {
                                selections.push(CaptureSelection {
                                    id: id.into(),
                                    path: path.into(),
                                    schedule: CaptureSchedule {
                                        decode: false,
                                        ..Default::default()
                                    },
                                    transform: CaptureTransform::FullTensor,
                                    slices: vec![],
                                });
                            }
                        }
                        selections
                    },
                    limits: CaptureLimits {
                        per_step: allowance,
                        cumulative: allowance.checked_mul(128).unwrap(),
                        physical_native_bytes: None,
                        on_limit: CaptureLimitPolicy::Fail,
                    },
                },
                interventions: eredu_core::intervention::InterventionPlan::none(),
                bounds: CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: 16,
                    max_context: None,
                    max_predictions: 28,
                },
            }
            .admit(&discovery)
            .unwrap(),
        )
    } else {
        None
    };
    let stream = runtime.backend().stream().clone();
    let binding = runtime.session().speculative_partition_binding();
    let DecodingMode::SelfSpeculation(options) = modes(adapter)[2] else {
        unreachable!()
    };
    let sampler = crate::composition::mlx::session::MlxTextSampler::from_config(
        &eredu_core::TextGenerationConfig::new(sampling()),
    )
    .unwrap();
    runtime
        .session_mut()
        .with_model_operation(|model| {
            let context =
                SpeculativeExecutionStreams::single(&stream).with_capture_binding(binding.as_ref());
            let draft = model
                .erased()
                .capability_estimate()
                .shared_backbone_draft()
                .unwrap()
                .prepare(options, 0.0)
                .unwrap();
            let mut executor = SharedBackboneExecutor::<MlxAutoregressiveMechanisms>::new(
                model.erased_mut(),
                std::num::NonZeroUsize::new(2).unwrap(),
            );
            if let Some(plan) = activations.clone() {
                executor
                    .configure_activation_capture(
                        plan,
                        eredu_core::SpeculativeRequestId::new(0),
                        context,
                    )
                    .unwrap();
            }
            let mut phases = Vec::new();
            let mut invocation = None;
            macro_rules! drain {
                () => {
                    while let Some(record) = executor.take_activation_capture() {
                        assert!(record.completed);
                        assert!(invocation.is_none_or(|previous| record.invocation > previous));
                        invocation = Some(record.invocation);
                        if vision && record.phase == eredu_core::speculative::SpeculativeActivationPhase::TargetPrefill {
                            let media = record.captures.invocation.unwrap().media.expect("exact media geometry precedes capture admission");
                            assert_eq!(media.patches, 4);
                            assert_eq!(media.projected, 1);
                            for (id, shape) in [("shared-vision-residual", vec![1,4,8]), ("shared-projector", vec![1,1,6])] {
                                let value = record.captures.records.iter().find(|value| value.selection_id == id).expect("selected public media point");
                                assert_eq!(value.outcome, eredu_core::capture::CaptureOutcome::Captured);
                                assert!(value.payload.is_some());
                                assert_eq!(value.source_shape.as_ref(), Some(&shape));
                            }
                        }
                        phases.push(record.phase);
                    }
                };
            }
            let mut cache = executor.new_cache(context)?;
            let output = SpeculativeOutputRuntime::new(
                MlxSpeculativeSampling::new(ConstrainedSampler::new(
                    sampler,
                    RingFilter(vocabulary),
                )),
                eredu_core::GenerationSequence::new(28, []),
                SpeculativeSemanticConstraint::plain(),
                SpeculativeCallbackPublisher::tokens(|_| Ok(())),
                Default::default(),
            );
            let mut table =
                GenerationRequestTable::new(Default::default(), context.topology()).unwrap();
            macro_rules! advance {
                () => {{
                    let advanced = table.step(&mut executor, false, context).unwrap();
                    drain!();
                    advanced
                }};
            }
            let id = table
                .submit(
                    &mut executor,
                    &mut cache,
                    SharedBackboneInput::new(
                        prompt(vision),
                        draft,
                        crate::backend::random::RandomState::with_seed(17)?,
                        crate::backend::random::RandomState::with_seed(29)?,
                    ),
                    GenerationRequestConfig {
                        max_tokens: 28,
                        ..Default::default()
                    },
                    output,
                    SpeculativeRandomness::new(None, None),
                    false,
                    context,
                )
                .map_err(|error| Error::Exception(safemlx::error::Exception::from_source(error)))?;
            drain!();
            if root_commands {
                table
                    .request(id)
                    .unwrap()
                    .request_decoding_mode(DecodingMode::SelfSpeculation(options))
                    .unwrap();
            }
            advance!();
            assert_eq!(
                table.request(id).unwrap().decoding_state().active,
                DecodingMode::SelfSpeculation(options)
            );
            assert_eq!(
                table
                    .request(id)
                    .unwrap()
                    .decoding_state()
                    .acknowledgement
                    .unwrap()
                    .committed_position,
                1
            );
            advance!();
            assert_eq!(table.request(id).unwrap().sequence().tokens().len(), 2);
            advance!();
            assert_eq!(table.request(id).unwrap().stats().draft_tokens(), 2);
            if root_commands {
                table
                    .request(id)
                    .unwrap()
                    .request_decoding_mode(DecodingMode::Autoregressive)
                    .unwrap();
            }
            loop {
                advance!();
                if table.request(id).unwrap().decoding_state().active
                    == DecodingMode::Autoregressive
                {
                    break;
                }
            }
            let diffusion = DecodingMode::Diffusion {
                policy: options.request().refinement,
            };
            for mode in [
                diffusion,
                DecodingMode::SelfSpeculation(options),
                diffusion,
                DecodingMode::Autoregressive,
            ] {
                if root_commands {
                    table
                        .request(id)
                        .unwrap()
                        .request_decoding_mode(mode)
                        .unwrap();
                }
                loop {
                    advance!();
                    if table.request(id).unwrap().decoding_state().active == mode
                        && table
                            .request(id)
                            .unwrap()
                            .decoding_state()
                            .pending
                            .is_none()
                    {
                        break;
                    }
                }
                assert!(
                    table
                        .request(id)
                        .unwrap()
                        .pending_committed_tokens()
                        .is_empty()
                );
                let position = table.request(id).unwrap().sequence().tokens().len();
                while table.request(id).unwrap().sequence().tokens().len() == position
                    || table.request(id).unwrap().has_tentative_refinement()
                {
                    advance!();
                }
            }
            let request = table.request(id).unwrap();
            assert_eq!(request.pending_committed_tokens().len(), 1);
            assert_eq!(
                request
                    .decoding_state()
                    .acknowledgement
                    .unwrap()
                    .committed_position
                    + 1,
                request.sequence().tokens().len() as u64
            );
            let budget = eredu_runtime::execution_control::SnapshotBudget::new(
                eredu_core::execution_control::SnapshotLimits {
                    max_snapshots: 1,
                    max_branches: 0,
                    retained_bytes: 128 << 20,
                    cumulative_copy_bytes: 512 << 20,
                },
            );
            let estimate = request.control_snapshot_estimate(&executor).unwrap();
            let _snapshot = budget
                .reserve(
                    eredu_core::execution_control::SnapshotResourceKind::Snapshot,
                    Some(estimate),
                )
                .unwrap();
            let saved = request.control_snapshot(&executor, context).unwrap();
            while advance!() {}
            let expected = table.request(id).unwrap().sequence().tokens().to_vec();
            let _restore = budget
                .reserve(
                    eredu_core::execution_control::SnapshotResourceKind::Restore,
                    Some(estimate),
                )
                .unwrap();
            table
                .request_mut(id)
                .unwrap()
                .restore_control_snapshot(&mut executor, &saved, context)
                .unwrap();
            while advance!() {}
            assert_eq!(table.request(id).unwrap().sequence().tokens(), expected);
            if activations.is_some() {
                assert!(
                    phases.contains(
                        &eredu_core::speculative::SpeculativeActivationPhase::TargetPrefill
                    )
                );
                assert!(phases.iter().any(|phase| matches!(
                    phase,
                    eredu_core::speculative::SpeculativeActivationPhase::MaskedDraft { .. }
                )));
                assert!(phases.iter().any(|phase| matches!(
                    phase,
                    eredu_core::speculative::SpeculativeActivationPhase::Refinement { .. }
                )));
                assert!(
                    phases.contains(
                        &eredu_core::speculative::SpeculativeActivationPhase::Verification
                    )
                );
            }
            Ok(expected)
        })
        .unwrap()
}

fn snapshot_branches<'a>(
    runtime: &mut ModelRuntime<MlxBackend<'a>>,
    vocabulary: usize,
    adapter: bool,
) {
    use eredu_core::{TextGenerationDriver, execution_control::SnapshotLimits};
    use eredu_runtime::execution_control::{
        ManagedTextContinuation, SnapshotBudget, TextBranchRequest, TextContinuationSnapshot,
    };
    runtime.reset().unwrap();
    let mut driver = TextGenerationDriver::new(runtime);
    let continuation = driver
        .start(
            prompt(true),
            eredu_core::TextGenerationConfig::new(sampling())
                .with_decoding_mode(Some(modes(adapter)[2])),
            RingFilter(vocabulary),
        )
        .unwrap();
    let mut state = ManagedTextContinuation::root(continuation);
    let mut committed = 0;
    while committed < 2 {
        let step = state.advance_operation(&mut driver).unwrap().unwrap();
        state.take_completed_step(&mut driver).unwrap();
        if matches!(step, TextAdvancement::Token(_)) {
            committed += 1;
        }
    }
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 128 << 20,
        cumulative_copy_bytes: 512 << 20,
    });
    let saved = TextContinuationSnapshot::capture(
        &mut state.boundary(&mut driver).unwrap(),
        &budget,
        Some(4096),
    )
    .unwrap();
    let growth = saved
        .native_continuation_growth(driver.runtime(), 10)
        .unwrap();
    let mut child = saved
        .fork(
            &mut state.boundary(&mut driver).unwrap(),
            &budget,
            TextBranchRequest {
                session_id: "pixtral-ring-fork",
                max_predictions: 10,
                capture_limits: None,
                intervention: None,
                host_bytes: Some(4096),
                continuation_growth_bytes: Some(growth + (1 << 20)),
            },
        )
        .unwrap();
    fn collect<'a>(
        driver: &mut TextGenerationDriver<'_, MlxBackend<'a>>,
        state: &mut ManagedTextContinuation<MlxBackend<'a>, RingFilter>,
    ) -> Vec<u32> {
        let mut output = Vec::new();
        while let Some(step) = state.advance_operation(driver).unwrap() {
            state.take_completed_step(driver).unwrap();
            if let TextAdvancement::Token(token) = step {
                output.push(token.token_id());
            }
        }
        output
    }
    let expected = collect(&mut driver, &mut state);
    let spent = budget.usage().cumulative_copy_bytes;
    saved
        .restore(&mut state.boundary(&mut driver).unwrap(), &budget)
        .unwrap();
    assert_eq!(collect(&mut driver, &mut state), expected);
    assert!(budget.usage().cumulative_copy_bytes > spent);
    child.exchange(&mut driver, &mut state).unwrap();
    assert_eq!(collect(&mut driver, &mut state), expected);
    child.exchange(&mut driver, &mut state).unwrap();
}

fn prompt(vision: bool) -> MlxModelInput {
    let mut parts = vec![input::token_ids_part(&Array::from_slice(&[1u32, 4], &[1, 2])).unwrap()];
    if vision {
        let patches = (0..48)
            .map(|i| (i as f32 - 23.0) / 37.0)
            .collect::<Vec<_>>();
        parts.push(
            input::input_part(
                eredu_core::InputModality::Image,
                input::InputPayload::Tensor(Array::from_slice(&patches, &[4, 12])),
                [(
                    eredu_core::InputMetadataKey::PatchGrid,
                    Array::from_slice(&[1i32, 2, 2], &[1, 3]),
                )],
                [eredu_core::InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 2,
                }],
            )
            .unwrap(),
        );
    }
    parts.push(input::token_ids_part(&Array::from_slice(&[2u32], &[1, 1])).unwrap());
    MlxModelInput::from(input::ModelInput::new(&parts))
}
const WORKER: &str = "EREDU_NEMOTRON_RING_WORKER";

fn modes(adapter: bool) -> [DecodingMode; 3] {
    [
        DecodingMode::Autoregressive,
        DecodingMode::Diffusion { policy: None },
        DecodingMode::SelfSpeculation(
            SpeculationOptionsRequest {
                max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
                draft_adapter: adapter,
                ..Default::default()
            }
            .try_into()
            .unwrap(),
        ),
    ]
}
fn sampling() -> eredu_core::generation::ResolvedGenerationConfig {
    eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(10),
            temperature: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        },
    )
    .unwrap()
}
fn output<'a>(
    runtime: &mut ModelRuntime<MlxBackend<'a>>,
    mode: DecodingMode,
    vision: bool,
    vocabulary: usize,
) -> Vec<u32> {
    runtime.reset().unwrap();
    ControlledTextGeneration::from_prompt(
        runtime,
        prompt(vision),
        eredu_core::TextGenerationConfig::new(sampling()).with_decoding_mode(Some(mode)),
        RingFilter(vocabulary),
    )
    .unwrap()
    .map(|token| token.unwrap().token_id())
    .collect()
}
fn handoffs<'a>(
    runtime: &mut ModelRuntime<MlxBackend<'a>>,
    request_owner: bool,
    vision: bool,
    vocabulary: usize,
    adapter: bool,
) -> Vec<Vec<u32>> {
    let mut trajectories = Vec::new();
    for (start, initial) in modes(adapter).into_iter().enumerate() {
        for (end, requested) in modes(adapter).into_iter().enumerate() {
            if start == end {
                continue;
            }
            runtime.reset().unwrap();
            let handle = GenerationControlHandle::default();
            let mut generation = ControlledTextGeneration::from_prompt(
                runtime,
                prompt(vision),
                eredu_core::TextGenerationConfig::new(sampling()).with_decoding_mode(Some(initial)),
                RingFilter(vocabulary),
            )
            .unwrap();
            generation.bind_control_handle(&handle).unwrap();
            let mut tokens = vec![generation.next().unwrap().unwrap().token_id()];
            if start == 2 {
                assert!(matches!(
                    generation.advance_operation().unwrap().unwrap(),
                    TextAdvancement::Progress(_)
                ));
            }
            if request_owner {
                handle.request_decoding_mode(requested).unwrap();
            }
            loop {
                match generation.advance_operation().unwrap().unwrap() {
                    TextAdvancement::Token(token) => tokens.push(token.token_id()),
                    TextAdvancement::Progress(_) => {}
                    TextAdvancement::ModeChanged(ack) => {
                        assert_eq!(ack.active, requested);
                        assert_eq!(ack.committed_position as usize, tokens.len());
                        break;
                    }
                }
            }
            tokens.push(generation.next().unwrap().unwrap().token_id());
            trajectories.push(tokens);
        }
    }
    trajectories
}

#[test]
fn native_shared_decoding_ring_worker() {
    let Some(root) = std::env::var_os(WORKER) else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    let vision = config.get("vision_config").is_some();
    let vocabulary = config["vocab_size"].as_u64().unwrap() as usize;
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("ring-expected.json")).unwrap()).unwrap();
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    let tp: usize = std::env::var("EREDU_NEMOTRON_RING_TP")
        .unwrap()
        .parse()
        .unwrap();
    let pp: usize = std::env::var("EREDU_NEMOTRON_RING_PP")
        .unwrap()
        .parse()
        .unwrap();
    let topology = eredu_core::ParallelTopology::new(tp, pp, 1, 1).unwrap();
    let rank = world.rank();
    let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights = Stream::new_with_device(&device);
    let adapter_source = std::env::var_os("EREDU_NEMOTRON_RING_ADAPTER")
        .map(|path| eredu_core::DraftAdapterSource::new(path, "nvidia/Nemotron-Labs-Diffusion-3B"));
    let adapter = adapter_source.is_some();
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        eprintln!("rank {rank}: {policy:?} loading vision={vision}");
        let backend = crate::native::distributed_backend(&stream, &weights, &world);
        let mut normalized = eredu_runtime::NormalizedLoadRequest::default()
            .with_weight_residency(eredu_runtime::WeightResidency::with_layers(policy.layers()));
        if let Some(source) = &adapter_source {
            normalized = normalized.with_draft_adapter(source.clone());
        }
        let request = crate::MlxLoadRequest::from_normalized(normalized)
            .with_parallel_topology(
                eredu_core::ParallelRankTopology::new(topology, rank).unwrap(),
                crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                16,
                crate::MlxLoadRequest::test_communication_completion_policy(),
            )
            .unwrap();
        let model = eredu_core::load_model(&backend, &root, request).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        if std::env::var_os("EREDU_NEMOTRON_RING_REQUEST_TABLE_ONLY").is_some() {
            assert_eq!(
                serde_json::to_value(request_table_handoffs(
                    &mut runtime,
                    rank == 0,
                    vision,
                    vocabulary,
                    adapter
                ))
                .unwrap(),
                expected["request_table"]
            );
            continue;
        }
        eprintln!("rank {rank}: loaded; AR vision={vision}");
        assert_eq!(
            serde_json::to_value(output(&mut runtime, modes(adapter)[0], vision, vocabulary))
                .unwrap(),
            expected["ar"]
        );
        eprintln!("rank {rank}: AR agrees; self vision={vision}");
        assert_eq!(
            serde_json::to_value(output(&mut runtime, modes(adapter)[2], vision, vocabulary))
                .unwrap(),
            expected["ar"]
        );
        eprintln!("rank {rank}: self agrees; handoffs vision={vision}");
        assert_eq!(
            serde_json::to_value(handoffs(
                &mut runtime,
                rank == 0,
                vision,
                vocabulary,
                adapter
            ))
            .unwrap(),
            expected["handoffs"]
        );
        if vision && tp == 2 && pp == 5 {
            eprintln!("rank {rank}: snapshot branches vision={vision}");
            snapshot_branches(&mut runtime, vocabulary, adapter);
        }
    }
}

#[test]
#[ignore = "spawns local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_shared_decoding_ring_tp_pp_and_combined_match_single_owner() {
    ring_matrix(false, false);
}

#[test]
#[ignore = "spawns local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_adapted_shared_decoding_ring_combined_vlm_matches_single_owner() {
    ring_matrix(true, false);
}

#[test]
#[ignore = "spawns local MLX Ring ranks with BF16 base weights; run explicitly"]
fn native_bfloat_shared_decoding_ring_matches_single_owner() {
    ring_matrix(false, true);
}

#[test]
#[ignore = "spawns local MLX Ring ranks with BF16 base and FP32 adapter weights"]
fn native_adapted_bfloat_shared_decoding_ring_combined_vlm_matches_single_owner() {
    ring_matrix(true, true);
}

fn ring_matrix(adapter: bool, bfloat: bool) {
    use std::{
        net::TcpListener,
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };
    struct Children(Vec<Child>);
    impl Drop for Children {
        fn drop(&mut self) {
            for child in &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eredu-architectures/tests/fixtures/nemotron_labs_diffusion/pixtral-numeric.json"
    )))
    .unwrap();
    let combined_only = adapter || std::env::var_os("EREDU_NEMOTRON_RING_COMBINED_ONLY").is_some();
    for (vision, mut config) in [(false, config()), (true, fixture["config"].clone())] {
        if combined_only && !vision {
            continue;
        }
        config["block_size"] = 3.into();
        if vision {
            config["vision_config"]["num_hidden_layers"] = 3.into();
        }
        let vocabulary = config["vocab_size"].as_u64().unwrap() as usize;
        config["num_attention_heads"] = 4.into();
        config["num_key_value_heads"] = 2.into();
        let adapter_root = adapter.then(|| super::adapter::fixture(config.clone()));
        let root = tiny_heterogeneous_artifact(config);
        if bfloat {
            convert_fixture_to_bfloat(root.path());
        }
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let mut normalized = eredu_runtime::NormalizedLoadRequest::default();
        if let Some(root) = &adapter_root {
            normalized = normalized.with_draft_adapter(eredu_core::DraftAdapterSource::new(
                root.path(),
                "nvidia/Nemotron-Labs-Diffusion-3B",
            ));
        }
        let model = eredu_core::load_model(
            &backend,
            root.path(),
            crate::MlxLoadRequest::from_normalized(normalized),
        )
        .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let expected = if std::env::var_os("EREDU_NEMOTRON_RING_REQUEST_TABLE_ONLY").is_some() {
            serde_json::json!({"request_table": request_table_handoffs(&mut runtime, true, vision, vocabulary, adapter)})
        } else {
            serde_json::json!({"ar": output(&mut runtime,modes(adapter)[0], vision, vocabulary), "handoffs": handoffs(&mut runtime,true, vision, vocabulary, adapter)})
        };
        drop(runtime);
        std::fs::write(
            root.path().join("ring-expected.json"),
            serde_json::to_vec(&expected).unwrap(),
        )
        .unwrap();
        let topologies = if vision {
            vec![(2, 1), (1, 5), (2, 5)]
        } else {
            vec![(2, 1), (1, 2), (2, 2)]
        };
        for (tp, pp) in topologies {
            if combined_only && (tp, pp) != (2, 5) {
                continue;
            }
            let participants = tp * pp;
            let sockets = (0..participants)
                .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
                .collect::<Vec<_>>();
            let hosts = sockets
                .iter()
                .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
                .collect::<Vec<_>>();
            let hostfile = root.path().join("ring-hosts.json");
            std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
            drop(sockets);
            let mut children = Children(Vec::new());
            for rank in 0..participants {
                let mut command = Command::new(std::env::current_exe().unwrap());
                if let Some(root) = &adapter_root {
                    command.env("EREDU_NEMOTRON_RING_ADAPTER", root.path());
                } else {
                    command.env_remove("EREDU_NEMOTRON_RING_ADAPTER");
                }
                children.0.push(command.args(["--exact",
                "composition::mlx::replicated_text::tests::nemotron_labs_diffusion::ring::native_shared_decoding_ring_worker", "--nocapture", "--test-threads=1"])
                .env(WORKER,root.path()).env("EREDU_NEMOTRON_RING_TP",tp.to_string()).env("EREDU_NEMOTRON_RING_PP",pp.to_string())
                .env("MLX_RANK",rank.to_string()).env("MLX_HOSTFILE",&hostfile).env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
            }
            let deadline = Instant::now() + Duration::from_secs(if adapter { 240 } else { 120 });
            loop {
                let statuses = children
                    .0
                    .iter_mut()
                    .map(|child| child.try_wait().unwrap())
                    .collect::<Vec<_>>();
                if statuses.iter().all(Option::is_some) {
                    break;
                }
                if Instant::now() >= deadline
                    || statuses.iter().flatten().any(|status| !status.success())
                {
                    for child in &mut children.0 {
                        let _ = child.kill();
                    }
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            let mut succeeded = true;
            let mut diagnostics = String::new();
            for (rank, child) in children.0.drain(..).enumerate() {
                let output = child.wait_with_output().unwrap();
                succeeded &= output.status.success();
                diagnostics.push_str(&format!(
                    "rank {rank}: {}\n{}\n{}\n",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            assert!(succeeded, "TP{tp}/PP{pp}: {diagnostics}");
        }
    }
}

fn convert_fixture_to_bfloat(root: &std::path::Path) {
    use safetensors::{
        Dtype, SafeTensors,
        tensor::{TensorView, serialize_to_file},
    };
    let path = root.join("model.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let converted = tensors
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            let (dtype, data) = if tensor.dtype() == Dtype::F32 {
                (
                    Dtype::BF16,
                    tensor
                        .data()
                        .chunks_exact(4)
                        .flat_map(|x| {
                            half::bf16::from_f32(f32::from_le_bytes(x.try_into().unwrap()))
                                .to_bits()
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                (tensor.dtype(), tensor.data().to_vec())
            };
            (name, tensor.shape().to_vec(), dtype, data)
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        converted.iter().map(|(name, shape, dtype, data)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        }),
        None,
        &path,
    )
    .unwrap();
}
