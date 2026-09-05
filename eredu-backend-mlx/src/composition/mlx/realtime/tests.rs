use super::*;
use eredu_architectures::moshi::EffectiveModelType;
use eredu_checkpoint::{
    schema::StoredDtypeConstraint, AffineQuantization, StoredDtype, WeightQuantization,
};
use eredu_core::{
    scheduler::{RequestId, RequestStatus, SchedulerLimits},
    RealtimeFrameConvention, RealtimeFrameForcing, RealtimeFrameScheduleState, RealtimeInputFrame,
    RealtimeSampling, RealtimeSpeechConfig,
};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, ExecutionResidency, LayerwiseLoadOptions, RealtimeGenerationState,
    RealtimeModelSessionIdentity, RealtimePayloadState, RealtimeSessionScheduler, WeightResidency,
};
use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
use std::path::Path;

fn prepare(path: &Path) -> RealtimePreparationPlan {
    eredu_architectures::moshi::prepare_realtime_model(path)
        .unwrap_or_else(|error| panic!("prepare realtime artifact {}: {error}", path.display()))
}

struct SelectedTestModel {
    backend: MlxRealtimeExecutionContext,
    model: MoshiRealtimeExecution<MlxRealtimeExecution>,
}

type TestScheduler = RealtimeSessionScheduler<
    RealtimePayloadState<MlxKeyValueState, MlxTensor>,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
    MlxPrepublicationFrame,
>;

fn load_selected_test_model(
    backend: MlxRealtimeExecutionContext,
    preparation: RealtimePreparationPlan,
    options: MlxLoadRequest,
) -> SelectedTestModel {
    let selected =
        MlxRealtimeExecutionContext::select_realtime_execution(preparation, &options, false)
            .expect("select realtime model");
    let model = backend
        .materialize_realtime_execution(selected, options)
        .expect("load selected realtime model");
    SelectedTestModel { backend, model }
}

fn selected_scheduler(
    model: &SelectedTestModel,
    request: RequestId,
    sampling: RealtimeSampling,
) -> TestScheduler {
    let mut scheduler = TestScheduler::new(
        RealtimeModelSessionIdentity::from_selected(model.model.selected()),
        SchedulerLimits::new(1, 1).unwrap(),
    )
    .unwrap();
    let schedule = model.model.execution_config().frame_schedule().clone();
    let samplers =
        eredu_architectures::moshi::realtime_generation_samplers(&schedule, sampling).unwrap();
    let model_state = RealtimePayloadState::fresh(
        model
            .backend
            .new_realtime_model_state(&model.model)
            .unwrap(),
        schedule.clone(),
    );
    let random = model
        .backend
        .realize_random_state(sampling.is_stochastic().then_some(sampling.seed()))
        .unwrap();
    scheduler
        .register(
            request,
            RealtimeGenerationState::new(model_state, schedule, sampling, samplers, random)
                .unwrap(),
        )
        .unwrap();
    scheduler
}

fn drive_selected_frame(
    model: &mut SelectedTestModel,
    scheduler: &mut TestScheduler,
    request: RequestId,
    frame: RealtimeInputFrame,
) -> RealtimeOutputFrame {
    scheduler.enqueue(request, frame).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let mut progress = scheduler
            .run_local_turn(std::time::Instant::now(), |_, frame, branch| {
                model
                    .backend
                    .submit_realtime_frame(&mut model.model, frame, branch)
            })
            .unwrap();
        if let Some((_, _, output)) = progress.committed.pop() {
            return output.into_host_output().unwrap();
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
}

#[test]
fn realtime_session_capabilities_fail_closed_for_activation_inspection() {
    let available = realtime_session_capabilities();
    assert!(available.persistent_cache());
    assert!(available.output_observation());
    assert!(!available.activation_inspection());

    let options = MlxLoadRequest::default().with_required_session_capabilities(
        eredu_core::SessionCapabilities::default().with_activation_inspection(true),
    );
    let error = validate_realtime_session_requirements(&options).unwrap_err();
    match error {
        Error::SessionCapability(error) => {
            assert_eq!(error.capability(), "activation_inspection")
        }
        error => panic!("expected session capability error, got {error:?}"),
    }
}

#[test]
fn realtime_capabilities_fail_closed_for_unimplemented_mechanisms_and_lowerings() {
    let mechanisms = mlx_realtime_mechanisms(false);
    assert!(!mechanisms.contains(&RealtimeMechanism::Collectives));
    assert!(!mechanisms.contains(&RealtimeMechanism::Observation));
    assert!(!mechanisms.contains(&RealtimeMechanism::Timing));

    let unsupported_transform = eredu_runtime::WeightLoweringDescriptor::new(
        SourceTensorEncoding::Safetensors(StoredDtype::F32),
        LinearFormat::Dense,
        vec![2, 2],
        vec![2, 2],
        None,
    )
    .unwrap();
    assert!(!mlx_supports_realtime_lowering(
        &unsupported_transform,
        WeightLoweringKind::Transform,
    ));
}

#[test]
fn prepared_frame_tensor_mechanisms_preserve_canonical_matrix_geometry() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut host = MlxRealtimeFrameTensorMechanisms::new(&stream);
    let matrix = host
        .materialize_i32(&[1, 2, 3, 4], [2, 2])
        .expect("materialize portable frame matrix");
    assert_eq!(matrix.as_array().shape(), &[2, 2]);
    assert_eq!(matrix.as_array().dtype(), Dtype::Int32);

    let mut tensors = MlxRealtimeFrameTensorMechanisms::new(&stream);
    let left = tensors.column(&matrix, 0).expect("select left column");
    let right = tensors.column(&matrix, 1).expect("select right column");
    assert_eq!(left.as_array().shape(), &[2, 1]);
    let stacked = tensors
        .stack_columns(&[left, right], 2)
        .expect("stack canonical columns");
    assert_eq!(stacked.as_array().shape(), &[2, 2]);
    let empty = tensors
        .stack_columns(&[], 2)
        .expect("stack empty generated-codebook set");
    assert_eq!(empty.as_array().shape(), &[2, 0]);
    let padding = tensors
        .filled_column(7, 2)
        .expect("materialize padding column");
    assert_eq!(padding.as_array().shape(), &[2, 1]);
}

const TINY_NATIVE_CONFIG: &str = r#"{
    "model_type": "moshi",
    "dim": 32,
    "text_card": 32,
    "n_q": 2,
    "dep_q": 1,
    "generated_audio_codebooks": 1,
    "card": 32,
    "num_heads": 4,
    "num_layers": 1,
    "dim_feedforward": 48,
    "causal": true,
    "context": 7,
    "max_period": 10000.0,
    "positional_embedding": "rope",
    "depformer_dim": 32,
    "depformer_dim_feedforward": 48,
    "depformer_num_heads": 4,
    "depformer_num_layers": 1,
    "depformer_context": 3,
    "depformer_max_period": 10000.0,
    "depformer_pos_emb": "none",
    "delays": [0, 0, 1]
}"#;

#[derive(Debug, Eq, PartialEq)]
struct TinyFrameTokens {
    text: Vec<i32>,
    sampled_audio: Vec<i32>,
    output_audio: Option<Vec<i32>>,
}

pub(crate) fn write_tiny_native_artifact(
    directory: &Path,
    quantization: Option<WeightQuantization>,
) {
    let mut config_json =
        serde_json::from_str::<serde_json::Value>(TINY_NATIVE_CONFIG).expect("tiny native JSON");
    if let Some(quantization) = quantization {
        config_json.as_object_mut().unwrap().insert(
            "quantization".into(),
            serde_json::to_value(quantization).expect("serialize tiny quantization"),
        );
    }
    let config_json = serde_json::to_string_pretty(&config_json).unwrap();
    let config = eredu_architectures::moshi::MoshiConfig::from_json(&config_json)
        .expect("tiny native Moshi config");
    let plan = eredu_architectures::moshi::safetensors_plan(&config)
        .expect("tiny native SafeTensors plan");
    assert!(plan.layout_groups.is_empty());

    // Derive every physical name and shape from the strict architecture
    // catalog. Zero matrices make greedy decisions exact across dense and
    // load-time packed execution; unit normalization scales remain valid.
    let tensors = plan
        .common_tensors
        .iter()
        .map(|constraint| {
            let dtype = match &constraint.dtype {
                StoredDtypeConstraint::Exact(dtype) => dtype.clone(),
                StoredDtypeConstraint::Floating => StoredDtype::F32,
                StoredDtypeConstraint::OneOf(dtypes) => dtypes
                    .iter()
                    .find(|dtype| **dtype == StoredDtype::F32)
                    .or_else(|| dtypes.first())
                    .cloned()
                    .expect("validated catalog dtype set"),
            };
            let elements = constraint.shape.iter().product::<usize>();
            let (dtype, bytes) = match dtype {
                StoredDtype::F32 => {
                    let value =
                        if constraint.key.contains("norm") || constraint.key.ends_with(".scales") {
                            1.0f32
                        } else {
                            0.0f32
                        };
                    (
                        SafeDtype::F32,
                        std::iter::repeat_n(value, elements)
                            .flat_map(f32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                }
                StoredDtype::U32 => (
                    SafeDtype::U32,
                    std::iter::repeat_n(0u32, elements)
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
                StoredDtype::U8 => (
                    SafeDtype::U8,
                    vec![
                        if constraint.key.ends_with(".scales") {
                            127
                        } else {
                            0
                        };
                        elements
                    ],
                ),
                dtype => panic!("tiny native writer does not support {dtype:?}"),
            };
            (
                constraint.key.clone(),
                constraint.shape.clone(),
                dtype,
                bytes,
            )
        })
        .collect::<Vec<_>>();
    let views = tensors.iter().map(|(name, shape, dtype, bytes)| {
        (
            name.as_str(),
            TensorView::new(*dtype, shape.clone(), bytes).expect("catalog-derived tensor view"),
        )
    });
    std::fs::write(directory.join("config.json"), config_json).expect("write tiny native config");
    serialize_to_file(views, None, &directory.join("model.safetensors"))
        .expect("write tiny native SafeTensors artifact");
}

fn artifact_files(directory: &Path) -> std::collections::BTreeSet<String> {
    std::fs::read_dir(directory)
        .expect("read tiny artifact directory")
        .map(|entry| {
            entry
                .expect("tiny artifact entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

fn host_i32(array: &Array) -> Vec<i32> {
    array
        .evaluated()
        .expect("evaluate realtime token array")
        .as_slice::<i32>()
        .to_vec()
}

fn run_tiny_realtime_frames(model: &mut SelectedTestModel) -> Vec<TinyFrameTokens> {
    let request = RequestId::new(8);
    let mut scheduler = selected_scheduler(model, request, RealtimeSampling::greedy());
    let inputs = [
        RealtimeInputFrame::new(1, vec![1]),
        RealtimeInputFrame::new(1, vec![2])
            .with_forced_text(vec![7])
            .with_forced_generated_audio(vec![9]),
        RealtimeInputFrame::new(1, vec![3]).with_forced_text(vec![11]),
        RealtimeInputFrame::new(1, vec![4]).with_forced_generated_audio(vec![13]),
        RealtimeInputFrame::new(1, vec![5]),
    ];
    let frames = inputs
        .into_iter()
        .map(|input| {
            let output = drive_selected_frame(model, &mut scheduler, request, input);
            TinyFrameTokens {
                text: output.text_tokens().to_vec(),
                sampled_audio: output.sampled_audio_tokens().to_vec(),
                output_audio: output.output_audio_tokens().map(<[i32]>::to_vec),
            }
        })
        .collect();
    scheduler
        .finish(request)
        .expect("finish tiny realtime request");
    frames
}

fn verify_tiny_native_hardware_matrix() {
    let directory = tempfile::tempdir().expect("tiny native artifact directory");
    write_tiny_native_artifact(directory.path(), None);
    let original_files = artifact_files(directory.path());
    assert_eq!(
        original_files,
        std::collections::BTreeSet::from([
            "config.json".to_string(),
            "model.safetensors".to_string(),
        ])
    );

    let device = safemlx::Device::new(safemlx::DeviceType::Gpu, 0);
    let weights_device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights_stream = Stream::new_with_device(&weights_device);
    let policies = [
        (
            WeightResidency::fully_resident(),
            ExecutionResidency::FullyResident,
        ),
        (
            WeightResidency::layerwise_host(LayerwiseLoadOptions::default()),
            ExecutionResidency::LayerwiseHost,
        ),
        (
            WeightResidency::dense_disk_stream(
                DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1)
                    .expect("tiny dense stream policy"),
            ),
            ExecutionResidency::DenseDiskStream,
        ),
    ];
    let expected = vec![
        TinyFrameTokens {
            text: vec![0],
            sampled_audio: vec![0],
            output_audio: None,
        },
        TinyFrameTokens {
            text: vec![7],
            sampled_audio: vec![9],
            output_audio: Some(vec![0]),
        },
        TinyFrameTokens {
            text: vec![11],
            sampled_audio: vec![0],
            output_audio: Some(vec![9]),
        },
        TinyFrameTokens {
            text: vec![0],
            sampled_audio: vec![13],
            output_audio: Some(vec![0]),
        },
        TinyFrameTokens {
            text: vec![0],
            sampled_audio: vec![0],
            output_audio: Some(vec![13]),
        },
    ];

    for (residency, execution) in policies {
        let backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
        let mut model = load_selected_test_model(
            backend,
            prepare(directory.path()),
            MlxLoadRequest::default().with_weight_residency(residency),
        );
        assert_eq!(model.model.executor().metadata().residency(), execution);
        assert_eq!(run_tiny_realtime_frames(&mut model), expected);
        let report = model
            .model
            .executor()
            .residency_report()
            .expect("tiny residency report");
        assert!(report.initialized());
        assert!(report.weight_store().physical_reads > 0);
        if execution == ExecutionResidency::DenseDiskStream {
            let dense = model
                .model
                .executor()
                .dense_stream_report()
                .expect("tiny dense stream report")
                .expect("selected dense-stream policy has a report");
            assert!(dense.planned_layer_count() > 0);
            assert!(dense.decode_forwards() > 0);
        }
    }

    for (request, quantization) in [
        (
            eredu_core::QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            },
            WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
        ),
        (
            eredu_core::QuantizationRequest::MxFp4,
            WeightQuantization::MxFp4,
        ),
    ] {
        let backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
        let mut model = load_selected_test_model(
            backend,
            prepare(directory.path()),
            MlxLoadRequest::with_quantization(request),
        );
        let metadata = model.model.executor().metadata();
        assert_eq!(metadata.quantization(), Some(quantization));
        let materialization = metadata
            .materialization()
            .expect("load-time quantization telemetry");
        assert!(materialization.transformed_weights > 0);
        assert!(materialization.source_bytes_read > 0);
        assert!(materialization.output_bytes > 0);
        assert_eq!(run_tiny_realtime_frames(&mut model), expected);
        drop(model);
        assert_eq!(
            artifact_files(directory.path()),
            original_files,
            "load-time {quantization:?} created a disk artifact"
        );
    }

    for quantization in [
        WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap()),
        WeightQuantization::MxFp4,
    ] {
        let packed_directory = tempfile::tempdir().expect("tiny packed artifact directory");
        write_tiny_native_artifact(packed_directory.path(), Some(quantization));
        let original_files = artifact_files(packed_directory.path());
        let backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
        let mut model = load_selected_test_model(
            backend,
            prepare(packed_directory.path()),
            MlxLoadRequest::default(),
        );
        let metadata = model.model.executor().metadata();
        assert_eq!(metadata.quantization(), Some(quantization));
        assert_eq!(metadata.materialization(), None);
        assert_eq!(run_tiny_realtime_frames(&mut model), expected);
        drop(model);
        assert_eq!(artifact_files(packed_directory.path()), original_files);
    }
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn moshi_mlx_scheduler_transaction_rollback_release_resume() {
    let directory = tempfile::tempdir().expect("tiny scheduler artifact directory");
    write_tiny_native_artifact(directory.path(), None);
    let execution =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxRealtimeExecutionContext::new(execution.stream(), &weights);
    let mut model = load_selected_test_model(
        backend,
        prepare(directory.path()),
        MlxLoadRequest::default(),
    );
    let request = RequestId::new(81);
    let mut scheduler = selected_scheduler(&model, request, RealtimeSampling::greedy());

    drive_selected_frame(
        &mut model,
        &mut scheduler,
        request,
        RealtimeInputFrame::new(1, vec![1]),
    );
    assert_eq!(
        scheduler
            .request_state(request)
            .unwrap()
            .generation()
            .schedule_state()
            .frontier(),
        1
    );
    let released = scheduler.release(request).unwrap();
    scheduler.resume(request, released).unwrap();

    drive_selected_frame(
        &mut model,
        &mut scheduler,
        request,
        RealtimeInputFrame::new(1, vec![2])
            .with_forced_text(vec![7])
            .with_forced_generated_audio(vec![9]),
    );
    assert_eq!(
        scheduler
            .request_state(request)
            .unwrap()
            .generation()
            .schedule_state()
            .frontier(),
        2
    );
    let released = scheduler.release(request).unwrap();
    scheduler.resume(request, released).unwrap();

    scheduler
        .enqueue(request, RealtimeInputFrame::new(1, vec![-1]))
        .unwrap();
    let error = match scheduler.run_local_turn(std::time::Instant::now(), |_, frame, branch| {
        model
            .backend
            .submit_realtime_frame(&mut model.model, frame, branch)
    }) {
        Ok(_) => panic!("invalid realtime input unexpectedly submitted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("Audio token -1"), "{error}");
    assert_eq!(
        scheduler.request_status(request),
        Some(RequestStatus::Failed)
    );
    assert!(scheduler.request_state(request).is_none());
}

#[test]
#[ignore = "runs the MLX operator, transaction, and tiny native model conformance suite"]
fn moshi_mlx_conformance_suite() {
    verify_tiny_native_hardware_matrix();
    const TESTS: &[&str] = &[
        "backend::nn::shared::neutral_semantic_operator_tests::mlx_dense_fused_projection_equivalence",
        "backend::nn::shared::neutral_semantic_operator_tests::mlx_affine_fused_projection_equivalence",
        "backend::nn::shared::neutral_semantic_operator_tests::mlx_mxfp4_fused_projection_equivalence",
        "backend::nn::shared::neutral_semantic_operator_tests::mlx_sentinel_embedding_validation",
        "backend::nn::shared::neutral_semantic_operator_tests::mlx_multi_table_embedding_sum_is_ordered_and_sentinel_safe",
        "backend::runtime::cache::state::semantic_transaction_tests::paged_depth_segment_reset_preserves_temporal_pages_and_later_rollback",
        "backend::runtime::cache::state::semantic_transaction_tests::mlx_realtime_transaction_paged_rollback_release_resume",
        "backend::runtime::residency::manager::tests::cross_unit_alias_reacquisition_reuses_one_pinned_owner_read",
        "backend::runtime::generation::backend::tests::mlx_token_domain_validation_is_deferred_to_completion",
        "composition::mlx::realtime::tests::mlx_realtime_input_domains_are_deferred_and_strict",
    ];
    let executable = std::env::current_exe().expect("current unit-test executable");
    for test in TESTS {
        let output = std::process::Command::new(&executable)
            .args(["--exact", test, "--ignored", "--nocapture"])
            .output()
            .unwrap_or_else(|error| {
                panic!("failed to launch MLX conformance test {test}: {error}")
            });
        assert!(
            output.status.success(),
            "MLX conformance test {test} failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

#[test]
fn partial_forcing_and_initialization_only_transition_are_portable() {
    let schedule = RealtimeSpeechConfig::new(
        4,
        2,
        2,
        3,
        100,
        64,
        RealtimeFrameConvention::AbsoluteDelayedSlots,
        vec![0, 0, 1, 0, 1],
    )
    .unwrap();
    let mut state = RealtimeFrameScheduleState::new(schedule.clone());
    let transition = state
        .advance(
            &schedule,
            &RealtimeFrameForcing::new(false, vec![true, false]),
        )
        .unwrap();
    assert!(!transition.model_call_required());
    assert_eq!(transition.forced_placements().len(), 1);
}

#[test]
fn portable_realtime_input_domains_are_strict_before_materialization() {
    let config = eredu_architectures::moshi::MoshiConfig::from_json(TINY_NATIVE_CONFIG)
        .expect("tiny native Moshi config");
    let ingress = eredu_architectures::moshi::realtime_ingress_contract(&config).unwrap();
    ingress
        .validate(
            &RealtimeInputFrame::new(1, vec![32])
                .with_forced_text(vec![32])
                .with_forced_generated_audio(vec![32]),
        )
        .unwrap();
    for input in [
        RealtimeInputFrame::new(1, vec![-1]),
        RealtimeInputFrame::new(1, vec![33]),
        RealtimeInputFrame::new(1, vec![0]).with_forced_text(vec![33]),
        RealtimeInputFrame::new(1, vec![0]).with_forced_generated_audio(vec![33]),
    ] {
        assert!(ingress.validate(&input).is_err());
    }
}

fn required_fixture_array<'a>(
    fixture: &'a std::collections::HashMap<String, Array>,
    key: &str,
) -> &'a Array {
    fixture
        .get(key)
        .unwrap_or_else(|| panic!("teacher-forced fixture is missing tensor {key}"))
}

fn assert_token_array_equal(actual: &Array, expected: &Array, label: &str, stream: &Stream) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape mismatch for {label}"
    );
    assert!(
        actual
            .eq(expected, stream)
            .expect("token comparison")
            .all(None, stream)
            .expect("token comparison reduction")
            .item::<bool>(stream),
        "token fixture differs at {label}"
    );
}

fn run_personaplex_frame_fixture(
    model: &mut SelectedTestModel,
    fixture: &std::collections::HashMap<String, Array>,
    prefix: &str,
    forced: bool,
) {
    let stream = model.backend.stream().clone();
    let user_key = if forced {
        format!("{prefix}.user_audio")
    } else {
        format!("{prefix}.input_audio")
    };
    let user = required_fixture_array(fixture, &user_key);
    let agent = forced.then(|| required_fixture_array(fixture, &format!("{prefix}.agent_audio")));
    let text = forced.then(|| required_fixture_array(fixture, &format!("{prefix}.text")));
    let request = RequestId::new(91);
    let mut scheduler = selected_scheduler(model, request, RealtimeSampling::greedy());
    let mut sampled = Vec::new();
    let mut output_audio = Vec::new();
    let mut emitted_steps = Vec::new();
    for step in 0..user.dim(2) {
        let user_step = user
            .try_index_device((.., .., step), &stream)
            .expect("PersonaPlex user frame");
        let mut input = RealtimeInputFrame::new(
            usize::try_from(user_step.dim(0)).unwrap(),
            host_i32(&user_step),
        );
        if let (Some(agent), Some(text)) = (agent, text) {
            let agent_step = agent
                .try_index_device((.., .., step), &stream)
                .expect("PersonaPlex agent frame");
            let text_step = text
                .try_index_device((.., .., step), &stream)
                .expect("PersonaPlex text frame");
            input = input
                .with_forced_generated_audio(host_i32(&agent_step))
                .with_forced_text(host_i32(&text_step));
        }
        let output = drive_selected_frame(model, &mut scheduler, request, input);
        if step > 0 {
            let values = output
                .text_tokens()
                .iter()
                .chain(output.sampled_audio_tokens())
                .copied()
                .collect::<Vec<_>>();
            sampled.push(Array::from_slice(
                &values,
                &[
                    i32::try_from(output.batch()).unwrap(),
                    i32::try_from(values.len() / output.batch()).unwrap(),
                ],
            ));
        }
        if let Some(audio) = output.output_audio_tokens() {
            output_audio.push(Array::from_slice(
                audio,
                &[
                    i32::try_from(output.batch()).unwrap(),
                    i32::try_from(audio.len() / output.batch()).unwrap(),
                ],
            ));
            emitted_steps.push(step);
        }
    }
    scheduler.finish(request).unwrap();
    let sampled = stack_axis(&sampled, 2, &stream).expect("PersonaPlex sampled transcript");
    let output_audio =
        stack_axis(&output_audio, 2, &stream).expect("PersonaPlex delayed audio transcript");
    let emitted_steps = Array::from_slice(&emitted_steps, &[output_audio.dim(2)]);
    async_eval_with_event([&sampled, &output_audio, &emitted_steps])
        .unwrap()
        .synchronize()
        .unwrap();
    assert_token_array_equal(
        &sampled,
        required_fixture_array(fixture, &format!("{prefix}.expected_sampled")),
        &format!("{prefix}.expected_sampled"),
        &stream,
    );
    assert_token_array_equal(
        &output_audio,
        required_fixture_array(fixture, &format!("{prefix}.expected_output_audio")),
        &format!("{prefix}.expected_output_audio"),
        &stream,
    );
    assert_token_array_equal(
        &emitted_steps,
        required_fixture_array(fixture, &format!("{prefix}.expected_emitted_steps")),
        &format!("{prefix}.expected_emitted_steps"),
        &stream,
    );
}

fn run_native_seeded_fixture(
    model: &mut SelectedTestModel,
    fixture: &std::collections::HashMap<String, Array>,
) {
    let stream = model.backend.stream().clone();
    let input = required_fixture_array(fixture, "generation.input_audio");
    let seed = required_fixture_array(fixture, "generation.seeded.seed")
        .clone()
        .item::<i64>(&stream) as u64;
    let text_temperature = required_fixture_array(fixture, "generation.seeded.text_temperature")
        .clone()
        .item::<f32>(&stream);
    let audio_temperature = required_fixture_array(fixture, "generation.seeded.audio_temperature")
        .clone()
        .item::<f32>(&stream);
    let request = RequestId::new(92);
    let sampling = RealtimeSampling::new(text_temperature, audio_temperature, seed).unwrap();
    let mut scheduler = selected_scheduler(model, request, sampling);
    let mut text = Vec::new();
    let mut audio = Vec::new();
    for step in 0..input.dim(2) {
        let frame = input
            .try_index_device((.., .., step), &stream)
            .expect("native seeded input frame");
        let output = drive_selected_frame(
            model,
            &mut scheduler,
            request,
            RealtimeInputFrame::new(usize::try_from(frame.dim(0)).unwrap(), host_i32(&frame)),
        );
        text.push(Array::from_slice(
            output.text_tokens(),
            &[i32::try_from(output.batch()).unwrap()],
        ));
        if let Some(tokens) = output.output_audio_tokens() {
            audio.push(Array::from_slice(
                tokens,
                &[
                    i32::try_from(output.batch()).unwrap(),
                    i32::try_from(tokens.len() / output.batch()).unwrap(),
                ],
            ));
        }
    }
    scheduler.finish(request).unwrap();
    let text = stack_axis(&text, 1, &stream).unwrap();
    let audio = if audio.is_empty() {
        Array::zeros::<i32>(
            &[
                input.dim(0),
                model
                    .model
                    .execution_config()
                    .frame_schedule()
                    .generated_audio_codebooks() as i32,
                0,
            ],
            &stream,
        )
        .unwrap()
    } else {
        stack_axis(&audio, 2, &stream).unwrap()
    };
    async_eval_with_event([&text, &audio])
        .unwrap()
        .synchronize()
        .unwrap();
    assert_token_array_equal(
        &text,
        required_fixture_array(fixture, "generation.seeded.expected_text"),
        "generation.seeded.expected_text",
        &stream,
    );
    assert_token_array_equal(
        &audio,
        required_fixture_array(fixture, "generation.seeded.expected_audio"),
        "generation.seeded.expected_audio",
        &stream,
    );
}

#[test]
#[ignore = "requires released PersonaPlex artifact and PyTorch realtime fixture"]
fn moshi_personaplex_prompt_realtime_and_residency_parity() {
    let model_path = std::env::var_os("EREDU_PERSONAPLEX_FIXTURE").expect(
        "EREDU_PERSONAPLEX_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
    );
    let fixture_path = std::env::var_os("EREDU_PERSONAPLEX_TEACHER_FIXTURE")
        .expect("EREDU_PERSONAPLEX_TEACHER_FIXTURE must accompany the model fixture");
    let execution =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let fixture = Array::load_safetensors(Path::new(&fixture_path), execution.stream())
        .expect("load PersonaPlex parity fixture");
    for residency in [
        WeightResidency::fully_resident(),
        WeightResidency::layerwise_host(LayerwiseLoadOptions::default()),
        WeightResidency::dense_disk_stream(
            DenseDiskStreamLoadOptions::new(1 << 30, 1 << 30, 1, 1).unwrap(),
        ),
    ] {
        let backend = MlxRealtimeExecutionContext::new(execution.stream(), &weights);
        let mut model = load_selected_test_model(
            backend,
            prepare(Path::new(&model_path)),
            MlxLoadRequest::default().with_weight_residency(residency),
        );
        assert_eq!(
            model.model.execution_config().effective_model_type(),
            EffectiveModelType::PersonaPlex
        );
        run_personaplex_frame_fixture(&mut model, &fixture, "generation", false);
        run_personaplex_frame_fixture(&mut model, &fixture, "prompt", true);
    }
}

#[test]
#[ignore = "requires released native Moshi artifact and seeded MLX fixture"]
fn moshi_native_multiframe_seeded_realtime_parity() {
    let model_path = std::env::var_os("EREDU_MOSHI_FIXTURE").expect(
        "EREDU_MOSHI_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
    );
    let fixture_path = std::env::var_os("EREDU_MOSHI_TEACHER_FIXTURE")
        .expect("EREDU_MOSHI_TEACHER_FIXTURE must accompany the model fixture");
    let execution =
        crate::backend::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxRealtimeExecutionContext::new(execution.stream(), &weights);
    let mut model = load_selected_test_model(
        backend,
        prepare(Path::new(&model_path)),
        MlxLoadRequest::default(),
    );
    let fixture = Array::load_safetensors(Path::new(&fixture_path), execution.stream())
        .expect("load native seeded fixture");
    run_native_seeded_fixture(&mut model, &fixture);
}

#[test]
#[ignore = "requires EREDU_MOSHI_FIXTURE and an MLX runtime"]
fn moshi_neutral_session_hook() {
    let fixture = std::env::var_os("EREDU_MOSHI_FIXTURE").expect(
        "EREDU_MOSHI_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
    );
    assert!(
        Path::new(&fixture).exists(),
        "EREDU_MOSHI_FIXTURE does not exist: {}",
        Path::new(&fixture).display()
    );
    let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let backend = MlxRealtimeExecutionContext::new(&stream, &stream);
    let scheduler_model = load_selected_test_model(
        backend,
        prepare(Path::new(&fixture)),
        MlxLoadRequest::default(),
    );
    let request = RequestId::new(93);
    let _scheduler = selected_scheduler(&scheduler_model, request, RealtimeSampling::greedy());
}

#[test]
#[ignore = "requires EREDU_PERSONAPLEX_FIXTURE and an MLX runtime"]
fn moshi_personaplex_fixture_session_hook() {
    let fixture = std::env::var_os("EREDU_PERSONAPLEX_FIXTURE").expect(
        "EREDU_PERSONAPLEX_FIXTURE must point at a released artifact when this ignored fixture test is explicitly enabled",
    );
    assert!(
        Path::new(&fixture).exists(),
        "EREDU_PERSONAPLEX_FIXTURE does not exist: {}",
        Path::new(&fixture).display()
    );
    let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let backend = MlxRealtimeExecutionContext::new(&stream, &stream);
    let scheduler_model = load_selected_test_model(
        backend,
        prepare(Path::new(&fixture)),
        MlxLoadRequest::default(),
    );
    assert_eq!(
        scheduler_model
            .model
            .execution_config()
            .effective_model_type(),
        EffectiveModelType::PersonaPlex
    );
    let request = RequestId::new(94);
    let _scheduler = selected_scheduler(&scheduler_model, request, RealtimeSampling::greedy());
}
