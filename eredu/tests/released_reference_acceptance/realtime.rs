use eredu_architectures::moshi::{
    prepare_moshi_realtime_with_support, EffectiveModelType, MoshiPreparationPlan,
    MoshiRealtimeExecution,
};
use eredu_backend_mlx::backend::random::RandomState;
use eredu_backend_mlx::backend::runtime::cache::MlxCacheBackend;
use eredu_backend_mlx::{
    backend::error::Error,
    native::{MlxRealtimeCompletion, MlxRealtimeExecutionContext, MlxRealtimeSupport},
    MlxTensor,
};
use eredu_core::{
    scheduler::{RequestId, SchedulerLimits},
    RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
};
use eredu_runtime::state::KeyValueState;
use eredu_runtime::{
    DenseDiskStreamLoadOptions, GenerationSampler, LayerwiseLoadOptions,
    RealtimeModelSessionIdentity, RealtimePayloadState, RealtimeSessionScheduler, WeightResidency,
};
use safemlx::{
    ops::{indexing::TryIndexOp, stack_axis},
    transforms::async_eval_with_event,
    Array, Stream,
};
use std::path::Path;

fn prepare(path: &Path) -> MoshiPreparationPlan {
    eredu_architectures::realtime::prepare_realtime_model(
        &eredu_architectures::realtime::RealtimeArtifactRequest::safetensors(path),
    )
    .map(|artifact| match artifact {
        eredu_architectures::realtime::PreparedRealtimeArtifact::Moshi(preparation) => preparation,
        _ => panic!("fixture must be a Moshi-family realtime artifact"),
    })
    .unwrap_or_else(|error| panic!("prepare realtime artifact {}: {error}", path.display()))
}

struct SelectedTestModel {
    backend: MlxRealtimeExecutionContext,
    model: MoshiRealtimeExecution<
        eredu_runtime::CategoricalRealtimeExecution<MlxRealtimeExecutionContext>,
    >,
}

type TestScheduler = RealtimeSessionScheduler<
    RealtimeInputFrame,
    RealtimePayloadState<KeyValueState<MlxCacheBackend>, MlxTensor>,
    eredu_core::RealtimeFrameScheduleState,
    RealtimeSampling,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
    eredu_runtime::CategoricalRealtimeFrame<
        MlxRealtimeExecutionContext,
        eredu_architectures::moshi::MoshiRealtimeOutputProjection,
    >,
>;

fn load_selected_test_model(
    backend: MlxRealtimeExecutionContext,
    preparation: MoshiPreparationPlan,
    options: eredu_core::ModelLoadOptions<
        eredu_runtime::NormalizedLoadRequest,
        eredu_backend_mlx::backend::DeviceAssignment,
    >,
) -> SelectedTestModel {
    let request = options.request();
    let rank = None;
    let selected =
        prepare_moshi_realtime_with_support(preparation, request, &MlxRealtimeSupport::new(false))
            .expect("select realtime model");
    let model = eredu_runtime::bind_prepared_realtime(&backend, selected, rank)
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
    model
        .backend
        .validate_execution(model.model.executor().queue())
        .unwrap();
    let generation = model
        .model
        .initialize_generation(&model.backend, sampling)
        .unwrap();
    scheduler.register(request, generation).unwrap();
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
                let plan = model
                    .model
                    .prepare_frame()
                    .map_err(Error::ArchitectureModel)?;
                eredu_runtime::submit_categorical_realtime_frame(
                    &model.backend,
                    model.model.executor(),
                    plan,
                    frame,
                    branch,
                )
                .map_err(|error| Error::Other(Box::new(error)))
            })
            .unwrap();
        if let Some((_, _, output)) = progress.committed.pop() {
            return output.into_host_output().unwrap();
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
}

fn host_i32(array: &Array) -> Vec<i32> {
    array
        .evaluated()
        .expect("evaluate realtime token array")
        .as_slice::<i32>()
        .to_vec()
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
    let execution = eredu_backend_mlx::backend::ExecutionContext::new(safemlx::Device::new(
        safemlx::DeviceType::Gpu,
        0,
    ));
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
            eredu_core::ModelLoadOptions::new(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
            ),
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
    let execution = eredu_backend_mlx::backend::ExecutionContext::new(safemlx::Device::new(
        safemlx::DeviceType::Gpu,
        0,
    ));
    let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let backend = MlxRealtimeExecutionContext::new(execution.stream(), &weights);
    let mut model = load_selected_test_model(
        backend,
        prepare(Path::new(&model_path)),
        eredu_core::ModelLoadOptions::default(),
    );
    let fixture = Array::load_safetensors(Path::new(&fixture_path), execution.stream())
        .expect("load native seeded fixture");
    run_native_seeded_fixture(&mut model, &fixture);
}
