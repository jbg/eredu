#[path = "realtime_fixtures.rs"]
mod fixtures;

use eredu_backend_mlx::backend::runtime::cache::MlxCacheBackend;
use eredu_runtime::state::KeyValueState;
use std::{
    cell::Cell,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use eredu_backend_mlx::{
    backend::DeviceAssignment,
    native::{
        MlxRealtimeCompletion, MlxRealtimeConsensusTransport, MlxRealtimeExecutionContext,
        RandomState,
    },
};
use eredu_core::scheduler::{RequestId, SchedulerLimits};
use eredu_core::{RealtimeInputFrame, RealtimeSampling};
use eredu_runtime::{GenerationSampler, RealtimeSessionScheduler};
use safemlx::{
    distributed::{self, Backend},
    Array, DeviceType, Stream,
};

const MODEL_WORKER_RANK: &str = "EREDU_MOSHI_RING_MODEL_WORKER";
const MODEL_WORKER_FIXTURE: &str = "EREDU_MOSHI_RING_MODEL_FIXTURE";
const MODEL_WORKER_PROFILE: &str = "EREDU_MOSHI_RING_MODEL_PROFILE";
const MODEL_WORKER_DISAGREE: &str = "EREDU_MOSHI_RING_MODEL_DISAGREE";
const NATIVE_FIXTURE: &str = "EREDU_MOSHI_NATIVE_FIXTURE";
const PERSONAPLEX_FIXTURE: &str = "EREDU_MOSHI_PERSONAPLEX_FIXTURE";

type PreparedRingRealtime = eredu_architectures::moshi::MoshiRealtimeExecution<
    eredu_runtime::CategoricalRealtimeExecution<MlxRealtimeExecutionContext>,
>;

fn push_tokens(serialized: &mut Vec<i32>, values: &[i32]) {
    serialized.push(i32::try_from(values.len()).unwrap());
    serialized.extend_from_slice(values);
}

type NativeScheduler = RealtimeSessionScheduler<
    RealtimeInputFrame,
    eredu_runtime::RealtimePayloadState<
        KeyValueState<MlxCacheBackend>,
        eredu_backend_mlx::MlxTensor,
    >,
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

fn submit_frame(
    backend: &MlxRealtimeExecutionContext,
    model: &PreparedRingRealtime,
    frame: &RealtimeInputFrame,
    branch: &mut eredu_runtime::CategoricalFrameBranch<MlxRealtimeExecutionContext>,
) -> Result<
    eredu_runtime::CategoricalRealtimeFrame<
        MlxRealtimeExecutionContext,
        eredu_architectures::moshi::MoshiRealtimeOutputProjection,
    >,
    eredu_backend_mlx::backend::error::Error,
> {
    let plan = model
        .prepare_frame()
        .map_err(eredu_backend_mlx::backend::error::Error::ArchitectureModel)?;
    eredu_runtime::submit_categorical_realtime_frame(backend, model.executor(), plan, frame, branch)
        .map_err(|error| eredu_backend_mlx::backend::error::Error::Other(Box::new(error)))
}

fn run_forced_and_greedy_sequence(
    backend: &MlxRealtimeExecutionContext,
    model: &mut PreparedRingRealtime,
    consensus: Option<&MlxRealtimeConsensusTransport>,
) -> Vec<i32> {
    let stream = backend.stream().clone();
    let schedule = model.execution_config().frame_schedule().clone();
    let input_codebooks = schedule.input_audio_codebooks();
    let generated_codebooks = schedule.generated_audio_codebooks();
    let request = RequestId::new(7);
    let mut scheduler = NativeScheduler::new(
        eredu_runtime::RealtimeModelSessionIdentity::from_selected(model.selected()),
        SchedulerLimits::new(1, 1).unwrap(),
    )
    .unwrap();
    let sampling = RealtimeSampling::greedy();
    backend
        .validate_execution(model.executor().queue())
        .unwrap();
    let generation = model.initialize_generation(backend, sampling).unwrap();
    scheduler.register(request, generation).unwrap();

    let forced_audio = Array::from_slice(
        &(0..generated_codebooks)
            .map(|codebook| i32::try_from(codebook + 2).unwrap())
            .collect::<Vec<_>>(),
        &[1, i32::try_from(generated_codebooks).unwrap()],
    );
    let partial_mask = (0..generated_codebooks)
        .map(|codebook| codebook != 0)
        .collect::<Vec<_>>();
    let mut serialized = Vec::new();
    for frame in 0..4 {
        let input = (0..input_codebooks)
            .map(|codebook| i32::try_from((frame + codebook) % 7).unwrap())
            .collect::<Vec<_>>();
        let input = match frame {
            0 | 3 => RealtimeInputFrame::new(1, input),
            1 => RealtimeInputFrame::new(1, input)
                .with_forced_text(vec![1])
                .with_forced_generated_audio(
                    forced_audio.evaluated().unwrap().as_slice::<i32>().to_vec(),
                ),
            2 => RealtimeInputFrame::new(1, input)
                .with_forced_text(vec![1])
                .with_partially_forced_generated_audio(
                    forced_audio.evaluated().unwrap().as_slice::<i32>().to_vec(),
                    partial_mask.clone(),
                ),
            _ => unreachable!(),
        };
        scheduler.enqueue(request, input).unwrap();
        let output = loop {
            let mut progress = match consensus {
                Some(transport) => scheduler.run_distributed_turn(
                    0x4d4f_5348_4954_5032,
                    transport,
                    Instant::now(),
                    |_, frame, branch| submit_frame(backend, model, frame, branch),
                ),
                None => scheduler.run_local_turn(Instant::now(), |_, frame, branch| {
                    submit_frame(backend, model, frame, branch)
                }),
            }
            .unwrap();
            if let Some((_, _, output)) = progress.committed.pop() {
                break output.into_host_output().unwrap();
            }
            thread::yield_now();
        };
        push_tokens(&mut serialized, output.text_tokens());
        push_tokens(&mut serialized, output.sampled_audio_tokens());
        match output.output_audio_tokens() {
            Some(tokens) => {
                serialized.push(1);
                push_tokens(&mut serialized, tokens);
            }
            None => serialized.push(0),
        }
    }
    scheduler.finish(request).unwrap();
    stream.synchronize().unwrap();
    serialized
}

fn assert_schedule_disagreement_submits_nothing(
    backend: &MlxRealtimeExecutionContext,
    model: &mut PreparedRingRealtime,
    consensus: &MlxRealtimeConsensusTransport,
    rank: usize,
) {
    let schedule = model.execution_config().frame_schedule().clone();
    let sampling = RealtimeSampling::greedy();
    let mut scheduler = NativeScheduler::new(
        eredu_runtime::RealtimeModelSessionIdentity::from_selected(model.selected()),
        SchedulerLimits::new(1, 1).unwrap(),
    )
    .unwrap();
    let request = RequestId::new(99);
    backend
        .validate_execution(model.executor().queue())
        .unwrap();
    let generation = model.initialize_generation(backend, sampling).unwrap();
    scheduler.register(request, generation).unwrap();
    let mut input = vec![0; schedule.input_audio_codebooks()];
    input[0] = i32::try_from(rank).unwrap();
    scheduler
        .enqueue(request, RealtimeInputFrame::new(1, input))
        .unwrap();

    let submissions = Cell::new(0usize);
    let result = scheduler.run_distributed_turn(
        0x4d4f_5348_4954_5032,
        consensus,
        Instant::now(),
        |_, frame, branch| {
            submissions.set(submissions.get() + 1);
            submit_frame(backend, model, frame, branch)
        },
    );
    let error = match result {
        Ok(_) => panic!("rank {rank} accepted a divergent realtime schedule"),
        Err(error) => error,
    };
    assert!(
        matches!(error, eredu_core::scheduler::SchedulerError::Consensus(_)),
        "rank {rank} returned an unexpected disagreement error: {error}"
    );
    assert_eq!(submissions.get(), 0, "rank {rank} submitted divergent work");
    assert_eq!(
        scheduler.report().completed_work,
        0,
        "rank {rank} published divergent work"
    );
}

#[test]
fn moshi_ring_model_parity_worker() {
    let Some(rank) = std::env::var_os(MODEL_WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let fixture = std::env::var_os(MODEL_WORKER_FIXTURE).unwrap();
    let expected_profile = std::env::var(MODEL_WORKER_PROFILE).unwrap();
    let device = DeviceAssignment::new(DeviceType::Cpu, 0);
    let preparation = eredu_architectures::realtime::prepare_realtime_model(
        &eredu_architectures::realtime::RealtimeArtifactRequest::safetensors(Path::new(&fixture)),
    )
    .map(|artifact| match artifact {
        eredu_architectures::realtime::PreparedRealtimeArtifact::Moshi(preparation) => preparation,
        _ => panic!("fixture must be a Moshi-family realtime artifact"),
    })
    .unwrap();
    let replicated_selected = eredu_architectures::moshi::prepare_moshi_realtime_with_support(
        preparation.clone(),
        &eredu_runtime::NormalizedLoadRequest::default(),
        &eredu_backend_mlx::native::MlxRealtimeSupport::new(false),
    )
    .unwrap();
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        expected_rank,
    )
    .unwrap();
    let parallel = eredu_runtime::ParallelLoadRequest::new(
        topology,
        eredu_runtime::PipelineWireContract::new(
            eredu_core::communication::PipelineActivationDtype::Float32,
        ),
        1,
        4096,
        eredu_core::communication::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
    )
    .unwrap();
    let normalized = eredu_runtime::NormalizedLoadRequest::default()
        .with_parallel_execution(parallel)
        .unwrap();
    let parallel_options = eredu_core::ModelLoadOptions::new(normalized).with_placement(device);
    let request = parallel_options.request();
    let native_rank =
        Some(eredu_backend_mlx::backend::MlxRankContext::new(2, expected_rank, device).unwrap());
    let parallel_selected = eredu_architectures::moshi::prepare_moshi_realtime_with_support(
        preparation,
        request,
        &eredu_backend_mlx::native::MlxRealtimeSupport::new(true),
    )
    .unwrap();

    // Exact replicated and TP selection both finish before the worker realizes
    // a native group, device, stream, checkpoint payload, module, or state.
    let group = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!((group.rank(), group.size()), (expected_rank, 2));
    let stream = Stream::new_with_device(&device.device().unwrap());
    let weights_stream = Stream::new_with_device(&device.device().unwrap());

    let replicated_backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
    let mut replicated =
        eredu_runtime::bind_prepared_realtime(&replicated_backend, replicated_selected, None)
            .unwrap();
    assert_eq!(
        replicated
            .execution_config()
            .effective_model_type()
            .as_str(),
        expected_profile
    );
    let config = replicated.execution_config().clone();
    let expected = run_forced_and_greedy_sequence(&replicated_backend, &mut replicated, None);
    drop(replicated);

    let backend =
        MlxRealtimeExecutionContext::new(&stream, &weights_stream).with_native_world(&group);
    let mut parallel =
        eredu_runtime::bind_prepared_realtime(&backend, parallel_selected, native_rank).unwrap();
    assert_eq!(
        parallel.execution_config().effective_model_type().as_str(),
        expected_profile
    );
    assert_eq!(parallel.execution_config(), &config);
    let consensus = MlxRealtimeConsensusTransport::new(&group, &stream);
    if std::env::var_os(MODEL_WORKER_DISAGREE).is_some() {
        assert_schedule_disagreement_submits_nothing(
            &backend,
            &mut parallel,
            &consensus,
            expected_rank,
        );
        return;
    }
    let actual = run_forced_and_greedy_sequence(&backend, &mut parallel, Some(&consensus));
    assert_eq!(actual, expected, "rank {expected_rank} TP output drifted");

    // Vocab-parallel projections already gather canonical logits internally.
    // Gather the committed frame transcript once more so both ranks prove
    // they observed the same canonical text/audio outputs as replication.
    let local = Array::from_slice(&actual, &[i32::try_from(actual.len()).unwrap()]);
    let gathered = distributed::all_gather(&local, &group, &stream).unwrap();
    let gathered = gathered.evaluated().unwrap();
    let gathered = gathered.as_slice::<i32>();
    assert_eq!(gathered.len(), expected.len() * 2);
    assert_eq!(&gathered[..expected.len()], expected.as_slice());
    assert_eq!(&gathered[expected.len()..], expected.as_slice());
}

fn run_model_parity_fixture(fixture_variable: &str, profile: &str) {
    let fixture = std::env::var_os(fixture_variable).unwrap_or_else(|| {
        panic!(
            "{profile} Ring parity requires {fixture_variable} to point at a complete fixture when this ignored test is explicitly enabled"
        )
    });
    assert!(
        Path::new(&fixture).exists(),
        "{fixture_variable} does not exist: {}",
        Path::new(&fixture).display()
    );
    run_model_parity_path(Path::new(&fixture), profile, false);
}

fn run_model_parity_path(fixture: &Path, profile: &str, disagree: bool) {
    super::workers::run(
        "realtime::moshi_ring_model_parity_worker",
        Duration::from_secs(15 * 60),
        |command, rank| {
            command
                .env(MODEL_WORKER_RANK, rank.to_string())
                .env(MODEL_WORKER_FIXTURE, fixture)
                .env(MODEL_WORKER_PROFILE, profile);
            if disagree {
                command.env(MODEL_WORKER_DISAGREE, "1");
            } else {
                command.env_remove(MODEL_WORKER_DISAGREE);
            }
        },
    );
}

/// Executes the selected synthetic model through replicated and pure-TP paths.
#[test]
#[ignore = "spawns two local Ring ranks and executes a synthetic selected model"]
fn moshi_ring_tp2_synthetic_model_parity() {
    let fixture = tempfile::tempdir().unwrap();
    fixtures::write_tiny_native_artifact(fixture.path());
    run_model_parity_path(fixture.path(), "moshi", false);
}

/// Proves a real two-rank Ring schedule disagreement reaches consensus before
/// either selected MLX executor is invoked or any frame is published.
#[test]
#[ignore = "spawns two local Ring ranks and executes a synthetic selected model"]
fn moshi_ring_tp2_schedule_disagreement_publishes_nothing() {
    let fixture = tempfile::tempdir().unwrap();
    fixtures::write_tiny_native_artifact(fixture.path());
    run_model_parity_path(fixture.path(), "moshi", true);
}

/// Optional production-model gate. Set `EREDU_MOSHI_NATIVE_FIXTURE`
/// to a released native Moshi model directory and select this ignored test
/// explicitly.
#[test]
#[ignore = "requires EREDU_MOSHI_NATIVE_FIXTURE and two local Ring ranks"]
fn moshi_ring_tp2_native_model_parity() {
    run_model_parity_fixture(NATIVE_FIXTURE, "moshi");
}

/// Optional production-model gate. Set
/// `EREDU_MOSHI_PERSONAPLEX_FIXTURE` to a released PersonaPlex model
/// directory and select this ignored test explicitly.
#[test]
#[ignore = "requires EREDU_MOSHI_PERSONAPLEX_FIXTURE and two local Ring ranks"]
fn moshi_ring_tp2_personaplex_model_parity() {
    run_model_parity_fixture(PERSONAPLEX_FIXTURE, "personaplex");
}
