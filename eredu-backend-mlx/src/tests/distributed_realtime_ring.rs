#![cfg(unix)]

use std::{
    cell::Cell,
    net::TcpListener,
    path::Path,
    process::{Child, Command, Output, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::{
    backend::DeviceAssignment,
    native::{
        MlxRealtimeCompletion, MlxRealtimeConsensusTransport, MlxRealtimeExecutionContext,
        MlxRealtimeHostObserver, RandomState,
    },
    MlxLoadRequest,
};
use eredu_architectures::moshi::{MoshiCollectiveCount, MoshiConfig};
use eredu_core::scheduler::{RequestId, SchedulerLimits};
use eredu_core::{RealtimeInputFrame, RealtimeSampling};
use eredu_runtime::{
    GenerationSampler, RealtimeGenerationState, RealtimePayloadState, RealtimeSessionScheduler,
};
use safemlx::{
    distributed::{self, Backend},
    Array, Device, DeviceType, Stream,
};

const WORKER_RANK: &str = "EREDU_MOSHI_RING_WORKER";
const MODEL_WORKER_RANK: &str = "EREDU_MOSHI_RING_MODEL_WORKER";
const MODEL_WORKER_FIXTURE: &str = "EREDU_MOSHI_RING_MODEL_FIXTURE";
const MODEL_WORKER_PROFILE: &str = "EREDU_MOSHI_RING_MODEL_PROFILE";
const MODEL_WORKER_DISAGREE: &str = "EREDU_MOSHI_RING_MODEL_DISAGREE";
const NATIVE_FIXTURE: &str = "EREDU_MOSHI_NATIVE_FIXTURE";
const PERSONAPLEX_FIXTURE: &str = "EREDU_MOSHI_PERSONAPLEX_FIXTURE";

type PreparedRingRealtime = eredu_architectures::moshi::MoshiRealtimeExecution<
    crate::composition::moshi::MlxRealtimeExecution,
>;

fn balanced_widths(total: usize) -> [usize; 2] {
    [total.div_ceil(2), total / 2]
}

fn verify_canonical_vocabulary(
    total: usize,
    rank: usize,
    group: &crate::backend::runtime::distributed::Group,
    stream: &Stream,
) {
    let widths = balanced_widths(total);
    let start = widths[..rank].iter().sum::<usize>();
    let end = start + widths[rank];
    let local = (start..end)
        .map(|token| i32::try_from(token).expect("test vocabulary fits i32"))
        .collect::<Vec<_>>();
    let local = Array::from_slice(&local, &[i32::try_from(local.len()).unwrap()]);
    let gathered =
        crate::backend::distributed::all_gather_uneven_axis(&local, 0, &widths, group, stream)
            .unwrap();
    let gathered = gathered.evaluated().unwrap();
    let expected = (0..total)
        .map(|token| i32::try_from(token).expect("test vocabulary fits i32"))
        .collect::<Vec<_>>();
    assert_eq!(gathered.as_slice::<i32>(), expected);
}

#[test]
fn moshi_ring_collective_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let group = crate::backend::runtime::distributed::Group::uncontracted(
        &distributed::init(true, Backend::Ring).unwrap(),
    );
    assert_eq!(group.size(), 2);
    assert_eq!(group.rank(), expected_rank);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let config = MoshiConfig::native_v0_1().unwrap();
    let temporal_layers = usize::try_from(config.temporal().num_hidden_layers()).unwrap();
    let depth_layers = usize::try_from(config.depth_template().num_hidden_layers()).unwrap();
    let depth_slices = config.frame_schedule().depth_audio_codebooks();
    let embedding_tables = config
        .frame_schedule()
        .total_audio_codebooks()
        .checked_add(1)
        .unwrap();

    // Canonical traversal order: summed embedding tables, two summed row
    // projections per temporal block, then input plus two row projections per
    // executed depth block. A rank-dependent marker detects order drift.
    let phase_counts = [
        embedding_tables,
        temporal_layers * 2,
        depth_slices * (1 + depth_layers * 2),
    ];
    let oracle = eredu_architectures::moshi::collective_count(&config, depth_slices).unwrap();
    let expected_all_sum = phase_counts.iter().sum::<usize>();
    assert_eq!(expected_all_sum, oracle.all_sum);
    let mut observed_all_sum = 0usize;
    for phase_count in phase_counts {
        for _ in 0..phase_count {
            let marker = i32::try_from(observed_all_sum + 1).unwrap();
            let local = Array::from_int(marker * 10 + expected_rank as i32);
            let reduced =
                crate::backend::runtime::distributed::all_sum(&local, &group, &stream).unwrap();
            let reduced = reduced.evaluated().unwrap();
            assert_eq!(reduced.item::<i32>(), marker * 20 + 1);
            observed_all_sum += 1;
        }
    }
    assert_eq!(observed_all_sum, expected_all_sum);

    // Text is gathered first, followed by one audio vocabulary per executed
    // depth slice. Uneven gathering must restore global row order on both
    // ranks, not rank-major padded order.
    let text_vocabulary = usize::try_from(config.text_vocabulary_size()).unwrap();
    let audio_vocabulary = usize::try_from(config.audio_vocabulary_size()).unwrap();
    let mut observed_all_gather = 0usize;
    verify_canonical_vocabulary(text_vocabulary, expected_rank, &group, &stream);
    observed_all_gather += 1;
    for _ in 0..depth_slices {
        verify_canonical_vocabulary(audio_vocabulary, expected_rank, &group, &stream);
        observed_all_gather += 1;
    }
    assert_eq!(observed_all_gather, oracle.all_gather);
}

fn push_tokens(serialized: &mut Vec<i32>, values: &[i32]) {
    serialized.push(i32::try_from(values.len()).unwrap());
    serialized.extend_from_slice(values);
}

type NativeScheduler = RealtimeSessionScheduler<
    eredu_runtime::RealtimePayloadState<
        crate::backend::runtime::cache::state::MlxKeyValueState,
        crate::MlxTensor,
    >,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
    eredu_runtime::PrepublicationRealtimeFrame<
        crate::MlxTensor,
        MlxRealtimeCompletion,
        MlxRealtimeHostObserver,
    >,
>;

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
    let samplers =
        eredu_architectures::moshi::realtime_generation_samplers(&schedule, sampling).unwrap();
    let model_state = RealtimePayloadState::fresh(
        backend.new_realtime_model_state(model).unwrap(),
        schedule.clone(),
    );
    let random = backend.realize_random_state(None).unwrap();
    scheduler
        .register(
            request,
            RealtimeGenerationState::new(model_state, schedule.clone(), sampling, samplers, random)
                .unwrap(),
        )
        .unwrap();

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
                    |_, frame, branch| backend.submit_realtime_frame(model, frame, branch),
                ),
                None => scheduler.run_local_turn(Instant::now(), |_, frame, branch| {
                    backend.submit_realtime_frame(model, frame, branch)
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

fn sequence_collective_oracle(config: &MoshiConfig) -> usize {
    let temporal_layers = usize::try_from(config.temporal().num_hidden_layers()).unwrap();
    let depth_layers = usize::try_from(config.depth_template().num_hidden_layers()).unwrap();
    let embedding_tables = config.frame_schedule().total_audio_codebooks() + 1;
    let generated = config.frame_schedule().generated_audio_codebooks();
    // Absolute-delay schedules initialize without model work. Feedback-aligned
    // schedules execute the first unforced frame greedily. The remaining frames
    // are fully forced, forced except for the first generated codebook, and greedy.
    let first = match config.frame_schedule().frame_convention() {
        eredu_core::realtime::RealtimeFrameConvention::FeedbackAlignedHistory => Some(generated),
        eredu_core::realtime::RealtimeFrameConvention::AbsoluteDelayedSlots => None,
        convention => panic!("unsupported synthetic Ring frame convention {convention:?}"),
    };
    first
        .into_iter()
        .chain([0, 1, generated])
        .map(|executed_depth_slices| {
            let actual =
                eredu_architectures::moshi::collective_count(config, executed_depth_slices)
                    .unwrap();
            assert_eq!(
                actual,
                MoshiCollectiveCount {
                    all_sum: embedding_tables
                        + 2 * temporal_layers
                        + executed_depth_slices * (1 + 2 * depth_layers),
                    all_gather: 1 + executed_depth_slices,
                }
            );
            actual.all_sum + actual.all_gather
        })
        .sum()
}

fn assert_schedule_disagreement_submits_nothing(
    backend: &MlxRealtimeExecutionContext,
    model: &mut PreparedRingRealtime,
    consensus: &MlxRealtimeConsensusTransport,
    rank: usize,
) {
    let schedule = model.execution_config().frame_schedule().clone();
    let sampling = RealtimeSampling::greedy();
    let samplers =
        eredu_architectures::moshi::realtime_generation_samplers(&schedule, sampling).unwrap();
    let model_state = RealtimePayloadState::fresh(
        backend.new_realtime_model_state(model).unwrap(),
        schedule.clone(),
    );
    let mut scheduler = NativeScheduler::new(
        eredu_runtime::RealtimeModelSessionIdentity::from_selected(model.selected()),
        SchedulerLimits::new(1, 1).unwrap(),
    )
    .unwrap();
    let request = RequestId::new(99);
    scheduler
        .register(
            request,
            RealtimeGenerationState::new(model_state, schedule.clone(), sampling, samplers, None)
                .unwrap(),
        )
        .unwrap();
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
            backend.submit_realtime_frame(model, frame, branch)
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
    let replicated_options = MlxLoadRequest::default();
    let replicated_selected = MlxRealtimeExecutionContext::select_realtime_execution(
        eredu_architectures::moshi::prepare_realtime_model(Path::new(&fixture)).unwrap(),
        &replicated_options,
        false,
    )
    .unwrap();
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        expected_rank,
    )
    .unwrap();
    let parallel_options = MlxLoadRequest::with_parallel(
        topology,
        device,
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        4096,
        MlxLoadRequest::test_communication_completion_policy(),
    );
    let parallel_selected = MlxRealtimeExecutionContext::select_realtime_execution(
        eredu_architectures::moshi::prepare_realtime_model(Path::new(&fixture)).unwrap(),
        &parallel_options,
        true,
    )
    .unwrap();

    // Exact replicated and TP selection both finish before the worker realizes
    // a native group, device, stream, checkpoint payload, module, or state.
    let group = Arc::new(crate::backend::runtime::distributed::Group::uncontracted(
        &distributed::init(true, Backend::Ring).unwrap(),
    ));
    assert_eq!((group.rank(), group.size()), (expected_rank, 2));
    let stream = Stream::new_with_device(&device.device().unwrap());
    let weights_stream = Stream::new_with_device(&device.device().unwrap());

    let replicated_backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream);
    let mut replicated = replicated_backend
        .materialize_realtime_execution(replicated_selected, replicated_options)
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

    let backend = MlxRealtimeExecutionContext::new(&stream, &weights_stream)
        .with_tensor_parallel_group(Arc::clone(&group));
    let mut parallel = backend
        .materialize_realtime_execution(parallel_selected, parallel_options)
        .unwrap();
    assert_eq!(
        parallel.execution_config().effective_model_type().as_str(),
        expected_profile
    );
    assert_eq!(parallel.execution_config(), &config);
    let consensus = MlxRealtimeConsensusTransport::new(group.native_group(), &stream);
    if std::env::var_os(MODEL_WORKER_DISAGREE).is_some() {
        assert_schedule_disagreement_submits_nothing(
            &backend,
            &mut parallel,
            &consensus,
            expected_rank,
        );
        return;
    }
    crate::backend::runtime::distributed::reset_native_collective_submissions();
    let actual = run_forced_and_greedy_sequence(&backend, &mut parallel, Some(&consensus));
    let model_collectives =
        crate::backend::runtime::distributed::contracted_collective_submissions();
    assert_eq!(actual, expected, "rank {expected_rank} TP output drifted");
    assert_eq!(
        model_collectives,
        sequence_collective_oracle(&config),
        "rank {expected_rank} production TP collective count drifted"
    );

    // Vocab-parallel projections already gather canonical logits internally.
    // Gather the committed frame transcript once more so both ranks prove
    // they observed the same canonical text/audio outputs as replication.
    let local = Array::from_slice(&actual, &[i32::try_from(actual.len()).unwrap()]);
    let gathered =
        crate::backend::runtime::distributed::all_gather(&local, group.as_ref(), &stream).unwrap();
    let gathered = gathered.evaluated().unwrap();
    let gathered = gathered.as_slice::<i32>();
    assert_eq!(gathered.len(), expected.len() * 2);
    assert_eq!(&gathered[..expected.len()], expected.as_slice());
    assert_eq!(&gathered[expected.len()..], expected.as_slice());
}

struct ChildGuard(Vec<Child>);

impl ChildGuard {
    fn finish(mut self) -> Vec<Output> {
        self.0
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
        }
        for child in &mut self.0 {
            let _ = child.wait();
        }
    }
}

fn reserve_two_ports() -> (TcpListener, TcpListener, u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let second = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let first_port = first.local_addr().unwrap().port();
    let second_port = second.local_addr().unwrap().port();
    (first, second, first_port, second_port)
}

fn render_failure(rank: usize, output: &Output) -> String {
    format!(
        "realtime Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run with:
/// `cargo test -p eredu-backend-mlx tests::distributed_realtime_ring::moshi_ring_tp2_collective_order_and_vocab -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns two local Ring ranks and opens loopback sockets; run explicitly"]
fn moshi_ring_tp2_collective_order_and_vocab() {
    assert!(
        distributed::is_available(Backend::Ring),
        "the requested Ring verification requires the MLX Ring backend"
    );
    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    drop(first_socket);
    drop(second_socket);

    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard(Vec::with_capacity(2));
    for rank in 0..2 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "tests::distributed_realtime_ring::moshi_ring_collective_worker",
                    "--nocapture",
                ])
                .env(WORKER_RANK, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }

    let deadline = Instant::now() + Duration::from_secs(90);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }

    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-rank realtime verification failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
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
    assert!(
        distributed::is_available(Backend::Ring),
        "{profile} Ring parity requires the MLX Ring backend when this ignored test is explicitly enabled"
    );

    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    drop(first_socket);
    drop(second_socket);

    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard(Vec::with_capacity(2));
    for rank in 0..2 {
        let mut command = Command::new(&executable);
        command
            .args([
                "--exact",
                "tests::distributed_realtime_ring::moshi_ring_model_parity_worker",
                "--nocapture",
            ])
            .env(MODEL_WORKER_RANK, rank.to_string())
            .env(MODEL_WORKER_FIXTURE, fixture)
            .env(MODEL_WORKER_PROFILE, profile)
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if disagree {
            command.env(MODEL_WORKER_DISAGREE, "1");
        }
        children.0.push(command.spawn().unwrap());
    }

    let deadline = Instant::now() + Duration::from_secs(15 * 60);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }

    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "{profile} model Ring parity failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}

/// Executes the selected synthetic model through replicated and pure-TP paths.
#[test]
#[ignore = "spawns two local Ring ranks and executes a synthetic selected model"]
fn moshi_ring_tp2_synthetic_model_parity() {
    let fixture = tempfile::tempdir().unwrap();
    crate::composition::mlx::realtime::tests::write_tiny_native_artifact(fixture.path(), None);
    run_model_parity_path(fixture.path(), "moshi", false);
}

/// Proves a real two-rank Ring schedule disagreement reaches consensus before
/// either selected MLX executor is invoked or any frame is published.
#[test]
#[ignore = "spawns two local Ring ranks and executes a synthetic selected model"]
fn moshi_ring_tp2_schedule_disagreement_publishes_nothing() {
    let fixture = tempfile::tempdir().unwrap();
    crate::composition::mlx::realtime::tests::write_tiny_native_artifact(fixture.path(), None);
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
