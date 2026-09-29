//! Retained Flash-Next MTP through ordinary and controlled native speculation.
use super::*;

mod captures;

pub(super) fn is_fixture(checkpoint: &Path) -> bool {
    if !checkpoint.is_dir() {
        return false;
    }
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap()).unwrap();
    config
        .get("mtp_num_hidden_layers")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|layers| layers > 0)
}
fn run(axes: &'static str, residency: WorkerResidency, media: bool) {
    assert!(distributed::is_available(Backend::Ring));
    let directory = tempfile::tempdir().unwrap();
    let fixture = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixture.safetensors_path).unwrap();
    if media {
        eredu_evaluation::qwen4_exp::add_vision_weights(&fixture.safetensors_path, 2);
    }
    run_ring_pipeline_processes(
        residency,
        FixtureFamily::Qwen4Exp,
        WorkerMode::OpaqueSession,
        directory,
        fixture.safetensors_path,
        Some(axes),
    );
}
#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_prediction_tp_safetensors_resident() {
    run("tp", WorkerResidency::FullyResident, false);
}
#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_prediction_pp_safetensors_host() {
    run("pp", WorkerResidency::LayerwiseHost, false);
}
#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_prediction_tp_pp_safetensors_disk() {
    run("tp-pp", WorkerResidency::DenseDiskStream, false);
}

#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_prediction_media_tp_pp_safetensors_host() {
    run("tp-pp", WorkerResidency::LayerwiseHost, true);
}

fn load_policy(
    layers: eredu_runtime::LayerWeightResidency,
    media: bool,
) -> eredu_runtime::NormalizedLoadRequest {
    let request = if media {
        super::media::load_policy(layers, 64)
    } else {
        eredu_runtime::NormalizedLoadRequest::default()
            .with_bounded_execution(eredu_evaluation::qwen4_exp::bounded_policy())
            .with_weight_residency(WeightResidency::with_layers(layers))
    };
    request
        .with_drafting(eredu_runtime::DraftingLoadRequest::embedded(1).unwrap())
        .with_required_session_capabilities(eredu_core::SessionCapabilities::new(true, true, true))
}

fn prompt(media: bool) -> MlxModelInput {
    if media {
        super::media::prompt()
            .with_semantic_content_fingerprint("qwen4-native-mtp-media-fixture")
            .unwrap()
    } else {
        let tokens = Array::from_slice(&PROMPT, &[1, PROMPT.len() as i32]);
        let parts = [crate::backend::runtime::media::input::token_ids_part(&tokens).unwrap()];
        synthetic_prediction_input(&parts, &PROMPT)
    }
}

fn generation_config() -> SpeculativeConfig {
    SpeculativeConfig {
        max_tokens: 16,
        max_draft_tokens: 1,
        temperature: 0.,
        eos_token_ids: vec![],
    }
}

fn generate(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    media: bool,
) -> eredu_core::SpeculativeGenerationOutput {
    run_neutral_embedded_mtp(runtime, prompt(media), generation_config()).unwrap()
}

fn controlled(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    media: bool,
    expected: &eredu_core::SpeculativeGenerationOutput,
    baseline_captures: &[eredu_core::speculative::SpeculativeActivationCapture],
) {
    use eredu_core::{execution_control::SnapshotLimits, speculative::SpeculativeControlError};
    use eredu_runtime::speculative::{
        ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
    };
    let admitted = captures::admit(runtime);
    let mut failure = None;
    let (result, _) = execute_neutral_embedded_mtp_with(
        runtime,
        prompt(media),
        generation_config(),
        DriveControlledSpeculation::new(
            Default::default(),
            ControlledSpeculativeOptions {
                activations: Some(admitted),
                snapshots: Some(SnapshotLimits {
                    max_snapshots: 1,
                    max_branches: 1,
                    retained_bytes: 64 << 20,
                    cumulative_copy_bytes: 256 << 20,
                }),
                ..Default::default()
            },
            |session: &mut dyn ControlledSpeculativeSession| {
                let prefill = session.step()?.expect("prefill advancement");
                assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
                let prefix = session.token_ids().to_vec();
                assert_eq!(prefix, expected.token_ids()[..prefix.len()]);
                let saved = session.snapshot()?;
                let child = session.fork(&saved)?;
                let mut captured = prefill.activations;
                let capture_start = captured.len();
                let mut first_pass = Vec::new();
                let mut last_sequence = prefill.sequence;
                let mut checked_tentative = false;
                // Each replay advances the same production scheduler. Delivery
                // coordinates continue monotonically across restore and exchange.
                for pass in 0..3 {
                    while let Some(step) = session.step()? {
                        assert!(step.sequence > last_sequence);
                        last_sequence = step.sequence;
                        assert_eq!(step.run_id, session.run_id());
                        assert_eq!(step.epoch, session.epoch());
                        captured.extend(step.activations);
                        if step.drafted.is_some() && !session.can_snapshot() {
                            let before = session.snapshot_usage();
                            assert!(matches!(
                                session.snapshot(),
                                Err(SpeculativeControlError::NotQuiescent)
                            ));
                            assert_eq!(session.snapshot_usage(), before);
                            checked_tentative = true;
                        }
                    }
                    assert_eq!(session.token_ids(), expected.token_ids());
                    if pass == 0 {
                        captures::compare(&captured, baseline_captures);
                        first_pass = captured.split_off(capture_start);
                        captured.clear();
                    } else {
                        captures::exact_replay(&captured, &first_pass);
                        captured.clear();
                    }
                    let copied = session.snapshot_usage().cumulative_copy_bytes;
                    match pass {
                        0 => {
                            session.restore(&saved)?;
                            assert_eq!(session.epoch(), 1);
                            assert_eq!(session.run_id(), 0);
                            assert_eq!(session.token_ids(), prefix);
                        }
                        1 => {
                            session.exchange(&child)?;
                            assert_eq!(session.epoch(), 2);
                            assert_ne!(session.run_id(), 0);
                            assert_eq!(session.token_ids(), prefix);
                            assert!(matches!(
                                session.restore(&saved),
                                Err(SpeculativeControlError::IncompatibleSnapshot)
                            ));
                        }
                        _ => {
                            session.exchange(&child)?;
                            assert_eq!(session.epoch(), 3);
                            assert_eq!(session.run_id(), 0);
                            assert_eq!(session.token_ids(), expected.token_ids());
                            assert_eq!(
                                session.branch_info(&child)?.token_ids,
                                expected.token_ids()
                            );
                        }
                    }
                    assert!(session.snapshot_usage().cumulative_copy_bytes > copied);
                }
                assert!(
                    checked_tentative,
                    "drafts exercised noncanonical snapshot rejection"
                );
                let copied = session.snapshot_usage().cumulative_copy_bytes;
                session.release_branch(&child)?;
                session.release_snapshot(&saved)?;
                let usage = session.snapshot_usage();
                assert_eq!(usage.snapshots, 0);
                assert_eq!(usage.branches, 0);
                assert_eq!(usage.retained_bytes, 0);
                assert_eq!(usage.cumulative_copy_bytes, copied);
                Ok(())
            },
            &mut failure,
        ),
    );
    assert!(failure.is_none(), "controlled distributed MTP: {failure:?}");
    let actual = result.unwrap();
    assert_eq!(actual.token_ids(), expected.token_ids());
    assert_eq!(
        actual.stats().draft_tokens(),
        expected.stats().draft_tokens()
    );
    assert_eq!(
        actual.stats().accepted_tokens(),
        expected.stats().accepted_tokens()
    );
    assert_eq!(actual.stats().accept_lens(), expected.stats().accept_lens());
}

fn cancel_pending(runtime: &mut ModelRuntime<MlxBackend<'_>>, media: bool) {
    use eredu_core::generation::SpeculativeRequestStatus;
    use eredu_runtime::speculative::{
        ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
    };
    let mut failure = None;
    let mut committed = Vec::new();
    let (result, _) = execute_neutral_embedded_mtp_with(
        runtime,
        prompt(media),
        generation_config(),
        DriveControlledSpeculation::new(
            Default::default(),
            ControlledSpeculativeOptions::default(),
            |session: &mut dyn ControlledSpeculativeSession| {
                let mut pending = false;
                while let Some(step) = session.step()? {
                    if step.status == SpeculativeRequestStatus::TargetVerificationInFlight {
                        pending = true;
                        committed = session.token_ids().to_vec();
                        session.cancel()?;
                        assert_eq!(session.status(), SpeculativeRequestStatus::Cancelled);
                        assert_eq!(session.token_ids(), committed);
                        assert!(!session.can_snapshot());
                        assert!(session.step()?.is_none());
                        break;
                    }
                }
                assert!(
                    pending,
                    "cancellation exercised retained verification completion"
                );
                Ok(())
            },
            &mut failure,
        ),
    );
    assert!(
        failure.is_none(),
        "distributed MTP cancellation: {failure:?}"
    );
    let actual = result.unwrap();
    assert_eq!(actual.token_ids(), committed);
    assert_eq!(actual.finish_reason(), FinishReason::Cancelled);
}

pub(super) fn worker(
    checkpoint: &Path,
    topology: eredu_core::ParallelRankTopology,
    device: DeviceAssignment,
    group: &safemlx::distributed::Group,
    stream: &Stream,
    weights_stream: &Stream,
) {
    let media = super::media::is_fixture(checkpoint);
    let backend = crate::native::backend(stream, weights_stream);
    let prepared = eredu_core::prepare_inspected_model(
        &backend,
        component_fixture_inspection(checkpoint),
        MlxLoadRequest::from_normalized(load_policy(
            eredu_runtime::LayerWeightResidency::FullyResident,
            media,
        )),
    )
    .unwrap();
    let mut baseline = ModelRuntime::from_prepared(backend, prepared).unwrap();
    let expected = generate(&mut baseline, media);
    assert_eq!(expected.token_ids().len(), 16);
    assert!(expected.stats().draft_tokens() > 0);
    if topology.global_rank() == 0 {
        eprintln!(
            "Flash-Next MTP media={media}: {} proposed, {} accepted, {} rounds",
            expected.stats().draft_tokens(),
            expected.stats().accepted_tokens(),
            expected.stats().rounds()
        );
    }
    let baseline_captures = captures::baseline(&mut baseline, media, &expected);
    drop(baseline);
    let layers = if std::env::var_os(DENSE_STREAM).is_some() {
        eredu_runtime::LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 25, 1 << 26, 1, 1).unwrap(),
        )
    } else if std::env::var_os(LAYERWISE_HOST).is_some() {
        eredu_runtime::LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(1 << 25), Some(1 << 26), 1).unwrap(),
        ))
    } else {
        eredu_runtime::LayerWeightResidency::FullyResident
    };
    let load = load_policy(layers, media);
    let invocation = load.bounded_execution().unwrap().invocation();
    let request = MlxLoadRequest::from_normalized(load)
        .with_parallel_topology(
            topology,
            device,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            invocation.batch(),
            invocation.chunk_tokens(),
            ring_completion_policy(),
        )
        .unwrap();
    let backend = crate::native::distributed_backend(stream, weights_stream, group);
    let prepared = eredu_core::prepare_inspected_model(
        &backend,
        component_fixture_inspection(checkpoint),
        request,
    )
    .unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, prepared).unwrap();
    assert!(has_selected_embedded_prediction(runtime.session_mut()));
    for _ in 0..2 {
        let actual = generate(&mut runtime, media);
        assert_eq!(
            actual.token_ids(),
            expected.token_ids(),
            "distributed MTP committed tokens and fresh cache replay"
        );
        assert_eq!(actual.stats().emitted_tokens(), 16);
        assert_eq!(
            actual.stats().draft_tokens(),
            expected.stats().draft_tokens()
        );
        assert_eq!(
            actual.stats().accepted_tokens(),
            expected.stats().accepted_tokens()
        );
        assert_eq!(actual.stats().accept_lens(), expected.stats().accept_lens());
    }
    controlled(&mut runtime, media, &expected, &baseline_captures);
    cancel_pending(&mut runtime, media);
    let recovered = generate(&mut runtime, media);
    assert_eq!(
        recovered.token_ids(),
        expected.token_ids(),
        "fresh run after cancellation"
    );
}
