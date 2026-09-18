//! Sequential native capture/copy phases keep unrelated test locals off the
//! default test-thread stack. The same session, snapshots and budgets cross
//! each boundary by value; all public operations and assertions remain here.
use super::*;

type Backend = eredu_backend_mlx::backend::MlxBackend<'static>;
type Session<'a> = PreparedChatSession<'a, Backend>;
type Snapshot = PreparedChatSnapshot<Backend>;
type Output = eredu_core::GenerationOutput<(), eredu_core::GenerationTokenIds>;

pub(super) struct Proof {
    pub(super) events: Vec<SemanticEvent>,
    pub(super) expected_ids: Vec<u32>,
    pub(super) expected_finish: eredu_core::FinishReason,
    pub(super) budget: SnapshotBudget,
    pub(super) spent: u64,
    pub(super) terminal_budget: SnapshotBudget,
    pub(super) terminal_spent: u64,
}

struct Checkpoint {
    saved: Snapshot,
    prefix: Vec<SemanticEvent>,
    budget: SnapshotBudget,
    sampling: eredu_core::SamplingStateFacts,
}

pub(super) fn check(
    model: &mut LoadedModel<Backend>,
    chat: &eredu::runtime::chat::PreparedChat,
    settings: &PreparedChatGenerationSettings,
    script: &[u32],
    count: u64,
    cancellation: &GenerationCancellationToken,
) -> Proof {
    let discovery = model.capture_discovery().unwrap_or_else(fail);
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "logits".into(),
        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    plan.limits.per_step = CaptureUsage {
        captures: 1,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    plan.limits.cumulative = CaptureUsage {
        captures: 3,
        retained_bytes: 128 << 20,
        host_bytes: 128 << 20,
        encoded_bytes: 128 << 20,
    };
    plan.limits.on_limit = CaptureLimitPolicy::Skip;
    let capture = SharedCapturePlan::new(
        plan.admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: count,
                max_predictions: 8,
            },
        )
        .unwrap(),
    );
    let mut request = PreparedChatRequest::new(chat, settings.clone());
    request.options = Some(TextPreparationOptions {
        capture: Some(capture),
        interventions: None,
    });
    let mut original_frames = Vec::new();
    let capture_deliveries = std::cell::Cell::new(0usize);
    let mut observer = |_: Option<u32>, frame: Option<SharedCapturedStep>, _: f64| {
        capture_deliveries.set(capture_deliveries.get() + 1);
        original_frames.push(frame.expect("admitted capture delivery"));
    };
    let mut events = Vec::new();
    let session = model
        .start_prepared_chat(request, cancellation)
        .unwrap_or_else(fail)
        .unwrap()
        .with_capture_observer(&mut observer);
    let (session, checkpoint) = checkpoint(session, script, cancellation, &mut events);
    let Checkpoint { saved, prefix, budget, sampling: saved_sampling } = checkpoint;
    let session = serial_branch(session, &saved, saved_sampling, script, cancellation);
    let (original, terminal_budget, terminal_spent) = terminal(
        session, cancellation, &mut events, &capture_deliveries, saved_sampling,
    );
    drop(observer);
    assert!(original.token_ids.starts_with(&script[..5]));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, SemanticEvent::ToolCallEnd { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
            .count(),
        1
    );
    assert_eq!(
        original_frames.last().unwrap().cumulative_usage().captures,
        3
    );
    resumes(model, &saved, &original, &prefix, &events, script, saved_sampling, cancellation);
    let spent = budget.usage().cumulative_copy_bytes;
    assert!(spent > 0);
    let expected_ids = original.token_ids.to_vec();
    let expected_finish = original.finish_reason;
    drop((saved, original, original_frames, prefix));
    Proof { events, expected_ids, expected_finish, budget, spent, terminal_budget, terminal_spent }
}

fn checkpoint<'a>(
    mut session: Session<'a>,
    script: &[u32],
    cancellation: &GenerationCancellationToken,
    events: &mut Vec<SemanticEvent>,
) -> (Session<'a>, Checkpoint) {
    for _ in 0..3 {
        session = session
            .advance(cancellation, &mut |event| events.push(event))
            .unwrap_or_else(fail);
    }
    assert_eq!(session.token_ids(), &script[..3]);
    let original_sampling = session.sampling_state().unwrap_or_else(fail);
    assert_eq!(original_sampling.temperature, 0.0);
    let invalid = session.override_sampling(eredu::api::SamplingOverride {
        temperature: Some(f32::NAN), reseed: None,
    }).err().expect("invalid sampler change must refuse");
    assert!(invalid.sampling_rejection().is_some());
    assert_eq!(session.sampling_state().unwrap_or_else(fail), original_sampling);
    assert_eq!(session.token_ids(), &script[..3]);
    drop(invalid);
    let saved_sampling = session.override_sampling(eredu::api::SamplingOverride {
        temperature: Some(0.7), reseed: Some(73),
    }).unwrap_or_else(fail);
    assert_eq!(saved_sampling.temperature, 0.7);
    assert!(saved_sampling.has_rng);
    assert_eq!(session.token_ids(), &script[..3]);
    session.force_next_token(script[3]).unwrap_or_else(fail);
    assert_eq!(session.pending_forced_token(), Some(script[3]));
    let prefix = events.clone();
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: CAPACITY,
        cumulative_copy_bytes: CAPACITY * 8,
    });
    let saved = session
        .snapshot(&budget, CAPACITY, WorkspaceCopyLimits::new(CAPACITY))
        .unwrap_or_else(fail);
    (session, Checkpoint { saved, prefix, budget, sampling: saved_sampling })
}

fn serial_branch<'a>(
    mut session: Session<'a>,
    saved: &Snapshot,
    saved_sampling: eredu_core::SamplingStateFacts,
    script: &[u32],
    cancellation: &GenerationCancellationToken,
) -> Session<'a> {
    let mut serial = session.fork_snapshot(saved, eredu::api::PreparedChatResumeSettings::default(),
        CAPACITY, cancellation).unwrap_or_else(fail).unwrap();
    assert_eq!(session.token_ids(), &script[..3]);
    assert_eq!(session.pending_forced_token(), Some(script[3]));
    session.exchange(&mut serial).unwrap_or_else(fail);
    assert_eq!(session.sampling_state().unwrap_or_else(fail), saved_sampling);
    let mut serial_events = Vec::new();
    session = session.advance(cancellation, &mut |event| serial_events.push(event)).unwrap_or_else(fail);
    assert_eq!(session.token_ids(), &script[..4]);
    session.exchange(&mut serial).unwrap_or_else(fail);
    assert_eq!(serial.token_ids(), &script[..4]);
    assert_eq!(session.token_ids(), &script[..3]);
    assert_eq!(session.pending_forced_token(), Some(script[3]));
    assert_eq!(session.sampling_state().unwrap_or_else(fail), saved_sampling);
    drop((serial, serial_events));
    assert!(session.restore_snapshot(saved, eredu::api::PreparedChatResumeSettings::default(),
        CAPACITY, cancellation).unwrap_or_else(fail));
    assert_eq!(session.token_ids(), &script[..3]);
    assert_eq!(session.pending_forced_token(), Some(script[3]));
    session
}

fn terminal(
    mut session: Session<'_>,
    cancellation: &GenerationCancellationToken,
    events: &mut Vec<SemanticEvent>,
    capture_deliveries: &std::cell::Cell<usize>,
    saved_sampling: eredu_core::SamplingStateFacts,
) -> (Output, SnapshotBudget, u64) {
    while session.finish_reason().is_none() {
        session = session.advance(cancellation, &mut |event| events.push(event)).unwrap_or_else(fail);
    }
    let terminal_reason = session.finish_reason();
    let terminal_status = session.status();
    let terminal_prediction = session.next_prediction();
    let terminal_ids = session.token_ids().to_vec();
    let delivered = capture_deliveries.get();
    let terminal_budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 3, max_branches: 1, retained_bytes: CAPACITY, cumulative_copy_bytes: CAPACITY * 8,
    });
    let terminal = session.snapshot(&terminal_budget, CAPACITY, WorkspaceCopyLimits::new(CAPACITY))
        .unwrap_or_else(|error| panic!("terminal snapshot failed: {error:?}"));
    let terminal_facts = terminal.resume_source_facts().expect("actual saved native facts");
    assert_eq!(terminal_facts.inherited_capture_usage.captures, 3);
    assert_eq!(terminal_facts.sampling_before, saved_sampling);
    assert!(session.restore_snapshot(&terminal, PreparedChatResumeSettings::default(), CAPACITY, cancellation)
        .unwrap_or_else(|error| panic!("terminal restore failed: {error:?}")));
    assert_eq!(session.status(), terminal_status);
    assert_eq!(session.finish_reason(), terminal_reason);
    assert_eq!(session.token_ids(), terminal_ids);
    assert_eq!(session.next_prediction(), terminal_prediction);
    assert_eq!(session.sampling_state().unwrap_or_else(|error| panic!("restored terminal sampling failed: {error:?}")), saved_sampling);
    assert_eq!(session.pending_forced_token(), None);
    let restored_terminal = session.snapshot(&terminal_budget, CAPACITY, WorkspaceCopyLimits::new(CAPACITY))
        .unwrap_or_else(|error| panic!("restored terminal snapshot failed: {error:?}"));
    assert_eq!(restored_terminal.remaining_tokens(), Some(0));
    assert_eq!(restored_terminal.resume_source_facts(), Some(terminal_facts));
    let mut terminal_branch = session.fork_snapshot(&restored_terminal, PreparedChatResumeSettings::default(),
        CAPACITY, cancellation).unwrap_or_else(|error| panic!("terminal fork failed: {error:?}")).unwrap();
    assert_eq!(terminal_branch.status(), terminal_status);
    assert_eq!(terminal_branch.next_prediction(), terminal_prediction);
    session.exchange(&mut terminal_branch).unwrap_or_else(|error| panic!("terminal branch installation failed: {error:?}"));
    assert_eq!(session.status(), terminal_status);
    assert_eq!(session.sampling_state().unwrap_or_else(|error| panic!("installed terminal sampling failed: {error:?}")), saved_sampling);
    let forked_terminal = session.snapshot(&terminal_budget, CAPACITY, WorkspaceCopyLimits::new(CAPACITY))
        .unwrap_or_else(|error| panic!("installed terminal snapshot failed: {error:?}"));
    assert_eq!(forked_terminal.remaining_tokens(), Some(0));
    assert_eq!(forked_terminal.resume_source_facts(), Some(terminal_facts));
    session = session.advance(cancellation, &mut |_| panic!("terminal branch emitted another event"))
        .unwrap_or_else(|error| panic!("terminal advance failed: {error:?}"));
    assert_eq!(session.token_ids(), terminal_ids);
    assert_eq!(session.next_prediction(), terminal_prediction);
    assert_eq!(capture_deliveries.get(), delivered, "terminal restore/fork issued another capture");
    session.exchange(&mut terminal_branch).unwrap_or_else(|error| panic!("terminal branch return failed: {error:?}"));
    assert_eq!(session.token_ids(), terminal_ids);
    let terminal_spent = terminal_budget.usage().cumulative_copy_bytes;
    assert!(terminal_spent > 0);
    drop((terminal_branch, forked_terminal, restored_terminal, terminal));
    let original = session.run(cancellation, &mut |_| panic!("terminal run emitted another event"))
        .unwrap_or_else(|error| panic!("terminal run failed: {error:?}"));
    (original, terminal_budget, terminal_spent)
}

fn resumes(
    model: &mut LoadedModel<Backend>,
    saved: &Snapshot,
    original: &Output,
    prefix: &[SemanticEvent],
    events: &[SemanticEvent],
    script: &[u32],
    saved_sampling: eredu_core::SamplingStateFacts,
    cancellation: &GenerationCancellationToken,
) {
    for branch in [false, true] {
        let mut resumed_events = prefix.to_vec();
        let mut frames = Vec::new();
        let mut observer = |_: Option<u32>, frame: Option<SharedCapturedStep>, _: f64| {
            frames.push(frame.expect("saved capture delivery"));
        };
        let mut session = if branch {
            model.fork_prepared_chat(saved, eredu::api::PreparedChatResumeSettings::default(), CAPACITY, cancellation)
        } else {
            model.restore_prepared_chat(saved, eredu::api::PreparedChatResumeSettings::default(), CAPACITY, cancellation)
        }
        .unwrap_or_else(fail)
        .unwrap()
        .with_capture_observer(&mut observer);
        assert_eq!(session.pending_forced_token(), Some(script[3]));
        assert_eq!(session.sampling_state().unwrap_or_else(fail), saved_sampling);
        let output = session
            .run(cancellation, &mut |event| resumed_events.push(event))
            .unwrap_or_else(fail);
        drop(observer);
        assert_eq!(output.token_ids.as_ref(), original.token_ids.as_ref());
        assert_eq!(&resumed_events, events);
        assert!(!frames.is_empty());
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(frame.prediction_index(), 3 + index as u64);
            assert_eq!(frame.cumulative_usage().captures, 3);
        }
        assert_eq!(saved.token_ids(), &script[..3]);
        drop((output, frames, resumed_events));
    }
}
