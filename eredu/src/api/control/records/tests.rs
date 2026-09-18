use super::*;
use eredu_core::{HostMetadataAccount, HostMetadataFundingError};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

#[derive(Debug, Default)]
struct State {
    calls: AtomicUsize,
    refuse: AtomicUsize,
    retired: AtomicBool,
    requests: Mutex<Vec<usize>>,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let call = self.0.calls.fetch_add(1, Ordering::SeqCst) + 1;
        self.0.requests.lock().unwrap().push(bytes);
        if call == self.0.refuse.load(Ordering::SeqCst) {
            Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            })
        } else {
            Ok(())
        }
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
fn payer() -> (HostMetadataFunding, Arc<State>) {
    let state = Arc::new(State::default());
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    state.calls.store(0, Ordering::SeqCst);
    state.requests.lock().unwrap().clear();
    (funding, state)
}
fn context() -> RecordContext {
    RecordContext {
        run_id: "run".into(),
        artifact_identity: Some("artifact".into()),
        parameter_overlay_id: Some("overlay".into()),
        session_id: "session".into(),
        capture_plan_id: Some("capture".into()),
        intervention_plan_id: Some("intervention".into()),
    }
}
fn progress() -> ControlEvent {
    ControlEvent::Existing(ObservedGenerationEvent::Lifecycle {
        status: GenerationStatus::Prepared,
        next_prediction: 0,
    })
}
fn snapshot_info() -> SnapshotInfo {
    SnapshotInfo {
        snapshot_id: "snapshot".into(),
        session_id: "session".into(),
        artifact_identity: Some("artifact".into()),
        capture_plan_id: Some("capture".into()),
        intervention_plan_id: Some("intervention".into()),
        tokenizer_identity: [3; 32],
        configuration_identity: [4; 32],
        pending_forced_token: Some(42),
        status: GenerationStatus::Prepared,
        output: GenerationOutputCheckpointData {
            run_id: "run".into(),
            epoch: 2,
            next_sequence: 7,
            next_prediction: 5,
        },
        retained_bytes: 51,
    }
}

#[test]
fn token_attribution_admits_every_original_destination_and_retains_failed_prefixes() {
    let ids = [1, 42, 65536];
    let (funding, state) = payer();
    let prompt = PromptRecord::from_tokens(&ids, &funding).unwrap();
    let count = state.calls.load(Ordering::SeqCst);
    assert!(count >= 8);
    let data = prompt.prepared().attribution();
    assert_eq!(data.complete_token_ids(), Some(ids.as_slice()));
    assert_eq!(
        data.semantic_content_identity,
        eredu_core::cache::prompt_cache_token_fingerprint(&ids)
    );
    assert_eq!(
        data.semantic_content_identity,
        "7444557477eb196f41f11e0b6808c9832aed5153123f2887b6ea442b7eae398d"
    );
    let mut short = String::with_capacity(63);
    assert!(!eredu_core::cache::prompt_cache_token_fingerprint_into(
        &ids, &mut short
    ));
    assert!(short.is_empty());
    assert_eq!(prompt.range(0).unwrap(), [0, 3]);
    assert_eq!(prompt.range(1).unwrap(), [3, 4]);
    let peer = prompt.clone();
    drop((prompt, funding));
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(peer);
    assert!(state.retired.load(Ordering::SeqCst));
    for cut in 1..=count {
        let (funding, state) = payer();
        state.refuse.store(cut, Ordering::SeqCst);
        let error = match PromptRecord::from_tokens(&ids, &funding) {
            Err(error) => error,
            Ok(_) => panic!("accepted refused attribution producer {cut}"),
        };
        assert!(matches!(
            error.funding_failure(),
            Some(HostMetadataFundingError::Capacity { available: 0, .. })
        ));
        assert_eq!(state.calls.load(Ordering::SeqCst), cut);
        drop(funding);
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(error);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}

#[test]
fn record_refuses_every_copy_and_shell_before_publication_and_aliases_keep_both_accounts() {
    let (prompt_funding, prompt_state) = payer();
    let prompt = PromptRecord::from_tokens(&[2, 3], &prompt_funding).unwrap();
    let c = context();
    let (funding, state) = payer();
    let record = ControlledGenerationRecord::record(
        &c,
        &prompt,
        progress(),
        5,
        2,
        GenerationTiming::default(),
        &funding,
    )
    .unwrap();
    let count = state.calls.load(Ordering::SeqCst);
    assert_eq!(count, 8); // named controls, six copied fields, and the actual shared shell
    assert_eq!(record.run_id, c.run_id);
    assert_ne!(record.run_id.as_ptr(), c.run_id.as_ptr());
    assert_eq!(record.sequence, 5);
    assert_eq!(record.epoch, 2);
    for cut in 1..=count {
        let (funding, state) = payer();
        state.refuse.store(cut, Ordering::SeqCst);
        let error = ControlledGenerationRecord::record(
            &c,
            &prompt,
            progress(),
            5,
            2,
            GenerationTiming::default(),
            &funding,
        )
        .unwrap_err();
        assert!(error.funding_failure().is_some());
        assert_eq!(state.calls.load(Ordering::SeqCst), cut);
        drop(funding);
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(error);
        assert!(state.retired.load(Ordering::SeqCst));
    }
    let peer = record.clone();
    drop((record, prompt, prompt_funding, funding));
    assert!(!state.retired.load(Ordering::SeqCst));
    assert!(!prompt_state.retired.load(Ordering::SeqCst));
    std::thread::spawn(move || drop(peer)).join().unwrap();
    assert!(state.retired.load(Ordering::SeqCst));
    assert!(prompt_state.retired.load(Ordering::SeqCst));
}

#[test]
fn owned_snapshot_fields_move_into_record_without_recopying_or_detaching_prompt() {
    let (funding, state) = payer();
    let prompt = PromptRecord::from_tokens(&[4, 5], &funding).unwrap();
    let info = snapshot_info();
    let snapshot_pointer = info.snapshot_id.as_ptr();
    let output_pointer = info.output.run_id.as_ptr();
    state.calls.store(0, Ordering::SeqCst);
    let record = ControlledGenerationRecord::record(
        &context(),
        &prompt,
        ControlEvent::SnapshotCreated { metadata: info },
        7,
        2,
        GenerationTiming::default(),
        &funding,
    )
    .unwrap();
    assert_eq!(state.calls.load(Ordering::SeqCst), 8);
    let ControlledGenerationEvent::SnapshotCreated { metadata } = &record.event else {
        panic!()
    };
    assert_eq!(metadata.snapshot_id.as_ptr(), snapshot_pointer);
    assert_eq!(metadata.output.run_id.as_ptr(), output_pointer);
    assert_eq!(metadata.pending_forced_token, Some(42));
    assert_eq!(metadata.prompt_attribution, *prompt.prepared());
    let wire = serde_json::to_string(&record).unwrap();
    let diagnostic = ControlledWireRecord::from_json(&wire).unwrap();
    assert_eq!(diagnostic.sequence, record.sequence);
    assert_eq!(diagnostic.instrumentation, record.instrumentation);
    drop((prompt, funding));
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(record);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn escaped_output_checkpoint_keeps_original_payer_and_refuses_each_destination() {
    let (funding, state) = payer();
    let output = GenerationOutputCheckpoint::prepare("run", 2, 7, 5, &funding).unwrap();
    let calls = state.calls.load(Ordering::SeqCst);
    assert!(calls >= 3);
    assert_eq!(output.run_id, "run");
    assert_eq!(
        (output.epoch, output.next_sequence, output.next_prediction),
        (2, 7, 5)
    );
    let wire = serde_json::to_string(&output).unwrap();
    let diagnostic: GenerationOutputCheckpointData = serde_json::from_str(&wire).unwrap();
    assert_eq!(diagnostic, *output);
    drop(funding);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(output);
    assert!(state.retired.load(Ordering::SeqCst));
    for cut in 1..=calls {
        let (funding, state) = payer();
        state.refuse.store(cut, Ordering::SeqCst);
        let error = GenerationOutputCheckpoint::prepare("run", 2, 7, 5, &funding).unwrap_err();
        assert!(error.funding_failure().is_some());
        assert_eq!(state.calls.load(Ordering::SeqCst), cut);
        drop(funding);
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(error);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}
