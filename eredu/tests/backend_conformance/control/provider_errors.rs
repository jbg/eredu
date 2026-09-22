use super::*;
use eredu::api::{ControlledGenerationError, GenerationBranchOptions, SamplingOverride};
use eredu_core::{BackendFailure, BackendFailureKind, SharedBackendFailure};
use eredu_runtime::execution_control::{SamplingOverrideError, TextSnapshotError};
use std::{
    cell::RefCell,
    error::Error as _,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

thread_local! {
    static FAILURE: RefCell<Option<(&'static str, SharedBackendFailure)>> = const { RefCell::new(None) };
}

#[derive(Debug, thiserror::Error)]
#[error("retained provider fixture cause")]
struct Cause(Arc<AtomicUsize>);
impl Drop for Cause {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

pub(crate) struct Armed {
    source: SharedBackendFailure,
    drops: Arc<AtomicUsize>,
}
impl Armed {
    pub(crate) fn new(at: &'static str) -> Self {
        let drops = Arc::new(AtomicUsize::new(0));
        let source = SharedBackendFailure::new(BackendFailureKind::Busy, Cause(drops.clone()));
        FAILURE.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some((at, source.retained()));
        });
        Self { source, drops }
    }
    pub(crate) fn assert_error(&self, error: &ControlledGenerationError) {
        let neutral = error
            .backend_failure()
            .expect("original provider classification");
        assert_eq!(neutral.kind(), BackendFailureKind::Busy);
        assert_eq!(neutral.operation(), "conformance-retained-hook");
        let actual = neutral.source().unwrap().downcast_ref::<Cause>().unwrap();
        let expected = self.source.source_error().downcast_ref::<Cause>().unwrap();
        assert!(
            std::ptr::eq(actual, expected),
            "conversion replaced the source owner"
        );
    }
}
impl Drop for Armed {
    fn drop(&mut self) {
        FAILURE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

pub(crate) fn check(at: &'static str) -> Result<(), MockError> {
    FAILURE.with(|slot| match slot.borrow().as_ref() {
        Some((selected, source)) if *selected == at => Err(MockError::ProviderRetained(
            source
                .retained()
                .into_failure()
                .with_operation("conformance-retained-hook"),
        )),
        _ => Ok(()),
    })
}

fn ignore(_: ControlledGenerationRecord) -> ControlFlow<()> {
    ControlFlow::Continue(())
}
fn no_record(_: ControlledGenerationRecord) -> ControlFlow<()> {
    panic!("rejected operation published a record")
}
fn copy_limits() -> SnapshotLimits {
    SnapshotLimits {
        max_snapshots: 3,
        max_branches: 2,
        retained_bytes: 64_000_000,
        cumulative_copy_bytes: 256_000_000,
    }
}
fn branch_options() -> GenerationBranchOptions {
    GenerationBranchOptions {
        trace_limits: limits(),
        capture_limits: None,
        sampling: None,
        intervention: None,
    }
}

#[test]
fn sampling_override_preserves_retained_provider_cause_and_invalid_control_branch() {
    let (mut model, chat, settings, _) = setup();
    let mut prepared =
        eredu::api::PreparedChatRequest::new(&chat, original_sources::settings(settings.clone()));
    let mut run = model
        .start_controlled_chat(prepared, limits(), Default::default(), ignore)
        .unwrap()
        .unwrap();
    let armed = Armed::new("sampling");
    // This invalid policy is checked before the backend callback. It must keep
    // the typed Invalid branch and leave the run able to try a valid request.
    let invalid = run
        .override_sampling(
            SamplingOverride {
                temperature: Some(0.7),
                reseed: None,
            },
            no_record,
        )
        .unwrap_err();
    assert!(invalid
        .session_failure()
        .unwrap()
        .sampling_rejection()
        .is_some());
    assert_eq!(run.status(), GenerationStatus::Prepared);
    let error = run
        .override_sampling(
            SamplingOverride {
                temperature: Some(0.7),
                reseed: Some(99),
            },
            no_record,
        )
        .unwrap_err();
    armed.assert_error(&error);
    assert_eq!(run.status(), GenerationStatus::Prepared);
    assert_eq!(run.next_prediction(), 0);
    assert!(run.token_ids().is_empty());
    let drops = armed.drops.clone();
    drop(run);
    drop(model);
    drop(armed);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn snapshot_capture_restore_and_branch_preserve_retained_provider_cause() {
    for at in ["capture", "copy", "growth"] {
        let (mut model, chat, settings, _) = super::snapshots::snapshot_setup();
        let mut prepared = eredu::api::PreparedChatRequest::new(
            &chat,
            original_sources::settings(settings.clone()),
        );
        let mut run = model
            .start_controlled_chat(prepared, limits(), Default::default(), ignore)
            .unwrap()
            .unwrap();
        let saved = {
            run.enable_snapshots(
                copy_limits(),
                crate::memory::limits(original_sources::CAPACITY),
                eredu_runtime::working_memory::WorkspaceCopyLimits::new(crate::memory::limits(
                    original_sources::CAPACITY,
                )),
            )
            .unwrap();
            Some(run.snapshot(ignore).unwrap())
        };
        let armed = Armed::new(at);
        let error = match at {
            "capture" => run.snapshot(no_record).map(|_| ()),
            "copy" => run.restore(saved.as_ref().unwrap(), no_record),
            "growth" => run
                .fork(saved.as_ref().unwrap(), branch_options(), no_record)
                .map(|_| ()),
            _ => unreachable!(),
        }
        .unwrap_err();
        armed.assert_error(&error);
        assert_eq!(run.next_prediction(), 0);
        assert!(run.token_ids().is_empty());
        let drops = armed.drops.clone();
        drop(saved);
        drop(run);
        drop(model);
        drop(armed);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(error);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn neutral_control_conversions_keep_classification_and_typed_host_failures() {
    let original =
        BackendFailure::from_error(std::io::Error::from(std::io::ErrorKind::InvalidInput));
    let error = ControlledGenerationError::Backend(original);
    assert_eq!(
        error.backend_failure().unwrap().kind(),
        BackendFailureKind::InvalidInput
    );
    assert!(error
        .backend_failure()
        .unwrap()
        .source()
        .unwrap()
        .is::<std::io::Error>());
    let error = ControlledGenerationError::Snapshot(eredu::api::GenerationSnapshotError::from(
        TextSnapshotError::<BackendFailure>::Host("host marker".into()),
    ));
    assert!(
        matches!(error,ControlledGenerationError::Snapshot(error) if matches!(error.cause(),TextSnapshotError::Host(message) if message=="host marker"))
    );
    let error = ControlledGenerationError::Snapshot(eredu::api::GenerationSnapshotError::from(
        TextSnapshotError::<BackendFailure>::HostPreparation(BackendFailure::new(
            BackendFailureKind::Busy,
            std::io::Error::other("host preparation"),
        )),
    ));
    assert_eq!(
        error.backend_failure().unwrap().kind(),
        BackendFailureKind::Busy
    );
}

#[test]
fn controlled_start_preserves_retained_provider_cause_after_model_drop() {
    let (mut model, chat, settings, _) = setup();
    let mut prepared =
        eredu::api::PreparedChatRequest::new(&chat, original_sources::settings(settings.clone()));
    let armed = Armed::new("start");
    let error = model
        .start_controlled_chat(prepared, limits(), Default::default(), no_record)
        .err()
        .unwrap();
    armed.assert_error(&error);
    let drops = armed.drops.clone();
    drop((model, armed));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
