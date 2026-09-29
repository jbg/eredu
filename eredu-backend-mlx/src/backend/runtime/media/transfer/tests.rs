//! Injected completion outcomes around real CPU copies. These tests exercise
//! the production loop and quarantine, not native device/driver fault behavior.

use super::*;
use crate::backend::runtime::media::input::{input_part, InputPayload, ModelInput};
use crate::backend::submission_recovery::{reap, wait_for_retirement};
use eredu_core::{InputMetadataKey, InputModality};
use std::{cell::Cell, panic::AssertUnwindSafe, rc::Rc};

#[derive(Clone, Copy, Debug)]
enum Fault {
    Error(usize),
    Panic(usize),
    FinalStatus,
}

#[derive(Clone, Copy, Debug)]
struct Observed {
    status: Status,
    arrays: usize,
    staging: bool,
}

struct InjectedCompletion {
    fault: Fault,
    calls: Rc<Cell<usize>>,
    observed: Rc<Cell<Option<Observed>>>,
    // This field drops only after all resource fields in the retention owner.
    _lifetime: Rc<()>,
}
impl Completion for InjectedCompletion {
    fn synchronize(&self, event: &safemlx::Event) -> Result<(), Error> {
        // Settle the real event before injecting its reported error/panic; an
        // artificial probe controls whether Recovery can release those roots.
        event.synchronize()?;
        let call = self.calls.get() + 1;
        self.calls.set(call);
        match self.fault {
            Fault::Error(at) if at == call => {
                Err(safemlx::error::Exception::custom("injected transfer completion error").into())
            }
            Fault::Panic(at) if at == call => panic!("injected transfer completion panic"),
            _ => Ok(()),
        }
    }
    fn observe(&self, status: Status, arrays: usize, staging: bool) {
        self.observed.set(Some(Observed {
            status,
            arrays,
            staging,
        }));
    }
}

struct ControlledProbe {
    scope: safemlx::SubmissionScope,
    status: Rc<Cell<Status>>,
}
impl Probe for ControlledProbe {
    fn seal(&mut self) {
        self.scope.seal();
    }
    fn progress(&self) -> Status {
        let native = self.scope.progress();
        let simulated = self.status.get();
        Status {
            settled: native.is_settled() && simulated.settled,
            failed: native.failed() || simulated.failed,
            blocked: native.blocked() || simulated.blocked,
        }
    }
}

fn source() -> PreparedModelInput {
    let parts = [
        input_part(
            InputModality::Text,
            InputPayload::TokenIds(Array::from_slice(&[16_777_217_u32, u32::MAX], &[1, 2])),
            [],
            [],
        )
        .unwrap(),
        input_part(
            InputModality::Image,
            InputPayload::Tensor(Array::from_slice(&[0.5_f32, -1.25, 2.0, 17.0], &[2, 2])),
            [(
                InputMetadataKey::PatchGrid,
                Array::from_slice(&[1_i32, 1, 2], &[1, 3]),
            )],
            [],
        )
        .unwrap(),
    ];
    let mut source = PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap();
    source.cache_identity = Some(source.inner.cache_identity("failure-source").unwrap());
    source
}

fn verify_values(source: &PreparedModelInput) {
    let arrays = source.wire_arrays();
    assert_eq!(
        arrays[0].evaluated().unwrap().as_slice::<u32>(),
        &[16_777_217, u32::MAX]
    );
    assert_eq!(
        arrays[1].evaluated().unwrap().as_slice::<f32>(),
        &[0.5, -1.25, 2.0, 17.0]
    );
    assert_eq!(arrays[2].evaluated().unwrap().as_slice::<i32>(), &[1, 1, 2]);
}

fn exercise(fault: Fault, failed: bool) {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let source = source();
    let identity = source.identity().clone();
    let cache_identity = source.cache_identity.clone();
    let resources = source.transfer_resources().unwrap();
    let calls = Rc::new(Cell::new(0));
    let observed = Rc::new(Cell::new(None));
    let lifetime = Rc::new(());
    let released = Rc::downgrade(&lifetime);
    let status = Rc::new(Cell::new(Status {
        settled: false,
        failed,
        blocked: !failed,
    }));
    let recovery = Recovery::with_probe(
        InputTransferRetention {
            arrays: source.wire_arrays(),
            staging: None,
            completion: InjectedCompletion {
                fault,
                calls: Rc::clone(&calls),
                observed: Rc::clone(&observed),
                _lifetime: lifetime,
            },
        },
        ControlledProbe {
            scope: safemlx::SubmissionScope::begin().unwrap(),
            status: Rc::clone(&status),
        },
    );
    let mut published = None;
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let (copied, _) = copy_with_recovery(&source, &stream, resources, recovery)?;
        published = Some(copied);
        Ok::<(), Error>(())
    }));
    match fault {
        Fault::Panic(_) => assert!(outcome.is_err()),
        Fault::Error(_) => {
            let Error::Exception(cause) = outcome.unwrap().unwrap_err() else {
                panic!("the exact completion failure must survive recovery");
            };
            assert_eq!(cause.what(), "injected transfer completion error");
        }
        Fault::FinalStatus => assert!(matches!(outcome.unwrap(), Err(Error::ArchitectureModel(_)))),
    }
    assert!(
        published.is_none(),
        "failed copies must not publish partial input"
    );
    assert_eq!(source.identity(), &identity);
    assert_eq!(source.cache_identity, cache_identity);
    verify_values(&source);
    // Repeated nonblocking housekeeping cannot reinterpret an error as a
    // terminal event, even when the underlying CPU transfer has completed.
    for _ in 0..3 {
        reap();
        assert!(released.upgrade().is_some());
    }
    let retained = observed
        .get()
        .expect("Recovery must observe its retained owner");
    assert!(!retained.status.settled);
    assert_eq!(retained.status.failed, failed);
    assert_eq!(retained.status.blocked, !failed);
    match fault {
        Fault::Error(at) | Fault::Panic(at) => {
            assert_eq!(calls.get(), at, "no later tensor may start after failure");
            assert_eq!(retained.arrays, 3 + at / 2);
            assert!(
                retained.staging,
                "current physical staging must remain owned"
            );
        }
        Fault::FinalStatus => {
            assert_eq!(calls.get(), 6);
            assert_eq!(retained.arrays, 6);
            assert!(
                !retained.staging,
                "completed serial copies retire their staging"
            );
        }
    }
    // The failed request remains quarantined while an independent ordinary
    // operation can still complete with exact source and semantic identities.
    let (copied, report) = source
        .transfer_to_stream(&stream, resources.into_budget())
        .unwrap();
    assert_eq!(report, resources);
    assert_eq!(copied.identity(), &identity);
    assert_eq!(copied.cache_identity, cache_identity);
    verify_values(&copied);
    assert!(released.upgrade().is_some());
    status.set(Status {
        settled: true,
        failed,
        blocked: false,
    });
    wait_for_retirement(|| released.upgrade().is_none());
    assert!(
        released.upgrade().is_none(),
        "terminal owner and roots must retire"
    );
}

#[test]
fn staged_download_completion_errors_retain_registered_staging_until_terminal() {
    for failed in [false, true] {
        for at in [1, 3] {
            exercise(Fault::Error(at), failed);
        }
    }
}

#[test]
fn staged_upload_completion_errors_retain_destinations_until_terminal() {
    for failed in [false, true] {
        for at in [2, 4] {
            exercise(Fault::Error(at), failed);
        }
    }
}

#[test]
fn staged_transfer_unwind_retains_current_staging_and_created_destinations() {
    for failed in [false, true] {
        for at in [1, 2, 4] {
            exercise(Fault::Panic(at), failed);
        }
    }
}

#[test]
fn staged_transfer_final_scope_failure_never_publishes_completed_destinations() {
    for failed in [false, true] {
        exercise(Fault::FinalStatus, failed);
    }
}
