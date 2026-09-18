use super::*;
use eredu_core::{BackendFailure, BackendFailureKind, SharedBackendFailure};
use std::{
    cell::{Cell, RefCell},
    error::Error as _,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Owner {
    Executor,
    Sampler,
}
struct Injection {
    at: &'static str,
    owner: Owner,
    source: SharedBackendFailure,
    ordinary: bool,
}
thread_local! {
    static INJECTION: RefCell<Option<Injection>> = const { RefCell::new(None) };
    static HOOKS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}
#[derive(Debug, thiserror::Error)]
#[error("retained speculative fixture cause")]
struct Cause(Arc<AtomicUsize>);
impl Drop for Cause {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct Armed {
    source: SharedBackendFailure,
    drops: Arc<AtomicUsize>,
}
impl Armed {
    fn new(at: &'static str, owner: Owner, ordinary: bool) -> Self {
        let drops = Arc::new(AtomicUsize::new(0));
        let source = SharedBackendFailure::new(BackendFailureKind::Busy, Cause(drops.clone()));
        INJECTION.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(Injection {
                at,
                owner,
                source: source.retained(),
                ordinary,
            });
        });
        HOOKS.set((0, 0));
        Self { source, drops }
    }
    fn clear(&self) {
        INJECTION.with(|slot| {
            slot.borrow_mut().take();
        });
    }
    fn assert_error(&self, error: &SpeculativeControlError) {
        let SpeculativeControlError::Backend(error) = error else {
            panic!("wrong control branch: {error}")
        };
        assert_eq!(error.kind(), BackendFailureKind::Busy);
        assert_eq!(error.operation(), "speculative retained owner");
        assert!(std::ptr::eq(
            error.source().unwrap().downcast_ref::<Cause>().unwrap(),
            self.source.source_error().downcast_ref::<Cause>().unwrap(),
        ));
    }
}
impl Drop for Armed {
    fn drop(&mut self) {
        self.clear();
    }
}

pub(crate) fn check(at: &'static str) -> Result<(), MockError> {
    INJECTION.with(|slot| match slot.borrow().as_ref() {
        Some(x) if x.at == at && x.ordinary => Err(MockError::Token(999)),
        Some(x) if x.at == at => Err(MockError::ProviderRetained(
            x.source
                .retained()
                .into_failure()
                .with_operation("speculative retained owner"),
        )),
        _ => Ok(()),
    })
}
pub(crate) fn take(error: MockError, owner: Owner) -> Result<BackendFailure, MockError> {
    INJECTION.with(|slot| {
        let slot = slot.borrow();
        let Some(x) = slot.as_ref() else {
            return Err(error);
        };
        HOOKS.with(|calls| {
            let (e, s) = calls.get();
            calls.set((
                e + usize::from(owner == Owner::Executor),
                s + usize::from(owner == Owner::Sampler),
            ));
        });
        if x.owner != owner {
            return Err(error);
        }
        match error {
            MockError::ProviderRetained(error) => Ok(error),
            error => Err(error),
        }
    })
}

#[test]
fn controlled_scheduler_transfers_executor_and_sampler_sources_through_actual_driver() {
    for (at, owner, calls) in [
        ("prefill", Owner::Executor, (1, 0)),
        ("sample", Owner::Sampler, (1, 1)),
    ] {
        let (mut model, chat, settings) = setup();
        let armed = Armed::new(at, owner, false);
        let mut events = Vec::new();
        let error = model
            .with_controlled_prepared_chat_speculative(
                PreparedChatSpeculativeRequest {
                    chat: &chat,
                    input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                    output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                    skip_special_tokens: true,
                    drafting: SpeculativeDraft::Embedded,
                    settings,
                    options: Default::default(),
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |e| events.push(e),
                },
                options(),
                |session| {
                    let error = session.step().unwrap_err();
                    armed.assert_error(&error);
                    assert!(session.token_ids().is_empty());
                    assert!(matches!(
                        session.step(),
                        Err(SpeculativeControlError::Failed)
                    ));
                    Err(error)
                },
            )
            .unwrap_err();
        armed.assert_error(error.control_failure().expect("retained controlled error"));
        assert_eq!(HOOKS.get(), calls);
        assert!(events.is_empty());
        let drops = armed.drops.clone();
        drop(model);
        drop(armed);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(error);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn repeated_reseed_errors_keep_sources_and_leave_sampling_and_tokens_unchanged() {
    let (mut model, chat, settings) = setup();
    let baseline = model
        .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
            chat: &chat,
            input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
            output_mode: eredu::api::PreparedChatOutputMode::Semantic,
            skip_special_tokens: true,
            drafting: SpeculativeDraft::Embedded,
            settings,
            options: Default::default(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        })
        .unwrap();
    let mut errors = Vec::new();
    let mut final_drops = None;
    let output = model
        .with_controlled_prepared_chat_speculative(
            PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                skip_special_tokens: true,
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            },
            options(),
            |session| {
                session.step()?;
                assert_eq!(session.token_ids(), [7]);
                let before = session.sampling_state();
                let armed = Armed::new("reseed", Owner::Sampler, false);
                assert!(matches!(
                    session.override_sampling(eredu::api::SamplingOverride {
                        temperature: Some(-1.0),
                        reseed: Some(5)
                    }),
                    Err(SpeculativeControlError::Invalid(_))
                ));
                assert_eq!(HOOKS.get(), (0, 0));
                for seed in [5, 6, 7] {
                    let error = session
                        .override_sampling(eredu::api::SamplingOverride {
                            temperature: Some(0.7),
                            reseed: Some(seed),
                        })
                        .unwrap_err();
                    armed.assert_error(&error);
                    assert_eq!(session.sampling_state(), before);
                    assert_eq!(session.token_ids(), [7]);
                    errors.push(error);
                }
                assert_eq!(HOOKS.get(), (0, 3));
                armed.clear();
                session.override_sampling(eredu::api::SamplingOverride {
                    temperature: Some(0.7),
                    reseed: Some(8),
                })?;
                assert_eq!(session.sampling_state().unwrap().temperature, 0.7);
                final_drops = Some(armed.drops.clone());
                drop(armed);
                while session.step()?.is_some() {}
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(output.token_ids(), baseline.token_ids());
    drop(model);
    let drops = final_drops.unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    while errors.len() > 1 {
        drop(errors.pop());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
    drop(errors);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn ordinary_speculative_failure_keeps_driver_wrapper_and_both_refused_hook_calls() {
    let (mut model, chat, settings) = setup();
    let armed = Armed::new("prefill", Owner::Executor, true);
    let error = model
        .with_controlled_prepared_chat_speculative(
            PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&[3, 4]),
                output_mode: eredu::api::PreparedChatOutputMode::Semantic,
                skip_special_tokens: true,
                drafting: SpeculativeDraft::Embedded,
                settings,
                options: Default::default(),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| panic!("failed prefill published"),
            },
            options(),
            |session| {
                session.step()?;
                Ok(())
            },
        )
        .unwrap_err();
    let Some(SpeculativeControlError::Backend(backend)) = error.control_failure() else {
        panic!("wrong backend branch")
    };
    assert_eq!(HOOKS.get(), (1, 1));
    assert_eq!(backend.kind(), BackendFailureKind::Other);
    let wrapped = backend
        .source()
        .unwrap()
        .downcast_ref::<eredu_core::SpeculativeDriverError<MockError>>()
        .unwrap();
    assert!(matches!(
        wrapped,
        eredu_core::SpeculativeDriverError::Backend(MockError::Token(999))
    ));
    assert!(wrapped.to_string().contains("999"));
    drop(armed);
}
