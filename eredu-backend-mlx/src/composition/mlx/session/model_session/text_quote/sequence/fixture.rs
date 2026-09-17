//! Scoped tests select only an actual private entry/target; claims stay core-owned.
use super::*;
use std::{error::Error as _, rc::Weak};

#[derive(Clone, Copy, Default)]
pub(in crate::composition::mlx::session::model_session::text_quote) enum Mode {
    #[default]
    Normal,
    Defer,
    DeferOriginalPrompt,
    CheckForeignOriginalInput,
    Fence,
    FenceProvider,
    TakeSampling,
    ReplaceWorkScope,
    InstallRowsTwice,
    Busy,
    PanicAfterTake,
    DecoderAcceptedError,
    DecoderAcceptedPanic,
    DecoderFundingError,
    DecoderFundingPanic,
}
#[derive(Clone, Copy, Debug)]
pub(in crate::composition::mlx::session::model_session::text_quote) struct Facts {
    pub(super) required: u64,
    pub(in crate::composition::mlx::session::model_session::text_quote) held: u64,
    pub(in crate::composition::mlx::session::model_session::text_quote) r: u64,
    pub(super) admission: u64,
    pub(super) work: u64,
    pub(super) capture: bool,
    pub(in crate::composition::mlx::session::model_session::text_quote) source: bool,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct InstalledFacts {
    pub(super) same_source: bool,
    pub(in crate::composition::mlx::session::model_session::text_quote) held: u64,
    pub(super) r: u64,
}
#[derive(Debug, thiserror::Error)]
#[error("test retained the actual prepared sampling state")]
pub(super) struct SamplingIntercepted;
struct Slot {
    session: Weak<Cell<bool>>,
    source: Option<eredu_core::SharedStorageIdentity>,
    rows: bool,
    mode: Cell<Mode>,
    target: RefCell<Option<MlxTextPreparation>>,
    admitted: RefCell<Option<MlxTextPreparation>>,
    calls: Cell<usize>,
    context: RefCell<Option<eredu_core::TextStepContext>>,
    decoder_location: Cell<usize>,
    decoder_takes: Cell<usize>,
    decoder_retirements: Cell<usize>,
    decoder_expected: RefCell<Option<(WorkingMemoryPool, u64)>>,
    facts: Cell<Option<Facts>>,
    installed: Cell<Option<InstalledFacts>>,
    sampling: RefCell<Option<MlxTextGenerationState>>,
    sampling_taken: Cell<bool>,
    operation_quote: RefCell<Option<super::super::TextExecutionQuoteOwner>>,
    foreign_scope: RefCell<Option<eredu_runtime::working_memory::WorkingMemoryFundingScope>>,
    operation_faults: Cell<usize>,
    installation_faults: Cell<usize>,
}
thread_local! {
    static CURRENT: RefCell<Option<Rc<Slot>>> = const { RefCell::new(None) };
}
pub(in crate::composition::mlx::session::model_session::text_quote) struct Probe(Rc<Slot>);
impl Probe {
    pub(in crate::composition::mlx::session::model_session::text_quote) fn new(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        source: Option<&eredu_core::capture::SharedCapturePlan>,
        rows: bool,
    ) -> Self {
        let slot = Rc::new(Slot {
            session: Rc::downgrade(&runtime.session().poison),
            source: source.map(|s| s.storage_identity().clone()),
            rows,
            mode: Cell::new(Mode::Normal),
            target: RefCell::new(None),
            admitted: RefCell::new(None),
            calls: Cell::new(0),
            context: RefCell::new(None),
            decoder_location: Cell::new(0),
            decoder_takes: Cell::new(0),
            decoder_retirements: Cell::new(0),
            decoder_expected: RefCell::new(None),
            facts: Cell::new(None),
            installed: Cell::new(None),
            sampling: RefCell::new(None),
            sampling_taken: Cell::new(false),
            operation_quote: RefCell::new(None),
            foreign_scope: RefCell::new(None),
            operation_faults: Cell::new(0),
            installation_faults: Cell::new(0),
        });
        CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            assert!(current.is_none());
            *current = Some(slot.clone());
        });
        Self(slot)
    }
    pub(in crate::composition::mlx::session::model_session::text_quote) fn context(
        &self,
    ) -> eredu_core::TextStepContext {
        self.0
            .context
            .borrow()
            .clone()
            .expect("actual original core binding")
    }
    pub(super) fn foreign_scope(
        &self,
        scope: eredu_runtime::working_memory::WorkingMemoryFundingScope,
    ) {
        assert!(self.0.foreign_scope.replace(Some(scope)).is_none());
    }
    pub(super) fn operation_faults(&self) -> usize {
        self.0.operation_faults.get()
    }
    pub(super) fn installation_faults(&self) -> usize {
        self.0.installation_faults.get()
    }
    pub(in crate::composition::mlx::session::model_session::text_quote) fn facts(&self) -> Facts {
        self.0.facts.get().unwrap()
    }
    pub(super) fn installed(&self) -> InstalledFacts {
        self.0
            .installed
            .get()
            .expect("actual successful capture installation")
    }
    pub(super) fn take_sampling(&self) -> MlxTextGenerationState {
        self.0
            .sampling
            .borrow_mut()
            .take()
            .expect("actual intercepted Sampling")
    }
    pub(in crate::composition::mlx::session::model_session::text_quote) fn take(
        &self,
    ) -> MlxTextPreparation {
        self.0
            .admitted
            .borrow_mut()
            .take()
            .expect("actual native original admission")
    }
    pub(in crate::composition::mlx::session::model_session::text_quote) fn mode(&self, mode: Mode) {
        self.0.mode.set(mode);
    }
    pub(super) fn target(&self, target: MlxTextPreparation) {
        let old = self.0.target.replace(Some(target));
        drop(old);
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        let removed = CURRENT.with(|current| current.borrow_mut().take());
        assert!(
            removed
                .as_ref()
                .is_some_and(|slot| Rc::ptr_eq(slot, &self.0))
        );
        drop(removed);
    }
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn decoder_taken(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: Option<&eredu_core::capture::SharedCapturePlan>,
    decoder: &Option<OriginalGenerationDecoderSource>,
) {
    let Some(slot) = current().filter(|slot| {
        matches(slot, runtime) && slot.source.as_ref() == source.map(|s| s.storage_identity())
    }) else {
        return;
    };
    if decoder.is_some() {
        slot.decoder_location
            .set(std::ptr::from_ref(decoder) as usize);
        slot.decoder_takes.set(slot.decoder_takes.get() + 1);
    }
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn decoder_checkpoint(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    funded: bool,
) -> Result<(), BackendFailure> {
    let Some(slot) =
        current().filter(|slot| matches(slot, runtime) && slot.decoder_location.get() != 0)
    else {
        return Ok(());
    };
    let mode = slot.mode.get();
    let selected = match mode {
        Mode::DecoderAcceptedError | Mode::DecoderAcceptedPanic => !funded,
        Mode::DecoderFundingError | Mode::DecoderFundingPanic => funded,
        _ => false,
    };
    if !selected {
        return Ok(());
    }
    slot.mode.set(Mode::Normal); // one-shot, reset before rejection/unwind
    let pool = runtime.session().payload.memory_pool.clone();
    let expected = pool.used_bytes().unwrap();
    assert!(
        slot.decoder_expected
            .replace(Some((pool, expected)))
            .is_none()
    );
    if matches!(mode, Mode::DecoderAcceptedPanic | Mode::DecoderFundingPanic) {
        panic!("actual original decoder staging boundary");
    }
    Err(Rejection::IdentityMismatch.into_backend_failure())
}
pub(super) fn decoder_retired(location: usize) {
    let Some(slot) = current().filter(|slot| slot.decoder_location.get() == location) else {
        return;
    };
    let expected = slot.decoder_expected.borrow().clone();
    if let Some((pool, expected)) = expected {
        // Source destruction already completed; its original charge still lives.
        assert_eq!(pool.used_bytes().unwrap(), expected);
        slot.decoder_retirements
            .set(slot.decoder_retirements.get() + 1);
    }
}
fn current() -> Option<Rc<Slot>> {
    CURRENT.with(|current| current.borrow().clone())
}
fn matches(slot: &Slot, runtime: &ModelRuntime<MlxBackend<'_>>) -> bool {
    slot.session
        .upgrade()
        .is_some_and(|session| Rc::ptr_eq(&session, &runtime.session().poison))
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn opening_rows(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: Option<&eredu_core::capture::SharedCapturePlan>,
) -> bool {
    current().is_some_and(|slot| {
        slot.rows
            && matches(&slot, runtime)
            && slot.source.as_ref() == source.map(|s| s.storage_identity())
    })
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn record(
    preparation: &MlxTextPreparation,
) {
    let Some(slot) = current() else {
        return;
    };
    let Some(quote) = preparation.quote.as_ref() else {
        return;
    };
    if !slot
        .session
        .upgrade()
        .is_some_and(|session| Rc::ptr_eq(&session, &quote.session))
    {
        return;
    }
    assert!(
        slot.admitted.borrow().is_none(),
        "take previous actual preparation first"
    );
    // The scoped probe can observe successive genuine admissions. Each keeps
    // its own once-only binding; recording a new one retires the old context.
    drop(slot.context.take());
    let (held, r, admission, work, source) = match &quote.capture {
        Some(capture) => capture.sequence_facts_for_test(),
        None => {
            let span = quote.sequence.as_ref().unwrap().span.as_ref().unwrap();
            let controls = span.workspace().text_controls().unwrap();
            (
                span.protected_host_bytes(),
                controls.sequence_storage_bytes(),
                controls.facts().admission_bytes().unwrap(),
                controls.facts().work_bytes().unwrap(),
                controls.source_identity().is_some(),
            )
        }
    };
    slot.facts.set(Some(Facts {
        required: quote.request().memory_reservation().unwrap().bytes(),
        held,
        r,
        admission,
        work,
        capture: quote.capture.is_some(),
        source,
    }));
    if matches!(slot.mode.get(), Mode::ReplaceWorkScope) {
        *slot.operation_quote.borrow_mut() = Some(quote.clone());
    }
    *slot.admitted.borrow_mut() = Some(preparation.clone());
}
// Capture-only setup has no sequence extraction callback. Record its actual
// admission only for this scoped TakeSampling test, with exact source/session
// matching. The shared core driver still binds the preparation and runs Prompt.
pub(in crate::composition::mlx::session::model_session::text_quote) fn record_capture_sampling(
    preparation: &MlxTextPreparation,
    source: &eredu_core::capture::SharedCapturePlan,
) {
    let Some(slot) = current().filter(|slot| {
        matches!(slot.mode.get(), Mode::TakeSampling)
            && !slot.rows
            && slot.source.as_ref() == Some(source.storage_identity())
    }) else {
        return;
    };
    let Some(quote) = preparation.quote.as_ref() else {
        return;
    };
    if quote.sequence.is_some() || !quote.has_capture() {
        return;
    }
    // record() independently checks this exact native session before keeping
    // the same original preparation; no state or claim is manufactured.
    drop(slot);
    record(preparation);
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(in crate::composition::mlx::session::model_session::text_quote) fn capture_only_sampling(
    runtime: &mut ModelRuntime<MlxBackend<'static>>,
    source: &eredu_core::capture::SharedCapturePlan,
    config: TextGenerationConfig,
) -> (MlxTextPreparation, MlxTextGenerationState) {
    use eredu_core::{TextGenerationDriver, TextGenerationInput, TextPreparationOptions};
    let probe = Probe::new(runtime, Some(source), false);
    probe.mode(Mode::TakeSampling);
    let mut driver = TextGenerationDriver::new(runtime);
    let rejected = driver.start_input_with_options(
        TextGenerationInput::TokenIds(vec![2, 5, 7]),
        config,
        crate::composition::mlx::session::model_session::disk_layerwise_tests::Controller::default(),
        TextPreparationOptions { interventions: None, capture: Some(source.clone()) },
    ).err().expect("actual capture-only Sampling interception");
    let mut cause: &(dyn std::error::Error + 'static) = &rejected;
    while !cause.is::<SamplingIntercepted>() {
        cause = cause
            .source()
            .expect("exact typed Sampling interception source");
    }
    drop((rejected, driver));
    assert_eq!(probe.0.calls.get(), 0, "no sequence claim or R extraction");
    assert_eq!(probe.facts().r, 0);
    assert!(probe.0.sampling_taken.get());
    let state = probe.take_sampling();
    let preparation = probe.take();
    assert!(preparation.quote.as_ref().unwrap().sequence.is_none());
    assert!(
        state
            .sampling
            .quote
            .as_ref()
            .unwrap()
            .same_owner(preparation.quote.as_ref().unwrap())
    );
    drop(probe);
    (preparation, state)
}

pub(in crate::composition::mlx::session::model_session::text_quote) fn prepare(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    preparation: &MlxTextPreparation,
    claim: GenerationSequencePreparation<'_, '_>,
) -> Result<RetainedGenerationSequence, BackendFailure> {
    let Some(slot) = current().filter(|slot| matches(slot, runtime)) else {
        return super::prepare(runtime, preparation, claim);
    };
    slot.calls.set(slot.calls.get().checked_add(1).unwrap());
    let target = slot.target.borrow().clone();
    let target = target.as_ref().unwrap_or(preparation);
    match slot.mode.get() {
        Mode::Defer => Err(Rejection::Unavailable.into_backend_failure()),
        Mode::DeferOriginalPrompt => {
            assert!(claim.request().token_input().is_some());
            let sequence = super::prepare(runtime, target, claim)?;
            drop(sequence);
            Err(Rejection::Unavailable.into_backend_failure())
        }
        Mode::CheckForeignOriginalInput => {
            // Two actual same-session admissions; neither a core claim nor its
            // source borrow is reconstructed. Rejections must precede I and R.
            slot.mode.set(Mode::Normal);
            let previous = target.quote.as_ref().unwrap();
            let actual = preparation.quote.as_ref().unwrap();
            assert!(!previous.same_owner(actual));
            let old_input = previous.original_token_input().unwrap();
            let new_input = actual.original_token_input().unwrap();
            assert!(old_input.is_unclaimed_for_test());
            assert!(new_input.is_unclaimed_for_test());
            for _ in 0..16 {
                let error = old_input
                    .prepare(target.request.as_ref().unwrap(), &claim)
                    .unwrap_err();
                assert_eq!(
                    error
                        .source()
                        .unwrap()
                        .downcast_ref::<eredu_core::TokenInputRejection>(),
                    Some(&eredu_core::TokenInputRejection::IdentityMismatch)
                );
                assert!(old_input.is_unclaimed_for_test());
                assert!(
                    previous
                        .sequence
                        .as_ref()
                        .unwrap()
                        .pending
                        .borrow()
                        .is_some()
                );
                assert!(actual.sequence.as_ref().unwrap().pending.borrow().is_some());
                let error = new_input
                    .prepare(target.request.as_ref().unwrap(), &claim)
                    .unwrap_err();
                assert_eq!(
                    error
                        .source()
                        .unwrap()
                        .downcast_ref::<eredu_core::TokenInputRejection>(),
                    Some(&eredu_core::TokenInputRejection::IdentityMismatch)
                );
                assert!(new_input.is_unclaimed_for_test());
                assert!(
                    previous
                        .sequence
                        .as_ref()
                        .unwrap()
                        .pending
                        .borrow()
                        .is_some()
                );
                assert!(actual.sequence.as_ref().unwrap().pending.borrow().is_some());
            }
            assert!(old_input.is_unclaimed_for_test());
            assert!(new_input.is_unclaimed_for_test());
            assert!(
                previous
                    .sequence
                    .as_ref()
                    .unwrap()
                    .pending
                    .borrow()
                    .is_some()
            );
            assert!(actual.sequence.as_ref().unwrap().pending.borrow().is_some());
            super::prepare(runtime, preparation, claim)
        }
        Mode::Fence => {
            target
                .quote
                .as_ref()
                .unwrap()
                .take_funding_run()
                .unwrap()
                .close()
                .unwrap();
            super::prepare(runtime, target, claim)
        }
        Mode::FenceProvider => {
            let sequence = super::prepare(runtime, target, claim)?;
            // Sampling has not yet taken this original run. Fence only after
            // the genuine claim has produced its dormant provider.
            target
                .quote
                .as_ref()
                .unwrap()
                .take_funding_run()
                .unwrap()
                .close()
                .unwrap();
            let cause = sequence.prepare_storage().unwrap_err();
            Err(BackendFailure::new(
                BackendFailureKind::InvalidSession,
                cause,
            ))
        }
        Mode::Busy => {
            let quote = target.quote.as_ref().unwrap();
            let loan = quote.sequence.as_ref().unwrap().pending.borrow();
            let result = super::prepare(runtime, target, claim);
            drop(loan);
            result
        }
        Mode::PanicAfterTake => {
            let bank = target
                .quote
                .as_ref()
                .unwrap()
                .sequence
                .as_ref()
                .unwrap()
                .take()
                .unwrap();
            let _bank = bank;
            panic!("native original bank taken before unwind");
        }
        Mode::Normal
        | Mode::TakeSampling
        | Mode::ReplaceWorkScope
        | Mode::InstallRowsTwice
        | Mode::DecoderAcceptedError
        | Mode::DecoderAcceptedPanic
        | Mode::DecoderFundingError
        | Mode::DecoderFundingPanic => super::prepare(runtime, target, claim),
    }
}

pub(in crate::composition::mlx::session::model_session::text_quote) fn intercept_claimed_work(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &MlxTextGenerationState,
    operation: &crate::composition::mlx::session::model_session::text_step::TextOperation<'_>,
) -> Result<(), Error> {
    let Some(slot) = current().filter(|slot| {
        matches!(slot.mode.get(), Mode::ReplaceWorkScope)
            && matches(slot, runtime)
            && state.sampling.quote.as_ref().is_some_and(|quote| {
                slot.operation_quote
                    .borrow()
                    .as_ref()
                    .is_some_and(|original| original.same_owner(quote))
            })
    }) else {
        return Ok(());
    };
    slot.mode.set(Mode::Normal);
    assert_eq!(slot.operation_faults.replace(1), 0);
    let foreign = slot
        .foreign_scope
        .borrow_mut()
        .take()
        .expect("second real admission scope");
    operation.replace_unsubmitted_scope_for_test(foreign)
}

pub(in crate::composition::mlx::session::model_session::text_quote) fn intercept_consumed_row_installation(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    preparation: &MlxTextPreparation,
    source: &eredu_core::capture::SharedCapturePlan,
    rows: &crate::composition::mlx::replicated_text::NativeOpeningRowsOwner,
) -> Result<(), Error> {
    let Some(slot) = current().filter(|slot| {
        matches!(slot.mode.get(), Mode::InstallRowsTwice)
            && matches(slot, runtime)
            && matches_preparation(slot, preparation)
            && slot.source.as_ref() == Some(source.storage_identity())
    }) else {
        return Ok(());
    };
    slot.mode.set(Mode::Normal);
    assert_eq!(slot.installation_faults.replace(1), 0);
    // Actual first installation consumes the one-use rows. The unchanged
    // production installation then performs the real duplicate rejection.
    runtime
        .session()
        .payload
        .model
        .erased()
        .install_opening_rows(rows)
}

// The quote is the exact one recorded during this scoped genuine admission.
// No RefCell loan escapes this identity check or survives the actual state move.
fn matches_preparation(slot: &Slot, preparation: &MlxTextPreparation) -> bool {
    let Some(quote) = preparation.quote.as_ref() else {
        return false;
    };
    slot.session
        .upgrade()
        .is_some_and(|session| Rc::ptr_eq(&session, &quote.session))
        && slot
            .admitted
            .borrow()
            .as_ref()
            .and_then(|p| p.quote.as_ref())
            .is_some_and(|actual| actual.same_owner(quote))
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn intercept_sampling(
    preparation: &MlxTextPreparation,
    state: MlxTextGenerationState,
) -> Result<MlxTextGenerationState, Error> {
    let Some(slot) = current().filter(|slot| {
        matches!(slot.mode.get(), Mode::TakeSampling) && matches_preparation(slot, preparation)
    }) else {
        return Ok(state);
    };
    assert!(
        state
            .sampling
            .quote
            .as_ref()
            .unwrap()
            .same_owner(preparation.quote.as_ref().unwrap())
    );
    assert!(
        !slot.sampling_taken.replace(true),
        "one actual Sampling interception"
    );
    let previous = slot.sampling.replace(Some(state));
    assert!(previous.is_none());
    drop(previous);
    Err(Error::Other(Box::new(SamplingIntercepted)))
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn record_installation(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    state: &MlxTextGenerationState,
    preparation: &MlxTextPreparation,
    source: &eredu_core::capture::SharedCapturePlan,
) {
    let Some(slot) = current().filter(|slot| {
        matches(slot, runtime)
            && matches_preparation(slot, preparation)
            && slot.source.as_ref() == Some(source.storage_identity())
    }) else {
        return;
    };
    assert!(
        state
            .sampling
            .quote
            .as_ref()
            .unwrap()
            .same_owner(preparation.quote.as_ref().unwrap())
    );
    let installed = state.funded_capture.as_ref().unwrap();
    let facts = InstalledFacts {
        same_source: installed.source().same_storage(source),
        held: installed.span_workspace().protected_host_bytes(),
        r: installed
            .span_workspace()
            .workspace()
            .text_controls()
            .unwrap()
            .sequence_storage_bytes(),
    };
    assert!(slot.installed.replace(Some(facts)).is_none());
}

#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
pub(in crate::composition::mlx::session::model_session::text_quote) mod tests;

pub(in crate::composition::mlx::session::model_session::text_quote) fn record_prediction_context(
    preparation: &MlxTextPreparation,
    context: &eredu_core::TextStepContext,
) {
    let Some(slot) = current().filter(|slot| matches_preparation(slot, preparation)) else {
        return;
    };
    assert!(slot.context.replace(Some(context.clone())).is_none());
}
