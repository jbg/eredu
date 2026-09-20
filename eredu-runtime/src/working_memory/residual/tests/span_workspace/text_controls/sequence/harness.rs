use super::*;
use eredu_core::run_preparation::{
    TextPreparationOutcome as Outcome, TextPreparationStage as Stage,
    TextPreparationStatus as Status,
};
use eredu_core::{
    BackendDescriptor, BackendFailure, BackendProvider, BackendSession, Completion,
    DeviceCapabilities, DeviceDescriptor, GenerationSequencePreparation, ModelRuntime,
    ObservationSet, PendingTextInput, PreparedModel, RetainedGenerationSequence,
    SessionCapabilities, Submission, TextGenerationBackend, TextGenerationConfig,
    TextPreparationInput, TextPreparationOptions, TextStepContext, TokenFilter,
    TokenFilterController, TokenOutput,
};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Mode {
    pub capture: bool,
    pub short: bool,
    pub abandon: bool,
    pub source: bool,
    pub explicit_source: bool,
    pub audit: bool,
    pub defer_bank: bool,
    pub fence_bank: bool,
    pub panic_after_take: bool,
    pub borrow_bank: bool,
    pub audit_decoder: bool,
    pub prediction_scopes: bool,
    pub prefill_scopes: bool,
    pub host_source_component: Option<u64>,
    pub reject_decoder_attachment: bool,
    pub defer_input: bool,
    pub fence_input: bool,
    pub input_retry: bool,
    pub fail_terminal_reserve: bool,
    pub copy_headroom: u64,
}
pub(super) struct State {
    pub mode: Mode,
    pub pool: WorkingMemoryPool,
    pub root: Option<WorkingMemoryStorage<u32>>,
    pub active: Option<Rc<Preparation>>,
    pub source_scope: Option<WorkingMemoryFundingScope>,
    pub source_run: Option<WorkingMemoryFundingRun>,
    pub source_reservation: Option<WorkingMemoryReservation>,
    pub source_envelope: u64,
    pub pending_bank: Option<crate::working_memory::OriginalGenerationSequenceBank>,
    pub foreign: Option<Rc<Preparation>>,
    pub order: Vec<&'static str>,
    pub votes: Vec<(Stage, Status)>,
    pub p: u64,
    pub r: u64,
    pub held: u64,
    pub decoder_takes: usize,
    pub loaded_bytes: u64,
    pub bound: Option<TextStepContext>,
    pub issued: Vec<crate::working_memory::OriginalTextPredictionScopeSet>,
    pub prefill_issued: Vec<crate::working_memory::OriginalTextPrefillScopeSet>,
    pub decoder_rejections_untaken: usize,
    pub input_bytes: u64,
    pub input_builds: usize,
    pub input_candidates: usize,
    pub input_address: usize,
    pub input_observed: Vec<u32>,
    pub pending_input: Option<crate::working_memory::OriginalTokenInputBank>,
}
pub(super) struct Preparation {
    pub request: InferenceTextPreparation,
    pub owner: RefCell<OwnedTextSpanWorkspace>,
    pub decoder_bank: RefCell<Option<crate::working_memory::OriginalGenerationSequenceBank>>,
    pub prediction_bank: RefCell<Option<crate::working_memory::OriginalTextPredictionScopes>>,
    pub prefill_bank: RefCell<Option<crate::working_memory::OriginalTextPrefillScopes>>,
    pub run: RefCell<Option<WorkingMemoryFundingRun>>,
    pub input: RefCell<Option<crate::working_memory::OwnedPromptTokenIds>>,
}
#[derive(Clone)]
pub(super) struct Backend(pub Rc<RefCell<State>>);
pub(super) struct Session;
#[derive(Debug, Clone)]
pub(super) struct Token {
    pub value: u32,
    pub receipt: crate::working_memory::InferenceTextStepReceipt,
}
pub(super) struct Done;
impl Completion for Done {
    type Error = WorkingMemoryError;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl TokenOutput for Token {
    type Error = WorkingMemoryError;
    fn token_id(&self) -> Result<u32, Self::Error> {
        Ok(self.value)
    }
}
impl BackendProvider for Backend {
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = WorkingMemoryError;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("original-r-neutral", "1")
    }
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        Ok(vec![])
    }
    fn prepare_model(&self, _: ()) -> Result<PreparedModel<()>, Self::Error> {
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }
    fn create_session(&self, _: PreparedModel<()>) -> Result<Session, Self::Error> {
        Ok(Session)
    }
    fn session_capability_mismatch(
        &self,
        _: SessionCapabilities,
        _: SessionCapabilities,
    ) -> Self::Error {
        WorkingMemoryError::IdentityMismatch
    }
}
impl BackendSession<Backend> for Session {
    type PrefillInput = Vec<u32>;
    type DecodeInput = Token;
    type Output = Token;
    type Completion = Done;
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }
    fn prefill(
        &mut self,
        _: &Backend,
        _: Vec<u32>,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        panic!("R preparation submits no model work")
    }
    fn decode(
        &mut self,
        _: &Backend,
        _: Token,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        panic!("R preparation submits no model work")
    }
    fn observe_output(&self, _: &Backend, _: &Token) -> Result<ObservationSet, WorkingMemoryError> {
        Ok(ObservationSet::new())
    }
}
fn memory(cause: WorkingMemoryError) -> BackendFailure {
    BackendFailure::from_error(cause)
}
impl TextGenerationBackend for Backend {
    type TextPreparation = Rc<Preparation>;
    type TextPreparationControl = ();
    type TextStepPermit = crate::working_memory::InferenceTextStep;
    type Prompt = Vec<u32>;
    type Token = Token;
    type TextGenerationState = ();
    type TextCompletion = Done;
    fn admit_text_preparation<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, Vec<u32>>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<Rc<Preparation>, BackendFailure> {
        Err(memory(WorkingMemoryError::UnknownBound))
    }
    fn admit_text_preparation_with_sequence<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, Vec<u32>>,
        config: TextGenerationConfig,
        _: &C,
        options: Option<&TextPreparationOptions>,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Rc<Preparation>, BackendFailure> {
        // Actual source take precedes all candidate estimates and admission.
        let mut decoder =
            crate::working_memory::OriginalGenerationDecoderSource::take_original(
                claim,
                &runtime.backend().0.borrow().pool,
            )?;
        let mut state = runtime.backend().0.borrow_mut();
        state.order.push("admit");
        if decoder.is_some() {
            state.decoder_takes += 1;
        }
        if state.mode.audit_decoder {
            for _ in 0..32 {
                let error =
                    crate::working_memory::OriginalGenerationDecoderSource::take_original(
                        claim,
                        &state.pool,
                    )
                    .unwrap_err();
                assert_eq!(
                    error
                        .source()
                        .unwrap()
                        .downcast_ref::<eredu_core::GenerationSequenceBankRejection>(),
                    Some(&eredu_core::GenerationSequenceBankRejection::Unavailable)
                );
            }
        }
        let input_bytes = claim
            .request()
            .token_input()
            .map(|p| {
                crate::working_memory::OriginalTokenInputLayout::prepare(p)
                    .unwrap()
                    .protected_bytes()
            })
            .unwrap_or(0);
        let g = InferenceGeometry {
            input_positions: claim
                .request()
                .token_input()
                .map(|p| p.tokens().len() as u64)
                .unwrap_or(geometry().input_positions),
            max_output_tokens: claim.request().max_new_tokens() as u64,
            ..geometry()
        };
        let g = if state.mode.input_retry {
            InferenceGeometry {
                prefill_chunk_positions: g.input_positions,
                ..g
            }
        } else {
            g
        };
        let mut original = replacement_quote(&state.pool, g, 0).into_incremental();
        if state.mode.source {
            let q = replacement_quote(&state.pool, g, 0).into_incremental();
            let (reservation, _) = sealed_plan(&state.pool, &q, 1_000_000).unwrap();
            let (reservation, run) = reservation.into_funding().unwrap();
            let scope = run.scope().unwrap();
            let source = scope
                .adopt_storage_individually([(30u32, 24)])
                .unwrap()
                .into_values()
                .next()
                .unwrap();
            original = original.with_registered_sources(source).unwrap();
            state.source_envelope = reservation.bytes();
            state.source_scope = Some(scope);
            state.source_run = Some(run);
            state.source_reservation = Some(reservation);
        }
        if state.mode.explicit_source {
            original = original
                .with_registered_sources(state.root.as_ref().unwrap().clone())
                .unwrap();
        }
        let controls = if state.mode.capture {
            let source = options
                .and_then(|o| o.capture.as_ref())
                .expect("actual capture option");
            PreparedTextControlWorkspace::prepare(
                source,
                g,
                original.span_workspace().plan(),
                facts(),
            )
            .unwrap()
            .with_generation_sequence(claim)
            .unwrap()
        } else {
            PreparedTextControlWorkspace::prepare_sequence(
                claim,
                g,
                original.span_workspace().plan(),
                facts(),
            )
            .unwrap()
        };
        assert_eq!(controls.source_identity().is_some(), state.mode.capture);
        assert!(controls.clone().with_generation_sequence(claim).is_err());
        if !state.mode.source {
            assert_eq!(state.pool.used_bytes().unwrap(), 64 + state.loaded_bytes);
        }
        if state.mode.audit {
            let foreign = replacement_quote(&state.pool, g, 0).into_incremental();
            assert!(matches!(
                foreign.with_span_workspace_and_text_controls(controls.clone()),
                Err(ResidualQuoteError::Storage(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
            let unknown = PreparedTextControlWorkspace::prepare_sequence(
                claim,
                g,
                original.span_workspace().plan(),
                TextHostControlFacts::new(None, Some(2), Some(3)),
            )
            .unwrap();
            assert!(matches!(
                original
                    .clone()
                    .with_span_workspace_and_text_controls(unknown),
                Err(ResidualQuoteError::Storage(
                    WorkingMemoryError::UnknownBound
                ))
            ));
            assert!(matches!(
                PreparedTextControlWorkspace::prepare_sequence(
                    claim,
                    g,
                    original.span_workspace().plan(),
                    TextHostControlFacts::new(None, Some(u64::MAX), Some(1))
                ),
                Err(WorkingMemoryError::Overflow)
            ));
            let changed = InferenceGeometry {
                max_output_tokens: g.max_output_tokens + 1,
                ..g
            };
            assert!(
                PreparedTextControlWorkspace::prepare_sequence(
                    claim,
                    changed,
                    original.span_workspace().plan(),
                    facts()
                )
                .is_err()
            );
        }
        let controls = if state.mode.prediction_scopes {
            controls
                .with_prediction_scopes(crate::working_memory::TextPredictionScopeFacts::new(
                    Some(101),
                    Some(103),
                    Some(107),
                    Some(109),
                    Some(113),
                ))
                .unwrap()
        } else {
            controls
        };
        let controls = if state.mode.prefill_scopes {
            controls
                .with_graph_metadata(
                    crate::working_memory::GraphMetadataFacts::new(
                        std::num::NonZeroU64::new(4096).unwrap(),
                        4096 + 29,
                    )
                    .unwrap(),
                )
                .unwrap()
                .with_prefill_scopes({
                    let facts = crate::working_memory::TextPrefillScopeFacts::new(
                        g,
                        [Some(37); 7],
                        Some(41),
                        8,
                        64,
                    )
                    .unwrap();
                    if let Some(bytes) = state.mode.host_source_component {
                        let source =
                            crate::working_memory::HostSourceConstructionFacts::new(64, 2, 0)
                                .unwrap();
                        let host = crate::working_memory::HostDestinationFacts::new(
                            bytes,
                            usize::from(bytes != 0),
                        )
                        .unwrap()
                        .with_source_constructions(source)
                        .unwrap();
                        facts
                            .with_source_constructions(Some(source))
                            .with_host_destinations(Some(host))
                    } else {
                        facts
                    }
                })
                .unwrap()
        } else {
            controls
        };
        let q = controls.facts().total_bytes().unwrap().unwrap();
        let r = controls.sequence_storage_bytes();
        let before = original.incremental_bytes();
        let quote = original
            .with_span_workspace_and_text_controls(controls)
            .unwrap();
        let p = quote.span_workspace().retention_peak_bytes().unwrap();
        assert_eq!(quote.incremental_bytes(), before + p + q + r + input_bytes);
        let exact = state.pool.used_bytes().unwrap() + quote.incremental_bytes();
        let capacity = if state.mode.short {
            exact - 1
        } else {
            exact + state.mode.copy_headroom
        };
        let execution = InferenceExecutionIdentity::default();
        let (reservation, accepted) = if state.mode.input_retry {
            assert!(!state.mode.capture && !state.mode.source && !state.mode.explicit_source);
            // A complete synthetic enclosing requirement is deliberately too
            // large only for the first chunk. The existing adaptive planner
            // chooses the next real quote; no hand-written candidate loop.
            let pool = state.pool.clone();
            plan_prefill_incremental_with_capacity(
                &execution,
                &pool,
                &capabilities(),
                request(g),
                g,
                exact + 100_000,
                |candidate| {
                    state.input_candidates += 1;
                    assert_eq!(state.input_builds, 0);
                    assert_eq!(pool.used_bytes().unwrap(), 64 + state.loaded_bytes);
                    let extra = if candidate.prefill_chunk_positions == g.prefill_chunk_positions {
                        1_000_000
                    } else {
                        0
                    };
                    let raw = replacement_quote(&pool, candidate, extra).into_incremental();
                    let controls = PreparedTextControlWorkspace::prepare_sequence(
                        claim,
                        candidate,
                        raw.span_workspace().plan(),
                        facts(),
                    )
                    .unwrap();
                    assert_eq!(controls.sequence_storage_bytes(), r);
                    Ok(raw.with_span_workspace_and_text_controls(controls).unwrap())
                },
            )
        } else {
            plan_prefill_incremental_with_capacity(
                &execution,
                &state.pool,
                &capabilities(),
                request(g),
                g,
                capacity,
                |_| Ok(quote.clone()),
            )
        }
        .map_err(BackendFailure::from_error)?;
        let p = accepted.span_workspace().retention_peak_bytes().unwrap();
        let g = accepted.geometry();
        let (reservation, run) = reservation.into_funding().unwrap();
        let (mut owner, witness) = accepted
            .into_funded_text_span_workspace(&run, &reservation)
            .unwrap();
        drop(witness);
        assert_eq!(owner.protected_host_bytes(), p + q + r + input_bytes);
        assert_eq!(
            owner.control_guard().custody.pin_source().is_ok(),
            state.mode.capture
        );
        assert_eq!(
            account(&state.pool, &reservation).1,
            p + q + r + input_bytes
        );
        let decoder_bank = if decoder.is_some() {
            let bank = if state.mode.reject_decoder_attachment {
                state
                    .pending_bank
                    .take()
                    .expect("actual foreign admitted bank")
            } else {
                owner.take_generation_sequence_bank().unwrap()
            };
            let bank = if state.mode.fail_terminal_reserve {
                bank.fail_terminal_reservation()
            } else {
                bank
            };
            let original_source = decoder.as_ref().map(std::ptr::from_ref);
            match bank.with_decoder_source(&mut decoder) {
                Ok(bank) => Some(bank),
                Err(error) => {
                    assert!(state.mode.reject_decoder_attachment);
                    assert_eq!(decoder.as_ref().map(std::ptr::from_ref), original_source);
                    assert!(
                        decoder.is_some(),
                        "foreign bank must not consume this source"
                    );
                    state.decoder_rejections_untaken += 1;
                    // The real new source remains under this new admission until
                    // explicitly retired; it cannot escape under the old bank.
                    drop(decoder);
                    return Err(error);
                }
            }
        } else {
            None
        };
        let request = InferenceRequest::from(&reservation)
            .prepare_text(&execution, g, config)
            .unwrap();
        let prediction_bank = owner.take_prediction_scopes().unwrap();
        let prefill_bank = owner.take_prefill_scopes().unwrap();
        let preparation = Rc::new(Preparation {
            request,
            owner: RefCell::new(owner),
            decoder_bank: RefCell::new(decoder_bank),
            prediction_bank: RefCell::new(prediction_bank),
            prefill_bank: RefCell::new(prefill_bank),
            run: RefCell::new(Some(run)),
            input: RefCell::new(None),
        });
        state.p = p;
        state.r = r;
        state.held = p + q + r + input_bytes;
        state.input_bytes = input_bytes;
        state.active = Some(Rc::clone(&preparation));
        Ok(preparation)
    }
    fn admit_text_preparation_with_token_input<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Vec<u32>>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&TextPreparationOptions>,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Rc<Preparation>, BackendFailure> {
        let TextPreparationInput::OriginalTokenIds(plan) = input else {
            panic!("missing original input")
        };
        assert!(std::ptr::eq(*plan, claim.request().token_input().unwrap()));
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }
    fn admit_text_preparation_with_sequence_consumer<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Vec<u32>>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&TextPreparationOptions>,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Rc<Preparation>, BackendFailure> {
        assert!(claim.request().consumer_layout().is_some());
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }
    fn admit_text_preparation_with_decoder<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Vec<u32>>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&TextPreparationOptions>,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Rc<Preparation>, BackendFailure> {
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }
    fn admit_text_preparation_with_plain_text<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        input: &TextPreparationInput<'_, Vec<u32>>,
        config: TextGenerationConfig,
        controller: &C,
        options: Option<&TextPreparationOptions>,
        claim: &GenerationSequencePreparation<'_, '_>,
    ) -> Result<Rc<Preparation>, BackendFailure> {
        Self::admit_text_preparation_with_sequence(
            runtime, input, config, controller, options, claim,
        )
    }
    fn bind_text_preparation_run<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        preparation: &Rc<Preparation>,
        _: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        runtime.backend().0.borrow_mut().order.push("bind");
        runtime.backend().0.borrow_mut().bound = Some(context.clone());
        preparation.request.bind_run(context).map_err(memory)
    }
    fn prepare_generation_sequence_admitted(
        runtime: &ModelRuntime<Self>,
        preparation: &Rc<Preparation>,
        claim: GenerationSequencePreparation<'_, '_>,
    ) -> Result<RetainedGenerationSequence, BackendFailure> {
        let (mode, target, pending) = {
            let mut state = runtime.backend().0.borrow_mut();
            state.order.push("extract");
            (
                state.mode,
                Rc::clone(state.foreign.as_ref().unwrap_or(preparation)),
                state.pending_bank.take(),
            )
        };
        if claim.request().token_input().is_some() {
            let pending = runtime.backend().0.borrow_mut().pending_input.take();
            let input_bank = if let Some(pending) = pending {
                // A foreign pending bank must not consume this request's bank.
                pending
            } else {
                let bank = target
                    .owner
                    .borrow_mut()
                    .take_token_input_bank()
                    .ok_or_else(|| {
                        eredu_core::TokenInputRejection::Unavailable.into_backend_failure()
                    })?;
                assert!(target.owner.borrow_mut().take_token_input_bank().is_none());
                bank
            };
            if mode.defer_input {
                runtime.backend().0.borrow_mut().pending_input = Some(input_bank);
                return Err(eredu_core::TokenInputRejection::Unavailable.into_backend_failure());
            }
            if mode.fence_input {
                target.run.borrow_mut().take();
            }
            let input = input_bank.construct(&preparation.request, &claim)?;
            assert_eq!(
                input.tokens(),
                claim.request().token_input().unwrap().tokens()
            );
            assert_ne!(
                input.tokens().as_ptr(),
                claim.request().token_input().unwrap().tokens().as_ptr()
            );
            let mut state = runtime.backend().0.borrow_mut();
            state.input_builds += 1;
            state.input_address = input.tokens().as_ptr() as usize;
            drop(state);
            *preparation.input.borrow_mut() = Some(input);
        }
        let busy_borrow = mode.borrow_bank.then(|| target.owner.borrow());
        let pending = pending.or_else(|| target.decoder_bank.borrow_mut().take());
        let bank = match pending {
            Some(bank) => bank,
            None => {
                let bank = {
                    let mut owner = target.owner.try_borrow_mut().map_err(|_| {
                        eredu_core::GenerationSequenceBankRejection::Busy.into_backend_failure()
                    })?;
                    owner.take_generation_sequence_bank()
                };
                bank.ok_or_else(|| {
                    eredu_core::GenerationSequenceBankRejection::Unavailable.into_backend_failure()
                })?
            }
        };
        drop(busy_borrow);
        if mode.defer_bank {
            runtime.backend().0.borrow_mut().pending_bank = Some(bank);
            return Err(
                eredu_core::GenerationSequenceBankRejection::Unavailable.into_backend_failure()
            );
        }
        if mode.panic_after_take {
            let _bank = bank;
            panic!("consumed original sequence bank unwind");
        }
        if mode.fence_bank {
            target.run.borrow_mut().take();
        }
        if mode.abandon {
            target
                .request
                .claim_generation_sequence(claim.context())
                .unwrap();
            assert!(target.request.claim_prompt().is_err());
            assert!(target.request.bind_prompt().is_err());
            assert!(
                target
                    .request
                    .claim_sampling(config(claim.request().max_new_tokens()))
                    .is_err()
            );
        }
        let context = claim.context().clone();
        let result = bank.prepare(&target.request, claim);
        if result.is_ok() {
            assert!(target.request.claim_generation_sequence(&context).is_err());
        }
        // Return the actual concrete core failure. No external retained error
        // slot, synthetic marker, native wrapper or extra Box is involved.
        result
    }
    fn reset_session(
        _: &Self,
        _: &mut Session,
        _claim: eredu_core::SessionResetClaim<'_>,
    ) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn synchronize_session(_: &Self, _: &Session) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn prepare_text_prompt(_: &Self, ids: Vec<u32>) -> Result<Vec<u32>, WorkingMemoryError> {
        Ok(ids)
    }
    fn prepare_original_text_prompt_admitted(
        backend: &Self,
        p: &Rc<Preparation>,
    ) -> Result<Vec<u32>, BackendFailure> {
        let input =
            p.input.borrow_mut().take().ok_or_else(|| {
                eredu_core::TokenInputRejection::Unavailable.into_backend_failure()
            })?;
        input.validate(&p.request).map_err(memory)?;
        p.request
            .claim_prompt()
            .map_err(memory)?
            .finish()
            .map_err(memory)?;
        {
            let mut state = backend.0.borrow_mut();
            state.order.push("original prompt");
            assert_eq!(input.tokens().as_ptr() as usize, state.input_address);
            // Test-only observation, after original construction. The neutral
            // fixture's opaque Prompt carries no native state or host copy.
            state.input_observed.extend_from_slice(input.tokens());
        }
        drop(input);
        Ok(Vec::new())
    }
    fn prepare_text_prompt_admitted(
        backend: &Self,
        ids: Vec<u32>,
        p: &Rc<Preparation>,
    ) -> Result<Vec<u32>, WorkingMemoryError> {
        backend.0.borrow_mut().order.push("prompt");
        p.request.claim_prompt()?.finish()?;
        Ok(ids)
    }
    fn bind_text_prompt_preparation(
        _: &Self,
        ids: Vec<u32>,
        p: &Rc<Preparation>,
    ) -> Result<Vec<u32>, WorkingMemoryError> {
        p.request.bind_prompt()?;
        Ok(ids)
    }
    fn start_text_generation(_: &Self, _: TextGenerationConfig) -> Result<(), WorkingMemoryError> {
        Ok(())
    }
    fn start_text_generation_admitted(
        backend: &Self,
        c: TextGenerationConfig,
        p: &Rc<Preparation>,
    ) -> Result<(), WorkingMemoryError> {
        backend.0.borrow_mut().order.push("sampling");
        p.request.claim_sampling(c)?.finish()
    }
    fn install_text_capture_admitted(
        _: &ModelRuntime<Self>,
        _: &mut (),
        p: &Rc<Preparation>,
        source: &SharedCapturePlan,
        _: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        assert_eq!(
            p.owner
                .borrow()
                .workspace()
                .text_controls()
                .unwrap()
                .source_identity(),
            Some(source.storage_identity())
        );
        Ok(())
    }
    fn agree_text_preparation(
        runtime: &ModelRuntime<Self>,
        stage: Stage,
        status: Status,
    ) -> Result<Outcome, BackendFailure> {
        runtime.backend().0.borrow_mut().votes.push((stage, status));
        Ok(match status {
            Status::Ready => Outcome::Ready,
            Status::Cancelled => Outcome::Cancelled,
            Status::Failed => Outcome::Rejected { rank: 0 },
        })
    }
    fn begin_text_step<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        preparation: &Rc<Preparation>,
        _: &(),
        _: &C,
        input: PendingTextInput<&Vec<u32>, &Token>,
        context: &TextStepContext,
    ) -> Result<Self::TextStepPermit, WorkingMemoryError> {
        assert!(
            runtime.backend().0.borrow().mode.prediction_scopes,
            "only the prediction fixture submits"
        );
        let input = match input {
            PendingTextInput::Prefill(_) => PendingTextInput::Prefill(()),
            PendingTextInput::Decode(token) => PendingTextInput::Decode(&token.receipt),
        };
        let is_prefill = matches!(input, PendingTextInput::Prefill(_));
        let step = preparation.request.claim_step(context, input)?;
        let roles = {
            let mut bank = preparation.prediction_bank.borrow_mut();
            let bank = bank.as_mut().unwrap();
            let roles = bank.claim(&step)?;
            assert!(matches!(
                bank.claim(&step),
                Err(WorkingMemoryError::TextStepOrdinalMismatch { .. })
            ));
            roles
        };
        runtime.backend().0.borrow_mut().issued.push(roles);
        if let Some(bank) = preparation.prefill_bank.borrow_mut().as_mut() {
            if is_prefill {
                let set = bank.claim(&step)?;
                assert!(matches!(
                    bank.claim(&step),
                    Err(WorkingMemoryError::AlreadyStarted)
                ));
                runtime.backend().0.borrow_mut().prefill_issued.push(set);
            } else {
                assert!(matches!(
                    bank.claim(&step),
                    Err(WorkingMemoryError::AlreadyStarted)
                ));
            }
        }
        Ok(step)
    }
    fn finish_text_step(step: Self::TextStepPermit) -> Result<(), WorkingMemoryError> {
        step.finish()
    }
    fn submit_text_prefill_permitted(
        _: &mut ModelRuntime<Self>,
        _: Vec<u32>,
        _: &eredu_core::TokenSamplingDecision<'_>,
        _: &mut (),
        _: &eredu_core::GenerationCancellationToken,
        step: &mut Self::TextStepPermit,
    ) -> Result<Option<Submission<Token, Done>>, WorkingMemoryError> {
        step.validate_input(PendingTextInput::Prefill(()))?;
        let request = step.request();
        request.begin_prefill(
            &request.memory_reservation().unwrap().0.execution,
            request.geometry(),
        )?;
        Ok(Some(Submission {
            output: Token {
                value: 7,
                receipt: step.receipt(),
            },
            completion: Done,
        }))
    }
    fn submit_text_decode_permitted(
        _: &mut ModelRuntime<Self>,
        token: Token,
        _: &eredu_core::TokenSamplingDecision<'_>,
        _: &mut (),
        step: &mut Self::TextStepPermit,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        step.validate_input(PendingTextInput::Decode(&token.receipt))?;
        Ok(Submission {
            output: Token {
                value: token.value + 1,
                receipt: step.receipt(),
            },
            completion: Done,
        })
    }
    fn submit_text_prefill(
        _: &mut ModelRuntime<Self>,
        _: Vec<u32>,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        panic!("no native submission")
    }
    fn submit_text_decode(
        _: &mut ModelRuntime<Self>,
        _: Token,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        panic!("no native submission")
    }
}
pub(super) fn config(maximum: usize) -> TextGenerationConfig {
    // User-facing resolution rejects zero. Match the core sequence fixture's
    // low-level resolved configuration to exercise its empty terminal state.
    let mut sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(maximum.max(1)),
            ..Default::default()
        },
    )
    .unwrap();
    sampling.max_new_tokens = Some(maximum);
    TextGenerationConfig::new(sampling)
}
pub(super) fn runtime(mode: Mode) -> (ModelRuntime<Backend>, Rc<RefCell<State>>) {
    runtime_with_pool(mode, WorkingMemoryPool::new(1_000_000, 0).unwrap())
}
pub(super) fn runtime_with_pool(
    mode: Mode,
    pool: WorkingMemoryPool,
) -> (ModelRuntime<Backend>, Rc<RefCell<State>>) {
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let state = Rc::new(RefCell::new(State {
        mode,
        pool,
        root: Some(root),
        active: None,
        source_scope: None,
        source_run: None,
        source_reservation: None,
        source_envelope: 0,
        pending_bank: None,
        foreign: None,
        order: vec![],
        votes: vec![],
        p: 0,
        r: 0,
        held: 0,
        decoder_takes: 0,
        loaded_bytes: 0,
        decoder_rejections_untaken: 0,
        input_bytes: 0,
        input_builds: 0,
        input_candidates: 0,
        input_address: 0,
        input_observed: Vec::new(),
        pending_input: None,
        bound: None,
        issued: Vec::new(),
        prefill_issued: Vec::new(),
    }));
    (
        ModelRuntime::prepare(Backend(Rc::clone(&state)), ()).unwrap(),
        state,
    )
}
