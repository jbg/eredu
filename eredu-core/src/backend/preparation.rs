//! One admission and native-preparation path for ordinary text machines.

use super::*;
mod sequence;
mod token_input;
pub(super) use sequence::PreparedSequence;
pub use sequence::{
    GenerationDecoderError, GenerationDecoderInput, GenerationDecoderOutput, GenerationPlainText,
    GenerationPlainTextEvent, GenerationPlainTextEvents, GenerationPlainTextProjection,
    GenerationSequenceAdmissionError, GenerationSequenceConsumerLayout,
    GenerationSequencePreparation, GenerationSequenceRequest,
};
pub use token_input::TokenIdsInputPlan;

/// Owned immutable sources considered by original text admission.
///
/// This carries no allocation allowance or execution permission. A backend must
/// price and retain supported sources under the original preparation account.
/// Even an empty shared capture plan has a physical source owner to account for.
#[derive(Debug, Default)]
pub struct TextPreparationOptions {
    /// Exact shared capture source to admit and install before machine exposure.
    pub capture: Option<crate::capture::SharedCapturePlan>,
    /// Exact immutable intervention source. Its declaration, payloads, native
    /// edits and outcome delivery must join the same original request admission.
    pub interventions: Option<crate::intervention::SharedInterventionPlan>,
}

/// Input owned by shared ordinary/controlled generation preparation.
pub enum TextGenerationInput<P> {
    /// Host token storage; native construction follows admission.
    TokenIds(Vec<u32>),
    /// Borrowed source is carried by the same original sequence request.
    OriginalTokenIds,
    /// Already prepared input whose existing storage must be accounted for.
    Prepared(P),
    /// Actual original prepared source, authenticated by the same sequence
    /// admission. This variant itself carries no allocation or work authority.
    OriginalPrepared(P),
}

/// Read-only input evidence available before native prompt or sampler creation.
pub enum TextPreparationInput<'a, P> {
    /// Complete host token payload, including spare owned capacity.
    TokenIds {
        /// Number of new prompt positions for one sequence.
        positions: u64,
        /// Full allocation capacity of the caller-owned U32 vector.
        capacity_bytes: u64,
    },
    /// Exact planned destination; the same source must be sealed in admission.
    OriginalTokenIds(&'a TokenIdsInputPlan<'a>),
    /// Existing opaque input; the backend inspects its retained metadata/storage.
    Prepared(&'a P),
    /// Borrow of the exact original prepared owner considered by this claim.
    /// Backend authentication is required; a caller may not relabel raw input.
    OriginalPrepared(&'a P),
}

impl<P> TextGenerationInput<P> {
    fn evidence<'a>(
        &'a self,
        original: Option<&'a TokenIdsInputPlan<'a>>,
    ) -> Result<TextPreparationInput<'a, P>, BackendFailure> {
        let overflow = || {
            BackendFailure::new(
                BackendFailureKind::InvalidInput,
                crate::CapabilityError::ArithmeticOverflow {
                    operation: "text preparation input capacity",
                },
            )
        };
        Ok(match self {
            Self::TokenIds(ids) => TextPreparationInput::TokenIds {
                positions: u64::try_from(ids.len()).map_err(|_| overflow())?,
                capacity_bytes: u64::try_from(ids.capacity())
                    .ok()
                    .and_then(|capacity| capacity.checked_mul(4))
                    .ok_or_else(overflow)?,
            },
            Self::OriginalTokenIds => TextPreparationInput::OriginalTokenIds(
                original
                    .ok_or_else(|| TokenInputRejection::IdentityMismatch.into_backend_failure())?,
            ),
            Self::Prepared(prompt) => TextPreparationInput::Prepared(prompt),
            Self::OriginalPrepared(prompt) => TextPreparationInput::OriginalPrepared(prompt),
        })
    }
}

type Prepared<B, C> = (
    C,
    <B as TextGenerationBackend>::Prompt,
    <B as TextGenerationBackend>::TextGenerationState,
    PreparedSequence,
    <B as TextGenerationBackend>::TextPreparation,
    Option<<B as TextGenerationBackend>::TextPreparationControl>,
);
type PreparationError<B, C> = ControlledTextGenerationError<
    <B as BackendProvider>::Error,
    <C as TokenFilterController>::Error,
>;

// Both public entry wrappers and the staged preparation worker use these exact
// representations. Naming them also permits allocation-free concrete layout
// reporting without changing their field/drop order.
pub(super) struct InputOwner<'e, B: TextGenerationBackend, C> {
    pub(super) controller: Option<C>,
    pub(super) input: Option<TextGenerationInput<B::Prompt>>,
    pub(super) sequence: Option<GenerationSequenceRequest<'e>>,
    pub(super) options: Option<TextPreparationOptions>,
}
struct Preparing<B: TextGenerationBackend, C, R> {
    controller: Option<C>,
    input: Option<TextGenerationInput<B::Prompt>>,
    prompt: Option<B::Prompt>,
    state: Option<B::TextGenerationState>,
    sequence: PreparedSequence,
    preparation: Option<B::TextPreparation>,
    control: Option<B::TextPreparationControl>,
    control_failed: bool,
    route: R,
}

pub(super) fn control_bytes<B: TextGenerationBackend, C: TokenFilterController>() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<InputOwner<'_, B, C>>(),
        size_of::<Preparing<B, C, Ordinary<'_, '_, B>>>(),
        size_of::<Ordinary<'_, '_, B>>(),
        size_of::<Prepared<B, C>>(),
        size_of::<Result<Option<Prepared<B, C>>, PreparationError<B, C>>>(),
        size_of::<PreparationError<B, C>>(),
        size_of::<Result<Option<bool>, PreparationError<B, C>>>(),
        size_of::<Result<Option<()>, PreparationError<B, C>>>(),
        size_of::<Result<B::Prompt, PreparationError<B, C>>>(),
        size_of::<Result<B::TextGenerationState, B::Error>>(),
        size_of::<TextPreparationInput<'_, B::Prompt>>(),
        size_of::<GenerationSequencePreparation<'_, '_>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

// Private adapters share readiness and ownership, not caller-provided execution
// callbacks. Ordinary preparation retains its existing immutable runtime API.
trait Route<B: TextGenerationBackend, C: TokenFilterController> {
    fn runtime(&self) -> &ModelRuntime<B>;
    fn restores_terminal(&self) -> bool { false }
    fn requires_control(&self,_input:&Option<TextGenerationInput<B::Prompt>>)->bool { false }
    fn cancellation(&self) -> Option<&crate::GenerationCancellationToken> {
        None
    }
    fn admit(
        &mut self,
        input: &Option<TextGenerationInput<B::Prompt>>,
        preparation: &mut Option<B::TextPreparation>,
        sequence: &mut PreparedSequence,
        control: &mut Option<B::TextPreparationControl>,
        control_failed: &mut bool,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure>;
    fn prompt(
        &mut self,
        input: &mut Option<TextGenerationInput<B::Prompt>>,
        preparation: &Option<B::TextPreparation>,
    ) -> Result<B::Prompt, PreparationError<B, C>>;
    fn sampling(
        &mut self,
        preparation: &Option<B::TextPreparation>,
        config: TextGenerationConfig,
    ) -> Result<B::TextGenerationState, B::Error>;
    fn install(
        &mut self,
        _state: &mut B::TextGenerationState,
        _controller: &C,
        _context: &TextStepContext,
    ) -> Result<(), B::Error> {
        Ok(())
    }
    fn has_instrumentation(&self) -> bool {
        false
    }
    fn instrumentation(
        &mut self,
        _state: &mut B::TextGenerationState,
        _preparation: &B::TextPreparation,
        _context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn finish(&mut self, _preparation: &mut Option<B::TextPreparation>) {}
}

struct Ordinary<'a, 'e, B: TextGenerationBackend> {
    runtime: &'a ModelRuntime<B>,
    options: Option<&'a TextPreparationOptions>,
    sequence: Option<&'a GenerationSequenceRequest<'e>>,
}
impl<B: TextGenerationBackend, C: TokenFilterController> Route<B, C> for Ordinary<'_, '_, B> {
    fn requires_control(&self,input:&Option<TextGenerationInput<B::Prompt>>)->bool {
        matches!(input,Some(TextGenerationInput::OriginalTokenIds|TextGenerationInput::OriginalPrepared(_)))
            && B::requires_original_preparation_control(self.runtime)
    }
    fn runtime(&self) -> &ModelRuntime<B> {
        self.runtime
    }
    fn admit(
        &mut self,
        input: &Option<TextGenerationInput<B::Prompt>>,
        preparation: &mut Option<B::TextPreparation>,
        sequence: &mut PreparedSequence,
        control: &mut Option<B::TextPreparationControl>,
        control_failed: &mut bool,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        let input = input.as_ref().expect("unprepared input");
        let original = self
            .sequence
            .and_then(GenerationSequenceRequest::token_input);
        if matches!(input, TextGenerationInput::OriginalTokenIds) != original.is_some() {
            return Err(TokenInputRejection::IdentityMismatch.into_backend_failure());
        }
        let original_prepared = matches!(input, TextGenerationInput::OriginalPrepared(_));
        if original_prepared && self.sequence.is_none() {
            return Err(PreparedRequestRejection::MissingSequence.into_backend_failure());
        }
        let evidence = input.evidence(original)?;
        if let Some(request) = self.sequence {
            if config.sampling().max_new_tokens != Some(request.max_new_tokens())
                || (request
                    .consumer_layout()
                    .is_some_and(|layout| layout.plain_text_output())
                    && !request.decoder_input().is_some_and(|input| {
                        input.output_kind() == GenerationDecoderOutput::PlainText
                    }))
            {
                if original_prepared {
                    return Err(PreparedRequestRejection::RequestMismatch.into_backend_failure());
                }
                return Err(BackendFailure::new(
                    BackendFailureKind::InvalidInput,
                    GenerationSequenceAdmissionError::RequestMismatch,
                ));
            }
            let claim = GenerationSequencePreparation::new(request, context);
            if original_prepared || request.token_input().is_some() {
                *control=match B::prepare_text_preparation_control(self.runtime,&evidence,config,&claim) {
                    Ok(source)=>source,
                    Err(error)=>{*control_failed=true;return Err(error);}
                };
                if control.is_none() && B::requires_original_preparation_control(self.runtime) {
                    *control_failed=true;
                    return Err(TokenInputRejection::Unsupported.into_backend_failure());
                }
            }
            *preparation = Some(if original_prepared {
                B::admit_text_preparation_with_original_prepared(
                    self.runtime,
                    &evidence,
                    config,
                    controller,
                    self.options,
                    &claim,
                )?
            } else if request.token_input().is_some() {
                B::admit_text_preparation_with_token_input(
                    self.runtime,
                    &evidence,
                    config,
                    controller,
                    self.options,
                    &claim,
                )?
            } else if request
                .decoder_input()
                .is_some_and(|input| input.output_kind() == GenerationDecoderOutput::PlainText)
            {
                B::admit_text_preparation_with_plain_text(
                    self.runtime,
                    &evidence,
                    config,
                    controller,
                    self.options,
                    &claim,
                )?
            } else if request.decoder_input().is_some() {
                B::admit_text_preparation_with_decoder(
                    self.runtime,
                    &evidence,
                    config,
                    controller,
                    self.options,
                    &claim,
                )?
            } else if request.consumer_layout().is_some() {
                B::admit_text_preparation_with_sequence_consumer(
                    self.runtime,
                    &evidence,
                    config,
                    controller,
                    self.options,
                    &claim,
                )?
            } else {
                B::admit_text_preparation_with_sequence(
                    self.runtime,
                    &evidence,
                    config,
                    controller,
                    self.options,
                    &claim,
                )?
            });
            let admitted = preparation.as_ref().expect("admitted sequence preparation");
            B::bind_text_preparation_run(self.runtime, admitted, controller, context)?;
            let retained = B::prepare_generation_sequence_admitted(self.runtime, admitted, claim)?;
            *sequence = PreparedSequence::install(retained, request)?;
            return Ok(());
        }
        *preparation = Some(match self.options {
            Some(options) => B::admit_text_preparation_with_options(
                self.runtime,
                &evidence,
                config,
                controller,
                options,
            )?,
            None => B::admit_text_preparation(self.runtime, &evidence, config, controller)?,
        });
        B::bind_text_preparation_run(
            self.runtime,
            preparation.as_ref().expect("admitted preparation"),
            controller,
            context,
        )
    }
    fn prompt(
        &mut self,
        input: &mut Option<TextGenerationInput<B::Prompt>>,
        preparation: &Option<B::TextPreparation>,
    ) -> Result<B::Prompt, PreparationError<B, C>> {
        let preparation = preparation.as_ref().expect("admitted preparation");
        let prompt = match input.take().expect("unprepared input") {
            TextGenerationInput::OriginalTokenIds => {
                B::prepare_original_text_prompt_admitted(self.runtime.backend(), preparation)
                    .map_err(ControlledTextGenerationError::Preparation)?
            }
            TextGenerationInput::TokenIds(ids) => {
                B::prepare_text_prompt_admitted(self.runtime.backend(), ids, preparation)
                    .map_err(ControlledTextGenerationError::Backend)?
            }
            TextGenerationInput::Prepared(prompt)
            | TextGenerationInput::OriginalPrepared(prompt) => prompt,
        };
        B::bind_text_prompt_preparation(self.runtime.backend(), prompt, preparation)
            .map_err(ControlledTextGenerationError::Backend)
    }
    fn sampling(
        &mut self,
        preparation: &Option<B::TextPreparation>,
        config: TextGenerationConfig,
    ) -> Result<B::TextGenerationState, B::Error> {
        B::start_text_generation_admitted(
            self.runtime.backend(),
            config,
            preparation.as_ref().expect("admitted preparation"),
        )
    }
    fn has_instrumentation(&self) -> bool {
        self.options.is_some()
    }

    fn instrumentation(
        &mut self,
        state: &mut B::TextGenerationState,
        preparation: &B::TextPreparation,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        if let Some(source) = self.options.and_then(|options| options.capture.as_ref()) {
            B::install_text_capture_admitted(self.runtime, state, preparation, source, context)?;
        }
        Ok(())
    }
}

struct Resume<'r, 's, B: TextResumeBackend> {
    runtime: &'r mut ModelRuntime<B>,
    source: &'s B::ResumeSource,
    cancellation: &'s crate::GenerationCancellationToken,
    prepared: Option<B::ResumePreparation>,
    host: Option<&'s HostPreparationAuthority>,
    options: &'s OriginalTextResumeOptions<'s>,
    displaced: &'s mut Option<B::DisplacedState>,
}
impl<B: TextResumeBackend, C: TokenFilterController> Route<B, C> for Resume<'_, '_, B> {
    fn restores_terminal(&self) -> bool { self.host.is_some() && self.options.terminal }
    fn requires_control(&self,_input:&Option<TextGenerationInput<B::Prompt>>)->bool {
        self.host.is_some() && B::requires_original_preparation_control(self.runtime)
    }
    fn runtime(&self) -> &ModelRuntime<B> {
        self.runtime
    }
    fn cancellation(&self) -> Option<&crate::GenerationCancellationToken> {
        Some(self.cancellation)
    }
    fn admit(
        &mut self,
        _: &Option<TextGenerationInput<B::Prompt>>,
        _: &mut Option<B::TextPreparation>,
        sequence: &mut PreparedSequence,
        control: &mut Option<B::TextPreparationControl>,
        control_failed: &mut bool,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), BackendFailure> {
        if self.options.terminal && config.sampling().max_new_tokens != Some(0) {
            return Err(crate::PreparedRequestRejection::RequestMismatch.into_backend_failure());
        }
        if let Some(host)=self.host {
            *control=match B::prepare_text_resume_control(self.runtime,self.source,config,context,host,self.options) {
                Ok(source)=>source,
                Err(error)=>{*control_failed=true;return Err(error);}
            };
            if control.is_none() && B::requires_original_preparation_control(self.runtime) {
                *control_failed=true;
                return Err(TokenInputRejection::Unsupported.into_backend_failure());
            }
        }
        self.prepared = Some(match self.host {
            Some(host) => B::admit_original_text_resume(
                self.runtime,
                self.source,
                config,
                controller,
                context,
                host,
                self.options,
            )?,
            None => B::admit_text_resume(self.runtime, self.source, config, controller, context)?,
        });
        if self.host.is_some() {
            // The separately copied retained cursor belongs to the facade.
            // None records extraction, never permission for ordinary copying.
            *sequence = PreparedSequence::Retained(None);
        }
        Ok(())
    }
    fn prompt(
        &mut self,
        _: &mut Option<TextGenerationInput<B::Prompt>>,
        _: &Option<B::TextPreparation>,
    ) -> Result<B::Prompt, PreparationError<B, C>> {
        B::prepare_text_resume_prompt(
            self.runtime,
            self.prepared.as_mut().expect("admitted resume"),
        )
        .map_err(ControlledTextGenerationError::Backend)
    }
    fn sampling(
        &mut self,
        _: &Option<B::TextPreparation>,
        _: TextGenerationConfig,
    ) -> Result<B::TextGenerationState, B::Error> {
        B::prepare_text_resume_sampling(
            self.runtime,
            self.prepared.as_mut().expect("admitted resume"),
        )
    }
    fn install(
        &mut self,
        state: &mut B::TextGenerationState,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), B::Error> {
        B::install_text_resume(
            self.runtime,
            self.prepared.as_mut().expect("admitted resume"),
            state,
            controller,
            context,
        )
    }
    fn finish(&mut self, preparation: &mut Option<B::TextPreparation>) {
        let (finished, displaced) = B::finish_text_resume(
            self.prepared.as_mut().expect("admitted resume"),
        );
        *preparation = Some(finished);
        *self.displaced = Some(displaced);
    }
}

fn cancelled<B: TextGenerationBackend, C: TokenFilterController>(route: &impl Route<B, C>) -> bool {
    route
        .cancellation()
        .is_some_and(crate::GenerationCancellationToken::is_cancelled)
}

fn agree<B: TextGenerationBackend, C: TokenFilterController, T>(
    route: &impl Route<B, C>,
    control: Option<&B::TextPreparationControl>,
    control_failed: bool,
    stage: crate::run_preparation::TextPreparationStage,
    local: Result<Option<T>, PreparationError<B, C>>,
) -> Result<Option<T>, PreparationError<B, C>> {
    if control_failed { return local; }
    if route.cancellation().is_some() {
        route.runtime().finish_text_preparation_control_cancellable(control,
            stage,
            local,
            ControlledTextGenerationError::Preparation,
        )
    } else {
        // Preserve ordinary preparation's existing peer-cancellation error.
        route
            .runtime()
            .finish_text_preparation_control(control,
                stage,
                local.map(|value| value.expect("ordinary preparation does not cancel")),
                ControlledTextGenerationError::Preparation,
            )
            .map(Some)
    }
}

fn run<B: TextGenerationBackend, C: TokenFilterController, R: Route<B, C>>(
    route: R,
    input: Option<TextGenerationInput<B::Prompt>>,
    config: TextGenerationConfig,
    controller: C,
    context: &TextStepContext,
) -> Result<Option<Prepared<B, C>>, PreparationError<B, C>> {
    use crate::run_preparation::TextPreparationStage as Stage;

    // Never pass funded payloads into agreement. This owner also covers hook
    // failures/unwinds before a machine exists. Resume staging authority stays
    // last, including during borrowed finish-only preparation extraction.
    let mut owned = Preparing::<B, C, R> {
        controller: Some(controller),
        input,
        prompt: None,
        state: None,
        sequence: PreparedSequence::Legacy,
        preparation: None,
        control: None,
        control_failed: false,
        route,
    };
    let control_required=owned.route.requires_control(&owned.input);
    let admission = (|| {
        config
            .inference_policy()
            .validate(config.sampling().max_new_tokens)
            .map_err(BackendFailure::from_error)
            .map_err(ControlledTextGenerationError::Preparation)?;
        if cancelled::<B, C>(&owned.route) {
            return Ok(None);
        }
        // A zero-output resume is Ready/no-work, not a cancellation vote. The
        // ordinary path deliberately keeps its existing zero-output behavior.
        if owned.route.cancellation().is_some() && config.sampling().max_new_tokens == Some(0)
            && !owned.route.restores_terminal() {
            return Ok(Some(false));
        }
        owned
            .route
            .admit(
                &owned.input,
                &mut owned.preparation,
                &mut owned.sequence,
                &mut owned.control,
                &mut owned.control_failed,
                config,
                owned.controller.as_ref().expect("owned controller"),
                context,
            )
            .map_err(ControlledTextGenerationError::Preparation)?;
        Ok(Some(true))
    })();
    if agree::<B, C, _>(&owned.route, owned.control.as_ref(), owned.control_failed || (control_required && owned.control.is_none()), Stage::Admission, admission)? != Some(true) {
        return Ok(None);
    }
    let prompt = (|| {
        if cancelled::<B, C>(&owned.route) {
            return Ok(None);
        }
        let prompt = owned.route.prompt(&mut owned.input, &owned.preparation)?;
        owned.prompt = Some(prompt);
        Ok((!cancelled::<B, C>(&owned.route)).then_some(()))
    })();
    if agree::<B, C, _>(&owned.route, owned.control.as_ref(), owned.control_failed || (control_required && owned.control.is_none()), Stage::Prompt, prompt)?.is_none() {
        return Ok(None);
    }
    let sampling = (|| {
        if cancelled::<B, C>(&owned.route) {
            return Ok(None);
        }
        let state = owned
            .route
            .sampling(&owned.preparation, config)
            .map_err(ControlledTextGenerationError::Backend)?;
        owned.state = Some(state);
        if cancelled::<B, C>(&owned.route) {
            return Ok(None);
        }
        owned
            .route
            .install(
                owned.state.as_mut().expect("prepared sampling state"),
                owned.controller.as_ref().expect("owned controller"),
                context,
            )
            .map_err(ControlledTextGenerationError::Backend)?;
        // Once installation begins, only its readiness outcome determines
        // publication. A late local cancellation is observed by the next step.
        Ok(Some(()))
    })();
    if agree::<B, C, _>(&owned.route, owned.control.as_ref(), owned.control_failed || (control_required && owned.control.is_none()), Stage::Sampling, sampling)?.is_none() {
        return Ok(None);
    }
    if owned.route.has_instrumentation() {
        let installed = owned
            .route
            .instrumentation(
                owned.state.as_mut().expect("prepared sampling state"),
                owned.preparation.as_ref().expect("admitted preparation"),
                context,
            )
            .map(|()| Some(()))
            .map_err(ControlledTextGenerationError::Preparation);
        if agree::<B, C, _>(&owned.route, owned.control.as_ref(), owned.control_failed || (control_required && owned.control.is_none()), Stage::Instrumentation, installed)?.is_none() {
            return Ok(None);
        }
    }
    owned.route.finish(&mut owned.preparation);
    Ok(Some((
        owned.controller.take().expect("owned controller"),
        owned.prompt.take().expect("prepared prompt"),
        owned.state.take().expect("prepared sampling state"),
        owned.sequence,
        owned.preparation.take().expect("finished preparation"),
        owned.control.take(),
    )))
}

pub(super) fn prepare<B: TextGenerationBackend, C: TokenFilterController>(
    runtime: &ModelRuntime<B>,
    input: TextGenerationInput<B::Prompt>,
    config: TextGenerationConfig,
    controller: C,
    context: &TextStepContext,
    options: Option<&TextPreparationOptions>,
    sequence: Option<&GenerationSequenceRequest<'_>>,
) -> Result<Prepared<B, C>, PreparationError<B, C>> {
    run(
        Ordinary {
            runtime,
            options,
            sequence,
        },
        Some(input),
        config,
        controller,
        context,
    )
    .map(|value| value.expect("ordinary preparation always produces a machine"))
}

pub(super) fn prepare_resume<B: TextResumeBackend, C: TokenFilterController>(
    runtime: &mut ModelRuntime<B>,
    source: &B::ResumeSource,
    config: TextGenerationConfig,
    controller: C,
    context: &TextStepContext,
    cancellation: &crate::GenerationCancellationToken,
    host: Option<&HostPreparationAuthority>,
    options: &OriginalTextResumeOptions<'_>,
) -> Result<Option<(Prepared<B, C>, B::DisplacedState)>, PreparationError<B, C>> {
    let mut displaced = None;
    let result = run(
        Resume {
            runtime,
            source,
            cancellation,
            prepared: None,
            host,
            options,
            displaced: &mut displaced,
        },
        None,
        config,
        controller,
        context,
    )?;
    Ok(result.map(|machine| (machine, displaced.expect("successful resume displaced state"))))
}

pub(super) fn resume_control_bytes<B: TextResumeBackend, C: TokenFilterController>() -> Option<usize>
{
    use std::mem::size_of;
    [
        size_of::<Preparing<B, C, Resume<'_, '_, B>>>(),
        size_of::<Resume<'_, '_, B>>(),
        size_of::<Prepared<B, C>>(),
        size_of::<Result<Option<Prepared<B, C>>, PreparationError<B, C>>>(),
        size_of::<Result<Option<bool>, PreparationError<B, C>>>(),
        size_of::<Result<Option<()>, PreparationError<B, C>>>(),
        size_of::<Result<B::Prompt, PreparationError<B, C>>>(),
        size_of::<Result<B::TextGenerationState, B::Error>>(),
        size_of::<Option<B::ResumePreparation>>(),
        size_of::<Option<B::DisplacedState>>(),
        size_of::<(Prepared<B, C>, B::DisplacedState)>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
