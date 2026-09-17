//! Fresh saved-state preparation feeding the shared ordinary text machine.

use super::*;
use crate::composition::mlx::session::{
    generation::{MlxOrdinarySampler, MlxTextSamplingState},
    model_session::text_quote::{PendingSavedTextAdmission, admit_original_saved, admit_saved},
    text_snapshot::MlxSavedTextComponents,
};
use eredu_core::{
    TextResumeBackend, TextStepContext, TokenFilterController,
    execution_control::NativeTextStateBackend,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Admitted,
    Prompt,
    Sampling,
    Installed,
    Finished,
}

/// Move-only custody for one fresh resume. The shared driver owns prompt and
/// sampling payload through readiness; this owner retains provisional/displaced
/// decoder state and the exact source until preparation succeeds or retires.
/// No public constructor or independently installable state is exposed.
pub struct MlxTextResumePreparation {
    native: Option<MlxNativeTextState>,
    key: Option<Array>,
    phase: Phase,
    // Payload-free exact session identity: retaining the actual target payload
    // would prevent the exclusive checked state exchange.
    poison: Rc<Cell<bool>>,
    armed: bool,
    // Source and preparation custody retire after provisional/displaced payload.
    admission: PendingSavedTextAdmission,
}

impl Drop for MlxTextResumePreparation {
    fn drop(&mut self) {
        if self.armed {
            // Every native copy already settled before installation. A rejected
            // readiness vote still cannot leave the replaced branch runnable.
            // This performs no native polling, completion claim or collective.
            self.poison.set(true);
        }
    }
}

impl MlxTextResumePreparation {
    fn new(runtime: &ModelRuntime<MlxBackend<'_>>, admission: PendingSavedTextAdmission) -> Self {
        Self {
            native: None,
            key: None,
            phase: Phase::Admitted,
            poison: Rc::clone(&runtime.session().poison),
            armed: false,
            admission,
        }
    }

    pub(super) fn preparation_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, eredu_core::BackendFailure>>(),
            size_of::<Result<MlxNativeTextState, Error>>(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
            size_of::<Result<MlxTextGenerationState, Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Option<crate::composition::mlx::session::model_session::control_slot::PreparedControlExchange>>(),
            size_of::<(&mut Self, &mut ModelRuntime<MlxBackend<'_>>)>() ,
            // Early resume readiness borrows the exact saved pair and invokes
            // the existing independently funded transport-source constructor.
            size_of::<Option<crate::backend::distributed::MlxTextPreparationControl>>(),
            size_of::<Result<Option<crate::backend::distributed::MlxTextPreparationControl>, Error>>(),
            size_of::<Result<Option<crate::backend::distributed::MlxTextPreparationControl>, eredu_core::BackendFailure>>(),
            size_of::<(&ModelRuntime<MlxBackend<'_>>, &MlxSavedTextComponents,
                TextGenerationConfig, &TextStepContext, &eredu_core::HostPreparationAuthority,
                eredu_core::OriginalTextResumeKind)>(),
            CopiedTextComponents::resume_origin_control_bytes()?,
            size_of::<Phase>(),
            size_of::<Rc<Cell<bool>>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    fn validate_target(
        poison: &Rc<Cell<bool>>,
        runtime: &ModelRuntime<MlxBackend<'_>>,
    ) -> Result<(), Error> {
        if !Rc::ptr_eq(poison, &runtime.session().poison) {
            return Err(mismatch());
        }
        runtime.session().validate_backend(runtime.backend())
    }

    fn require(&self, phase: Phase) -> Result<(), Error> {
        if self.phase != phase {
            return Err(mismatch());
        }
        Ok(())
    }
}

impl TextResumeBackend for MlxBackend<'_> {
    type ResumeSource = MlxSavedTextComponents;
    type ResumePreparation = MlxTextResumePreparation;

    fn prepare_text_resume_control(
        runtime: &ModelRuntime<Self>, saved: &Self::ResumeSource,
        config: TextGenerationConfig, context: &TextStepContext,
        host: &eredu_core::HostPreparationAuthority, _kind: eredu_core::OriginalTextResumeKind,
    ) -> Result<Option<Self::TextPreparationControl>, eredu_core::BackendFailure> {
        let session=runtime.session();
        let Some(transport)=session.payload.distributed.as_ref() else {return Ok(None);};
        let result=(|| {
            let source=saved.funded_source()?;
            source.validate_resume_origin_fixed(runtime)
                .map_err(|cause|memory(cause.into_memory()))?;
            if context.attempt()!=0 || !session.payload.target.has_retained_world() {
                return Err(mismatch());
            }
            runtime.backend().validate_original_stream_owners()?;
            let selected=session.payload.model.inference_blueprint().ok_or_else(unknown)?.selected();
            let manifest=selected.communication_manifest().ok_or_else(unknown)?;
            let capacity=config.inference_policy().managed_memory_capacity_bytes.ok_or_else(unknown)?;
            transport.prepare_original_readiness(manifest,transport.native_world(),
                runtime.backend().memory_pool(),session.payload.model.erased().inference_execution_identity(),
                capacity,context).map(Some)
        })();
        result.map_err(|cause|eredu_core::BackendFailure::from_error(
            super::resume_prompt::ResumeFailure::new(cause,host)))
    }

    fn admit_text_resume<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        saved: &Self::ResumeSource,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<Self::ResumePreparation, eredu_core::BackendFailure> {
        let source = saved
            .funded_source()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let admission = admit_saved(runtime, source, config, controller, context)?;
        Ok(MlxTextResumePreparation::new(runtime, admission))
    }

    fn admit_original_text_resume<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        saved: &Self::ResumeSource,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Self::ResumePreparation, eredu_core::BackendFailure> {
        Self::admit_original_text_resume_with_kind(
            runtime,
            saved,
            config,
            controller,
            context,
            host,
            eredu_core::OriginalTextResumeKind::Restore,
        )
    }

    fn admit_original_text_resume_with_kind<C: TokenFilterController>(
        runtime: &ModelRuntime<Self>,
        saved: &Self::ResumeSource,
        config: TextGenerationConfig,
        controller: &C,
        context: &TextStepContext,
        host: &eredu_core::HostPreparationAuthority,
        kind: eredu_core::OriginalTextResumeKind,
    ) -> Result<Self::ResumePreparation, eredu_core::BackendFailure> {
        let source = saved.funded_source().map_err(|cause| {
            eredu_core::BackendFailure::from_error(super::resume_prompt::ResumeFailure::new(
                cause, host,
            ))
        })?;
        let admission =
            admit_original_saved(runtime, source, config, controller, context, host, kind)?;
        Ok(MlxTextResumePreparation::new(runtime, admission))
    }

    fn prepare_text_resume_prompt(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut Self::ResumePreparation,
    ) -> Result<Self::Prompt, Error> {
        let host = prepared.admission.host_preparation().cloned();
        let result = (|| {
            prepared.require(Phase::Admitted)?;
            MlxTextResumePreparation::validate_target(&prepared.poison, runtime)?;
            let provisional =
                super::resume_prompt::construct_saved_resume_prompt(runtime, &prepared.admission)?;
            let (state, key, mut prompt) = provisional.into_parts();
            // The closed slot retains its constructor H through installation and
            // destruction of any displaced state; no bare funded Box escapes.
            prepared.native = Some(state);
            prepared.key = key;
            prompt.quote = Some(prepared.admission.quote().clone());
            text_step::validate_prompt_binding(&prompt, prepared.admission.quote())?;
            prepared
                .admission
                .preparation()
                .bind_prompt()
                .map_err(memory)?;
            prepared.phase = Phase::Prompt;
            #[cfg(all(
                test,
                target_vendor = "apple",
                feature = "metal",
                not(feature = "cuda")
            ))]
            failure_tests::checkpoint(failure_tests::Point::AfterPrompt, runtime)?;
            Ok(prompt)
        })();
        result.map_err(|cause| match &host {
            Some(host) => super::resume_prompt::ResumeFailure::retain(cause, host),
            None => cause,
        })
    }

    fn prepare_text_resume_sampling(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut Self::ResumePreparation,
    ) -> Result<Self::TextGenerationState, Error> {
        let host = prepared.admission.host_preparation().cloned();
        let result = (|| {
            prepared.require(Phase::Prompt)?;
            MlxTextResumePreparation::validate_target(&prepared.poison, runtime)?;
            prepared
                .admission
                .source()
                .validate_resume_origin(runtime)?;
            let quote = prepared.admission.quote();
            let source = &prepared.admission.source().sampling;
            if prepared.key.is_some() != source.arrays.key.is_some() {
                return Err(mismatch());
            }
            let plan = source
                .sampler
                .borrow_funded()
                .prepare_resume(quote.config())
                .map_err(memory)?;
            let (stage, original_custody) =
                quote.claim_saved_sampling(prepared.admission.preparation())?;
            let (sampler, completion) = stage
                .construct_resumed_sampler(plan, quote.sampler_scope()?)
                .map_err(memory)?;
            let mut state = MlxTextGenerationState {
                sampling: MlxTextSamplingState {
                    temperature: source.temperature,
                    prng: prepared.key.take().map(RandomState::from_key),
                    sampler: MlxOrdinarySampler::Funded(sampler),
                    next_prediction: source.next_prediction,
                    parameter_epoch: source.parameter_epoch,
                    inference_retention: Default::default(),
                    memory_retention: Default::default(),
                    quote: Some(quote.clone()),
                },
                capture: None,
                funded_capture: None,
                funding: None,
            };
            state.sampling.inference_retention.retain(quote.request());
            // The key was already settled and published by prompt construction.
            // This stage copies only protected host history; no reseeding or native
            // callback can occur between host construction and completion.
            completion.finish().map_err(memory)?;
            drop(original_custody);
            state.funding = Some(quote.take_funding_run()?);
            state.funded_capture = prepared.admission.take_capture(runtime)?;
            prepared.phase = Phase::Sampling;
            #[cfg(all(
                test,
                target_vendor = "apple",
                feature = "metal",
                not(feature = "cuda")
            ))]
            failure_tests::checkpoint(failure_tests::Point::AfterSampling, runtime)?;
            Ok(state)
        })();
        result.map_err(|cause| match &host {
            Some(host) => super::resume_prompt::ResumeFailure::retain(cause, host),
            None => cause,
        })
    }

    fn install_text_resume<C: TokenFilterController>(
        runtime: &mut ModelRuntime<Self>,
        prepared: &mut Self::ResumePreparation,
        sampling: &mut Self::TextGenerationState,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), Error> {
        let host = prepared.admission.host_preparation().cloned();
        let result = (|| {
            prepared.require(Phase::Sampling)?;
            prepared
                .admission
                .validate_before_install(runtime, controller, context)?;
            let quote = prepared.admission.quote();
            let source = &prepared.admission.source().sampling;
            if !Rc::ptr_eq(&prepared.poison, &runtime.session().poison)
                || !sampling
                    .sampling
                    .quote
                    .as_ref()
                    .is_some_and(|actual| actual.same_owner(quote))
                || !sampling.sampling.sampler.is_funded()
                || sampling.sampling.next_prediction != source.next_prediction
                || sampling.sampling.parameter_epoch != source.parameter_epoch
                || sampling.sampling.temperature != source.temperature
                || sampling.sampling.prng.is_some() != source.arrays.key.is_some()
                || sampling.capture.is_some()
                || sampling.funding.is_none()
            {
                return Err(mismatch());
            }
            let exchange = prepared.admission.take_original_exchange()?;
            let slot = prepared.native.as_mut().ok_or_else(mismatch)?;
            // Arm before the first installation mutation. Keep it armed through
            // actual post-exchange sealing and the shared Sampling readiness vote.
            prepared.armed = true;
            let transition = match exchange {
                Some(exchange) => {
                    let metadata = prepared
                        .admission
                        .planning_metadata()
                        .ok_or_else(mismatch)?;
                    let media = prepared
                        .admission
                        .source()
                        .sampling
                        .pending_media()
                        .map(|media| media.semantics().binding());
                    prepared.admission.with_parallel_control(runtime, |runtime| {
                        exchange.install(runtime, slot, metadata, media)
                    })?
                }
                None => {
                    Self::exchange_native_text_state(runtime, slot)?;
                    None
                }
            };
            #[cfg(all(
                test,
                target_vendor = "apple",
                feature = "metal",
                not(feature = "cuda")
            ))]
            failure_tests::checkpoint(failure_tests::Point::AfterExchange, runtime)?;
            prepared
                .admission
                .seal_installed(runtime, transition.as_ref())?;
            prepared.phase = Phase::Installed;
            Ok(())
        })();
        result.map_err(|cause| match &host {
            Some(host) => super::resume_prompt::ResumeFailure::retain(cause, host),
            None => cause,
        })
    }

    fn text_resume_capture_source(
        state: &Self::TextGenerationState,
    ) -> Option<&eredu_core::capture::SharedCapturePlan> {
        state
            .funded_capture
            .as_ref()
            .map(super::super::super::text_capture::InstalledCapture::source)
    }

    fn finish_text_resume(prepared: &mut Self::ResumePreparation) -> MlxTextPreparation {
        assert!(
            prepared.phase == Phase::Installed,
            "resume was not installed"
        );
        let ordinary = prepared.admission.finish();
        prepared.phase = Phase::Finished;
        prepared.armed = false;
        ordinary
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
pub(in crate::composition::mlx::session) mod failure_tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod family_tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod state_family_tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod pooling_tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod layerwise_tests;
