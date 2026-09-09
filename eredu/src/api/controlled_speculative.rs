//! Controlled speculation shares facade preparation and the ordinary scheduler.

use super::request::PreparedGenerationMode;
use super::{LoadedModel, PreparedChatSpeculativeError, PreparedChatSpeculativeGenerationRequest};
pub use eredu_core::speculative::{
    SpeculativeCaptureRole, SpeculativeControlError, SpeculativePredictionCapture,
    SpeculativeProposalView,
};
use eredu_core::{
    generation::SemanticEvent, SpeculativeGenerationBackend, SpeculativeGenerationOutput,
};
pub use eredu_runtime::speculative::{
    ControlledSpeculativeOptions, ControlledSpeculativeSession, ControlledSpeculativeStep,
    SpeculativeProposalDisposition, SpeculativeSnapshotHandle, SpeculativeVerificationRecord,
};

/// Preparation or controlled speculative execution failure, with neutral causes.
#[derive(Debug, thiserror::Error)]
pub enum ControlledSpeculativeGenerationError {
    /// Existing speculative admission, tokenizer, constraint or backend failure.
    #[error(transparent)]
    Prepared(#[from] PreparedChatSpeculativeError),
    /// Controlled advancement, snapshot or delivery failure.
    #[error(transparent)]
    Control(#[from] SpeculativeControlError),
}

impl<B: SpeculativeGenerationBackend> LoadedModel<B> {
    /// Admits bounded speculative prediction captures against this loaded model.
    /// Each raw-logit observation is one vocabulary row, including target prefill;
    /// sequence slices therefore address a single row rather than the full prompt.
    /// MLX currently accepts model.logits selections for both target and draft.
    pub fn prepare_speculative_capture(
        &self,
        settings: super::PreparedChatGenerationSettings,
        plan: eredu_core::capture::CapturePlan,
    ) -> Result<eredu_core::capture::AdmittedCapturePlan, super::PreparedChatError> {
        let (_, maximum) = self.resolve_text_generation_settings(settings)?;
        let (plan, _) = self.admit_capture(
            plan,
            eredu_core::capture::CaptureRequestShape {
                batch: 1,
                prompt_tokens: 1,
                max_predictions: maximum.get() as u64,
            },
        )?;
        B::validate_speculative_capture(&self.runtime, &plan)?;
        Ok(plan)
    }

    /// Lends a controlled speculative chat to an Inspector or worker command loop.
    /// Uses the same drafting, acceptance, stop, decoding and cancellation path as
    /// `generate_prepared_chat_speculative`. No token is generated before `step`.
    /// Pausing means returning to the command loop without calling `step`.
    /// Native resources stay borrowed for the closure; unfinished work is cancelled
    /// and safely settled when it returns. The returned output includes active TTFT.
    pub fn with_controlled_chat_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<'a, B, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        self.with_controlled_speculative(request, options, drive, PreparedGenerationMode::Semantic)
    }

    /// Literal-text equivalent, including templates without semantic recognition.
    ///
    /// ```no_run
    /// # use eredu::api::*;
    /// # use eredu::runtime::chat::PreparedChat;
    /// # use eredu_core::{SpeculativeGenerationBackend, SpeculativeDraft, SpeculativeGenerationOutput};
    /// # fn inspect<B: SpeculativeGenerationBackend>(model: &mut LoadedModel<B>, chat: &PreparedChat,
    /// #     drafting: SpeculativeDraft<'_, B::Drafter>) -> Result<SpeculativeGenerationOutput, Box<dyn std::error::Error>> {
    /// let output = model.with_controlled_text_speculative(
    ///     PreparedChatSpeculativeGenerationRequest {
    ///         input: PreparedChatInput::rendered_prompt(chat), drafting,
    ///         settings: Default::default(), options: Default::default(),
    ///         caller_stop_sequences: &[], cancellation: Default::default(),
    ///         on_event: |event| println!("committed: {event:?}"),
    ///     },
    ///     ControlledSpeculativeOptions::default(),
    ///     |session| {
    ///         // A worker may wait for the next Inspector command before each step.
    ///         while let Some(step) = session.step()? {
    ///             println!("draft: {:?}, verification: {:?}", step.drafted, step.verification);
    ///         }
    ///         Ok(())
    ///     },
    /// )?;
    /// # Ok(output)
    /// # }
    /// ```
    pub fn with_controlled_text_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<'a, B, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        self.with_controlled_speculative(request, options, drive, PreparedGenerationMode::Text)
    }

    fn with_controlled_speculative<'a, F, D>(
        &mut self,
        request: PreparedChatSpeculativeGenerationRequest<'a, B, B::Drafter, F>,
        options: ControlledSpeculativeOptions,
        drive: D,
        mode: PreparedGenerationMode,
    ) -> Result<SpeculativeGenerationOutput, ControlledSpeculativeGenerationError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut dyn ControlledSpeculativeSession) -> Result<(), SpeculativeControlError>,
    {
        let capture = options.capture.clone();
        let mut failure = None;
        let driver = eredu_runtime::speculative::DriveControlledSpeculation::new(
            request.options.scheduler,
            options,
            drive,
            &mut failure,
        );
        if let Some(plan) = capture.as_ref() {
            if plan.request().prompt_tokens != 1 || plan.request().batch != 1 {
                return Err(SpeculativeControlError::Capture(eredu_core::capture::CaptureError::Invalid(
                    "speculative captures require one-row admission from prepare_speculative_capture".into())).into());
            }
            B::validate_speculative_capture(&self.runtime, plan)
                .map_err(SpeculativeControlError::from)?;
            let (_, maximum) = self
                .resolve_text_generation_settings(request.settings)
                .map_err(PreparedChatSpeculativeError::from)?;
            if plan.request().max_predictions < maximum.get() as u64 {
                return Err(SpeculativeControlError::Capture(
                    eredu_core::capture::CaptureError::Invalid(
                        "capture admission does not cover the requested speculative token limit"
                            .into(),
                    ),
                )
                .into());
            }
        }
        let result = self.generate_prepared_speculative_with(request, mode, driver);
        if let Some(error) = failure {
            return Err(error.into());
        }
        result.map_err(Into::into)
    }
}
