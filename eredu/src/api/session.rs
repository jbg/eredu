//! Generation over a retained target and optional drafting resources.

use super::request::SourceRequest;
use super::{
    GenerationForecast, GenerationForecastBackend, GenerationForecastError,
    GenerationForecastOptions, GenerationRequest, LoadedModel, PreparedChatError,
    PreparedChatGenerationSettings, SpeculativeForecastBackend,
};
use eredu_core::decoding::{DecodingMode, SpeculationOptions};
use eredu_core::{
    DraftingPlan, GenerationOutput, RealizedDrafting, SemanticEvent, SpeculativeDraft,
    SpeculativeGenerationBackend, SpeculativeStats, TextGenerationBackend,
};
use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

/// Diagnostics for the mechanisms used by one generation.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Present when a retained speculative executor ran, including an early cancellation.
    pub drafting: Option<SpeculativeStats>,
}

/// A target-only source, requiring no speculative backend implementation.
pub struct NoDrafting;

/// A generation owner retaining its target, source and selected execution policy.
///
/// Factories return an owned session. `LoadedModel::session` and
/// `LoadedModel::drafting_session` lend the same interface for explicitly assembled
/// models. The request never borrows drafting resources.
pub struct PlannedModel<B: TextGenerationBackend, S = NoDrafting, M = LoadedModel<B>> {
    model: M,
    drafting: S,
    drafting_plan: Option<DraftingPlan>,
    options: Option<SpeculationOptions>,
    backend: PhantomData<fn() -> B>,
}

impl<B: TextGenerationBackend, D> PlannedModel<B, RealizedDrafting<D>> {
    pub(crate) fn new(
        model: LoadedModel<B>,
        drafting: RealizedDrafting<D>,
        plan: DraftingPlan,
    ) -> Self {
        Self {
            model,
            drafting,
            drafting_plan: Some(plan),
            options: None,
            backend: PhantomData,
        }
    }
}

impl<B: TextGenerationBackend, S, M: Borrow<LoadedModel<B>>> PlannedModel<B, S, M> {
    /// Loaded target and its metadata, capability and preparation operations.
    pub fn model(&self) -> &LoadedModel<B> {
        self.model.borrow()
    }
    /// Retained source resources.
    pub fn drafting(&self) -> &S {
        &self.drafting
    }
    /// Load-time plan, absent for an explicitly assembled borrowed session.
    pub fn drafting_plan(&self) -> Option<&DraftingPlan> {
        self.drafting_plan.as_ref()
    }

    /// Default drafting controls retained by this session.
    pub fn drafting_options(
        &self,
    ) -> Result<Option<SpeculationOptions>, eredu_core::GenerationError> {
        let Some(plan) = &self.drafting_plan else {
            return Ok(self.options);
        };
        let Some(maximum) = plan.max_draft_tokens() else {
            return Ok(None);
        };
        let maximum = std::num::NonZeroUsize::new(maximum)
            .ok_or(eredu_core::GenerationError::ZeroDraftTokens)?;
        let mut scheduler =
            eredu_core::SpeculativeSchedulerOptions::default().with_lookahead(plan.lookahead());
        scheduler.adaptive_lookahead = plan.adaptive_lookahead();
        Ok(Some(
            eredu_core::decoding::SpeculationOptionsRequest {
                max_draft_tokens: maximum,
                scheduler,
                ..Default::default()
            }
            .try_into()?,
        ))
    }

    fn resolve_settings(
        &self,
        mut settings: PreparedChatGenerationSettings,
    ) -> Result<PreparedChatGenerationSettings, eredu_core::GenerationError> {
        let options = self.drafting_options()?;
        if settings.decoding_mode.is_none() {
            settings.decoding_mode = options.map(DecodingMode::SelfSpeculation);
        }
        if let Some(DecodingMode::SelfSpeculation(requested)) = settings.decoding_mode {
            if let Some(ceiling) = options {
                eredu_core::decoding::SpeculationLimits::from(ceiling).validate(requested)?;
            } else if self.drafting_plan.is_some() {
                return Err(eredu_core::decoding::DecodingModeError::Unsupported(
                    "the selected plan has no drafting resources",
                )
                .into());
            }
        }
        Ok(settings)
    }
}

impl<B: TextGenerationBackend, S, M: BorrowMut<LoadedModel<B>>> PlannedModel<B, S, M> {
    /// Mutably borrows the target for preparation and lifecycle operations.
    pub fn model_mut(&mut self) -> &mut LoadedModel<B> {
        self.model.borrow_mut()
    }
    /// Borrows target and source for resource inspection.
    pub fn parts_mut(&mut self) -> (&mut LoadedModel<B>, &mut S) {
        (self.model.borrow_mut(), &mut self.drafting)
    }
    /// Releases drafting resources and keeps a target-only owner. This projection
    /// requires no speculative backend implementation, including for factory loads.
    pub fn into_target_only(self) -> PlannedModel<B, NoDrafting, M> {
        PlannedModel {
            model: self.model,
            drafting: NoDrafting,
            drafting_plan: Some(DraftingPlan::Disabled),
            options: None,
            backend: PhantomData,
        }
    }

    /// Releases the generation owner into its target and source.
    pub fn into_parts(self) -> (M, S) {
        (self.model, self.drafting)
    }
}

// Sources are sealed by visibility. Concrete source types determine which neutral
// backend contracts are required; target-only sessions never require speculation.
#[allow(private_bounds)]
impl<B: TextGenerationBackend, S: Source<B>, M: BorrowMut<LoadedModel<B>>> PlannedModel<B, S, M> {
    /// Runs one request with the retained source and output policy.
    /// Only canonical committed events enter the callback. Reset the target before
    /// an independent request; cancellation preserves the committed prefix.
    pub fn generate<F: FnMut(SemanticEvent)>(
        &mut self,
        mut request: GenerationRequest<'_, B, F>,
    ) -> Result<GenerationOutput<GenerationStats>, PreparedChatError> {
        let started = std::time::Instant::now();
        request.link_cancellation();
        request.settings = match self.resolve_settings(request.settings) {
            Ok(settings) => settings,
            Err(error) => {
                return self.model.borrow_mut().runtime.finish_text_preparation(
                    eredu_core::run_preparation::TextPreparationStage::Request,
                    Err(PreparedChatError::Generation(error)),
                    PreparedChatError::Backend,
                )
            }
        };
        let limits = self.drafting_options()?;
        self.drafting
            .generate(self.model.borrow_mut(), request, limits, started)
    }
}

#[allow(private_bounds)]
impl<B: GenerationForecastBackend, S: ForecastSource<B>, M: BorrowMut<LoadedModel<B>>>
    PlannedModel<B, S, M>
{
    /// Forecasts the same resolved source and request as `generate`, without
    /// consuming input, invoking callbacks or advancing execution.
    pub fn forecast_generation<F>(
        &mut self,
        request: &GenerationRequest<'_, B, F>,
        options: &GenerationForecastOptions,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        let settings = self.resolve_settings(request.settings)?;
        let limits = self.drafting_options()?;
        self.drafting
            .forecast(self.model.borrow(), request, settings, options, limits)
    }
}

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Lends a target-only generation owner with no speculative backend bound.
    pub fn session(&mut self) -> PlannedModel<B, NoDrafting, &mut Self> {
        PlannedModel {
            model: self,
            drafting: NoDrafting,
            drafting_plan: None,
            options: None,
            backend: PhantomData,
        }
    }
    /// Owns a target-only generation session.
    pub fn into_session(self) -> PlannedModel<B> {
        PlannedModel {
            model: self,
            drafting: NoDrafting,
            drafting_plan: None,
            options: None,
            backend: PhantomData,
        }
    }
    /// Lends an explicitly paired source and its default speculative policy.
    /// `None` starts the source in causal mode unless the request selects another
    /// mode. Explicit options supply speculative defaults and resource ceilings.
    /// Native pairing validates compatibility.
    pub fn drafting_session<'a, D>(
        &'a mut self,
        drafting: SpeculativeDraft<'a, D>,
        options: Option<SpeculationOptions>,
    ) -> PlannedModel<B, SpeculativeDraft<'a, D>, &'a mut Self> {
        PlannedModel {
            model: self,
            drafting,
            drafting_plan: None,
            options,
            backend: PhantomData,
        }
    }
}

fn speculative_output(
    output: eredu_core::SpeculativeGenerationOutput,
) -> GenerationOutput<GenerationStats> {
    output.map_stats(|stats| GenerationStats {
        drafting: Some(stats),
    })
}

fn with_source<'a, B: TextGenerationBackend, D, F>(
    request: GenerationRequest<'a, B, F>,
    drafting: SpeculativeDraft<'a, D>,
    limits: Option<SpeculationOptions>,
    started: std::time::Instant,
) -> SourceRequest<'a, B, D, F> {
    SourceRequest {
        request,
        drafting,
        limits,
        started,
    }
}

trait Source<B: TextGenerationBackend> {
    fn generate<F: FnMut(SemanticEvent)>(
        &mut self,
        model: &mut LoadedModel<B>,
        request: GenerationRequest<'_, B, F>,
        limits: Option<SpeculationOptions>,
        started: std::time::Instant,
    ) -> Result<GenerationOutput<GenerationStats>, PreparedChatError>;
}
impl<B: TextGenerationBackend> Source<B> for NoDrafting {
    fn generate<F: FnMut(SemanticEvent)>(
        &mut self,
        model: &mut LoadedModel<B>,
        request: GenerationRequest<'_, B, F>,
        _limits: Option<SpeculationOptions>,
        started: std::time::Instant,
    ) -> Result<GenerationOutput<GenerationStats>, PreparedChatError> {
        model.generate_request(request, started)
    }
}
impl<B: SpeculativeGenerationBackend> Source<B> for SpeculativeDraft<'_, B::Drafter> {
    fn generate<F: FnMut(SemanticEvent)>(
        &mut self,
        model: &mut LoadedModel<B>,
        request: GenerationRequest<'_, B, F>,
        limits: Option<SpeculationOptions>,
        started: std::time::Instant,
    ) -> Result<GenerationOutput<GenerationStats>, PreparedChatError> {
        model
            .generate_speculative_request(with_source(request, self.reborrow(), limits, started))
            .map(speculative_output)
    }
}
impl<B: SpeculativeGenerationBackend> Source<B> for RealizedDrafting<B::Drafter> {
    fn generate<F: FnMut(SemanticEvent)>(
        &mut self,
        model: &mut LoadedModel<B>,
        request: GenerationRequest<'_, B, F>,
        limits: Option<SpeculationOptions>,
        started: std::time::Instant,
    ) -> Result<GenerationOutput<GenerationStats>, PreparedChatError> {
        match self.as_speculative_draft() {
            Some(drafting) => model
                .generate_speculative_request(with_source(request, drafting, limits, started))
                .map(speculative_output),
            None => model.generate_request(request, started),
        }
    }
}

trait ForecastSource<B: GenerationForecastBackend> {
    fn forecast<F>(
        &mut self,
        model: &LoadedModel<B>,
        request: &GenerationRequest<'_, B, F>,
        settings: PreparedChatGenerationSettings,
        options: &GenerationForecastOptions,
        limits: Option<SpeculationOptions>,
    ) -> Result<GenerationForecast, GenerationForecastError>;
}
impl<B: GenerationForecastBackend> ForecastSource<B> for NoDrafting {
    fn forecast<F>(
        &mut self,
        model: &LoadedModel<B>,
        request: &GenerationRequest<'_, B, F>,
        settings: PreparedChatGenerationSettings,
        options: &GenerationForecastOptions,
        _limits: Option<SpeculationOptions>,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        model.forecast_input(&request.input, settings, options, false)
    }
}
impl<B, D> ForecastSource<B> for SpeculativeDraft<'_, D>
where
    B: GenerationForecastBackend + SpeculativeForecastBackend<D>,
{
    fn forecast<F>(
        &mut self,
        model: &LoadedModel<B>,
        request: &GenerationRequest<'_, B, F>,
        settings: PreparedChatGenerationSettings,
        options: &GenerationForecastOptions,
        limits: Option<SpeculationOptions>,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        let mode = settings.decoding_mode.unwrap_or_default();
        let mut target_settings = settings;
        target_settings.decoding_mode = None;
        let mut forecast = model.forecast_input(&request.input, target_settings, options, false)?;
        model.project_speculation(&mut forecast, self, mode, options, limits)?;
        Ok(forecast)
    }
}
impl<B, D> ForecastSource<B> for RealizedDrafting<D>
where
    B: GenerationForecastBackend + SpeculativeForecastBackend<D>,
{
    fn forecast<F>(
        &mut self,
        model: &LoadedModel<B>,
        request: &GenerationRequest<'_, B, F>,
        settings: PreparedChatGenerationSettings,
        options: &GenerationForecastOptions,
        limits: Option<SpeculationOptions>,
    ) -> Result<GenerationForecast, GenerationForecastError> {
        match self.as_speculative_draft() {
            Some(mut draft) => draft.forecast(model, request, settings, options, limits),
            None => model.forecast_input(&request.input, settings, options, false),
        }
    }
}

/// Bounded inspection controls for either generation mechanism.
#[derive(Debug, Clone)]
pub struct GenerationControlOptions {
    /// Ordinary forward capture, or a plan to admit for speculative prediction rows.
    pub capture: eredu_core::capture::CapturePlan,
    /// Record transport bounds shared by ordinary and speculative delivery.
    pub trace_limits: super::TraceLimits,
    /// Already admitted target/draft prediction capture, excluding `capture` above.
    pub speculative_capture: Option<eredu_core::capture::AdmittedCapturePlan>,
    /// Phase-aware internal forward capture and interventions.
    pub speculative_activations: Option<eredu_core::speculative::AdmittedSpeculativeActivations>,
    /// Speculative snapshot retention and copy bounds. Ordinary snapshots use
    /// the typed controller's `enable_snapshots` after startup.
    pub speculative_snapshots: Option<eredu_core::execution_control::SnapshotLimits>,
}
impl Default for GenerationControlOptions {
    fn default() -> Self {
        Self {
            capture: eredu_core::capture::CapturePlan::none(),
            trace_limits: super::TraceLimits {
                per_record_bytes: 1 << 20,
                total_bytes: 64 << 20,
            },
            speculative_capture: None,
            speculative_activations: None,
            speculative_snapshots: None,
        }
    }
}

/// Generation preparation or explicit advancement failed.
#[derive(Debug, thiserror::Error)]
pub enum SessionControlError {
    /// Shared request admission failed.
    #[error(transparent)]
    Generation(#[from] PreparedChatError),
    /// Ordinary controlled execution or observation failed.
    #[error(transparent)]
    Ordinary(#[from] super::ControlledGenerationError),
    /// Speculative controlled execution or observation failed.
    #[error(transparent)]
    Speculative(#[from] eredu_core::speculative::SpeculativeControlError),
}
impl From<eredu_core::decoding::DecodingModeError> for SessionControlError {
    fn from(error: eredu_core::decoding::DecodingModeError) -> Self {
        Self::Generation(error.into())
    }
}
impl From<super::ControlledSpeculativeGenerationError> for SessionControlError {
    fn from(error: super::ControlledSpeculativeGenerationError) -> Self {
        match error {
            super::ControlledSpeculativeGenerationError::Prepared(error) => Self::Generation(error),
            super::ControlledSpeculativeGenerationError::Control(error) => Self::Speculative(error),
        }
    }
}

/// One bounded physical advancement, retaining its mechanism-specific evidence.
#[derive(Debug)]
pub enum GenerationStep {
    /// Ordinary physical progress and its already charged records.
    Ordinary(Vec<super::ControlledGenerationRecord>),
    /// Proposal, verification or commitment action from the shared scheduler.
    Speculative(super::ControlledSpeculativeStep),
}

/// Scoped advancement over the same drivers used by uninterrupted generation.
/// Common operations do not require selecting an execution mechanism. Inspectors
/// can borrow the typed controller for its specialized snapshots and interventions.
pub enum GenerationControl<'a, 'm, B: TextGenerationBackend> {
    /// Ordinary target driver and synchronous committed-event delivery.
    Ordinary {
        /// Native state stays borrowed for this scope.
        session: &'a mut super::ControlledGenerationSession<'m, B>,
        /// Shared request callback, excluding tentative output.
        on_event: &'a mut dyn FnMut(SemanticEvent),
    },
    /// Retained speculative source and its existing scheduler.
    Speculative(&'a mut dyn super::ControlledSpeculativeSession),
}
impl<'a, 'm, B: TextGenerationBackend> GenerationControl<'a, 'm, B> {
    /// Canonical committed output, excluding draft proposals.
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Ordinary { session, .. } => session.token_ids(),
            Self::Speculative(session) => session.token_ids(),
        }
    }
    /// Timing under the common active-time contract.
    pub fn timing(&self) -> eredu_core::GenerationTiming {
        match self {
            Self::Ordinary { session, .. } => session.timing(),
            Self::Speculative(session) => session.timing(),
        }
    }
    /// Performs one physical advancement, publishing canonical events through the request callback.
    pub fn step(&mut self) -> Result<Option<GenerationStep>, SessionControlError> {
        match self {
            Self::Speculative(session) => Ok(session.step()?.map(GenerationStep::Speculative)),
            Self::Ordinary { session, on_event } => {
                if session.finish_reason().is_some() {
                    return Ok(None);
                }
                let mut records = Vec::new();
                session.step(|record| {
                    if let super::ObservedGenerationEvent::Semantic { event, .. } =
                        &record.generation.event
                    {
                        on_event(event.clone());
                    }
                    records.push(record);
                    std::ops::ControlFlow::Continue(())
                })?;
                Ok(Some(GenerationStep::Ordinary(records)))
            }
        }
    }
    /// Advances to termination. A caller pauses by returning to its command loop between steps.
    pub fn run(&mut self) -> Result<(), SessionControlError> {
        loop {
            if let Self::Ordinary { session, .. } = self {
                let control = session.control_handle();
                if control.pause_requested() && !session.cancellation_requested() {
                    break;
                }
            }
            if self.step()?.is_none() {
                break;
            }
        }
        Ok(())
    }
    /// Borrows the ordinary controller for typed snapshots and interventions.
    pub fn ordinary(&mut self) -> Option<&mut super::ControlledGenerationSession<'m, B>> {
        match self {
            Self::Ordinary { session, .. } => Some(*session),
            _ => None,
        }
    }
    /// Borrows the proposal-aware inspector when this session retains a speculative executor.
    pub fn speculative(&mut self) -> Option<&mut dyn super::ControlledSpeculativeSession> {
        match self {
            Self::Speculative(session) => Some(*session),
            _ => None,
        }
    }
}

#[allow(private_bounds)]
impl<B: TextGenerationBackend, S: ControlSource<B>, M: BorrowMut<LoadedModel<B>>>
    PlannedModel<B, S, M>
{
    /// Lends explicit advancement with the same request, defaults and result as `generate`.
    /// Dropping an unfinished scope cancels and settles its existing driver.
    pub fn with_controlled<F, D>(
        &mut self,
        mut request: GenerationRequest<'_, B, F>,
        options: GenerationControlOptions,
        drive: D,
    ) -> Result<GenerationOutput<GenerationStats>, SessionControlError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut GenerationControl<'_, '_, B>) -> Result<(), SessionControlError>,
    {
        let started = std::time::Instant::now();
        request.link_cancellation();
        request.settings = match self.resolve_settings(request.settings) {
            Ok(settings) => settings,
            Err(error) => {
                return self.model.borrow_mut().runtime.finish_text_preparation(
                    self.drafting.admission_stage(),
                    Err(SessionControlError::Generation(error.into())),
                    |error| SessionControlError::Generation(PreparedChatError::Backend(error)),
                )
            }
        };
        let limits = self
            .drafting_options()
            .map_err(PreparedChatError::Generation)?;
        self.drafting.controlled(
            self.model.borrow_mut(),
            request,
            options,
            limits,
            started,
            drive,
        )
    }
    /// Runs explicit advancement continuously, delivering bounded records on demand.
    /// Breaking delivery cancels the same driver and closes further step delivery.
    pub fn generate_observed<F, D>(
        &mut self,
        request: GenerationRequest<'_, B, F>,
        options: GenerationControlOptions,
        mut on_step: D,
    ) -> Result<GenerationOutput<GenerationStats>, SessionControlError>
    where
        F: FnMut(SemanticEvent),
        D: FnMut(GenerationStep) -> std::ops::ControlFlow<()>,
    {
        self.with_controlled(request, options, |session| {
            while let Some(step) = session.step()? {
                if on_step(step).is_break() {
                    match session {
                        GenerationControl::Speculative(session) => session.cancel()?,
                        GenerationControl::Ordinary { session, on_event } => {
                            session.cancel(|record| {
                                if let super::ObservedGenerationEvent::Semantic { event, .. } =
                                    record.generation.event
                                {
                                    on_event(event);
                                }
                                std::ops::ControlFlow::Continue(())
                            })?;
                        }
                    }
                    break;
                }
            }
            Ok(())
        })
    }
}

trait ControlSource<B: TextGenerationBackend> {
    fn admission_stage(&self) -> eredu_core::run_preparation::TextPreparationStage {
        eredu_core::run_preparation::TextPreparationStage::Instrumentation
    }
    fn controlled<F, D>(
        &mut self,
        model: &mut LoadedModel<B>,
        request: GenerationRequest<'_, B, F>,
        options: GenerationControlOptions,
        limits: Option<SpeculationOptions>,
        started: std::time::Instant,
        drive: D,
    ) -> Result<GenerationOutput<GenerationStats>, SessionControlError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut GenerationControl<'_, '_, B>) -> Result<(), SessionControlError>;
}
impl<B: eredu_core::ModelCapabilityBackend> ControlSource<B> for NoDrafting {
    fn admission_stage(&self) -> eredu_core::run_preparation::TextPreparationStage {
        eredu_core::run_preparation::TextPreparationStage::Request
    }
    fn controlled<F, D>(
        &mut self,
        model: &mut LoadedModel<B>,
        request: GenerationRequest<'_, B, F>,
        options: GenerationControlOptions,
        _limits: Option<SpeculationOptions>,
        started: std::time::Instant,
        drive: D,
    ) -> Result<GenerationOutput<GenerationStats>, SessionControlError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut GenerationControl<'_, '_, B>) -> Result<(), SessionControlError>,
    {
        use super::PreparedChatInput;
        let prepared = (|| -> Result<_, PreparedChatError> {
            if request.cancellation.is_cancelled() {
                return Ok(None);
            }
            if options.speculative_capture.is_some()
                || options.speculative_activations.is_some()
                || options.speculative_snapshots.is_some()
            {
                return Err(eredu_core::decoding::DecodingModeError::Unsupported(
                    "draft-phase inspection requires a speculative executor",
                )
                .into());
            }
            request
                .input
                .validate_session(&model.session_identity)
                .map_err(PreparedChatError::Backend)?;
            let chat = request.input.prepared_chat().clone();
            let input = request.input.into_unbound();
            let prepared = match input {
                PreparedChatInput::RenderedPrompt(_) => model.prepare_observed_chat(
                    &chat,
                    request.settings,
                    options.capture,
                    options.trace_limits,
                )?,
                PreparedChatInput::TokenIds { token_ids, .. } => model.prepare_observed_token_ids(
                    &chat,
                    token_ids,
                    request.settings,
                    options.capture,
                    options.trace_limits,
                )?,
                PreparedChatInput::PreparedBackendInput { prompt, .. } => {
                    let count = model.count_prepared_input(&prompt)?;
                    let ids = B::prepared_input_token_ids(&model.runtime, &prompt)
                        .map_err(PreparedChatError::Backend)?;
                    if u64::try_from(ids.len()).ok() != Some(count.model_positions) {
                        return Err(PreparedChatError::Alignment {
                            expected: count.model_positions,
                            actual: ids.len(),
                        });
                    }
                    model.prepare_observed_input(
                        &chat,
                        prompt,
                        ids,
                        request.settings,
                        options.capture,
                        options.trace_limits,
                    )?
                }
                PreparedChatInput::SessionBound { .. } => {
                    unreachable!("validated input was unbound")
                }
            };
            Ok(Some(prepared))
        })();
        let prepared = model.runtime.finish_text_preparation_cancellable(
            eredu_core::run_preparation::TextPreparationStage::Request,
            prepared,
            PreparedChatError::Backend,
        )?;
        let mut on_event = request.on_event;
        let Some(mut prepared) = prepared else {
            on_event(SemanticEvent::Finished {
                reason: eredu_core::FinishReason::Cancelled,
            });
            return Ok(GenerationOutput::new(
                Vec::new(),
                eredu_core::FinishReason::Cancelled,
                eredu_core::GenerationTiming::default(),
                GenerationStats::default(),
            ));
        };
        let control = request.control_handle.cloned().unwrap_or_else(|| {
            eredu_core::execution_control::GenerationControlHandle::new(
                request.cancellation.clone(),
            )
        });
        prepared.control_handle = Some(control.clone());
        prepared.cancellation = Some(request.cancellation.clone());
        let emit = |_: super::ControlledGenerationRecord| std::ops::ControlFlow::Continue(());
        prepared.preparation_elapsed = started.elapsed();
        let mut session = model.start_controlled(
            prepared,
            request.caller_stop_sequences,
            control,
            emit,
            request.output_mode,
        )?;
        drive(&mut GenerationControl::Ordinary {
            session: &mut session,
            on_event: &mut on_event,
        })?;
        if session.finish_reason().is_none() {
            session.cancel(|record| {
                if let super::ObservedGenerationEvent::Semantic { event, .. } =
                    record.generation.event
                {
                    on_event(event);
                }
                std::ops::ControlFlow::Continue(())
            })?;
        }
        Ok(GenerationOutput::new(
            session.token_ids().to_vec(),
            session.finish_reason().expect("settled controlled request"),
            session.timing(),
            GenerationStats::default(),
        ))
    }
}
impl<B> ControlSource<B> for SpeculativeDraft<'_, B::Drafter>
where
    B: SpeculativeGenerationBackend + SpeculativeForecastBackend<B::Drafter>,
{
    fn controlled<F, D>(
        &mut self,
        model: &mut LoadedModel<B>,
        request: GenerationRequest<'_, B, F>,
        options: GenerationControlOptions,
        limits: Option<SpeculationOptions>,
        started: std::time::Instant,
        drive: D,
    ) -> Result<GenerationOutput<GenerationStats>, SessionControlError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut GenerationControl<'_, '_, B>) -> Result<(), SessionControlError>,
    {
        let mode = request.output_mode;
        let mut options = options;
        let admission = (|| -> Result<(), PreparedChatError> {
            if !options.capture.selections.is_empty() {
                if options.speculative_capture.is_some() {
                    return Err(PreparedChatError::Capture(eredu_core::capture::CaptureError::Invalid(
                        "capture is supplied both as a plan and as an admitted speculative selection".into())));
                }
                options.speculative_capture =
                    Some(model.prepare_speculative_capture(request.settings, options.capture)?);
            }
            Ok(())
        })();
        if let Err(error) = admission {
            return model.runtime.finish_text_preparation(
                eredu_core::run_preparation::TextPreparationStage::Instrumentation,
                Err(SessionControlError::Generation(error)),
                |error| SessionControlError::Generation(PreparedChatError::Backend(error)),
            );
        }
        let mut caller_failure = None;
        let result = model.with_controlled_speculative(
            with_source(request, self.reborrow(), limits, started),
            eredu_runtime::speculative::ControlledSpeculativeOptions {
                trace_limits: options.trace_limits,
                capture: options.speculative_capture,
                activations: options.speculative_activations,
                snapshots: options.speculative_snapshots,
            },
            |session| match drive(&mut GenerationControl::Speculative(session)) {
                Ok(()) => Ok(()),
                Err(error) => {
                    caller_failure = Some(error);
                    Err(
                        eredu_core::speculative::SpeculativeControlError::Unsupported(
                            "generation control callback failed",
                        ),
                    )
                }
            },
            mode,
        );
        if let Some(error) = caller_failure {
            return Err(error);
        }
        Ok(speculative_output(result?))
    }
}
impl<B> ControlSource<B> for RealizedDrafting<B::Drafter>
where
    B: SpeculativeGenerationBackend + SpeculativeForecastBackend<B::Drafter>,
{
    fn admission_stage(&self) -> eredu_core::run_preparation::TextPreparationStage {
        if self.is_enabled() {
            eredu_core::run_preparation::TextPreparationStage::Instrumentation
        } else {
            eredu_core::run_preparation::TextPreparationStage::Request
        }
    }
    fn controlled<F, D>(
        &mut self,
        model: &mut LoadedModel<B>,
        request: GenerationRequest<'_, B, F>,
        options: GenerationControlOptions,
        limits: Option<SpeculationOptions>,
        started: std::time::Instant,
        drive: D,
    ) -> Result<GenerationOutput<GenerationStats>, SessionControlError>
    where
        F: FnMut(SemanticEvent),
        D: FnOnce(&mut GenerationControl<'_, '_, B>) -> Result<(), SessionControlError>,
    {
        match self.as_speculative_draft() {
            Some(mut draft) => draft.controlled(model, request, options, limits, started, drive),
            None => NoDrafting.controlled(model, request, options, limits, started, drive),
        }
    }
}
