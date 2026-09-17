//! Source-explicit capture through the ordinary managed plain session.
use super::*;
use eredu_core::capture::{CapturedStepDelivery, SharedCapturePlan};
use eredu_core::intervention::SharedInterventionPlan;

impl<'a, B: TextGenerationBackend> ManagedPlainTextSession<'a, B> {
    /// Attaches a caller-borrowed observer to this session's existing capture
    /// source. This changes delivery only: it cannot install a capture plan,
    /// reset usage, admit work or change native completion.
    ///
    /// The callback receives the optional committed token, an optional completed
    /// capture frame, and the active native step duration in seconds. A failed
    /// or cancelled step can deliver a frame without a token. It runs before
    /// that token's text events, through the same delivery agreement as ordinary
    /// generation. Cancel the shared cancellation token to stop advancement.
    ///
    /// Shared frames move with their original custody; callers may retain or
    /// clone their shared owner without copying the payload. No raw DTO clone,
    /// JSON buffer or separate trace budget is created by this interface.
    /// A restored session can attach a fresh observer by this same method;
    /// callback state is caller-owned and is never snapshotted.
    pub fn with_capture_observer(
        mut self,
        observer: &'a mut dyn FnMut(Option<u32>, Option<CapturedStepDelivery>, f64),
    ) -> Self {
        self.0.set_capture_observer(observer);
        self
    }
}

impl<B: OriginalTokenizerBackend> LoadedModel<B> {
    /// Starts managed plain generation with an explicitly supplied immutable
    /// capture source. Managed admission starts at this already admitted plan;
    /// constructing the caller's plan and discovery are separate operations.
    ///
    /// The source must describe this selected model and the actual prompt,
    /// cached frontier and prediction allowance. Core preparation passes it to
    /// the selected backend, which validates and accounts for its exact source,
    /// collector and native transformations before exposing the session. Missing
    /// producer bounds refuse normally; this method supplies no ordinary fallback.
    ///
    /// The returned session uses the same `advance` and `run` as unobserved
    /// managed generation. Its borrowed observer receives retained capture
    /// deliveries, including available failure/cancellation frames; see
    /// [`ManagedPlainTextSession::with_capture_observer`]. Pre-start cancellation
    /// returns `None` without calling the observer or preparing the request.
    pub fn start_observed_managed_plain_text<'a>(
        &'a mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextRequest<'_>,
        capture: SharedCapturePlan,
        cancellation: &GenerationCancellationToken,
        observer: &'a mut dyn FnMut(Option<u32>, Option<CapturedStepDelivery>, f64),
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedPlainTextError> {
        self.start_managed_plain_text_with_options(
            source,
            request,
            cancellation,
            Some(eredu_core::TextPreparationOptions {
                interventions: None, capture: Some(capture),
            }),
        )
        .map(|session| session.map(|session| session.with_capture_observer(observer)))
    }

    /// Starts the same controlled managed session with admitted intervention
    /// and capture sources. An empty capture selection can still deliver edit
    /// outcomes through the observer. Both plans must describe this request.
    ///
    /// Advancement, cancellation, text termination and retained frame delivery
    /// use the ordinary managed driver. The selected backend must admit the
    /// actual edits before returning a session.
    pub fn start_intervened_managed_plain_text<'a>(
        &'a mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextRequest<'_>,
        capture: SharedCapturePlan,
        interventions: SharedInterventionPlan,
        cancellation: &GenerationCancellationToken,
        observer: &'a mut dyn FnMut(Option<u32>, Option<CapturedStepDelivery>, f64),
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedPlainTextError> {
        self.start_managed_plain_text_with_options(
            source, request, cancellation,
            Some(eredu_core::TextPreparationOptions {
                capture: Some(capture), interventions: Some(interventions),
            }),
        ).map(|session| session.map(|session| session.with_capture_observer(observer)))
    }

    /// Runs the same observed managed session to completion. Capture delivery
    /// and text delivery use the existing shared source/cursor; neither callback
    /// needs an owned event buffer. Escaped shared frames and terminal output
    /// retain their own actual storage custody after this call returns.
    pub fn generate_observed_managed_plain_text(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPlainTextRequest<'_>,
        capture: SharedCapturePlan,
        cancellation: &GenerationCancellationToken,
        observer: &mut dyn FnMut(Option<u32>, Option<CapturedStepDelivery>, f64),
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<Option<GenerationPlainTextOutput>, ManagedPlainTextError> {
        self.start_observed_managed_plain_text(source, request, capture, cancellation, observer)?
            .map(|session| {
                session
                    .run(cancellation, emit)
                    .map_err(|cause| ManagedPlainTextError::new(Cause::Backend(cause)))
            })
            .transpose()
    }
}

impl<B: OriginalTokenizerBackend> LoadedModel<B> {
    /// Starts the shared managed driver from an authenticated original media
    /// input and an explicit capture source. Media positions and alignment come
    /// from that input; the selected backend admits the exact observation
    /// transformations and retained roots before returning the session.
    pub fn start_observed_managed_prepared_input<'a>(
        &'a mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedInputRequest<'_, B::Prompt>,
        capture: SharedCapturePlan,
        cancellation: &GenerationCancellationToken,
        observer: &'a mut dyn FnMut(Option<u32>, Option<CapturedStepDelivery>, f64),
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedPlainTextError> {
        self.start_managed_prepared_input_with_options(
            source, request, cancellation,
            Some(eredu_core::TextPreparationOptions { interventions: None, capture: Some(capture) }),
        ).map(|session| session.map(|session| session.with_capture_observer(observer)))
    }

    /// Starts the same managed session from authenticated media with admitted
    /// edit and capture sources. Preparation keeps the input's media alignment;
    /// outcomes use the ordinary capture observer and cumulative schedule.
    pub fn start_intervened_managed_prepared_input<'a>(
        &'a mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedInputRequest<'_, B::Prompt>,
        capture: SharedCapturePlan,
        interventions: SharedInterventionPlan,
        cancellation: &GenerationCancellationToken,
        observer: &'a mut dyn FnMut(Option<u32>, Option<CapturedStepDelivery>, f64),
    ) -> Result<Option<ManagedPlainTextSession<'a, B>>, ManagedPlainTextError> {
        self.start_managed_prepared_input_with_options(
            source, request, cancellation,
            Some(eredu_core::TextPreparationOptions {
                capture: Some(capture), interventions: Some(interventions),
            }),
        ).map(|session| session.map(|session| session.with_capture_observer(observer)))
    }

    /// Runs the same observed media session through ordinary commitment, capture
    /// delivery, decoding and termination. Escaped frames keep their own payer.
    pub fn generate_observed_managed_prepared_input(
        &mut self,
        source: &ManagedPlainTextSource,
        request: ManagedPreparedInputRequest<'_, B::Prompt>,
        capture: SharedCapturePlan,
        cancellation: &GenerationCancellationToken,
        observer: &mut dyn FnMut(Option<u32>, Option<CapturedStepDelivery>, f64),
        emit: &mut impl for<'e> FnMut(GenerationPlainTextEvent<'e>),
    ) -> Result<Option<GenerationPlainTextOutput>, ManagedPlainTextError> {
        self.start_observed_managed_prepared_input(
            source, request, capture, cancellation, observer,
        )?.map(|session| session.run(cancellation, emit)
            .map_err(|cause| ManagedPlainTextError::new(Cause::Backend(cause))))
            .transpose()
    }
}
