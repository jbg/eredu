//! The ordinary equation visitor with a borrowed, span-aware observation program.
use super::*;
use eredu_runtime::{
    SharedLayeredObservationPaths,
    working_memory::{InferenceObservationError as ObservationError, InferenceWorkspaceObserver},
};
use std::cell::RefCell;

/// Actual observation source and logical prediction coordinate for one model
/// invocation. This borrows the same observer/path contracts as ordinary text
/// quoting; it grants neither a capture source nor native execution authority.
/// The coordinate comes from the caller's actual invocation provenance, not its
/// physical cache position or this equation's zero future-output count.
pub struct InvocationWorkspaceObservation<'a> {
    pub(super) paths: &'a SharedLayeredObservationPaths,
    pub(super) observer: &'a mut dyn InferenceWorkspaceObserver,
    pub(super) prediction: u64,
}
impl<'a> InvocationWorkspaceObservation<'a> {
    pub fn new(
        paths: &'a SharedLayeredObservationPaths,
        observer: &'a mut dyn InferenceWorkspaceObserver,
        prediction: u64,
    ) -> Self {
        Self {
            paths,
            observer,
            prediction,
        }
    }
}

fn source_error<E: std::error::Error + Send + Sync + 'static>(
    context: &WorkspaceContext,
    cause: E,
) -> Error {
    if context.uses_checked_metadata() {
        context.metadata_source(cause)
    } else {
        Error::backend_retained_source(cause)
    }
}

#[derive(Clone, Copy)]
pub(super) struct ObservationRef<'a, 'observer> {
    pub(super) paths: &'a SharedLayeredObservationPaths,
    pub(super) observer: &'a RefCell<&'observer mut dyn InferenceWorkspaceObserver>,
    // A source-explicit equation invocation can observe output even though its
    // geometry describes no future generation. Ordinary requests keep None.
    pub(super) invocation_prediction: Option<u64>,
}
impl ObservationRef<'_, '_> {
    pub(super) fn validate_selection(
        self,
        geometry: InferenceGeometry,
        context: &WorkspaceContext,
    ) -> Result<bool, Error> {
        let observer = self.observer.borrow();
        let Some(bound) = observer.prefill_selection() else {
            return Ok(false);
        };
        if bound.geometry() != geometry {
            return Err(source_error(
                context,
                eredu_runtime::layered::PreparedCaptureSelectionError::Identity,
            ));
        }
        bound
            .selection()
            .validate_sources(bound.selection().source(), self.paths)
            .map_err(|cause| source_error(context, cause))?;
        Ok(true)
    }
    pub(super) fn requires_sequence_readout(self) -> bool {
        self.observer.borrow().requires_sequence_readout()
    }
    pub(super) fn begin_span(
        self,
        geometry: InferenceGeometry,
        span: &InferenceWorkspaceSpan,
        context: &WorkspaceContext,
    ) -> Result<bool, Error> {
        let prediction = match self.invocation_prediction {
            Some(prediction) => Some(prediction),
            None => match span {
                InferenceWorkspaceSpan::Sampling(_) => {
                    unreachable!("model equation scheduler emits only prefill/decode spans")
                }
                InferenceWorkspaceSpan::Prefill(_) => (geometry.max_output_tokens > 0).then_some(0),
                InferenceWorkspaceSpan::Decode { index, .. } => Some(
                    index
                        .checked_add(1)
                        .ok_or_else(|| source_error(context, ObservationError::Overflow))?,
                )
                .filter(|prediction| *prediction < geometry.max_output_tokens),
            },
        };
        let Some(prediction) = prediction else {
            return Ok(false);
        };
        let bound_rows = self.validate_selection(geometry, context)?;
        let active = self
            .observer
            .borrow_mut()
            .begin_span(geometry, span, prediction, context)?;
        if active && !bound_rows {
            if let InferenceWorkspaceSpan::Prefill(chunk) = span {
                let partial = chunk.input.start != 0 || chunk.input.end != geometry.input_positions;
                if partial && self.observer.borrow().prefill_invocation_span() != Some(chunk) {
                    return Err(source_error(context, ObservationError::PartialPrefill));
                }
            }
        }
        Ok(active)
    }
    pub(super) fn append_retained(
        self,
        roots: &mut Vec<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        let mut result = Ok(());
        self.observer.borrow().visit_retained(&mut |value| {
            if result.is_ok() {
                result = context.reserve_metadata_vec(roots, 1);
                if result.is_ok() {
                    roots.push(value.clone());
                }
            }
        });
        result
    }
}

pub(super) fn finish_logits(
    observer: &mut dyn InferenceWorkspaceObserver,
    output: Option<WorkspaceTensor>,
    demand: eredu_core::OutputDemand,
    context: &WorkspaceContext,
) -> Result<Option<WorkspaceTensor>, Error> {
    let output = match output {
        Some(output) => Some(eredu_runtime::observe_model_logits(observer, &output)?),
        None if demand == eredu_core::OutputDemand::StateOnly => None,
        None => return Err(source_error(context, ObservationError::MissingLogits)),
    };
    observer.finish()?;
    Ok(output)
}

/// Selects the final position after full [batch,sequence,vocabulary] observation.
/// The interval and axis removal match the native output adapter and retain
/// whatever complete backing the selected mechanisms report.
pub(super) fn sampling_row(
    scores: &WorkspaceTensor,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    if scores.shape().len() != 3 || scores.shape().iter().any(|&n| n <= 0) {
        return Err(source_error(
            context,
            ObservationError::InvalidLogitsGeometry,
        ));
    }
    let end = scores.shape()[1];
    scores
        .narrow_axis(1, end - 1, end, context)?
        .squeeze_axes(&[1], context)
}

/// Private composition receives only the prepared borrowed-hook mechanism's
/// fixed extent. No caller bytes/report can construct an observed quote.
pub(super) fn with_hook_workspace(
    mut report: eredu_nn::workspace::WorkspaceTraceReport,
    bytes: u64,
    context: &WorkspaceContext,
) -> Result<eredu_nn::workspace::WorkspaceTraceReport, Error> {
    if let Some(domains) = &mut report.physical_domains {
        domains
            .add_host_workspace(
                context.memory_topology().ok_or_else(|| {
                    context.metadata_source(
                        eredu_nn::workspace::WorkspacePlacementError::MissingTopology,
                    )
                })?,
                bytes,
            )
            .map_err(|cause| context.metadata_source(cause))?;
    }
    let attributed = report.physical_domains.is_some();
    let add = |value: Option<u64>, bytes: u64| -> Result<Option<u64>, Error> {
        value
            .map(|value| {
                value
                    .checked_add(bytes)
                    .map(Some)
                    .or(attributed.then_some(None))
                    .ok_or_else(|| {
                        if context.uses_checked_metadata() {
                            context.metadata_source(ObservationError::Overflow)
                        } else {
                            Error::backend_retained_source(ObservationError::Overflow)
                        }
                    })
            })
            .transpose()
            .map(Option::flatten)
    };
    report.host_workspace_bytes = add(report.host_workspace_bytes, bytes)?;
    report.total_bytes = add(report.total_bytes, bytes)?;
    report.transient_bytes = add(report.transient_bytes, bytes)?;
    if let Some(state) = report.state.as_mut() {
        state.transient_bytes = add(state.transient_bytes, bytes)?;
    }
    if let Some(residual) = report.residual.as_mut() {
        residual.total_bytes = add(residual.total_bytes, bytes)?;
        residual.transient_bytes = add(residual.transient_bytes, bytes)?;
    }
    context.reserve_metadata_vec(&mut report.assumptions, 1)?;
    report.assumptions.push(context.metadata_string(format_args!("prepared borrowed observation-hook inline move overlap; registered shared path payload, observer host receipts and architecture-internal hook workspace remain separately owned"))?);
    Ok(report)
}

impl PreparedInferenceBlueprint {
    fn quote_text_observed_impl(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: Option<TextSamplingInput<'_>>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
        trace: Option<EquationTraceRef<'_, '_>>,
    ) -> Result<EquationQuote, PreparedExecutionError<Error>> {
        if geometry.max_output_tokens > 0
            && observer.requires_sequence_readout()
            && geometry.output != eredu_core::OutputDemand::Sequence
        {
            return Err(PreparedExecutionError::Backend(source_error(
                context,
                ObservationError::SequenceReadoutRequired,
            )));
        }
        if sampling.is_some()
            && (geometry.batch_size != 1 || geometry.output == eredu_core::OutputDemand::StateOnly)
        {
            return Err(preparation_message(
                context,
                format_args!("observed sampling requires one sequence with score readout"),
            ));
        }
        context
            .charge_metadata(std::mem::size_of::<(
                ObservationRef<'_, '_>,
                RefCell<&mut dyn InferenceWorkspaceObserver>,
                Option<EquationTraceRef<'_, '_>>,
                Result<EquationQuote, PreparedExecutionError<Error>>,
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let observer = RefCell::new(&mut *observer);
        let observation = ObservationRef {
            paths,
            observer: &observer,
            invocation_prediction: None,
        };
        observation
            .validate_selection(geometry, context)
            .map_err(PreparedExecutionError::Backend)?;
        self.quote_text_with_trace_impl(
            geometry,
            state,
            context,
            sampling,
            parameters,
            Some(observation),
            trace,
        )
    }
}

impl PreparedInferenceBlueprint {
    /// Quotes the same selected equations with actual retained observation paths
    /// and a span-aware program on the original workspace context. Full-sequence
    /// demand must be in original geometry; partial prefill additionally requires
    /// an exact retained causal row companion from the observer.
    /// Prior live native observations remain workspace through later spans.
    /// Source registration, observer host receipts/sidecars, and native admission
    /// remain separate; this method grants no execution or allocation authority.
    pub fn quote_replicated_resident_text_observed(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        self.quote_text_observed_impl(geometry, state, context, None, None, paths, observer, None)
            .map(|(equations, _)| equations)
    }
    /// Quotes the same selected equations with actual retained observation paths
    /// and a span-aware program on the original workspace context. Full-sequence
    /// demand must be in original geometry; partial prefill additionally requires
    /// an exact retained causal row companion from the observer.
    /// Prior live native observations remain workspace through later spans.
    /// Source registration, observer host receipts/sidecars, and native admission
    /// remain separate; this method grants no execution or allocation authority.
    pub fn quote_replicated_resident_text_with_sampling_observed<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        let (equations, sampling) = self.quote_text_observed_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Configured(config, filter.into())),
            None,
            paths,
            observer,
            None,
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("observed sampling requested"),
        })
    }
    /// Quotes the same selected equations with actual retained observation paths
    /// and a span-aware program on the original workspace context. Full-sequence
    /// demand must be in original geometry; partial prefill additionally requires
    /// an exact retained causal row companion from the observer.
    /// Prior live native observations remain workspace through later spans.
    /// Source registration, observer host receipts/sidecars, and native admission
    /// remain separate; this method grants no execution or allocation authority.
    pub fn quote_replicated_resident_text_with_existing_sampling_observed(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        let (equations, sampling) = self.quote_text_observed_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Borrowed(sampling)),
            None,
            paths,
            observer,
            None,
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("observed sampling requested"),
        })
    }
    /// Retained sampling, observation and native trace consumers share this same
    /// selected equation loop. Saved origin and account proofs remain with the
    /// caller; this method neither replays sampling nor constructs execution.
    pub fn quote_replicated_resident_text_with_existing_sampling_observed_and_trace(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_replicated_text_with_existing_sampling_observed_and_trace(
            geometry, state, context, sampling, None, paths, observer, trace,
        )
    }

    /// Observes the selected equations with the exact optional layerwise source
    /// and lends their trace to the native recorder. Source rows supply physical
    /// scalar facts at the ordinary unit binding boundary; no native allocation
    /// or execution authority is conferred by this quotation.
    pub fn quote_replicated_text_with_existing_sampling_observed_and_trace(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        context
            .charge_metadata(std::mem::size_of::<(
                Option<&dyn WorkspaceLayerwiseParameters>,
                RefCell<&mut dyn InferenceEquationTraceObserver>,
                PreparedTextGenerationWorkspace,
                Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>>,
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let trace = RefCell::new(trace);
        let (equations, sampling) = self.quote_text_observed_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Borrowed(sampling)),
            parameters,
            paths,
            observer,
            Some(EquationTraceRef(&trace)),
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("observed sampling requested"),
        })
    }

    /// Quotes the same selected equations with actual retained observation paths
    /// and a span-aware program on the original workspace context. Full-sequence
    /// demand must be in original geometry; partial prefill additionally requires
    /// an exact retained causal row companion from the observer.
    /// Prior live native observations remain workspace through later spans.
    /// Source registration, observer host receipts/sidecars, and native admission
    /// remain separate; this method grants no execution or allocation authority.
    pub fn quote_replicated_layerwise_text_observed(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: &dyn WorkspaceLayerwiseParameters,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        self.quote_text_observed_impl(
            geometry,
            state,
            context,
            None,
            Some(parameters),
            paths,
            observer,
            None,
        )
        .map(|(equations, _)| equations)
    }
    /// Quotes the same selected equations with actual retained observation paths
    /// and a span-aware program on the original workspace context. Full-sequence
    /// demand must be in original geometry; partial prefill additionally requires
    /// an exact retained causal row companion from the observer.
    /// Prior live native observations remain workspace through later spans.
    /// Source registration, observer host receipts/sidecars, and native admission
    /// remain separate; this method grants no execution or allocation authority.
    pub fn quote_replicated_layerwise_text_with_sampling_observed<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        parameters: &dyn WorkspaceLayerwiseParameters,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        let (equations, sampling) = self.quote_text_observed_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Configured(config, filter.into())),
            Some(parameters),
            paths,
            observer,
            None,
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("observed sampling requested"),
        })
    }
    /// Quotes the same selected equations with actual retained observation paths
    /// and a span-aware program on the original workspace context. Full-sequence
    /// demand must be in original geometry; partial prefill additionally requires
    /// an exact retained causal row companion from the observer.
    /// Prior live native observations remain workspace through later spans.
    /// Source registration, observer host receipts/sidecars, and native admission
    /// remain separate; this method grants no execution or allocation authority.
    pub fn quote_replicated_layerwise_text_with_existing_sampling_observed(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        sampling: BorrowedTextSamplingWorkspace<'_>,
        parameters: &dyn WorkspaceLayerwiseParameters,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        let (equations, sampling) = self.quote_text_observed_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Borrowed(sampling)),
            Some(parameters),
            paths,
            observer,
            None,
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("observed sampling requested"),
        })
    }
    /// Runs the same observed equations and sampling with their actual span reports
    /// lent to a backend recipe recorder. Observation paths, physical sequence
    /// demand, logical capture limits and every retained output remain unchanged.
    /// The recorder supplies no capture source, host destination or native authority.
    pub fn quote_replicated_resident_text_with_sampling_observed_and_trace<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        self.quote_replicated_text_with_sampling_observed_and_trace(
            geometry, state, context, config, filter, None, paths, observer, trace,
        )
    }

    /// Observes the selected equations with the exact optional layerwise source
    /// and lends their trace to the native recorder. Source rows supply physical
    /// scalar facts at the ordinary unit binding boundary; no native allocation
    /// or execution authority is conferred by this quotation.
    pub fn quote_replicated_text_with_sampling_observed_and_trace<'a>(
        &self,
        geometry: InferenceGeometry,
        state: &ResidentState,
        context: &WorkspaceContext,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        paths: &SharedLayeredObservationPaths,
        observer: &mut dyn InferenceWorkspaceObserver,
        trace: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>> {
        context
            .charge_metadata(std::mem::size_of::<(
                Option<&dyn WorkspaceLayerwiseParameters>,
                RefCell<&mut dyn InferenceEquationTraceObserver>,
                PreparedTextGenerationWorkspace,
                Result<PreparedTextGenerationWorkspace, PreparedExecutionError<Error>>,
            )>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let trace = RefCell::new(trace);
        let (equations, sampling) = self.quote_text_observed_impl(
            geometry,
            state,
            context,
            Some(TextSamplingInput::Configured(config, filter.into())),
            parameters,
            paths,
            observer,
            Some(EquationTraceRef(&trace)),
        )?;
        Ok(PreparedTextGenerationWorkspace {
            equations,
            sampling: sampling.expect("observed sampling requested"),
        })
    }
}

#[cfg(test)]
mod tests;
