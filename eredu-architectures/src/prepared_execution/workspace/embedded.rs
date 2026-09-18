//! Target-only equations of the retained embedded prediction selection.
use super::*;
use eredu_nn::workspace::{WorkspaceDtype, WorkspaceMetadataError};
use eredu_runtime::{
    SharedLayeredObservationPaths,
    speculative::embedded_occurrence::{EmbeddedInvocationWorkspace, EmbeddedOccurrenceError},
    working_memory::InferenceWorkspaceObserver,
};
use std::cell::RefCell;

/// Actual observation source and logical prediction coordinate for one target
/// invocation. This borrows the same observer/path contracts as ordinary text
/// quoting; it grants neither a capture source nor native execution authority.
/// The coordinate comes from the caller's actual invocation provenance, not its
/// physical target-cache position or this equation's zero future-output count.
pub struct EmbeddedTargetWorkspaceObservation<'a> {
    paths: &'a SharedLayeredObservationPaths,
    observer: &'a mut dyn InferenceWorkspaceObserver,
    prediction: u64,
}
impl<'a> EmbeddedTargetWorkspaceObservation<'a> {
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

impl PreparedInferenceBlueprint {
    /// Traces one target prefill span, verification or target replay through the
    /// same selected architecture construction and resident/layerwise equations.
    /// The selected embedded source pairing is validated without constructing its
    /// prediction extension or changing its classification/placement. `state`
    /// must project the actual target at the descriptor's frontier into `context`.
    ///
    /// Vocabulary demand comes from the descriptor. The actual target hidden
    /// capture is retained independently, including state-only prefill spans,
    /// and is lent separately to `observe_target_with_storage`. Observation must
    /// provide its actual source and coordinate; sequence readout requirements
    /// must already be present in the descriptor. The caller separately owns
    /// input preparation, source pins, prediction equations, checkpoint copies,
    /// completion, sampling and native admission.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_embedded_target_invocation(
        &self,
        workspace: EmbeddedInvocationWorkspace,
        input_dtype: WorkspaceDtype,
        media: Option<OriginalMediaWorkspaceInput>,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observation: Option<EmbeddedTargetWorkspaceObservation<'_>>,
        trace: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        let controls = [
            std::mem::size_of::<EmbeddedInvocationWorkspace>(),
            std::mem::size_of::<Result<EmbeddedInvocationWorkspace, EmbeddedOccurrenceError>>(),
            std::mem::size_of::<Option<&dyn WorkspaceLayerwiseParameters>>(),
            std::mem::size_of::<Option<EmbeddedTargetWorkspaceObservation<'_>>>(),
            std::mem::size_of::<
                Option<(
                    &SharedLayeredObservationPaths,
                    RefCell<&mut dyn InferenceWorkspaceObserver>,
                    u64,
                )>,
            >(),
            std::mem::size_of::<Option<ObservationRef<'_, '_>>>(),
            std::mem::size_of::<InferenceGeometry>(),
            std::mem::size_of::<EquationVisitor<'_, '_, '_>>(),
            std::mem::size_of::<EquationQuote>(),
            std::mem::size_of::<Result<EquationQuote, PreparedExecutionError<Error>>>(),
            std::mem::size_of::<Result<InferenceWorkspaceReport, PreparedExecutionError<Error>>>(),
            std::mem::size_of::<RefCell<&mut dyn InferenceEquationTraceObserver>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| {
                PreparedExecutionError::Metadata(WorkspaceMetadataError::Overflow.into())
            })?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let target = EmbeddedInvocationWorkspace::target_with_readout(
            workspace.invocation(),
            workspace.geometry().output,
        )
        .map_err(|cause| PreparedExecutionError::Metadata(context.metadata_source(cause)))?;
        if target != workspace {
            return Err(PreparedExecutionError::Metadata(
                context.metadata_source(EmbeddedOccurrenceError::Geometry),
            ));
        }
        if !matches!(input_dtype, WorkspaceDtype::Int32 | WorkspaceDtype::Uint32) {
            return Err(preparation_message(
                context,
                format_args!("embedded target requires token-index input"),
            ));
        }
        let geometry = workspace.geometry();
        let observation = observation.map(|source| {
            (
                source.paths,
                RefCell::new(source.observer),
                source.prediction,
            )
        });
        let observation =
            observation
                .as_ref()
                .map(|(paths, observer, prediction)| ObservationRef {
                    paths,
                    observer,
                    invocation_prediction: Some(*prediction),
                });
        if let Some(observation) = observation {
            if observation.requires_sequence_readout()
                && geometry.output != eredu_core::OutputDemand::Sequence
            {
                return Err(PreparedExecutionError::Metadata(context.metadata_source(
                    eredu_runtime::working_memory::InferenceObservationError::SequenceReadoutRequired,
                )));
            }
            observation
                .validate_selection(geometry, context)
                .map_err(PreparedExecutionError::Backend)?;
        }
        let trace = RefCell::new(trace);
        context.charge_metadata(std::mem::size_of::<RefCell<Option<OriginalMediaWorkspaceInput>>>())
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        let has_media = media.is_some();
        let media = RefCell::new(media);
        self.quote_text_with_target_capture(
            geometry,
            state,
            context,
            None,
            parameters,
            observation,
            Some(EquationTraceRef(&trace)),
            Some(input_dtype),
            true,
            None,
            has_media.then_some(MediaEquationRef { input: &media, intervals: None }),
        )
        .map(|(report, _)| report)
    }
}

#[cfg(test)]
mod tests;
