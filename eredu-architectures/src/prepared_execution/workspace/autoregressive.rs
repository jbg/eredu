//! Selected independent invocation over the same ordinary unit equations.
use super::*;
use eredu_nn::workspace::WorkspaceDtype;
use eredu_runtime::speculative::autoregressive::{
    AutoregressiveInvocation, AutoregressiveSchedulePlan,
};
use std::num::NonZeroU64;

impl PreparedInferenceBlueprint {
    /// Quotes one actual independent-decoder invocation using the ordinary
    /// architecture unit worker. The caller must establish that its selected
    /// native strategy uses those units; provider/partition execution requires
    /// its own adapter. The invocation retains the explicit semantic pass even
    /// though the ordinary unit strategy, like its native counterpart, does not
    /// dispatch on it. Routed providers consume that same explicit pass for
    /// acquisition and chunk policy. A multi-token Decode is never reclassified
    /// as Prefill merely because its finite storage report has one Prefill span.
    ///
    /// `state` is the actual lane projection at `frontier`; this method neither
    /// fabricates future state nor treats the largest shape as a bound for all
    /// alternatives. No sampler, checkpoint-copy worker or submission is priced
    /// by this equation-only report.
    pub fn quote_ordinary_autoregressive_invocation(
        &self,
        schedule: &AutoregressiveSchedulePlan<'_>,
        frontier: u64,
        invocation: AutoregressiveInvocation,
        prefill_chunk_positions: NonZeroU64,
        input_dtype: WorkspaceDtype,
        media: Option<(
            OriginalMediaWorkspaceInput,
            &eredu_runtime::working_memory::MediaSessionBinding,
        )>,
        state: &ResidentState,
        context: &WorkspaceContext,
        observer: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        self.quote_ordinary_autoregressive_invocation_with_parameters(
            schedule,
            frontier,
            invocation,
            prefill_chunk_positions,
            input_dtype,
            media,
            state,
            context,
            None,
            observer,
        )
    }

    /// Same actual invocation with the selected source's borrowed layerwise
    /// parameter inventory. Materialization, source pins and native transfer
    /// admission remain the caller's separate obligations.
    pub fn quote_ordinary_autoregressive_invocation_with_parameters(
        &self,
        schedule: &AutoregressiveSchedulePlan<'_>,
        frontier: u64,
        invocation: AutoregressiveInvocation,
        prefill_chunk_positions: NonZeroU64,
        input_dtype: WorkspaceDtype,
        media: Option<(
            OriginalMediaWorkspaceInput,
            &eredu_runtime::working_memory::MediaSessionBinding,
        )>,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observer: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        self.quote_ordinary_autoregressive_invocation_observed(
            schedule,
            frontier,
            invocation,
            prefill_chunk_positions,
            input_dtype,
            media,
            state,
            context,
            parameters,
            None,
            observer,
        )
    }

    /// The same selected invocation with source-bound observation hooks. The
    /// caller supplies the actual logical prediction coordinate separately from
    /// the lane frontier and retains native capture/admission ownership.
    pub fn quote_ordinary_autoregressive_invocation_observed(
        &self,
        schedule: &AutoregressiveSchedulePlan<'_>,
        frontier: u64,
        invocation: AutoregressiveInvocation,
        prefill_chunk_positions: NonZeroU64,
        input_dtype: WorkspaceDtype,
        media: Option<(
            OriginalMediaWorkspaceInput,
            &eredu_runtime::working_memory::MediaSessionBinding,
        )>,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observation: Option<InvocationWorkspaceObservation<'_>>,
        observer: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        self.quote_autoregressive_invocation_observed(
            schedule,
            frontier,
            invocation,
            prefill_chunk_positions,
            input_dtype,
            media,
            state,
            context,
            parameters,
            observation,
            observer,
            None,
        )
    }

    /// Quotes the actual independent invocation through its retained constructor.
    /// A partition source retains its rank-local communication declaration and
    /// uses the same partition equation worker as ordinary text. This describes
    /// execution and grants no native admission or submission authority.
    #[allow(clippy::too_many_arguments)]
    pub fn quote_autoregressive_invocation_observed(
        &self,
        schedule: &AutoregressiveSchedulePlan<'_>,
        frontier: u64,
        invocation: AutoregressiveInvocation,
        prefill_chunk_positions: NonZeroU64,
        input_dtype: WorkspaceDtype,
        media: Option<(
            OriginalMediaWorkspaceInput,
            &eredu_runtime::working_memory::MediaSessionBinding,
        )>,
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observation: Option<InvocationWorkspaceObservation<'_>>,
        observer: &mut dyn InferenceEquationTraceObserver,
        communication: Option<&eredu_runtime::RetainedCommunicationSource>,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        let controls = [
            std::mem::size_of::<Option<&eredu_runtime::RetainedCommunicationSource>>(),
            std::mem::size_of::<AutoregressiveInvocation>(),
            std::mem::size_of::<Option<InvocationWorkspaceObservation<'_>>>(),
            std::mem::size_of::<
                Option<(
                    &eredu_runtime::SharedLayeredObservationPaths,
                    std::cell::RefCell<
                        &mut dyn eredu_runtime::working_memory::InferenceWorkspaceObserver,
                    >,
                    u64,
                )>,
            >(),
            std::mem::size_of::<Option<ObservationRef<'_, '_>>>(),
            std::mem::size_of::<Option<&dyn WorkspaceLayerwiseParameters>>(),
            std::mem::size_of::<InferenceGeometry>(),
            std::mem::size_of::<EquationVisitor<'_, '_, '_>>(),
            std::mem::size_of::<EquationQuote>(),
            std::mem::size_of::<Result<EquationQuote, PreparedExecutionError<Error>>>(),
            std::mem::size_of::<Result<InferenceWorkspaceReport, PreparedExecutionError<Error>>>(),
            std::mem::size_of::<std::cell::RefCell<&mut dyn InferenceEquationTraceObserver>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| {
                PreparedExecutionError::Metadata(
                    eredu_nn::workspace::WorkspaceMetadataError::Overflow.into(),
                )
            })?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| PreparedExecutionError::Metadata(cause.into()))?;
        if !matches!(input_dtype, WorkspaceDtype::Int32 | WorkspaceDtype::Uint32) {
            return Err(preparation_message(
                context,
                format_args!("independent decoder requires token-index input"),
            ));
        }
        let mut geometry = schedule
            .workspace_geometry(frontier, invocation, prefill_chunk_positions)
            .map_err(|cause| PreparedExecutionError::Metadata(context.metadata_source(cause)))?;
        // Observation declares its actual readout source before equations. The
        // native admission companion independently authenticates this change.
        if invocation.execution_pass() == eredu_runtime::ExpertPass::Prefill
            && observation
                .as_ref()
                .is_some_and(|source| source.observer.requires_sequence_readout())
        {
            geometry.output = eredu_core::OutputDemand::Sequence;
        }
        let observation = observation.map(|source| {
            (
                source.paths,
                std::cell::RefCell::new(source.observer),
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
        let observer = std::cell::RefCell::new(observer);
        if let Some((input, binding)) = media {
            if invocation.execution_pass() != eredu_runtime::ExpertPass::Prefill {
                return Err(preparation_message(
                    context,
                    format_args!("media ingress requires an actual prefill invocation"),
                ));
            }
            return self
                .quote_original_media_with_trace(
                    input,
                    binding,
                    geometry,
                    state,
                    context,
                    parameters,
                    None,
                    Some(EquationTraceRef(&observer)),
                    observation,
                    communication,
                )
                .map(|report| report.into_equations())
                .map_err(|cause| {
                    PreparedExecutionError::Metadata(context.metadata_source(cause.into_failure()))
                });
        }
        if let Some(communication) = communication {
            geometry
                .validate()
                .map_err(|cause| preparation_message(context, format_args!("{cause}")))?;
            if matches!(
                self.selected().text_realization().residency(),
                eredu_runtime::LayerWeightResidency::FullyResident
            ) != parameters.is_none()
            {
                return Err(preparation_message(
                    context,
                    format_args!(
                        "partition invocation parameter source differs from selected residency"
                    ),
                ));
            }
            let source = self
                .sources
                .construction_semantics()
                .direct_partition
                .get()
                .ok_or_else(|| {
                    preparation_message(
                        context,
                        format_args!(
                            "partition invocation requires its completed constructor source"
                        ),
                    )
                })?;
            let visitor = EquationVisitor {
                geometry,
                state,
                context,
                sampling: None,
                parameters,
                unpriced_execution: None,
                observation,
                trace: Some(EquationTraceRef(&observer)),
                media: None,
                input_dtype: Some(input_dtype),
                target_capture: false,
                routed_pass: Some(invocation.execution_pass()),
                external_target: None,
            };
            let (equations, _) = source
                .quote_media(&self.sources, Some(communication), visitor)
                .map_err(PreparedExecutionError::Metadata)?;
            return Ok(equations);
        }
        let (equations, _) = self.quote_text_with_input_dtype(
            geometry,
            state,
            context,
            None,
            parameters,
            observation,
            Some(EquationTraceRef(&observer)),
            Some(input_dtype),
            Some(invocation.execution_pass()),
        )?;
        Ok(equations)
    }
}
