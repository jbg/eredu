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
        state: &ResidentState,
        context: &WorkspaceContext,
        parameters: Option<&dyn WorkspaceLayerwiseParameters>,
        observer: &mut dyn InferenceEquationTraceObserver,
    ) -> Result<InferenceWorkspaceReport, PreparedExecutionError<Error>> {
        let controls = [
            std::mem::size_of::<AutoregressiveInvocation>(),
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
        let geometry = schedule
            .workspace_geometry(frontier, invocation, prefill_chunk_positions)
            .map_err(|cause| PreparedExecutionError::Metadata(context.metadata_source(cause)))?;
        let observer = std::cell::RefCell::new(observer);
        let (equations, _) = self.quote_text_with_input_dtype(
            geometry,
            state,
            context,
            None,
            parameters,
            None,
            Some(EquationTraceRef(&observer)),
            Some(input_dtype),
            Some(invocation.execution_pass()),
        )?;
        Ok(equations)
    }
}
