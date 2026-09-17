//! Source-owned companion selections on the actual pre/post-edit model graph.
use super::*;
use crate::backend::array_copy::{CaptureTensorSelection, PreparedCaptureSummary};
use eredu_core::checkpoint::TensorDtype;
use eredu_runtime::{
    capture::CaptureObservationStep,
    working_memory::{CaptureInterventionEvidenceClaim, CaptureInterventionEvidenceKind},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum State {
    Inactive,
    MetadataSkipped,
    Empty,
    ValueSkipped,
    Traced,
}
enum Program<'a> {
    Preview(CaptureTensorGeometry<'a>),
    Summary(CaptureSummaryGeometry<'a>),
}
impl Program<'_> {
    fn empty(&self) -> bool {
        match self {
            Self::Preview(g) => g.starts().iter().zip(g.ends()).any(|(a, b)| a == b),
            Self::Summary(g) => g.starts().iter().zip(g.ends()).any(|(a, b)| a == b),
        }
    }
    fn validate(&self, value: &WorkspaceTensor, ctx: &WorkspaceContext) -> Result<(), Failure> {
        match self {
            Self::Preview(g) => {
                CaptureTensorSelection::from_geometry(g)?.validate_workspace_source(value, ctx)
            }
            Self::Summary(g) => {
                PreparedCaptureSummary::from_geometry(g)?.validate_workspace_source(value, ctx)
            }
        }
    }
    fn usage(&self) -> Result<CaptureUsage, CaptureError> {
        use crate::composition::mlx::session::bounded_capture;
        match self {
            Self::Preview(g) => bounded_capture::estimate_tensor_geometry(g),
            Self::Summary(g) => bounded_capture::estimate_summary(g),
        }
    }
    fn trace(
        &self,
        value: &WorkspaceTensor,
        ctx: &WorkspaceContext,
        roots: &mut Vec<WorkspaceTensor>,
    ) -> Result<CaptureNativePopulation, Failure> {
        let population = match self {
            Self::Preview(_) => CaptureNativePopulation::within_raw(),
            Self::Summary(g) => {
                CaptureNativePopulation::within_summary(&PreparedCaptureSummary::from_geometry(g)?)
            }
        }
        .ok_or(Failure::GeometryOverflow)?;
        // Native shared readout retains its actual input alias too. All aliases
        // remain in the enclosing model role's final root union and Recovery.
        ctx.reserve_metadata_vec(roots, 1)?;
        roots.push(value.clone());
        match self {
            Self::Preview(g) => {
                CaptureTensorSelection::from_geometry(g)?
                    .trace_retained_within(value, ctx, roots)?;
            }
            Self::Summary(g) => {
                PreparedCaptureSummary::from_geometry(g)?.trace(value, ctx, roots)?
            }
        }
        Ok(population)
    }
}
pub(super) fn trace(
    model: &PreparedModelInterventions,
    operation: usize,
    side: InterventionEvidenceSide,
    input: &WorkspaceTensor,
    context: &WorkspaceContext,
    ledger: &mut CaptureLedger,
    roots: &mut Vec<WorkspaceTensor>,
) -> Result<(State, CaptureNativePopulation), eredu_nn::Error> {
    context.charge_metadata(planning_controls().ok_or(WorkspaceMetadataError::Overflow)?)?;
    let Some(companion) = model.source.plan().evidence(operation) else {
        return Ok((State::Inactive, CaptureNativePopulation::default()));
    };
    if model
        .evidence_skips
        .as_ref()
        .is_some_and(|rows| rows[operation][side.index()].is_some())
    {
        return Ok((State::Inactive, CaptureNativePopulation::default()));
    }
    let policy = CaptureObservationStep::with_invocation(
        companion.geometry_source(),
        model.phase,
        model.prediction,
        model.invocation,
    )
    .and_then(|policy| policy.with_window(model.window))
    .map_err(|cause| context.metadata_source(cause))?;
    let controls = execution_controls().ok_or(WorkspaceMetadataError::Overflow)?;
    let empty = CaptureNativePopulation {
        controls,
        ..Default::default()
    };
    if let Some(usage) = policy
        .window_metadata_usage(side.index())
        .map_err(|cause| context.metadata_source(cause))?
    {
        if ledger
            .reserve(usage)
            .map_err(|cause| context.metadata_source(cause))?
            .is_some()
        {
            return Ok((State::MetadataSkipped, empty));
        }
    }
    let program = match companion.geometry_source().plan().selections[side.index()].transform {
        CaptureTransform::Preview { .. } => Program::Preview(
            policy
                .tensor_geometry(side.index())
                .map_err(|cause| context.metadata_source(cause))?,
        ),
        CaptureTransform::Summary => Program::Summary(
            policy
                .summary_geometry(side.index())
                .map_err(|cause| context.metadata_source(cause))?,
        ),
        _ => return Err(context.metadata_source(Failure::ClaimMismatch)),
    };
    program
        .validate(input, context)
        .map_err(|cause| context.metadata_source(cause))?;
    if model.window.is_some() && program.empty() {
        return Ok((State::Empty, empty));
    }
    let usage = program
        .usage()
        .map_err(|cause| context.metadata_source(cause))?;
    if ledger
        .reserve(usage)
        .map_err(|cause| context.metadata_source(cause))?
        .is_some()
    {
        return Ok((State::ValueSkipped, empty));
    }
    let population = program
        .trace(input, context, roots)
        .map_err(|cause| context.metadata_source(cause))?
        .checked_add(empty)
        .ok_or(WorkspaceMetadataError::Overflow)?;
    Ok((State::Traced, population))
}
impl PreparedModelInterventions {
    /// Exact source, model account, side and physical source geometry. This does
    /// not settle a value or turn geometry into native source authority.
    pub(crate) fn check_evidence(
        &self,
        value: &Array,
        claim: &CaptureInterventionEvidenceClaim<'_, '_>,
        custody: &OriginalSpeculativeBudgetCustody,
        execute: bool,
    ) -> Result<(), Failure> {
        self.check_evidence_with(value, claim, NativeCustody::Model(custody), execute)
    }
    pub(crate) fn check_evidence_scheduled(
        &self, value: &Array, claim: &CaptureInterventionEvidenceClaim<'_, '_>,
        scope: &WorkingMemoryFundingScope, execute: bool,
    ) -> Result<(), Failure> {
        self.check_evidence_with(value, claim, NativeCustody::Scheduled(scope), execute)
    }
    fn check_evidence_with(
        &self, value: &Array, claim: &CaptureInterventionEvidenceClaim<'_, '_>,
        custody: NativeCustody<'_>, execute: bool,
    ) -> Result<(), Failure> {
        claim.validate_source(&self.source)?;
        match custody {
            NativeCustody::Model(custody) => claim.validate_model_custody(custody)?,
            NativeCustody::Scheduled(scope) => claim.validate_native_custody(scope)?,
        }
        let (index, side, phase, prediction) = claim.coordinate();
        if !self.sealed
            || (phase, prediction) != (self.phase, self.prediction)
            || claim.invocation() != self.invocation
            || claim.invocation_window() != self.window
        {
            return Err(Failure::ClaimMismatch);
        }
        let Some(Row::Ready(edit)) = self.rows.get(index) else {
            return Err(Failure::ClaimMismatch);
        };
        let state = edit.evidence[side.index()];
        if matches!(state, State::Inactive | State::MetadataSkipped)
            || execute && state != State::Traced
        {
            return Err(Failure::ClaimMismatch);
        }
        let dtype = match value.dtype() {
            safemlx::Dtype::Float32 => InterventionDtype::Float32,
            safemlx::Dtype::Float16 => InterventionDtype::Float16,
            safemlx::Dtype::Bfloat16 => InterventionDtype::Bfloat16,
            other => return Err(Failure::UnsupportedDtype(other)),
        };
        if dtype != edit.dtype
            || value.shape().len() != edit.shape.len()
            || value
                .shape()
                .iter()
                .zip(&edit.shape)
                .any(|(a, b)| u64::try_from(*a).ok() != Some(*b))
        {
            return Err(Failure::ShapeMismatch);
        }
        if (state == State::Empty) != claim.window_empty() {
            return Err(Failure::ClaimMismatch);
        }
        Ok(())
    }
}
pub(super) fn planning_controls() -> Option<usize> {
    let frames = [
        size_of::<Program<'static>>(),
        size_of::<CaptureTensorSelection>(),
        size_of::<PreparedCaptureSummary>(),
        size_of::<Result<CaptureTensorSelection, Failure>>(),
        size_of::<Result<PreparedCaptureSummary, Failure>>(),
        size_of::<Result<CaptureNativePopulation, Failure>>(),
        size_of::<(
            &Program<'static>,
            &WorkspaceTensor,
            &WorkspaceContext,
            &mut Vec<WorkspaceTensor>,
        )>(),
        size_of::<Result<Program<'static>, eredu_nn::Error>>(),
        size_of::<CaptureTensorGeometry<'static>>(),
        size_of::<CaptureSummaryGeometry<'static>>(),
        size_of::<CaptureObservationStep<'static>>(),
        size_of::<Result<CaptureObservationStep<'static>, CaptureError>>(),
        size_of::<Result<Option<CaptureUsage>, CaptureError>>(),
        size_of::<CaptureNativePopulation>(),
        size_of::<(State, CaptureNativePopulation)>(),
        size_of::<Result<(State, CaptureNativePopulation), eredu_nn::Error>>(),
        size_of::<(
            &PreparedModelInterventions,
            usize,
            InterventionEvidenceSide,
            &WorkspaceTensor,
            &WorkspaceContext,
            &mut CaptureLedger,
            &mut Vec<WorkspaceTensor>,
        )>(),
        CaptureTensorGeometry::preparation_control_bytes()?,
        CaptureSummaryGeometry::preparation_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
pub(super) fn execution_controls() -> Option<usize> {
    use eredu_runtime::working_memory::{ClaimedInterventionEvidence, InterventionEvidenceReceipt};
    let frames = [
        size_of::<(&PreparedModelInterventions, &Array, &CaptureInterventionEvidenceClaim<'_, '_>,
            NativeCustody<'_>, bool)>(),
        size_of::<(&PreparedModelInterventions, &Array, &CaptureInterventionEvidenceClaim<'_, '_>,
            &WorkingMemoryFundingScope, bool)>(),
        size_of::<CaptureInterventionEvidenceClaim<'static, 'static>>(),
        size_of::<CaptureInterventionEvidenceKind<'static, 'static>>(),
        size_of::<InterventionEvidenceReceipt<'static>>(),
        size_of::<ClaimedInterventionEvidence<'static>>(),
        size_of::<Option<ClaimedInterventionEvidence<'static>>>(),
        size_of::<Result<ClaimedInterventionEvidence<'static>, FundedCaptureError<Failure>>>(),
        size_of::<Result<(TensorDtype, CaptureUsage), FundedCaptureError<Failure>>>(),
        size_of::<Result<(), Failure>>(),
        size_of::<(
            &PreparedModelInterventions,
            &Array,
            &CaptureInterventionEvidenceClaim<'static, 'static>,
            &OriginalSpeculativeBudgetCustody,
            bool,
        )>(),
        size_of::<(usize, InterventionEvidenceSide, CapturePhase, u64)>(),
        size_of::<(
            &crate::MlxTensor,
            CaptureInterventionEvidenceClaim<'static, 'static>,
        )>(),
        size_of::<(
            &crate::MlxTensor,
            &CaptureInterventionEvidenceClaim<'static, 'static>,
        )>(),
        size_of::<(State, InterventionDtype)>(),
        size_of::<CaptureUsage>(),
        size_of::<TensorDtype>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

#[cfg(test)]
mod tests;
