//! The exact admitted edit source joins the existing cumulative host schedule.
use super::*;
use crate::composition::mlx::session::{
    OriginalInterventionDeclaration, intervention::NativeInterventionEstimator,
};
use eredu_core::{
    HostPreparationAuthority,
    intervention::{PreparedInterventionPlanCopy, SharedInterventionPlan},
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::{
    intervention::StaticInterventionPreflight,
    working_memory::{OriginalInterventionSource, OriginalInterventionSourceError},
};
use std::mem::{size_of, size_of_val};

impl<'a> CaptureAdmission<'a> {
    /// Validate the actual loaded declaration before copying its immutable source.
    /// Only the same original pool can compile the source that the host bank
    /// later retains. This does not admit native edits or install an observer.
    pub(in crate::composition::mlx::session::model_session::text_quote) fn with_intervention_source(
        mut self,
        source: &SharedInterventionPlan,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<Self, Error> {
        let pool = &self.session.payload.memory_ledger;
        let funding = pool
            .prepare_workspace_metadata(
                self.session
                    .payload
                    .model
                    .erased()
                    .inference_execution_identity(),
                capacity,
            )
            .map_err(Error::WorkspacePlanning)?;
        let overflow = || Error::WorkspacePlanning(HostMetadataFundingError::Overflow);
        let frames = [
            size_of::<(&Self, &SharedInterventionPlan, u64)>(),
            size_of::<HostMetadataFunding>(),
            size_of::<OriginalInterventionSource>(),
            size_of::<Option<OriginalInterventionDeclaration>>(),
            size_of::<Result<Option<OriginalInterventionDeclaration>, Error>>(),
            size_of::<Result<OriginalInterventionSource, OriginalInterventionSourceError>>(),
            size_of::<Result<Self, Error>>(),
            PreparedInterventionPlanCopy::inspection_control_bytes().ok_or_else(overflow)?,
            OriginalInterventionDeclaration::validation_control_bytes().ok_or_else(overflow)?,
            NativeInterventionEstimator::prepared_preflight_control_bytes().ok_or_else(overflow)?,
            StaticInterventionPreflight::required_bytes().ok_or_else(overflow)?,
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                .ok_or_else(overflow)?,
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let retain =
            |cause| crate::composition::mlx::model::retain_planning_error(cause, funding.clone());
        let declaration = self
            .session
            .original_intervention_declaration(&funding)?
            .ok_or_else(|| retain(WorkingMemoryError::UnknownBound))?;
        declaration.validate(source.admission())?;
        let copy = PreparedInterventionPlanCopy::inspect(source.admission()).map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })?;
        let original = pool.compile_intervention_source(copy).map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })?;
        let mut scratch =
            StaticInterventionPreflight::prepare(HostPreparationAuthority::retain(funding.clone()))
                .map_err(|cause| {
                    crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
                })?;
        scratch
            .run_with_source(
                self.source.admission(),
                original.plan(),
                &NativeInterventionEstimator,
            )
            .map_err(|cause| {
                crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
            })?;
        self.host = self.host.with_interventions(&original).map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })?;
        Ok(self)
    }

    pub(in crate::composition::mlx::session::model_session::text_quote) fn intervention_source(
        &self,
    ) -> Option<&OriginalInterventionSource> {
        self.host.intervention_source()
    }
}

/// Saved cold quotation and accepted reconstruction both authenticate the
/// retained edit declaration against the actual destination model before rows
/// or native work are prepared. This neither recopies the plan nor resets usage.
pub(super) fn validate_saved_source(
    session: &MlxModelSession,
    source: Option<&OriginalInterventionSource>,
    metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
) -> Result<(), Error> {
    let Some(source) = source else {
        return Ok(());
    };
    let parts = [
        size_of::<(
            &MlxModelSession,
            Option<&OriginalInterventionSource>,
            eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
        )>(),
        size_of::<Option<HostMetadataFunding>>(),
        size_of::<HostMetadataFunding>(),
        size_of::<Option<OriginalInterventionDeclaration>>(),
        size_of::<Result<Option<OriginalInterventionDeclaration>, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        OriginalInterventionSource::validation_control_bytes().ok_or_else(unknown)?,
        OriginalInterventionDeclaration::validation_control_bytes().ok_or_else(unknown)?,
    ];
    metadata
        .charge(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(unknown)?,
        )
        .map_err(|cause| Error::Neural(metadata.error(cause)))?;
    source
        .validate_pool(&session.payload.memory_ledger)
        .map_err(|cause| Error::Neural(metadata.source(cause)))?;
    let funding = metadata.funding().ok_or_else(unknown)?;
    let declaration = session
        .original_intervention_declaration(&funding)?
        .ok_or_else(|| Error::Neural(metadata.source(WorkingMemoryError::UnknownBound)))?;
    declaration.validate(source.plan().admission())
}
