//! Actual original observer source prepared on this equation's own trace.
use crate::{
    backend::{array_copy::CaptureNativePopulation, error::Error},
    composition::mlx::{
        session::{
            capture_workspace::CaptureWorkspaceObserver, intervention::PreparedModelInterventions,
        },
        speculative::OriginalSpeculativeNumericalSources,
    },
};
use eredu_core::capture::CaptureInvocationShape;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::{
    capture::OriginalSpeculativeCaptureInvocation,
    speculative::embedded_occurrence::EmbeddedInvocationWorkspace,
    working_memory::{EmbeddedCaptureHostPlan, WorkingMemoryError},
};
use std::{
    cell::{Cell, RefCell},
    mem::{size_of, size_of_val},
};

/// Borrowed declaration plus a caller-owned destination for the same host plan.
/// Neither this handoff nor its trace grants native/source authority.
pub(in crate::composition::mlx::replicated_text) struct CaptureWorkspaceInput<'source, 'slot> {
    pub invocation: OriginalSpeculativeCaptureInvocation<'source>,
    pub host: &'slot mut Option<EmbeddedCaptureHostPlan<'source>>,
    pub edits: &'slot RefCell<Option<PreparedModelInterventions>>,
}
impl<'source, 'slot> CaptureWorkspaceInput<'source, 'slot> {
    pub(super) fn prepare<'trace>(
        self,
        workspace: EmbeddedInvocationWorkspace,
        context: &WorkspaceContext,
        transfers: &'trace Cell<CaptureNativePopulation>,
        sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<CaptureWorkspaceObserver<'trace>, Error>
    where
        'source: 'trace,
        'slot: 'trace,
    {
        let invocation = self.invocation;
        if self
            .edits
            .try_borrow()
            .map_err(|cause| sources.retain_startup_error(cause))?
            .is_some()
        {
            return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        let fixed = [
            size_of::<Self>(),
            eredu_runtime::capture::OriginalSpeculativeCapturePrefix::control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            size_of::<RefCell<Option<PreparedModelInterventions>>>(),
            size_of::<std::cell::RefMut<'_,Option<PreparedModelInterventions>>>(),
            size_of::<std::cell::BorrowMutError>(),
            size_of::<CaptureInvocationShape>(),
            size_of::<Cell<CaptureNativePopulation>>(),
            size_of::<Option<EmbeddedCaptureHostPlan<'source>>>(),
            size_of::<Result<CaptureWorkspaceObserver<'trace>, Error>>(),
            eredu_runtime::working_memory::OriginalSpeculativeRequest::embedded_capture_inspection_control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            EmbeddedCaptureHostPlan::inspection_control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        ];
        context
            .charge_metadata(
                fixed
                    .into_iter()
                    .try_fold(size_of_val(&fixed), usize::checked_add)
                    .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        invocation
            .source()
            .validate_pool(sources.pool())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let inherited = match invocation.lineage() {
            Some(lineage) => sources
                .request()
                .inspect_embedded_capture_lineage_usage(invocation.source(), lineage),
            None => sources
                .request()
                .inspect_embedded_capture_usage(invocation.source()),
        }
        .map_err(|cause| sources.retain_startup_error(cause))?;
        let geometry = workspace.geometry();
        if self.host.is_some()
            || invocation.phase() != workspace.invocation().phase()
            || usize::try_from(invocation.sequence()).ok()
                != Some(workspace.invocation().positions())
        {
            return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        let bounds = invocation
            .source()
            .plan()
            .admission()
            .invocation_bounds()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let shape = CaptureInvocationShape {
            batch: geometry.batch_size,
            sequence: invocation.sequence(),
            context: bounds
                .max_context
                .map(|_| {
                    geometry
                        .cached_positions
                        .checked_add(geometry.input_positions)
                        .and_then(|n| u64::try_from(n).ok())
                        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
                })
                .transpose()?,
        };
        let (observer, plan) = CaptureWorkspaceObserver::with_invocation_window(
            invocation.source().plan(),
            geometry,
            context,
            transfers,
            invocation.capture_phase(),
            invocation.origin().prediction as u64,
            shape,
            invocation.selected(),
            invocation
                .window()
                .map_err(|cause| sources.retain_startup_error(cause))?,
        )?;
        let observer = observer
            .with_inherited_usage(inherited)?
            .with_prepared_prefix(invocation.prepared_prefix())?
            .with_envelope_usage(
                invocation
                    .envelope_usage()
                    .map_err(|cause| sources.retain_startup_error(cause))?,
            )?;
        let plan = plan
            .with_skip_reasons(invocation.skip_reasons())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let plan = EmbeddedCaptureHostPlan::prepare(
            invocation.source(),
            plan,
            workspace,
            invocation.origin(),
        )
        .map_err(|cause| sources.retain_startup_error(cause))?;
        let (observer, plan) = match invocation.interventions() {
            None => (observer, plan),
            Some((source, selected)) => {
                source
                    .validate_pool(sources.pool())
                    .map_err(|cause| sources.retain_startup_error(cause))?;
                let plan = plan
                    .with_intervention_evidence(source, selected, invocation.intervention_evidence_skips())
                    .map_err(|cause| sources.retain_startup_error(cause))?;
                *self
                    .edits
                    .try_borrow_mut()
                    .map_err(|cause| sources.retain_startup_error(cause))? = Some(
                    PreparedModelInterventions::prepare_with_evidence(
                        source,
                        selected,
                        invocation.capture_phase(),
                        invocation.origin().prediction as u64,
                        shape,
                        invocation.window().map_err(|cause| sources.retain_startup_error(cause))?,
                        invocation.intervention_evidence_skips(),
                        context,
                    )
                    .map_err(|cause| sources.retain_startup_error(cause))?,
                );
                (
                    observer
                        .with_model_interventions(self.edits)
                        .map_err(|cause| sources.retain_startup_error(cause))?,
                    plan,
                )
            }
        };
        let plan = match invocation.lineage() {
            Some(lineage) => plan
                .with_lineage(lineage)
                .map_err(|cause| sources.retain_startup_error(cause))?,
            None => plan,
        };
        *self.host = Some(plan.with_quoted_usage(inherited));
        Ok(observer)
    }
}
