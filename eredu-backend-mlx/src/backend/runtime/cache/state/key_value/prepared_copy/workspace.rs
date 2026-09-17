//! Actual saved/live source projected through the closed dense-copy program.

use super::PreparedResidentKvCopy;
use crate::backend::{
    array_copy::IsolatedArrayCopy,
    nn::workspace::{ExistingArrayProjection, ProjectedNativeStorage},
};
use eredu_nn::{
    Error,
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTraceReport},
};
use eredu_runtime::{DeviceState, working_memory::WorkspaceResidentLayerState};
use std::num::NonZeroU32;

/// Independent destination roots with their actual isolated-copy bounds. Native
/// sources stay separately witnessed; no source allocation is relabeled as a
/// destination or charged twice as an independently copied alias.
pub(crate) struct ProjectedDenseResidentKvCopy {
    pub(crate) state: DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    pub(crate) source_storage: ProjectedNativeStorage,
    /// Complete copy span with both source and final destination roots retained.
    /// Source registration, host table construction and future equations must
    /// still be composed explicitly; this report grants no allocation authority.
    pub(crate) copy: WorkspaceTraceReport,
}

impl<'a> PreparedResidentKvCopy<'a> {
    /// Trace the same independently copied operands that fill the dense native
    /// table. All source roots enter the opening span before any copy equation;
    /// repeated operands preserve source aliases but produce distinct outputs.
    /// The selected exact-concatenation controls and geometry are validated by
    /// the same import used by ordinary live-state inspection.
    ///
    /// Only metadata and temporary inspection handles are allocated. No native
    /// evaluation, completion, publication or destination allocation occurs.
    /// Unknown backing remains unknown. Keep the actual source settled and
    /// exclusive for this quote and its subsequent admission. Future equation
    /// spans may advance the returned state; quote copy preparation separately.
    /// This route traces before returning its source witnesses. It is a full
    /// conservative quote: do not bind borrowed-storage credit to this context
    /// afterward. Such credit must be installed before any traced operation.
    pub(crate) fn project_dense_workspace(
        &self,
        batch: NonZeroU32,
        context: &'a WorkspaceContext,
    ) -> Result<ProjectedDenseResidentKvCopy, Error> {
        let mut projection = ExistingArrayProjection::new(context);
        let mut retained = projection.project_roots(|visitor| self.visit_operands(visitor))?;
        context.begin_state_span(&retained)?;
        let state = super::super::workspace::project_layers(
            self.shared_layout(),
            self.len(),
            batch,
            context,
            |index, policy| {
                self.layer(index)
                    .ok_or_else(|| {
                        context.metadata_error(format_args!("missing resident source layer"))
                    })?
                    .project_workspace_attention_with(policy, context, |array| {
                        let copied = IsolatedArrayCopy::new(array).trace(&mut projection)?;
                        context.reserve_metadata_vec(&mut retained, 1)?;
                        retained.push(copied.clone());
                        Ok(copied)
                    })
            },
        )?;
        let copy = context.finish_report(&retained)?;
        let source_storage = projection.try_into_storage()?;
        Ok(ProjectedDenseResidentKvCopy {
            state,
            source_storage,
            copy,
        })
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
