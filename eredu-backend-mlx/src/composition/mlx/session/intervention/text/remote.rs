//! The same scheduled owner retains its actual remote publication source.
use super::*;
use eredu_architectures::component_partition::ComponentPartitionLayouts;
impl PreparedTextInterventions {
    pub(crate) fn trace_remote_output(
        &mut self,
        layouts: &ComponentPartitionLayouts,
        rank: usize,
        value: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<()> {
        self.check_context(context)?;
        context.charge_metadata(
            size_of::<(
                &mut Self,
                &ComponentPartitionLayouts,
                usize,
                &WorkspaceTensor,
                &WorkspaceContext,
            )>()
            .checked_add(size_of::<Result<()>>())
            .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let index = self
            .active
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?;
        if let Some(fragment) = self.prefill_active {
            self.prefill_rows[fragment].trace_remote_output(layouts, rank, value, context)?;
        }
        self.rows[index].trace_remote_output(layouts, rank, value, context)
    }
}
