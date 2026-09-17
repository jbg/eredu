//! Descriptive scalar/layout rows from the same retained copy/read source.
use super::*;
use eredu_nn::{
    workspace::{WorkspaceMetadataError, WorkspaceParameterRepresentation},
    ParameterId,
};
use eredu_runtime::working_memory::WorkspaceParameterRows;
use std::mem::size_of;

impl LayerwiseWorkspace {
    /// Extends static parameter evidence for a resident-shaped cold equation.
    /// This creates no loaded parameter, read, copy, or executable authority;
    /// native source/copy/constructor admission remains with this exact owner.
    pub(crate) fn extend_workspace_parameter_representations(
        &self,
        context: &WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        context.charge_metadata(size_of::<(
            &Self,
            &WorkspaceContext,
            WorkspaceParameterRepresentation,
            eredu_runtime::working_memory::WorkspaceParameterUnit<'_>,
            eredu_runtime::working_memory::WorkspaceParameterRow<'_>,
            Result<(), eredu_nn::Error>,
            Option<eredu_nn::workspace::WorkspaceRepresentation>,
            Vec<WorkspaceParameterRepresentation>,
            ParameterId,
            WorkspaceLayout,
            Result<ParameterId, eredu_nn::ParameterTopologyError>,
            Result<
                eredu_runtime::working_memory::WorkspaceParameterRow<'_>,
                eredu_runtime::working_memory::WorkspaceParameterSourceError,
            >,
            Result<
                eredu_runtime::working_memory::WorkspaceParameterUnit<'_>,
                eredu_runtime::working_memory::WorkspaceParameterSourceError,
            >,
            [usize; 3],
        )>())?;
        let error = |cause| context.metadata_source(cause);
        let mut count = 0usize;
        for unit in 0..self.unit_count() {
            count = count
                .checked_add(self.unit(unit).map_err(error)?.rows)
                .ok_or(WorkspaceMetadataError::Overflow)?;
        }
        let mut rows = context.metadata_vec(count)?;
        for unit in 0..self.unit_count() {
            for index in 0..self.unit(unit).map_err(error)?.rows {
                let row = self.row(unit, index).map_err(error)?;
                let id = ParameterId::new(
                    context.metadata_string(format_args!("{}", row.binding.name()))?,
                )
                .map_err(|cause| context.metadata_source(cause))?;
                let layout = context
                    .layout(row.shape, row.dtype)?
                    .with_representation(row.representation);
                rows.push(WorkspaceParameterRepresentation::new(id, layout));
            }
        }
        context.extend_parameter_representations(rows)
    }
}
