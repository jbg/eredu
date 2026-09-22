//! Immutable completed replacement facts beside the original materialization source.
use super::*;
use eredu_nn::workspace::{
    WorkspaceFloatingType, WorkspaceParameterRepresentation, WorkspaceRepresentation,
};
use eredu_runtime::parameter_operations::ParameterReplacementValues;

#[derive(Debug)]
pub(super) struct ReplacementRow {
    pub(super) name: String,
    pub(super) shape: Vec<i32>,
    pub(super) dtype: WorkspaceDtype,
    pub(super) representation: Option<WorkspaceRepresentation>,
    pub(super) logical_bytes: u64,
    pub(super) allocation: safemlx::ArrayAllocationInfo,
    pub(super) placement: &'static eredu_core::MemoryPlacement,
}

pub(super) fn snapshot(
    values: &ParameterReplacementValues<MlxTensor>,
    context: Option<&WorkspaceContext>,
) -> Result<Vec<ReplacementRow>, Error> {
    let mut rows = match context {
        Some(context) => context.metadata_vec(values.len()).map_err(Error::Neural)?,
        None => Vec::with_capacity(values.len()),
    };
    for (name, value) in values.iter() {
        if let Some(context) = context {
            let bytes = std::mem::size_of::<(
                ReplacementRow,
                &str,
                &MlxTensor,
                safemlx::ArrayDescriptorFacts,
                Option<safemlx::ArrayAllocationInfo>,
                Result<Vec<ReplacementRow>, Error>,
            )>()
            .checked_add(safemlx::Array::descriptor_control_bytes().ok_or_else(unknown)?)
            .ok_or_else(overflow)?;
            context
                .charge_metadata(bytes)
                .map_err(|cause| Error::Neural(cause.into()))?;
        }
        let descriptor = value
            .as_array()
            .try_descriptor()
            .map_err(|cause| Error::Neural(eredu_nn::Error::backend_retained_source(cause)))?;
        let facts = descriptor.facts();
        let allocation = facts.allocation().ok_or_else(unknown)?;
        let placement = crate::backend::managed_memory::cold_allocation_placement(&allocation)
            .ok_or_else(unknown)?;
        let (dtype, floating) = match facts.dtype() {
            safemlx::Dtype::Float32 => (
                WorkspaceDtype::Float32,
                Some(WorkspaceFloatingType::Float32),
            ),
            safemlx::Dtype::Float16 => (
                WorkspaceDtype::Float32,
                Some(WorkspaceFloatingType::Float16),
            ),
            safemlx::Dtype::Bfloat16 => (
                WorkspaceDtype::Float32,
                Some(WorkspaceFloatingType::Bfloat16),
            ),
            safemlx::Dtype::Int32 => (WorkspaceDtype::Int32, None),
            safemlx::Dtype::Uint32 => (WorkspaceDtype::Uint32, None),
            safemlx::Dtype::Uint8 => (WorkspaceDtype::Uint8, None),
            safemlx::Dtype::Bool => (WorkspaceDtype::Bool, None),
            _ => return Err(unknown()),
        };
        let mut shape = match context {
            Some(context) => context.metadata_vec(facts.rank()).map_err(Error::Neural)?,
            None => Vec::with_capacity(facts.rank()),
        };
        shape.extend_from_slice(descriptor.shape());
        let name = match context {
            Some(context) => context
                .metadata_string(format_args!("{name}"))
                .map_err(Error::Neural)?,
            None => name.to_owned(),
        };
        rows.push(ReplacementRow {
            name,
            shape,
            dtype,
            representation: floating.map(|kind| {
                WorkspaceRepresentation::new(kind, descriptor.row_contiguous().unwrap_or(false))
            }),
            logical_bytes: u64::try_from(facts.logical_bytes()).map_err(|_| overflow())?,
            allocation,
            placement,
        });
    }
    Ok(rows)
}
fn unknown() -> Error {
    Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
}
fn overflow() -> Error {
    Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
}

impl LayerwiseWorkspace {
    pub(crate) fn replacement_values(&self) -> &ParameterReplacementValues<MlxTensor> {
        &self.identity.replacements
    }
    pub(super) fn replacement_row(&self, name: &str) -> Option<&ReplacementRow> {
        self.replacement_rows
            .binary_search_by(|row| row.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.replacement_rows[index])
    }
    /// Joins the exact retained arrays with the same map as static parameters.
    /// Native allocation generations, capacities and placement decide sharing.
    pub(crate) fn install_replacement_parameter_representations(
        &self,
        context: &WorkspaceContext,
        backings: &mut crate::backend::nn::workspace::ParameterWorkspaceBackings,
    ) -> Result<(), eredu_nn::Error> {
        let mut rows = context.metadata_vec(self.replacement_rows.len())?;
        for row in &self.replacement_rows {
            let backing = backings.import_observed(row.allocation, context)?;
            let id =
                eredu_nn::ParameterId::new(context.metadata_string(format_args!("{}", row.name))?)
                    .map_err(|cause| context.metadata_source(cause))?;
            let layout = context
                .layout(&row.shape, row.dtype)?
                .with_representation(row.representation);
            rows.push(WorkspaceParameterRepresentation::new(id, layout).with_backing(backing));
        }
        if !rows.is_empty() {
            context.extend_parameter_representations(rows)?;
        }
        Ok(())
    }
}
