//! Identity-keyed binding of already projected layerwise parameter storage.

use eredu_nn::{workspace::WorkspaceTensor, Error, ParameterId, Parameterized};
use std::collections::BTreeMap;

/// Binds supplied metadata values after validating the complete immutable and
/// mutable parameter traversals. Missing, unexpected or repeated identities,
/// traversal disagreement, and incompatible shapes, contexts or dtypes reject
/// before any parameter is replaced.
///
/// Floating native representations share the conservative `Float32` workspace
/// dtype; packed and integer representations must match their exact descriptor.
/// The provider supplies actual storage capacities and aliases. This does not
/// infer capacity from logical shape: broadcast views and floating projections
/// can validly have smaller physical backing. Unknown backing stays unknown.
/// No checkpoint data, tensor values or native resources are read or allocated.
pub fn bind_workspace_parameters<M: Parameterized<WorkspaceTensor>>(
    module: &mut M,
    weights: BTreeMap<ParameterId, WorkspaceTensor>,
) -> Result<(), Error> {
    crate::parameter::bind_parameter_values(
        module,
        weights,
        |_| false,
        |parameter: &WorkspaceTensor, weight: &WorkspaceTensor| {
            validate_workspace_binding(parameter, weight).map_err(|cause| match cause {
                PreparedWorkspaceBindingCause::Shape => Error::backend(format!(
                    "workspace parameter shape {:?} does not match supplied weight {:?}",
                    parameter.layout().shape(), weight.layout().shape(),
                )),
                PreparedWorkspaceBindingCause::Context => Error::backend(
                    "workspace parameter and supplied weight belong to different contexts",
                ),
                PreparedWorkspaceBindingCause::Dtype => Error::backend(format!(
                    "workspace parameter dtype {:?} does not match supplied weight {:?}",
                    parameter.layout().dtype(), weight.layout().dtype(),
                )),
            })
        },
        |parameter, weight| *parameter = weight,
    )
    .map_err(Error::backend_source)
}

#[cfg(test)]
mod tests;

/// Fixed prepublication mismatch shared by map and finite-row workspace binding.
#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum PreparedWorkspaceBindingCause {
    /// Represented logical shapes differ.
    #[error("workspace parameter shape differs from its source")]
    Shape,
    /// Values belong to different workspace traces.
    #[error("workspace parameter and source belong to different contexts")]
    Context,
    /// Represented scalar/packed dtypes differ.
    #[error("workspace parameter dtype differs from its source")]
    Dtype,
}
fn validate_workspace_binding(parameter: &WorkspaceTensor, weight: &WorkspaceTensor)
    -> Result<(), PreparedWorkspaceBindingCause>
{
    if parameter.layout().shape() != weight.layout().shape() {
        return Err(PreparedWorkspaceBindingCause::Shape);
    }
    if !parameter.same_context(weight) { return Err(PreparedWorkspaceBindingCause::Context); }
    if parameter.layout().dtype() != weight.layout().dtype() {
        return Err(PreparedWorkspaceBindingCause::Dtype);
    }
    Ok(())
}

/// Finite caller-owned rows through the same prepublication parameter traversal.
/// The caller pays row storage, retains source/control custody, and supplies the
/// actual context. Neither this function nor matching geometry grants source
/// residency, native execution, or completion authority.
pub fn bind_prepared_workspace_parameters<M: Parameterized<WorkspaceTensor>>(
    module: &mut M,
    rows: &mut [crate::PreparedParameterBinding<'_, WorkspaceTensor>],
    context: &eredu_nn::workspace::WorkspaceContext,
) -> Result<(), Error> {
    use eredu_nn::workspace::WorkspaceMetadataError;
    use std::mem::{size_of, size_of_val};
    let all = crate::prepared_parameter_binding_control_bytes::<
        WorkspaceTensor, WorkspaceTensor, PreparedWorkspaceBindingCause,
    >(rows.len()).ok_or(WorkspaceMetadataError::Overflow)?;
    // Row storage is the caller's already-paid destination; retain all shared
    // traversal and return controls from the actual finite binding query.
    let row_bytes = std::alloc::Layout::array::<crate::PreparedParameterBinding<'_, WorkspaceTensor>>(rows.len())
        .map_err(|_| WorkspaceMetadataError::Overflow)?.size();
    let frames = [all.checked_sub(row_bytes).ok_or(WorkspaceMetadataError::Overflow)?,
        size_of::<&mut M>(), size_of::<&mut [crate::PreparedParameterBinding<'_, WorkspaceTensor>]>(),
        size_of::<Result<(), Error>>()];
    context.charge_metadata(frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
        .ok_or(WorkspaceMetadataError::Overflow)?)?;
    crate::bind_prepared_parameter_values(module, rows, |_| false,
        validate_workspace_binding, |parameter, weight| *parameter = weight)
        .map_err(|cause| context.metadata_source(cause))
}
