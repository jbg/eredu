//! Shared fixture producers with the same explicit destination as the contracts.
use super::*;
use eredu_nn::workspace::WorkspaceContext;

pub(super) fn vector<T, const N: usize>(
    values: [T; N],
    metadata: Option<&WorkspaceContext>,
) -> Result<Vec<T>, Error> {
    if let Some(metadata) = metadata {
        metadata.charge_metadata(std::mem::size_of::<(
            [T; N], Option<&WorkspaceContext>, Vec<T>, Result<Vec<T>, Error>,
        )>())?;
    }
    let mut output = match metadata {
        Some(metadata) => metadata.metadata_vec(N)?,
        None => Vec::with_capacity(N),
    };
    output.extend(values);
    Ok(output)
}

pub(super) fn layout(
    policies: Vec<LayerCachePolicy>,
    metadata: Option<&WorkspaceContext>,
) -> Result<StateLayout, Error> {
    if let Some(metadata) = metadata {
        metadata.charge_metadata(std::mem::size_of::<(
            Vec<LayerCachePolicy>, LayerSchedule<LayerCachePolicy>, StateLayout,
            Option<&WorkspaceContext>, Result<StateLayout, StateError>, Result<StateLayout, Error>,
        )>())?;
        if policies.capacity() != policies.len() {
            metadata.charge_metadata(std::alloc::Layout::array::<LayerCachePolicy>(policies.len())
                .map_err(|_| eredu_nn::workspace::WorkspaceMetadataError::Overflow)?.size())?;
        }
    }
    let schedule = LayerSchedule::new(policies.len(), policies).map_err(|cause| match metadata {
        Some(metadata) => metadata.metadata_source(cause),
        None => Error::backend_retained_source(cause),
    })?;
    let result = match metadata {
        Some(metadata) => StateLayout::new_with_metadata(schedule, metadata),
        None => StateLayout::new(schedule),
    };
    result.map_err(|cause| match cause {
        StateError::WorkspaceConstruction(cause) => cause,
        cause => match metadata {
            Some(metadata) => metadata.metadata_source(cause),
            None => Error::backend_retained_source(cause),
        },
    })
}

pub(super) fn identity(
    name: &str,
    fingerprint: &str,
    layers: usize,
    state: &eredu_runtime::PartitionState,
    topology: eredu_core::cache::PromptCacheTopology,
    metadata: Option<&WorkspaceContext>,
) -> Result<eredu_runtime::ModelStateIdentity, Error> {
    if let Some(metadata) = metadata {
        metadata.charge_metadata(std::mem::size_of::<(
            &str, &str, usize, &eredu_runtime::PartitionState,
            eredu_core::cache::PromptCacheTopology, Option<&WorkspaceContext>,
            String, String, String, eredu_runtime::ModelStateIdentity,
            Result<eredu_runtime::ModelStateIdentity, Error>,
        )>())?;
    }
    let text = |value: &str| match metadata {
        Some(metadata) => metadata.metadata_string(format_args!("{value}")),
        None => Ok(value.to_owned()),
    };
    eredu_runtime::ModelStateIdentity::new_with_diagnostic(
        text(name)?, text(name)?, text(fingerprint)?, layers,
        state.global_layer_offset(), 0, topology,
        |cause| match metadata {
            Some(metadata) => metadata.metadata_error(cause),
            None => Error::backend(cause),
        },
    )
}

pub(super) fn text(value: std::fmt::Arguments<'_>, metadata: Option<&WorkspaceContext>) -> Result<String, Error> {
    match metadata {
        Some(metadata) => metadata.metadata_string(value),
        None => Ok(value.to_string()),
    }
}
pub(super) fn diagnostic(value: std::fmt::Arguments<'_>, metadata: Option<&WorkspaceContext>) -> Error {
    match metadata {
        Some(metadata) => metadata.metadata_error(value),
        None => Error::backend(value),
    }
}
