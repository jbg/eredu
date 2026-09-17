//! Same neutral fixed sequence provider, funded before every actual birth.
use super::*;
use eredu_core::{HostPreparationAuthority, SpeculativeSequence, SpeculativeSequenceRef};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};

fn controls(maximum: usize, eos: usize) -> Option<usize> {
    let parts = [
        SpeculativeSequence::retained_control_bytes(maximum, eos)?,
        HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()?,
        std::mem::size_of::<Result<SpeculativeSequence, Error>>(),
        std::mem::size_of::<(usize, &[u32], SpeculativeExecutionStreams<'static>)>(),
        std::mem::size_of::<(
            SpeculativeSequenceRef<'static>,
            SpeculativeExecutionStreams<'static>,
        )>(),
        std::mem::size_of::<Result<SpeculativeSequence, eredu_core::SpeculativeDriverError<Error>>>(
        ),
        std::mem::size_of::<
            Option<(
                &crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
                &crate::backend::OriginalCopyEnvironment<'static>,
            )>,
        >(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}
fn authority(
    maximum: usize,
    eos: usize,
    sources: &crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
) -> Result<HostPreparationAuthority, Error> {
    let bytes = controls(maximum, eos).ok_or(Error::WorkspacePlanning(
        WorkspaceMetadataFundingError::Overflow,
    ))?;
    sources
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    Ok(HostPreparationAuthority::retain(
        sources.metadata_funding().clone(),
    ))
}
pub(in crate::composition::mlx) fn new(
    maximum: usize,
    eos: &[u32],
    context: SpeculativeExecutionStreams<'_>,
) -> Result<SpeculativeSequence, Error> {
    let Some((sources, environment)) = context.original_numerical() else {
        return Ok(eredu_core::GenerationSequence::new(maximum, eos.iter().copied()).into());
    };
    sources.validate_environment(environment)?;
    let authority = authority(maximum, eos.len(), sources)?;
    SpeculativeSequence::try_new_retained(maximum, eos, authority)
        .map_err(|cause| sources.retain_startup_error(cause))
}
pub(in crate::composition::mlx::speculative) fn copy(
    source: SpeculativeSequenceRef<'_>,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<SpeculativeSequence, eredu_core::SpeculativeDriverError<Error>> {
    use eredu_core::SpeculativeDriverError as DriverError;
    let Some((sources, environment)) = context.original_numerical() else {
        return source.copy_ordinary().map_err(DriverError::Preparation);
    };
    let result = (|| {
        sources.validate_environment(environment)?;
        let (maximum, eos) = source.retained_geometry().ok_or_else(|| {
            sources.retain_startup_error(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )
        })?;
        let authority = authority(maximum, eos, sources)?;
        source
            .try_copy_retained(authority)
            .map_err(|cause| sources.retain_startup_error(cause))
    })();
    result.map_err(DriverError::Backend)
}
pub(in crate::composition::mlx::speculative) fn copy_bytes(source: &SpeculativeSequence) -> Option<u64> {
    match source {
        SpeculativeSequence::Ordinary(_) => source.snapshot_storage_bytes(),
        SpeculativeSequence::Retained(_) => {
            let (m, e) = SpeculativeSequenceRef::from(source).retained_geometry()?;
            u64::try_from(controls(m, e)?).ok()
        }
    }
}
