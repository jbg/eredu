//! Source-funded persistence identity copies and fixed contract diagnostics.
use eredu_core::cache::PromptCacheDescriptor;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use std::mem::{size_of, size_of_val};

pub(crate) fn clone_descriptor(
    source: &PromptCacheDescriptor,
    context: Option<&WorkspaceContext>,
) -> Result<PromptCacheDescriptor, Error> {
    if let Some(context) = context {
        let frames = [
            size_of::<PromptCacheDescriptor>(),
            size_of::<(&PromptCacheDescriptor, Option<&WorkspaceContext>)>(),
            size_of::<Result<PromptCacheDescriptor, Error>>(),
            crate::StateLayout::schedule_clone_metadata_bytes(source.layer_layout())?,
            source
                .layer_prefix_offsets()
                .len()
                .checked_mul(size_of::<i32>())
                .ok_or(WorkspaceMetadataError::Overflow)?,
            source
                .state_segments()
                .len()
                .checked_mul(size_of::<eredu_core::cache::PromptCacheStateSegment>())
                .ok_or(WorkspaceMetadataError::Overflow)?,
        ];
        let mut bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        for text in [
            source.model_family(),
            source.effective_model_type(),
            source.checkpoint_fingerprint(),
            source.prefix_content_fingerprint(),
            source.architecture_fingerprint(),
        ] {
            bytes = bytes
                .checked_add(text.len())
                .ok_or(WorkspaceMetadataError::Overflow)?;
        }
        for segment in source.state_segments() {
            bytes = bytes
                .checked_add(segment.id().len())
                .ok_or(WorkspaceMetadataError::Overflow)?;
        }
        context.charge_metadata(bytes)?;
    }
    Ok(source.clone())
}

/// Fixed diagnostics retain the actual persistence payer without allocating a
/// message. They carry no source, admission or native execution authority.
#[derive(Debug, thiserror::Error)]
pub(crate) enum CacheContractFailure {
    #[error("{0}")]
    Static(&'static str),
    #[error("another rank failed distributed cache control at {0:?}")]
    Remote(crate::DistributedExecutionPhase),
}
