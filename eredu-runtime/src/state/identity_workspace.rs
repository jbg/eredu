//! Counted final identity construction from actual architecture-owned facts.
use super::*;
use eredu_core::cache::PromptCacheDiagnosticKind;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use std::mem::{size_of, size_of_val};

fn diagnostic(
    context: &WorkspaceContext,
    kind: PromptCacheDiagnosticKind,
    text: std::fmt::Arguments<'_>,
) -> Error {
    match context.metadata_string(text) {
        Ok(text) => context.metadata_source(kind.into_error(text)),
        Err(cause) => cause,
    }
}

impl ModelStateIdentity {
    /// Consumes the actual architecture identity; its three strings are moved,
    /// never cloned or substituted with expected selection fingerprints. Their
    /// original producer has its own construction/accounting responsibility.
    pub(crate) fn into_prompt_cache_identity_workspace(
        self,
        layout: &StateLayout,
        context: &WorkspaceContext,
    ) -> Result<PromptCacheModelIdentity, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<PromptCacheModelIdentity>(),
            size_of::<Result<PromptCacheModelIdentity, Error>>(),
            size_of::<PromptCacheStateSegment>(),
            size_of::<Result<PromptCacheStateSegment, Error>>(),
            size_of::<PromptCacheDiagnosticKind>(),
            size_of::<(&StateLayout, &WorkspaceContext)>(),
        ];
        context.charge_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let global_layer_end = self
            .global_layer_start
            .checked_add(layout.len())
            .ok_or_else(|| {
                diagnostic(
                    context,
                    PromptCacheDiagnosticKind::Malformed,
                    format_args!("owned layer range overflowed"),
                )
            })?;
        let layer_layout = layout.clone_layers_workspace(context)?;
        let mut offsets = context.metadata_vec(layout.len())?;
        offsets.extend(layout.iter_layer_prefix_offsets());
        let mut segments = context.metadata_vec(layout.segments().len())?;
        for segment in layout.segments() {
            let id = context.metadata_string(format_args!("{}", segment.id().as_str()))?;
            segments.push(PromptCacheStateSegment::new_with_diagnostic(
                id,
                segment.layers(),
                |text| diagnostic(context, PromptCacheDiagnosticKind::Malformed, text),
            )?);
        }
        PromptCacheModelIdentity::new_with_diagnostic(
            self.model_family,
            self.effective_model_type,
            self.architecture_fingerprint,
            self.layer_count,
            self.global_layer_start,
            global_layer_end,
            self.sink_tokens,
            self.topology,
            layer_layout,
            offsets,
            segments,
            |kind, text| diagnostic(context, kind, text),
        )
    }
}
