//! One source contract shared by cold model quotation and remote publication.
use super::*;
use eredu_architectures::component_partition::{ComponentPartitionLayouts,ContiguousPartitionCaptureSource};
use eredu_runtime::inspection::ObservationHookSite;

pub(in crate::composition::mlx) fn validate(selection:&CaptureSelection,
    placement:(&ComponentPartitionLayouts,usize),context:&WorkspaceContext)->Result<()> {
    let metadata=Metadata::new(context)?;
    let parts=[size_of::<(&CaptureSelection,(&ComponentPartitionLayouts,usize),&WorkspaceContext)>(),
        size_of::<Option<eredu_architectures::component_partition::CompletePartitionCaptureSource>>(),
        size_of::<ContiguousPartitionCaptureSource<'_>>(),
        size_of::<std::result::Result<ContiguousPartitionCaptureSource<'_>,eredu_architectures::component_partition::PartitionCaptureSourceError>>(),
        size_of::<Result<()>>(),size_of::<ObservationHookSite>(),size_of::<bool>()*2,
        ComponentPartitionLayouts::complete_capture_source_control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        ContiguousPartitionCaptureSource::control_bytes().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?];
    context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
    if placement.1>=placement.0.topology().world_size() {
        return Err(metadata.coordinate());
    }
    if matches!(selection.transform,CaptureTransform::RoutedUnits) {
        context.charge_metadata(
            eredu_architectures::component_partition::RoutedPartitionCaptureSource::control_bytes()
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?)?;
        let source=placement.0.routed_capture_source(&selection.path).map_err(|cause|metadata.error(cause))?;
        if source.world_size()!=placement.0.topology().world_size() {return Err(metadata.coordinate());}
        return Ok(());
    }
    if placement.0.complete_capture_source(&selection.path).is_some() {
        if matches!(selection.transform,CaptureTransform::FullTensor|CaptureTransform::Slice
            |CaptureTransform::Preview{..}|CaptureTransform::Summary|CaptureTransform::Histogram{..}
            |CaptureTransform::TopCandidates{..}|CaptureTransform::TokenScores{..}) {return Ok(());}
        return Err(metadata.coordinate());
    }
    // Both admitted initial-prefill and decode coordinates have the shared
    // projected equation, original per-frame Host, raw additive assembly and
    // typed late Source vote. Vocabulary selection needs a separate projected
    // producer; explicit nested invocation windows keep their own source rule.
    if !matches!(selection.transform,CaptureTransform::FullTensor|CaptureTransform::Slice
            |CaptureTransform::Preview{..}|CaptureTransform::Summary|CaptureTransform::Histogram{..}) {
        return Err(metadata.coordinate());
    }
    let source=placement.0.contiguous_capture_source(&selection.path).map_err(|cause|metadata.error(cause))?;
    if source.world_size()!=placement.0.topology().world_size()
        || !matches!(source.site(),ObservationHookSite::Input|ObservationHookSite::Unit|ObservationHookSite::Readout) {
        return Err(metadata.coordinate());
    }
    // Presence remains separate from export. Cold end_fragment_span handles an
    // absent PP invocation; the real local callback supplies scalar/geometry for
    // replicas and empty shards, with no fabricated tensor on absent ranks.
    Ok(())
}
