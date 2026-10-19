//! Complete post-reduction observation producers for internal prediction transport.
use super::*;

impl ComponentPartitionLayouts {
    /// Selects one authoritative complete producer per ordered observation.
    ///
    /// Shapes are exact invocation geometry; retained component coordinates prove
    /// the scalar width and invocation ownership. This does not assemble sharded
    /// inputs or additive contributions. Those require their declared reduction
    /// before a complete-tensor capture can use this publication mechanism.
    pub fn whole_tensor_capture_bundle(
        &self,
        captures: &[(String, Vec<i32>)],
        element: eredu_nn::TensorElementType,
    ) -> Result<eredu_runtime::ForwardCaptureBundle, ComponentPartitionError> {
        use eredu_core::capture::CaptureError;
        let mut slots = Vec::with_capacity(captures.len());
        for (path, shape) in captures {
            let width = shape
                .last()
                .copied()
                .filter(|width| *width > 0)
                .ok_or_else(|| {
                    CaptureError::Invalid("capture geometry has no positive scalar width".into())
                })? as usize;
            if self.observation_combination(path)? != PartitionCaptureCombination::Disjoint {
                return Err(CaptureError::Unsupported(
                    "capture requires architecture reduction before publication".into(),
                )
                .into());
            }
            let owner = self
                .layouts
                .iter()
                .filter_map(|layout| {
                    let observation = layout.observation(path)?;
                    let coordinates = observation.coordinates()?;
                    (observation.exports()
                        && coordinates.global_count() == width
                        && coordinates.contiguous_range() == Some(0..width))
                    .then_some(layout.topology().global_rank())
                })
                .min()
                .ok_or_else(|| {
                    CaptureError::Unsupported(format!(
                        "capture has no authoritative complete producer: {path}"
                    ))
                })?;
            slots.push(
                eredu_runtime::ForwardCaptureSlot::new(path.clone(), owner, shape.clone(), element)
                    .map_err(CaptureError::Invalid)?,
            );
        }
        eredu_runtime::ForwardCaptureBundle::new(slots)
            .map_err(|detail| CaptureError::Invalid(detail).into())
    }
}

/// Composite block nodes can contain only structural metadata. Their parameter
/// declarations belong to descendant sublayers, all of which must prove the
/// same physical invocation owner before the block boundary can be replicated.
pub(super) fn block_invocation_owner<'a>(
    descriptor: &ArchitectureDescriptor,
    parameters: &'a ArchitectureParameterDescription,
    block: &str,
) -> Result<&'a eredu_runtime::ParameterGroupOwner, ComponentPartitionError> {
    use eredu_core::capture::CaptureError;
    let parents = descriptor
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node.parent.as_deref()))
        .collect::<BTreeMap<_, _>>();
    let mut owner = None;
    for candidate in &descriptor.nodes {
        if candidate.parameter_groups.is_empty() {
            continue;
        }
        let mut current = Some(candidate.id.as_str());
        let mut contained = false;
        for _ in 0..=descriptor.nodes.len() {
            let Some(id) = current else {
                break;
            };
            if id == block {
                contained = true;
                break;
            }
            current = parents.get(id).copied().flatten();
        }
        if !contained {
            continue;
        }
        let actual = node_invocation_owner(descriptor, parameters, &candidate.id)?;
        if owner.is_some_and(|previous| previous != actual) {
            return Err(CaptureError::Invalid(
                "decoder block spans different invocation owners".into(),
            )
            .into());
        }
        owner = Some(actual);
    }
    owner.ok_or_else(|| {
        CaptureError::Invalid("decoder block has no parameter-owned invocation".into()).into()
    })
}
