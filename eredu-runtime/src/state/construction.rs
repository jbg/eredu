//! The ordinary layout validator with explicit destinations for metadata births.
use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};
use std::mem::{size_of, size_of_val};

impl StateLayout {
    /// Copies the exact intersecting state policies and segments through the
    /// supplied metadata context, using the ordinary slice validator.
    pub fn slice_with_metadata(
        &self,
        layers: Range<usize>,
        context: &WorkspaceContext,
    ) -> Result<Self, StateError> {
        slice(
            self,
            layers,
            Some(context).filter(|context| context.uses_checked_metadata()),
        )
    }

    /// Constructs the actual policy's components and default segment through the
    /// supplied metadata context. The policy schedule must already be owned.
    pub fn new_with_metadata(
        layers: LayerSchedule<LayerCachePolicy>,
        context: &WorkspaceContext,
    ) -> Result<Self, StateError> {
        simple(
            layers,
            Some(context).filter(|context| context.uses_checked_metadata()),
        )
    }

    /// Consumes already-constructed segment declarations without copying them.
    /// The same validator expands every policy and validates the exact partition.
    pub fn segmented_with_metadata(
        layers: LayerSchedule<LayerCachePolicy>,
        segments: Vec<StateSegmentSpec>,
        context: &WorkspaceContext,
    ) -> Result<Self, StateError> {
        segmented(
            layers,
            || Ok(segments),
            Some(context).filter(|context| context.uses_checked_metadata()),
        )
    }
}

impl StateSegmentSpec {
    /// Constructs the original segment identity in a counted string destination.
    pub fn new_with_metadata(
        id: &str,
        layers: Range<usize>,
        lifetime: StateSegmentLifetime,
        processed_token_offset: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, StateError> {
        if !context.uses_checked_metadata() {
            return Self::new(id, layers, lifetime, processed_token_offset);
        }
        context
            .charge_metadata(size_of::<(Self, Result<Self, StateError>, &WorkspaceContext)>())
            .map_err(eredu_nn::Error::from)?;
        let id = context.metadata_string(format_args!("{id}"))?;
        Self::new(id, layers, lifetime, processed_token_offset)
    }
}

pub(super) fn simple(
    layers: LayerSchedule<LayerCachePolicy>,
    context: Option<&WorkspaceContext>,
) -> Result<StateLayout, StateError> {
    if layers.is_empty() {
        return Err(StateError::EmptyLayout);
    }
    let count = layers.len();
    // Preserve creation of the default segment before per-policy validation.
    let segment = match context {
        Some(context) => StateSegmentSpec::new_with_metadata(
            DEFAULT_STATE_SEGMENT_ID,
            0..count,
            StateSegmentLifetime::Persistent,
            0,
            context,
        )?,
        None => StateSegmentSpec::new(
            DEFAULT_STATE_SEGMENT_ID,
            0..count,
            StateSegmentLifetime::Persistent,
            0,
        )?,
    };
    segmented(
        layers,
        || {
            let mut segments = vector(1, context)?;
            segments.push(segment);
            Ok(segments)
        },
        context,
    )
}

fn vector<T>(count: usize, context: Option<&WorkspaceContext>) -> Result<Vec<T>, StateError> {
    match context {
        Some(context) => context.metadata_vec(count).map_err(StateError::from),
        None => Ok(Vec::with_capacity(count)),
    }
}

fn identity(
    id: &StateSegmentId,
    context: Option<&WorkspaceContext>,
) -> Result<StateSegmentId, StateError> {
    match context {
        Some(context) => Ok(StateSegmentId(
            context.metadata_string(format_args!("{id}"))?,
        )),
        None => Ok(id.clone()),
    }
}

pub(super) fn segmented(
    layers: LayerSchedule<LayerCachePolicy>,
    segments: impl FnOnce() -> Result<Vec<StateSegmentSpec>, StateError>,
    context: Option<&WorkspaceContext>,
) -> Result<StateLayout, StateError> {
    if let Some(context) = context {
        let controls = [
            size_of::<StateLayout>(),
            size_of::<Result<StateLayout, StateError>>(),
            size_of::<StateError>(),
            size_of::<(&LayerSchedule<LayerCachePolicy>, &WorkspaceContext)>(),
            size_of::<(usize, &StateSegmentSpec)>(),
            size_of::<Result<Vec<StateComponentPolicy>, eredu_nn::Error>>(),
            size_of_val(&segments),
        ];
        context
            .charge_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(WorkspaceMetadataError::Overflow)
                    .map_err(eredu_nn::Error::from)?,
            )
            .map_err(eredu_nn::Error::from)?;
    }
    if layers.is_empty() {
        return Err(StateError::EmptyLayout);
    }
    for (layer, policy) in layers.iter().enumerate() {
        policy.validate_with_diagnostic(|text| match context {
            Some(context) => StateError::workspace_invalid_layer(context, layer, text),
            None => StateError::InvalidLayer {
                layer,
                reason: text.to_string(),
            },
        })?;
    }
    let mut components = vector(layers.len(), context)?;
    for policy in layers.iter() {
        components.push(policy.components_with_storage(
            |count| vector(count, context),
            |count| vector(count, context),
        )?);
    }
    let mut segments = segments()?;
    if segments.is_empty() {
        return Err(StateError::EmptySegments);
    }
    // Equal keys have the same identity and range, so stability cannot affect
    // the duplicate/coverage result. In-place sorting needs no scratch owner.
    segments.sort_unstable_by(|left, right| {
        left.layers
            .start
            .cmp(&right.layers.start)
            .then_with(|| left.layers.end.cmp(&right.layers.end))
            .then_with(|| left.id.cmp(&right.id))
    });
    let mut frontier = 0usize;
    for (index, segment) in segments.iter().enumerate() {
        if segments[..index]
            .iter()
            .any(|earlier| earlier.id == segment.id)
        {
            return Err(StateError::DuplicateSegment {
                segment: identity(&segment.id, context)?,
            });
        }
        if segment.layers.end > layers.len() {
            return Err(StateError::SegmentOutOfBounds {
                segment: identity(&segment.id, context)?,
                start: segment.layers.start,
                end: segment.layers.end,
                layers: layers.len(),
            });
        }
        if segment.layers.start < frontier {
            return Err(StateError::OverlappingSegment {
                segment: identity(&segment.id, context)?,
                start: segment.layers.start,
                frontier,
            });
        }
        if segment.layers.start > frontier {
            return Err(StateError::UnassignedStateLayer { layer: frontier });
        }
        frontier = segment.layers.end;
    }
    if frontier != layers.len() {
        return Err(StateError::UnassignedStateLayer { layer: frontier });
    }
    Ok(StateLayout {
        layers,
        components,
        segments,
    })
}

// Shared ordinary/counted slice: validation and policy/segment ordering remain
// the existing StateLayout::slice semantics. No borrowed metadata is authority
// to construct another range or to issue any state/native allocation grant.
pub(super) fn slice(
    source: &StateLayout,
    layers: Range<usize>,
    context: Option<&WorkspaceContext>,
) -> Result<StateLayout, StateError> {
    if layers.is_empty() || layers.end > source.len() {
        return Err(StateError::InvalidLayoutSlice {
            start: layers.start,
            end: layers.end,
            layers: source.len(),
        });
    }
    if let Some(context) = context {
        context
            .charge_metadata(size_of::<(
                StateLayout,
                Result<StateLayout, StateError>,
                Vec<LayerCachePolicy>,
                Vec<StateSegmentSpec>,
                LayerSchedule<LayerCachePolicy>,
                LayerCachePolicy,
                StateSegmentSpec,
                Range<usize>,
                usize,
                usize,
            )>())
            .map_err(eredu_nn::Error::from)?;
    }
    let mut policies = vector(layers.len(), context)?;
    for policy in source.layers.iter().skip(layers.start).take(layers.len()) {
        if let Some(context) = context {
            context
                .charge_metadata(
                    StateLayout::policy_clone_payload_bytes(policy)
                        .map_err(eredu_nn::Error::from)?,
                )
                .map_err(eredu_nn::Error::from)?;
        }
        policies.push(policy.clone());
    }
    let segment_count = source
        .segments
        .iter()
        .filter(|segment| {
            segment.layers.start.max(layers.start) < segment.layers.end.min(layers.end)
        })
        .count();
    let mut segments = vector(segment_count, context)?;
    for segment in &source.segments {
        let start = segment.layers.start.max(layers.start);
        let end = segment.layers.end.min(layers.end);
        if start < end {
            let range = start - layers.start..end - layers.start;
            segments.push(match context {
                Some(context) => StateSegmentSpec::new_with_metadata(
                    segment.id.as_str(),
                    range,
                    segment.lifetime,
                    segment.processed_token_offset,
                    context,
                )?,
                None => StateSegmentSpec::new(
                    segment.id.as_str(),
                    range,
                    segment.lifetime,
                    segment.processed_token_offset,
                )?,
            });
        }
    }
    let layers = LayerSchedule::new(policies.len(), policies).map_err(|error| match context {
        Some(context) => match context.metadata_string(format_args!("{error}")) {
            Ok(reason) => StateError::InvalidResidency(reason),
            Err(cause) => StateError::from(cause),
        },
        None => StateError::InvalidResidency(error.to_string()),
    })?;
    segmented(layers, || Ok(segments), context)
}
