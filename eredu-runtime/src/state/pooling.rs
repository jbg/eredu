//! Shared declaration validation for local-key and append-only pooling state.

use eredu_core::cache::{
    LayerCachePolicy, PoolingStateComponent, StateTensorDimension, StateTensorPolicy,
    StateTensorPresence, StateTensorRole,
};
use std::fmt;

/// Validated geometry for the local-key and append-only pooling mechanism.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PoolingAttentionGeometry {
    /// Number of local source positions, including the current position.
    pub sliding_window: i32,
    /// Source positions represented by one output in each contiguous stream.
    pub stream_ratios: Vec<i32>,
}

/// Validates architecture-declared local keys and complete pooling components.
/// Native and metadata mechanisms consume this same role/ratio contract.
pub fn pooling_attention_geometry(
    layer: usize,
    policy: &LayerCachePolicy,
) -> Result<PoolingAttentionGeometry, eredu_nn::Error> {
    let plan =
        pooling_attention_geometry_plan(layer, policy, |args| eredu_nn::Error::backend(args))?;
    Ok(PoolingAttentionGeometry {
        sliding_window: plan.sliding_window,
        stream_ratios: plan.stream_ratios().to_vec(),
    })
}

/// Inline result of the shared pooling declaration validator. The existing
/// mechanism supports no streams, one non-overlapping stream, or two overlapping
/// streams. All supplied declarations are validated before that final check.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PoolingAttentionGeometryPlan {
    /// Number of local source positions, including the current position.
    pub sliding_window: i32,
    ratios: [i32; 2],
    streams: usize,
}
impl PoolingAttentionGeometryPlan {
    /// Ratios in contiguous stream order, borrowed without a temporary vector.
    pub fn stream_ratios(&self) -> &[i32] {
        &self.ratios[..self.streams]
    }
}

/// Validates the same pooling mechanism without allocating maps or result
/// vectors. The caller owns diagnostic construction; the callback receives only
/// deterministic borrowed formatting of the selected declarations.
pub fn pooling_attention_geometry_plan(
    layer: usize,
    policy: &LayerCachePolicy,
    mut error: impl FnMut(fmt::Arguments<'_>) -> eredu_nn::Error,
) -> Result<PoolingAttentionGeometryPlan, eredu_nn::Error> {
    let (attention, heads, tensors) = match policy {
        LayerCachePolicy::KeyOnly {
            attention,
            num_key_heads,
            ..
        } => (attention, num_key_heads, &[][..]),
        LayerCachePolicy::KeyOnlyWithFixedState {
            attention,
            num_key_heads,
            tensors,
            ..
        } => (attention, num_key_heads, tensors.as_slice()),
        _ => {
            return Err(error(format_args!(
                "pooling-attention state requires key-only policy at layer {layer}: {policy:?}"
            )));
        }
    };
    if heads.get() != 1 {
        return Err(error(format_args!(
            "pooling local keys require one head at layer {layer}"
        )));
    }
    let sliding_window = attention
        .sliding_window_i32()
        .map_err(|cause| error(format_args!("{cause}")))?
        .ok_or_else(|| {
            error(format_args!(
                "pooling-attention state requires a sliding window at layer {layer}"
            ))
        })?;
    // Preserve the ordinary validator's input-order non-pooling rejection and
    // sorted stream traversal, without building its two levels of maps.
    for tensor in tensors {
        if !matches!(tensor.role, StateTensorRole::Pooling { .. }) {
            return Err(error(format_args!(
                "pooling-attention state found non-pooling component at layer {layer}: {:?}",
                tensor.role
            )));
        }
    }
    let mut ratios = [0; 2];
    let mut overlaps = [false; 2];
    let mut count = 0usize;
    let mut previous = None;
    loop {
        let next = tensors
            .iter()
            .filter_map(|tensor| {
                let StateTensorRole::Pooling { stream, .. } = tensor.role else {
                    unreachable!()
                };
                previous
                    .is_none_or(|previous| stream > previous)
                    .then_some(stream)
            })
            .min();
        let Some(stream) = next else { break };
        if stream as usize != count {
            return Err(error(format_args!(
                "pooling-attention streams must be contiguous at layer {layer}, expected {count}, got {stream}"
            )));
        }
        let (ratio, overlapping) = pooling_stream_geometry(layer, stream, tensors, &mut error)?;
        if count < ratios.len() {
            ratios[count] = ratio;
            overlaps[count] = overlapping;
        }
        count += 1;
        previous = Some(stream);
    }
    if !(count == 0 || count == 1 && !overlaps[0] || count == 2 && overlaps == [true, true]) {
        return Err(error(format_args!(
            "pooling-attention stream overlap layout is unsupported at layer {layer}"
        )));
    }
    Ok(PoolingAttentionGeometryPlan {
        sliding_window,
        ratios,
        streams: count,
    })
}

fn pooling_stream_geometry(
    layer: usize,
    stream: u32,
    tensors: &[StateTensorPolicy],
    error: &mut impl FnMut(fmt::Arguments<'_>) -> eredu_nn::Error,
) -> Result<(i32, bool), eredu_nn::Error> {
    // BTreeMap::insert historically selected the last duplicate declaration.
    let find = |component| {
        tensors
            .iter()
            .rev()
            .find(|tensor| tensor.role == StateTensorRole::Pooling { stream, component })
    };
    let pooled = find(PoolingStateComponent::Pooled).ok_or_else(|| {
        error(format_args!(
            "pooling stream {stream} at layer {layer} is missing {:?}",
            PoolingStateComponent::Pooled
        ))
    })?;
    let ratio = match (pooled.shape.as_slice(), pooled.presence) {
        (
            [
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensDiv(shape_ratio),
                StateTensorDimension::Fixed(_),
            ],
            StateTensorPresence::PrefixAtLeast(presence_ratio),
        ) if shape_ratio == &presence_ratio => *shape_ratio,
        _ => {
            return Err(error(format_args!(
                "pooling stream {stream} at layer {layer} has invalid pooled geometry"
            )));
        }
    };
    for component in [
        PoolingStateComponent::PendingValues,
        PoolingStateComponent::PendingGates,
    ] {
        let pending = find(component).ok_or_else(|| {
            error(format_args!(
                "pooling stream {stream} at layer {layer} is missing {component:?}"
            ))
        })?;
        match (pending.shape.as_slice(), pending.presence) {
            (
                [
                    StateTensorDimension::Batch,
                    StateTensorDimension::PrefixTokensRem(shape_ratio),
                    StateTensorDimension::Fixed(_),
                ],
                StateTensorPresence::PrefixRemainderNonZero(presence_ratio),
            ) if shape_ratio == &ratio && presence_ratio == ratio => {}
            _ => {
                return Err(error(format_args!(
                    "pooling stream {stream} at layer {layer} has invalid {component:?} geometry"
                )));
            }
        }
    }
    let overlap_values = find(PoolingStateComponent::OverlapValues);
    let overlap_gates = find(PoolingStateComponent::OverlapGates);
    if overlap_values.is_some() != overlap_gates.is_some() {
        return Err(error(format_args!(
            "pooling stream {stream} at layer {layer} has incomplete overlap geometry"
        )));
    }
    for (component, overlap) in [
        (PoolingStateComponent::OverlapValues, overlap_values),
        (PoolingStateComponent::OverlapGates, overlap_gates),
    ] {
        let Some(overlap) = overlap else { continue };
        match (overlap.shape.as_slice(), overlap.presence) {
            (
                [
                    StateTensorDimension::Batch,
                    StateTensorDimension::Fixed(shape_ratio),
                    StateTensorDimension::Fixed(_),
                ],
                StateTensorPresence::PrefixAtLeast(presence_ratio),
            ) if shape_ratio == &ratio && presence_ratio == ratio => {}
            _ => {
                return Err(error(format_args!(
                    "pooling stream {stream} at layer {layer} has invalid {component:?} geometry"
                )));
            }
        }
    }
    let component_count = tensors.iter().enumerate().filter(|(index, tensor)| {
        matches!(tensor.role, StateTensorRole::Pooling { stream: candidate, .. } if candidate == stream)
            && !tensors[..*index].iter().any(|previous| previous.role == tensor.role)
    }).count();
    if component_count != 3 + usize::from(overlap_values.is_some()) * 2 {
        return Err(error(format_args!(
            "pooling stream {stream} at layer {layer} has undeclared components"
        )));
    }
    i32::try_from(ratio.get())
        .map(|ratio| (ratio, overlap_values.is_some()))
        .map_err(|_| error(format_args!("pooling ratio exceeds runtime range")))
}
