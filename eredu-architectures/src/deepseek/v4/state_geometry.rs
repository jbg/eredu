//! The actual V4 cache-policy equations with shared ordinary/counted storage.
use super::*;
use crate::state_geometry::Destination;
pub(super) fn layout<D: Destination>(args: &V4Args, out: &D) -> Result<StateLayout, D::Error> {
    out.controls::<(
        StateLayout,
        &V4Args,
        Vec<LayerCachePolicy>,
        Vec<StateSegmentSpec>,
    )>()?;
    args.validate_with_diagnostic(|message| out.error(message))?;
    let layers = usize::try_from(
        args.num_hidden_layers
            .checked_add(args.num_nextn_predict_layers)
            .ok_or_else(|| out.error(format_args!("V4 total state layer count overflowed")))?,
    )
    .map_err(|cause| out.error(format_args!("{cause}")))?;
    let attention = AttentionPolicy::sliding(
        u32::try_from(args.sliding_window).map_err(|cause| out.error(format_args!("{cause}")))?,
    )
    .map_err(|cause| out.error(format_args!("{cause}")))?;
    let policies = out.collect((0..layers).map(|layer| {
        let mut fixed = match args.attention_policy(layer) {
            Some(V4AttentionPolicy::Local) => out.vector(0)?,
            Some(V4AttentionPolicy::Compressed { ratio }) => {
                let mut tensors = pooling_stream(0, ratio, args.head_dim, ratio == 4, out)?;
                if ratio == 4 {
                    for tensor in pooling_stream(1, ratio, args.index_head_dim, true, out)? {
                        out.push(&mut tensors, tensor)?;
                    }
                }
                tensors
            }
            None => return Err(out.error(format_args!("missing V4 layer policy {layer}"))),
        };
        if fixed.is_empty() {
            LayerCachePolicy::key_only_with_diagnostic(attention, 1, args.head_dim, |message| {
                out.error(message)
            })
        } else {
            LayerCachePolicy::key_only_with_fixed_state_with_diagnostic(
                attention,
                1,
                args.head_dim,
                std::mem::take(&mut fixed),
                |message| out.error(message),
            )
        }
    }))?;
    let target_layers = usize::try_from(args.num_hidden_layers)
        .map_err(|cause| out.error(format_args!("{cause}")))?;
    let mut segments = out.vector(1 + usize::from(layers > target_layers))?;
    segments.push(out.segment(
        super::super::TARGET_STATE_SEGMENT,
        0..target_layers,
        StateSegmentLifetime::Persistent,
        0,
    )?);
    if layers > target_layers {
        segments.push(out.segment(
            super::super::PREDICTION_STATE_SEGMENT,
            target_layers..layers,
            StateSegmentLifetime::Persistent,
            if args.dspark.is_some() { 0 } else { -1 },
        )?);
    }
    out.segmented(out.schedule(layers, policies)?, segments)
}
fn pooling_stream<D: Destination>(
    stream: u32,
    ratio: i32,
    pooled_width: i32,
    overlapping: bool,
    out: &D,
) -> Result<Vec<StateTensorPolicy>, D::Error> {
    out.controls::<(
        Vec<StateTensorPolicy>,
        StateTensorPolicy,
        Vec<StateTensorDimension>,
        NonZeroU32,
    )>()?;
    let ratio =
        NonZeroU32::new(u32::try_from(ratio).map_err(|cause| out.error(format_args!("{cause}")))?)
            .ok_or_else(|| out.error(format_args!("V4 pooling ratio must be positive")))?;
    let source_width = if overlapping {
        pooled_width
            .checked_mul(2)
            .ok_or_else(|| out.error(format_args!("V4 pooling source width overflowed")))?
    } else {
        pooled_width
    };
    let role = |component| StateTensorRole::Pooling { stream, component };
    let pending = |component| {
        out.tensor(
            role(component),
            out.values([
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensRem(ratio),
                out.fixed(source_width)?,
            ])?,
            StateTensorDtype::Floating,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .map(|policy| policy.when_prefix_remainder_nonzero(ratio))
    };
    let mut tensors = out.vector(if overlapping { 5 } else { 3 })?;
    tensors.push(pending(PoolingStateComponent::PendingValues)?);
    tensors.push(pending(PoolingStateComponent::PendingGates)?);
    tensors.push(
        StateTensorPolicy::new_with_residency_and_diagnostic(
            role(PoolingStateComponent::Pooled),
            out.values([
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensDiv(ratio),
                out.fixed(pooled_width)?,
            ])?,
            StateTensorDtype::Floating,
            StateResidencyClass::SealablePaged,
            |message| out.error(message),
        )?
        .when_prefix_at_least(ratio),
    );
    if overlapping {
        for component in [
            PoolingStateComponent::OverlapValues,
            PoolingStateComponent::OverlapGates,
        ] {
            tensors.push(
                out.tensor(
                    role(component),
                    out.values([
                        StateTensorDimension::Batch,
                        StateTensorDimension::Fixed(ratio),
                        out.fixed(pooled_width)?,
                    ])?,
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )?
                .when_prefix_at_least(ratio),
            );
        }
    }
    Ok(tensors)
}
