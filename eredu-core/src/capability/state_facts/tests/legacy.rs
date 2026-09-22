//! Independent pre-D2 state/dimension implementation, retained as a behavioral oracle.
//! Original bodies come from the exact archived D2 base; only the local function
//! receiver/call changes prevent the oracle from delegating to the new kernel.
use super::*;
use crate::cache::CachePolicyError;
fn checked_add(a: u64, b: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    a.checked_add(b)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}
fn checked_mul(a: u64, b: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    a.checked_mul(b)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}
fn resolved_shape(
    tensor: &StateTensorPolicy,
    batch_size: usize,
    prefix_tokens: usize,
) -> Result<Vec<i32>, CachePolicyError> {
    tensor
        .shape
        .iter()
        .map(|dimension| match dimension {
            StateTensorDimension::Batch => i32::try_from(batch_size),
            StateTensorDimension::PrefixTokens => i32::try_from(prefix_tokens),
            StateTensorDimension::PrefixTokensDiv(divisor) => {
                i32::try_from(prefix_tokens / divisor.get() as usize)
            }
            StateTensorDimension::PrefixTokensRem(divisor) => {
                i32::try_from(prefix_tokens % divisor.get() as usize)
            }
            StateTensorDimension::Fixed(value) => i32::try_from(value.get()),
            StateTensorDimension::Scalar => Ok(1),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            CachePolicyError::Invalid(
                "fixed-state tensor dimension exceeds runtime i32 range".into(),
            )
        })
}

fn attention_scalars_per_position(policy: &LayerCachePolicy) -> Result<u64, CapabilityError> {
    let scalars = match policy {
        LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyValueWithFixedState {
            num_key_value_heads,
            head_dim,
            ..
        } => checked_mul(
            checked_mul(
                u64::from(num_key_value_heads.get()),
                u64::from(head_dim.get()),
                "key/value heads times head dimension",
            )?,
            2,
            "key plus value scalars",
        )?,
        LayerCachePolicy::KeyOnly {
            num_key_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyOnlyWithFixedState {
            num_key_heads,
            head_dim,
            ..
        } => checked_mul(
            u64::from(num_key_heads.get()),
            u64::from(head_dim.get()),
            "key heads times head dimension",
        )?,
        LayerCachePolicy::CompressedLatentRotary {
            latent_dim,
            rotary_dim,
            ..
        } => checked_add(
            u64::from(latent_dim.get()),
            u64::from(rotary_dim.get()),
            "compressed latent plus rotary width",
        )?,
        LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => 0,
    };
    Ok(scalars)
}

fn is_context_dependent_dimension(dimension: &StateTensorDimension) -> bool {
    matches!(
        dimension,
        StateTensorDimension::PrefixTokens
            | StateTensorDimension::PrefixTokensDiv(_)
            | StateTensorDimension::PrefixTokensRem(_)
    )
}

fn state_tensor_dtype_bytes(tensor: &StateTensorPolicy, floating_scalar_bytes: u64) -> u64 {
    match tensor.dtype {
        StateTensorDtype::Floating => floating_scalar_bytes,
        StateTensorDtype::Float32 | StateTensorDtype::Int32 | StateTensorDtype::Uint32 => 4,
    }
}

fn state_tensor_is_present(tensor: &StateTensorPolicy, prefix_tokens: usize) -> bool {
    match tensor.presence {
        StateTensorPresence::Required => true,
        // Prepared prefix embeddings are accounted once through the input's
        // authoritative media-position count below. Any other optional state
        // is included conservatively because its request-time presence is not
        // otherwise represented in the portable input descriptor.
        StateTensorPresence::Optional => !matches!(tensor.role, StateTensorRole::PrefixEmbedding),
        StateTensorPresence::PrefixRemainderNonZero(divisor) => {
            !prefix_tokens.is_multiple_of(divisor.get() as usize)
        }
        StateTensorPresence::PrefixAtLeast(divisor) => prefix_tokens >= divisor.get() as usize,
    }
}

fn state_tensor_bytes(
    tensor: &StateTensorPolicy,
    batch_size: usize,
    prefix_tokens: usize,
    floating_scalar_bytes: u64,
) -> Result<u64, CapabilityError> {
    if !state_tensor_is_present(tensor, prefix_tokens) {
        return Ok(0);
    }
    let shape = resolved_shape(tensor, batch_size, prefix_tokens).map_err(|error| {
        CapabilityError::InvalidConfiguration {
            field: "layer_layout",
            detail: error.to_string(),
        }
    })?;
    let scalars = shape.into_iter().try_fold(1_u64, |scalars, dimension| {
        checked_mul(
            scalars,
            u64::try_from(dimension).map_err(|_| CapabilityError::InvalidConfiguration {
                field: "layer_layout",
                detail: "runtime state tensor has a negative resolved dimension".into(),
            })?,
            "runtime state tensor scalar count",
        )
    })?;
    checked_mul(
        scalars,
        state_tensor_dtype_bytes(tensor, floating_scalar_bytes),
        "runtime state tensor bytes",
    )
}

fn state_tensor_bytes_per_position_per_batch(
    tensor: &StateTensorPolicy,
    floating_scalar_bytes: u64,
) -> Result<u64, CapabilityError> {
    let mut scalars = 1_u64;
    let mut divisor = 1_u64;
    let mut unbounded = false;
    for dimension in &tensor.shape {
        match dimension {
            StateTensorDimension::Batch | StateTensorDimension::Scalar => {}
            StateTensorDimension::Fixed(value) => {
                scalars =
                    checked_mul(scalars, u64::from(value.get()), "state growth scalar count")?;
            }
            StateTensorDimension::PrefixTokens => unbounded = true,
            StateTensorDimension::PrefixTokensDiv(value) => {
                unbounded = true;
                divisor = checked_mul(divisor, u64::from(value.get()), "state growth divisor")?;
            }
            StateTensorDimension::PrefixTokensRem(_) => return Ok(0),
        }
    }
    if !unbounded {
        return Ok(0);
    }
    let bytes = checked_mul(
        scalars,
        state_tensor_dtype_bytes(tensor, floating_scalar_bytes),
        "state growth bytes",
    )?;
    Ok(bytes.div_ceil(divisor))
}

/// Estimates request state from exact executable layer policies.
pub fn estimate_runtime_state(
    layout: &StateMemoryLayout,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
    floating_state_dtype_bytes: NonZeroU8,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    if batch_size == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "batch_size",
            detail: "must be positive".into(),
        });
    }
    let requested_positions = checked_add(
        input.model_positions,
        max_output_tokens,
        "prompt plus output positions",
    )?;
    let floating_scalar_bytes = u64::from(floating_state_dtype_bytes.get());
    let batch_size_usize =
        usize::try_from(batch_size).map_err(|_| CapabilityError::InvalidConfiguration {
            field: "batch_size",
            detail: "exceeds the runtime state shape range".into(),
        })?;
    let mut fixed_state_bytes = 0;
    let mut context_state_bytes = 0;
    let mut unbounded_per_position = 0;
    let mut sliding_window_bounds = Vec::new();
    for (layer, policy) in layout.layer_layout.iter().enumerate() {
        let layer_positions = requested_positions
            .saturating_sub(u64::from(layout.layer_prefix_offsets[layer].unsigned_abs()));
        let layer_positions_usize = usize::try_from(layer_positions).map_err(|_| {
            CapabilityError::InvalidConfiguration {
                field: "requested_positions",
                detail: "exceeds the runtime state shape range".into(),
            }
        })?;
        if let Some(attention) = policy.attention() {
            let per_position = attention_scalars_per_position(policy)?;
            let retained = match attention {
                AttentionPolicy::Sliding { window } => {
                    let window = u64::from(window.get());
                    sliding_window_bounds.push(window);
                    layer_positions.min(window)
                }
                AttentionPolicy::Full => {
                    let adjustment = layout.allocation_granularity - 1;
                    checked_add(layer_positions, adjustment, "cache allocation rounding")?
                        / layout.allocation_granularity
                        * layout.allocation_granularity
                }
            };
            let bytes = checked_mul(
                checked_mul(
                    checked_mul(per_position, retained, "attention context scalars")?,
                    batch_size,
                    "attention context batch",
                )?,
                floating_scalar_bytes,
                "attention context bytes",
            )?;
            context_state_bytes =
                checked_add(context_state_bytes, bytes, "context state byte total")?;
            if matches!(attention, AttentionPolicy::Full) {
                unbounded_per_position = checked_add(
                    unbounded_per_position,
                    checked_mul(
                        per_position,
                        floating_scalar_bytes,
                        "unbounded bytes per position",
                    )?,
                    "unbounded bytes-per-position total",
                )?;
            }
        }
        for tensor in policy.fixed_state() {
            let bytes = state_tensor_bytes(
                tensor,
                batch_size_usize,
                layer_positions_usize,
                floating_scalar_bytes,
            )?;
            if tensor.shape.iter().any(is_context_dependent_dimension) {
                context_state_bytes =
                    checked_add(context_state_bytes, bytes, "context state byte total")?;
                unbounded_per_position = checked_add(
                    unbounded_per_position,
                    state_tensor_bytes_per_position_per_batch(tensor, floating_scalar_bytes)?,
                    "unbounded bytes-per-position total",
                )?;
            } else {
                fixed_state_bytes =
                    checked_add(fixed_state_bytes, bytes, "fixed state byte total")?;
            }
        }
    }
    sliding_window_bounds.sort_unstable();
    sliding_window_bounds.dedup();
    let multimodal_embedding_bytes = checked_mul(
        checked_mul(
            checked_mul(
                input.media_positions,
                layout.hidden_size,
                "media positions times hidden size",
            )?,
            batch_size,
            "media embeddings times batch",
        )?,
        floating_scalar_bytes,
        "media embedding bytes",
    )?;
    let media_execution_workspace_bytes = checked_mul(
        input.media_execution_workspace_bytes,
        batch_size,
        "media execution workspace times batch",
    )?;
    let requested_state_bytes = checked_add(
        checked_add(
            checked_add(
                fixed_state_bytes,
                context_state_bytes,
                "fixed plus context state",
            )?,
            multimodal_embedding_bytes,
            "persistent plus multimodal embedding state",
        )?,
        media_execution_workspace_bytes,
        "persistent plus media execution workspace",
    )?;
    // A state layout never bounds native text execution workspace. In
    // particular, a conservative media estimate cannot upgrade missing text
    // bounds to a complete estimate.
    let completeness = EstimationCompleteness::PersistentStateOnly;
    Ok(RuntimeStateEstimate {
        physical_domains: None,
        fixed_state_bytes,
        bytes_per_position_per_batch: unbounded_per_position,
        context_state_bytes,
        selected_state_backing: None,
        multimodal_embedding_bytes,
        media_execution_workspace_bytes,
        requested_state_bytes,
        execution_workspace: None,
        persistent_state_completeness: if layout.completeness
            == EstimationCompleteness::PersistentStateOnly
        {
            layout.completeness
        } else {
            match (input.kind, input.media_execution_workspace_kind) {
                (ObservationKind::Exact, ObservationKind::Exact) => layout.completeness,
                (
                    ObservationKind::Exact | ObservationKind::Conservative,
                    ObservationKind::Exact | ObservationKind::Conservative,
                ) => EstimationCompleteness::Conservative,
                _ => EstimationCompleteness::PersistentStateOnly,
            }
        },
        assumptions: StateMemoryAssumptions {
            floating_state_dtype_bytes,
            batch_size,
            requested_positions,
            sliding_window_bounds,
            allocation_granularity: layout.allocation_granularity,
        },
        completeness,
    })
}
