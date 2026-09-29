//! Architecture-owned persistent media position offset over ordinary fixed state.
use eredu_core::cache::{
    LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
    StateTensorPolicy, StateTensorRole,
};
use eredu_nn::{NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::RuntimeStateComponents;

/// Invalid persisted or requested rotary-position provenance.
#[derive(Debug, thiserror::Error)]
pub enum PositionError {
    /// Populated target state must retain the exact offset that authored its keys.
    #[error("populated qwen4_exp state has no persisted rotary position delta")]
    Missing,
    /// The offset uses exact signed I32 scalar storage.
    #[error("invalid qwen4_exp position delta geometry or scalar type")]
    Geometry,
    /// A continuation cannot reinterpret previously cached positions.
    #[error("qwen4_exp continuation changes its committed rotary position delta")]
    Changed,
    /// Checked causal-to-rotary position addition failed.
    #[error("qwen4_exp rotary position is negative or exceeds i32")]
    Overflow,
    /// The selected profile lacks the declared component.
    #[error(transparent)]
    State(#[from] eredu_runtime::StateError),
    /// Native integer transfer failed.
    #[error(transparent)]
    Tensor(#[from] eredu_nn::Error),
}

pub(crate) fn policy(mut policy: LayerCachePolicy) -> Result<LayerCachePolicy, eredu_nn::Error> {
    let tensor = StateTensorPolicy::new(
        StateTensorRole::PositionDelta,
        vec![StateTensorDimension::Scalar],
        StateTensorDtype::Int32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .map_err(eredu_nn::Error::backend)?;
    match &mut policy {
        LayerCachePolicy::FixedState { tensors }
        | LayerCachePolicy::KeyValueWithState { tensors, .. } => tensors.push(tensor),
        _ => {
            return Err(eredu_nn::Error::backend(
                "target ingress owner lacks its fixed state profile",
            ))
        }
    }
    policy.validate().map_err(eredu_nn::Error::backend)?;
    Ok(policy)
}

pub(crate) fn resolve<B: NeuralBackend, S: RuntimeStateComponents<B>>(
    state: &mut S,
    offset: i32,
    requested: Option<i32>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<i32, PositionError> {
    let stored = state.fixed_component(StateTensorRole::PositionDelta)?;
    let previous = match stored {
        Some(value) => {
            if value.shape() != [1] || value.element_type() != Some(TensorElementType::I32) {
                return Err(PositionError::Geometry);
            }
            let values = value.to_i32_vec(context)?;
            if values.len() != 1 {
                return Err(PositionError::Geometry);
            }
            values[0]
        }
        None if offset == 0 => 0,
        None => return Err(PositionError::Missing),
    };
    if offset > 0 && requested.is_some_and(|next| next != previous) {
        return Err(PositionError::Changed);
    }
    Ok(requested.unwrap_or(previous))
}

pub(crate) fn rotary_offset(offset: i32, delta: i32) -> Result<i32, PositionError> {
    offset
        .checked_add(delta)
        .filter(|v| *v >= 0)
        .ok_or(PositionError::Overflow)
}

pub(crate) fn commit<B: NeuralBackend, S: RuntimeStateComponents<B>>(
    state: &mut S,
    delta: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(), PositionError> {
    let next = B::Tensor::from_i32_slice(&[delta], &[1], context)?;
    if next.shape() != [1] || next.element_type() != Some(TensorElementType::I32) {
        return Err(PositionError::Geometry);
    }
    *state.fixed_component(StateTensorRole::PositionDelta)? = Some(next);
    Ok(())
}
