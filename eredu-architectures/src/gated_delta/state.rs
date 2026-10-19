//! Rank-local mutable geometry shared by all gated-delta families.
use eredu_core::cache::{
    LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
    StateTensorPolicy, StateTensorRole,
};
use eredu_nn::Error;

/// Head and convolution geometry; no checkpoint or native backend identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaStateGeometry {
    /// Query/key heads before expansion to value heads.
    pub key_heads: i32,
    /// Independent recurrent matrices.
    pub value_heads: i32,
    /// Rows in each recurrent matrix.
    pub key_dim: i32,
    /// Columns in each recurrent matrix.
    pub value_dim: i32,
    /// Number of causal convolution taps.
    pub kernel: i32,
    /// Distance between convolution taps.
    pub dilation: i32,
}

impl GatedDeltaStateGeometry {
    /// Fixed history and FP32 recurrence, with their ordinary residency policies.
    /// A one-tap convolution has no history component.
    pub fn state_policy(self) -> Result<LayerCachePolicy, Error> {
        if self.key_heads <= 0
            || self.value_heads <= 0
            || self.value_heads % self.key_heads != 0
            || self.key_dim <= 0
            || self.value_dim <= 0
            || self.kernel <= 0
            || self.dilation <= 0
        {
            return Err(Error::backend("invalid gated-delta state geometry"));
        }
        let width = self
            .key_heads
            .checked_mul(self.key_dim)
            .and_then(|key| key.checked_mul(2))
            .and_then(|key| {
                self.value_heads
                    .checked_mul(self.value_dim)
                    .and_then(|value| key.checked_add(value))
            })
            .ok_or_else(|| Error::backend("gated-delta state width overflowed"))?;
        let history = (self.kernel - 1)
            .checked_mul(self.dilation)
            .filter(|length| *length < i32::MAX)
            .ok_or_else(|| Error::backend("gated-delta history exceeds i32"))?;
        let fixed = |value| StateTensorDimension::fixed(value).map_err(Error::backend);
        let mut components = Vec::with_capacity(2);
        if history != 0 {
            components.push(
                StateTensorPolicy::new(
                    StateTensorRole::Convolution { slot: 0 },
                    vec![StateTensorDimension::Batch, fixed(history)?, fixed(width)?],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .map_err(Error::backend)?,
            );
        }
        components.push(
            StateTensorPolicy::new(
                StateTensorRole::Recurrent,
                vec![
                    StateTensorDimension::Batch,
                    fixed(self.value_heads)?,
                    fixed(self.key_dim)?,
                    fixed(self.value_dim)?,
                ],
                StateTensorDtype::Float32,
                MutableStateResidency::LayerScopedOffloadable,
            )
            .map_err(Error::backend)?,
        );
        LayerCachePolicy::fixed_only(components).map_err(Error::backend)
    }
}
