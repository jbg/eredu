//! Rank-local mutable geometry shared by all gated-delta families.
use eredu_core::cache::{
    LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
    StateTensorPolicy, StateTensorRole,
};
use eredu_nn::{Error, GatedDeltaHeadGeometry};

/// Head and convolution geometry; no checkpoint or native backend identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaStateGeometry {
    /// Checked neural axes shared with parameter construction.
    pub heads: GatedDeltaHeadGeometry,
    /// Number of causal convolution taps.
    pub kernel: i32,
    /// Distance between convolution taps.
    pub dilation: i32,
}

impl GatedDeltaStateGeometry {
    /// Fixed history and FP32 recurrence, with their ordinary residency policies.
    /// A one-tap convolution has no history component.
    pub fn state_policy(self) -> Result<LayerCachePolicy, Error> {
        if self.kernel <= 0 || self.dilation <= 0 {
            return Err(Error::backend("invalid gated-delta state geometry"));
        }
        let width = self.heads.qkv_width()?;
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
                    fixed(self.heads.value_heads())?,
                    fixed(self.heads.key_dimensions())?,
                    fixed(self.heads.value_dimensions())?,
                ],
                StateTensorDtype::Float32,
                MutableStateResidency::LayerScopedOffloadable,
            )
            .map_err(Error::backend)?,
        );
        LayerCachePolicy::fixed_only(components).map_err(Error::backend)
    }
}
