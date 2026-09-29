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

#[cfg(test)]
mod tests {
    use super::*;
    fn geometry() -> GatedDeltaStateGeometry {
        GatedDeltaStateGeometry {
            key_heads: 2,
            value_heads: 6,
            key_dim: 3,
            value_dim: 5,
            kernel: 4,
            dilation: 3,
        }
    }
    #[test]
    fn state_shapes_preserve_grouped_heads_precision_and_dilated_history() {
        let policy = geometry().state_policy().unwrap();
        let state = policy.fixed_state();
        let fixed = |n| StateTensorDimension::fixed(n).unwrap();
        assert_eq!(state[0].role, StateTensorRole::Convolution { slot: 0 });
        assert_eq!(
            state[0].shape,
            vec![StateTensorDimension::Batch, fixed(9), fixed(42)]
        );
        assert_eq!(state[0].dtype, StateTensorDtype::Floating);
        assert_eq!(state[1].role, StateTensorRole::Recurrent);
        assert_eq!(
            state[1].shape,
            vec![StateTensorDimension::Batch, fixed(6), fixed(3), fixed(5)]
        );
        assert_eq!(state[1].dtype, StateTensorDtype::Float32);
        let one = GatedDeltaStateGeometry {
            kernel: 1,
            ..geometry()
        }
        .state_policy()
        .unwrap();
        assert_eq!(one.fixed_state().len(), 1);
        assert_eq!(one.fixed_state()[0].role, StateTensorRole::Recurrent);
    }
    #[test]
    fn malformed_or_overflowing_state_fails_before_allocation() {
        for g in [
            GatedDeltaStateGeometry {
                key_heads: 0,
                ..geometry()
            },
            GatedDeltaStateGeometry {
                value_heads: 5,
                ..geometry()
            },
            GatedDeltaStateGeometry {
                key_dim: i32::MAX,
                ..geometry()
            },
            GatedDeltaStateGeometry {
                value_dim: 0,
                ..geometry()
            },
            GatedDeltaStateGeometry {
                kernel: 0,
                ..geometry()
            },
            GatedDeltaStateGeometry {
                dilation: 0,
                ..geometry()
            },
            GatedDeltaStateGeometry {
                dilation: i32::MAX,
                ..geometry()
            },
        ] {
            assert!(g.state_policy().is_err(), "accepted {g:?}");
        }
    }
}
