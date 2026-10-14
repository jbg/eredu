//! Masked proposal geometry shares target parameters and keeps block K/V transient.
use super::*;
use crate::speculative::shared_backbone::{
    SharedBackboneExecutionPolicy, SharedDraftAdapterGeometry,
};

/// Invocation costs distinct from the shared model's retained parameter catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedBackboneMemoryPlan {
    /// Complete trained block, including positions beyond a shortened proposal budget.
    pub canvas_positions: u64,
    /// Selected draft-only additive work; absence means adapters are disabled for this request.
    pub adapter: Option<SharedDraftAdapterGeometry>,
}
impl SharedBackboneMemoryPlan {
    /// Projects immutable architecture declarations without reading weights or allocating state.
    pub fn from_policy(policy: SharedBackboneExecutionPolicy) -> Self {
        Self {
            canvas_positions: u64::from(policy.geometry().canvas_length.get()),
            adapter: policy.adapter_geometry(),
        }
    }
    pub(super) fn validate(&self, width: u64) -> Result<(), CapabilityError> {
        if self.canvas_positions < 2 || width >= self.canvas_positions {
            return Err(CapabilityError::InvalidConfiguration {
                field: "shared draft canvas",
                detail: "proposals must fit within the complete seeded masked invocation".into(),
            });
        }
        Ok(())
    }
    pub(super) fn adapter_workspace(&self, graphs: u64) -> Result<MemoryBytes, CapabilityError> {
        let Some(adapter) = self.adapter else {
            return Ok(MemoryBytes::exact(0));
        };
        let channels = add(
            u64::from(adapter.input.get()),
            add(
                u64::from(adapter.rank.get()),
                mul(2, u64::from(adapter.output.get()))?,
            )?,
        )?;
        let bytes = mul(
            mul(mul(self.canvas_positions, channels)?, 4)?,
            mul(u64::from(adapter.layers.get()), graphs)?,
        )?;
        Ok(MemoryBytes::estimated(
            0,
            bytes,
            "draft-only low-rank input promotion, rank activation, expansion and additive output; F32 envelope for all live layer graphs; base and adapter parameters already belong to target residency",
        ))
    }
}
