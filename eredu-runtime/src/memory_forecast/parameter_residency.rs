//! Managed parameter pools, distinct from transfers and operator workspace.

use super::*;

fn add(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_add(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "managed parameter capacity sum",
    })
}

/// Upper bounds for parameter copies charged to one selected residency manager.
/// Host capacities include transfer-buffer rounding; device bytes are payload,
/// excluding allocator overhead. Detached leases, source caches, materialization
/// scratch and parameter conversions require separate accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterResidencyMemoryBounds {
    residency: crate::LayerWeightResidency,
    host_capacity_bytes: u64,
    device_payload_bytes: u64,
}

impl ParameterResidencyMemoryBounds {
    /// Combines exact unit payloads and backend-proven host capacity bounds with
    /// the manager's enforced budgets. Include every physical owner once, even
    /// when execution visits only one group at a time. A working window is a
    /// minimum admission requirement, not a maximum cache occupancy.
    pub fn from_unit_capacities(
        units: impl IntoIterator<Item = (u64, u64)>,
        residency: crate::LayerWeightResidency,
    ) -> Result<Self, CapabilityError> {
        let (mut payload, mut capacity) = (0u64, 0u64);
        for (device, host) in units {
            if host < device {
                return Err(CapabilityError::InvalidConfiguration {
                    field: "parameter residency capacity",
                    detail: "host capacity is smaller than the unit payload".into(),
                });
            }
            payload = add(payload, device)?;
            capacity = add(capacity, host)?;
        }
        let limits =
            residency
                .offload()
                .map_err(|error| CapabilityError::InvalidConfiguration {
                    field: "parameter residency policy",
                    detail: error.to_string(),
                })?;
        Ok(Self {
            residency,
            host_capacity_bytes: if residency.is_fully_resident() {
                0
            } else {
                limits
                    .host_budget_bytes()
                    .map_or(capacity, |n| n.min(capacity))
            },
            device_payload_bytes: limits
                .device_budget_bytes()
                .map_or(payload, |n| n.min(payload)),
        })
    }

    /// Exact selected policy whose enforced limits justify these bounds.
    pub const fn residency(&self) -> crate::LayerWeightResidency {
        self.residency
    }

    /// Managed host copies, including native transfer-buffer capacity rounding.
    pub const fn host_capacity_bytes(&self) -> u64 {
        self.host_capacity_bytes
    }

    /// Managed device parameter payload, excluding conversions and allocator slack.
    pub const fn device_payload_bytes(&self) -> u64 {
        self.device_payload_bytes
    }
}
