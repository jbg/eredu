//! Parameter-loading payloads retained outside managed cache occupancy.

use super::*;

/// Source ownership and transfer/materialization payload envelopes. Native
/// operator scratch and allocator/reader overhead remain separate contributions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterLoadingMemoryProfile {
    /// Immutable host payload owned by retained sources, independently of leases.
    pub source_payload: MemoryBytes,
    /// Lease-owned encoded reads, host indices and transfer preparation payloads.
    pub host_workspace: MemoryBytes,
    /// Source-stream intermediate arrays and destination transfer payloads.
    pub execution_workspace: MemoryBytes,
}

impl ParameterLoadingMemoryProfile {
    /// Bounds all outstanding materializations using the manager's reservations,
    /// not merely its minimum forward window. A unit's workspace must cover its
    /// complete retained read/recipe graph until its transfer lease is released.
    /// Every unit declares device payload, host capacity, host workspace and
    /// execution workspace. Aliases belong only to their physical owner.
    pub fn from_unit_workspaces(
        units: impl IntoIterator<Item = (u64, u64, u64, u64)>,
        residency: crate::LayerWeightResidency,
        source_payload: MemoryBytes,
    ) -> Result<Self, CapabilityError> {
        source_payload.validate()?;
        let units = units.into_iter().collect::<Vec<_>>();
        let bounds = ParameterResidencyMemoryBounds::from_unit_capacities(
            units.iter().map(|&(device, host, _, _)| (device, host)),
            residency,
        )?;
        for &(device, _, host_workspace, execution_workspace) in &units {
            if device == 0 && (host_workspace != 0 || execution_workspace != 0) {
                return Err(CapabilityError::InvalidConfiguration {
                    field: "parameter loading workspace",
                    detail: "nonempty workspace has no reserved physical owner".into(),
                });
            }
        }
        // Both host- and device-tier reservations can own live source graphs.
        // The fractional relaxation allows the final unit to be partially filled,
        // so it bounds every subset of complete units, including uneven sizes.
        let envelope = |workspace: fn(&(u64, u64, u64, u64)) -> u64| {
            let device = fractional_envelope(
                units.iter().map(|u| (u.0, workspace(u))),
                bounds.device_payload_bytes(),
            )?;
            let host = fractional_envelope(
                units.iter().map(|u| (u.1, workspace(u))),
                bounds.host_capacity_bytes(),
            )?;
            device
                .checked_add(host)
                .ok_or(CapabilityError::ArithmeticOverflow {
                    operation: "parameter loading tier overlap",
                })
        };
        Ok(Self {
            source_payload,
            host_workspace: MemoryBytes::estimated(0, envelope(|u| u.2)?,
                "encoded reads and host recipe payloads covered by outstanding host/device reservations"),
            execution_workspace: MemoryBytes::estimated(0, envelope(|u| u.3)?,
                "native recipe and transfer payloads covered by outstanding host/device reservations"),
        })
    }

    pub(super) fn validate(&self) -> Result<(), CapabilityError> {
        self.source_payload.validate()?;
        self.host_workspace.validate()?;
        self.execution_workspace.validate()
    }

    pub(super) fn staging(
        &self,
        host: bool,
        execution: bool,
    ) -> Result<MemoryBytes, CapabilityError> {
        let mut result = MemoryBytes::exact(0);
        if host {
            result = self.source_payload.add(&self.host_workspace)?;
        }
        if execution {
            result = result.add(&self.execution_workspace)?;
        }
        result.detail = "retained checkpoint-source payload and reservation-bounded materialization/transfer payload; native operator scratch and allocator/reader overhead excluded".into();
        Ok(result)
    }
}

fn fractional_envelope(
    units: impl Iterator<Item = (u64, u64)>,
    mut budget: u64,
) -> Result<u64, CapabilityError> {
    let mut units = units.filter(|&(bytes, _)| bytes != 0).collect::<Vec<_>>();
    units.sort_unstable_by(|&(a, aw), &(b, bw)| {
        (u128::from(bw) * u128::from(a)).cmp(&(u128::from(aw) * u128::from(b)))
    });
    let mut total = 0u64;
    for (bytes, workspace) in units {
        let used = budget.min(bytes);
        let contribution = (u128::from(workspace) * u128::from(used)).div_ceil(u128::from(bytes));
        total =
            total
                .checked_add(contribution as u64)
                .ok_or(CapabilityError::ArithmeticOverflow {
                    operation: "parameter loading workspace envelope",
                })?;
        budget -= used;
        if budget == 0 {
            break;
        }
    }
    Ok(total)
}
