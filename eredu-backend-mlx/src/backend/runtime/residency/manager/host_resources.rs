//! Physical host-buffer identities follow allocations through aliases and transfers.
use super::*;
use eredu_core::{resources::*, Observed};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) fn identity(scope: &str) -> ResourceIdentity {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let key = NEXT
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("native residency resource identity exhausted");
    ResourceIdentity {
        scope: scope.into(),
        key: key.to_string(),
    }
}

/// One owned native allocation. Sharing this object preserves its identity;
/// rematerializing the same checkpoint selection creates a different allocation.
pub(super) struct ResidentHostAllocation {
    pub(super) buffer: ImmutableHostTransferBuffer,
    identity: ResourceIdentity,
    extent: ResourceExtent,
}
impl ResidentHostAllocation {
    pub(super) fn new(buffer: ImmutableHostTransferBuffer) -> Result<Self, ResidencyError> {
        let inspect = |value: safemlx::error::Result<usize>| -> Result<u64, ResidencyError> {
            u64::try_from(value.map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "host allocation resource metadata",
                source,
            })?)
            .map_err(|_| ResidencyError::ArithmeticOverflow {
                context: "host allocation extent",
            })
        };
        // This buffer is immutable and its transfer completed before publication.
        // Retain exact metadata so later reporting needs no native API or lock.
        let extent = ResourceExtent {
            payload: ResourceByteBounds::exact(inspect(buffer.nbytes())?),
            capacity: ResourceByteBounds::exact(inspect(buffer.capacity())?),
        };
        extent.validate()?;
        Ok(Self {
            buffer,
            extent,
            identity: identity("mlx.resident_host_allocation"),
        })
    }
}

impl ResidencyManager {
    /// Bounded snapshot of this calling thread's detached submissions, across
    /// managers. This does not reap, poll, acquire native locks or advance work.
    pub fn detached_storage_resources(
        limits: eredu_runtime::detached_resources::DetachedResourceLimits,
    ) -> Result<eredu_runtime::detached_resources::DetachedResourceInventory, ResidencyError> {
        Ok(crate::backend::submission_recovery::describe_resources(
            limits,
        )?)
    }

    /// Observes currently cached host allocations under one storage lock. No
    /// payload reads, evaluation, completion polling or native allocation occur.
    /// Eviction removes cache ownership, not independently retained transfers.
    pub fn host_storage_resources(&self) -> Result<ResourceDescription, ResidencyError> {
        let state = self.observation_lock()?;
        describe(
            self.inner.resource_identity.clone(),
            state.storage.values().filter_map(|s| s.host.as_deref()),
            None,
        )
    }
}

pub(super) fn describe<'a>(
    scope: ResourceIdentity,
    hosts: impl Iterator<Item = &'a ResidentHostBuffers>,
    transfer_owner: Option<&ResourceIdentity>,
) -> Result<ResourceDescription, ResidencyError> {
    let mut allocations: BTreeMap<ResourceIdentity, ResourceAllocation> = BTreeMap::new();
    for host in hosts {
        let usage = ResourceUse {
            owner: transfer_owner.unwrap_or(&host.owner).clone(),
            role: ResourceRole::Parameters,
        };
        for allocation in host.buffers.values() {
            if let Some(prior) = allocations.get_mut(&allocation.identity) {
                if !prior.uses.contains(&usage) {
                    prior.uses.push(usage.clone());
                }
                continue;
            }
            allocations.insert(
                allocation.identity.clone(),
                ResourceAllocation {
                    identity: allocation.identity.clone(),
                    uses: vec![usage.clone()],
                    placement: Observed::unavailable(
                        "host transfer storage has no bound physical capacity pool",
                    ),
                    size: ResourceSize::Fixed {
                        extent: allocation.extent.clone(),
                    },
                },
            );
        }
    }
    let resources = ResourceDescription {
        schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
        scope,
        context: ResourceContext::default(),
        horizon: ResourceContext::default(),
        coverage: ResourceCoverage::Partial { reasons: vec![
            "only retained native host-transfer buffers are described; device arrays, checkpoint caches, bookkeeping and materialization scratch are excluded".into(),
            if transfer_owner.is_some() {
                "cache ownership, destination leases, other transfers and detached recovery holdings are not inventoried here"
            } else {
                "detached transfer and external lease ownership are separate retention references"
            }.into(),
        ] },
        allocations: allocations.into_values().collect(),
    };
    resources.validate()?;
    Ok(resources)
}
