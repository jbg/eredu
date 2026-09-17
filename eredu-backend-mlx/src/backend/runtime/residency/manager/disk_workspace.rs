//! Prepared direct disk reads and cold facts for their exact owner dispatch.
//!
//! Header/source admission happens once in `prepare_disk_reads`, during loading.
//! Candidate inspection reuses that metadata without touching the source store.
//! No descriptor retains a native array or pins a device residency window.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Weak, atomic::Ordering},
};

use eredu_checkpoint::store::CheckpointSource;
use eredu_core::residency::{MemoryTier, OffloadUnitId};
use eredu_runtime::{OffloadUnit, WeightBinding, working_memory::WorkingMemoryError};
use safemlx::{Array, ArrayAllocationInfo, Dtype, Stream};

use super::{ManagerWeak, ResidencyError, ResidencyManager, transfer::ManagerState};
use crate::backend::{
    nn::workspace::MetalAllocationFacts,
    runtime::checkpoint::recipe::{PreparedDirectReadError, PreparedDirectReadPlan},
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum DiskCopyWorkspaceError {
    #[error("disk-copy workspace is unavailable: {reason}")]
    Unproved {
        reason: &'static str,
        #[source]
        source: WorkingMemoryError,
    },
    #[error("disk-copy residency inspection failed: {0}")]
    Residency(#[source] ResidencyError),
    #[error("disk-copy direct read plan failed: {0}")]
    Direct(#[from] PreparedDirectReadError),
    #[error("disk-copy native metadata inspection failed: {0}")]
    Metadata(#[source] safemlx::ArrayMetadataError),
    #[error("disk-copy stream inspection failed: {0}")]
    Native(#[from] safemlx::error::Exception),
}
impl DiskCopyWorkspaceError {
    fn unknown(reason: &'static str) -> Self {
        Self::Unproved {
            reason,
            source: WorkingMemoryError::UnknownBound,
        }
    }
    fn mismatch(reason: &'static str) -> Self {
        Self::Unproved {
            reason,
            source: WorkingMemoryError::IdentityMismatch,
        }
    }
}
fn add(a: u64, b: u64) -> Result<u64, DiskCopyWorkspaceError> {
    a.checked_add(b).ok_or(DiskCopyWorkspaceError::Unproved {
        reason: "capacity sum overflow",
        source: WorkingMemoryError::Overflow,
    })
}

/// Explicit physical owner, also present for owners in another residency unit.
/// Its unit must remain available whenever this binding is consumed; it is not
/// an extra destination allocation in the direct read dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DiskCopyOwner {
    unit: OffloadUnitId,
    name: String,
}
impl DiskCopyOwner {
    pub(crate) fn unit(&self) -> &OffloadUnitId {
        &self.unit
    }
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone)]
pub(crate) struct PreparedDiskReadUnit {
    definition: OffloadUnit,
    source: eredu_checkpoint::store::CheckpointSourceIdentity,
    direct: PreparedDirectReadPlan,
    owners: BTreeMap<String, DiskCopyOwner>,
    aliases: BTreeMap<String, AliasGeometry>,
}

#[derive(Debug, Clone)]
struct AliasGeometry {
    shape: Vec<i32>,
    dtype: Dtype,
    bytes: u64,
}
impl std::fmt::Debug for PreparedDiskReadUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedDiskReadUnit")
            .field("definition", &self.definition)
            .field("direct", &self.direct)
            .field("owners", &self.owners)
            .field("aliases", &self.aliases)
            .finish()
    }
}
impl PreparedDiskReadUnit {
    pub(crate) fn id(&self) -> &OffloadUnitId {
        self.definition.id()
    }
    pub(crate) fn definition(&self) -> &OffloadUnit {
        &self.definition
    }
    pub(crate) fn direct(&self) -> &PreparedDirectReadPlan {
        &self.direct
    }
    pub(crate) fn owner(&self, name: &str) -> Option<&DiskCopyOwner> {
        self.owners.get(name)
    }

    /// Executes this exact retained plan. External aliases are supplied by name
    /// from their canonical owner units; local aliases reuse the freshly read
    /// owner. No alias name is independently read or materialized. The caller
    /// must retain the supplied arrays and all callback values in armed native
    /// recovery through exact completion, and enforce residency/source custody.
    /// Shared values must be the manager's validated canonical owner arrays,
    /// covered by the admitted owner envelope. Geometry checks below reject
    /// wrong bindings but do not manufacture physical-owner provenance.
    pub(crate) fn materialize(
        &self,
        shared: &BTreeMap<String, Array>,
        source_stream: &Stream,
        execution_stream: &Stream,
        mut retain: impl FnMut(&Array),
    ) -> Result<BTreeMap<String, Array>, DiskCopyWorkspaceError> {
        let expected = self
            .owners
            .iter()
            .filter(|(_, owner)| owner.unit() != self.id())
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();
        if expected != shared.keys().map(String::as_str).collect() {
            return Err(DiskCopyWorkspaceError::mismatch(
                "external alias names differ from retained owner dispatch",
            ));
        }
        self.validate_external_values(shared)?;
        // Retain borrowed owners before the first fallible read/copy operation.
        for value in shared.values() {
            retain(value);
        }
        let mut arrays = self
            .direct
            .materialize(source_stream, execution_stream, &mut retain)?;
        for (name, value) in shared {
            arrays.insert(name.clone(), value.clone());
        }
        self.bind_local_aliases(&mut arrays, &mut retain)?;
        Ok(arrays)
    }

    /// Same admitted direct-read worker before external alias publication.
    /// The caller must bind and validate every external owner from the complete
    /// canonical batch before publication; this method is not a full plan result.
    pub(super) fn materialize_canonical(
        &self,
        source_stream: &Stream,
        execution_stream: &Stream,
        mut retain: impl FnMut(&Array),
    ) -> Result<BTreeMap<String, Array>, DiskCopyWorkspaceError> {
        let mut arrays = self
            .direct
            .materialize(source_stream, execution_stream, &mut retain)?;
        self.bind_local_aliases(&mut arrays, &mut retain)?;
        Ok(arrays)
    }

    /// Borrowed sink companion for a prepared canonical destination. Local
    /// and external aliases are attached together after every unit is filled.
    pub(super) fn materialize_canonical_into<
        E: From<crate::backend::runtime::checkpoint::recipe::PreparedDirectReadError>,
    >(
        &self,
        source_stream: &Stream,
        execution_stream: &Stream,
        retain: impl FnMut(&Array),
        publish: impl FnMut(&str, Array) -> Result<(), E>,
    ) -> Result<(), E> {
        self.direct
            .materialize_into(source_stream, execution_stream, retain, publish)
    }

    fn bind_local_aliases(
        &self,
        arrays: &mut BTreeMap<String, Array>,
        mut retain: impl FnMut(&Array),
    ) -> Result<(), DiskCopyWorkspaceError> {
        for binding in self
            .definition
            .bindings()
            .iter()
            .filter(|binding| binding.is_alias())
        {
            let owner = &self.owners[binding.name()];
            if owner.unit() == self.id() {
                let value = arrays
                    .get(owner.name())
                    .ok_or_else(|| {
                        DiskCopyWorkspaceError::mismatch("local canonical owner is absent")
                    })?
                    .clone();
                retain(&value);
                arrays.insert(binding.name().to_owned(), value);
            }
        }
        Ok(())
    }

    /// Same external geometry checks, usable after all canonical unit outputs
    /// exist. The full ordinary adapter separately validates its exact key set.
    pub(super) fn validate_external_values(
        &self,
        arrays: &BTreeMap<String, Array>,
    ) -> Result<(), DiskCopyWorkspaceError> {
        self.validate_external_lookup(|name| arrays.get(name))
    }

    pub(super) fn validate_external_lookup<'a>(
        &self,
        lookup: impl Fn(&str) -> Option<&'a Array>,
    ) -> Result<(), DiskCopyWorkspaceError> {
        for (name, owner) in self
            .owners
            .iter()
            .filter(|(_, owner)| owner.unit() != self.id())
        {
            let _ = owner;
            let value = lookup(name).ok_or_else(|| {
                DiskCopyWorkspaceError::mismatch(
                    "external alias names differ from retained owner dispatch",
                )
            })?;
            let expected = &self.aliases[name];
            let actual = value
                .try_metadata_snapshot()
                .map_err(DiskCopyWorkspaceError::Metadata)?;
            if actual.shape() != expected.shape
                || actual.dtype() != expected.dtype
                || actual.nbytes() as u64 != expected.bytes
            {
                return Err(DiskCopyWorkspaceError::mismatch(
                    "external alias geometry differs from canonical read output",
                ));
            }
        }
        Ok(())
    }
}

/// Immutable plans include the transitive canonical-owner unit closure. Weak
/// source identities do not extend source or wrapper payload lifetimes. The
/// encoded reads retain their own admitted-file and read metadata independently.
#[derive(Debug, Clone)]
pub(crate) struct PreparedDiskReadPlans {
    manager: ManagerWeak,
    requested: Vec<OffloadUnitId>,
    units: Vec<PreparedDiskReadUnit>,
}
impl PreparedDiskReadPlans {
    pub(crate) fn requested_units(&self) -> &[OffloadUnitId] {
        &self.requested
    }
    pub(crate) fn units(&self) -> &[PreparedDiskReadUnit] {
        &self.units
    }
    pub(crate) fn unit(&self, id: &OffloadUnitId) -> Option<&PreparedDiskReadUnit> {
        self.units.iter().find(|unit| unit.id() == id)
    }

    fn validate_locked(
        &self,
        manager: &ResidencyManager,
        state: &ManagerState,
    ) -> Result<(), DiskCopyWorkspaceError> {
        if !self.manager.ptr_eq(&manager.inner.downgrade()) {
            return Err(DiskCopyWorkspaceError::mismatch(
                "prepared reads belong to another residency manager",
            ));
        }
        for unit in &self.units {
            if state.control.unit(unit.id()) != Some(unit.definition()) {
                return Err(DiskCopyWorkspaceError::mismatch(
                    "unit definition changed after direct read preparation",
                ));
            }
            let source = manager.inner.sources.retained(unit.id()).ok_or_else(|| {
                DiskCopyWorkspaceError::mismatch("source is an immutable prepared host catalog")
            })?;
            if unit.source != source.identity() {
                return Err(DiskCopyWorkspaceError::mismatch(
                    "unit source differs from retained admitted read source",
                ));
            }
            for binding in unit.definition.bindings() {
                let expected = state
                    .control
                    .binding_owner(unit.id(), binding)
                    .map(|(id, binding)| DiskCopyOwner {
                        unit: id.clone(),
                        name: binding.name().to_owned(),
                    })
                    .unwrap_or_else(|| DiskCopyOwner {
                        unit: unit.id().clone(),
                        name: binding.name().to_owned(),
                    });
                if unit.owner(binding.name()) != Some(&expected) {
                    return Err(DiskCopyWorkspaceError::mismatch(
                        "canonical alias owner changed",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DiskCopyBinding {
    binding: WeightBinding,
    owner: DiskCopyOwner,
    shape: Vec<i32>,
    dtype: Dtype,
    output_capacity_bytes: u64,
    current_allocation: Option<ArrayAllocationInfo>,
}
impl DiskCopyBinding {
    pub(crate) fn name(&self) -> &str {
        self.binding.name()
    }
    pub(crate) fn binding(&self) -> &WeightBinding {
        &self.binding
    }
    pub(crate) fn owner(&self) -> &DiskCopyOwner {
        &self.owner
    }
    pub(crate) fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub(crate) fn logical_bytes(&self) -> u64 {
        self.binding.expected_bytes()
    }
    pub(crate) fn output_capacity_bytes(&self) -> u64 {
        self.output_capacity_bytes
    }
    pub(crate) fn current_allocation(&self) -> Option<ArrayAllocationInfo> {
        self.current_allocation
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DiskCopyUnit {
    definition: OffloadUnit,
    bindings: Vec<DiskCopyBinding>,
    fresh_capacity_bytes: u64,
    current_device: bool,
}
impl DiskCopyUnit {
    pub(crate) fn id(&self) -> &OffloadUnitId {
        self.definition.id()
    }
    pub(crate) fn definition(&self) -> &OffloadUnit {
        &self.definition
    }
    pub(crate) fn bindings(&self) -> &[DiskCopyBinding] {
        &self.bindings
    }
    /// All owner outputs of a future direct refill, even if currently warm.
    pub(crate) fn fresh_capacity_bytes(&self) -> u64 {
        self.fresh_capacity_bytes
    }
    pub(crate) fn currently_on_device(&self) -> bool {
        self.current_device
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DiskCopyWorkspace {
    plans: PreparedDiskReadPlans,
    allocation: MetalAllocationFacts,
    units: Vec<DiskCopyUnit>,
    fresh_capacity_bytes: u64,
    current_device_bytes: u64,
}
impl DiskCopyWorkspace {
    pub(crate) fn plans(&self) -> &PreparedDiskReadPlans {
        &self.plans
    }
    pub(crate) fn units(&self) -> &[DiskCopyUnit] {
        &self.units
    }
    pub(crate) fn requested_units(&self) -> &[OffloadUnitId] {
        self.plans.requested_units()
    }
    /// Sum includes dependency owners outside the requested residency windows.
    /// Window schedulers must account for that explicit owner closure separately.
    pub(crate) fn fresh_capacity_bytes(&self) -> u64 {
        self.fresh_capacity_bytes
    }
    pub(crate) fn current_device_bytes(&self) -> u64 {
        self.current_device_bytes
    }
    pub(crate) fn host_staging_bytes(&self) -> u64 {
        0
    }
    pub(crate) fn validate_sources(
        &self,
        manager: &ResidencyManager,
    ) -> Result<(), DiskCopyWorkspaceError> {
        let current = manager.disk_copy_workspace(&self.plans, self.allocation)?;
        if current.units != self.units {
            return Err(DiskCopyWorkspaceError::mismatch(
                "disk source readiness or device backing changed after snapshot",
            ));
        }
        Ok(())
    }
}

impl ResidencyManager {
    /// Loading-time preparation may admit checkpoint headers and metadata. It
    /// performs no payload reads, native work, residency acquisition or reaping.
    /// Candidates must subsequently call `disk_copy_workspace` with these plans.
    pub(crate) fn prepare_disk_reads(
        &self,
        ids: &[OffloadUnitId],
    ) -> Result<PreparedDiskReadPlans, DiskCopyWorkspaceError> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| DiskCopyWorkspaceError::Residency(ResidencyError::StatePoisoned))?;
        let mut seen = BTreeSet::new();
        let mut pending = Vec::new();
        for id in ids {
            if !seen.insert(id.clone()) {
                return Err(DiskCopyWorkspaceError::mismatch("duplicate requested unit"));
            }
            pending.push(id.clone());
        }
        let mut units = Vec::new();
        let mut index = 0;
        while index < pending.len() {
            let id = pending[index].clone();
            let definition = state.control.unit(&id).ok_or_else(|| {
                DiskCopyWorkspaceError::mismatch("unknown requested or dependency unit")
            })?;
            let mut owners = BTreeMap::new();
            for binding in definition.bindings() {
                let (owner_id, owner_name) = if binding.is_alias() {
                    let (owner_id, owner) =
                        state.control.binding_owner(&id, binding).ok_or_else(|| {
                            DiskCopyWorkspaceError::mismatch("alias has no canonical owner")
                        })?;
                    (owner_id.clone(), owner.name().to_owned())
                } else {
                    (id.clone(), binding.name().to_owned())
                };
                if seen.insert(owner_id.clone()) {
                    pending.push(owner_id.clone());
                }
                owners.insert(
                    binding.name().to_owned(),
                    DiskCopyOwner {
                        unit: owner_id,
                        name: owner_name,
                    },
                );
            }
            let source = self.inner.sources.retained(&id).ok_or_else(|| {
                DiskCopyWorkspaceError::mismatch("source is an immutable prepared host catalog")
            })?;
            let bindings = definition
                .bindings()
                .iter()
                .filter(|binding| !binding.is_alias())
                .cloned()
                .collect::<Vec<_>>();
            let direct = PreparedDirectReadPlan::prepare(source.as_ref(), &bindings)?;
            units.push(PreparedDiskReadUnit {
                definition: definition.clone(),
                source: source.identity(),
                direct,
                owners,
                aliases: BTreeMap::new(),
            });
            index += 1;
        }
        // Resolve immutable alias geometry after the complete owner closure is
        // prepared. This does not infer recipes again or read source payloads.
        let alias_geometry = units
            .iter()
            .map(|unit| {
                unit.definition
                    .bindings()
                    .iter()
                    .filter(|binding| binding.is_alias())
                    .map(|binding| {
                        let owner = &unit.owners[binding.name()];
                        let output = units
                            .iter()
                            .find(|unit| unit.id() == owner.unit())
                            .and_then(|unit| {
                                unit.direct
                                    .outputs()
                                    .iter()
                                    .find(|output| output.name() == owner.name())
                            })
                            .ok_or_else(|| {
                                DiskCopyWorkspaceError::mismatch(
                                    "alias canonical read output is absent",
                                )
                            })?;
                        if output.logical_bytes() != binding.expected_bytes() {
                            return Err(DiskCopyWorkspaceError::mismatch(
                                "alias and canonical output byte counts differ",
                            ));
                        }
                        Ok((
                            binding.name().to_owned(),
                            AliasGeometry {
                                shape: output.shape().to_vec(),
                                dtype: output.dtype(),
                                bytes: output.logical_bytes(),
                            },
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, DiskCopyWorkspaceError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (unit, aliases) in units.iter_mut().zip(alias_geometry) {
            unit.aliases = aliases;
        }
        Ok(PreparedDiskReadPlans {
            manager: self.inner.downgrade(),
            requested: ids.to_vec(),
            units,
        })
    }

    /// Entirely cold: no source APIs, payload reads, native housekeeping, graph
    /// evaluation, completion polling, fallback, pinning or residency mutation.
    /// Existing Host copies and in-flight/failed transfers remain unproved.
    pub(crate) fn disk_copy_workspace(
        &self,
        plans: &PreparedDiskReadPlans,
        allocation: MetalAllocationFacts,
    ) -> Result<DiskCopyWorkspace, DiskCopyWorkspaceError> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| DiskCopyWorkspaceError::Residency(ResidencyError::StatePoisoned))?;
        self.disk_copy_workspace_locked(plans, allocation, &state)
    }

    fn disk_copy_workspace_locked(
        &self,
        plans: &PreparedDiskReadPlans,
        allocation: MetalAllocationFacts,
        state: &ManagerState,
    ) -> Result<DiskCopyWorkspace, DiskCopyWorkspaceError> {
        plans.validate_locked(self, state)?;
        state
            .control
            .ledger()
            .require_initialized()
            .map_err(|error| DiskCopyWorkspaceError::Residency(error.into()))?;
        if self.inner.failed_transfer.load(Ordering::Acquire) {
            return Err(DiskCopyWorkspaceError::unknown(
                "manager has failed transfer ownership",
            ));
        }
        if state.device_stream.get_device()?.get_type()? != safemlx::DeviceType::Gpu {
            return Err(DiskCopyWorkspaceError::unknown(
                "destination is not selected Metal",
            ));
        }
        for unit in state.control.units() {
            for tier in [MemoryTier::Host, MemoryTier::Device] {
                if state
                    .control
                    .ledger()
                    .copy_status(unit.id(), tier)
                    .map_err(|error| DiskCopyWorkspaceError::Residency(error.into()))?
                    .is_some_and(|copy| copy.in_flight().is_some())
                {
                    return Err(DiskCopyWorkspaceError::unknown(
                        "manager transfer remains in flight",
                    ));
                }
            }
        }
        let mut observations = BTreeMap::new();
        let mut physical = BTreeMap::new();
        for unit in &plans.units {
            let storage = state.storage.get(unit.id());
            if storage.is_some_and(|storage| storage.host.is_some())
                || state
                    .control
                    .ledger()
                    .copy_status(unit.id(), MemoryTier::Host)
                    .map_err(|error| DiskCopyWorkspaceError::Residency(error.into()))?
                    .is_some()
            {
                return Err(DiskCopyWorkspaceError::unknown(
                    "host promotion has no retained direct-read proof",
                ));
            }
            let device = storage.and_then(|storage| storage.device.as_ref());
            let ready = state
                .control
                .ledger()
                .copy_status(unit.id(), MemoryTier::Device)
                .map_err(|error| DiskCopyWorkspaceError::Residency(error.into()))?
                .is_some();
            if ready != device.is_some() {
                return Err(DiskCopyWorkspaceError::mismatch(
                    "device publication and residency ledger differ",
                ));
            }
            if let Some(device) = device {
                if device.arrays.len() != unit.definition.bindings().len() {
                    return Err(DiskCopyWorkspaceError::mismatch(
                        "warm device binding names differ",
                    ));
                }
                for binding in unit.definition.bindings() {
                    let value = device.arrays.get(binding.name()).ok_or_else(|| {
                        DiskCopyWorkspaceError::mismatch("warm device binding is absent")
                    })?;
                    let metadata = value.try_metadata_snapshot().map_err(|error| match error {
                        safemlx::ArrayMetadataError::RuntimeBusy => {
                            DiskCopyWorkspaceError::unknown("native metadata lock is busy")
                        }
                        error => DiskCopyWorkspaceError::Metadata(error),
                    })?;
                    let backing = metadata.allocation().ok_or_else(|| {
                        DiskCopyWorkspaceError::unknown("warm device backing is not certified")
                    })?;
                    if backing.bytes() < metadata.nbytes() {
                        return Err(DiskCopyWorkspaceError::mismatch(
                            "warm device backing is smaller than logical payload",
                        ));
                    }
                    if let Some(previous) =
                        physical.insert(backing.identity(), backing.bytes() as u64)
                    {
                        if previous != backing.bytes() as u64 {
                            return Err(DiskCopyWorkspaceError::mismatch(
                                "one allocation has conflicting capacities",
                            ));
                        }
                    }
                    observations.insert((unit.id().clone(), binding.name().to_owned()), metadata);
                }
            }
        }
        let mut units = Vec::with_capacity(plans.units.len());
        let mut fresh_capacity_bytes = 0;
        for unit in &plans.units {
            let fresh = unit
                .direct
                .workspace_bound(allocation)?
                .bytes()
                .ok_or_else(|| {
                    DiskCopyWorkspaceError::unknown("direct read capacity is unavailable")
                })?;
            fresh_capacity_bytes = add(fresh_capacity_bytes, fresh)?;
            let mut bindings = Vec::with_capacity(unit.definition.bindings().len());
            for binding in unit.definition.bindings() {
                let owner = &unit.owners[binding.name()];
                let owner_unit = plans.unit(owner.unit()).ok_or_else(|| {
                    DiskCopyWorkspaceError::mismatch("canonical owner omitted from read closure")
                })?;
                let output = owner_unit
                    .direct
                    .outputs()
                    .iter()
                    .find(|output| output.name() == owner.name())
                    .ok_or_else(|| {
                        DiskCopyWorkspaceError::mismatch(
                            "canonical owner omitted from direct outputs",
                        )
                    })?;
                if output.logical_bytes() != binding.expected_bytes() {
                    return Err(DiskCopyWorkspaceError::mismatch(
                        "alias and owner byte counts differ",
                    ));
                }
                let current = observations.get(&(unit.id().clone(), binding.name().to_owned()));
                let owner_current =
                    observations.get(&(owner.unit().clone(), owner.name().to_owned()));
                for metadata in [current, owner_current].into_iter().flatten() {
                    if metadata.shape() != output.shape()
                        || metadata.dtype() != output.dtype()
                        || metadata.nbytes() as u64 != output.logical_bytes()
                    {
                        return Err(DiskCopyWorkspaceError::mismatch(
                            "warm device geometry differs from exact read output",
                        ));
                    }
                }
                if binding.is_alias()
                    && current.is_some()
                    && current.and_then(|metadata| metadata.allocation())
                        != owner_current.and_then(|metadata| metadata.allocation())
                {
                    return Err(DiskCopyWorkspaceError::mismatch(
                        "warm alias does not share its canonical owner backing",
                    ));
                }
                let output_capacity_bytes = output.capacity_bytes(allocation)?.max(
                    owner_current
                        .and_then(|metadata| metadata.allocation())
                        .map_or(0, |backing| backing.bytes() as u64),
                );
                bindings.push(DiskCopyBinding {
                    binding: binding.clone(),
                    owner: owner.clone(),
                    shape: output.shape().to_vec(),
                    dtype: output.dtype(),
                    output_capacity_bytes,
                    current_allocation: current.and_then(|metadata| metadata.allocation()),
                });
            }
            units.push(DiskCopyUnit {
                definition: unit.definition.clone(),
                bindings,
                fresh_capacity_bytes: fresh,
                current_device: state
                    .storage
                    .get(unit.id())
                    .is_some_and(|storage| storage.device.is_some()),
            });
        }
        let current_device_bytes = physical
            .values()
            .try_fold(0_u64, |sum, bytes| add(sum, *bytes))?;
        Ok(DiskCopyWorkspace {
            plans: plans.clone(),
            allocation,
            units,
            fresh_capacity_bytes,
            current_device_bytes,
        })
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;

mod route;
pub(super) use route::{
    DiskRouteActivation, materialize_admitted_disk, materialize_admitted_disk_into,
    validate_admitted_disk_aliases, validate_disk_access, validate_disk_acquisition,
};
pub(crate) use route::{DiskRouteGuard, DiskRouteReceipt};
