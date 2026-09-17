//! Operation-local restrictions for one previously priced foreground disk route.
//! Core SessionOperation remains the submission authority. These metadata-only
//! receipts restrict materialization choices; they neither reserve nor fund work.

use super::*;
use crate::backend::runtime::residency::manager::{
    ResidencySources, ResidentTransferResources, materialization::PreparedResidentArrays,
};
use std::thread::ThreadId;

fn fenced(reason: &'static str) -> DiskCopyWorkspaceError {
    DiskCopyWorkspaceError::Unproved {
        reason,
        source: WorkingMemoryError::ExecutionFenced,
    }
}
fn residency(error: DiskCopyWorkspaceError) -> ResidencyError {
    ResidencyError::AdmittedDiskRoute {
        source: Box::new(error),
    }
}

#[derive(Debug)]
struct DiskRouteDefinition {
    workspace: DiskCopyWorkspace,
    windows: Vec<Vec<OffloadUnitId>>,
    persistent: BTreeSet<OffloadUnitId>,
}

/// Cold reusable proof. It retains neither manager/source payloads nor an active
/// operation. Cloning it cannot prolong the guard's live manager restriction.
#[derive(Debug, Clone)]
pub(crate) struct DiskRouteReceipt(Arc<DiskRouteDefinition>);

#[derive(Debug)]
pub(crate) struct DiskRouteActivation {
    receipt: DiskRouteReceipt,
    thread: ThreadId,
}

/// Retained by existing submission recovery until publication and settlement.
/// Drop releases only immutable metadata; it takes no manager/native lock,
/// performs no eviction, and does not reset another operation's window.
#[derive(Debug)]
pub(crate) struct DiskRouteGuard {
    _activation: Arc<DiskRouteActivation>,
}

impl DiskRouteReceipt {
    /// Borrows the already validated closure membership without rebuilding a set.
    pub(crate) fn parameter_owner_is_persistent(&self, unit: &OffloadUnitId) -> bool {
        self.0.persistent.contains(unit)
    }
    /// Borrows the actual retained permitted window; grants no active route.
    pub(crate) fn parameter_window(&self, ordinal: usize) -> Option<&[OffloadUnitId]> {
        self.0.windows.get(ordinal).map(Vec::as_slice)
    }
}

impl DiskCopyWorkspace {
    /// Complete owner units that existing cross-unit alias pins can keep alive
    /// across rotating windows. Includes owners that also occur in requested
    /// units, and recursively included owner units. Price each whole unit once.
    pub(crate) fn persistent_units(&self) -> Vec<OffloadUnitId> {
        self.plans
            .units
            .iter()
            .flat_map(|unit| {
                unit.owners
                    .values()
                    .filter(move |owner| owner.unit() != unit.id())
                    .map(|owner| owner.unit().clone())
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub(crate) fn receipt(
        &self,
        windows: Vec<Vec<OffloadUnitId>>,
    ) -> Result<DiskRouteReceipt, DiskCopyWorkspaceError> {
        let requested = self
            .requested_units()
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut covered = BTreeSet::new();
        if windows.is_empty() {
            return Err(DiskCopyWorkspaceError::mismatch(
                "disk receipt has no execution windows",
            ));
        }
        for window in &windows {
            let distinct = window.iter().cloned().collect::<BTreeSet<_>>();
            if window.is_empty()
                || distinct.len() != window.len()
                || !distinct.is_subset(&requested)
            {
                return Err(DiskCopyWorkspaceError::mismatch(
                    "disk receipt window differs from prepared requested units",
                ));
            }
            covered.extend(distinct);
        }
        if covered != requested {
            return Err(DiskCopyWorkspaceError::mismatch(
                "disk receipt windows omit requested units",
            ));
        }
        Ok(DiskRouteReceipt(Arc::new(DiskRouteDefinition {
            workspace: self.clone(),
            windows,
            persistent: self.persistent_units().into_iter().collect(),
        })))
    }

    /// Revalidates current warm storage against the immutable admitted envelope;
    /// replacement and eviction are allowed, while shape/dtype/capacity growth,
    /// changed sources, Host fallback and unresolved work remain rejected.
    pub(crate) fn validate_envelope(
        &self,
        manager: &ResidencyManager,
    ) -> Result<(), DiskCopyWorkspaceError> {
        let state = manager
            .inner
            .state
            .lock()
            .map_err(|_| DiskCopyWorkspaceError::Residency(ResidencyError::StatePoisoned))?;
        self.validate_envelope_locked(manager, &state)
    }

    fn validate_envelope_locked(
        &self,
        manager: &ResidencyManager,
        state: &ManagerState,
    ) -> Result<(), DiskCopyWorkspaceError> {
        let current = manager.disk_copy_workspace_locked(&self.plans, self.allocation, state)?;
        for (expected, actual) in self.units.iter().zip(&current.units) {
            if expected.definition != actual.definition
                || expected.bindings.len() != actual.bindings.len()
            {
                return Err(DiskCopyWorkspaceError::mismatch(
                    "disk unit changed after admission",
                ));
            }
            for (expected, actual) in expected.bindings.iter().zip(&actual.bindings) {
                if expected.binding != actual.binding
                    || expected.owner != actual.owner
                    || expected.shape != actual.shape
                    || expected.dtype != actual.dtype
                    || actual.output_capacity_bytes > expected.output_capacity_bytes
                {
                    return Err(DiskCopyWorkspaceError::mismatch(
                        "warm device backing exceeds admitted parameter envelope",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl DiskRouteReceipt {
    fn manager(&self) -> Result<ResidencyManager, DiskCopyWorkspaceError> {
        self.0
            .workspace
            .plans
            .manager
            .upgrade()
            .map(|inner| ResidencyManager { inner })
            .ok_or_else(|| DiskCopyWorkspaceError::mismatch("admitted disk manager has retired"))
    }
    pub(crate) fn validate(&self) -> Result<(), DiskCopyWorkspaceError> {
        self.0.workspace.validate_envelope(&self.manager()?)
    }
    pub(crate) fn activate(&self) -> Result<DiskRouteGuard, DiskCopyWorkspaceError> {
        self.manager()?.activate_disk_route(self)
    }
}

impl ResidencyManager {
    pub(crate) fn activate_disk_route(
        &self,
        receipt: &DiskRouteReceipt,
    ) -> Result<DiskRouteGuard, DiskCopyWorkspaceError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| DiskCopyWorkspaceError::Residency(ResidencyError::StatePoisoned))?;
        if state.admitted_disk_route.upgrade().is_some() {
            return Err(fenced("another disk operation is active"));
        }
        receipt.0.workspace.validate_envelope_locked(self, &state)?;
        let activation = Arc::new(DiskRouteActivation {
            receipt: receipt.clone(),
            thread: std::thread::current().id(),
        });
        state.admitted_disk_window.clear();
        state.admitted_disk_route = Arc::downgrade(&activation);
        Ok(DiskRouteGuard {
            _activation: activation,
        })
    }
    pub(crate) fn admitted_disk_route_active(&self) -> bool {
        // A poisoned manager must enter the strict path, which returns its typed
        // error; it must never silently choose ordinary materialization.
        self.inner
            .state
            .lock()
            .map(|state| state.admitted_disk_route.upgrade().is_some())
            .unwrap_or(true)
    }
    pub(crate) fn admitted_disk_persistent_units(&self) -> Vec<OffloadUnitId> {
        self.inner
            .state
            .lock()
            .ok()
            .and_then(|state| state.admitted_disk_route.upgrade())
            .map(|route| route.receipt.0.persistent.iter().cloned().collect())
            .unwrap_or_default()
    }
    pub(crate) fn set_admitted_disk_window(
        &self,
        window: &[OffloadUnitId],
    ) -> Result<bool, ResidencyError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResidencyError::StatePoisoned)?;
        let Some(route) = state.admitted_disk_route.upgrade() else {
            return Ok(false);
        };
        if route.thread != std::thread::current().id() {
            return Err(residency(fenced(
                "background work cannot change an admitted foreground window",
            )));
        }
        if !route
            .receipt
            .0
            .windows
            .iter()
            .any(|allowed| allowed == window)
        {
            return Err(residency(DiskCopyWorkspaceError::mismatch(
                "requested disk window was not priced",
            )));
        }
        let next_window = window
            .iter()
            .cloned()
            .chain(route.receipt.0.persistent.iter().cloned())
            .collect::<BTreeSet<_>>();
        if state.failed_transfer.load(Ordering::Acquire) {
            return Err(residency(fenced(
                "admitted disk operation has failed native transfer ownership",
            )));
        }
        // This preflight precedes caller-side constructor allocations. Pending
        // copies created by this activation remain valid through their recorded
        // plan provenance; no completion polling or retirement occurs here.
        for id in &next_window {
            validate_unit(&route, &state, &self.inner.sources, id).map_err(residency)?;
        }
        // Window changes deliberately precede policy draining/eviction. New
        // acquisitions below enforce that obsolete caches have actually retired.
        state.admitted_disk_window = next_window;
        Ok(true)
    }
}

fn check_access(
    route: &DiskRouteActivation,
    state: &ManagerState,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
) -> Result<(), DiskCopyWorkspaceError> {
    if route.thread != std::thread::current().id() {
        return Err(fenced(
            "background acquisition cannot use admitted foreground authority",
        ));
    }
    if tier != MemoryTier::Device {
        return Err(fenced(
            "admitted direct disk operation forbids Host materialization",
        ));
    }
    for id in ids {
        if !state.admitted_disk_window.contains(id) {
            return Err(fenced(
                "acquisition is outside the priced disk window and owner closure",
            ));
        }
    }
    if state.failed_transfer.load(Ordering::Acquire) {
        return Err(fenced(
            "admitted disk operation has failed native transfer ownership",
        ));
    }
    Ok(())
}

/// Membership checks precede waits, demand/prefetch telemetry, recursive owner
/// acquisition and any ordinary ready-copy branch.
pub(in super::super) fn validate_disk_access(
    state: &ManagerState,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
) -> Result<(), ResidencyError> {
    if let Some(route) = state.admitted_disk_route.upgrade() {
        check_access(&route, state, ids, tier).map_err(residency)?;
    }
    Ok(())
}

fn validate_unit(
    route: &DiskRouteActivation,
    state: &ManagerState,
    sources: &ResidencySources,
    id: &OffloadUnitId,
) -> Result<(), DiskCopyWorkspaceError> {
    let proof =
        route.receipt.0.workspace.plans.unit(id).ok_or_else(|| {
            DiskCopyWorkspaceError::mismatch("unit has no retained direct read plan")
        })?;
    if state.control.unit(id) != Some(proof.definition()) {
        return Err(DiskCopyWorkspaceError::mismatch(
            "unit definition changed under active disk route",
        ));
    }
    let source = sources.retained(id).ok_or_else(|| {
        DiskCopyWorkspaceError::mismatch("source is an immutable prepared host catalog")
    })?;
    if proof.source != source.identity() {
        return Err(DiskCopyWorkspaceError::mismatch(
            "unit source changed under active disk route",
        ));
    }
    let storage = state
        .storage
        .get(id)
        .ok_or_else(|| DiskCopyWorkspaceError::mismatch("unit publication is absent"))?;
    if storage.host.is_some()
        || state
            .control
            .ledger()
            .copy_status(id, MemoryTier::Host)
            .map_err(|error| DiskCopyWorkspaceError::Residency(error.into()))?
            .is_some()
    {
        return Err(fenced(
            "Host promotion is forbidden by the admitted direct read route",
        ));
    }
    let status = state
        .control
        .ledger()
        .copy_status(id, MemoryTier::Device)
        .map_err(|error| DiskCopyWorkspaceError::Residency(error.into()))?;
    let pending = status.is_some_and(|copy| copy.in_flight().is_some());
    let own_pending = pending
        && storage
            .device_disk_route
            .upgrade()
            .is_some_and(|owner| std::ptr::eq(owner.as_ref(), route));
    if pending && !own_pending {
        return Err(fenced(
            "in-flight device copy has no current admitted plan provenance",
        ));
    }
    if status.is_some() != storage.device.is_some() {
        return Err(DiskCopyWorkspaceError::mismatch(
            "device publication and ledger differ",
        ));
    }
    let Some(device) = &storage.device else {
        return Ok(());
    };
    let envelope = route
        .receipt
        .0
        .workspace
        .units
        .iter()
        .find(|unit| unit.id() == id)
        .ok_or_else(|| DiskCopyWorkspaceError::mismatch("unit envelope is absent"))?;
    if device.arrays.len() != envelope.bindings.len() {
        return Err(DiskCopyWorkspaceError::mismatch(
            "warm binding names changed",
        ));
    }
    for expected in &envelope.bindings {
        let value = device
            .arrays
            .get(expected.name())
            .ok_or_else(|| DiskCopyWorkspaceError::mismatch("warm binding is absent"))?;
        let metadata = value
            .try_metadata_snapshot()
            .map_err(DiskCopyWorkspaceError::Metadata)?;
        let backing = metadata.allocation();
        if backing.is_none() && !own_pending {
            return Err(DiskCopyWorkspaceError::unknown(
                "warm device backing is not certified",
            ));
        }
        if metadata.shape() != expected.shape
            || metadata.dtype() != expected.dtype
            || metadata.nbytes() as u64 != expected.logical_bytes()
            || backing
                .is_some_and(|backing| backing.bytes() as u64 > expected.output_capacity_bytes)
        {
            return Err(DiskCopyWorkspaceError::mismatch(
                "warm backing exceeds admitted shape, dtype or capacity",
            ));
        }
        if expected.binding.is_alias() {
            let owner = state
                .storage
                .get(expected.owner.unit())
                .and_then(|unit| unit.device.as_ref())
                .and_then(|unit| unit.arrays.get(expected.owner.name()))
                .ok_or_else(|| {
                    DiskCopyWorkspaceError::mismatch("warm alias canonical owner is absent")
                })?;
            let owner_backing = owner
                .try_metadata_snapshot()
                .map_err(DiskCopyWorkspaceError::Metadata)?
                .allocation();
            if !own_pending && owner_backing != backing {
                return Err(DiskCopyWorkspaceError::mismatch(
                    "warm alias changed physical owner",
                ));
            }
        }
    }
    Ok(())
}

/// Complete preflight of requested units and all recursive owners, before the
/// first owner can allocate. Other in-flight units in this same window may run
/// concurrently under the existing transfer recovery and window envelope.
pub(in super::super) fn validate_disk_acquisition(
    state: &ManagerState,
    sources: &ResidencySources,
    ids: &[OffloadUnitId],
    tier: MemoryTier,
) -> Result<(), ResidencyError> {
    let Some(route) = state.admitted_disk_route.upgrade() else {
        return Ok(());
    };
    check_access(&route, state, ids, tier).map_err(residency)?;
    let mut pending = ids.to_vec();
    let mut seen = BTreeSet::new();
    let mut index = 0;
    while index < pending.len() {
        let id = pending[index].clone();
        index += 1;
        if !seen.insert(id.clone()) {
            continue;
        }
        check_access(&route, state, std::slice::from_ref(&id), tier).map_err(residency)?;
        validate_unit(&route, state, sources, &id).map_err(residency)?;
        let unit = route
            .receipt
            .0
            .workspace
            .plans
            .unit(&id)
            .expect("validated retained unit");
        pending.extend(
            unit.owners
                .values()
                .filter(|owner| owner.unit() != &id)
                .map(|owner| owner.unit().clone()),
        );
    }
    if ids.iter().any(|id| {
        state
            .storage
            .get(id)
            .is_some_and(|unit| unit.device.is_none())
    }) {
        for unit in route.receipt.0.workspace.plans.units() {
            if !state.admitted_disk_window.contains(unit.id())
                && state
                    .storage
                    .get(unit.id())
                    .is_some_and(|unit| unit.device.is_some())
            {
                return Err(residency(fenced(
                    "obsolete device window must be evicted before refill",
                )));
            }
        }
    }
    Ok(())
}

/// Some means a live admitted route was selected; its failure never falls back
/// to re-planning, Host promotion or the ordinary conversion materializer.
pub(in super::super) fn materialize_admitted_disk(
    state: &ManagerState,
    id: &OffloadUnitId,
    retained: &mut ResidentTransferResources,
) -> Option<Result<PreparedResidentArrays, ResidencyError>> {
    let route = state.admitted_disk_route.upgrade()?;
    Some((|| {
        let plan = route.receipt.0.workspace.plans.unit(id).ok_or_else(|| {
            residency(DiskCopyWorkspaceError::mismatch(
                "unit has no admitted direct read plan",
            ))
        })?;
        let arrays = plan
            .materialize_canonical(&state.source_stream, &state.device_stream, |array| {
                retained.retained_arrays.push(array.clone())
            })
            .map_err(residency)?;
        Ok(PreparedResidentArrays {
            arrays: arrays.into(),
            direction: eredu_core::residency::TransferDirection::DiskToDevice,
        })
    })())
}

// Exact admitted-plan failure transport; no ordinary source fallback.
enum PreparedDiskFillError {
    Read(crate::backend::runtime::checkpoint::recipe::PreparedDirectReadError),
    Destination(super::super::NamedArrayError),
}
impl From<crate::backend::runtime::checkpoint::recipe::PreparedDirectReadError>
    for PreparedDiskFillError
{
    fn from(cause: crate::backend::runtime::checkpoint::recipe::PreparedDirectReadError) -> Self {
        Self::Read(cause)
    }
}

pub(in super::super) fn materialize_admitted_disk_into(
    state: &ManagerState,
    id: &OffloadUnitId,
    arrays: &mut super::super::NamedArrays,
    retained: &mut ResidentTransferResources,
) -> Option<Result<(), ResidencyError>> {
    let route = state.admitted_disk_route.upgrade()?;
    Some((|| {
        let plan = route.receipt.0.workspace.plans.unit(id).ok_or_else(|| {
            residency(DiskCopyWorkspaceError::mismatch(
                "unit has no admitted direct read plan",
            ))
        })?;
        plan.materialize_canonical_into::<PreparedDiskFillError>(
            &state.source_stream,
            &state.device_stream,
            |array| retained.retained_arrays.push(array.clone()),
            |name, array| {
                arrays
                    .put(name, array)
                    .map_err(|(cause, _retained_array)| PreparedDiskFillError::Destination(cause))
            },
        )
        .map_err(|cause| match cause {
            PreparedDiskFillError::Read(cause) => residency(cause.into()),
            PreparedDiskFillError::Destination(cause) => cause.into(),
        })
    })())
}

/// Mandatory second phase for admitted canonical batches. This retains every
/// source/geometry check; a failed admitted route never enters ordinary reads.
pub(in super::super) fn validate_admitted_disk_aliases(
    state: &ManagerState,
    prepared: &[(OffloadUnitId, PreparedResidentArrays)],
) -> Result<(), ResidencyError> {
    let Some(route) = state.admitted_disk_route.upgrade() else {
        return Ok(());
    };
    for (id, item) in prepared {
        let plan = route.receipt.0.workspace.plans.unit(id).ok_or_else(|| {
            residency(DiskCopyWorkspaceError::mismatch(
                "unit has no admitted direct read plan",
            ))
        })?;
        plan.validate_external_lookup(|name| item.arrays.get(name))
            .map_err(residency)?;
    }
    Ok(())
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
