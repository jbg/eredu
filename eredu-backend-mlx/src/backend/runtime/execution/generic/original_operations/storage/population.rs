//! Source-derived selected Device acquisition bounds. Fixed recovery owners
//! and dynamic storage qualification remain distinct; counts grant no authority.
use super::*;
use crate::backend::runtime::{
    checkpoint::store::MaterializationPayloadShape,
    residency::manager::{TransferPayloadShape, WindowPopulation},
};

impl TransferPayloadShape {
    pub(crate) fn window(source: WindowPopulation) -> Option<Self> {
        Some(Self {
            leases: source.requested,
            lease_id_bytes: source.requested_id_bytes,
            unit_ids: source.units,
            unit_id_bytes: source.unit_id_bytes,
            pending_sources: source.recipe_pending.checked_mul(4)?,
            // Each unit may run twice, each binding may run twice. Direct
            // batches retain input+output; ordinary recipes retain host+output;
            // three slots per physical binding also cover partial promotion.
            retained_arrays: source
                .physical_bindings
                .checked_mul(3)?
                .checked_add(source.aliases)?
                .checked_mul(4)?,
            retained_host: source.units.checked_mul(2)?,
            retained_events: source.physical_bindings.checked_mul(2)?,
            output_arrays: source.bindings,
            prepared_units: source.units,
            binding_rows: source.bindings,
            binding_name_bytes: source.binding_name_bytes,
            alias_assignments: source.aliases,
            alias_owner_name_bytes: source.alias_owner_name_bytes,
            declaration_clone_bytes: source.declaration_clone_bytes,
            declaration_clone_allocations: source.declaration_clone_allocations,
        })
    }
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            leases: self.leases.max(other.leases),
            lease_id_bytes: self.lease_id_bytes.max(other.lease_id_bytes),
            unit_ids: self.unit_ids.max(other.unit_ids),
            unit_id_bytes: self.unit_id_bytes.max(other.unit_id_bytes),
            pending_sources: self.pending_sources.max(other.pending_sources),
            retained_arrays: self.retained_arrays.max(other.retained_arrays),
            retained_host: self.retained_host.max(other.retained_host),
            retained_events: self.retained_events.max(other.retained_events),
            output_arrays: self.output_arrays.max(other.output_arrays),
            prepared_units: self.prepared_units.max(other.prepared_units),
            binding_rows: self.binding_rows.max(other.binding_rows),
            binding_name_bytes: self.binding_name_bytes.max(other.binding_name_bytes),
            alias_assignments: self.alias_assignments.max(other.alias_assignments),
            alias_owner_name_bytes: self
                .alias_owner_name_bytes
                .max(other.alias_owner_name_bytes),
            declaration_clone_bytes: self
                .declaration_clone_bytes
                .max(other.declaration_clone_bytes),
            declaration_clone_allocations: self
                .declaration_clone_allocations
                .max(other.declaration_clone_allocations),
        }
    }
}

/// Named remaining mechanisms, rather than pretending scalar source counts
/// cover the still-allocating producer. No value of this type certifies storage.
#[derive(Clone, Copy, Debug)]
pub(crate) struct UnpreparedResidencyStorage {
    pub(crate) transfer: TransferPayloadShape,
    pub(crate) materialization: MaterializationPayloadShape,
    pub(crate) controller_units: usize,
    // Pending source leases own source-specific metadata/cache/read buffers;
    // recipe clones/inference/errors own maps, Strings and shape/index Vecs.
    // Their closed layout/fill APIs are not supplied by a recovery-node bank.
}

pub(crate) const TRANSFERS_PER_NONEMPTY_WINDOW: usize = 2;

pub(super) struct AttemptPopulation {
    pub(super) transfers: usize,
    pub(super) observations: usize,
    pub(super) pending: usize,
    pub(super) weight: usize,
    pub(super) transfer: TransferPayloadShape,
    pub(super) materialization: MaterializationPayloadShape,
}
impl AttemptPopulation {
    pub(super) fn new(source: WindowPopulation) -> Result<Self, Error> {
        let transfer = TransferPayloadShape::window(source).ok_or_else(overflow)?;
        let nonempty = source.requested != 0;
        // One warm owner plus optional one missing-closure owner. Inner binding
        // and whole-unit retries each run at most twice; each may detach once.
        let pending = if nonempty {
            source.recipe_pending.checked_mul(4).ok_or_else(overflow)?
        } else {
            0
        };
        let weight = if nonempty {
            source
                .recipe_materializations
                .checked_mul(4)
                .and_then(|n| {
                    source
                        .physical_bindings
                        .checked_mul(2)
                        .and_then(|b| n.checked_add(b))
                })
                .and_then(|n| n.checked_add(source.units))
                .ok_or_else(overflow)?
        } else {
            0
        };
        Ok(Self {
            transfers: if nonempty {
                TRANSFERS_PER_NONEMPTY_WINDOW
            } else {
                0
            },
            observations: usize::from(nonempty),
            pending,
            weight,
            materialization: MaterializationPayloadShape {
                inputs: 1,
                outputs: transfer.retained_arrays.max(1),
                pending_sources: transfer.pending_sources,
            },
            transfer,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ResidencyPopulation {
    pub(crate) forwards: usize,
    pub(crate) controller_units: usize,
    pub(crate) transfers: usize,
    pub(crate) observations: usize,
    pub(crate) host: usize,
    pub(crate) pending: usize,
    pub(crate) weight: usize,
    pub(crate) materialization: usize,
    pub(crate) unprepared: UnpreparedResidencyStorage,
}
impl ResidencyPopulation {
    pub(crate) fn selected_native_transfers(window: WindowPopulation)
        -> Result<(usize, usize, usize, usize), Error> {
        let attempt = AttemptPopulation::new(window)?;
        Ok((attempt.transfers, attempt.observations, attempt.transfer.output_arrays,
            TRANSFERS_PER_NONEMPTY_WINDOW))
    }
    pub(crate) fn pending_for_window(window: WindowPopulation) -> Result<usize, Error> {
        Ok(AttemptPopulation::new(window)?.pending)
    }

    pub(crate) fn from_policy<U: 'static, P>(
        policy: &MlxLayerwisePolicy<U, P>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Option<Self>, Error> {
        let Ok(source) = &policy.operation_source else {
            return Ok(None);
        };
        Self::from_windows(source.controller_units, source.windows(), geometry).map(Some)
    }
    pub(crate) fn from_windows(
        controller_units: usize,
        windows: &[WindowPopulation],
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        let plan = PrefillControlPlan::new(geometry, true).map_err(memory)?;
        let forwards = plan
            .span_count()
            .checked_add(geometry.max_output_tokens.saturating_sub(1))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(overflow)?;
        Self::from_window_forwards(controller_units, windows, forwards)
    }
    pub(crate) fn from_window_forwards(
        controller_units: usize,
        windows: &[WindowPopulation],
        forwards: usize,
    ) -> Result<Self, Error> {
        let mut transfers = 0usize;
        let mut observations = 0usize;
        let mut pending = 0usize;
        let mut weight = 0usize;
        let mut payload = TransferPayloadShape::default();
        for source in windows {
            if source.requested == 0 {
                continue;
            }
            let attempt = AttemptPopulation::new(*source)?;
            transfers = transfers
                .checked_add(attempt.transfers)
                .ok_or_else(overflow)?;
            observations = observations
                .checked_add(attempt.observations)
                .ok_or_else(overflow)?;
            pending = pending.checked_add(attempt.pending).ok_or_else(overflow)?;
            weight = weight.checked_add(attempt.weight).ok_or_else(overflow)?;
            payload = payload.merge(attempt.transfer);
        }
        Ok(Self {
            forwards,
            controller_units,
            transfers: transfers.checked_mul(forwards).ok_or_else(overflow)?,
            observations: observations.checked_mul(forwards).ok_or_else(overflow)?,
            host: 0, // Selected Device promotion uses retained host buffers.
            pending: pending.checked_mul(forwards).ok_or_else(overflow)?,
            weight: weight.checked_mul(forwards).ok_or_else(overflow)?,
            materialization: 0, // Recipe detach/validation synchronizes directly.
            unprepared: UnpreparedResidencyStorage {
                transfer: payload,
                materialization: MaterializationPayloadShape {
                    inputs: 1, // NegLog validation; detach-only preparations use zero.
                    outputs: payload.retained_arrays.max(1),
                    pending_sources: payload.pending_sources,
                },
                controller_units,
            },
        })
    }
    pub(crate) fn known_control_bytes(&self, windows: &[WindowPopulation]) -> Option<u64> {
        let scratch =
            eredu_runtime::residency::ResidencyClosureSlot::layout(self.controller_units)?.size();
        let shared_plan = eredu_core::residency::ResidencyPlanSource::shared_plan_layout()?.size();
        super::windows::layout(windows, self.forwards)?
            .checked_add(u64::try_from(shared_plan).ok()?)?
            .checked_add(
                u64::try_from(scratch.checked_add(OriginalResidencySlots::control_bytes()?)?)
                    .ok()?,
            )
    }
    pub(crate) fn prepared_payload_control_bytes(
        &self,
        windows: &[WindowPopulation],
    ) -> Option<u64> {
        // Per-window recovery and paired-bank cache keys/nodes/groups are
        // concrete; G4 final tickets are in the paired source component.
        // A supplied cache capsule covers its fixed shared Arc/PAL lifetime.
        // An explicitly compiled immutable GGUF catalog also retains its own
        // cold constructor account. Supplied source construction covers reader,
        // materializer and StoreInner storage; a built-in GGUF union additionally
        // covers its child vector/key directory and initial recipe controls.
        // Ready-host owners use the separate authenticated constructor join
        // below. Other routes still need their actual shared manager/source
        // producer; legacy pins retain their ordinary constructor.
        // Each transfer now owns its final publication-row Vec and two ID
        // destinations per canonical unit before activation. The shared ready-
        // host/direct worker moves those IDs without collecting or cloning
        // declarations; their exact layouts enter the existing transfer bank.
        // The additional original pin destinations/node are priced by their
        // owning HostCopyWorkspace and retain raw custody through deferred drop.
        // Direct routes retain DiskRouteDefinition,
        // window/owner maps and per-read metadata/native destinations. Those
        // complete selected controls, future recipe entries and manager baselines
        // remain unresolved. A sealed empty GGUF bank supplies none of them.
        let _ = self.known_control_bytes(windows)?;
        None
    }

    pub(crate) fn prepared_foreground_payload_control_bytes(
        &self,
        windows: &crate::backend::runtime::residency::manager::OperationWindows,
        source: &super::super::super::host_workspace::LayerwiseWorkspace,
        manager: &ResidencyManager,
    ) -> Option<u64> {
        self.known_control_bytes(windows.as_slice())?;
        if !source.has_original_foreground_source(manager, windows) {
            return None;
        }
        // Final read destinations and their source backing belong to the exact
        // request slot plan. The source constructor owns descriptor, window and
        // controller metadata; ordinary disk receipts cannot qualify this path.
        // Existing transfer/publication/closure banks cover their own payloads.
        let fixed = [
            size_of::<(
                &Self,
                &crate::backend::runtime::residency::manager::OperationWindows,
                &super::super::super::host_workspace::LayerwiseWorkspace,
                &ResidencyManager,
            )>(),
            size_of::<Option<u64>>(),
            size_of::<bool>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        u64::try_from(fixed).ok()
    }

    pub(crate) fn prepared_ready_host_payload_control_bytes(
        &self,
        windows: &crate::backend::runtime::residency::manager::OperationWindows,
        source: &super::super::super::host_workspace::LayerwiseWorkspace,
        manager: &ResidencyManager,
    ) -> Option<u64> {
        self.known_control_bytes(windows.as_slice())?;
        // The retained route creates no pending recipe/materializer slots. Its
        // controller, closure, named publication, transfer and binding storage
        // are already in the request's known controls; source construction and
        // every shared alias remain charged to the authenticated source owner.
        if !source.has_original_ready_host_source(manager, windows) {
            return None;
        }
        // Only the new borrowed qualification frames are additional request
        // controls. No source payload or already-paid window bank is charged twice.
        let fixed = [
            size_of::<(
                &Self,
                &crate::backend::runtime::residency::manager::OperationWindows,
                &super::super::super::host_workspace::LayerwiseWorkspace,
                &ResidencyManager,
            )>(),
            size_of::<(
                &super::super::super::host_workspace::LayerwiseWorkspace,
                &ResidencyManager,
                &crate::backend::runtime::residency::manager::OperationWindows,
            )>(),
            size_of::<(
                &crate::backend::runtime::residency::manager::HostCopyWorkspace,
                &ResidencyManager,
                &crate::backend::runtime::residency::manager::OperationWindows,
            )>(),
            size_of::<(
                &ResidencyManager,
                &crate::backend::runtime::residency::manager::OperationWindows,
            )>(),
            size_of::<[bool; 3]>(),
            size_of::<Option<u64>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        u64::try_from(fixed).ok()
    }
}

#[cfg(test)]
mod tests;
