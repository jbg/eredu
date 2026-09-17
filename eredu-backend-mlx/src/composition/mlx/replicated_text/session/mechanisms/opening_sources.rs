//! Private lexical handoff of live native sources at the canonical chunk opening.
//!
//! This is an inventory collector, not publication, complete outer-model source
//! validation, budget evidence, a funding scope, or native completion authority.
use super::*;
use crate::backend::runtime::residency::storage::RetainedStorage;
use eredu_core::DistributedCommitEpoch;
use eredu_runtime::{
    inspection::PrefillChunkRetentionContext, prefill::PrefillChunk,
    working_memory::InferenceRequest,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Debug, thiserror::Error)]
pub(in crate::composition::mlx::replicated_text) enum NativeOpeningSourceError {
    #[error("native opening source slot is borrowed")]
    Busy,
    #[error("native opening source binding is inactive")]
    Inactive,
    #[error("native opening source inventory is still pending")]
    Pending,
    #[error("native opening source coordinates differ")]
    Coordinates,
    #[error("native opening source inventory is incomplete")]
    Incomplete,
}
fn error(cause: NativeOpeningSourceError) -> Error {
    Error::Other(Box::new(cause))
}

/// An absent slot adds no heap allocation to ordinary execution. The fixed
/// representation remains a separate enclosing-control obligation.
#[derive(Default)]
pub(in crate::composition::mlx::replicated_text) struct NativeOpeningSourceBinding {
    slot: Option<Rc<OpeningSourceSlot>>,
}
struct OpeningSourceSlot {
    // Inventory payloads retire before the request/account owner. No slot holds
    // a model, observer, FundedWork, native scope or a reference back to itself.
    pending: RefCell<Option<NativeOpeningSourceInventory>>,
    active: Cell<bool>,
    request: InferenceRequest,
}

/// One private caller keeps this guard alongside its original operation. Drop
/// stops collection but never discards a pending partial or complete inventory.
pub(in crate::composition::mlx::replicated_text) struct NativeOpeningSourceGuard {
    slot: Rc<OpeningSourceSlot>,
}
impl Drop for NativeOpeningSourceGuard {
    fn drop(&mut self) {
        self.slot.active.set(false);
    }
}

impl NativeOpeningSourceBinding {
    pub(super) fn is_active(&self) -> bool {
        self.slot.as_ref().is_some_and(|slot| slot.active.get())
    }

    pub(super) fn bind(
        &mut self,
        request: &InferenceRequest,
    ) -> Result<NativeOpeningSourceGuard, Error> {
        if let Some(slot) = &self.slot {
            if slot.active.get() {
                return Err(error(NativeOpeningSourceError::Pending));
            }
            if slot
                .pending
                .try_borrow()
                .map_err(|_| error(NativeOpeningSourceError::Busy))?
                .is_some()
            {
                return Err(error(NativeOpeningSourceError::Pending));
            }
        }
        let slot = Rc::new(OpeningSourceSlot {
            pending: RefCell::new(None),
            active: Cell::new(true),
            request: request.clone(),
        });
        // The old empty slot/request is destroyed outside a RefCell loan.
        let old = self.slot.replace(slot.clone());
        drop(old);
        Ok(NativeOpeningSourceGuard { slot })
    }

    pub(super) fn collect<S: MlxStateMechanisms>(
        &self,
        state: &S,
        context: &PrefillChunkRetentionContext<'_>,
        parameters: impl FnOnce() -> Result<RetainedStorage, Error>,
        execution: impl FnOnce(&mut dyn FnMut(&MlxTensor)) -> bool,
    ) -> Result<(), Error> {
        let slot = self
            .slot
            .as_ref()
            .filter(|slot| slot.active.get())
            .ok_or_else(|| error(NativeOpeningSourceError::Inactive))?;
        slot.request
            .validate_same_request(context.request())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let mut pending = slot
            .pending
            .try_borrow_mut()
            .map_err(|_| error(NativeOpeningSourceError::Busy))?;
        if pending.is_some() {
            return Err(error(NativeOpeningSourceError::Pending));
        }
        *pending = Some(NativeOpeningSourceInventory {
            parts: OpeningSourceParts::default(),
            chunk: context.chunk().clone(),
            epoch: context.epoch(),
            request: context.request().clone(),
        });
        // Put the owner in the slot before any fallible builder/visitor. Errors
        // and unwinding retain all completed domains and execution prefix here.
        pending
            .as_mut()
            .expect("new opening owner")
            .parts
            .collect(state, parameters, execution)
    }

    /// Fixed new mechanisms-field bytes only; no Rc allocation or map nodes.
    pub(in crate::composition::mlx::replicated_text) fn inline_control_bytes() -> Option<u64> {
        std::mem::size_of::<Self>().try_into().ok()
    }

    /// Measured fixed controls/construction moves only. Rc allocator overhead,
    /// all inventory map/vector/source-store allocations, existing builder
    /// temporaries, visitor controls and enclosing owners remain separate.
    pub(in crate::composition::mlx::replicated_text) fn active_fixed_control_peak_bytes(
    ) -> Option<u64> {
        std::mem::size_of::<OpeningSourceSlot>()
            .checked_add(std::mem::size_of::<NativeOpeningSourceGuard>())?
            .checked_add(std::mem::size_of::<NativeOpeningSourceInventory>())?
            .checked_add(std::mem::size_of::<RetainedStorage>())?
            .checked_add(std::mem::size_of::<Error>())?
            .checked_add(std::mem::size_of::<usize>().checked_mul(2)?)? // Rc counts, not allocator overhead
            .checked_mul(3)?
            .try_into()
            .ok()
    }
}

/// Payload fields precede the exact request. Host-slot tokens retain accounting
/// custody only; the original session must also retain the actual state tables.
pub(in crate::composition::mlx::replicated_text) struct NativeOpeningSourceInventory {
    parts: OpeningSourceParts,
    chunk: PrefillChunk,
    epoch: DistributedCommitEpoch,
    request: InferenceRequest,
}
impl NativeOpeningSourceInventory {
    fn validate(&self, context: &PrefillChunkRetentionContext<'_>) -> Result<(), Error> {
        self.request
            .validate_same_request(context.request())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        if self.chunk != *context.chunk() || self.epoch != context.epoch() {
            return Err(error(NativeOpeningSourceError::Coordinates));
        }
        Ok(())
    }

    /// No merged temporary, deduplicated-byte claim or source-origin health
    /// validation. The eventual consumer must join these actual domains with
    /// enclosing sources under its original account and exact scope.
    pub(in crate::composition::mlx::replicated_text) fn visit_domains(
        &self,
        visit: &mut dyn FnMut(&RetainedStorage),
    ) {
        self.parts.visit_domains(visit);
    }

    pub(in crate::composition::mlx::replicated_text) fn is_complete(&self) -> bool {
        self.parts.complete
    }
}
impl NativeOpeningSourceGuard {
    /// Moves the matching inventory once. This is not a retirement, release or
    /// native-work grant. Unknown/error inventories stay pending for recovery.
    pub(in crate::composition::mlx::replicated_text) fn take(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<NativeOpeningSourceInventory, Error> {
        self.slot
            .request
            .validate_same_request(context.request())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let mut pending = self
            .slot
            .pending
            .try_borrow_mut()
            .map_err(|_| error(NativeOpeningSourceError::Busy))?;
        let inventory = pending
            .as_ref()
            .ok_or_else(|| error(NativeOpeningSourceError::Inactive))?;
        inventory.validate(context)?;
        if !inventory.is_complete() {
            return Err(error(NativeOpeningSourceError::Incomplete));
        }
        let inventory = pending.take().expect("validated opening owner");
        drop(pending);
        Ok(inventory)
    }

    /// Terminal recovery move, including a failed or unwound prefix. The caller
    /// must retain this payload under the same original operation recovery.
    pub(in crate::composition::mlx::replicated_text) fn take_for_recovery(
        &mut self,
    ) -> Result<Option<NativeOpeningSourceInventory>, Error> {
        self.slot.active.set(false);
        let mut pending = self
            .slot
            .pending
            .try_borrow_mut()
            .map_err(|_| error(NativeOpeningSourceError::Busy))?;
        let inventory = pending.take();
        drop(pending);
        Ok(inventory)
    }
}

#[derive(Default)]
struct OpeningSourceParts {
    state: Option<RetainedStorage>,
    host: Option<RetainedStorage>,
    parameters: Option<RetainedStorage>,
    execution: RetainedStorage,
    complete: bool,
}
impl OpeningSourceParts {
    fn collect<S: MlxStateMechanisms>(
        &mut self,
        state: &S,
        parameters: impl FnOnce() -> Result<RetainedStorage, Error>,
        execution: impl FnOnce(&mut dyn FnMut(&MlxTensor)) -> bool,
    ) -> Result<(), Error> {
        self.state = Some(state.retained_storage()?);
        self.host = Some(state.retained_host_storage()?);
        self.parameters = Some(parameters()?);
        let mut first_error = None;
        let known = execution(&mut |value| {
            if first_error.is_none() {
                if let Err(cause) = self.execution.include_array(value.as_array()) {
                    first_error = Some(cause);
                }
            }
        });
        if !known {
            self.execution.mark_incomplete();
        }
        if let Some(cause) = first_error {
            return Err(cause.into());
        }
        let mut complete = known;
        let mut bound_error = None;
        self.visit_domains(&mut |domain| match domain.byte_bound() {
            Ok(Some(_)) => {}
            Ok(None) => complete = false,
            Err(cause) => {
                if bound_error.is_none() {
                    bound_error = Some(cause);
                }
            }
        });
        if let Some(cause) = bound_error {
            return Err(cause.into());
        }
        if !complete {
            return Err(error(NativeOpeningSourceError::Incomplete));
        }
        self.complete = true;
        Ok(())
    }

    fn visit_domains(&self, visit: &mut dyn FnMut(&RetainedStorage)) {
        if let Some(state) = &self.state {
            visit(state);
        }
        if let Some(host) = &self.host {
            visit(host);
        }
        if let Some(parameters) = &self.parameters {
            visit(parameters);
        }
        visit(&self.execution);
    }
}

#[cfg(test)]
mod tests;

mod capacity;
mod fixed;

#[cfg(test)]
pub(in crate::composition::mlx::replicated_text) use capacity::{
    OpeningPinFailure, PreparedOpeningPins,
};

#[cfg(test)]
pub(in crate::composition::mlx::replicated_text) use capacity::OpeningPinSetup;

mod rows;
pub(crate) use rows::{
    NativeOpeningRows, NativeOpeningRowsOwner, NativeOpeningRowsPlan, RetiredOpeningRow,
    SealedOpeningRows,
};

pub(in crate::composition::mlx::replicated_text) use rows::InstalledOpeningRows;
