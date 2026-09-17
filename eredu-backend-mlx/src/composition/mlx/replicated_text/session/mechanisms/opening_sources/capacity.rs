//! One lexical source topology, sealed before original admission. No publication.
use super::fixed::{FixedOpeningOwners, OpeningEntry, OpeningError};
use super::*;
use crate::backend::runtime::residency::manager::{
    PreparedWeightOwnerSlotBounds, ResidencyManager,
};
use eredu_nn::Parameterized;
use eredu_runtime::ArchitectureParameters;
use eredu_runtime::{
    layered::{BoundCaptureSelection, PreparedLayeredObservationPaths},
    parameter_operations::LayeredParameterOwner,
    working_memory::{
        CapturePlanStorageKey, IncrementalInferenceQuote, InferenceSpanWorkspacePlan,
        OwnedTextSpanWorkspace, PreparedCapturePlanPublication, PreparedPrefillStoragePinPlan,
        PreparedTextControlWorkspace, ResidualQuoteError, TextHostControlFacts,
    },
    LayerwisePolicy, LayerwiseRuntime, ResidentRuntime,
};
use std::mem::size_of;

pub(super) mod registered;
pub(in crate::composition::mlx::replicated_text) use registered::{
    OpeningPinFailure, PreparedOpeningPins,
};

mod paired;

/// Not implemented by external callers. The actual prepared runtime token must
/// match; equal semantic declarations from another runtime are insufficient.
pub(super) trait OpeningExecution<S: MlxStateMechanisms> {
    type Architecture: LayeredArchitecture<MlxNeuralBackend, S>;
    fn owner_slot_bound(&self) -> Option<usize>;
    fn visit_owners(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool;
    fn validate_paths(&self, paths: &PreparedLayeredObservationPaths) -> Result<(), OpeningError>;
}
impl<A, S> OpeningExecution<S> for ResidentRuntime<A, MlxNeuralBackend, S>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S>,
{
    type Architecture = A;
    fn owner_slot_bound(&self) -> Option<usize> {
        self.units().iter().flatten().try_fold(
            self.architecture().retained_static_value_slot_bound()?,
            |sum, unit| sum.checked_add(unit.retained_value_slot_bound()?),
        )
    }
    fn visit_owners(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        let mut complete = self.architecture().visit_retained_static_values(visitor);
        for unit in self.units().iter().flatten() {
            complete &= unit.visit_retained_values(visitor);
        }
        complete
    }
    fn validate_paths(&self, paths: &PreparedLayeredObservationPaths) -> Result<(), OpeningError> {
        self.validate_observation_binding(paths)
            .map_err(|_| OpeningError::Identity)
    }
}
impl<A, S, P> OpeningExecution<S> for LayerwiseRuntime<A, MlxNeuralBackend, S, P>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    P: LayerwisePolicy<MlxNeuralBackend, A::Unit>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    type Architecture = A;
    fn owner_slot_bound(&self) -> Option<usize> {
        self.retained_value_slot_bound()
    }
    fn visit_owners(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        self.visit_retained_values(visitor)
    }
    fn validate_paths(&self, paths: &PreparedLayeredObservationPaths) -> Result<(), OpeningError> {
        self.validate_observation_binding(paths)
            .map_err(|_| OpeningError::Identity)
    }
}

impl<A, S> MlxReplicatedTextMechanisms<A, S>
where
    S: MlxStateMechanisms,
    A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
{
    /// Closed native role selection: the mechanisms lend their own physical
    /// checkpoint source and actual prepared weight manager. The future paired
    /// runtime inspector must lend the same session's state and execution.
    pub(super) fn prepare_fixed_opening_capacity<'a, E>(
        &'a self,
        state: &'a S,
        execution: &'a E,
        selection: BoundCaptureSelection<'a>,
        paths: &'a PreparedLayeredObservationPaths,
    ) -> Result<NativeOpeningCapacityPlan<'a, S, E>, OpeningError>
    where
        E: OpeningExecution<S, Architecture = A>,
    {
        let weights = self
            .residency_manager
            .as_ref()
            .ok_or(OpeningError::Unknown)?;
        NativeOpeningCapacityPlan::prepare(
            state,
            execution,
            self.store.as_ref(),
            Some(weights),
            selection,
            paths,
        )
    }
}

/// Handle counts, including duplicate/zero owners. These are not payload bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct OpeningSlots {
    pub(super) arrays: usize,
    pub(super) hosts: usize,
    pub(super) bytes: usize,
    pub(super) sources: usize,
    pub(super) layouts: usize,
    pub(super) tables: usize,
}

/// Non-Clone, allocation-free diagnostic retaining the exact borrowed owners.
/// The borrow prohibits state/execution replacement for this lexical mechanism;
/// it is not a persistent installation/rebinding contract. Interior mutability
/// still obeys each source's slot contract and the retaining visitor's checks.
pub(super) struct NativeOpeningCapacityPlan<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> {
    pub(super) state: &'a S,
    pub(super) execution: &'a E,
    pub(super) store: &'a dyn CheckpointSource,
    pub(super) weights: Option<PreparedWeightOwnerSlotBounds<'a>>,
    pub(super) selection: BoundCaptureSelection<'a>,
    pub(super) paths: &'a PreparedLayeredObservationPaths,
    pub(super) slots: OpeningSlots,
}
impl<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> NativeOpeningCapacityPlan<'a, S, E> {
    /// Actual owners only; no supplied counts, byte certificate or guard. The
    /// enclosing native adapter must lend the mechanisms' own store/manager and
    /// the state/execution paired by its existing quiescent session inspection.
    /// No new session inspector or capture gate is installed by this unit.
    fn prepare(
        state: &'a S,
        execution: &'a E,
        store: &'a dyn CheckpointSource,
        weights: Option<&'a ResidencyManager>,
        selection: BoundCaptureSelection<'a>,
        paths: &'a PreparedLayeredObservationPaths,
    ) -> Result<Self, OpeningError> {
        execution.validate_paths(paths)?;
        if !paths.source().same_storage(selection.selection().paths()) {
            return Err(OpeningError::Identity);
        }
        state
            .validate_text_frontier(selection.geometry().cached_positions)
            .map_err(OpeningError::StateOwner)?;
        let state_slots = state
            .retained_owner_slot_counts()
            .ok_or(OpeningError::Unknown)?;
        // No record-growth or exact future-acquisition contract exists here.
        if state_slots.manager_roles != 0 {
            return Err(OpeningError::CacheCatalog);
        }
        let execution_slots = execution.owner_slot_bound().ok_or(OpeningError::Unknown)?;
        let source_slots = store
            .source_storage_slot_bound()?
            .ok_or(OpeningError::Unknown)?;
        let weights = weights
            .map(ResidencyManager::prepare_owner_slot_bounds)
            .transpose()?
            .map(|p| p.ok_or(OpeningError::Unknown))
            .transpose()?;
        let mut slots = OpeningSlots {
            arrays: state_slots
                .arrays
                .checked_add(execution_slots)
                .ok_or(OpeningError::Overflow)?,
            sources: source_slots,
            layouts: state_slots.layouts,
            tables: state_slots.slot_tables,
            ..OpeningSlots::default()
        };
        if let Some(w) = &weights {
            slots.arrays = slots
                .arrays
                .checked_add(w.array_slots())
                .ok_or(OpeningError::Overflow)?;
            slots.hosts = w.host_slots();
            slots.sources = slots
                .sources
                .checked_add(w.source_slots())
                .ok_or(OpeningError::Overflow)?;
        }
        let plan = Self {
            state,
            execution,
            store,
            weights,
            selection,
            paths,
            slots,
        };
        plan.control_peak_bytes()?;
        Ok(plan)
    }

    /// Concrete buffers, fact slots, borrowed bindings and construction/error
    /// controls only. Numeric owners/source catalogs/enclosing models are not
    /// relabeled as these controls. The optional pin bank prices its own buffers.
    pub(super) fn control_peak_bytes(&self) -> Result<u64, OpeningError> {
        let controls = size_of::<Self>()
            .checked_add(size_of::<OpeningControlProposal<'a, S, E>>())
            .and_then(|n| n.checked_add(size_of::<SealedOpeningCapacity<'a, S, E>>()))
            .and_then(|n| n.checked_add(size_of::<PreparedOpeningInventory<'a, S, E>>()))
            .and_then(|n| n.checked_add(size_of::<OpeningAllocationFailure<'a, S, E>>()))
            .and_then(|n| n.checked_add(size_of::<OpeningError>()))
            .and_then(|n| n.checked_add(size_of::<OpeningCollectionFailure>()))
            .and_then(|n| n.checked_add(size_of::<OpeningEntry<'_>>()))
            .and_then(|n| n.checked_add(registered::wrapper_control_bytes()))
            .and_then(|n| {
                n.checked_add(size_of::<registered::OpeningPinPreparationFailure<'a, S, E>>())
            })
            .and_then(|n| n.checked_add(size_of::<paired::PairedOpeningError>()))
            .and_then(|n| n.checked_mul(3))
            .ok_or(OpeningError::Overflow)?;
        u64::try_from(controls)
            .map_err(|_| OpeningError::Overflow)?
            .checked_add(FixedOpeningOwners::storage_bytes(self.slots)?)
            .ok_or(OpeningError::Overflow)
    }

    /// Consume the sole native plan while producing the exact original control
    /// diagnostic. The caller seals THIS receipt into its existing text quote
    /// before reservation. A later equal-content receipt does not match.
    pub(super) fn seal(
        self,
        span: &InferenceSpanWorkspacePlan,
        existing: TextHostControlFacts,
    ) -> Result<OpeningControlProposal<'a, S, E>, OpeningError> {
        let extra = self.control_peak_bytes()?;
        let carrier = existing
            .carrier_bytes()
            .map(|n| n.checked_add(extra).ok_or(OpeningError::Overflow))
            .transpose()?;
        let facts =
            TextHostControlFacts::new(existing.admission_bytes(), carrier, existing.work_bytes());
        let controls = PreparedTextControlWorkspace::prepare(
            self.selection.selection().source(),
            self.selection.geometry(),
            span,
            facts,
        )?
        .with_prefill_capture_selection(self.selection)?
        .with_preparation_scopes(
            crate::composition::mlx::session::preparation_scope_facts()
                .map_err(|_| OpeningError::Overflow)?,
        )?;
        let controls = controls.with_prediction_scopes(
            crate::composition::mlx::session::prediction_scope_facts()
                .map_err(|_| OpeningError::Overflow)?,
        )?;
        Ok(OpeningControlProposal {
            plan: self,
            controls,
            opening_pins: false,
        })
    }
}

/// Specific consuming enrichments preserve this plan's original binding identity.
pub(super) struct OpeningControlProposal<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> {
    plan: NativeOpeningCapacityPlan<'a, S, E>,
    controls: PreparedTextControlWorkspace,
    opening_pins: bool,
}
impl<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> OpeningControlProposal<'a, S, E> {
    pub(super) fn with_generation_sequence(
        mut self,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
    ) -> Result<Self, OpeningError> {
        self.controls = self.controls.with_generation_sequence(claim)?;
        Ok(self)
    }

    /// Derive the existing-only bank from this actual native topology and span.
    /// The returned snapshot is not later current-state or readiness evidence.
    pub(super) fn with_existing_opening_pins(mut self) -> Result<Self, OpeningError> {
        let count = registered::descriptor_count(self.plan.slots)?;
        let pins = PreparedPrefillStoragePinPlan::<registered::Key>::prepare(
            self.controls.plan(),
            |_| Some(count),
        )?;
        self.controls = self.controls.with_prefill_storage_pins(pins)?;
        self.opening_pins = true;
        Ok(self)
    }
    pub(super) fn with_prefill_storage_pins<K: Ord + Send + 'static>(
        mut self,
        pins: PreparedPrefillStoragePinPlan<K>,
    ) -> Result<Self, OpeningError> {
        self.controls = self.controls.with_prefill_storage_pins(pins)?;
        Ok(self)
    }
    pub(super) fn with_capture_plan_publication<K: CapturePlanStorageKey>(
        mut self,
        publication: PreparedCapturePlanPublication<K>,
    ) -> Result<Self, OpeningError> {
        self.controls = self.controls.with_capture_plan_publication(publication)?;
        Ok(self)
    }
    /// Existing one-time quote sealing rejects a sealed/accepted input. There
    /// is no arbitrary final-control replacement or post-admission repair.
    pub(super) fn finish(
        self,
        quote: IncrementalInferenceQuote,
    ) -> Result<(SealedOpeningCapacity<'a, S, E>, IncrementalInferenceQuote), ResidualQuoteError>
    {
        let quote = quote.with_span_workspace_and_text_controls(self.controls.clone())?;
        Ok((
            SealedOpeningCapacity {
                plan: self.plan,
                controls: self.controls,
                opening_pins: self.opening_pins,
            },
            quote,
        ))
    }
}

pub(super) struct SealedOpeningCapacity<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> {
    plan: NativeOpeningCapacityPlan<'a, S, E>,
    controls: PreparedTextControlWorkspace,
    opening_pins: bool,
}
pub(super) struct OpeningAllocationFailure<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> {
    pub(super) cause: OpeningError,
    pub(super) plan: SealedOpeningCapacity<'a, S, E>,
}
impl<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> SealedOpeningCapacity<'a, S, E> {
    pub(super) fn allocate(
        self,
        accepted: &OwnedTextSpanWorkspace,
    ) -> Result<PreparedOpeningInventory<'a, S, E>, OpeningAllocationFailure<'a, S, E>> {
        let validate = || {
            let actual = accepted
                .workspace()
                .text_controls()
                .ok_or(OpeningError::Identity)?;
            if !self.controls.same_binding(actual) {
                return Err(OpeningError::Identity);
            }
            self.plan.execution.validate_paths(self.plan.paths)?;
            accepted
                .control_guard()
                .validate_reservation(accepted.reservation())?;
            Ok(())
        };
        if let Err(cause) = validate() {
            return Err(OpeningAllocationFailure { plan: self, cause });
        }
        // Custody exists before the first buffer allocation and drops last,
        // including partial construction failure. No owners have been visited.
        let mut inventory = PreparedOpeningInventory {
            owners: FixedOpeningOwners::empty(self.plan.slots),
            plan: self,
            reservation: accepted.reservation().clone(),
            custody: accepted.control_guard(),
        };
        if let Err(cause) = inventory.owners.reserve().and_then(|()| {
            inventory
                .custody
                .validate_reservation(&inventory.reservation)
                .map_err(Into::into)
        }) {
            let PreparedOpeningInventory {
                owners,
                plan,
                reservation,
                custody,
            } = inventory;
            drop(owners);
            drop(reservation);
            // The returned plan aliases the aggregate custody via span records.
            drop(custody);
            return Err(OpeningAllocationFailure { plan, cause });
        }
        Ok(inventory)
    }
}

/// One owner capsule; no Clone, reset, raw owning export or second allocation.
/// All payloads precede the original host custody on success, failure and unwind.
pub(super) struct PreparedOpeningInventory<'a, S: MlxStateMechanisms, E: OpeningExecution<S>> {
    pub(super) owners: FixedOpeningOwners,
    plan: SealedOpeningCapacity<'a, S, E>,
    reservation: eredu_runtime::working_memory::WorkingMemoryReservation,
    custody: eredu_runtime::working_memory::OriginalTextControlGuard,
}
/// Escaped diagnostics keep original control custody without owning the
/// capsule payload. The original cause is dropped before the last guard.
#[derive(Debug)]
pub(super) struct OpeningCollectionFailure {
    pub(super) cause: OpeningError,
    custody: eredu_runtime::working_memory::OriginalTextControlGuard,
}
impl std::fmt::Display for OpeningCollectionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("native opening collection failed")
    }
}
impl std::error::Error for OpeningCollectionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl<S: MlxStateMechanisms, E: OpeningExecution<S>> PreparedOpeningInventory<'_, S, E> {
    pub(super) fn collect(&mut self) -> Result<(), OpeningCollectionFailure> {
        self.collect_inner()
            .map_err(|cause| OpeningCollectionFailure {
                cause,
                custody: self.custody.clone(),
            })
    }
    fn collect_inner(&mut self) -> Result<(), OpeningError> {
        self.owners.begin()?;
        self.custody.validate_reservation(&self.reservation)?;
        let p = &self.plan.plan;
        p.execution.validate_paths(p.paths)?;
        p.state
            .validate_text_frontier(p.selection.geometry().cached_positions)
            .map_err(OpeningError::StateOwner)?;
        let mut first = None;
        let state_result = p.state.visit_all_retained_values(&mut |v| {
            if first.is_none() {
                first = self.owners.retain_array(v.as_array()).err();
            }
        });
        if let Some(error) = first {
            return Err(error);
        }
        state_result?;
        if let Some(layout) = p.state.shared_layout() {
            self.owners.retain_layout(layout)?;
        }
        let mut first = None;
        let metadata_result = p.state.visit_slot_metadata(&mut |v| {
            if first.is_none() {
                first = self.owners.retain_table(v).err();
            }
            Ok(())
        });
        if let Some(error) = first {
            return Err(error);
        }
        metadata_result.map_err(OpeningError::StateOwner)?;
        self.owners.retain_checkpoint(p.store)?;
        if let Some(weights) = &p.weights {
            let complete = weights
                .source()
                .try_visit_retained_storage(&mut |v| self.owners.retain(v))?;
            if !complete {
                return Err(OpeningError::Incomplete);
            }
        }
        let mut first = None;
        let complete = p.execution.visit_owners(&mut |v| {
            if first.is_none() {
                first = self.owners.retain_array(v.as_array()).err();
            }
        });
        if let Some(error) = first {
            return Err(error);
        }
        if !complete {
            return Err(OpeningError::Incomplete);
        }
        // No manager, state-table or execution visitor loan remains here.
        self.owners.inspect_facts()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(in crate::composition::mlx::replicated_text) use registered::tests::OpeningPinSetup;
