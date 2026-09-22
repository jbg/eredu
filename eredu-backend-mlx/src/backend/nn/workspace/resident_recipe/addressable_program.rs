//! The complete cold constructor/read program and exact report coordinates.
use super::*;
use crate::backend::{
    error::Error as NativeError,
    submission_recovery::addressable::request_sources::AddressableRequestSourcePlan,
};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, InferenceTextStep, WorkingMemoryError,
};
use std::cell::{Cell, RefCell};
#[derive(Default)]
pub(super) struct RetainedProgram {
    plan: RefCell<Option<AddressableRequestSourcePlan>>,
    inputs: Cell<
        Option<(
            Option<HostSourceConstructionFacts>,
            Option<HostSourceConstructionFacts>,
        )>,
    >,
    facts: Cell<Option<HostSourceConstructionFacts>>,
    controls: Cell<u64>,
    arenas: Cell<u64>,
    taken: Cell<bool>,
}
impl std::fmt::Debug for RetainedProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddressableSourceProgram")
            .field("facts", &self.facts.get())
            .field("taken", &self.taken.get())
            .finish_non_exhaustive()
    }
}
fn memory(cause: WorkingMemoryError) -> NativeError {
    NativeError::PrefillControl(cause)
}
fn identity() -> NativeError {
    memory(WorkingMemoryError::IdentityMismatch)
}
fn overflow() -> NativeError {
    memory(WorkingMemoryError::Overflow)
}
impl ResidentNativeRecipe {
    /// Same report ordering as the request source plan. Counts are physical
    /// executions, one per retained prefill/decode row, never grant multipliers.
    pub(crate) fn prepare_addressable_source_program(
        &self,
        target: Option<HostSourceConstructionFacts>,
        paged: Option<HostSourceConstructionFacts>,
        funding: &HostMetadataFunding,
    ) -> Result<Option<HostSourceConstructionFacts>, NativeError> {
        let state = &self.addressable_program;
        if let Some(inputs) = state.inputs.get() {
            if inputs != (target, paged) || state.taken.get() {
                return Err(identity());
            }
            return Ok(state.facts.get());
        }
        let count = self
            .records
            .iter()
            .filter_map(ResidentSpanRecipe::addressable)
            .count();
        if count == 0 && !(target.is_some() && paged.is_some()) {
            state.inputs.set(Some((target, paged)));
            return Ok(None);
        }
        if self
            .planning_metadata
            .as_ref()
            .is_none_or(|own| !own.same_account(funding))
        {
            return Err(identity());
        }
        funding
            .reserve_metadata(std::mem::size_of::<(
                RetainedProgram,
                Result<Option<HostSourceConstructionFacts>, NativeError>,
                AddressableRequestSourcePlan,
            )>())
            .map_err(NativeError::WorkspacePlanning)?;
        let mut rows = funding.metadata_vec(count).map_err(NativeError::Neural)?;
        let mut controls = 0u64;
        let mut arenas = 0u64;
        for row in self
            .records
            .iter()
            .filter_map(ResidentSpanRecipe::addressable)
        {
            if !row.source().funding().same_account(funding) {
                return Err(identity());
            }
            for (_, quote) in row.occurrences() {
                let native = quote
                    .native_capacity()
                    .graph
                    .checked_add(quote.native_capacity().records)
                    .ok_or_else(overflow)?;
                arenas = arenas
                    .checked_add(u64::try_from(native).map_err(|_| overflow())?)
                    .ok_or_else(overflow)?;
            }
            rows.push((row, 1usize));
        }
        let providers=crate::backend::submission_recovery::addressable::AddressableRequestOwner::runtime_control_bytes(&rows).ok_or_else(overflow)?;
        controls = controls
            .checked_add(u64::try_from(providers).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        let plan = AddressableRequestSourcePlan::prepare(&rows, target, paged, funding)?;
        controls = controls
            .checked_add(u64::try_from(plan.runtime_control_bytes()?).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        let facts = plan.facts();
        let mut destination = state.plan.try_borrow_mut().map_err(|_| identity())?;
        if destination.is_some() {
            return Err(identity());
        }
        *destination = Some(plan);
        state.inputs.set(Some((target, paged)));
        state.facts.set(Some(facts));
        state.controls.set(controls);
        state.arenas.set(arenas);
        Ok(Some(facts))
    }
    pub(crate) fn addressable_source_facts(&self) -> Option<HostSourceConstructionFacts> {
        self.addressable_program.facts.get()
    }
    pub(crate) fn take_addressable_source_program(
        &self,
    ) -> Result<Option<AddressableRequestSourcePlan>, NativeError> {
        let state = &self.addressable_program;
        if state.inputs.get().is_none() {
            if self.records.iter().any(|row| row.addressable().is_some()) {
                return Err(memory(WorkingMemoryError::UnknownBound));
            }
            return Ok(None);
        }
        if state.taken.replace(true) {
            return Err(identity());
        }
        state
            .plan
            .try_borrow_mut()
            .map_err(|_| identity())
            .map(|mut value| value.take())
    }
    /// These native arenas belong to each addressable role, outside the model
    /// graph. Its backing is already present in the model's mutable population.
    pub(crate) fn addressable_request_control_bytes(&self) -> Result<u64, NativeError> {
        let state = &self.addressable_program;
        if self.records.iter().any(|row| row.addressable().is_some())
            && state.inputs.get().is_none()
        {
            return Err(memory(WorkingMemoryError::UnknownBound));
        }
        state
            .controls
            .get()
            .checked_add(state.arenas.get())
            .ok_or_else(overflow)
    }
    fn addressable_for_row(
        &self,
        selected: &ResidentSpanRecipe,
    ) -> Result<Option<(usize, AddressableInvocation)>, NativeError> {
        if selected.addressable().is_none() {
            return Ok(None);
        }
        let (ordinal, invocation) = self
            .records
            .iter()
            .filter_map(|row| row.addressable().map(|source| (row, source)))
            .enumerate()
            .find_map(|(ordinal, (row, source))| {
                std::ptr::eq(row, selected).then_some((ordinal, source))
            })
            .ok_or_else(identity)?;
        Ok(Some((
            ordinal,
            invocation
                .try_clone_for_retention()
                .map_err(NativeError::Neural)?,
        )))
    }
    pub(crate) fn addressable_for_prefill(
        &self,
        role: eredu_runtime::prefill::PrefillControlRole,
    ) -> Result<Option<(usize, AddressableInvocation)>, NativeError> {
        let eredu_runtime::prefill::PrefillControlRole::Span {
            input_start,
            input_end,
            position,
            output,
            ..
        } = role
        else {
            return Err(NativeError::PrefillScopeUnavailable);
        };
        let row=self.records.iter().find(|row|matches!(&row.span,InferenceWorkspaceSpan::Prefill(chunk)
            if chunk.input.start==input_start&&chunk.input.end==input_end&&chunk.position==position&&chunk.output==output)).ok_or(NativeError::PrefillScopeUnavailable)?;
        self.addressable_for_row(row)
    }
    pub(crate) fn addressable_for_step(
        &self,
        step: &InferenceTextStep,
    ) -> Result<Option<(usize, AddressableInvocation)>, NativeError> {
        if step.request().geometry() != self.plan.geometry() {
            return Err(NativeError::PrefillScopeUnavailable);
        }
        let index = step
            .attempt()
            .checked_sub(1)
            .ok_or(NativeError::PrefillScopeUnavailable)?;
        let row=self.records.iter().find(|row|matches!(&row.span,InferenceWorkspaceSpan::Decode{index:actual,..}if *actual==index)).ok_or(NativeError::PrefillScopeUnavailable)?;
        self.addressable_for_row(row)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
