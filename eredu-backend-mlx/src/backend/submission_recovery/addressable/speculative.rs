//! Per-equation source programs consumed after the existing speculative claim.
//! Descriptions and accepted roots remain separate; every actual prefill span
//! gets its own one-use constructor/read program and its exact target child.
use super::request_sources::{AddressableRequestSourcePlan, AddressableRequestSources};
use super::{AddressableExecutionRow, AddressableRequestOwner};
use crate::backend::{
    error::Error,
    nn::workspace::{AddressableInvocation, ResidentSpanRecipe},
    runtime::execution::generic::OriginalSelectedResidencyAccess,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, OriginalHostSourceBank, SpeculativeHostSourceSpans,
    WorkingMemoryError,
};
use safemlx::{OriginalBufferBudget, OriginalScopeObserver};
use std::{
    cell::{Cell, RefCell},
    mem::{size_of, size_of_val},
};
fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}
fn identity() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn sum(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .copied()
        .try_fold(size_of_val(values), usize::checked_add)
}
struct Row {
    taken: Cell<bool>,
    value: RefCell<Option<SpeculativeAddressableSpan>>,
}
pub(crate) struct SpeculativeAddressableSources {
    rows: Vec<Row>,
    facts: Option<SpeculativeHostSourceSpans>,
    controls: u64,
    _funding: HostMetadataFunding,
}
pub(crate) struct SpeculativeAddressableSpan {
    plan: Option<AddressableRequestSourcePlan>,
    invocation: Option<AddressableInvocation>,
    accepted: Option<AddressableRequestSources>,
    runtime_funding: HostMetadataFunding,
    funding: HostMetadataFunding,
}
impl SpeculativeAddressableSources {
    pub(crate) fn prepare(
        records: &[ResidentSpanRecipe],
        target: Option<HostSourceConstructionFacts>,
        funding: &HostMetadataFunding,
    ) -> Result<Option<Self>, Error> {
        Self::prepare_with_paged(records, target, None, funding)
    }
    /// The paged program spans the actual invocation and is constructed once,
    /// from its first row's admitted bank. Other source components remain per row.
    pub(crate) fn prepare_with_paged(
        records: &[ResidentSpanRecipe],
        target: Option<HostSourceConstructionFacts>,
        paged: Option<HostSourceConstructionFacts>,
        funding: &HostMetadataFunding,
    ) -> Result<Option<Self>, Error> {
        if paged.is_none()
            && !records
                .iter()
                .any(|row| row.addressable().is_some() || row.parallel().is_some())
        {
            return Ok(None);
        }
        let fixed = sum(&[
            size_of::<Self>(),
            size_of::<Result<Option<Self>, Error>>(),
            size_of::<(
                &[ResidentSpanRecipe],
                Option<HostSourceConstructionFacts>,
                &HostMetadataFunding,
            )>(),
            size_of::<(
                &[ResidentSpanRecipe],
                Option<HostSourceConstructionFacts>,
                Option<HostSourceConstructionFacts>,
                &HostMetadataFunding,
            )>(),
            size_of::<Vec<Row>>(),
            size_of::<SpeculativeHostSourceSpans>(),
            size_of::<AddressableRequestSourcePlan>(),
            size_of::<SpeculativeAddressableSpan>(),
            size_of::<Option<AddressableInvocation>>(),
            size_of::<[(&AddressableInvocation, usize); 1]>(),
            size_of::<&[(&AddressableInvocation, usize)]>(),
            size_of::<Option<HostSourceConstructionFacts>>(),
            size_of::<Result<Option<HostSourceConstructionFacts>, Error>>(),
            size_of::<bool>(),
            size_of::<usize>() * 5,
            size_of::<u64>() * 3,
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, ResidentSpanRecipe>>>(),
        ])
        .ok_or_else(overflow)?;
        funding
            .reserve_metadata(fixed)
            .map_err(Error::WorkspacePlanning)?;
        if records.is_empty() {
            return Err(identity());
        }
        let mut rows = funding.metadata_vec(records.len()).map_err(Error::Neural)?;
        let mut facts =
            SpeculativeHostSourceSpans::new(records.len(), funding).map_err(Error::Neural)?;
        let mut controls = 0u64;
        let mut any_component = false;
        for (ordinal, record) in records.iter().enumerate() {
            let paged = if ordinal == 0 { paged } else { None };
            let invocation = record.addressable();
            let initialized = record.initialized_input_source_facts()?;
            let value = if invocation.is_some() || paged.is_some() || initialized.is_some() {
                any_component = true;
                let selected;
                let selected = match invocation {
                    Some(invocation) => {
                        selected = [(invocation, 1usize)];
                        &selected[..]
                    }
                    None => &[],
                };
                let plan = AddressableRequestSourcePlan::prepare_with_initialized(
                    selected,
                    target,
                    paged,
                    initialized,
                    funding,
                )?;
                facts
                    .push(Some(plan.facts()))
                    .map_err(Error::PrefillControl)?;
                let wrappers = match invocation {
                    Some(_) => AddressableRequestOwner::prepared_runtime_control_bytes(selected)
                        .ok_or_else(overflow)?,
                    None => 0,
                };
                let plan_controls = plan.runtime_control_bytes()?;
                let wrappers = wrappers
                    .checked_add(SpeculativeAddressableSpan::control_bytes().ok_or_else(overflow)?)
                    .and_then(|n| n.checked_add(plan_controls))
                    .ok_or_else(overflow)?;
                let runtime_funding = prepare_wrapper_funding(wrappers, funding)?;
                // Graph/Record owners are allocated by each dedicated region;
                // Buffer births already belong to its numerical population.
                if let Some(invocation) = invocation {
                    for (_, quote) in invocation.occurrences() {
                        let arenas = quote
                            .capacity
                            .graph
                            .checked_add(quote.capacity.records)
                            .ok_or_else(overflow)?;
                        controls = controls
                            .checked_add(u64::try_from(arenas).map_err(|_| overflow())?)
                            .ok_or_else(overflow)?;
                    }
                }
                Some(SpeculativeAddressableSpan {
                    plan: Some(plan),
                    invocation: invocation
                        .map(|source| source.try_clone_for_retention())
                        .transpose()
                        .map_err(Error::Neural)?,
                    accepted: None,
                    runtime_funding,
                    funding: funding.clone(),
                })
            } else {
                facts.push(target).map_err(Error::PrefillControl)?;
                None
            };
            rows.push(Row {
                taken: Cell::new(false),
                value: RefCell::new(value),
            });
        }
        if !any_component {
            return Ok(None);
        }
        Ok(Some(Self {
            rows,
            facts: Some(facts),
            controls,
            _funding: funding.clone(),
        }))
    }
    /// Native child Graph/Record storage only. Wrapper metadata is prepaid
    /// on its retained request account before this source can reach admission.
    pub(crate) fn controls(&self) -> u64 {
        self.controls
    }
    pub(crate) fn take_facts(&mut self) -> Result<SpeculativeHostSourceSpans, Error> {
        self.facts.take().ok_or_else(identity)
    }
    pub(crate) fn take(&self, index: usize) -> Result<Option<SpeculativeAddressableSpan>, Error> {
        let row = self.rows.get(index).ok_or_else(identity)?;
        if row.taken.replace(true) {
            return Err(identity());
        }
        row.value
            .try_borrow_mut()
            .map_err(|_| identity())
            .map(|mut value| value.take())
    }
}
impl SpeculativeAddressableSpan {
    pub(crate) fn has_addressable(&self) -> bool {
        self.invocation.is_some()
    }
    pub(crate) fn take_paged(&self) -> Result<Option<OriginalHostSourceBank>, Error> {
        self.accepted.as_ref().ok_or_else(identity)?.take_paged()
    }
    pub(crate) fn take_initialized(&self) -> Result<Option<OriginalHostSourceBank>, Error> {
        self.accepted
            .as_ref()
            .ok_or_else(identity)?
            .take_initialized()
    }
    fn control_bytes() -> Option<usize> {
        sum(&[
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Option<AddressableRequestSources>>(),
            size_of::<AddressableRequestSources>(),
            size_of::<Option<AddressableRequestSourcePlan>>(),
            size_of::<Option<OriginalHostSourceBank>>(),
            size_of::<Result<Option<OriginalHostSourceBank>, Error>>(),
            size_of::<(&mut Self, Option<OriginalHostSourceBank>)>(),
            // Paged and initialized components have independent call/results.
            size_of::<(&Self, Result<Option<OriginalHostSourceBank>, Error>)>(),
            size_of::<&Self>(),
            size_of::<bool>(),
            size_of::<(
                Self,
                OriginalSelectedResidencyAccess,
                OriginalBufferBudget,
                &OriginalScopeObserver,
                Option<std::time::Duration>,
                &HostMetadataFunding,
            )>(),
            size_of::<AddressableRequestOwner>(),
            size_of::<AddressableExecutionRow>(),
            size_of::<Result<AddressableExecutionRow, Error>>(),
            size_of::<&mut Self>(),
        ])
    }
    /// Called only by the selected operation bank after its cumulative claim.
    pub(crate) fn accept(
        &mut self,
        root: Option<OriginalHostSourceBank>,
    ) -> Result<Option<OriginalHostSourceBank>, Error> {
        self.runtime_funding
            .reserve_metadata(Self::control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if self.accepted.is_some() {
            return Err(identity());
        }
        let source = self
            .plan
            .take()
            .ok_or_else(identity)?
            .accept_with_funding(root.ok_or_else(identity)?, &self.runtime_funding)?;
        let target = source.take_target()?;
        self.accepted = Some(source);
        Ok(target)
    }
    pub(crate) fn activate(
        self,
        access: OriginalSelectedResidencyAccess,
        budget: OriginalBufferBudget,
        observer: &OriginalScopeObserver,
        timeout: Option<std::time::Duration>,
        funding: &HostMetadataFunding,
    ) -> Result<AddressableExecutionRow, Error> {
        if self.plan.is_some() || !self.funding.same_account(funding) {
            return Err(identity());
        }
        let owner = AddressableRequestOwner::new_prepared(
            self.accepted.ok_or_else(identity)?,
            access,
            budget,
            observer,
            timeout,
            &self.runtime_funding,
            funding,
        )?;
        owner.row(0, 0, self.invocation.ok_or_else(identity)?)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

/// Same shared metadata workers, bounded by their owning prospective query.
/// The child spends its prepaid partition without another ledger reservation.
pub(crate) fn prepare_wrapper_funding(
    bytes: usize,
    parent: &HostMetadataFunding,
) -> Result<HostMetadataFunding, Error> {
    let limit = bytes
        .checked_add(HostMetadataFunding::prepaid_control_bytes().ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    let frames = [
        size_of::<(usize, &HostMetadataFunding)>(),
        size_of::<HostMetadataFunding>(),
        size_of::<eredu_core::HostPreparationAuthority>(),
        size_of::<Result<HostMetadataFunding, Error>>(),
        size_of::<Result<HostMetadataFunding, HostMetadataFundingError>>(),
        size_of::<usize>(),
        eredu_core::HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
            .ok_or_else(overflow)?,
    ];
    let charge = sum(&frames)
        .and_then(|n| n.checked_add(limit))
        .ok_or_else(overflow)?;
    parent
        .reserve_metadata(charge)
        .map_err(Error::WorkspacePlanning)?;
    HostMetadataFunding::from_prepaid(
        limit,
        eredu_core::HostPreparationAuthority::retain(parent.clone()),
    )
    .map_err(Error::WorkspacePlanning)
}
