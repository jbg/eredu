//! One request's exact source program keeps parameter, paged-state, initialized
//! input, indexed-constructor and retained-reader allowances independent.
use crate::backend::{
    error::Error,
    nn::workspace::{AddressableInvocation, AddressableQuoteRef},
    runtime::{
        execution::generic::OriginalSelectedResidencyAccess,
        residency::{
            manager::{ForegroundDiskSourceCapacity, ForegroundDiskSourceSeries},
            parameter_bank::IndexedConstructorPartitions,
        },
    },
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError, WorkspaceContext};
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, HostSourceConstructionProgram, OriginalHostSourceBank,
    OriginalHostSourceProgramBanks, WorkingMemoryError,
};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};
fn memory(cause: WorkingMemoryError) -> Error {
    Error::PrefillControl(cause)
}
fn overflow() -> Error {
    Error::WorkspacePlanning(HostMetadataFundingError::Overflow)
}
fn identity() -> Error {
    memory(WorkingMemoryError::IdentityMismatch)
}
fn sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .copied()
        .try_fold(size_of_val(parts), usize::checked_add)
}
fn reserve(funding: &HostMetadataFunding, bytes: Option<usize>) -> Result<(), Error> {
    funding
        .reserve_metadata(bytes.ok_or_else(overflow)?)
        .map_err(Error::WorkspacePlanning)
}
/// Original report coordinates and exact immutable source for one physical use.
#[derive(Debug)]
pub(crate) struct AddressableSourceOccurrence {
    pub(crate) row: usize,
    pub(crate) repetition: usize,
    pub(crate) operation: usize,
    pub(crate) quote: AddressableQuoteRef,
    reader: Option<usize>,
}
struct Reader {
    series: Option<ForegroundDiskSourceSeries>,
}
/// Cold source description. No native read owner or source authority is created.
pub(crate) struct AddressableRequestSourcePlan {
    occurrences: Vec<AddressableSourceOccurrence>,
    readers: Vec<Reader>,
    constructors: Option<HostSourceConstructionFacts>,
    program: HostSourceConstructionProgram,
    target: Option<usize>,
    paged: Option<usize>,
    initialized: Option<usize>,
    funding: HostMetadataFunding,
}
impl AddressableRequestSourcePlan {
    /// Rows are the actual retained reports and their finite physical occurrence
    /// counts. Repeated descriptions still consume distinct constructor calls.
    /// With no indexed rows, this same program composes only the actual target
    /// and paged sources; no constructor component or allowance is invented.
    pub(crate) fn prepare(
        rows: &[(&AddressableInvocation, usize)],
        target: Option<HostSourceConstructionFacts>,
        paged: Option<HostSourceConstructionFacts>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::prepare_with_initialized(rows, target, paged, None, funding)
    }
    /// Initialized integer sources remain a distinct accepted component from
    /// parameters, cache state, indexed constructors and read capacities.
    pub(crate) fn prepare_with_initialized(
        rows: &[(&AddressableInvocation, usize)],
        target: Option<HostSourceConstructionFacts>,
        paged: Option<HostSourceConstructionFacts>,
        initialized: Option<HostSourceConstructionFacts>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let count = rows
            .iter()
            .try_fold(0usize, |sum, (row, repeats)| {
                sum.checked_add(row.occurrences().len().checked_mul(*repeats)?)
            })
            .ok_or_else(overflow)?;
        if count == 0 && target.is_none() && paged.is_none() && initialized.is_none() {
            return Err(identity());
        }
        reserve(funding, Self::planning_control_bytes())?;
        let mut occurrences = funding.metadata_vec(count).map_err(Error::Neural)?;
        // A distinct reader cannot outnumber actual region occurrences. This is
        // the actual paid table capacity; no hidden growth occurs while grouping.
        let mut readers: Vec<Reader> = funding.metadata_vec(count).map_err(Error::Neural)?;
        let (mut bytes, mut attempts, mut partitions) = (0u64, 0usize, 0usize);
        for (row, (invocation, repeats)) in rows.iter().enumerate() {
            for repetition in 0..*repeats {
                for (operation, quote) in invocation.occurrences() {
                    let actual = quote.residency.constructor_facts().ok_or_else(overflow)?;
                    if actual != quote.constructor_facts
                        || quote.residency.read_facts() != quote.read_facts
                    {
                        return Err(identity());
                    }
                    bytes = bytes
                        .checked_add(actual.capacity_bytes())
                        .ok_or_else(overflow)?;
                    attempts = attempts
                        .checked_add(actual.maximum_attempts())
                        .ok_or_else(overflow)?;
                    partitions = partitions
                        .checked_add(actual.maximum_partitions())
                        .ok_or_else(overflow)?;
                    let reader = if quote.residency.requires_reads() {
                        let selected = readers.iter().position(|reader| {
                            reader
                                .series
                                .as_ref()
                                .is_some_and(|series| quote.residency.read_source_matches(series))
                        });
                        let index = match selected {
                            Some(index) => index,
                            None => {
                                let index = readers.len();
                                readers.push(Reader { series: None });
                                index
                            }
                        };
                        quote
                            .residency
                            .include_read_source(&mut readers[index].series)?;
                        if readers[index].series.is_none() {
                            return Err(identity());
                        }
                        Some(index)
                    } else {
                        None
                    };
                    occurrences.push(AddressableSourceOccurrence {
                        row,
                        repetition,
                        operation: *operation,
                        quote: quote.clone(),
                        reader,
                    });
                }
            }
        }
        let constructors = if count == 0 {
            None
        } else {
            Some(HostSourceConstructionFacts::new(bytes, attempts, partitions).map_err(memory)?)
        };
        let components = readers
            .len()
            .checked_add(usize::from(constructors.is_some()))
            .and_then(|n| n.checked_add(usize::from(target.is_some())))
            .and_then(|n| n.checked_add(usize::from(paged.is_some())))
            .and_then(|n| n.checked_add(usize::from(initialized.is_some())))
            .ok_or_else(overflow)?;
        let mut facts = funding.metadata_vec(components).map_err(Error::Neural)?;
        if let Some(constructors) = constructors {
            facts.push(constructors);
        }
        for reader in &readers {
            facts.push(
                reader
                    .series
                    .as_ref()
                    .ok_or_else(identity)?
                    .facts()
                    .ok_or_else(overflow)?,
            );
        }
        let target = target.map(|facts_row| {
            let index = facts.len();
            facts.push(facts_row);
            index
        });
        let paged = paged.map(|facts_row| {
            let index = facts.len();
            facts.push(facts_row);
            index
        });
        let initialized = initialized.map(|facts_row| {
            let index = facts.len();
            facts.push(facts_row);
            index
        });
        let program = HostSourceConstructionProgram::from_components(facts).map_err(memory)?;
        Ok(Self {
            occurrences,
            readers,
            constructors,
            program,
            target,
            paged,
            initialized,
            funding: funding.clone(),
        })
    }
    pub(crate) fn facts(&self) -> HostSourceConstructionFacts {
        self.program.facts()
    }
    pub(crate) fn target_facts(&self) -> Option<HostSourceConstructionFacts> {
        self.target.map(|index| self.program.components()[index])
    }
    pub(crate) fn occurrences(&self) -> &[AddressableSourceOccurrence] {
        &self.occurrences
    }
    /// Wrapper tables, direct chunk directories and exactly one actual capacity
    /// initialization per retained reader. Their tensor/backing facts are already
    /// in the source program and are not added to this metadata account again.
    pub(crate) fn runtime_control_bytes(&self) -> Result<usize, Error> {
        reserve(
            &self.funding,
            Some(size_of::<(
                &Self,
                usize,
                [bool; 3],
                std::array::IntoIter<bool, 3>,
                std::slice::Iter<'_, AddressableSourceOccurrence>,
                std::slice::Iter<'_, Reader>,
                Result<usize, Error>,
            )>()),
        )?;
        let mut bytes = Self::accept_control_bytes()
            .and_then(|n| {
                n.checked_add(WorkspaceContext::metadata_vec_bytes::<ReaderState>(
                    self.readers.len(),
                )?)
            })
            .and_then(|n| {
                n.checked_add(WorkspaceContext::metadata_vec_bytes::<OccurrenceState>(
                    self.occurrences.len(),
                )?)
            })
            .ok_or_else(overflow)?;
        bytes = bytes
            .checked_add(
                Self::take_control_bytes()
                    .and_then(|n| n.checked_mul(self.occurrences.len()))
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        for occurrence in &self.occurrences {
            bytes = bytes
                .checked_add(
                    occurrence
                        .quote
                        .residency
                        .constructor_partition_control_bytes()
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)?;
        }
        for present in [
            self.target.is_some(),
            self.paged.is_some(),
            self.initialized.is_some(),
        ] {
            if present {
                bytes = bytes
                    .checked_add(Self::component_control_bytes().ok_or_else(overflow)?)
                    .ok_or_else(overflow)?;
            }
        }
        for _ in &self.readers {
            bytes = bytes
                .checked_add(ForegroundDiskSourceSeries::control_bytes().ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        Ok(bytes)
    }
    /// A cold speculative source prepays the exact wrapper requirement on its
    /// original metadata account. Rebind only metadata spending; the accepted
    /// source program below still authenticates its independent native owner.
    pub(crate) fn accept_with_funding(
        mut self,
        root: OriginalHostSourceBank,
        funding: &HostMetadataFunding,
    ) -> Result<AddressableRequestSources, Error> {
        self.funding = funding.clone();
        self.accept(root)
    }
    /// Partition only the exact admitted aggregate. Native reader capacities
    /// remain absent until an enclosing original selected access authenticates.
    pub(crate) fn accept(
        self,
        root: OriginalHostSourceBank,
    ) -> Result<AddressableRequestSources, Error> {
        reserve(&self.funding, Self::accept_control_bytes())?;
        let mut program = root
            .partition_program(&self.program)
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let constructors = match self.constructors {
            Some(facts) => {
                let root = program.take(0).map_err(memory)?;
                if !root.matches_facts(facts) {
                    return Err(identity());
                }
                Some(root)
            }
            None => None,
        };
        let mut readers = self
            .funding
            .metadata_vec(self.readers.len())
            .map_err(Error::Neural)?;
        for index in 0..self.readers.len() {
            readers.push(ReaderState {
                root: Some(
                    program
                        .take(index + usize::from(self.constructors.is_some()))
                        .map_err(memory)?,
                ),
                capacity: None,
            });
        }
        let target = match self.target {
            Some(index) => Some(program.take(index).map_err(memory)?),
            None => None,
        };
        let paged = match self.paged {
            Some(index) => Some(program.take(index).map_err(memory)?),
            None => None,
        };
        let initialized = match self.initialized {
            Some(index) => Some(program.take(index).map_err(memory)?),
            None => None,
        };
        let mut spent = self
            .funding
            .metadata_vec(self.occurrences.len())
            .map_err(Error::Neural)?;
        spent.resize(self.occurrences.len(), OccurrenceState::Fresh);
        Ok(AddressableRequestSources {
            plan: self,
            state: RefCell::new(State {
                constructors,
                readers,
                spent,
                target,
                paged,
                initialized,
                _program: program,
            }),
        })
    }
    /// The report/repetition table is in canonical order; this borrowed lookup
    /// creates no occurrence or source authority.
    pub(crate) fn occurrence_range(
        &self,
        row: usize,
        repetition: usize,
    ) -> Option<std::ops::Range<usize>> {
        let start = self
            .occurrences
            .iter()
            .position(|o| o.row == row && o.repetition == repetition)?;
        let count = self.occurrences[start..]
            .iter()
            .take_while(|o| o.row == row && o.repetition == repetition)
            .count();
        Some(start..start.checked_add(count)?)
    }
    fn component_control_bytes() -> Option<usize> {
        Some(size_of::<(
            &AddressableRequestSources,
            Option<OriginalHostSourceBank>,
            Result<Option<OriginalHostSourceBank>, Error>,
            std::cell::RefMut<'_, State>,
        )>())
    }
    fn planning_control_bytes() -> Option<usize> {
        sum(&[
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<AddressableSourceOccurrence>(),
            size_of::<Reader>(),
            size_of::<Vec<Reader>>(),
            size_of::<Vec<AddressableSourceOccurrence>>(),
            size_of::<(
                &[(&AddressableInvocation, usize)],
                Option<HostSourceConstructionFacts>,
                Option<HostSourceConstructionFacts>,
                &HostMetadataFunding,
            )>(),
            size_of::<(
                &[(&AddressableInvocation, usize)],
                Option<HostSourceConstructionFacts>,
                Option<HostSourceConstructionFacts>,
                Option<HostSourceConstructionFacts>,
                &HostMetadataFunding,
            )>(),
            size_of::<[usize; 8]>(),
            size_of::<u64>(),
            size_of::<Option<usize>>(),
            size_of::<HostMetadataFunding>(),
            size_of::<std::slice::Iter<'_, (usize, AddressableQuoteRef)>>(),
            HostSourceConstructionProgram::plan_control_bytes()?,
        ])
    }
    fn accept_control_bytes() -> Option<usize> {
        sum(&[
            size_of::<AddressableRequestSources>(),
            size_of::<State>(),
            size_of::<ReaderState>(),
            size_of::<Result<AddressableRequestSources, Error>>(),
            size_of::<OriginalHostSourceProgramBanks>(),
            size_of::<OriginalHostSourceBank>(),
            size_of::<(Self, OriginalHostSourceBank)>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Option<OriginalHostSourceBank>>() * 3,
            size_of::<(Self, OriginalHostSourceBank, &HostMetadataFunding)>(),
        ])
    }
    fn take_control_bytes() -> Option<usize> {
        sum(&[
            size_of::<(
                &AddressableRequestSources,
                usize,
                &OriginalSelectedResidencyAccess,
            )>(),
            size_of::<std::cell::RefMut<'_, State>>(),
            size_of::<IndexedConstructorPartitions>(),
            size_of::<Option<ForegroundDiskSourceCapacity>>(),
            size_of::<ForegroundDiskSourceCapacity>(),
            size_of::<
                Result<
                    (
                        IndexedConstructorPartitions,
                        Option<ForegroundDiskSourceCapacity>,
                    ),
                    Error,
                >,
            >(),
            size_of::<Option<usize>>(),
            size_of::<usize>(),
            size_of::<bool>(),
        ])
    }
}
struct ReaderState {
    root: Option<OriginalHostSourceBank>,
    capacity: Option<ForegroundDiskSourceCapacity>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum OccurrenceState {
    Fresh,
    Attempted,
    Issued,
}
struct State {
    constructors: Option<OriginalHostSourceBank>,
    readers: Vec<ReaderState>,
    spent: Vec<OccurrenceState>,
    target: Option<OriginalHostSourceBank>,
    paged: Option<OriginalHostSourceBank>,
    initialized: Option<OriginalHostSourceBank>,
    _program: OriginalHostSourceProgramBanks,
}
/// Source bank owner shared by the request's ordered row providers. It retains
/// all unused roots/capacities; no occurrence can replenish another's allowance.
pub(crate) struct AddressableRequestSources {
    plan: AddressableRequestSourcePlan,
    state: RefCell<State>,
}
impl AddressableRequestSources {
    pub(crate) fn occurrences(&self) -> &[AddressableSourceOccurrence] {
        self.plan.occurrences()
    }
    pub(crate) fn occurrence_range(
        &self,
        row: usize,
        repetition: usize,
    ) -> Option<std::ops::Range<usize>> {
        self.plan.occurrence_range(row, repetition)
    }
    /// Exact retained target source description, separate from paged state.
    pub(crate) fn target_facts(&self) -> Option<HostSourceConstructionFacts> {
        self.plan.target_facts()
    }
    /// Move the unchanged target source back to the original operation bank.
    pub(crate) fn take_target(&self) -> Result<Option<OriginalHostSourceBank>, Error> {
        if self.plan.target.is_none() {
            return Ok(None);
        }
        reserve(
            &self.plan.funding,
            AddressableRequestSourcePlan::component_control_bytes(),
        )?;
        let mut state = self.state.try_borrow_mut().map_err(|_| identity())?;
        if self.plan.target.is_some() && state.target.is_none() {
            return Err(memory(WorkingMemoryError::AlreadyStarted));
        }
        Ok(state.target.take())
    }
    /// Return the unchanged paged component to its existing source initializer.
    pub(crate) fn take_paged(&self) -> Result<Option<OriginalHostSourceBank>, Error> {
        if self.plan.paged.is_none() {
            return Ok(None);
        }
        reserve(
            &self.plan.funding,
            AddressableRequestSourcePlan::component_control_bytes(),
        )?;
        let mut state = self.state.try_borrow_mut().map_err(|_| identity())?;
        if self.plan.paged.is_some() && state.paged.is_none() {
            return Err(memory(WorkingMemoryError::AlreadyStarted));
        }
        Ok(state.paged.take())
    }
    /// Move this span's initialized integer source into its control owner.
    pub(crate) fn take_initialized(&self) -> Result<Option<OriginalHostSourceBank>, Error> {
        if self.plan.initialized.is_none() {
            return Ok(None);
        }
        reserve(
            &self.plan.funding,
            AddressableRequestSourcePlan::component_control_bytes(),
        )?;
        let mut state = self.state.try_borrow_mut().map_err(|_| identity())?;
        if state.initialized.is_none() {
            return Err(memory(WorkingMemoryError::AlreadyStarted));
        }
        Ok(state.initialized.take())
    }
    /// Authenticate the actual enclosing scope before creating a read capacity
    /// or taking direct constructor children. A failed physical attempt stays
    /// spent; all surviving original roots remain held by this request owner.
    pub(crate) fn take(
        &self,
        index: usize,
        access: &OriginalSelectedResidencyAccess,
    ) -> Result<
        (
            IndexedConstructorPartitions,
            Option<ForegroundDiskSourceCapacity>,
        ),
        Error,
    > {
        reserve(
            &self.plan.funding,
            AddressableRequestSourcePlan::take_control_bytes(),
        )?;
        let occurrence = self.plan.occurrences.get(index).ok_or_else(identity)?;
        let mut state = self.state.try_borrow_mut().map_err(|_| identity())?;
        access.validate_source_bank(state.constructors.as_ref().ok_or_else(identity)?)?;
        let spent = state.spent.get_mut(index).ok_or_else(identity)?;
        if *spent != OccurrenceState::Fresh {
            return Err(memory(WorkingMemoryError::AlreadyStarted));
        }
        *spent = OccurrenceState::Attempted;
        let constructors = occurrence
            .quote
            .residency
            .partition_constructors_with_funding(
                access,
                state.constructors.as_mut().ok_or_else(identity)?,
                &self.plan.funding,
            )?;
        let reads = if let Some(reader) = occurrence.reader {
            let series = self
                .plan
                .readers
                .get(reader)
                .and_then(|r| r.series.as_ref())
                .ok_or_else(identity)?;
            let slot = state.readers.get_mut(reader).ok_or_else(identity)?;
            if slot.capacity.is_none() {
                reserve(
                    &self.plan.funding,
                    ForegroundDiskSourceSeries::control_bytes(),
                )?;
                let bank = slot.root.take().ok_or_else(identity)?;
                slot.capacity = Some(access.prepare_read_series(series, bank)?);
            }
            Some(slot.capacity.as_ref().ok_or_else(identity)?.clone())
        } else {
            None
        };
        state.spent[index] = OccurrenceState::Issued;
        Ok((constructors, reads))
    }
    /// Read-only source-issuance audit. Native completion remains the owning
    /// region worker's responsibility; a failed source take is not an issuance.
    pub(crate) fn complete(&self) -> Result<bool, Error> {
        let state = self.state.try_borrow().map_err(|_| identity())?;
        Ok(state
            .spent
            .iter()
            .all(|spent| *spent == OccurrenceState::Issued)
            && state.constructors.as_ref().is_none_or(|root| {
                root.remaining_bytes() == 0
                    && root.remaining_attempts() == 0
                    && root.remaining_partitions() == 0
            }))
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
