//! Request population for the actual foreground disk read constructors.
use super::*;
use crate::backend::runtime::residency::manager::{
    ForegroundDiskPopulation, ForegroundDiskSourceCapacity, ForegroundDiskWindowPlan,
};
use eredu_runtime::working_memory::WorkingMemoryPool;
use std::mem::size_of_val;
mod background;
pub(in crate::backend::runtime::execution::generic::original_operations) use background::BackgroundSelection;
use background::BackgroundRequestPlan;

pub(crate) struct ForegroundDiskRequestPlan {
    windows: Vec<ForegroundDiskWindowPlan>,
    pool: WorkingMemoryPool,
    forwards: usize,
    temporary_bytes: usize,
    nested_source_peak: Option<usize>,
    background: Option<BackgroundRequestPlan>,
}
impl ForegroundDiskRequestPlan {
    pub(crate) fn new(
        manager: &ResidencyManager,
        pool: &WorkingMemoryPool,
        windows: &[crate::backend::runtime::residency::manager::WindowPopulation],
        ids: &[OffloadUnitId],
        population: ResidencyPopulation,
    ) -> Result<Self, Error> {
        Self::new_with_metadata(manager, pool, windows, ids, population, None)
    }
    pub(in crate::backend::runtime::execution::generic) fn new_with_metadata(
        manager: &ResidencyManager,
        pool: &WorkingMemoryPool,
        windows: &[crate::backend::runtime::residency::manager::WindowPopulation],
        ids: &[OffloadUnitId],
        population: ResidencyPopulation,
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<Self, Error> {
        if let Some(funding) = funding {
            let bytes = [
                Layout::array::<eredu_runtime::residency::ResidencyClosureSlot>(
                    population.controller_units,
                )
                .map_err(|_| overflow())?
                .size(),
                Layout::array::<ForegroundDiskWindowPlan>(windows.len())
                    .map_err(|_| overflow())?
                    .size(),
                size_of::<Self>(),
                size_of::<Result<Self, Error>>(),
                size_of::<Vec<ForegroundDiskWindowPlan>>(),
                size_of::<Vec<eredu_runtime::residency::ResidencyClosureSlot>>(),
                size_of::<std::collections::TryReserveError>(),
                size_of::<Box<std::collections::TryReserveError>>(),
                size_of::<eredu_nn::workspace::HostMetadataFunding>(),
            ];
            funding
                .reserve_metadata(
                    bytes
                        .into_iter()
                        .try_fold(size_of_val(&bytes), usize::checked_add)
                        .ok_or_else(overflow)?,
                )
                .map_err(Error::WorkspacePlanning)?;
        }
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(population.controller_units)
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        scratch.resize(
            population.controller_units,
            eredu_runtime::residency::ResidencyClosureSlot::default(),
        );
        let mut out = Vec::new();
        out.try_reserve_exact(windows.len())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        for window in windows {
            out.push(ForegroundDiskWindowPlan::new_with_metadata(
                manager,
                pool,
                *window,
                ids,
                &mut scratch,
                funding,
            )?);
        }
        let temporary_bytes =
            Layout::array::<eredu_runtime::residency::ResidencyClosureSlot>(scratch.capacity())
                .map_err(|_| overflow())?
                .size();
        let value = Self {
            windows: out,
            pool: pool.clone(),
            forwards: population.forwards,
            temporary_bytes,
            nested_source_peak: None,
            background: None,
        };
        value.control_bytes().ok_or_else(overflow)?;
        value.population().ok_or_else(overflow)?;
        Ok(value)
    }
    pub(crate) fn with_background(mut self, selection: Option<&BackgroundSelection>, manager: &ResidencyManager, ids: &[OffloadUnitId], device_sources: &[crate::backend::runtime::residency::manager::WindowPopulation], funding: Option<&eredu_nn::workspace::HostMetadataFunding>) -> Result<Self, Error> {
        if let Some(selection) = selection {
            self.background = Some(BackgroundRequestPlan::new(selection, manager, &self.pool, ids, &self.windows, device_sources, self.forwards, funding.ok_or_else(unknown)?)?);
        }
        Ok(self)
    }
    pub(crate) fn background(&self) -> Option<&BackgroundRequestPlan> { self.background.as_ref() }
    pub(in crate::backend::runtime::execution::generic::original_operations) fn select_background_execution_ordinals(&mut self, ordinals: &[usize]) -> Result<(), Error> {
        self.background.as_mut().ok_or_else(identity)?.select_execution_ordinals(ordinals)
    }
    /// Only live operation destinations; the borrowed cold declaration has
    /// already been paid by its retained source quote.
    pub(crate) fn operation_control_bytes(&self) -> Option<u64> {
        if let Some(background) = &self.background {
            return (0..self.windows.len()).try_fold(u64::try_from(ForegroundDiskSourceCapacity::control_bytes()?).ok()?, |sum, ordinal| {
                sum.checked_add(background.direct(ordinal)?.attempt_control_bytes()?.checked_mul(u64::try_from(self.forwards).ok()?)?)
            });
        }
        self.windows.iter().try_fold(
            u64::try_from(ForegroundDiskSourceCapacity::control_bytes()?).ok()?,
            |sum, window| {
                sum.checked_add(
                    window
                        .attempt_control_bytes()?
                        .checked_mul(u64::try_from(self.forwards).ok()?)?,
                )
            },
        )
    }
    /// Source backing is admitted once at the bounded live capacity. Constructor
    /// controls remain charged for every finite slot/attempt. Native output
    /// births use the separate logical-output population and allocator padding.
    pub(crate) fn control_bytes(&self) -> Option<u64> {
        let fixed = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<WorkingMemoryPool>(),
            size_of::<Vec<ForegroundDiskWindowPlan>>(),
            size_of::<Vec<eredu_runtime::residency::ResidencyClosureSlot>>(),
            size_of::<ForegroundDiskPopulation>(),
            size_of::<Result<(), Error>>(),
        ];
        let bytes = fixed.into_iter().try_fold(
            self.temporary_bytes
                .checked_add(
                    Layout::array::<ForegroundDiskWindowPlan>(self.windows.capacity())
                        .ok()?
                        .size(),
                )?
                .checked_add(size_of_val(&fixed))?,
            usize::checked_add,
        )?;
        let bytes = bytes.checked_add(ForegroundDiskSourceCapacity::control_bytes()?)?;
        self.windows.iter().enumerate().try_fold(
            u64::try_from(bytes).ok()?,
            |sum, (ordinal, window)| {
                let reads = match &self.background {
                    Some(background) => background.direct(ordinal)?,
                    None => window,
                };
                sum.checked_add(window.retained_control_bytes()?)?.checked_add(
                    reads.attempt_control_bytes()?.checked_mul(u64::try_from(self.forwards).ok()?)?,
                )
            },
        )
    }
    /// The shared original acquisition drains the preceding unit/transfer
    /// before starting another window. Native receipt completion destroys its
    /// buffer-retention handlers before publishing terminal status. Thus only
    /// one canonical window can retain disk source backing on successful
    /// progression; include its initial plus one whole-unit retry population.
    /// Failure stops progression and retains capacity until native destruction.
    /// Per-call metadata remains separate from this physical backing amount.
    pub(crate) fn source_backing_capacity(&self) -> Option<usize> {
        if self.forwards == 0 {
            return Some(0);
        }
        if let Some(background) = &self.background { return background.backing_capacity(); }
        if let Some(peak) = self.nested_source_peak {
            return Some(peak);
        }
        self.windows.iter().try_fold(0usize, |maximum, window| {
            Some(maximum.max(window.population()?.source_backing_bytes))
        })
    }
    pub(crate) fn source_facts(
        &self,
    ) -> Option<eredu_runtime::working_memory::HostSourceConstructionFacts> {
        if let Some(background) = &self.background {
            let population = background.population()?.checked_mul(self.forwards)?;
            let bytes = background.source_control_bytes()?.checked_mul(u64::try_from(self.forwards).ok()?)?;
            let selection = background.peak_selection(u64::try_from(self.source_backing_capacity()?).ok()?);
            return eredu_runtime::working_memory::HostSourceConstructionFacts::new(bytes, population.outputs, population.attempts).ok()?.with_peak_backing(selection).ok();
        }
        let population = self.population()?;
        let bytes = self
            .windows
            .iter()
            .try_fold(0u64, |sum, window| {
                sum.checked_add(window.source_control_bytes()?)
            })?
            .checked_mul(u64::try_from(self.forwards).ok()?)?;
        let selection = self
            .windows
            .first()?
            .peak_selection(u64::try_from(self.source_backing_capacity()?).ok()?);
        eredu_runtime::working_memory::HostSourceConstructionFacts::new(
            bytes,
            population.outputs,
            population.attempts,
        )
        .ok()?
        .with_peak_backing(selection)
        .ok()
    }
    pub(crate) fn host_facts(&self) -> Option<eredu_runtime::working_memory::HostDestinationFacts> {
        eredu_runtime::working_memory::HostDestinationFacts::new(0, 0)
            .ok()?
            .with_source_constructions(self.source_facts()?)
            .ok()
    }
    pub(crate) fn prepare_capacity(
        &self,
        custody: OriginalTextControlGuard,
        reservation: &eredu_runtime::working_memory::WorkingMemoryReservation,
        mut bank: eredu_runtime::working_memory::OriginalHostDestinationBank,
    ) -> Result<ForegroundDiskSourceCapacity, Error> {
        if !custody
            .metadata_custody()
            .matches_domain(self.pool.shared_storage_domain())
        {
            return Err(identity());
        }
        if !bank.belongs_to(&custody) || !bank.matches_facts(self.host_facts().ok_or_else(unknown)?)
        {
            return Err(identity());
        }
        self.prepare_source_capacity(
            custody.into(),
            Some(reservation),
            bank.take_source_constructions().ok_or_else(identity)?,
        )
    }
    pub(crate) fn prepare_source_capacity(
        &self,
        custody: eredu_runtime::working_memory::OriginalHostSourceCustody,
        reservation: Option<&eredu_runtime::working_memory::WorkingMemoryReservation>,
        bank: eredu_runtime::working_memory::OriginalHostSourceBank,
    ) -> Result<ForegroundDiskSourceCapacity, Error> {
        if !custody.metadata_custody().matches_domain(self.pool.shared_storage_domain())
            || !bank.belongs_to_source(&custody)
            || !bank.matches_facts(self.source_facts().ok_or_else(unknown)?) {
            return Err(identity());
        }
        let bytes = u64::try_from(self.source_backing_capacity().ok_or_else(overflow)?).map_err(|_| overflow())?;
        let selection = if let Some(background) = &self.background { background.peak_selection(bytes) } else { self.windows.first().ok_or_else(identity)?.peak_selection(
            u64::try_from(self.source_backing_capacity().ok_or_else(overflow)?)
                .map_err(|_| overflow())?,
        ) };
        ForegroundDiskSourceCapacity::with_source_account(selection, bank, custody, reservation)
            .map_err(Error::PrefillControl)
    }
    /// Descriptive actual constructor/copy populations, never complete native fit.
    pub(crate) fn population(&self) -> Option<ForegroundDiskPopulation> {
        self.forward_population()?.checked_mul(self.forwards)
    }
    pub(crate) fn forward_population(&self) -> Option<ForegroundDiskPopulation> {
        self.windows
            .iter()
            .try_fold(ForegroundDiskPopulation::default(), |sum, window| {
                sum.checked_add(window.population()?)
            })
    }
    pub(crate) fn windows(&self) -> &[ForegroundDiskWindowPlan] { &self.windows }
    pub(crate) fn window(&self, ordinal: usize) -> Option<&ForegroundDiskWindowPlan> {
        self.windows.get(ordinal)
    }
    pub(crate) fn pool(&self) -> &WorkingMemoryPool {
        &self.pool
    }
    pub(crate) fn matches(&self, windows: usize, forwards: usize) -> bool {
        self.windows.len() == windows && self.forwards == forwards
    }
}

/// The actual one-forward disk declaration lives with the immutable source
/// snapshot. Its vectors/source aliases retire before the quote's funding.
pub(crate) struct PreparedSpeculativeForegroundSource {
    plan: ForegroundDiskRequestPlan,
    _funding: eredu_nn::workspace::HostMetadataFunding,
}
impl PreparedSpeculativeForegroundSource {
    pub(crate) fn new(
        manager: &ResidencyManager, pool: &WorkingMemoryPool,
        windows: &[crate::backend::runtime::residency::manager::WindowPopulation],
        ids: &[OffloadUnitId], population: ResidencyPopulation,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<Self, Error> {
        funding
            .reserve_metadata(
                size_of::<Self>()
                    .checked_add(size_of::<Result<Self, Error>>())
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let funding = funding.clone();
        let plan = ForegroundDiskRequestPlan::new_with_metadata(
            manager,
            pool,
            windows,
            ids,
            population,
            Some(&funding),
        )?;
        Ok(Self {
            plan,
            _funding: funding,
        })
    }
    pub(crate) fn with_background(mut self, selection: Option<&BackgroundSelection>, manager: &ResidencyManager, ids: &[OffloadUnitId], sources: &[crate::backend::runtime::residency::manager::WindowPopulation]) -> Result<Self, Error> {
        self.plan = self.plan.with_background(selection, manager, ids, sources, Some(&self._funding))?;
        Ok(self)
    }
    pub(crate) fn plan(&self) -> &ForegroundDiskRequestPlan {
        &self.plan
    }
    /// Repeated/nested module entry order is separate from unique source rows.
    /// Each exit ordinal is consumed once; an outer window remains charged
    /// while any nested window acquires and executes under its live lease.
    pub(crate) fn set_nested_completion_order<I>(&mut self, order: I) -> Result<(), Error>
    where
        I: IntoIterator<Item = usize>,
    {
        if self.plan.forwards != 1 || self.plan.nested_source_peak.is_some() {
            return Err(identity());
        }
        let frames = [
            size_of::<I>(),
            size_of::<I::IntoIter>(),
            size_of::<(usize, usize)>(),
            size_of::<Vec<(usize, usize)>>(),
            size_of::<Result<(), Error>>(),
            size_of::<[usize; 4]>(),
        ];
        self._funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let mut stack = self
            ._funding
            .metadata_vec::<(usize, usize)>(self.plan.windows.len())
            .map_err(Error::from)?;
        let mut next = 0usize;
        let mut live = 0usize;
        let mut peak = 0usize;
        let mut entries = 0usize;
        for completion in order {
            let window = self.plan.windows.get(entries).ok_or_else(identity)?;
            if completion >= self.plan.windows.len() {
                return Err(identity());
            }
            while stack.last().is_some_and(|(prior, _)| *prior < completion) {
                let (prior, bytes) = stack.pop().expect("checked stack");
                if prior != next {
                    return Err(identity());
                }
                next = next.checked_add(1).ok_or_else(overflow)?;
                live = live.checked_sub(bytes).ok_or_else(identity)?;
            }
            if stack.last().is_some_and(|(prior, _)| *prior == completion) || completion < next {
                return Err(identity());
            }
            let bytes = window
                .population()
                .ok_or_else(overflow)?
                .source_backing_bytes;
            live = live.checked_add(bytes).ok_or_else(overflow)?;
            peak = peak.max(live);
            stack.push((completion, bytes));
            entries = entries.checked_add(1).ok_or_else(overflow)?;
        }
        if entries != self.plan.windows.len() {
            return Err(identity());
        }
        while let Some((completion, _)) = stack.pop() {
            if completion != next {
                return Err(identity());
            }
            next = next.checked_add(1).ok_or_else(overflow)?;
        }
        if next != entries {
            return Err(identity());
        }
        self.plan.nested_source_peak = Some(peak);
        Ok(())
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
