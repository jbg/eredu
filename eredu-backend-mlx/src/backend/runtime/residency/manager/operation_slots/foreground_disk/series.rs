//! One retained source root for sequential regions of the same actual reader.
use super::*;
use eredu_runtime::working_memory::{
    HostSourceConstructionFacts, OriginalHostSourceBank, OriginalHostSourcePeakCapacity,
};

/// Counts come only from exact source-derived region ceilings. Peak backing is
/// the maximum sequential region, while all constructor/read calls accumulate.
#[derive(Clone)]
pub(crate) struct ForegroundDiskSourceSeries {
    source: ForegroundDiskDescriptors,
    controls: u64,
    outputs: usize,
    attempts: usize,
    backing: usize,
}
impl ForegroundDiskSourceSeries {
    pub(crate) fn new(source: &ForegroundDiskSubsetCeiling, calls: usize) -> Option<Self> {
        let population = source.population.checked_mul(calls)?;
        Some(Self {
            source: source.source.clone(),
            controls: source
                .source_controls
                .checked_mul(u64::try_from(calls).ok()?)?,
            outputs: population.sources,
            attempts: population.attempts,
            backing: if calls == 0 {
                0
            } else {
                source.population.source_backing_bytes
            },
        })
    }
    pub(crate) fn matches_ceiling(&self, source: &ForegroundDiskSubsetCeiling) -> bool {
        self.source.same_source(&source.source)
    }
    pub(crate) fn include(
        &mut self,
        source: &ForegroundDiskSubsetCeiling,
        calls: usize,
    ) -> Option<()> {
        if !self.source.same_source(&source.source) {
            return None;
        }
        let next = Self::new(source, calls)?;
        let controls = self.controls.checked_add(next.controls)?;
        let outputs = self.outputs.checked_add(next.outputs)?;
        let attempts = self.attempts.checked_add(next.attempts)?;
        self.controls = controls;
        self.outputs = outputs;
        self.attempts = attempts;
        self.backing = self.backing.max(next.backing);
        Some(())
    }
    pub(crate) fn facts(&self) -> Option<HostSourceConstructionFacts> {
        HostSourceConstructionFacts::new(self.controls, self.outputs, self.attempts)
            .ok()?
            .with_peak_backing(
                self.source
                    .peak_selection(u64::try_from(self.backing).ok()?),
            )
            .ok()
    }
    pub(crate) fn prepare_capacity(
        &self,
        bank: OriginalHostSourceBank,
        custody: OriginalHostSourceCustody,
        reservation: Option<&WorkingMemoryReservation>,
    ) -> Result<ForegroundDiskSourceCapacity, WorkingMemoryError> {
        if !bank.belongs_to_source(&custody)
            || !bank.matches_facts(self.facts().ok_or(WorkingMemoryError::Overflow)?)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        ForegroundDiskSourceCapacity::with_source_account(
            self.source.peak_selection(
                u64::try_from(self.backing).map_err(|_| WorkingMemoryError::Overflow)?,
            ),
            bank,
            custody,
            reservation,
        )
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<HostSourceConstructionFacts>(),
            size_of::<OriginalHostSourceBank>(),
            size_of::<OriginalHostSourceCustody>(),
            size_of::<Result<ForegroundDiskSourceCapacity, WorkingMemoryError>>(),
            size_of::<(&Self, &ForegroundDiskSubsetCeiling, usize)>(),
            size_of::<[usize; 4]>(),
            size_of::<[u64; 2]>(),
            ForegroundDiskSourceCapacity::control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
impl ForegroundDiskSubsetCeiling {
    /// The exact same native descriptor producer and accepted capacity, never
    /// caller-provided byte authority. Runtime read counters remain shared.
    pub(crate) fn matches_capacity(&self, capacity: &ForegroundDiskSourceCapacity) -> bool {
        capacity.matches_source(&self.source)
            && u64::try_from(self.population.source_backing_bytes)
                .is_ok_and(|bytes| capacity.backing_bytes() >= bytes)
    }
}
