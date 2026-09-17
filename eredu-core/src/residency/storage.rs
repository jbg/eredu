//! Finite constructor storage and shared named-window transitions.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

/// Exact requested ledger arrays, identifier copies and shared plan header.
/// The already-owned input plan payload and allocator rounding are separate.
/// The shared header uses ResidencyPlanSource::shared_plan_layout; callers must
/// bind its existing Rust Arc representation qualification before claiming fit.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyLedgerStorageLayout {
    bytes: usize,
}
impl ResidencyLedgerStorageLayout {
    /// Requested constructor and retained storage, including fixed transports.
    pub const fn required_bytes(self) -> usize {
        self.bytes
    }
    /// Inspect actual immutable plan and declared original window names.
    pub fn inspect<G: AsRef<str>>(plan: &OffloadPlan, groups: &[G]) -> Option<Self> {
        let mut bytes = Layout::array::<LedgerUnit>(plan.units().len()).ok()?.size();
        for spec in plan.units() {
            bytes = bytes.checked_add(spec.id().as_str().len())?;
        }
        bytes = bytes.checked_add(Layout::array::<WindowRow>(groups.len()).ok()?.size())?;
        let flags = Layout::array::<[bool; 2]>(plan.units().len()).ok()?.size();
        for (index, group) in groups.iter().enumerate() {
            if group.as_ref().trim().is_empty()
                || groups[..index]
                    .iter()
                    .any(|prior| prior.as_ref() == group.as_ref())
            {
                return None;
            }
            bytes = bytes
                .checked_add(group.as_ref().len())?
                .checked_add(flags)?;
        }
        bytes = bytes.checked_add(ResidencyPlanSource::shared_plan_layout()?.size())?;
        for control in [
            size_of::<ResidencyLedger>(),
            size_of::<Self>(),
            size_of::<Result<ResidencyLedger, ResidencyStorageError>>(),
            size_of::<LedgerUnits>(),
            size_of::<LedgerWindows>(),
            size_of::<WindowRow>(),
            size_of::<OffloadUnitSpec>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<ResidencyStorageError>(),
        ] {
            bytes = bytes.checked_add(control)?;
        }
        Some(Self { bytes })
    }
}
/// Concrete constructor refusal before publication; partial arrays retire in the caller's custody.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyStorageError {
    /// Requested array arithmetic or declared group population is invalid.
    #[error("invalid residency constructor storage layout")]
    Layout,
    /// One actual finite destination reserve failed.
    #[error("residency constructor storage reserve failed")]
    Reserve(#[source] TryReserveError),
}

#[derive(Debug)]
pub(super) struct LedgerUnits(Vec<LedgerUnit>);
impl LedgerUnits {
    pub(super) fn construct(plan: &OffloadPlan) -> Result<Self, ResidencyStorageError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(plan.units().len())
            .map_err(ResidencyStorageError::Reserve)?;
        for spec in plan.units() {
            let mut id = String::new();
            id.try_reserve_exact(spec.id().as_str().len())
                .map_err(ResidencyStorageError::Reserve)?;
            id.push_str(spec.id().as_str());
            rows.push(LedgerUnit {
                spec: OffloadUnitSpec {
                    id: OffloadUnitId(id),
                    bytes: spec.bytes,
                    policy: spec.policy,
                    tier: spec.tier,
                },
                host: None,
                device: None,
            });
        }
        Ok(Self(rows))
    }
    pub(super) fn ordinal(&self, id: &OffloadUnitId) -> Option<usize> {
        self.0.binary_search_by(|row| row.spec.id().cmp(id)).ok()
    }
    pub(super) fn contains_key(&self, id: &OffloadUnitId) -> bool {
        self.ordinal(id).is_some()
    }
    pub(super) fn get(&self, id: &OffloadUnitId) -> Option<&LedgerUnit> {
        self.ordinal(id).map(|index| &self.0[index])
    }
    pub(super) fn get_mut(&mut self, id: &OffloadUnitId) -> Option<&mut LedgerUnit> {
        self.ordinal(id).map(|index| &mut self.0[index])
    }
    pub(super) fn values(&self) -> std::slice::Iter<'_, LedgerUnit> {
        self.0.iter()
    }
    pub(super) fn values_mut(&mut self) -> std::slice::IterMut<'_, LedgerUnit> {
        self.0.iter_mut()
    }
}
impl std::ops::Index<&OffloadUnitId> for LedgerUnits {
    type Output = LedgerUnit;
    fn index(&self, id: &OffloadUnitId) -> &LedgerUnit {
        self.get(id).expect("known ledger unit")
    }
}
#[derive(Debug)]
struct WindowRow {
    name: String,
    units: Vec<[bool; 2]>,
}
#[derive(Debug)]
pub(super) struct LedgerWindows {
    rows: Vec<WindowRow>,
    finite: bool,
}
impl LedgerWindows {
    pub(super) fn ordinary() -> Self {
        Self {
            rows: Vec::new(),
            finite: false,
        }
    }
    pub(super) fn construct<G: AsRef<str>>(
        count: usize,
        groups: &[G],
    ) -> Result<Self, ResidencyStorageError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(groups.len())
            .map_err(ResidencyStorageError::Reserve)?;
        for group in groups {
            let mut name = String::new();
            name.try_reserve_exact(group.as_ref().len())
                .map_err(ResidencyStorageError::Reserve)?;
            name.push_str(group.as_ref());
            let mut units = Vec::new();
            units
                .try_reserve_exact(count)
                .map_err(ResidencyStorageError::Reserve)?;
            units.resize(count, [false; 2]);
            rows.push(WindowRow { name, units });
        }
        Ok(Self { rows, finite: true })
    }
    pub(super) fn prepared_name(&self, group: &str) -> Option<&str> {
        self.finite.then(|| self.rows.iter().find(|row| row.name == group))
            .flatten().map(|row| row.name.as_str())
    }
    pub(super) fn contains(&self, ordinal: usize, tier: MemoryTier) -> bool {
        let Some(tier) = window_tier(tier) else {
            return false;
        };
        self.rows.iter().any(|row| row.units[ordinal][tier])
    }
    pub(super) fn set(
        &mut self,
        group: &str,
        active: &[OffloadUnitId],
        tier: MemoryTier,
        units: &LedgerUnits,
    ) -> Result<(), ResidencyLedgerError> {
        let tier = window_tier(tier).expect("validated resident tier");
        let index = match self.rows.iter().position(|row| row.name == group) {
            Some(index) => index,
            None if self.finite => return Err(ResidencyLedgerError::UnknownPreparedGroup),
            None if active.is_empty() => return Ok(()),
            None => {
                self.rows.push(WindowRow {
                    name: group.to_owned(),
                    units: vec![[false; 2]; units.0.len()],
                });
                self.rows.len() - 1
            }
        };
        let row = &mut self.rows[index];
        for flags in &mut row.units {
            flags[tier] = false;
        }
        for id in active {
            row.units[units.ordinal(id).expect("validated group unit")][tier] = true;
        }
        if !self.finite && row.units.iter().all(|flags| !flags[0] && !flags[1]) {
            self.rows.remove(index);
        }
        Ok(())
    }
}
fn window_tier(tier: MemoryTier) -> Option<usize> {
    match tier {
        MemoryTier::Host => Some(0),
        MemoryTier::Device => Some(1),
        MemoryTier::Disk => None,
    }
}
