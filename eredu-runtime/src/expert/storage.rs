//! Installed sparse parameter pools and their shared residency accounting.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use eredu_checkpoint::{quantization::WeightMaterializationReport, WeightQuantization};
use eredu_core::{
    parameters::ParameterError,
    residency::{MemoryTier, OffloadUnitId},
    BackendFailure,
};

use crate::{
    dense::DenseResidencyMechanisms,
    parameter_operations::{
        prepare_bank_parameter_replacements, BankParameterReplacementFacts,
        PreparedBankParameterIndex, PreparedBankParameterMember,
    },
    residency::ResidencyDriver,
    ParameterBankAccess, ParameterBankKey, ParameterBankLoadOptions,
};

use super::{
    AddressableBankStorage, BankAcquisitionError, BankAcquisitionResources,
    ParameterBankResidencyReport, ParameterBankStatistics, RowLookupPoolReport,
};

/// Shared entry catalog, scheduler, residency manager, and telemetry.
pub struct AddressableParameterBank<N: DenseResidencyMechanisms, T> {
    parameter_members: PreparedBankParameterIndex,
    parameter_replacements: BTreeMap<String, T>,
    effective_member_bytes: BTreeMap<ParameterBankKey, u64>,
    pool_id: u64,
    manager: ResidencyDriver<N>,
    catalog: BTreeMap<ParameterBankKey, u64>,
    unit_banks: BTreeMap<OffloadUnitId, usize>,
    host_budget: Option<u64>,
    scratch_limit: u64,
    bulk_bank_target: u64,
    statistics: Mutex<BTreeMap<usize, ParameterBankStatistics>>,
    weight_quantizations: Vec<WeightQuantization>,
    placements: BTreeMap<ParameterBankKey, crate::AddressableBankMemberPlacement>,
    materialization: Option<WeightMaterializationReport>,
}

impl<N: DenseResidencyMechanisms, T> AddressableParameterBank<N, T> {
    /// Owns one installed residency driver and its exact admitted parameter catalog.
    pub fn from_installed(installed: AddressableBankStorage<ResidencyDriver<N>>) -> Self {
        let (residency, catalog, members, placements, options, formats, materialization) =
            installed.into_parts();
        Self::from_residency(
            residency,
            catalog,
            members,
            options,
            MemoryTier::Disk,
            formats,
            placements,
            materialization,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Assembles the state of one admitted pool.
    pub fn from_residency(
        manager: ResidencyDriver<N>,
        catalog: BTreeMap<ParameterBankKey, u64>,
        parameter_members: PreparedBankParameterIndex,
        options: ParameterBankLoadOptions,
        initial_tier: MemoryTier,
        weight_quantizations: Vec<WeightQuantization>,
        placements: BTreeMap<ParameterBankKey, crate::AddressableBankMemberPlacement>,
        materialization: Option<WeightMaterializationReport>,
    ) -> Self {
        static NEXT_POOL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            parameter_members,
            parameter_replacements: BTreeMap::new(),
            effective_member_bytes: catalog.clone(),
            pool_id: NEXT_POOL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            manager,
            unit_banks: catalog
                .keys()
                .map(|key| (key.unit_id(), key.bank()))
                .collect(),
            catalog,
            host_budget: if initial_tier == MemoryTier::Device {
                Some(0)
            } else {
                options.offload().host_budget_bytes()
            },
            scratch_limit: options.compact_bank_scratch_bytes(),
            bulk_bank_target: options.prefill_compact_bank_target_bytes(),
            statistics: Mutex::new(BTreeMap::new()),
            weight_quantizations,
            placements,
            materialization,
        }
    }

    /// Returns all load-time encodings of transformed entry bindings.
    pub fn weight_quantizations(&self) -> &[WeightQuantization] {
        &self.weight_quantizations
    }

    /// Returns the underlying reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyDriver<N> {
        &self.manager
    }

    /// Returns the largest source member in the generic storage catalog.
    pub fn maximum_member_bytes(&self) -> u64 {
        self.catalog.values().copied().max().unwrap_or(0)
    }

    /// Returns the hard compact-bank byte limit.
    pub const fn compact_bank_scratch_bytes(&self) -> u64 {
        self.scratch_limit
    }

    /// Returns the bulk compact-bank working-set target.
    pub const fn bulk_compact_bank_target_bytes(&self) -> u64 {
        self.bulk_bank_target
    }

    /// Returns the admitted global member span for a caller-owned namespace.
    pub fn namespace_global_span(&self, namespace: usize) -> Option<usize> {
        self.catalog
            .keys()
            .filter(|key| key.unit() == namespace)
            .map(|key| key.member() + 1)
            .max()
    }

    /// Forms the exact borrows used by one scoped acquisition.
    pub fn acquisition_resources(
        &self,
        scope: Option<usize>,
    ) -> BankAcquisitionResources<'_, ResidencyDriver<N>>
    where
        N: Clone + Send + Sync + 'static,
    {
        BankAcquisitionResources {
            residency: &self.manager,
            catalog: &self.catalog,
            effective_member_bytes: &self.effective_member_bytes,
            unit_banks: &self.unit_banks,
            host_budget: self.host_budget,
            scratch_limit: self.scratch_limit,
            statistics: &self.statistics,
            scope,
        }
    }

    /// Returns parameter declarations in scope for an installed executable.
    fn bank_members(
        &self,
        scope: Option<usize>,
    ) -> impl Iterator<Item = &PreparedBankParameterMember> {
        self.parameter_members
            .members()
            .iter()
            .filter(move |member| scope.is_none_or(|bank| bank == member.key.bank()))
    }

    /// Returns whether the pool owns storage for this logical bank.
    fn contains_bank(&self, bank: usize) -> bool {
        self.catalog.keys().any(|key| key.bank() == bank)
    }

    /// Observes the shared pool as a row provider without duplicating its counters.
    pub fn row_report(
        &self,
        requirements: crate::SelectedRowLookupRequirements,
    ) -> Result<RowLookupPoolReport, N::Error> {
        Ok(RowLookupPoolReport {
            pool_id: self.pool_id,
            residency: self.manager.report()?,
            requirements,
        })
    }

    /// Stable identity shared by grouped members and row ranges in this pool.
    pub const fn pool_id(&self) -> u64 {
        self.pool_id
    }

    /// Exact source bytes for this catalog.
    pub fn catalog(&self) -> &BTreeMap<ParameterBankKey, u64> {
        &self.catalog
    }

    /// Current member bytes after accounting for published replacements.
    pub fn effective_member_bytes(&self) -> &BTreeMap<ParameterBankKey, u64> {
        &self.effective_member_bytes
    }

    /// Current canonical replacement, if any, for this parameter.
    pub fn replacement(&self, parameter: &str) -> Option<&T> {
        self.parameter_replacements.get(parameter)
    }

    /// Resolves a physical member and binding to its current replacement row.
    pub fn replacement_row(&self, key: ParameterBankKey, binding: &str) -> Option<(usize, &T)> {
        let address = self.parameter_members.binding(key, binding)?;
        Some((
            address.row(),
            self.replacement(&address.member().parameter)?,
        ))
    }

    fn publish_replacements(
        &mut self,
        values: BTreeMap<String, T>,
        member_bytes: BTreeMap<ParameterBankKey, u64>,
    ) {
        self.parameter_replacements = values;
        self.effective_member_bytes = member_bytes;
    }

    /// Records a completed compact-bank construction.
    pub fn record_compact_bank(
        &self,
        bank: usize,
        pass: ParameterBankAccess,
        bytes: u64,
        duration: Duration,
    ) -> Result<(), BankAcquisitionError<N::Error>> {
        if bytes > self.scratch_limit {
            return Err(crate::AddressableBankDemandError::ScratchLimitExceeded {
                required_bytes: bytes,
                limit_bytes: self.scratch_limit,
                distinct_entries: 0,
            }
            .into());
        }
        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| BankAcquisitionError::StatisticsPoisoned)?;
        statistics
            .entry(bank)
            .or_default()
            .record_compaction(pass, bytes, duration);
        Ok(())
    }

    /// Returns current entry residency, transfer, storage, and pass statistics.
    pub fn report(&self) -> Result<ParameterBankResidencyReport, BankAcquisitionError<N::Error>> {
        self.report_scoped(None)
    }

    /// Reports one logical bank in the shared pool.
    fn report_scoped(
        &self,
        scope: Option<usize>,
    ) -> Result<ParameterBankResidencyReport, BankAcquisitionError<N::Error>> {
        let residency = self
            .manager
            .report()
            .map_err(BankAcquisitionError::Residency)?;
        let catalog = self
            .catalog
            .iter()
            .filter(|(key, _)| scope.is_none_or(|bank| key.bank() == bank))
            .collect::<BTreeMap<_, _>>();
        let ids = catalog
            .keys()
            .map(|key| key.unit_id())
            .collect::<std::collections::BTreeSet<_>>();
        let mut host_resident_entries = 0;
        let mut device_resident_entries = 0;
        let mut host_resident_bytes = 0u64;
        let mut device_resident_bytes = 0u64;
        for unit in residency
            .units()
            .iter()
            .filter(|unit| ids.contains(unit.id()))
        {
            if unit.host_resident() {
                host_resident_entries += 1;
                host_resident_bytes =
                    host_resident_bytes.saturating_add(unit.host_allocated_bytes());
            }
            if unit.device_resident() {
                device_resident_entries += 1;
                device_resident_bytes =
                    device_resident_bytes.saturating_add(unit.device_allocated_bytes());
            }
        }
        let statistics = self
            .statistics
            .lock()
            .map_err(|_| BankAcquisitionError::StatisticsPoisoned)?;
        Ok(ParameterBankResidencyReport {
            pool_id: self.pool_id,
            weight_quantizations: self.weight_quantizations.clone(),
            placements: self
                .placements
                .iter()
                .filter(|(key, _)| scope.is_none_or(|bank| key.bank() == bank))
                .map(|(key, placement)| (*key, placement.clone()))
                .collect(),
            owned_entries: catalog.len(),
            owned_bytes: catalog.values().map(|bytes| **bytes).sum(),
            host_resident_entries,
            device_resident_entries,
            host_resident_bytes,
            device_resident_bytes,
            peak_host_resident_bytes: scope.map_or_else(
                || {
                    residency
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Host)
                },
                |bank| {
                    statistics
                        .get(&bank)
                        .map_or(0, |stats| stats.peak_host_bytes)
                },
            ),
            peak_device_resident_bytes: scope.map_or_else(
                || {
                    residency
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                },
                |bank| {
                    statistics
                        .get(&bank)
                        .map_or(0, |stats| stats.peak_device_bytes)
                },
            ),
            bulk: statistics
                .iter()
                .filter(|(bank, _)| scope.is_none_or(|scope| scope == **bank))
                .map(|(_, stats)| stats.bulk)
                .sum(),
            incremental: statistics
                .iter()
                .filter(|(bank, _)| scope.is_none_or(|scope| scope == **bank))
                .map(|(_, stats)| stats.incremental)
                .sum(),
            residency,
            materialization: self.materialization.clone(),
        })
    }
}

/// One logical view of a shared portable sparse residency pool.
pub struct SharedAddressableParameterBank<N: DenseResidencyMechanisms, T> {
    pool: Arc<Mutex<AddressableParameterBank<N, T>>>,
    scope: Option<usize>,
}

impl<N: DenseResidencyMechanisms, T> From<AddressableBankStorage<ResidencyDriver<N>>>
    for SharedAddressableParameterBank<N, T>
{
    fn from(storage: AddressableBankStorage<ResidencyDriver<N>>) -> Self {
        Self::new(AddressableParameterBank::from_installed(storage))
    }
}

impl<N: DenseResidencyMechanisms, T> Clone for SharedAddressableParameterBank<N, T> {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
            scope: self.scope,
        }
    }
}

impl<N: DenseResidencyMechanisms, T> SharedAddressableParameterBank<N, T> {
    /// Shares an admitted pool with its providers and reporting consumers.
    pub fn new(bank: AddressableParameterBank<N, T>) -> Self {
        Self {
            pool: Arc::new(Mutex::new(bank)),
            scope: None,
        }
    }

    /// Restricts a view to an exact logical bank without allocating storage.
    pub fn scoped(&self, bank: usize) -> Result<Self, ParameterError> {
        let pool = self.lock_pool()?;
        if !self.owns_parameter_bank(bank) || !pool.contains_bank(bank) {
            return Err(ParameterError::Invalid(
                "bank scope is absent from the selected pool".into(),
            ));
        }
        Ok(Self {
            pool: Arc::clone(&self.pool),
            scope: Some(bank),
        })
    }

    /// Whether this view permits operations on the named bank.
    pub const fn owns_parameter_bank(&self, bank: usize) -> bool {
        match self.scope {
            None => true,
            Some(scope) => scope == bank,
        }
    }

    /// Restriction on the logical bank ordinals visible to this view.
    pub const fn scope(&self) -> Option<usize> {
        self.scope
    }

    /// Borrows the shared pool for setup or an admitted operation.
    pub fn lock_pool(
        &self,
    ) -> Result<MutexGuard<'_, AddressableParameterBank<N, T>>, ParameterError> {
        self.pool.lock().map_err(|_| {
            ParameterError::Backend(BackendFailure::from_error(std::io::Error::other(
                "addressable parameter bank lock was poisoned",
            )))
        })
    }

    /// Observes this logical bank's counters and the current shared residency.
    pub fn report(&self) -> Result<ParameterBankResidencyReport, ParameterError> {
        self.lock_pool()?
            .report_scoped(self.scope)
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    }

    /// Observes the same physical pool for attributed row diagnostics.
    pub fn row_report(
        &self,
        requirements: crate::SelectedRowLookupRequirements,
    ) -> Result<RowLookupPoolReport, ParameterError> {
        self.lock_pool()?
            .row_report(requirements)
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    }
}

impl<N: DenseResidencyMechanisms, T> crate::executable_binding::TextParameterBanks
    for SharedAddressableParameterBank<N, T>
{
    fn bank_members(&self) -> Result<Vec<PreparedBankParameterMember>, ParameterError> {
        Ok(self
            .lock_pool()?
            .bank_members(self.scope)
            .cloned()
            .collect())
    }
}

/// Publishes ordinary and sparse handles after each distinct pool validates the replacement.
pub fn publish_bank_parameter_replacements<N, T, E>(
    banks: &[SharedAddressableParameterBank<N, T>],
    values: &BTreeMap<String, T>,
    active: bool,
    facts: impl Fn(&T) -> BankParameterReplacementFacts,
    ordinary: impl FnOnce() -> Result<bool, E>,
) -> Result<bool, E>
where
    N: DenseResidencyMechanisms,
    T: Clone,
    E: From<ParameterError>,
{
    let mut pools = banks.iter().collect::<Vec<_>>();
    pools.sort_unstable_by_key(|bank| Arc::as_ptr(&bank.pool) as usize);
    pools.dedup_by_key(|bank| Arc::as_ptr(&bank.pool));
    let mut guards = pools
        .into_iter()
        .map(SharedAddressableParameterBank::lock_pool)
        .collect::<Result<Vec<_>, _>>()?;
    let mut prepared = Vec::with_capacity(guards.len());
    for bank in &guards {
        let (parameters, bytes) = prepare_bank_parameter_replacements(
            bank.parameter_members.members(),
            &bank.catalog,
            active,
            |parameter| values.get(parameter).map(&facts),
        )
        .map_err(|error| ParameterError::Backend(BackendFailure::from_error(error)))?
        .into_parts();
        let replacements = parameters
            .into_iter()
            .map(|parameter| (parameter.clone(), values[&parameter].clone()))
            .collect();
        prepared.push((replacements, bytes));
    }
    if !ordinary()? {
        return Ok(false);
    }
    for (bank, (replacements, bytes)) in guards.iter_mut().zip(prepared) {
        bank.publish_replacements(replacements, bytes);
    }
    Ok(true)
}

/// Restricts an installed portable pool to one named logical bank.
pub trait ScopedParameterBank: Sized {
    /// Adopts a scoped view over the existing residency owner.
    fn scope_to(&self, bank: usize) -> Result<Self, ParameterError>;
}

impl<N: DenseResidencyMechanisms, T> ScopedParameterBank for SharedAddressableParameterBank<N, T> {
    fn scope_to(&self, bank: usize) -> Result<Self, ParameterError> {
        self.scoped(bank)
    }
}
