//! Addressable entry acquisition, residency, and completion.

use super::*;

mod prepared;
pub(crate) use prepared::PreparedAddressableSource;
mod ordinary_host;
pub(crate) use ordinary_host::{compact_transport_control_bytes, OrdinaryBankHostSource};

fn prepared_parameter_members(
    entries: &[ParameterBankEntry],
    targets: &BTreeMap<(ParameterBankKey, String), String>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<
    Vec<eredu_runtime::parameter_operations::PreparedBankParameterMember>,
    AddressableParameterBankError,
> {
    let result = entries
        .iter()
        .flat_map(|entry| {
            entry
                .unit
                .bindings()
                .iter()
                .map(move |binding| (entry, binding))
        })
        .map(|(entry, binding)| {
            let parameter = targets
                .get(&(entry.identity, binding.name().to_owned()))
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "selected bank binding has no logical destination".into(),
                    )
                })?;
            Ok(
                eredu_runtime::parameter_operations::PreparedBankParameterMember {
                    key: entry.identity,
                    binding: binding.name().to_owned(),
                    parameter: parameter.clone(),
                    materialized: binding.source_recipe().infer(store)?,
                },
            )
        })
        .collect::<Result<Vec<_>, Error>>();
    result.map_err(|source| AddressableParameterBankError::Transformation {
        source: Box::new(source),
    })
}

/// Shared entry catalog, scheduler, residency manager, and telemetry.
pub struct AddressableParameterBank {
    pub(super) parameter_members:
        Vec<eredu_runtime::parameter_operations::PreparedBankParameterMember>,
    pub(super) parameter_replacements:
        eredu_runtime::parameter_operations::ParameterReplacementValues<MlxTensor>,
    pub(super) parameter_revision: u64,
    pub(super) effective_member_bytes: BTreeMap<ParameterBankKey, u64>,
    pub(super) pool_id: u64,
    pub(super) manager: ResidencyManager,
    pub(super) catalog: BTreeMap<ParameterBankKey, u64>,
    pub(super) unit_banks: BTreeMap<OffloadUnitId, usize>,
    #[cfg(test)]
    pub(super) namespace_entry_counts: BTreeMap<usize, usize>,
    #[cfg(test)]
    pub(super) namespace_global_spans: BTreeMap<usize, usize>,
    pub(super) host_budget: Option<u64>,
    pub(super) scratch_limit: u64,
    #[cfg(test)]
    pub(super) bulk_bank_target: u64,
    pub(super) statistics: Mutex<ParameterBankStatisticsTable>,
    pub(super) weight_quantizations: Vec<WeightQuantization>,
    pub(super) placements:
        BTreeMap<ParameterBankKey, eredu_runtime::AddressableBankMemberPlacement>,
    pub(super) materialization: Option<WeightMaterializationReport>,
}

/// Cloneable handle to one independently addressable parameter bank.
///
/// Partitioned execution retains this handle outside the monomorphized routed
/// provider so generic session telemetry can observe the same live cache. The
/// handle adds no routing or architecture policy; it only serializes access to
/// the native storage mechanism.
#[derive(Clone)]
pub struct SharedAddressableParameterBank {
    pub(super) inner: Arc<Mutex<AddressableParameterBank>>,
    pub(super) scope: Option<usize>,
}

pub(super) fn preflight_selected_entry_bindings(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    entries: &[ParameterBankEntry],
) -> Result<(), AddressableParameterBankError> {
    for entry in entries {
        eredu_runtime::preflight_bindings::<crate::backend::nn::shared::MlxNeuralBackend>(
            store,
            entry.unit.bindings(),
        )
        .map_err(|error| AddressableParameterBankError::Transformation {
            source: Box::new(Error::Other(Box::new(error))),
        })?;
    }
    Ok(())
}

impl SharedAddressableParameterBank {
    /// Inventories the entire shared physical pool, including replacement values.
    /// A logical bank scope cannot discount other storage retained by the pool.
    pub fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        self.collect_retained_storage(&mut storage)?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    pub fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        let bank = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
        })?;
        bank.manager.collect_retained_storage(storage)?;
        for value in bank.parameter_replacements.values() {
            storage.include_array(value.as_array())?;
        }
        Ok(())
    }

    /// Reads exact member destinations retained after lowering, without source work.
    pub fn prepared_parameter_members(
        &self,
    ) -> Result<Vec<eredu_runtime::parameter_operations::PreparedBankParameterMember>, Error> {
        let bank = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
        })?;
        Ok(bank
            .parameter_members
            .iter()
            .filter(|member| self.scope.is_none_or(|scope| member.key.bank() == scope))
            .cloned()
            .collect())
    }
    /// Wraps one selected native bank for shared provider/telemetry ownership.
    pub fn new(bank: AddressableParameterBank) -> Self {
        Self {
            inner: Arc::new(Mutex::new(bank)),
            scope: None,
        }
    }

    /// Restricts a handle to one bank while retaining the shared residency budget.
    pub fn scoped(&self, bank: usize) -> Result<Self, Error> {
        let inner = self.inner.lock().map_err(|_| {
            Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
        })?;
        if self.scope.is_some_and(|scope| scope != bank)
            || !inner.catalog.keys().any(|key| key.bank() == bank)
        {
            return Err(Error::ArchitectureModel(
                "bank scope is absent from the selected pool".into(),
            ));
        }
        Ok(Self {
            inner: Arc::clone(&self.inner),
            scope: Some(bank),
        })
    }

    /// Returns current telemetry for the shared native bank.
    pub fn report(&self) -> Result<ParameterBankResidencyReport, Error> {
        self.inner
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?
            .report_scoped(self.scope)
            .map_err(Into::into)
    }
}

impl AddressableParameterBank {
    /// Creates a disk-planned cache over exactly the supplied owned entries.
    #[cfg(test)]
    pub(crate) fn new<S, O>(
        store: Arc<S>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        options: O,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError>
    where
        S: eredu_checkpoint::store::CheckpointSource + 'static,
        O: Into<ParameterBankOptions>,
    {
        let store: eredu_checkpoint::store::RetainedCheckpointSource = store.into();
        Self::new_shared(store, entries, options.into(), source_stream, device_stream)
    }

    /// Creates a cache from an already type-erased checkpoint store.
    #[cfg(test)]
    pub(crate) fn new_shared<O>(
        store: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        options: O,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError>
    where
        O: Into<ParameterBankOptions>,
    {
        let store = store.into();
        Self::new_shared_with_policy(
            store,
            entries,
            options.into(),
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
            source_stream,
            device_stream,
            Vec::new(),
            BTreeMap::new(),
            None,
        )
    }

    /// Creates an ordinary cache from the exact selected transformation tasks.
    pub fn new_selected_shared<O>(
        store: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        selected: SelectedAddressableEntries,
        options: O,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError>
    where
        O: Into<ParameterBankOptions>,
    {
        Self::new_selected_shared_with_manager(
            store,
            selected,
            options,
            source_stream,
            device_stream,
            None,
        )
    }

    /// Prepares the actual selected catalog and its original manager before
    /// ordinary model ownership. Selected transforms execute once here; the
    /// move-only result retains their exact store, recipes and source identity.
    pub(crate) fn prepare_selected_manager(
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        selected: SelectedAddressableEntries,
        options: impl Into<ParameterBankOptions>,
        source_stream: &Stream,
        device_stream: &Stream,
        pool: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<Option<PreparedAddressableSource>, AddressableParameterBankError> {
        PreparedAddressableSource::prepare(
            store,
            selected,
            options.into(),
            source_stream,
            device_stream,
            pool,
        )
    }

    /// Consumes a matching prepared catalog, or runs the same ordinary resolver.
    pub(crate) fn new_selected_shared_with_manager<O>(
        store: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        selected: SelectedAddressableEntries,
        options: O,
        source_stream: Stream,
        device_stream: Stream,
        prepared: Option<PreparedAddressableSource>,
    ) -> Result<Self, AddressableParameterBankError>
    where
        O: Into<ParameterBankOptions>,
    {
        let store = store.into();
        let options = options.into();
        let (resolved, manager) = match prepared {
            Some(prepared) => prepared.into_selected(&store, &selected, options)?,
            None => (
                prepared::resolve_selected(store, selected, options, &source_stream)?,
                None,
            ),
        };
        resolved.into_bank(options, source_stream, device_stream, manager)
    }

    /// Creates a fully resident store over exactly the supplied owned entries.
    ///
    /// Every entry is pinned on the execution device during construction. The
    /// same selection compaction and backend-neutral binding machinery is used
    /// by sparse and resident execution, but resident entries cannot be evicted
    /// and never trigger checkpoint reads during a forward pass.
    #[cfg(test)]
    pub(crate) fn new_resident_shared(
        store: impl Into<eredu_checkpoint::store::RetainedCheckpointSource>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError> {
        let store = store.into();
        Self::new_shared_with_policy(
            store,
            entries,
            ParameterBankOptions::default(),
            ResidencyPolicy::Pinned,
            MemoryTier::Device,
            source_stream,
            device_stream,
            Vec::new(),
            BTreeMap::new(),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_shared_with_policy(
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        options: ParameterBankOptions,
        policy: ResidencyPolicy,
        initial_tier: MemoryTier,
        source_stream: Stream,
        device_stream: Stream,
        weight_quantizations: Vec<WeightQuantization>,
        placements: BTreeMap<ParameterBankKey, eredu_runtime::AddressableBankMemberPlacement>,
        materialization: Option<WeightMaterializationReport>,
    ) -> Result<Self, AddressableParameterBankError> {
        Self::new_shared_with_prepared_policy(
            store,
            entries,
            options,
            policy,
            initial_tier,
            source_stream,
            device_stream,
            weight_quantizations,
            placements,
            materialization,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_shared_with_prepared_policy(
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        options: ParameterBankOptions,
        policy: ResidencyPolicy,
        initial_tier: MemoryTier,
        source_stream: Stream,
        device_stream: Stream,
        weight_quantizations: Vec<WeightQuantization>,
        placements: BTreeMap<ParameterBankKey, eredu_runtime::AddressableBankMemberPlacement>,
        materialization: Option<WeightMaterializationReport>,
        prepared_manager: Option<ResidencyManager>,
    ) -> Result<Self, AddressableParameterBankError> {
        options.validate()?;
        let mut catalog = BTreeMap::new();
        let mut definitions = Vec::new();
        let mut specs = Vec::new();
        #[cfg(test)]
        let mut namespace_entry_counts = BTreeMap::new();
        #[cfg(test)]
        let mut namespace_global_spans = BTreeMap::new();
        for entry in entries {
            if catalog.insert(entry.identity, entry.bytes).is_some() {
                return Err(AddressableParameterBankError::DuplicateEntry {
                    identity: entry.identity,
                });
            }
            #[cfg(test)]
            {
                *namespace_entry_counts
                    .entry(entry.identity.unit())
                    .or_insert(0) += 1;
            }
            #[cfg(test)]
            {
                namespace_global_spans
                    .entry(entry.identity.unit())
                    .and_modify(|span: &mut usize| *span = (*span).max(entry.identity.member() + 1))
                    .or_insert(entry.identity.member() + 1);
            }
            specs.push(OffloadUnitSpec::new(
                entry.identity.unit_id(),
                entry.bytes,
                policy,
                initial_tier,
            )?);
            definitions.push(entry.unit);
        }
        if catalog.is_empty() {
            return Err(AddressableParameterBankError::EmptyCatalog);
        }
        let plan = OffloadPlan::new(options.storage, specs)?;
        let manager = match prepared_manager {
            Some(manager) => {
                manager.validate_original_layerwise_preparation(
                    &store,
                    &BTreeMap::new(),
                    &plan,
                    &definitions,
                    &source_stream,
                    &device_stream,
                )?;
                manager
            }
            None => {
                let manager = ResidencyManager::new_shared(
                    store,
                    plan,
                    definitions,
                    source_stream,
                    device_stream,
                )?;
                manager.initialize()?;
                manager
            }
        };
        let statistics = ParameterBankStatisticsTable::new(&catalog);
        static NEXT_POOL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Ok(Self {
            parameter_members: Vec::new(),
            parameter_replacements: Default::default(),
            parameter_revision: 0,
            effective_member_bytes: catalog.clone(),
            pool_id: NEXT_POOL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            manager,
            unit_banks: catalog
                .keys()
                .map(|key| (key.unit_id(), key.bank()))
                .collect(),
            catalog,
            #[cfg(test)]
            namespace_entry_counts,
            #[cfg(test)]
            namespace_global_spans,
            host_budget: if initial_tier == MemoryTier::Device {
                Some(0)
            } else {
                options.storage.host_budget_bytes()
            },
            scratch_limit: options.compact_bank_scratch_bytes,
            #[cfg(test)]
            bulk_bank_target: options.bulk_compact_bank_target_bytes,
            statistics: Mutex::new(statistics),
            weight_quantizations,
            placements,
            materialization,
        })
    }

    /// Returns all load-time encodings of transformed entry bindings.
    pub fn weight_quantizations(&self) -> &[WeightQuantization] {
        &self.weight_quantizations
    }

    /// Returns the underlying reusable residency manager.
    pub const fn residency_manager(&self) -> &ResidencyManager {
        &self.manager
    }

    /// Returns the largest source member in the generic storage catalog.
    #[cfg(test)]
    pub(crate) fn maximum_member_bytes(&self) -> u64 {
        self.catalog.values().copied().max().unwrap_or(0)
    }

    /// Returns the hard compact-bank byte limit.
    #[cfg(test)]
    pub(crate) const fn compact_bank_scratch_bytes(&self) -> u64 {
        self.scratch_limit
    }

    /// Returns the bulk compact-bank working-set target.
    #[cfg(test)]
    pub(crate) const fn bulk_compact_bank_target_bytes(&self) -> u64 {
        self.bulk_bank_target
    }

    /// Returns the admitted global member span for a caller-owned namespace.
    #[cfg(test)]
    pub(crate) fn namespace_global_span(&self, namespace: usize) -> Option<usize> {
        self.namespace_global_spans.get(&namespace).copied()
    }

    /// Completes one generic acquisition after its dependent output is evaluated.
    #[cfg(test)]
    pub(crate) fn complete_acquisition(
        &self,
        mut acquisition: AcquiredParameterGroups,
        output: &Array,
    ) -> Result<(), AddressableParameterBankError> {
        eval([output])?;
        acquisition.transfer.synchronize()?;
        Ok(())
    }

    /// Acquires a caller-provided selection table while preserving its exact shape and order.
    #[cfg(test)]
    pub fn acquire_selection_slice(
        &self,
        namespace: usize,
        grouped_ids: &[i32],
        selection_shape: &[i32],
        pass: BankAccessClass,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        let expected_elements = selection_shape
            .iter()
            .try_fold(1usize, |count, dimension| {
                let dimension = usize::try_from(*dimension).map_err(|_| {
                    AddressableParameterBankError::InvalidSelectionShape(selection_shape.to_vec())
                })?;
                count.checked_mul(dimension).ok_or_else(|| {
                    AddressableParameterBankError::InvalidSelectionShape(selection_shape.to_vec())
                })
            })?;
        if expected_elements != grouped_ids.len() {
            return Err(AddressableParameterBankError::SelectionShapeMismatch {
                shape: selection_shape.to_vec(),
                elements: grouped_ids.len(),
            });
        }
        let namespace_count = self
            .namespace_entry_counts
            .get(&namespace)
            .copied()
            .ok_or(AddressableParameterBankError::UnknownNamespace { namespace })?;
        let mut demand = BTreeMap::<ParameterBankKey, u64>::new();
        for id in grouped_ids {
            let global_entry = usize::try_from(*id).map_err(|_| {
                AddressableParameterBankError::InvalidEntryId {
                    namespace,
                    entry: i64::from(*id),
                    known_owned_entries: namespace_count,
                }
            })?;
            let identity = ParameterBankKey::new(0, namespace, global_entry);
            if !self.catalog.contains_key(&identity) {
                return Err(AddressableParameterBankError::MissingOwnedEntry { identity });
            }
            let count = demand.entry(identity).or_insert(0);
            *count = count.saturating_add(1);
        }

        let compact_ids = demand.keys().copied().collect::<Vec<_>>();
        self.acquire_demand(demand.into_iter().collect(), compact_ids,
            u64::try_from(grouped_ids.len()).map_err(|_| AddressableParameterBankError::ByteOverflow)?,
            pass, stream, None)
    }

    fn acquire_demand(
        &self,
        demand: Vec<(ParameterBankKey, u64)>,
        compact_ids: Vec<ParameterBankKey>,
        selection_count: u64,
        pass: BankAccessClass,
        stream: &Stream,
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        let bank = compact_ids.first().map(|key| key.bank());
        if compact_ids.iter().any(|key| Some(key.bank()) != bank) {
            return Err(AddressableParameterBankError::MixedBanks);
        }
        let scratch_bytes = demand.iter().try_fold(0u64, |total, (identity, _)| {
            total
                .checked_add(self.effective_member_bytes[identity])
                .ok_or(AddressableParameterBankError::ByteOverflow)
        })?;
        if scratch_bytes > self.scratch_limit {
            return Err(AddressableParameterBankError::ScratchLimitExceeded {
                required_bytes: scratch_bytes,
                limit_bytes: self.scratch_limit,
                distinct_entries: demand.len(),
            });
        }
        let before = self.resident_snapshot(funding)?;
        let started = Instant::now();
        let mut host_hits = 0u64;
        let mut host_misses = 0u64;
        let mut device_hits = 0u64;
        let mut device_misses = 0u64;
        let mut requests = ordinary_host::vector(compact_ids.len(), funding)?;
        let mut host_requests = ordinary_host::vector(compact_ids.len(), funding)?;
        for &(identity, selection_demand) in &demand {
            let unit = ordinary_host::unit_id(identity, funding)?;
            let host_hit = self.manager.is_resident(&unit, MemoryTier::Host)?;
            let device_hit = self.manager.is_resident(&unit, MemoryTier::Device)?;
            if host_hit {
                host_hits = host_hits.saturating_add(1);
            } else {
                host_misses = host_misses.saturating_add(1);
            }
            if device_hit {
                device_hits = device_hits.saturating_add(1);
            } else {
                device_misses = device_misses.saturating_add(1);
            }
            if host_hit || (!device_hit && self.host_budget != Some(0)) {
                host_requests.push((ordinary_host::clone_unit_id(&unit, funding)?, selection_demand));
            }
            requests.push((unit, selection_demand));
        }
        if !host_requests.is_empty() {
            let host = match funding {
                Some(funding) => self.manager.acquire_many_with_host_transfer(
                    &host_requests, MemoryTier::Host, funding).and_then(|mut transfer| transfer.synchronize()),
                None => self.manager.acquire_many_with_demand(&host_requests, MemoryTier::Host).map(drop),
            };
            match host {
                Ok(()) => {},
                Err(ResidencyError::Ledger(ResidencyLedgerError::BudgetExhausted {
                    tier: MemoryTier::Host,
                    ..
                })) => {}
                Err(error) => return Err(error.into()),
            }
        }
        let transfer = match funding {
            Some(funding) => self.manager.acquire_many_with_host_transfer(&requests, MemoryTier::Device, funding)?,
            None => self.manager.acquire_many_with_transfer(&requests, MemoryTier::Device)?,
        };
        transfer.order_after(stream)?;
        let wait = started.elapsed();
        let after = self.resident_snapshot(funding)?;
        let (host_evictions, host_eviction_bytes) = before.evicted(&after, MemoryTier::Host);
        let (device_evictions, device_eviction_bytes) = before.evicted(&after, MemoryTier::Device);

        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| AddressableParameterBankError::StatisticsPoisoned)?;
        if let Some(bank) = bank {
            let stats = statistics.get_mut(bank)?.pass_mut(pass);
            let distinct = compact_ids.len() as u64;
            stats.requested_selections = stats.requested_selections.saturating_add(selection_count);
            stats.distinct_entries = stats.distinct_entries.saturating_add(distinct);
            stats.coalesced_duplicates = stats
                .coalesced_duplicates
                .saturating_add(selection_count.saturating_sub(distinct));
            stats.materialization_wait = stats.materialization_wait.saturating_add(wait);
            stats.host.requests = stats.host.requests.saturating_add(distinct);
            stats.host.hits = stats.host.hits.saturating_add(host_hits);
            stats.host.misses = stats.host.misses.saturating_add(host_misses);
            stats.host.evictions = stats.host.evictions.saturating_add(host_evictions);
            stats.host.eviction_bytes = stats
                .host
                .eviction_bytes
                .saturating_add(host_eviction_bytes);
            stats.device.requests = stats.device.requests.saturating_add(distinct);
            stats.device.hits = stats.device.hits.saturating_add(device_hits);
            stats.device.misses = stats.device.misses.saturating_add(device_misses);
            stats.device.evictions = stats.device.evictions.saturating_add(device_evictions);
            stats.device.eviction_bytes = stats
                .device
                .eviction_bytes
                .saturating_add(device_eviction_bytes);
        }
        // Each actual catalog bank has one fixed statistics row. The borrowed
        // residency snapshot supplies occupancy without another dynamic map.
        for (&bank, stats) in statistics.iter_mut() {
            let (host, device) = after.rows.iter().filter(|(id, _, _)| self.unit_banks[*id] == bank)
                .try_fold((0u64, 0u64), |(host, device), (_, h, d)| {
                    Some((host.checked_add(h.unwrap_or(0))?, device.checked_add(d.unwrap_or(0))?))
                }).ok_or(AddressableParameterBankError::ByteOverflow)?;
            stats.peak_host_bytes = stats.peak_host_bytes.max(host);
            stats.peak_device_bytes = stats.peak_device_bytes.max(device);
        }
        drop(statistics);

        let mut counts = ordinary_host::vector(demand.len(), funding)?;
        counts.extend(demand.into_iter().map(|(_, count)| count));
        Ok(AcquiredParameterGroups {
            identities: compact_ids,
            demand: counts,
            scratch_bytes,
            pass,
            transfer,
            original: None,
            ordinary: None,
            ordinary_chunk: None,
        })
    }

    /// Acquires a deterministic, already coalesced generic entry demand.
    pub fn acquire_entry_demand(
        &self,
        entries: &[(ParameterBankKey, u64)],
        pass: BankAccessClass,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        self.acquire_entry_demand_with_host(entries, pass, stream, None)
    }

    fn acquire_entry_demand_with_host(
        &self,
        entries: &[(ParameterBankKey, u64)],
        pass: BankAccessClass,
        stream: &Stream,
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        ordinary_host::fund_acquisition(funding)?;
        if entries.is_empty() {
            return Err(AddressableParameterBankError::EmptyDemand);
        }
        let mut demand = ordinary_host::vector(entries.len(), funding)?;
        let mut selection_count = 0u64;
        for &(identity, count) in entries {
            if count == 0 {
                return Err(AddressableParameterBankError::ZeroDemand { identity });
            }
            if !self.catalog.contains_key(&identity) {
                return Err(AddressableParameterBankError::MissingOwnedEntry { identity });
            }
            demand.push((identity, count));
            selection_count = selection_count
                .checked_add(count)
                .ok_or(AddressableParameterBankError::ByteOverflow)?;
        }
        demand.sort_unstable_by_key(|(identity, _)| *identity);
        if let Some(pair) = demand.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(AddressableParameterBankError::DuplicateDemand { identity: pair[0].0 });
        }
        let mut compact_ids = ordinary_host::vector(demand.len(), funding)?;
        compact_ids.extend(demand.iter().map(|(key, _)| *key));
        self.acquire_demand(demand, compact_ids, selection_count, pass, stream, funding)
    }

    /// Records a completed compact-bank construction.
    pub fn record_compact_bank(
        &self,
        bank: usize,
        pass: BankAccessClass,
        bytes: u64,
        duration: Duration,
    ) -> Result<(), AddressableParameterBankError> {
        if bytes > self.scratch_limit {
            return Err(AddressableParameterBankError::ScratchLimitExceeded {
                required_bytes: bytes,
                limit_bytes: self.scratch_limit,
                distinct_entries: 0,
            });
        }
        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| AddressableParameterBankError::StatisticsPoisoned)?;
        let stats = statistics.get_mut(bank)?.pass_mut(pass);
        stats.compact_banks = stats.compact_banks.saturating_add(1);
        stats.compact_bank_bytes = stats.compact_bank_bytes.saturating_add(bytes);
        stats.peak_compact_bank_bytes = stats.peak_compact_bank_bytes.max(bytes);
        stats.compact_bank_time = stats.compact_bank_time.saturating_add(duration);
        Ok(())
    }

    /// Returns current entry residency, transfer, storage, and pass statistics.
    pub fn report(&self) -> Result<ParameterBankResidencyReport, AddressableParameterBankError> {
        self.report_scoped(None)
    }

    fn report_scoped(
        &self,
        scope: Option<usize>,
    ) -> Result<ParameterBankResidencyReport, AddressableParameterBankError> {
        let residency = self.manager.report()?;
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
            .map_err(|_| AddressableParameterBankError::StatisticsPoisoned)?;
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

    fn resident_snapshot(&self, funding: Option<&eredu_nn::workspace::HostMetadataFunding>)
        -> Result<ResidentSnapshot<'_>, AddressableParameterBankError> {
        let mut rows = ordinary_host::vector(self.unit_banks.len(), funding)?;
        rows.extend(self.unit_banks.keys().map(|id| (id, None, None)));
        self.manager.fill_resident_capacities(&mut rows)?;
        Ok(ResidentSnapshot { rows })
    }
}

struct ResidentSnapshot<'a> {
    rows: Vec<(&'a OffloadUnitId, Option<u64>, Option<u64>)>,
}

impl ResidentSnapshot<'_> {
    fn evicted(&self, after: &Self, tier: MemoryTier) -> (u64, u64) {
        self.rows.iter().zip(&after.rows).fold((0u64, 0u64), |(count, bytes), (before, after)| {
            debug_assert!(std::ptr::eq(before.0, after.0));
            let (before, after) = match tier {
                MemoryTier::Host => (before.1, after.1),
                MemoryTier::Device => (before.2, after.2),
                MemoryTier::Disk => return (count, bytes),
            };
            match before.filter(|_| after.is_none()) {
                Some(size) => (count.saturating_add(1), bytes.saturating_add(size)),
                None => (count, bytes),
            }
        })
    }
}

/// A deterministic compact selection table and the leases protecting its sources.
pub struct AcquiredParameterGroups {
    pub(super) identities: Vec<ParameterBankKey>,
    pub(super) demand: Vec<u64>,
    pub(super) scratch_bytes: u64,
    pub(super) pass: BankAccessClass,
    pub(super) transfer: ResidentTransfer,
    // Transfer/payloads retire before their issuing demand and source custody.
    pub(super) original: Option<super::movement::OriginalIndexedChunkSource>,
    pub(super) ordinary: Option<OrdinaryBankHostSource>,
    pub(super) ordinary_chunk: Option<super::movement::OrdinaryIndexedChunkSource>,
}

impl AcquiredParameterGroups {
    /// Returns selected entries in compact-bank order.
    pub fn identities(&self) -> &[ParameterBankKey] {
        &self.identities
    }

    /// Returns duplicate-preserving demand counts in compact-bank order.
    pub fn demand(&self) -> &[u64] {
        &self.demand
    }

    /// Returns the conservatively reserved compact-bank byte count.
    pub const fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }

    /// Returns the execution-path classification used for telemetry.
    pub const fn pass(&self) -> BankAccessClass {
        self.pass
    }

    /// Returns source leases in the same order as [`Self::identities`].
    #[cfg(test)]
    pub fn leases(&self) -> &[ResidentUnitLease] {
        self.transfer.leases()
    }

    /// Concatenates one required per-entry binding along its leading axis.
    pub fn compact_binding(
        &self,
        name: &str,
        stream: &Stream,
    ) -> Result<Array, AddressableParameterBankError> {
        let values = self
            .transfer
            .leases()
            .iter()
            .map(|lease| lease.device_value(name).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        if values.is_empty() {
            return Err(AddressableParameterBankError::EmptyCompactBinding {
                name: name.to_string(),
            });
        }
        Ok(concatenate_axis(&values, 0, stream)?)
    }

    /// Concatenates an optional companion binding when every entry provides it.
    pub fn optional_compact_binding(
        &self,
        name: &str,
        stream: &Stream,
    ) -> Result<Option<Array>, AddressableParameterBankError> {
        let present = self
            .transfer
            .leases()
            .iter()
            .map(|lease| lease.binding_names().any(|binding| binding == name))
            .collect::<Vec<_>>();
        if present.iter().all(|value| !value) {
            return Ok(None);
        }
        if present.iter().any(|value| !value) {
            return Err(AddressableParameterBankError::InconsistentCompanion {
                name: name.to_string(),
            });
        }
        self.compact_binding(name, stream).map(Some)
    }

    /// Returns whether no grouped entries were selected.
    pub fn is_empty(&self) -> bool {
        self.identities.is_empty()
    }
}
