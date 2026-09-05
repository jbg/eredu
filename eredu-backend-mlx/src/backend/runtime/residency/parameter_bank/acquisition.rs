//! Addressable entry acquisition, residency, and completion.

use super::*;

/// Shared entry catalog, scheduler, residency manager, and telemetry.
pub struct AddressableParameterBank {
    pub(super) manager: ResidencyManager,
    pub(super) catalog: BTreeMap<ParameterBankKey, u64>,
    #[cfg(test)]
    pub(super) namespace_entry_counts: BTreeMap<usize, usize>,
    #[cfg(test)]
    pub(super) namespace_global_spans: BTreeMap<usize, usize>,
    pub(super) host_budget: Option<u64>,
    pub(super) scratch_limit: u64,
    #[cfg(test)]
    pub(super) bulk_bank_target: u64,
    pub(super) statistics: Mutex<ParameterBankStatistics>,
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
            source: Box::new(Error::ArchitectureModel(error.to_string())),
        })?;
    }
    Ok(())
}

impl SharedAddressableParameterBank {
    /// Wraps one selected native bank for shared provider/telemetry ownership.
    pub fn new(bank: AddressableParameterBank) -> Self {
        Self {
            inner: Arc::new(Mutex::new(bank)),
        }
    }

    /// Returns current telemetry for the shared native bank.
    pub fn report(&self) -> Result<ParameterBankResidencyReport, Error> {
        self.inner
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?
            .report()
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
        let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> = store;
        Self::new_shared(store, entries, options.into(), source_stream, device_stream)
    }

    /// Creates a cache from an already type-erased checkpoint store.
    #[cfg(test)]
    pub(crate) fn new_shared<O>(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        options: O,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError>
    where
        O: Into<ParameterBankOptions>,
    {
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

    /// Creates a cache from exact per-binding selected transformation tasks.
    pub fn new_selected_shared<O>(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        selected: SelectedAddressableEntries,
        options: O,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError>
    where
        O: Into<ParameterBankOptions>,
    {
        let options = options.into();
        preflight_selected_entry_bindings(store.as_ref(), &selected.entries)?;
        let selected_keys = selected
            .entries
            .iter()
            .map(|entry| entry.identity)
            .collect::<std::collections::BTreeSet<_>>();
        if selected_keys
            != selected
                .placements
                .keys()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
        {
            return Err(AddressableParameterBankError::Transformation {
                source: Box::new(Error::ArchitectureModel(
                    "selected addressable placements do not cover the exact entry keys".into(),
                )),
            });
        }
        for entry in &selected.entries {
            let expected = selected
                .expected_bytes
                .get(&entry.identity)
                .ok_or_else(|| AddressableParameterBankError::Transformation {
                    source: Box::new(Error::ArchitectureModel(format!(
                        "selected addressable entry {:?} has no neutral byte total",
                        entry.identity
                    ))),
                })?;
            let projected = entry
                .unit
                .bindings()
                .iter()
                .try_fold(0u64, |total, binding| {
                    let bytes = if let Some(transform) = selected
                        .transformations
                        .get(&(entry.identity, binding.name().to_owned()))
                    {
                        let metadata = binding.source_recipe().infer(store.as_ref())?;
                        packed_projection_bytes(
                            metadata.shape(),
                            transform.quantization,
                            &transform.companion_dtype,
                        )?
                    } else {
                        binding.expected_bytes()
                    };
                    total.checked_add(bytes).ok_or_else(|| {
                        Error::ArchitectureModel(
                            "selected addressable entry bytes overflowed".into(),
                        )
                    })
                })
                .map_err(|source| AddressableParameterBankError::Transformation {
                    source: Box::new(source),
                })?;
            if projected != *expected {
                return Err(AddressableParameterBankError::Transformation {
                    source: Box::new(Error::ArchitectureModel(format!(
                        "selected addressable entry {:?} bytes differ: expected {}, projected {}",
                        entry.identity, expected, projected
                    ))),
                });
            }
        }
        if selected.transformations.is_empty() {
            return Self::new_shared_with_policy(
                store,
                selected.entries,
                options,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
                source_stream,
                device_stream,
                Vec::new(),
                selected.placements,
                None,
            );
        }
        let telemetry_formats = selected_transformation_formats(&selected.transformations);
        let transformed = quantize_selected_entry_catalog(
            store,
            selected.entries,
            selected.transformations,
            options.compact_bank_scratch_bytes,
            &source_stream,
        )
        .map_err(|source| AddressableParameterBankError::Transformation {
            source: Box::new(source),
        })?;
        for entry in &transformed.entries {
            if selected.expected_bytes.get(&entry.identity) != Some(&entry.bytes) {
                return Err(AddressableParameterBankError::Transformation {
                    source: Box::new(Error::ArchitectureModel(format!(
                        "materialized addressable entry {:?} differs from its neutral selected bytes",
                        entry.identity
                    ))),
                });
            }
        }
        Self::new_shared_with_policy(
            transformed.store,
            transformed.entries,
            options,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
            source_stream,
            device_stream,
            telemetry_formats,
            selected.placements,
            Some(transformed.report),
        )
    }

    /// Creates a fully resident store over exactly the supplied owned entries.
    ///
    /// Every entry is pinned on the execution device during construction. The
    /// same selection compaction and backend-neutral binding machinery is used
    /// by sparse and resident execution, but resident entries cannot be evicted
    /// and never trigger checkpoint reads during a forward pass.
    #[cfg(test)]
    pub(crate) fn new_resident_shared(
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
        entries: impl IntoIterator<Item = ParameterBankEntry>,
        source_stream: Stream,
        device_stream: Stream,
    ) -> Result<Self, AddressableParameterBankError> {
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
        store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
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
                    .entry(entry.identity.namespace)
                    .or_insert(0) += 1;
            }
            #[cfg(test)]
            {
                namespace_global_spans
                    .entry(entry.identity.namespace)
                    .and_modify(|span: &mut usize| *span = (*span).max(entry.identity.index + 1))
                    .or_insert(entry.identity.index + 1);
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
        let manager =
            ResidencyManager::new_shared(store, plan, definitions, source_stream, device_stream)?;
        manager.initialize()?;
        Ok(Self {
            manager,
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
            statistics: Mutex::new(ParameterBankStatistics::default()),
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
            let identity = ParameterBankKey::new(namespace, global_entry);
            if !self.catalog.contains_key(&identity) {
                return Err(AddressableParameterBankError::MissingOwnedEntry { identity });
            }
            let count = demand.entry(identity).or_insert(0);
            *count = count.saturating_add(1);
        }

        let compact_ids = demand.keys().copied().collect::<Vec<_>>();
        self.acquire_demand(demand, compact_ids, grouped_ids.len() as u64, pass, stream)
    }

    fn acquire_demand(
        &self,
        demand: BTreeMap<ParameterBankKey, u64>,
        compact_ids: Vec<ParameterBankKey>,
        selection_count: u64,
        pass: BankAccessClass,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        let scratch_bytes = demand.keys().try_fold(0u64, |total, identity| {
            total
                .checked_add(self.catalog[identity])
                .ok_or(AddressableParameterBankError::ByteOverflow)
        })?;
        if scratch_bytes > self.scratch_limit {
            return Err(AddressableParameterBankError::ScratchLimitExceeded {
                required_bytes: scratch_bytes,
                limit_bytes: self.scratch_limit,
                distinct_entries: demand.len(),
            });
        }
        let before = self.resident_snapshot()?;
        let started = Instant::now();
        let mut host_hits = 0u64;
        let mut host_misses = 0u64;
        let mut device_hits = 0u64;
        let mut device_misses = 0u64;
        let mut requests = Vec::with_capacity(compact_ids.len());
        let mut host_requests = Vec::new();
        for identity in &compact_ids {
            let unit = identity.unit_id();
            let selection_demand = demand[identity];
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
                host_requests.push((unit.clone(), selection_demand));
            }
            requests.push((unit, selection_demand));
        }
        if !host_requests.is_empty() {
            match self
                .manager
                .acquire_many_with_demand(&host_requests, MemoryTier::Host)
            {
                Ok(host) => drop(host),
                Err(ResidencyError::Ledger(ResidencyLedgerError::BudgetExhausted {
                    tier: MemoryTier::Host,
                    ..
                })) => {}
                Err(error) => return Err(error.into()),
            }
        }
        let transfer = self
            .manager
            .acquire_many_with_transfer(&requests, MemoryTier::Device)?;
        transfer.order_after(stream)?;
        let wait = started.elapsed();
        let after = self.resident_snapshot()?;
        let (host_evictions, host_eviction_bytes) = before.evicted(&after, MemoryTier::Host);
        let (device_evictions, device_eviction_bytes) = before.evicted(&after, MemoryTier::Device);

        let mut statistics = self
            .statistics
            .lock()
            .map_err(|_| AddressableParameterBankError::StatisticsPoisoned)?;
        let stats = statistics.pass_mut(pass);
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
        drop(statistics);

        Ok(AcquiredParameterGroups {
            identities: compact_ids,
            demand: demand.into_values().collect(),
            scratch_bytes,
            pass,
            transfer,
        })
    }

    /// Acquires a deterministic, already coalesced generic entry demand.
    pub fn acquire_entry_demand(
        &self,
        entries: &[(ParameterBankKey, u64)],
        pass: BankAccessClass,
        stream: &Stream,
    ) -> Result<AcquiredParameterGroups, AddressableParameterBankError> {
        if entries.is_empty() {
            return Err(AddressableParameterBankError::EmptyDemand);
        }
        let mut demand = BTreeMap::new();
        let mut selection_count = 0u64;
        for &(identity, count) in entries {
            if count == 0 {
                return Err(AddressableParameterBankError::ZeroDemand { identity });
            }
            if !self.catalog.contains_key(&identity) {
                return Err(AddressableParameterBankError::MissingOwnedEntry { identity });
            }
            if demand.insert(identity, count).is_some() {
                return Err(AddressableParameterBankError::DuplicateDemand { identity });
            }
            selection_count = selection_count
                .checked_add(count)
                .ok_or(AddressableParameterBankError::ByteOverflow)?;
        }
        let compact_ids = demand.keys().copied().collect::<Vec<_>>();
        self.acquire_demand(demand, compact_ids, selection_count, pass, stream)
    }

    /// Records a completed compact-bank construction.
    pub fn record_compact_bank(
        &self,
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
        let stats = statistics.pass_mut(pass);
        stats.compact_banks = stats.compact_banks.saturating_add(1);
        stats.compact_bank_bytes = stats.compact_bank_bytes.saturating_add(bytes);
        stats.peak_compact_bank_bytes = stats.peak_compact_bank_bytes.max(bytes);
        stats.compact_bank_time = stats.compact_bank_time.saturating_add(duration);
        Ok(())
    }

    /// Returns current entry residency, transfer, storage, and pass statistics.
    pub fn report(&self) -> Result<ParameterBankResidencyReport, AddressableParameterBankError> {
        let residency = self.manager.report()?;
        let mut host_resident_entries = 0;
        let mut device_resident_entries = 0;
        let mut host_resident_bytes = 0u64;
        let mut device_resident_bytes = 0u64;
        for unit in residency.units() {
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
            weight_quantizations: self.weight_quantizations.clone(),
            placements: self
                .placements
                .iter()
                .map(|(key, placement)| (*key, placement.clone()))
                .collect(),
            owned_entries: self.catalog.len(),
            owned_bytes: self.catalog.values().copied().sum(),
            host_resident_entries,
            device_resident_entries,
            host_resident_bytes,
            device_resident_bytes,
            peak_host_resident_bytes: residency
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Host),
            peak_device_resident_bytes: residency
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device),
            bulk: statistics.bulk,
            incremental: statistics.incremental,
            residency,
            materialization: self.materialization.clone(),
        })
    }

    fn resident_snapshot(&self) -> Result<ResidentSnapshot, AddressableParameterBankError> {
        let report = self.manager.report()?;
        Ok(ResidentSnapshot {
            host: report
                .units()
                .iter()
                .filter(|unit| unit.host_resident())
                .map(|unit| (unit.id().clone(), unit.host_allocated_bytes()))
                .collect(),
            device: report
                .units()
                .iter()
                .filter(|unit| unit.device_resident())
                .map(|unit| (unit.id().clone(), unit.device_allocated_bytes()))
                .collect(),
        })
    }
}

struct ResidentSnapshot {
    pub(super) host: BTreeMap<OffloadUnitId, u64>,
    pub(super) device: BTreeMap<OffloadUnitId, u64>,
}

impl ResidentSnapshot {
    fn evicted(&self, after: &Self, tier: MemoryTier) -> (u64, u64) {
        let (before, after) = match tier {
            MemoryTier::Host => (&self.host, &after.host),
            MemoryTier::Device => (&self.device, &after.device),
            MemoryTier::Disk => return (0, 0),
        };
        before
            .iter()
            .filter(|(id, _)| !after.contains_key(*id))
            .fold((0u64, 0u64), |(count, bytes), (_, size)| {
                (count.saturating_add(1), bytes.saturating_add(*size))
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
