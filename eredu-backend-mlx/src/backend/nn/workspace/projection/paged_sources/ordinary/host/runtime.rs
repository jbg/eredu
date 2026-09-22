//! Runtime loans retain each transfer prefix outside the catalog borrow.
use super::*;

pub(super) struct Promotion<'a, 'source> {
    pub(super) source: &'a TransferSource<'source>,
    pub(super) id: &'a CacheBlockId,
}
impl PreparedHostPromotion for Promotion<'_, '_> {
    type Cause = OrdinaryPagedCause;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        self.source.validate_manager(manager, generation)
    }
    fn validate_id(&self, id: &CacheBlockId) -> Result<(), Exception> {
        if id != self.id {
            return Err(self.error(CacheSourceError::Identity.into()));
        }
        Ok(())
    }
    fn validate_context(&self, context: &WorkspaceContext) -> Result<(), Exception> {
        let prepared = &self.source.work.program.inner.context;
        let valid = prepared.shares_trace(context)
            && prepared
                .metadata_funding()
                .zip(context.metadata_funding())
                .is_some_and(|(expected, actual)| expected.same_account(&actual));
        if !valid {
            return Err(self.error(CacheSourceError::Identity.into()));
        }
        self.source.validate_manager(
            self.source.installed.manager(),
            self.source.installed.initial_generation(),
        )
    }
    fn publication_controls(&self) -> usize {
        self.source.installed.publication_control_bytes()
    }
    fn error(&self, cause: Self::Cause) -> Exception {
        self.source.error(cause)
    }
}
struct Return<'a, 'source> {
    source: &'a TransferSource<'source>,
    id: &'a CacheBlockId,
    owned_pins: usize,
}
impl PreparedHostReturn for Return<'_, '_> {
    fn id(&self) -> &CacheBlockId {
        self.id
    }
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        self.source.validate_manager(manager, generation)
    }
    fn validate_pin_counts(
        &self,
        lifecycle: &eredu_runtime::CacheBlockLifecycle,
    ) -> Result<(), Exception> {
        validate_pins(self.source, self.id, self.owned_pins, lifecycle)
    }
    fn error(&self, cause: OrdinaryPagedCause) -> Exception {
        self.source.error(cause)
    }
}
fn validate_pins(
    source: &TransferSource<'_>,
    id: &CacheBlockId,
    owned: usize,
    lifecycle: &eredu_runtime::CacheBlockLifecycle,
) -> Result<(), Exception> {
    if lifecycle
        .is_device_leased(id)
        .map_err(|cause| source.error(CacheSourceError::Lifecycle(cause).into()))?
        || lifecycle
            .source_pin_count(id)
            .map_err(|cause| source.error(CacheSourceError::Lifecycle(cause).into()))?
            != owned
    {
        return Err(source.error(CacheSourceError::Busy.into()));
    }
    Ok(())
}
impl HostProgram {
    pub(super) fn pin_count(
        &self,
        work: &OrdinaryPagedWork,
        manager: &CacheResidencyManager,
        id: &CacheBlockId,
    ) -> Result<usize, CacheSourceError> {
        let sources = work.program.inner.sources.sources();
        let mut count = 0usize;
        for source in sources {
            if source.manager().same_catalog(manager) {
                count = count
                    .checked_add(source.owned_source_pin_count(id))
                    .ok_or(CacheSourceError::Overflow)?;
            }
        }
        for store in &self.stores {
            if &store.id == id && sources[store.source].manager().same_catalog(manager) {
                count = count
                    .checked_add(usize::from(store.pin.is_some()))
                    .and_then(|n| {
                        n.checked_add(
                            store
                                .write
                                .as_ref()
                                .map_or(0, DiskWriteOperation::completed_source_pin_count),
                        )
                    })
                    .ok_or(CacheSourceError::Overflow)?;
            }
        }
        for load in &self.loads {
            let store = &self.stores[load.store];
            if &store.id == id && sources[store.source].manager().same_catalog(manager) {
                count = count
                    .checked_add(usize::from(load.promotion.owns_pin()))
                    .and_then(|n| n.checked_add(load.promotion.read_source_pin_count()))
                    .and_then(|n| {
                        n.checked_add(
                            load.read_operation
                                .as_ref()
                                .map_or(0, DiskReadOperation::completed_source_pin_count),
                        )
                    })
                    .ok_or(CacheSourceError::Overflow)?;
            }
        }
        let state = work
            .program
            .inner
            .state
            .try_borrow()
            .map_err(|_| CacheSourceError::Busy)?;
        for row in state.rows.iter().flatten() {
            if !sources[row.source].manager().same_catalog(manager) {
                continue;
            }
            if let Some(scan) = &row.scan {
                for (key, row) in scan.ids.iter().zip(&scan.rows) {
                    if key == id {
                        count = count
                            .checked_add(usize::from(row.pin.is_some()))
                            .ok_or(CacheSourceError::Overflow)?;
                    }
                }
            }
        }
        Ok(count)
    }
    fn ensure_initial(
        &mut self,
        index: usize,
        proof: &TransferSource<'_>,
    ) -> Result<(), Exception> {
        let store = &mut self.stores[index];
        if store.pair.values().is_some() {
            if store.pin.is_none() {
                return Err(proof.error(CacheSourceError::Identity.into()));
            }
            return Ok(());
        }
        proof
            .installed
            .with_source(selection(&store.id), |mut loan| {
                if store.pin.is_some() {
                    return Err(CacheSourceError::Identity);
                }
                let row = loan
                    .blocks()
                    .find(|row| row.id() == &store.id)
                    .ok_or(CacheSourceError::Identity)?;
                let arrays = row.device().ok_or(CacheSourceError::PromotionRequired)?;
                store.pin = Some(loan.pin_prepared_block(&store.id, self._funding.clone())?);
                store.pair.fill_for_inspection(arrays, true)
            })
            .map_err(|cause| proof.error(cause.into()))
    }
    #[allow(clippy::too_many_arguments)]
    fn rebalance(
        &mut self,
        proof: &TransferSource<'_>,
        ordinal: usize,
        required: Option<&CacheBlockId>,
        additional: u64,
        tail: Option<(usize, u64)>,
        allow_recent: bool,
        stream: &Stream,
    ) -> Result<(), Exception> {
        loop {
            for store in &mut self.stores {
                if let Some(mover) = &mut store.mover {
                    mover.reclaim_replaced_device();
                }
                if let Some(backed) = &mut store.initial_disk {
                    backed.reclaim_replaced_device();
                }
                if let Some(retirement) = &mut store.write_retirement {
                    retirement.reclaim();
                }
            }
            for load in &mut self.loads {
                load.promotion.reclaim_replaced_device();
            }
            let Some(id) = proof.source.manager().prepared_host_victim(
                proof,
                required,
                additional,
                tail,
                allow_recent,
            )?
            else {
                return Ok(());
            };
            let sources = proof.work.program.inner.sources.sources();
            let index = self
                .stores
                .iter()
                .position(|store| {
                    store.id == id
                        && store.first_ordinal <= ordinal
                        && sources[store.source]
                            .manager()
                            .same_catalog(proof.source.manager())
                })
                .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
            let victim = TransferSource {
                source: &sources[self.stores[index].source],
                ..*proof
            };
            if let Some(load) = self.stores[index].current_load {
                if self.loads[load].promotion.has_backing() {
                    let pins = self
                        .pin_count(proof.work, proof.source.manager(), &id)
                        .map_err(|cause| proof.error(cause.into()))?;
                    let permission = Return {
                        source: &victim,
                        id: &id,
                        owned_pins: pins,
                    };
                    self.loads[load].promotion.return_to_disk(&permission)?;
                    continue;
                }
            }
            if self.stores[index].current_load.is_none()
                && self.stores[index].initial_disk.is_some()
            {
                self.ensure_initial(index, &victim)?;
                let pins = self
                    .pin_count(proof.work, proof.source.manager(), &id)
                    .map_err(|cause| proof.error(cause.into()))?;
                let store = &mut self.stores[index];
                let arrays = store
                    .pair
                    .values()
                    .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
                let permission = Eviction {
                    source: &victim,
                    id: &store.id,
                    arrays,
                    owned_pins: pins,
                };
                store
                    .initial_disk
                    .as_mut()
                    .expect("retained backed source")
                    .run_ordinary(&permission, &proof.work.program.inner.context)?;
                store.pair.arrays = [None, None];
                continue;
            }
            let capacity = if let Some(load) = self.stores[index].current_load {
                self.loads[load].promotion.host_capacity()
            } else {
                self.stores[index]
                    .mover
                    .as_ref()
                    .and_then(PreparedOrdinaryCacheHostDemotion::host_capacity)
            }
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
            if self.requires_disk {
                self.make_disk_room(&victim, ordinal, required, capacity, false)?;
            }
            if self.stores[index].current_load.is_none() {
                self.ensure_initial(index, &victim)?;
            }
            let pins = self
                .pin_count(proof.work, proof.source.manager(), &id)
                .map_err(|cause| proof.error(cause.into()))?;
            if let Some(load) = self.stores[index].current_load {
                let permission = Return {
                    source: &victim,
                    id: &id,
                    owned_pins: pins,
                };
                self.loads[load].promotion.return_to_host(&permission)?;
            } else {
                let store = &mut self.stores[index];
                let arrays = store
                    .pair
                    .values()
                    .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
                let permission = Eviction {
                    source: &victim,
                    id: &store.id,
                    arrays,
                    owned_pins: pins,
                };
                store
                    .mover
                    .as_mut()
                    .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
                    .run(&permission, stream)?;
                store.pair.arrays = [None, None];
            }
            if self.requires_disk {
                self.make_disk_room(&victim, ordinal, required, 0, true)?;
            }
        }
    }
    fn acquire(
        &mut self,
        proof: &TransferSource<'_>,
        ordinal: usize,
        id: &CacheBlockId,
        pass: usize,
        stream: &Stream,
    ) -> Result<(usize, PinnedCacheBlockLease), Exception> {
        let sources = proof.work.program.inner.sources.sources();
        let index = self
            .loads
            .iter()
            .position(|load| {
                !load.consumed
                    && load.ordinal == ordinal
                    && load.pass == pass
                    && self.stores[load.store].id == *id
                    && sources[self.stores[load.store].source]
                        .manager()
                        .same_catalog(proof.source.manager())
            })
            .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
        let store_index = self.loads[index].store;
        self.loads[index].consumed = true;
        let (device, disk, bytes) = proof
            .installed
            .with_source(selection(id), |loan| {
                let row = loan
                    .blocks()
                    .find(|row| row.id() == id)
                    .ok_or(CacheSourceError::Identity)?;
                let disk = row.phase() == eredu_runtime::CacheStoragePhase::DiskReady;
                if row.device().is_none() && row.host().is_none() && !disk {
                    return Err(CacheSourceError::PendingStorage);
                }
                Ok((row.device().is_some(), disk, row.logical_bytes()))
            })
            .map_err(|cause| proof.error(cause.into()))?;
        let read = if device {
            match self.stores[store_index].current_load {
                Some(prior) => {
                    if self.loads[prior].promotion.was_demoted() {
                        return Err(proof.error(CacheSourceError::Identity.into()));
                    }
                    Read::Promoted(prior)
                }
                None => {
                    self.ensure_initial(store_index, proof)?;
                    Read::Initial(store_index)
                }
            }
        } else {
            if disk {
                self.read_disk_source(proof, ordinal, index)?;
            }
            self.rebalance(proof, ordinal, Some(id), bytes, None, true, stream)?;
            if disk {
                self.promote_disk_read(proof, id, index, stream)?;
            } else {
                let store = &self.stores[store_index];
                let buffers = store
                    .mover
                    .as_ref()
                    .and_then(PreparedOrdinaryCacheHostDemotion::buffers)
                    .or_else(|| {
                        store
                            .initial_host
                            .as_ref()
                            .map(|buffers| [buffers[0].as_ref(), buffers[1].as_ref()])
                    })
                    .ok_or_else(|| proof.error(CacheSourceError::PromotionRequired.into()))?;
                let permission = Promotion { source: proof, id };
                self.loads[index]
                    .promotion
                    .run(&permission, buffers, stream)?;
            }
            self.stores[store_index].current_load = Some(index);
            Read::Promoted(index)
        };
        self.loads[index].read = Some(read);
        let lease = match read {
            Read::Initial(index) => self.stores[index]
                .pin
                .as_ref()
                .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
                .acquire(),
            Read::Promoted(index) => self.loads[index].promotion.acquire(),
        }
        .map_err(|cause| proof.error(cause.into()))?;
        Ok((index, lease))
    }
    fn values(&self, load: usize) -> Option<[&Array; 2]> {
        match self.loads.get(load)?.read? {
            Read::Initial(index) => self.stores.get(index)?.pair.values(),
            Read::Promoted(index) => self.loads.get(index)?.promotion.values(),
        }
    }
}
pub(super) fn selection(id: &CacheBlockId) -> eredu_runtime::CacheBlockSelection {
    eredu_runtime::CacheBlockSelection::new(id.global_layer, id.representation, id.start, id.end, 0)
}
impl<'a> Checkout<'a> {
    fn take(
        work: &'a OrdinaryPagedWork,
        host: &HostPreparationAuthority,
    ) -> Result<Option<Self>, Exception> {
        let value = work
            .program
            .inner
            .host
            .try_borrow_mut()
            .map_err(|_| failure(CacheSourceError::Busy, host))?
            .take();
        Ok(value.map(|value| Self {
            work,
            value: Some(value),
        }))
    }
}
impl OrdinaryPagedHostScan<'_> {
    pub(crate) fn with_acquired<R>(
        &self,
        id: &CacheBlockId,
        pass: usize,
        stream: &Stream,
        run: impl FnOnce([&Array; 2], PinnedCacheBlockLease) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let proof = TransferSource {
            work: self.work,
            source: self.source,
            installed: self.installed,
            host: self.host,
        };
        proof.validate_manager(
            self.installed.manager(),
            self.installed.initial_generation(),
        )?;
        let mut checkout = Checkout::take(self.work, self.host)?
            .ok_or_else(|| failure(CacheSourceError::Identity, self.host))?;
        let host = checkout.value.as_mut().expect("checked-out itinerary");
        let (index, lease) = host.acquire(&proof, self.ordinal, id, pass, stream)?;
        let arrays = host
            .values(index)
            .ok_or_else(|| failure(CacheSourceError::Identity, self.host))?;
        run(arrays, lease).map_err(|cause| failure(cause, self.host))
    }
}
impl OrdinaryPagedAppend {
    pub(crate) fn rebalance_host(
        &mut self,
        additional: u64,
        replacement_tail: Option<u64>,
        required: Option<&CacheBlockId>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let Some(mut checkout) = Checkout::take(&self.work, &self.host)? else {
            return Ok(());
        };
        let source = &self.work.program.inner.sources.sources()[self.source];
        let proof = TransferSource {
            work: &self.work,
            source,
            installed: &self.installed,
            host: &self.host,
        };
        proof.validate_manager(
            self.installed.manager(),
            self.installed.initial_generation(),
        )?;
        checkout
            .value
            .as_mut()
            .expect("checked-out itinerary")
            .rebalance(
                &proof,
                self.row().ordinal,
                required,
                additional,
                replacement_tail.map(|bytes| (self.layer(), bytes)),
                false,
                stream,
            )
    }
    pub(crate) fn validate_retained_storage(
        &self,
        id: &CacheBlockId,
        phase: eredu_runtime::CacheStoragePhase,
        file: Option<&crate::backend::runtime::cache::residency::CacheFileSource>,
    ) -> Result<(), Exception> {
        if !matches!(
            phase,
            eredu_runtime::CacheStoragePhase::HostUnbacked
                | eredu_runtime::CacheStoragePhase::HostBacked
                | eredu_runtime::CacheStoragePhase::DiskReady
        ) {
            return Err(self.error(CacheSourceError::PromotionRequired));
        }
        let program = self
            .work
            .program
            .inner
            .host
            .try_borrow()
            .map_err(|_| self.error(CacheSourceError::Busy))?;
        let valid = program.as_ref().is_some_and(|program| {
            program.stores.iter().any(|store| {
                store.source == self.source
                    && &store.id == id
                    && store.first_ordinal <= self.row().ordinal
                    && (if phase == eredu_runtime::CacheStoragePhase::DiskReady {
                        file.is_some_and(|actual| {
                            self.work.program.inner.sources.sources()[self.source]
                                .retained_file(id)
                                .is_some_and(|retained| retained.same_source(actual))
                                || store
                                    .write
                                    .as_ref()
                                    .and_then(DiskWriteOperation::committed_file)
                                    .is_some_and(|retained| actual.same_live_source(retained))
                        })
                    } else {
                        store.initial_host.is_some()
                            || store
                                .mover
                                .as_ref()
                                .is_some_and(|mover| mover.buffers().is_some())
                    })
            })
        });
        if !valid {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
}

impl HostProgram {
    /// Finite caller envelope from the retained one-use store/load itinerary.
    /// A prior load may be evicted by this step, so a step includes each actual
    /// earlier destination's return path. Whole-program quotation counts each
    /// destination once. Numerical copies and Event waits remain in the trace.
    pub(in super::super) fn caller_controls(
        &self,
        rows: &[Option<Row>],
        ordinals: Range<usize>,
    ) -> Option<OrdinaryCallControls> {
        let mut controls = self.disk_call_controls(ordinals.clone())?;
        let mut stores = 0usize;
        let mut loads = 0usize;
        let mut visits = 0usize;
        for store in &self.stores {
            if store.first_ordinal >= ordinals.end {
                continue;
            }
            stores = stores.checked_add(1)?;
            if let Some(mover) = &store.mover {
                controls =
                    controls.append(PreparedOrdinaryCacheHostDemotion::source_controls::<
                        Eviction<'_, '_>,
                    >()?)?;
                controls.observed = controls.observed.append(mover.constructor_controls())?;
            }
        }
        for load in &self.loads {
            if load.ordinal >= ordinals.end {
                continue;
            }
            loads = loads.checked_add(1)?;
            controls = controls.append(
                load.promotion
                    .source_controls::<Promotion<'_, '_>, Return<'_, '_>>()?,
            )?;
            if ordinals.contains(&load.ordinal) {
                visits = visits.checked_add(1)?;
            }
        }
        let mut entries = visits;
        for row in rows
            .iter()
            .flatten()
            .filter(|row| ordinals.contains(&row.ordinal))
        {
            // publish_tail calls rebalancing once for each exact append step;
            // publish_block calls it once for each actual sealed publication.
            entries = entries
                .checked_add(row.append.steps().count())?
                .checked_add(row.publications.len())?;
        }
        // Every initial source can be stored once and every actual load can
        // return once. Each loop also performs its final no-victim policy check.
        let mutations = stores.checked_add(loads)?;
        let iterations = mutations.checked_add(entries)?;
        let runtime = runtime_control_bytes()?;
        let policy =
            CacheResidencyManager::prepared_host_policy_control_bytes::<TransferSource<'_>>()?;
        let bind = super::super::super::scan_program::ScanSourcePair::inspection_control_bytes()?
            .checked_add(PinnedCacheBlock::fixed_controls()?)?
            .checked_add(InstalledManagerCatalog::source_control_bytes::<()>(
                size_of::<(&mut Store, &Option<HostMetadataFunding>)>(),
            )?)?;
        let inspect =
            InstalledManagerCatalog::source_control_bytes::<(bool, bool, u64)>(size_of::<
                &CacheBlockId,
            >())?;
        let bytes = runtime
            .checked_mul(iterations.checked_add(visits)?.checked_add(1)?)?
            .checked_add(policy.checked_mul(iterations)?)?
            .checked_add(bind.checked_mul(stores)?)?
            .checked_add(inspect.checked_mul(visits)?)?
            .checked_add(PinnedCacheBlock::fixed_controls()?.checked_mul(visits)?)?;
        controls.metadata(bytes)
    }
}
fn runtime_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<TransferSource<'_>>(),
        size_of::<Eviction<'_, '_>>(),
        size_of::<Promotion<'_, '_>>(),
        size_of::<Return<'_, '_>>(),
        size_of::<Read>(),
        size_of::<Option<Read>>(),
        size_of::<Checkout<'_>>(),
        size_of::<(
            &mut HostProgram,
            &TransferSource<'_>,
            usize,
            Option<&CacheBlockId>,
            u64,
            Option<(usize, u64)>,
            bool,
            &Stream,
        )>(),
        size_of::<(
            &mut HostProgram,
            &TransferSource<'_>,
            usize,
            &CacheBlockId,
            usize,
            &Stream,
        )>(),
        size_of::<(
            &HostProgram,
            &OrdinaryPagedWork,
            &CacheResidencyManager,
            &CacheBlockId,
        )>(),
        size_of::<(
            &TransferSource<'_>,
            &CacheBlockId,
            usize,
            &eredu_runtime::CacheBlockLifecycle,
        )>(),
        size_of::<(&mut HostProgram, usize, &TransferSource<'_>)>(),
        size_of::<(
            &mut OrdinaryPagedAppend,
            u64,
            Option<u64>,
            Option<&CacheBlockId>,
            &Stream,
        )>(),
        size_of::<(
            &OrdinaryPagedAppend,
            &CacheBlockId,
            eredu_runtime::CacheStoragePhase,
            Option<&crate::backend::runtime::cache::residency::CacheFileSource>,
        )>(),
        size_of::<Result<(usize, PinnedCacheBlockLease), Exception>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<usize, CacheSourceError>>(),
        size_of::<Option<[&Array; 2]>>(),
        size_of::<Option<[&ImmutableHostTransferBuffer; 2]>>(),
        size_of::<std::slice::Iter<'_, Store>>(),
        size_of::<std::slice::IterMut<'_, Store>>(),
        size_of::<std::slice::Iter<'_, Load>>(),
        size_of::<std::slice::IterMut<'_, Load>>(),
        size_of::<std::cell::Ref<'_, State>>(),
        size_of::<std::cell::Ref<'_, Option<HostProgram>>>(),
        size_of::<std::cell::RefMut<'_, Option<HostProgram>>>(),
        size_of::<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>>(),
        size_of::<Result<Option<crate::backend::nn::shared::OrdinaryExecutionOwner>, Exception>>(),
        size_of::<(usize, usize, u64, bool)>(),
        size_of::<CacheBlockId>(),
        size_of::<Failure>(),
        Exception::retained_source_control_bytes::<Failure>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
