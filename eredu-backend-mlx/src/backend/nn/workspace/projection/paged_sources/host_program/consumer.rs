//! Existing policy and transfer workers consume this exact finite itinerary.
use super::super::scan_claim::{
    OriginalPagedDiskWriteSource, OriginalPagedHostEviction, OriginalPagedHostReturn,
    OriginalPagedScanSource,
};
use super::*;
use crate::backend::{
    error::Error as NativeError,
    runtime::cache::residency::{
        CacheResidencyManager, PinnedCacheBlockLease, StoredCacheHostSource,
    },
    submission_recovery::prefill::TransientRootsProjection,
};
use safemlx::{Array, Stream, error::Exception};
#[derive(Clone, Copy)]
pub(super) enum HostRead {
    Initial(usize),
    Promotion(usize),
}

/// The native owner is outside every RefCell/manager loan while work executes.
/// Drop restores the complete attempted prefix even when a callback unwinds.
pub(in super::super) struct HostCheckout<'a> {
    source: &'a PagedSourceBank,
    pub(in super::super) value: Option<PreparedPagedHostProgram>,
}
impl<'a> HostCheckout<'a> {
    pub(in super::super) fn take(
        source: &'a PagedSourceBank,
        proof: &OriginalPagedScanSource<'_>,
    ) -> Result<Self, Exception> {
        let mut slot = source
            .host
            .try_borrow_mut()
            .map_err(|_| proof.error(CacheSourceError::Busy))?;
        if slot.as_ref().is_some_and(|host| !host.constructed) {
            return Err(proof.error(CacheSourceError::Identity));
        }
        let value = slot.take();
        Ok(Self { source, value })
    }
}
impl Drop for HostCheckout<'_> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            let previous = self.source.host.borrow_mut().replace(value);
            debug_assert!(previous.is_none());
            drop(previous);
        }
    }
}
pub(super) fn failure(proof: &OriginalPagedScanSource<'_>, cause: CacheSourceFailure) -> Exception {
    proof.error(NativeError::Neural(proof.context().metadata_source(cause)))
}
pub(super) fn for_source<'a>(
    proof: &'a OriginalPagedScanSource<'_>,
    source: &'a ProjectedPagedSource,
) -> OriginalPagedScanSource<'a> {
    OriginalPagedScanSource {
        source,
        installed: proof.installed,
        observer: proof.observer.clone(),
        context: proof.context,
        custody: proof.custody.clone(),
    }
}
impl PreparedPagedHostProgram {
    pub(in super::super) fn contains(
        &self,
        source: &ProjectedPagedSource,
        id: &CacheBlockId,
        bank: &PagedSourceBank,
    ) -> bool {
        self.stores.iter().any(|store| {
            &store.id == id
                && bank.sources[store.source]
                    .manager()
                    .same_catalog(source.manager())
        })
    }
    /// The canonical file must be this bank's exact initial source or a
    /// successfully committed own-writer result. This grants no read/promotion.
    pub(in super::super) fn retains_file(
        &self,
        source: &ProjectedPagedSource,
        id: &CacheBlockId,
        file: &crate::backend::runtime::cache::residency::CacheFileSource,
        bank: &PagedSourceBank,
    ) -> bool {
        self.stores.iter().any(|store| {
            &store.id == id
                && bank.sources[store.source]
                    .manager()
                    .same_catalog(source.manager())
                && (bank.sources[store.source]
                    .retained_file(id)
                    .is_some_and(|held| held.same_source(file))
                    || store
                        .write
                        .as_ref()
                        .and_then(|write| write.committed_file())
                        .is_some_and(|held| file.same_live_source(held)))
        })
    }
    pub(in super::super) fn retained_file_control_bytes() -> usize {
        size_of::<(
            &Self,
            &ProjectedPagedSource,
            &CacheBlockId,
            &crate::backend::runtime::cache::residency::CacheFileSource,
            &PagedSourceBank,
            std::slice::Iter<'static, StoreSlot>,
            Option<&'static crate::backend::runtime::cache::residency::CacheFileSource>,
            bool,
        )>() + crate::backend::runtime::cache::residency::DiskWriteOperation::committed_file_control_bytes()
    }
    fn pin_count(
        &self,
        bank: &PagedSourceBank,
        id: &CacheBlockId,
        manager: &CacheResidencyManager,
    ) -> Result<usize, CacheSourceError> {
        let mut count = 0usize;
        for source in &bank.sources {
            if source.manager().same_catalog(manager) {
                count = count
                    .checked_add(source.owned_source_pin_count(id))
                    .ok_or(CacheSourceError::Overflow)?;
            }
        }
        for store in &self.stores {
            if &store.id == id && bank.sources[store.source].manager().same_catalog(manager) {
                count = count
                    .checked_add(usize::from(store.pin.is_some()))
                    .and_then(|n| {
                        n.checked_add(
                            store
                                .write
                                .as_ref()
                                .map_or(0, |write| write.completed_source_pin_count()),
                        )
                    })
                    .ok_or(CacheSourceError::Overflow)?;
            }
        }
        for load in &self.loads {
            let store = &self.stores[load.store];
            if &store.id == id && bank.sources[store.source].manager().same_catalog(manager) {
                count = count
                    .checked_add(
                        load.read_operation
                            .as_ref()
                            .map_or(0, |read| read.completed_source_pin_count()),
                    )
                    .and_then(|n| {
                        n.checked_add(usize::from(
                            load.promotion
                                .as_ref()
                                .is_some_and(|value| value.owns_pin()),
                        ))
                    })
                    .ok_or(CacheSourceError::Overflow)?;
            }
        }
        // Other already checked-in programs retain their own actual row pins.
        // The current Host scan/append borrows directly from this itinerary.
        let catalogs = bank
            .catalogs
            .try_borrow()
            .map_err(|_| CacheSourceError::Busy)?;
        if let Some(catalogs) = catalogs.as_ref() {
            for program in catalogs.programs.iter().flatten() {
                if !bank.sources[program.source].manager().same_catalog(manager) {
                    continue;
                }
                let rows = program.scan.iter().flat_map(|scan| scan.rows.iter()).chain(
                    program
                        .visible
                        .iter()
                        .flat_map(|visible| visible.rows.iter()),
                );
                {
                    for row in rows {
                        if &row.id == id {
                            count = count
                                .checked_add(usize::from(row.pin.is_some()))
                                .ok_or(CacheSourceError::Overflow)?;
                        }
                    }
                }
            }
        }
        Ok(count)
    }
    /// Prepares the shared durable writer for the actual Host row while retaining
    /// every request-bank source owner. Its caller keeps the returned task in
    /// the finite Disk itinerary before submitting it to the installed worker.
    pub(in super::super) fn prepare_disk_write(
        &mut self,
        bank: &PagedSourceBank,
        proof: &OriginalPagedScanSource<'_>,
        ordinal: usize,
        id: &CacheBlockId,
    ) -> Result<crate::backend::runtime::cache::residency::PreparedDiskWrite, Exception> {
        type Write = crate::backend::runtime::cache::residency::PreparedDiskWrite;
        let context = proof.context();
        let controls = [
            size_of::<(
                &Self,
                &PagedSourceBank,
                &OriginalPagedScanSource<'_>,
                usize,
                &CacheBlockId,
            )>(),
            size_of::<OriginalPagedDiskWriteSource<'_, '_>>(),
            size_of::<(
                &OriginalPagedDiskWriteSource<'_, '_>,
                &CacheBlockSourceLoan<'_>,
            )>(),
            size_of::<(&CacheBlockSourceLoan<'_>, &CacheBlockId)>(),
            size_of::<Result<(usize, bool), CacheSourceError>>(),
            size_of::<(usize, bool)>(),
            size_of::<std::slice::Iter<'_, StoreSlot>>(),
            size_of::<std::slice::IterMut<'_, StoreSlot>>(),
            size_of::<std::slice::IterMut<'_, LoadSlot>>(),
            size_of::<std::slice::Iter<'_, LoadSlot>>(),
            size_of::<std::slice::Iter<'_, ProjectedPagedSource>>(),
            size_of::<std::slice::Iter<'_, super::super::scan_program::ScanSourceRow>>(),
            size_of::<std::cell::Ref<'_, Option<PreparedPagedCatalogs>>>(),
            size_of::<Result<Write, Exception>>(),
            size_of::<Result<usize, CacheSourceError>>(),
            size_of::<Result<bool, eredu_runtime::CacheLifecycleError>>(),
            size_of::<Result<usize, eredu_runtime::CacheLifecycleError>>(),
            CacheResidencyManager::source_loan_control_bytes::<Write>(size_of::<(
                &OriginalPagedDiskWriteSource<'_, '_>,
                &OriginalPagedScanSource<'_>,
            )>())
            .ok_or_else(|| proof.error(CacheSourceError::Overflow))?,
        ];
        context
            .charge_metadata(
                controls
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                    .ok_or_else(|| proof.error(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| proof.error(NativeError::Neural(cause.into())))?;
        let index = self
            .stores
            .iter()
            .position(|store| {
                store.id == *id
                    && store.first_ordinal <= ordinal
                    && bank.sources[store.source]
                        .manager()
                        .same_catalog(proof.manager())
            })
            .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
        let owned_pins = self
            .pin_count(bank, id, proof.manager())
            .map_err(|cause| proof.error(cause))?;
        let source = OriginalPagedDiskWriteSource {
            source: proof,
            id,
            owned_pins,
        };
        let destination = self.stores[index]
            .disk
            .take()
            .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
        proof
            .manager()
            .with_original_host_source(proof, id, |loan| {
                destination
                    .bind(loan, &source)
                    .map_err(|cause| failure(proof, cause))
            })
    }
    fn ensure_initial(
        &mut self,
        index: usize,
        proof: &OriginalPagedScanSource<'_>,
    ) -> Result<(), Exception> {
        let store = &mut self.stores[index];
        if store.pair.values().is_some() {
            if store.pin.is_none() {
                return Err(proof.error(CacheSourceError::Identity));
            }
            return Ok(());
        }
        proof
            .manager()
            .with_original_host_source(proof, &store.id, |loan| {
                if store.pin.is_some() {
                    return Err(proof.error(CacheSourceError::Identity));
                }
                let source = loan
                    .blocks()
                    .find(|row| row.id() == &store.id)
                    .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
                let arrays = source
                    .device()
                    .ok_or_else(|| proof.error(CacheSourceError::PromotionRequired))?;
                proof.validate_arrays(arrays, store.id.end - store.id.start)?;
                for array in arrays {
                    proof.observer().validate_completed_array(array)?;
                }
                // The outer checkout keeps this pin on a later partial fill failure.
                store.pin = Some(
                    loan.pin_prepared_block(&store.id, proof.funding())
                        .map_err(|cause| proof.error(cause))?,
                );
                store.pair.fill(arrays, proof.observer())
            })
    }
    pub(in super::super) fn rebalance(
        &mut self,
        bank: &PagedSourceBank,
        proof: &OriginalPagedScanSource<'_>,
        ordinal: usize,
        required: Option<&CacheBlockId>,
        additional: u64,
        tail: Option<(usize, u64)>,
        allow_recent: bool,
        roots: &TransientRootsProjection,
        stream: &Stream,
    ) -> Result<(), Exception> {
        loop {
            // A previous completed demotion may have retained an escaped numerical
            // alias until this boundary. Drain only these exact native-dead owners.
            for store in &mut self.stores {
                if let Some(mover) = &mut store.mover {
                    mover.reclaim_replaced_device();
                }
            }
            for load in &mut self.loads {
                if let Some(promotion) = &mut load.promotion {
                    promotion.reclaim_replaced_device();
                }
            }
            let Some(id) = proof.manager().original_host_victim(
                proof,
                required,
                additional,
                tail,
                allow_recent,
            )?
            else {
                return Ok(());
            };
            let index = self
                .stores
                .iter()
                .position(|store| {
                    store.id == id
                        && store.first_ordinal <= ordinal
                        && bank.sources[store.source]
                            .manager()
                            .same_catalog(proof.manager())
                })
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
            let source = &bank.sources[self.stores[index].source];
            let victim = for_source(proof, source);
            if let Some(load) = self.stores[index].current_load {
                if self.loads[load]
                    .promotion
                    .as_ref()
                    .is_some_and(|value| value.has_backing())
                {
                    let pins = self
                        .pin_count(bank, &id, proof.manager())
                        .map_err(|cause| proof.error(cause))?;
                    let permission = OriginalPagedHostReturn {
                        source: &victim,
                        id: &id,
                        owned_pins: pins,
                        roots,
                    };
                    self.loads[load]
                        .promotion
                        .as_mut()
                        .expect("selected promotion")
                        .return_to_disk(&permission)?;
                    continue;
                }
            }
            if self.stores[index].current_load.is_none()
                && self.stores[index].initial_disk.is_some()
            {
                self.ensure_initial(index, &victim)?;
                let pins = self
                    .pin_count(bank, &id, proof.manager())
                    .map_err(|cause| proof.error(cause))?;
                let permission = OriginalPagedHostReturn {
                    source: &victim,
                    id: &id,
                    owned_pins: pins,
                    roots,
                };
                let store = &mut self.stores[index];
                let arrays = store
                    .pair
                    .values()
                    .ok_or_else(|| victim.error(CacheSourceError::Identity))?;
                store
                    .initial_disk
                    .as_mut()
                    .expect("initial backed source")
                    .run(&permission, arrays)?;
                continue;
            }
            let capacity = if let Some(load) = self.stores[index].current_load {
                self.loads[load]
                    .promotion
                    .as_ref()
                    .and_then(|promotion| promotion.host_capacity())
                    .ok_or_else(|| victim.error(CacheSourceError::Identity))?
            } else {
                self.stores[index]
                    .mover
                    .as_ref()
                    .map(|mover| mover.host_capacity())
                    .ok_or_else(|| victim.error(CacheSourceError::Identity))?
            };
            self.make_disk_room(bank, &victim, ordinal, required, capacity, false)?;
            if self.stores[index].current_load.is_none() {
                self.ensure_initial(index, &victim)?;
            }
            let pins = self
                .pin_count(bank, &id, proof.manager())
                .map_err(|cause| proof.error(cause))?;
            if let Some(load) = self.stores[index].current_load {
                let promotion = self.loads[load]
                    .promotion
                    .as_mut()
                    .ok_or_else(|| victim.error(CacheSourceError::Identity))?;
                let permission = OriginalPagedHostReturn {
                    source: &victim,
                    id: &id,
                    owned_pins: pins,
                    roots,
                };
                promotion.return_to_host(&permission)?;
            } else {
                let store = &mut self.stores[index];
                let arrays = store
                    .pair
                    .values()
                    .ok_or_else(|| victim.error(CacheSourceError::Identity))?;
                let eviction = OriginalPagedHostEviction {
                    source: &victim,
                    id: &store.id,
                    arrays,
                    owned_pins: pins,
                    roots,
                };
                store
                    .mover
                    .as_mut()
                    .ok_or_else(|| victim.error(CacheSourceError::Identity))?
                    .run(&eviction, stream)?;
                // The completed Host source replaces this initial Device read.
                // Any escaped native alias keeps the backing-attached reservation.
                store.pair.arrays = [None, None];
            }
            self.make_disk_room(bank, &victim, ordinal, required, 0, true)?;
        }
    }
    pub(in super::super) fn acquire(
        &mut self,
        bank: &PagedSourceBank,
        proof: &OriginalPagedScanSource<'_>,
        ordinal: usize,
        pass: usize,
        id: &CacheBlockId,
        roots: &TransientRootsProjection,
        stream: &Stream,
    ) -> Result<(usize, PinnedCacheBlockLease), Exception> {
        let load_index = self
            .loads
            .iter()
            .position(|load| {
                !load.consumed
                    && load.ordinal == ordinal
                    && load.pass == pass
                    && bank.sources[self.stores[load.store].source]
                        .manager()
                        .same_catalog(proof.manager())
                    && self.stores[load.store].id.global_layer == proof.layer()
            })
            .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
        let store_index = self.loads[load_index].store;
        if &self.stores[store_index].id != id {
            return Err(proof.error(CacheSourceError::Identity));
        }
        self.loads[load_index].consumed = true;
        let (device, disk, bytes) =
            proof
                .manager()
                .with_original_host_source(proof, id, |loan| {
                    let row = loan
                        .blocks()
                        .find(|row| row.id() == id)
                        .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
                    let disk = row.phase() == eredu_runtime::CacheStoragePhase::DiskReady;
                    if row.device().is_none() && row.host().is_none() && !disk {
                        return Err(proof.error(CacheSourceError::PendingStorage));
                    }
                    Ok((row.device().is_some(), disk, row.logical_bytes()))
                })?;
        let read = if device {
            match self.stores[store_index].current_load {
                Some(index) => {
                    if self.loads[index]
                        .promotion
                        .as_ref()
                        .is_none_or(|value| value.was_demoted())
                    {
                        return Err(proof.error(CacheSourceError::Identity));
                    }
                    HostRead::Promotion(index)
                }
                None => {
                    self.ensure_initial(store_index, proof)?;
                    HostRead::Initial(store_index)
                }
            }
        } else {
            if disk {
                self.read_disk_source(bank, proof, ordinal, load_index)?;
            }
            self.rebalance(
                bank,
                proof,
                ordinal,
                Some(id),
                bytes,
                None,
                true,
                roots,
                stream,
            )?;
            let slots = self.loads[load_index]
                .slots
                .take()
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
            let promotion = if let Some(read) = self.loads[load_index].read_operation.as_mut() {
                read.prepare_host_promotion_from_slots(proof, slots)
                    .map_err(|cause| failure(proof, cause))?
            } else {
                let stored = self.stores[store_index]
                    .mover
                    .as_ref()
                    .map(|mover| {
                        mover
                            .stored_source()
                            .ok_or_else(|| proof.error(CacheSourceError::Identity))
                    })
                    .transpose()?;
                proof
                    .manager()
                    .with_original_host_source(proof, id, |loan| {
                        slots
                            .bind(loan, proof, stored)
                            .map_err(|cause| failure(proof, cause))
                    })?
            };
            self.loads[load_index].promotion = Some(promotion);
            self.loads[load_index]
                .promotion
                .as_mut()
                .expect("retained before native work")
                .run(proof, roots, stream)?;
            self.stores[store_index].current_load = Some(load_index);
            HostRead::Promotion(load_index)
        };
        self.loads[load_index].read = Some(read);
        let lease = match read {
            HostRead::Initial(index) => self.stores[index]
                .pin
                .as_ref()
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?
                .acquire(),
            HostRead::Promotion(index) => self.loads[index]
                .promotion
                .as_ref()
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?
                .acquire(),
        }
        .map_err(|cause| proof.error(cause))?;
        Ok((load_index, lease))
    }
    pub(in super::super) fn values(&self, load: usize) -> Option<[&Array; 2]> {
        match self.loads.get(load)?.read? {
            HostRead::Initial(index) => self.stores.get(index)?.pair.values(),
            HostRead::Promotion(index) => self.loads.get(index)?.promotion.as_ref()?.values(),
        }
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    let frames = [
        size_of::<HostCheckout<'_>>(),
        size_of::<HostRead>(),
        size_of::<Option<HostRead>>(),
        size_of::<[Option<Array>; 2]>(),
        size_of::<OriginalPagedScanSource<'_>>(),
        size_of::<OriginalPagedHostEviction<'_, '_>>(),
        size_of::<(
            &mut PreparedPagedHostProgram,
            &PagedSourceBank,
            &OriginalPagedScanSource<'_>,
            usize,
            Option<&CacheBlockId>,
            u64,
            Option<(usize, u64)>,
            bool,
            &TransientRootsProjection,
            &Stream,
        )>(),
        size_of::<(
            &mut PreparedPagedHostProgram,
            &PagedSourceBank,
            &OriginalPagedScanSource<'_>,
            usize,
            usize,
            &CacheBlockId,
            &TransientRootsProjection,
            &Stream,
        )>(),
        size_of::<std::cell::RefMut<'_, Option<PreparedPagedHostProgram>>>(),
        size_of::<std::cell::Ref<'_, Option<PreparedPagedCatalogs>>>(),
        size_of::<Option<StoredCacheHostSource>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<(usize, PinnedCacheBlockLease), Exception>>(),
        size_of::<(usize, usize, u64, bool, bool)>(),
        size_of::<std::slice::Iter<'_, StoreSlot>>(),
        size_of::<std::slice::Iter<'_, LoadSlot>>(),
        size_of::<std::slice::Iter<'_, ProjectedPagedSource>>(),
        CacheResidencyManager::original_host_policy_control_bytes()?,
        CacheResidencyManager::source_loan_control_bytes::<(bool, bool, u64)>(size_of::<(
            &OriginalPagedScanSource<'_>,
            &CacheBlockId,
        )>())?,
        CacheResidencyManager::source_loan_control_bytes::<()>(size_of::<(
            &mut StoreSlot,
            &OriginalPagedScanSource<'_>,
        )>())?,
        CacheResidencyManager::source_loan_control_bytes::<PreparedCacheHostPromotion>(
            size_of::<(
                PreparedCacheHostPromotionSlots,
                Option<StoredCacheHostSource>,
                &OriginalPagedScanSource<'_>,
            )>(),
        )?,
        OriginalPagedHostEviction::control_bytes()?,
        disk::consumer_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
