//! Concrete destinations prepared under the existing selected Work control hold.
use super::*;
use std::{alloc::Layout, mem::size_of};

pub(super) fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<(), Error> {
    values
        .try_reserve_exact(count)
        .map_err(|e| Error::PrefillControl(WorkingMemoryError::ControlStorageReserve(e)))?;
    if size_of::<T>() != 0 && values.capacity() != count {
        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
    }
    Ok(())
}
impl FundedWork {
    pub(in crate::composition::mlx::session::model_session) fn original_metadata_custody(
        &self,
    ) -> Option<eredu_runtime::working_memory::OriginalTextMetadataCustody> {
        self.native_storage.as_ref()?;
        Some(self._controls.as_ref()?.metadata_custody())
    }
    pub(super) fn prepare_collectors(&self) -> Result<(), Error> {
        if self.native_storage.is_none() && self.ordinary.borrow().is_none() {
            return Err(Error::OriginalSourceContract {
                stage: "prepared work publication collectors",
                cause: WorkingMemoryError::UnknownBound,
            });
        }
        if self.collectors_prepared.replace(true) {
            return Err(Error::PrefillControl(WorkingMemoryError::AlreadyStarted));
        }
        if !qualified() {
            return Err(Error::OriginalSourceContract {
                stage: "publication borrowed-owner allocation layout",
                cause: WorkingMemoryError::UnknownBound,
            });
        }
        let rows = if let Some(bank) = &self.native_storage {
            bank.try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?
                .collector_rows()
        } else {
            self.ordinary.borrow().as_ref().expect("ordinary plan").0
        };
        let custody = self.original_metadata_custody();
        // Prepare final destinations before any original native worker. A failed
        // prefix remains inside this Work until its original scope is cancelled.
        reserve(&mut self.roots.borrow_mut(), rows)?;
        reserve(&mut self.metadata.borrow_mut(), rows)?;
        reserve(&mut self.publications.borrow_mut(), 2)?;
        reserve(&mut self.inventories.borrow_mut(), 2)?;
        reserve(&mut self.root_clones.borrow_mut(), rows)?;
        for _ in 0..rows {
            let clone =
                safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(|cause| {
                    Error::from(
                        crate::backend::runtime::residency::manager::ResidencyError::OriginalClone(
                            cause,
                        ),
                    )
                })?;
            self.root_clones.borrow_mut().push(clone);
        }
        for _ in 0..2 {
            let inventory = if let Some(custody) = &custody {
                RetainedStorage::prepare_original(rows, custody.clone())?
            } else {
                let ordinary = self.ordinary.borrow();
                let (_, host) = ordinary.as_ref().expect("ordinary plan");
                let scope = self.scope.borrow();
                RetainedStorage::prepare_ordinary(
                    rows,
                    scope.as_ref().expect("active scope").pool(),
                    host,
                )?
            };
            self.inventories.borrow_mut().push(inventory);
        }
        Ok(())
    }
    /// Ordinary callers retain the previous collector. Original callers consume
    /// one of this Work's two already prepared publication destinations.
    pub(in crate::composition::mlx::session::model_session) fn prepare_inventory(
        &self,
    ) -> Result<RetainedStorage, Error> {
        if self.native_storage.is_none()
            && self.snapshot.is_none()
            && self.ordinary.borrow().is_none()
        {
            return Err(Error::OriginalSourceContract {
                stage: "prepared work publication inventory",
                cause: WorkingMemoryError::UnknownBound,
            });
        }
        if let Some(cause) = self.take_collection_failure() {
            return Err(cause);
        }
        let mut inventories = self.inventories.borrow_mut();
        let capacity = inventories.capacity();
        let used = capacity - inventories.len();
        let inventory = inventories.pop();
        drop(inventories);
        inventory.ok_or_else(|| {
            self.fail_collection(Error::PrefillControl(
                WorkingMemoryError::CollectorCapacity {
                    kind: CollectorCapacityKind::WorkInventories,
                    used,
                    capacity,
                },
            ))
        })
    }
}
fn qualified() -> bool {
    crate::backend::runtime::residency::storage::native_storage::Bank::shared_borrowed_owner_bytes()
        .is_some()
}
fn array<T>(n: usize) -> Option<usize> {
    Some(Layout::array::<T>(n).ok()?.size())
}
fn work_bytes(rows: usize) -> Option<usize> {
    let fixed = [
        array::<Array>(rows)?,
        array::<safemlx::PreparedArrayClone>(rows)?,
        array::<eredu_runtime::SharedHostMetadata>(rows)?,
        array::<RetainedStoragePublication>(2)?,
        array::<RetainedStorage>(2)?,
        rows.checked_mul(Array::inspection_clone_handle_bytes())?,
        // These sequential clone workers reuse their call frames. The actual
        // retained slot/Array vectors and native handles remain per-row above.
        safemlx::PreparedArrayClone::control_bytes()?,
        // Two actual inventories are prepared for every Work, even the prompt
        // and sampler roles whose single publication leaves one unused.
        RetainedStorage::original_collector_control_bytes(rows)?.checked_mul(2)?,
        size_of::<Result<RetainedStorage, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<std::collections::TryReserveError>(),
        size_of::<Option<Error>>(),
        size_of::<Error>(), // first owned collection failure returned without cloning
        // Post-retirement health check moves the first callback cause out of
        // the funded owner before completion can be cached as successful.
        size_of::<Option<Error>>(),
        size_of::<std::cell::Ref<'static, Option<super::owner::FundedWorkOwner>>>(),
        size_of::<WorkingMemoryError>(),
        size_of::<CollectorCapacityKind>(),
        size_of::<std::cell::RefMut<'static, Vec<Array>>>(),
        size_of::<std::cell::RefMut<'static, Vec<safemlx::PreparedArrayClone>>>(),
        size_of::<std::cell::RefMut<'static, Vec<eredu_runtime::SharedHostMetadata>>>(),
        size_of::<std::cell::RefMut<'static, Vec<RetainedStoragePublication>>>(),
        size_of::<std::cell::RefMut<'static, Vec<RetainedStorage>>>(),
        size_of::<eredu_runtime::working_memory::OriginalTextMetadataCustody>(),
        size_of::<usize>() * 3,
    ];
    fixed
        .into_iter()
        .try_fold(std::mem::size_of_val(&fixed), usize::checked_add)
}
pub(super) fn work_control_bytes(rows: usize) -> Option<u64> {
    u64::try_from(work_bytes(rows)?).ok()
}

pub(super) fn control_bytes(
    attempts: usize,
    rows: usize,
    works: usize,
) -> Result<Option<u64>, Error> {
    if !qualified() {
        return Ok(None);
    }
    let result = (|| {
        let work = u64::try_from(work_bytes(rows)?)
            .ok()?
            .checked_mul(u64::try_from(works).ok()?)?;
        let publication =
            native_publication::control_bytes(rows)?.checked_mul(u64::try_from(attempts).ok()?)?;
        work.checked_add(publication)
    })();
    result
        .map(Some)
        .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))
}
