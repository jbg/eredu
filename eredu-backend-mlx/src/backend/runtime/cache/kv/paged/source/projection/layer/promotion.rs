//! Final source-owned transfer slots; checkout releases every source loan.
use super::*;
use crate::backend::nn::workspace::OriginalPagedScanSource;
use crate::backend::submission_recovery::prefill::TransientRootsProjection;
use safemlx::error::Exception;

impl ProjectedPagedSource {
    pub(crate) fn promotion_access_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Checkout<'_>>(),
            size_of::<Option<PreparedCacheHostPromotion>>(),
            size_of::<(usize, PreparedCacheHostPromotion)>(),
            size_of::<(
                &Self,
                &CacheBlockId,
                &OriginalPagedScanSource<'_>,
                &TransientRootsProjection,
                &Stream,
            )>(),
            size_of::<std::cell::Ref<'_, Vec<Option<PreparedCacheHostPromotion>>>>(),
            size_of::<std::cell::RefMut<'_, Vec<Option<PreparedCacheHostPromotion>>>>(),
            size_of::<std::slice::Iter<'_, Option<PreparedCacheHostPromotion>>>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Option<usize>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }

    pub(crate) fn retain_promotion<F>(
        &mut self,
        id: &CacheBlockId,
        acquire: F,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure>
    where
        F: FnOnce() -> Result<PreparedCacheHostPromotion, CacheSourceFailure>,
    {
        context
            .charge_metadata(size_of::<(
                &mut Self,
                &CacheBlockId,
                F,
                PreparedCacheHostPromotion,
                Result<(), CacheSourceFailure>,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let slots = self.promotions.get_mut();
        if slots.len() >= self.promotion_capacity
            || slots.iter().flatten().any(|slot| slot.id() == id)
            || !self.geometry.blocks.iter().any(|block| {
                &block.id == id
                    && matches!(
                        block.phase,
                        CacheStoragePhase::HostUnbacked | CacheStoragePhase::HostBacked
                    )
            })
        {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Identity,
                context,
            ));
        }
        let value = acquire()?;
        // No fallible operation before the source is safely in its outer owner.
        slots.push(Some(value));
        if slots
            .last()
            .and_then(Option::as_ref)
            .is_none_or(|value| value.id() != id)
        {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Identity,
                context,
            ));
        }
        Ok(())
    }
    pub(crate) fn has_host_promotion(&self, id: &CacheBlockId) -> bool {
        self.promotions
            .try_borrow()
            .ok()
            .is_some_and(|slots| slots.iter().flatten().any(|slot| slot.id() == id))
    }
    pub(crate) fn promote_host(
        &self,
        id: &CacheBlockId,
        proof: &OriginalPagedScanSource<'_>,
        roots: &TransientRootsProjection,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let (index, value) = {
            let mut slots = self
                .promotions
                .try_borrow_mut()
                .map_err(|_| proof.error(CacheSourceError::Busy))?;
            let index = slots
                .iter()
                .position(|slot| slot.as_ref().is_some_and(|value| value.id() == id))
                .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
            (index, slots[index].take().expect("selected one-use source"))
        };
        let mut checkout = Checkout {
            slots: &self.promotions,
            index,
            value: Some(value),
        };
        let value = checkout.value.as_mut().expect("checked out source");
        if value.values().is_none() {
            value.run(proof, roots, stream)?;
        }
        Ok(())
    }
}
struct Checkout<'a> {
    slots: &'a RefCell<Vec<Option<PreparedCacheHostPromotion>>>,
    index: usize,
    value: Option<PreparedCacheHostPromotion>,
}
impl Drop for Checkout<'_> {
    fn drop(&mut self) {
        let value = self.value.take().expect("one source restoration");
        let previous = self.slots.borrow_mut()[self.index].replace(value);
        // Native and pin teardown is always outside the source loan.
        drop(previous);
    }
}
