use super::*;
use crate::{error::Exception, OriginalScopeObserver, SubmissionGraphQuota};
use std::{mem::size_of, ptr};

/// Final native handle storage for one future clone of an existing array.
///
/// Preparation creates no descriptor or numerical buffer. Filling shares the
/// actual retained source descriptor and consumes the slot once. The caller
/// retains its admitted storage custody through this slot and the resulting
/// ordinary [`Array`], including that array's final deallocation.
#[derive(Debug)]
pub struct PreparedArrayClone {
    storage: *mut c_void,
    arena: Option<SubmissionGraphQuota>,
}

/// Fixed preparation/borrow refusal; no native formatted diagnostic is needed.
#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum PreparedArrayCloneCause {
    /// Exact handle storage allocation was refused.
    #[error("native clone handle allocation failed")]
    Allocation,
    /// The existing metadata arena has insufficient unoccupied capacity.
    #[error("native clone handle metadata capacity is exhausted")]
    Capacity,
    /// Slot is absent or was already consumed.
    #[error("native clone handle is unavailable or already consumed")]
    InvalidStorage,
    /// Another thread owns the serialized native runtime.
    #[error("native runtime is busy during retained descriptor inspection")]
    RuntimeBusy,
}

impl PreparedArrayClone {
    /// Allocate the same exact native handle without a runtime callback or
    /// formatted native error. Caller retains prior storage funding on failure.
    pub fn try_prepare_for_inspection() -> Result<Self, PreparedArrayCloneCause> {
        let mut slot = Self {
            storage: ptr::null_mut(),
            arena: None,
        };
        // SAFETY: unique initially-null pointer; no descriptor exists yet.
        let status = unsafe { safemlx_sys::mlx_array_clone_storage_new_fixed(&mut slot.storage) };
        match status {
            0 => Ok(slot),
            6 => Err(PreparedArrayCloneCause::Allocation),
            _ => Err(PreparedArrayCloneCause::InvalidStorage),
        }
    }

    /// Exact fresh metadata arena capacity for this producer's one final handle.
    /// The caller also pays the existing arena's layout and its retained owner.
    pub fn arena_capacity() -> Option<usize> {
        let mut capacity = 0;
        // SAFETY: checked pure layout query writes only this scalar.
        (unsafe { safemlx_sys::mlx_array_clone_storage_arena_capacity(&mut capacity) } == 0)
            .then_some(capacity)
    }

    /// Preallocate one handle in an existing paid metadata arena. Filling and
    /// dropping use the same closed prepared-handle owner as native inputs;
    /// the source backing never retains this handle's metadata payer.
    /// This method creates no submission or execution authority.
    pub fn try_prepare_in(arena: &SubmissionGraphQuota) -> Result<Self, PreparedArrayCloneCause> {
        let mut slot = Self {
            storage: ptr::null_mut(),
            arena: Some(arena.clone()),
        };
        // SAFETY: live retained arena, exclusive initially empty destination.
        match unsafe { safemlx_sys::mlx_array_clone_storage_new_in(&mut slot.storage, arena.raw()) }
        {
            0 => Ok(slot),
            2 => Err(PreparedArrayCloneCause::Capacity),
            _ => Err(PreparedArrayCloneCause::InvalidStorage),
        }
    }

    /// Named controls for the paid-arena form, in addition to that arena's layout.
    pub fn arena_control_bytes() -> Option<usize> {
        Self::control_bytes()?
            .checked_add(unsafe {
                // SAFETY: sizeof inventory only, without native initialization.
                safemlx_sys::mlx_array_clone_storage_arena_control_bytes()
            })?
            .checked_add(size_of::<SubmissionGraphQuota>())?
            .checked_add(size_of::<Option<SubmissionGraphQuota>>())?
            .checked_add(size_of::<usize>())
    }

    unsafe fn fill_raw(&mut self, output: &mut mlx_array, source: &Array) -> u32 {
        if let Some(arena) = &self.arena {
            // SAFETY: caller holds the inspection/original runtime loan and
            // this slot's arena owns the exclusive uninitialized block.
            unsafe {
                safemlx_sys::mlx_array_clone_storage_fill_in(
                    output,
                    &mut self.storage,
                    source.as_ptr(),
                    arena.raw(),
                )
            }
        } else {
            // SAFETY: same unique slot and retained source, ordinary allocation.
            unsafe {
                safemlx_sys::mlx_array_clone_storage_fill(
                    output,
                    &mut self.storage,
                    source.as_ptr(),
                )
            }
        }
    }

    /// Consume a prepared handle to retain this already-owned descriptor.
    /// Same no-hooks inspection lock as Array::try_clone_for_inspection; no
    /// evaluation, numerical allocation, current Scope or authority is created.
    pub fn fill_for_inspection(
        &mut self,
        source: &Array,
    ) -> Result<Array, PreparedArrayCloneCause> {
        runtime_lock::try_retire(|| {
            let mut output = mlx_array {
                ctx: ptr::null_mut(),
                prepared_owner: ptr::null_mut(),
            };
            // SAFETY: unique slot, valid source and empty result under runtime
            // serialization. Native copy construction is statically noexcept.
            let status = unsafe { self.fill_raw(&mut output, source) };
            if status != 0 {
                return Err(PreparedArrayCloneCause::InvalidStorage);
            }
            Ok(Array { c_array: output })
        })
        .ok_or(PreparedArrayCloneCause::RuntimeBusy)?
    }

    /// Cold allocation of exactly one final native handle. A preparation error
    /// preserves the actual native error; no source array is involved yet.
    pub fn try_new() -> Result<Self, Exception> {
        let mut slot = Self {
            storage: ptr::null_mut(),
            arena: None,
        };
        <() as Guarded>::try_from_op(|_| {
            // SAFETY: output is initially null and uniquely owned by this slot.
            unsafe { safemlx_sys::mlx_array_clone_storage_new(&mut slot.storage) }
        })?;
        Ok(slot)
    }

    /// Named preparation/fill/drop controls, excluding the one native handle
    /// allocation reported by [`Array::inspection_clone_handle_bytes`].
    /// This does not price arbitrary native errors or allocator infrastructure.
    pub fn control_bytes() -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<Result<Self, PreparedArrayCloneCause>>(),
            size_of::<Result<Array, PreparedArrayCloneCause>>(),
            size_of::<Option<Result<Array, PreparedArrayCloneCause>>>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Exception>>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<mlx_array>(),
            size_of::<u32>(),
            size_of::<&Array>(),
            size_of::<&OriginalScopeObserver>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<Result<OriginalScopeObserver, Exception>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
        ]
        .into_iter()
        .try_fold(
            // SAFETY: pure sizeof inventory, independent of runtime state.
            unsafe { safemlx_sys::mlx_array_clone_storage_control_bytes() },
            usize::checked_add,
        )
    }

    /// Fill once under the exact current original role. Runtime/owner refusal
    /// and reuse of a consumed slot preserve both source and slot. This performs
    /// no evaluation, hooks, allocation, descriptor reconstruction or payload copy.
    pub fn fill_in_original_scope(
        &mut self,
        source: &Array,
        observer: &OriginalScopeObserver,
    ) -> Result<Array, Exception> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(observer) {
            return Err(observer.error(4));
        }
        let mut output = mlx_array {
            ctx: ptr::null_mut(),
            prepared_owner: ptr::null_mut(),
        };
        // SAFETY: unique slot, live source, empty destination, exact current
        // original owner and no-hooks native serialization. On success the
        // constructor transfers the sole wrapper ownership into output.
        let status = unsafe { self.fill_raw(&mut output, source) };
        if status != 0 {
            return Err(observer.error(status));
        }
        Ok(Array { c_array: output })
    }
}

impl Drop for PreparedArrayClone {
    fn drop(&mut self) {
        // Only unfilled storage can remain. There is no array or descriptor to
        // destroy; an arena returns its block through the existing paid owner.
        if let Some(arena) = &self.arena {
            // SAFETY: the retained arena owns this possibly unfilled block.
            // A successful fill transferred it to the returned Array.
            unsafe { safemlx_sys::mlx_array_clone_storage_free_in(self.storage, arena.raw()) };
        } else {
            unsafe { safemlx_sys::mlx_array_clone_storage_free(self.storage) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
    fn hook() {
        HOOKS.with(|n| n.set(n.get() + 1));
    }
    struct RemoveHook;
    impl Drop for RemoveHook {
        fn drop(&mut self) {
            runtime_lock::unregister_housekeeping_hook(hook);
        }
    }

    #[test]
    fn prepared_inspection_clone_preserves_backing_and_consumes_exact_slot_without_hooks() {
        let source = Array::from_slice(&[7u32, 13, 29], &[1, 3]);
        let before = source.try_allocation_info().unwrap().unwrap();
        let mut slot = PreparedArrayClone::try_prepare_for_inspection().unwrap();
        runtime_lock::register_housekeeping_hook(hook);
        let guard = RemoveHook;
        HOOKS.with(|n| n.set(0));
        let retained = slot.fill_for_inspection(&source).unwrap();
        assert_eq!(retained.try_allocation_info().unwrap(), Some(before));
        assert!(matches!(
            slot.fill_for_inspection(&source),
            Err(PreparedArrayCloneCause::InvalidStorage)
        ));
        assert_eq!(HOOKS.with(Cell::get), 0);
        drop(guard);
        drop(source);
        assert_eq!(
            retained.evaluated().unwrap().as_slice::<u32>(),
            &[7, 13, 29]
        );
        assert_eq!(retained.try_allocation_info().unwrap(), Some(before));
    }

    #[test]
    fn arena_clone_failure_fill_and_drop_retire_independently_of_source_aliases() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct Retired(Arc<AtomicUsize>);
        impl Drop for Retired {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let source = Array::from_slice(&[7u32, 13, 29], &[1, 3]);
        let source_alias = source.clone();
        let before = source.try_allocation_info().unwrap().unwrap();
        let arena = crate::PreparedSubmissionGraphQuota::try_new(
            PreparedArrayClone::arena_capacity().unwrap(),
            Retired(dropped.clone()),
        )
        .unwrap()
        .try_allocate()
        .unwrap();
        let mut slot = PreparedArrayClone::try_prepare_in(&arena).unwrap();
        let occupied = arena.occupied_bytes();
        assert!(occupied > 0);
        let invalid = Array {
            c_array: mlx_array {
                ctx: ptr::null_mut(),
                prepared_owner: ptr::null_mut(),
            },
        };
        assert!(matches!(
            slot.fill_for_inspection(&invalid),
            Err(PreparedArrayCloneCause::InvalidStorage)
        ));
        assert_eq!(arena.occupied_bytes(), occupied);
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        runtime_lock::register_housekeeping_hook(hook);
        let guard = RemoveHook;
        HOOKS.with(|n| n.set(0));
        let retained = slot.fill_for_inspection(&source).unwrap();
        assert_eq!(HOOKS.with(Cell::get), 0);
        assert_eq!(retained.try_allocation_info().unwrap(), Some(before));
        assert!(matches!(
            slot.fill_for_inspection(&source),
            Err(PreparedArrayCloneCause::InvalidStorage)
        ));
        drop(guard);
        drop(slot);
        drop(arena);
        crate::reclaim_allocation_owners();
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        // An ordinary descriptor alias has its own handle; it must not keep the
        // discarded prepared handle's payer while sharing its source backing.
        let retained_alias = retained.clone();
        drop(retained);
        crate::reclaim_allocation_owners();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        assert_eq!(
            source_alias.evaluated().unwrap().as_slice::<u32>(),
            &[7, 13, 29]
        );
        assert_eq!(
            retained_alias.evaluated().unwrap().as_slice::<u32>(),
            &[7, 13, 29]
        );
        assert_eq!(source.try_allocation_info().unwrap(), Some(before));

        let arena = crate::PreparedSubmissionGraphQuota::try_new(
            PreparedArrayClone::arena_capacity().unwrap(),
            Retired(dropped.clone()),
        )
        .unwrap()
        .try_allocate()
        .unwrap();
        let unfilled = PreparedArrayClone::try_prepare_in(&arena).unwrap();
        drop(arena);
        crate::reclaim_allocation_owners();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        drop(unfilled);
        crate::reclaim_allocation_owners();
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }
}
