use super::*;
use crate::{error::Exception, OriginalScopeObserver};
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
}

/// Fixed preparation/borrow refusal; no native formatted diagnostic is needed.
#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum PreparedArrayCloneCause {
    /// Exact handle storage allocation was refused.
    #[error("native clone handle allocation failed")]
    Allocation,
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
        };
        // SAFETY: unique initially-null pointer; no descriptor exists yet.
        let status = unsafe { safemlx_sys::mlx_array_clone_storage_new_fixed(&mut slot.storage) };
        match status {
            0 => Ok(slot),
            6 => Err(PreparedArrayCloneCause::Allocation),
            _ => Err(PreparedArrayCloneCause::InvalidStorage),
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
            let status = unsafe {
                safemlx_sys::mlx_array_clone_storage_fill(
                    &mut output,
                    &mut self.storage,
                    source.as_ptr(),
                )
            };
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
        let status = unsafe {
            safemlx_sys::mlx_array_clone_storage_fill(
                &mut output,
                &mut self.storage,
                source.as_ptr(),
            )
        };
        if status != 0 {
            return Err(observer.error(status));
        }
        Ok(Array { c_array: output })
    }
}

impl Drop for PreparedArrayClone {
    fn drop(&mut self) {
        // SAFETY: only unfilled operator-new storage can remain. There is no
        // array/descriptor/callback to destroy and no runtime entry is needed.
        unsafe { safemlx_sys::mlx_array_clone_storage_free(self.storage) };
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
        assert_eq!(retained.evaluated().unwrap().as_slice::<u32>(), &[7, 13, 29]);
        assert_eq!(retained.try_allocation_info().unwrap(), Some(before));
    }
}
