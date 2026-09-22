//! Prepared physical-backing attachment; no funding or completion authority.
use super::RetiredOwner;
use super::original_buffer::{OriginalBufferCause, OriginalBufferError};
use crate::{Array, error::Exception, utils::guard::Guarded, utils::runtime_lock};
use std::{alloc::Layout, ffi::c_void, fmt, mem, ptr::NonNull};

mod retirement;
pub use retirement::PreparedAllocationRetirement;

// The optional completion slot belongs to this exact prepared attachment.
// Other allocation-owner producers keep their existing node representation.
#[repr(C)]
struct OwnedNode<T> {
    retired: RetiredOwner,
    retirement: Option<std::sync::Arc<retirement::Slot>>,
    owner: T,
}
fn take_owner<T>(node: Box<OwnedNode<T>>) -> T {
    let OwnedNode {
        retirement, owner, ..
    } = *node;
    // Actual node and optional shared shell retire before the accounted payload.
    drop(retirement);
    owner
}
unsafe fn destroy<T>(node: *mut RetiredOwner) {
    // SAFETY: repr(C) places this exact node's retirement header first.
    let owner = take_owner(unsafe { Box::from_raw(node.cast::<OwnedNode<T>>()) });
    drop(owner);
}

/// Fixed representation facts for one attachment of owner type `T`. Existing
/// payload allocations, allocator bookkeeping and backing are excluded. These
/// facts supply no admission: a closed backend plan must count actual owner
/// types, finite slots and overlapping wrappers before constructing them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationOwnerLayout {
    rust_node_bytes: usize,
    native_node_bytes: usize,
    native_list_bytes: usize,
    prepared_bytes: usize,
    preparation_control_bytes: usize,
    preparation_failure_bytes: usize,
    attachment_failure_bytes: usize,
    original_attachment_control_bytes: usize,
}
impl AllocationOwnerLayout {
    /// One Rust heap node, including the actual owner representation.
    pub fn rust_node_bytes(self) -> usize {
        self.rust_node_bytes
    }
    /// One exact linked native heap node; no shared control block is created.
    pub fn native_node_bytes(self) -> usize {
        self.native_node_bytes
    }
    /// Inline canonical list in each actual Data/HostTransferStorage, replacing
    /// the previous vector. This is not allocated once per attachment.
    pub fn native_list_bytes(self) -> usize {
        self.native_list_bytes
    }
    /// Actual named construction owner plus its allocation-layout/pointer locals.
    /// Conservatively retain this alongside the result/error wrapper at handoff.
    pub fn preparation_control_bytes(self) -> usize {
        self.preparation_control_bytes
    }
    /// Inline representation of one successfully prepared owner wrapper.
    pub fn prepared_bytes(self) -> usize {
        self.prepared_bytes
    }
    /// Inline preparation error, including the original owner representation.
    pub fn preparation_failure_bytes(self) -> usize {
        self.preparation_failure_bytes
    }
    /// Inline attachment error, including the complete prepared wrapper.
    pub fn attachment_failure_bytes(self) -> usize {
        self.attachment_failure_bytes
    }
    /// Named C/Rust transports for the checked original-birth attachment,
    /// including its owning failure. Separate from the two prepaid node layouts.
    pub fn original_attachment_control_bytes(self) -> usize {
        self.original_attachment_control_bytes
    }
    /// Checked sum of the two newly allocated heap representations.
    pub fn allocation_bytes(self) -> Option<usize> {
        self.rust_node_bytes.checked_add(self.native_node_bytes)
    }
}

/// A prepared handoff could not proceed; no failure certifies completion.
#[derive(Debug, thiserror::Error)]
pub enum PreparedAllocationOwnerCause {
    /// The nonblocking native runtime loan could not be acquired.
    #[error("native runtime is busy during prepared owner attachment")]
    RuntimeBusy,
    /// The destination has not established supported completed backing.
    #[error("backing is unfinished or uncertified")]
    UncertifiedBacking,
    /// The destination has no physical allocation to retain the owner.
    #[error("value has no physical allocation")]
    NoAllocation,
    /// A fixed Rust or native preparation node could not be allocated.
    #[error("prepared owner node allocation failed")]
    AllocationFailed,
    /// The native attachment entry returned the preserved native exception.
    #[error("native owner attachment failed: {0}")]
    Native(#[source] Exception),
}
/// Error first, entire original unattached owner last, preserving its custody.
pub struct PreparedAllocationOwnerError<O> {
    cause: PreparedAllocationOwnerCause,
    owner: O,
}
impl<O> PreparedAllocationOwnerError<O> {
    /// Borrows the failure without consuming the retained original owner.
    pub fn cause(&self) -> &PreparedAllocationOwnerCause {
        &self.cause
    }
    /// Borrows the entire owner returned by the failed operation.
    pub fn owner(&self) -> &O {
        &self.owner
    }
    /// Returns the failure and original owner without changing native visibility.
    pub fn into_parts(self) -> (PreparedAllocationOwnerCause, O) {
        (self.cause, self.owner)
    }
}
impl<O> fmt::Debug for PreparedAllocationOwnerError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedAllocationOwnerError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<O> fmt::Display for PreparedAllocationOwnerError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<O> std::error::Error for PreparedAllocationOwnerError<O> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
// Actual constructor state, with all in-flight payload ownership explicit.
// Empty native storage drops before Rust payloads on an allocation failure/panic.
struct Preparation<T> {
    native: Option<NativeNode>,
    node: Option<Box<OwnedNode<T>>>,
    owner: Option<T>,
}
impl<T> Drop for Preparation<T> {
    fn drop(&mut self) {
        drop(self.native.take());
        if let Some(node) = self.node.take() {
            let owner = take_owner(node);
            drop(owner);
        }
        // An original T not yet moved into a Rust node remains in owner and
        // drops after these now-empty node fields, including on unwind.
    }
}
struct NativeNode(NonNull<c_void>);
// SAFETY: an unattached node owns no native runtime object or payload. It is
// exclusive movable host storage, freed without a runtime call or callback.
unsafe impl Send for NativeNode {}
impl Drop for NativeNode {
    fn drop(&mut self) {
        // SAFETY: only unattached nodes reach here; success forgets this guard.
        unsafe { safemlx_sys::mlx_allocation_owner_node_free(self.0.as_ptr()) }
    }
}

/// Exclusive preparation for one completed physical-backing attachment.
///
/// Cold preparation initializes the native error handler (and may wait for its
/// one-time initialization) before allocating two fixed nodes. Attachment
/// allocates nothing, performs
/// no evaluation, polling, wait, reclamation, housekeeping or owner clone. The
/// owner must not retain the destination backing or its graph, directly or
/// indirectly; payload-free accounting handles are suitable. This supplies no
/// funding, registration or completion authority. Unfinished or uncertified backing
/// and values with no physical allocation reject. A logically empty array with
/// certified physical backing can retain the owner. Known allocation facts can
/// also describe an allocation-free empty value through the zero identity
/// sentinel; that value still rejects attachment. Deferred attachment keeps
/// its separate legacy semantics.
///
/// Every failure returns the entire preparation. Success consumes it once;
/// backing destruction frees the native node before queueing the payload for
/// `reclaim_allocation_owners`, including reclamation by another thread. Both
/// queued and unattached retirement free the Rust node before dropping T.
/// Earlier successes and remaining preparations must remain in original recovery
/// custody if a later attachment fails. No rollback of native visibility occurs.
pub struct PreparedAllocationOwner<T: Send + 'static> {
    // Empty native storage retires before the Rust payload/custody on failure.
    native: Option<NativeNode>,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> Drop for PreparedAllocationOwner<T> {
    fn drop(&mut self) {
        drop(self.native.take());
        if let Some(node) = self.node.take() {
            let owner = take_owner(node);
            drop(owner);
        }
    }
}
// SAFETY: exclusive native/Rust host nodes have not been attached; their links
// are null, and T: Send permits later queue reclamation on another host thread.
unsafe impl<T: Send + 'static> Send for PreparedAllocationOwner<T> {}
impl<T: Send + 'static> fmt::Debug for PreparedAllocationOwner<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedAllocationOwner")
            .finish_non_exhaustive()
    }
}
// Private selection from an actual borrowed native witness; callers cannot
// choose a kind while supplying their own key/capacity.
enum AllocationExpectation {
    Original(Option<safemlx_sys::mlx_original_buffer_budget>),
    Ordinary,
    Immutable,
    HostTransfer(bool),
    HostTransferView(u64),
}

impl<T: Send + 'static> PreparedAllocationOwner<T> {
    /// Cold exact facts, without a lock, allocation or runtime activity.
    pub fn layout() -> AllocationOwnerLayout {
        AllocationOwnerLayout {
            rust_node_bytes: mem::size_of::<OwnedNode<T>>(),
            // SAFETY: linked sizeof-only producers take no pointers or locks.
            native_node_bytes: unsafe { safemlx_sys::mlx_allocation_owner_node_bytes() },
            native_list_bytes: unsafe { safemlx_sys::mlx_allocation_owner_list_bytes() },
            prepared_bytes: mem::size_of::<Self>(),
            preparation_control_bytes: mem::size_of::<Preparation<T>>()
                + mem::size_of::<Layout>()
                + mem::size_of::<Option<NonNull<u8>>>(),
            preparation_failure_bytes: mem::size_of::<PreparedAllocationOwnerError<T>>(),
            attachment_failure_bytes: mem::size_of::<PreparedAllocationOwnerError<Self>>(),
            // SAFETY: sizeof-only C transport query; no runtime entry.
            original_attachment_control_bytes: unsafe {
                safemlx_sys::mlx_original_buffer_attachment_control_bytes()
            } + mem::size_of::<Self>()
                + mem::size_of::<super::original_buffer::OriginalBufferWitness<'static>>()
                + mem::size_of::<super::original_buffer::OriginalBufferAliasWitness<'static>>()
                + mem::size_of::<super::original_buffer::ImmutableSourceWitness<'static>>()
                + mem::size_of::<super::original_buffer::OrdinaryBufferWitness<'static>>()
                + mem::size_of::<(
                    &Array,
                    safemlx_sys::mlx_original_buffer_info,
                    AllocationExpectation,
                )>()
                + mem::size_of::<AllocationExpectation>()
                + mem::size_of::<safemlx_sys::mlx_original_buffer_info>()
                + mem::size_of::<Option<safemlx_sys::mlx_original_buffer_budget>>()
                // The checked wrapper and shared node-move worker each own
                // their actual handoff/result frame; callback transport above
                // remains separate from these two ownership representations.
                + mem::size_of::<Self>()
                + mem::size_of::<Result<(), OriginalBufferError<Self>>>() * 2
                + mem::size_of::<Result<(), OriginalBufferCause>>()
                + mem::size_of::<runtime_lock::RuntimeLockGuard>()
                + mem::size_of::<u32>()
                + mem::size_of::<*mut c_void>() * 2,
        }
    }
    /// Allocates fixed host representations before any attachment, without a
    /// runtime lock or housekeeping. This cold boundary may wait for the error
    /// handler's one-time initialization; attachment never enters that initializer.
    /// Failure returns the original owner.
    pub fn try_new(owner: T) -> Result<Self, PreparedAllocationOwnerError<T>> {
        Self::prepare_with(
            owner,
            || {
                // SAFETY: fresh empty native node or null, without runtime state.
                unsafe { safemlx_sys::mlx_allocation_owner_node_new() }
            },
            |layout| {
                // SAFETY: nonzero OwnedNode layout; null is handled below.
                unsafe { std::alloc::alloc(layout) }
            },
        )
    }
    fn prepare_with(
        owner: T,
        native: impl FnOnce() -> *mut c_void,
        allocate: impl FnOnce(Layout) -> *mut u8,
    ) -> Result<Self, PreparedAllocationOwnerError<T>> {
        // Every preparation, including the private allocation-failure seam, is
        // initialized before a node can escape into the bounded attach path.
        crate::error::ensure_mlx_error_handler();
        let mut preparation = Preparation {
            native: None,
            node: None,
            owner: Some(owner),
        };
        let Some(native) = NonNull::new(native()) else {
            return Err(PreparedAllocationOwnerError {
                cause: PreparedAllocationOwnerCause::AllocationFailed,
                owner: preparation.owner.take().unwrap(),
            });
        };
        preparation.native = Some(NativeNode(native));
        let Some(node) = NonNull::new(allocate(Layout::new::<OwnedNode<T>>())) else {
            // Release constructor storage before handing the original owner out.
            drop(preparation.native.take());
            return Err(PreparedAllocationOwnerError {
                cause: PreparedAllocationOwnerCause::AllocationFailed,
                owner: preparation.owner.take().unwrap(),
            });
        };
        // SAFETY: exact allocator/Layout pair, initialized once before Box owns
        // it. OwnedNode is nonzero-sized because it contains RetiredOwner. The
        // remaining moves cannot call providers, allocate or unwind.
        preparation.node = Some(unsafe {
            let node = node.cast::<OwnedNode<T>>().as_ptr();
            node.write(OwnedNode {
                retired: RetiredOwner {
                    next: std::ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                retirement: None,
                owner: preparation.owner.take().unwrap(),
            });
            Box::from_raw(node)
        });
        Ok(Self {
            native: preparation.native.take(),
            node: preparation.node.take(),
        })
    }
    /// Original payload, read-only; no destination or backing authority.
    pub fn owner(&self) -> &T {
        &self.node.as_ref().unwrap().owner
    }
    /// Abandons unattached preparation, returning the original owner.
    pub fn into_owner(mut self) -> T {
        drop(self.native.take());
        take_owner(self.node.take().unwrap())
    }
    /// Nonblocking completed-Array handoff; physical aliases share the owner.
    /// Independently allocated results need their own attachment.
    pub fn try_attach(self, array: &Array) -> Result<(), PreparedAllocationOwnerError<Self>> {
        self.attach_with(|outcome, node, payload| unsafe {
            // SAFETY: exclusive fresh node/payload and actual borrowed array;
            // callback only queues the matching Rust retirement node.
            safemlx_sys::mlx_array_attach_prepared_allocation_owner(
                outcome,
                array.as_ptr(),
                node,
                payload,
                Some(retirement::retire::<T>),
            )
        })
    }
    pub(crate) fn attach_host(
        self,
        buffer: safemlx_sys::mlx_host_transfer_buffer,
    ) -> Result<(), PreparedAllocationOwnerError<Self>> {
        self.attach_with(|outcome, node, payload| unsafe {
            // SAFETY: a borrowed immutable owner or fresh exclusive writer
            // retains this exact buffer throughout the nonallocating handoff.
            safemlx_sys::mlx_host_transfer_buffer_attach_prepared_allocation_owner(
                outcome,
                buffer,
                node,
                payload,
                Some(retirement::retire::<T>),
            )
        })
    }
    pub(crate) fn host_attachment_control_bytes() -> Option<usize> {
        let native = unsafe {
            // SAFETY: linked sizeof-only query; no native object or allocation.
            safemlx_sys::mlx_host_transfer_buffer_prepared_owner_control_bytes()
        };
        let frames = [
            native,
            mem::size_of::<Self>(),
            mem::size_of::<Result<(), PreparedAllocationOwnerError<Self>>>(),
            mem::size_of::<PreparedAllocationOwnerCause>(),
            mem::size_of::<safemlx_sys::mlx_host_transfer_buffer>(),
            mem::size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            mem::size_of::<(i32, *mut i32, *mut c_void, *mut c_void)>(),
            mem::size_of::<Result<(), Exception>>(),
            mem::size_of::<Option<NativeNode>>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
        ];
        frames
            .into_iter()
            .try_fold(mem::size_of_val(&frames), usize::checked_add)
    }
    // Both closed witnesses use this exact node move. No initialized exception
    // handler is entered: all producer refusals are fixed, allocation-free statuses.
    pub(super) fn attach_original(
        self,
        array: &Array,
        expected: safemlx_sys::mlx_original_buffer_info,
        budget: Option<safemlx_sys::mlx_original_buffer_budget>,
    ) -> Result<(), OriginalBufferError<Self>> {
        self.attach_checked(array, expected, AllocationExpectation::Original(budget))
    }
    pub(super) fn attach_ordinary(
        self,
        array: &Array,
        expected: safemlx_sys::mlx_original_buffer_info,
    ) -> Result<(), OriginalBufferError<Self>> {
        self.attach_checked(array, expected, AllocationExpectation::Ordinary)
    }
    pub(super) fn attach_immutable(
        self,
        array: &Array,
        expected: safemlx_sys::mlx_original_buffer_info,
    ) -> Result<(), OriginalBufferError<Self>> {
        self.attach_checked(array, expected, AllocationExpectation::Immutable)
    }
    pub(super) fn attach_host_array(
        self,
        array: &Array,
        expected: safemlx_sys::mlx_immutable_host_transfer_info,
    ) -> Result<(), OriginalBufferError<Self>> {
        self.attach_checked(
            array,
            expected.backing,
            AllocationExpectation::HostTransfer(expected.prepared_source),
        )
    }
    pub(super) fn attach_host_view(
        self,
        array: &Array,
        expected: safemlx_sys::mlx_host_transfer_view_info,
    ) -> Result<(), OriginalBufferError<Self>> {
        self.attach_checked(
            array,
            expected.backing,
            AllocationExpectation::HostTransferView(expected.view_identity),
        )
    }
    pub(crate) fn attach_immutable_host(
        self,
        buffer: safemlx_sys::mlx_host_transfer_buffer,
        expected: safemlx_sys::mlx_immutable_host_transfer_info,
    ) -> Result<(), OriginalBufferError<Self>> {
        self.attach_original_with(|node, payload| unsafe {
            // SAFETY: called only by a live borrowed immutable Host witness.
            safemlx_sys::mlx_immutable_host_transfer_attach(
                buffer,
                &expected,
                node,
                payload,
                Some(retirement::retire::<T>),
            )
        })
    }
    fn attach_checked(
        self,
        array: &Array,
        expected: safemlx_sys::mlx_original_buffer_info,
        kind: AllocationExpectation,
    ) -> Result<(), OriginalBufferError<Self>> {
        self.attach_original_with(|node, payload| unsafe {
            // SAFETY: the borrowed witness retains the actual source; the
            // shared worker supplies exclusive nodes under one runtime loan.
            match kind {
                AllocationExpectation::Original(Some(budget)) => {
                    safemlx_sys::mlx_original_buffer_array_attach(
                        array.as_ptr(),
                        budget,
                        &expected,
                        node,
                        payload,
                        Some(retirement::retire::<T>),
                    )
                }
                AllocationExpectation::Original(None) => {
                    safemlx_sys::mlx_original_buffer_array_alias_attach(
                        array.as_ptr(),
                        &expected,
                        node,
                        payload,
                        Some(retirement::retire::<T>),
                    )
                }
                AllocationExpectation::Immutable => safemlx_sys::mlx_immutable_source_array_attach(
                    array.as_ptr(),
                    &expected,
                    node,
                    payload,
                    Some(retirement::retire::<T>),
                ),
                AllocationExpectation::HostTransfer(prepared_source) => {
                    safemlx_sys::mlx_host_transfer_array_alias_attach(
                        array.as_ptr(),
                        &safemlx_sys::mlx_immutable_host_transfer_info {
                            backing: expected,
                            prepared_source,
                        },
                        node,
                        payload,
                        Some(retirement::retire::<T>),
                    )
                }
                AllocationExpectation::HostTransferView(view_identity) => {
                    safemlx_sys::mlx_host_transfer_array_view_attach(
                        array.as_ptr(),
                        &safemlx_sys::mlx_host_transfer_view_info {
                            backing: expected,
                            view_identity,
                        },
                        node,
                        payload,
                        Some(retirement::retire::<T>),
                    )
                }
                AllocationExpectation::Ordinary => safemlx_sys::mlx_ordinary_buffer_array_attach(
                    array.as_ptr(),
                    &expected,
                    node,
                    payload,
                    Some(retirement::retire::<T>),
                ),
            }
        })
    }
    fn attach_original_with(
        mut self,
        attach: impl FnOnce(*mut c_void, *mut c_void) -> u32,
    ) -> Result<(), OriginalBufferError<Self>> {
        let status = {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(OriginalBufferError::new(
                    OriginalBufferCause::RuntimeBusy,
                    self,
                ));
            };
            let node = self.native.as_ref().unwrap().0.as_ptr();
            let payload = std::ptr::from_mut(&mut self.node.as_mut().unwrap().retired).cast();
            // Only status zero consumes these original exclusive nodes.
            attach(node, payload)
        };
        // Release the runtime loan before any caller-owned failure can retire.
        match OriginalBufferCause::check(status) {
            Ok(()) => {
                mem::forget(self.native.take().unwrap());
                let _ = Box::into_raw(self.node.take().unwrap());
                Ok(())
            }
            Err(cause) => Err(OriginalBufferError::new(cause, self)),
        }
    }
    fn attach_with(
        mut self,
        attach: impl FnOnce(*mut i32, *mut c_void, *mut c_void) -> i32,
    ) -> Result<(), PreparedAllocationOwnerError<Self>> {
        let mut outcome = 0;
        let result = {
            let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
                return Err(PreparedAllocationOwnerError {
                    cause: PreparedAllocationOwnerCause::RuntimeBusy,
                    owner: self,
                });
            };
            <() as Guarded>::try_from_initialized_op(|_| {
                attach(
                    &mut outcome,
                    self.native.as_ref().unwrap().0.as_ptr(),
                    std::ptr::from_mut(&mut self.node.as_mut().unwrap().retired).cast(),
                )
            })
        };
        // Guard is gone before returned-owner teardown or any assertion panic.
        if outcome == 2 {
            // Empty both Drop slots only after the native producer consumed
            // them. No payload destructor or allocation follows this handoff.
            mem::forget(self.native.take().unwrap());
            let _ = Box::into_raw(self.node.take().unwrap());
            debug_assert!(result.is_ok(), "handoff cannot fail after consumption");
            return Ok(());
        }
        let cause = match result {
            Err(error) => PreparedAllocationOwnerCause::Native(error),
            Ok(()) if outcome == 1 => PreparedAllocationOwnerCause::NoAllocation,
            Ok(()) => PreparedAllocationOwnerCause::UncertifiedBacking,
        };
        Err(PreparedAllocationOwnerError { cause, owner: self })
    }
}
#[cfg(test)]
mod tests;
