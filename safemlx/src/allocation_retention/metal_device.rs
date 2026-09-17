//! One actual embedded-library Device birth, before the input allocator.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner};
use crate::utils::runtime_lock;
use std::{alloc::Layout, ffi::c_void, fmt, mem};

/// Fixed construction stage or unchanged-owner refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MetalDeviceCause {
    /// The compiled producer cannot supply a qualified construction layout.
    #[error("Metal Device producer layout is unqualified")]
    UnknownLayout,
    /// The supplied initialization context is invalid.
    #[error("invalid Metal Device initialization context")]
    Invalid,
    /// Another initialization or native runtime operation holds the required loan.
    #[error("Metal Device initialization is busy")]
    Busy,
    /// Storage for the initialization owner could not be allocated.
    #[error("Metal Device owner allocation failed")]
    AllocationFailed,
    /// The selected native Metal device is unavailable.
    #[error("no Metal Device is available")]
    NoDevice,
    /// An ordinary device already exists and cannot acquire admitted provenance.
    #[error("Metal Device has an ordinary predecessor")]
    OrdinaryPredecessor,
    /// The singleton already has an admitted initialization owner.
    #[error("Metal Device already has an admitted owner")]
    AlreadyInitialized,
    /// The retained identity does not name the initialized singleton.
    #[error("Metal Device initialization identity mismatch")]
    IdentityMismatch,
    /// No further initialization identity can be issued.
    #[error("Metal Device initialization identity supply exhausted")]
    IdentityExhausted,
    /// The selected library source has no qualified construction recipe.
    #[error("selected Metal library source is unqualified")]
    UnqualifiedSource,
    /// The selected source no longer matches the retained construction layout.
    #[error("selected Metal Device source changed")]
    SourceChanged,
    /// Native residency initialization failed.
    #[error("Metal residency initialization failed")]
    ResidencyFailed,
    /// The embedded library could not be decompressed.
    #[error("embedded Metal library decompression failed")]
    DecompressionFailed,
    /// The decompressed library could not be transferred to dispatch data.
    #[error("embedded Metal library dispatch data failed")]
    DispatchFailed,
    /// Metal could not load the embedded library.
    #[error("embedded Metal library load failed")]
    LibraryFailed,
    /// The native architecture name is invalid for this producer.
    #[error("Metal architecture name is invalid")]
    InvalidArchitecture,
}
fn cause(status: u32) -> MetalDeviceCause {
    use MetalDeviceCause::*;
    match status {
        1 => UnknownLayout,
        3 => Busy,
        4 => AllocationFailed,
        5 => NoDevice,
        6 => OrdinaryPredecessor,
        7 => AlreadyInitialized,
        8 => IdentityMismatch,
        9 => IdentityExhausted,
        10 => UnqualifiedSource,
        11 => SourceChanged,
        12 => ResidencyFailed,
        13 => DecompressionFailed,
        14 => DispatchFailed,
        15 => LibraryFailed,
        16 => InvalidArchitecture,
        _ => Invalid,
    }
}
/// The fixed cause precedes exact retained source/accounting custody.
pub struct MetalDeviceError<T> {
    cause: MetalDeviceCause,
    owner: T,
}
impl<T> MetalDeviceError<T> {
    /// Converts the retained owner while preserving the exact refusal cause.
    /// A caller may cancel an unadopted preparation to free its shell and keep
    /// the same source custody in the returned error.
    pub fn map_owner<U>(self, map: impl FnOnce(T) -> U) -> MetalDeviceError<U> {
        MetalDeviceError {
            cause: self.cause,
            owner: map(self.owner),
        }
    }

    /// The fixed initialization stage or refusal cause.
    pub fn cause(&self) -> MetalDeviceCause {
        self.cause
    }
    /// Borrow the original owner retained by this failure.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Recover the cause and the exact owner that was not transferred.
    pub fn into_parts(self) -> (MetalDeviceCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for MetalDeviceError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalDeviceError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for MetalDeviceError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for MetalDeviceError<T> {}

/// Pure fixed module/source storage, separate from one dynamic constructor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalDeviceStaticLayout {
    /// Requested fixed module and embedded-source storage in bytes.
    pub bytes: usize,
    /// Whether this compiled producer qualifies the reported static storage.
    pub qualified: bool,
    /// Whether the selected backend requires this static contribution.
    pub required: bool,
}
/// Query fixed Device/source storage without initializing a device or runtime.
pub fn metal_device_static_layout() -> MetalDeviceStaticLayout {
    let mut native = safemlx_sys::mlx_device_initialization_static_layout::default();
    // SAFETY: initialized fixed repr(C) output; no runtime, source or Device read.
    unsafe { safemlx_sys::mlx_device_initialization_static_layout_for(&mut native) };
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    let bytes = native
        .bytes
        .checked_add(safemlx_sys::MLX_METALLIB_LZFSE.len())
        .and_then(|n| n.checked_add(mem::size_of_val(&safemlx_sys::MLX_METALLIB_LZFSE)));
    #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
    let bytes = Some(native.bytes);
    MetalDeviceStaticLayout {
        bytes: bytes.unwrap_or(native.bytes),
        qualified: native.qualified == 1 && bytes.is_some(),
        required: native.required != 0,
    }
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn source() -> safemlx_sys::mlx_device_initialization_source {
    // These generated static bytes are the only safe source constructor.
    safemlx_sys::mlx_device_initialization_source {
        compressed_data: safemlx_sys::MLX_METALLIB_LZFSE.as_ptr(),
        compressed_size: safemlx_sys::MLX_METALLIB_LZFSE.len(),
        uncompressed_size: safemlx_sys::MLX_METALLIB_UNCOMPRESSED_SIZE,
    }
}
// CPU/CUDA builds have no embedded Metal source. The native layout query
// retains its typed unavailable response and cannot construct a Metal device.
#[cfg(not(all(feature = "metal", target_vendor = "apple")))]
fn source() -> safemlx_sys::mlx_device_initialization_source {
    safemlx_sys::mlx_device_initialization_source {
        compressed_data: std::ptr::null(),
        compressed_size: 0,
        uncompressed_size: 0,
    }
}
/// Actual selected constructor payload and named controls. No Device/runtime
/// resource or source registration is performed by this layout query.
pub struct MetalDeviceLayout<T> {
    owner_type: std::marker::PhantomData<fn() -> T>,
    native: safemlx_sys::mlx_device_initialization_layout,
    rust_node_bytes: usize,
    control_bytes: usize,
}
impl<T> Copy for MetalDeviceLayout<T> {}
impl<T> Clone for MetalDeviceLayout<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> fmt::Debug for MetalDeviceLayout<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalDeviceLayout")
            .field("native", &self.native)
            .field("rust_node_bytes", &self.rust_node_bytes)
            .field("control_bytes", &self.control_bytes)
            .finish()
    }
}
impl<T> MetalDeviceLayout<T> {
    /// Requested storage for the Rust node that retains the supplied owner.
    pub fn rust_node_bytes(&self) -> usize {
        self.rust_node_bytes
    }
    /// Named construction, query, result and retirement control storage.
    pub fn control_bytes(&self) -> usize {
        self.control_bytes
    }
    /// Requested native Device object storage.
    pub fn object_bytes(&self) -> usize {
        self.native.object_bytes
    }
    /// Storage for the decompressed embedded library.
    pub fn decoded_bytes(&self) -> usize {
        self.native.decoded_bytes
    }
    /// Library copy storage retained by dispatch data during construction.
    pub fn dispatch_copy_bytes(&self) -> usize {
        self.native.dispatch_copy_bytes
    }
    /// Decompression scratch storage required by the selected source.
    pub fn scratch_bytes(&self) -> usize {
        self.native.scratch_bytes
    }
    /// Storage required to retain the observed library-source override.
    pub fn override_bytes(&self) -> usize {
        self.native.override_bytes
    }
    /// Checked sum of this producer's dynamic storage and named controls.
    pub fn required_bytes(&self) -> Option<usize> {
        [
            self.native.object_bytes,
            self.native.decoded_bytes,
            self.native.dispatch_copy_bytes,
            self.native.scratch_bytes,
            self.native.override_bytes,
            self.rust_node_bytes,
            self.control_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}
/// Private completed singleton identity; no raw Device pointer or account grant.
#[derive(Debug)]
pub struct InitializedMetalDevice {
    pub(super) identity: u64,
}
impl InitializedMetalDevice {
    /// Validate this retained singleton identity without creating a device.
    pub fn try_borrow(&self) -> Result<(), MetalDeviceCause> {
        let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
            return Err(MetalDeviceCause::Busy);
        };
        // SAFETY: private identity came only from successful same-slot construction.
        let status = unsafe { safemlx_sys::mlx_device_initialized_borrow(self.identity) };
        if status == 0 {
            Ok(())
        } else {
            Err(cause(status))
        }
    }
}
/// One exact prepared queue node. Success alone transfers it to native storage.
/// The supplied owner is custody, not a byte grant or a library-source override.
pub struct PreparedMetalDevice<T: Send + 'static> {
    layout: MetalDeviceLayout<T>,
    node: Option<Box<OwnedNode<T>>>,
}
impl<T: Send + 'static> fmt::Debug for PreparedMetalDevice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedMetalDevice")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}
impl<T: Send + 'static> Drop for PreparedMetalDevice<T> {
    fn drop(&mut self) {
        if let Some(node) = self.node.take() {
            drop(take_owner(node));
        }
    }
}
impl<T: Send + 'static> PreparedMetalDevice<T> {
    /// Query the exact owner-typed constructor layout before reserving storage.
    pub fn layout() -> Result<MetalDeviceLayout<T>, MetalDeviceCause> {
        let mut native = safemlx_sys::mlx_device_initialization_layout::default();
        // SAFETY: generated immutable source and initialized fixed output.
        let status =
            unsafe { safemlx_sys::mlx_device_initialization_layout_for(&mut native, source()) };
        if status != 0 {
            return Err(cause(status));
        }
        let controls = [
            native.controls,
            mem::size_of::<Self>(),
            mem::size_of::<T>(),
            mem::size_of::<MetalDeviceLayout<T>>(),
            mem::size_of::<Option<usize>>(),
            mem::size_of::<Result<MetalDeviceLayout<T>, MetalDeviceCause>>(),
            mem::size_of::<safemlx_sys::mlx_device_initialization_source>(),
            mem::size_of::<safemlx_sys::mlx_device_initialization_layout>(),
            mem::size_of::<Layout>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<OwnedNode<T>>(),
            mem::size_of::<Box<OwnedNode<T>>>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<InitializedMetalDevice>(),
            mem::size_of::<Result<Self, MetalDeviceError<T>>>(),
            mem::size_of::<Result<InitializedMetalDevice, MetalDeviceError<Self>>>(),
            mem::size_of::<Result<(), MetalDeviceCause>>(),
            mem::size_of::<runtime_lock::RuntimeLockGuard>(),
            mem::size_of::<u64>(),
            mem::size_of::<u32>(),
            mem::size_of::<*mut c_void>(),
            mem::size_of::<super::RetirementBatch>(),
            mem::size_of::<*mut RetiredOwner>(),
            mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
            mem::size_of::<T>(),
        ];
        let control_bytes = controls
            .into_iter()
            .try_fold(mem::size_of_val(&controls), usize::checked_add)
            .ok_or(MetalDeviceCause::UnknownLayout)?;
        let layout = MetalDeviceLayout {
            owner_type: std::marker::PhantomData,
            native,
            rust_node_bytes: mem::size_of::<OwnedNode<T>>(),
            control_bytes,
        };
        layout
            .required_bytes()
            .ok_or(MetalDeviceCause::UnknownLayout)?;
        Ok(layout)
    }
    /// Query the constructor and allocate its owner node, returning custody on refusal.
    /// The caller must already protect these allocations with its accounting policy.
    pub fn try_new(owner: T) -> Result<Self, MetalDeviceError<T>> {
        let layout = match Self::layout() {
            Ok(value) => value,
            Err(cause) => return Err(MetalDeviceError { cause, owner }),
        };
        Self::with_layout(layout, owner)
    }
    /// Consume the exact immutable queried layout after its account comparison.
    /// Configuration is revalidated against this plan before native allocation;
    /// a larger reached override cannot silently replace the admitted bound.
    pub fn with_layout(
        layout: MetalDeviceLayout<T>,
        owner: T,
    ) -> Result<Self, MetalDeviceError<T>> {
        // Layout carries the owner type: reject a query prepared for a different
        // node representation before allocating this actual shell.
        if layout.rust_node_bytes != mem::size_of::<OwnedNode<T>>() {
            return Err(MetalDeviceError {
                cause: MetalDeviceCause::Invalid,
                owner,
            });
        }
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: matching allocation, initialized before Box takes ownership.
        let pointer = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if pointer.is_null() {
            return Err(MetalDeviceError {
                cause: MetalDeviceCause::AllocationFailed,
                owner,
            });
        }
        let node = unsafe {
            pointer.write(OwnedNode {
                retired: RetiredOwner {
                    next: std::ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                owner,
            });
            Box::from_raw(pointer)
        };
        Ok(Self {
            layout,
            node: Some(node),
        })
    }
    /// Borrow the original custody retained by this prepared node.
    pub fn owner(&self) -> &T {
        &self.node.as_ref().expect("live Device initializer").owner
    }
    /// Release the unused node's storage before returning its original custody.
    pub fn into_owner(mut self) -> T {
        take_owner(self.node.take().expect("live Device initializer"))
    }
    /// Initialize the singleton, transferring custody only after native success.
    /// A refusal returns this same prepared node for recovery or a later attempt.
    pub fn try_initialize(mut self) -> Result<InitializedMetalDevice, MetalDeviceError<Self>> {
        let mut identity = 0;
        let status;
        {
            let Some(_loan) = runtime_lock::try_enter_for_recovery() else {
                return Err(MetalDeviceError {
                    cause: MetalDeviceCause::Busy,
                    owner: self,
                });
            };
            let owner = (&mut **self.node.as_mut().expect("live Device initializer")
                as *mut OwnedNode<T>)
                .cast();
            // SAFETY: source is static; every refusal leaves node/output intact.
            status = unsafe {
                safemlx_sys::mlx_device_initialize(
                    &mut identity,
                    source(),
                    self.layout.native,
                    owner,
                    Some(retire),
                )
            };
            if status == 0 {
                let _ = Box::into_raw(self.node.take().expect("live Device initializer"));
            }
        }
        if status == 0 {
            Ok(InitializedMetalDevice { identity })
        } else {
            Err(MetalDeviceError {
                cause: cause(status),
                owner: self,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    };
    #[derive(Debug)]
    struct Owner(Arc<AtomicUsize>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn device_initializer_busy_preserves_same_prepared_node_or_typed_unknown() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let owner = Owner(dropped.clone());
        let layout = PreparedMetalDevice::<Owner>::layout();
        if std::env::var_os("EREDU_REQUIRE_METAL_DEVICE_INITIALIZATION_QUALIFICATION").is_some() {
            assert!(layout.is_ok(), "{layout:?}");
        }
        if matches!(
            layout,
            Err(MetalDeviceCause::UnknownLayout | MetalDeviceCause::UnqualifiedSource)
        ) {
            let error = PreparedMetalDevice::try_new(owner).unwrap_err();
            assert!(matches!(
                error.cause(),
                MetalDeviceCause::UnknownLayout | MetalDeviceCause::UnqualifiedSource
            ));
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
            drop(error);
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
            return;
        }
        let prepared = PreparedMetalDevice::try_new(owner).unwrap();
        let pointer = &**prepared.node.as_ref().unwrap() as *const OwnedNode<Owner>;
        let (entered, wait_entered) = mpsc::channel();
        let (release, wait_release) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _loan = runtime_lock::enter();
            entered.send(()).unwrap();
            let _ = wait_release.recv();
        });
        wait_entered.recv().unwrap();
        let result = prepared.try_initialize();
        release.send(()).unwrap();
        worker.join().unwrap();
        let error = result.unwrap_err();
        assert_eq!(error.cause(), MetalDeviceCause::Busy);
        assert_eq!(
            &**error.owner().node.as_ref().unwrap() as *const OwnedNode<Owner>,
            pointer
        );
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        let (_, prepared) = error.into_parts();
        drop(prepared.into_owner());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
