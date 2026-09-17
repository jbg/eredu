//! Prepaid immutable copies of existing stream values; no worker/queue creation.
use super::{destroy, retire, take_owner, OwnedNode, RetiredOwner};
use crate::Stream;
use std::{
    alloc::Layout,
    ffi::c_void,
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ptr,
    sync::Arc,
};

/// Explicit implementation copied into CPU Matmul/AddMM primitives.
/// Choosing a kernel provides no model, allocation, source or submission grant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuMatmulKernel {
    /// The ordinary platform selection, including BLAS where applicable.
    PlatformDefault,
    /// The bundled fixed 16x16 F32 tile worker and its declared SIMD reduction.
    Float32Tiles,
    /// Bundled F32 and F16 inputs, both using F32 tile accumulation.
    Float32AndFloat16Tiles,
}
/// Pure source facts for bundled CPU tiles with F32 accumulation.
#[derive(Clone, Copy, Debug)]
pub struct CpuMatmulFacts(safemlx_sys::mlx_cpu_matmul_facts);
impl CpuMatmulFacts {
    /// Inspect the compiled worker without creating a runtime or native context.
    pub fn inspect() -> Option<Self> {
        let mut source = safemlx_sys::mlx_cpu_matmul_facts::default();
        // SAFETY: pure fixed-output source query; no pointer is retained.
        (unsafe { safemlx_sys::mlx_cpu_matmul_facts_for(&mut source) } == 0).then_some(Self(source))
    }
    /// Whether the compiled shared tile worker supports F16 inputs.
    pub fn float16_tiles(self) -> bool { self.0.float16_tiles }
    /// Whether ordinary platform F16 already selects that exact SIMD worker.
    pub fn platform_float16_tiles(self) -> bool { self.0.platform_float16_tiles }
    /// Square accumulation tile edge.
    pub fn tile_edge(self) -> usize { self.0.tile_edge }
    /// Actual number of values in each SIMD partial reduction.
    pub fn reduction_lanes(self) -> usize { self.0.reduction_lanes }
    /// Maximum normalized matrix rank covered by this source.
    pub fn max_rank(self) -> usize { self.0.max_rank }
    /// Checked signed element-index domain of the existing worker.
    pub fn max_elements(self) -> usize { self.0.max_elements }
    /// Source inspection and fixed selection controls, without wrapper storage.
    pub fn control_bytes(self) -> Option<usize> {
        let parts=[mem::size_of::<Self>(),mem::size_of::<Option<Self>>(),mem::size_of::<CpuMatmulKernel>(),
            mem::size_of::<safemlx_sys::mlx_cpu_matmul_facts>()];
        parts.into_iter().try_fold(self.0.controls.checked_add(mem::size_of_val(&parts))?,usize::checked_add)
    }
}

/// A fixed copy/preparation refusal, without native error allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum StreamCopyCause {
    /// The existing stream or native layout was invalid.
    #[error("invalid prepared stream copy")]
    Invalid,
    /// The exact wrapper or source node could not be allocated.
    #[error("prepared stream copy allocation failed")]
    AllocationFailed,
    /// A checked layout sum overflowed.
    #[error("prepared stream copy layout overflow")]
    Overflow,
}
/// Fixed cause followed by the exact untransferred owner.
pub struct StreamCopyError<T> {
    cause: StreamCopyCause,
    owner: T,
}
impl<T> StreamCopyError<T> {
    /// The original fixed refusal.
    pub fn cause(&self) -> StreamCopyCause {
        self.cause
    }
    /// The original untransferred custody.
    pub fn owner(&self) -> &T {
        &self.owner
    }
    /// Recover the cause and intact owner.
    pub fn into_parts(self) -> (StreamCopyCause, T) {
        (self.cause, self.owner)
    }
}
impl<T> fmt::Debug for StreamCopyError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamCopyError")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for StreamCopyError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<T> std::error::Error for StreamCopyError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
struct Body<T> {
    stream: ManuallyDrop<Stream>,
    owner: Option<Box<OwnedNode<T>>>,
}
// SAFETY: this wrapper's native Stream value is immutable. Safe Stream methods
// only read that value; runtime effects keep their existing runtime guard. T is
// read-only until final exclusive retirement. No mutable Stream/raw Arc escapes.
unsafe impl<T: Send + Sync> Send for Body<T> {}
unsafe impl<T: Send + Sync> Sync for Body<T> {}
impl<T> Drop for Body<T> {
    fn drop(&mut self) {
        // SAFETY: exact copy_new allocation, final exclusive body after Arc's
        // no-Weak into_inner freed its control. Scalar deletion has no runtime.
        unsafe { safemlx_sys::mlx_stream_copy_free(self.stream.c_stream) };
        if let Some(node) = self.owner.take() {
            // Queue its preallocated retirement header only after the C wrapper
            // and Arc storage have retired. Arbitrary T::drop stays unlocked.
            unsafe { retire(Box::into_raw(node).cast::<c_void>()) };
        }
    }
}
/// Owning immutable copied stream. Cloning this type allocates nothing.
///
/// Ordinary [`Stream::clone`] remains an independent mutable C wrapper. This
/// separate owner shares only its own explicitly immutable wrapper; no Weak,
/// mutable Stream or bare Arc is exported. Borrowing it grants no queue, worker,
/// submission or original-funding authority. Those resources need separate fits.
/// Final drop only queues custody for reclaim_allocation_owners, so arbitrary
/// destructors never run inside a caller's runtime or cache/manager mutex.
pub struct PreparedStreamCopy<T: Send + Sync + 'static>(Option<Arc<Body<T>>>);
impl<T: Send + Sync + 'static> Clone for PreparedStreamCopy<T> {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live stream copy"))))
    }
}
impl<T: Send + Sync + 'static> Drop for PreparedStreamCopy<T> {
    fn drop(&mut self) {
        // Only this type owns Arc aliases, so the last concurrent into_inner
        // frees the actual shared allocation before Body can retire its source.
        if let Some(body) = self.0.take() {
            drop(Arc::into_inner(body));
        }
    }
}
impl<T: Send + Sync + 'static> fmt::Debug for PreparedStreamCopy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedStreamCopy").finish_non_exhaustive()
    }
}
impl<T: Send + Sync + 'static> PreparedStreamCopy<T> {
    /// Borrow this explicitly immutable value for ordinary native operations.
    ///
    /// Raw interop through the borrowed Stream must not mutate, replace or free
    /// its C wrapper. Copy it with ordinary Stream::clone for an independent
    /// mutable raw wrapper. This borrow cannot outlive this shared owner.
    pub fn as_stream(&self) -> &Stream {
        &self.0.as_ref().expect("live stream copy").stream
    }
    /// Borrow the original account/custody; no new allowance is produced.
    pub fn owner(&self) -> &T {
        &self
            .0
            .as_ref()
            .expect("live stream copy")
            .owner
            .as_ref()
            .expect("live source")
            .owner
    }
    /// Shared wrapper identity, with no raw pointer or mutable handle export.
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live stream copy"),
            other.0.as_ref().expect("live stream copy"),
        )
    }
}
/// Sealed copy recipe captured from an actual borrowed native Stream.
/// It contains only the exact scalar value, never a borrowed raw wrapper. The
/// source may retire after capture; this does not extend its worker/queue life.
pub struct StreamCopyPlan<T: Send + Sync + 'static> {
    value: safemlx_sys::mlx_stream_copy_value,
    native: safemlx_sys::mlx_stream_copy_layout,
    marker: PhantomData<fn() -> T>,
}
impl<T: Send + Sync + 'static> Copy for StreamCopyPlan<T> {}
impl<T: Send + Sync + 'static> Clone for StreamCopyPlan<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Send + Sync + 'static> fmt::Debug for StreamCopyPlan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamCopyPlan")
            .field("native", &self.native)
            .finish_non_exhaustive()
    }
}
impl<T: Send + Sync + 'static> StreamCopyPlan<T> {
    /// Pure scalar snapshot and owning layout query. No allocation, runtime
    /// initialization, housekeeping, lock, native default lookup or clone.
    pub fn capture(source: &Stream) -> Result<Self, StreamCopyCause> {
        let mut value = safemlx_sys::mlx_stream_copy_value::default();
        let mut native = safemlx_sys::mlx_stream_copy_layout::default();
        // SAFETY: the actual source wrapper remains borrowed through its scalar
        // copy. Both C entries write only the local fixed output on success.
        let ok = unsafe {
            safemlx_sys::mlx_stream_copy_snapshot(&mut value, source.c_stream) == 0
                && safemlx_sys::mlx_stream_copy_layout_for(&mut native) == 0
        };
        if !ok {
            return Err(StreamCopyCause::Invalid);
        }
        Ok(Self {
            value,
            native,
            marker: PhantomData,
        })
    }
    /// Compare the exact captured native value with an existing borrowed stream.
    /// This creates no stream, queue, worker, clone or submission authority.
    pub fn matches_source(&self, source: &Stream) -> bool {
        let mut value = safemlx_sys::mlx_stream_copy_value::default();
        // SAFETY: source remains borrowed; the fixed snapshot only reads its value.
        (unsafe { safemlx_sys::mlx_stream_copy_snapshot(&mut value, source.c_stream) == 0 })
            && value.index == self.value.index
            && value.device_index == self.value.device_index
            && value.device_kind == self.value.device_kind
            && value.cpu_matmul == self.value.cpu_matmul
    }
    /// Select a CPU numerical implementation on this immutable copied plan.
    /// The source context is untouched; realization still pays the ordinary
    /// wrapper/retirement owners. GPU contexts refuse instead of changing device.
    pub fn with_cpu_matmul(mut self, kernel: CpuMatmulKernel) -> Result<Self, StreamCopyCause> {
        let kernel=match kernel { CpuMatmulKernel::PlatformDefault=>0,CpuMatmulKernel::Float32Tiles=>1, CpuMatmulKernel::Float32AndFloat16Tiles=>2 };
        // SAFETY: only our private fixed scalar snapshot can be modified. Native
        // validation leaves it unchanged on refusal; no native owner is created.
        if unsafe { safemlx_sys::mlx_stream_copy_select_cpu_matmul(&mut self.value,kernel) } != 0 {
            return Err(StreamCopyCause::Invalid);
        }
        Ok(self)
    }
    /// Exact CPU numerical implementation preserved by this stream copy.
    pub fn cpu_matmul(&self) -> CpuMatmulKernel {
        match self.value.cpu_matmul {
            0=>CpuMatmulKernel::PlatformDefault,1=>CpuMatmulKernel::Float32Tiles,2=>CpuMatmulKernel::Float32AndFloat16Tiles,
            _=>unreachable!("validated native stream kernel"),
        }
    }
    /// Named controls for the allocation-free source comparison above.
    pub fn source_comparison_control_bytes(&self) -> Option<usize> {
        self.native
            .controls
            .checked_add(mem::size_of::<&Self>())?
            .checked_add(mem::size_of::<&Stream>())?
            .checked_add(mem::size_of::<bool>())
    }

    /// Device type recorded by this existing stream's scalar snapshot.
    /// This does not construct a Device or initialize its runtime.
    pub fn device_type(&self) -> crate::DeviceType {
        match self.value.device_kind {
            0 => crate::DeviceType::Cpu,
            1 => crate::DeviceType::Gpu,
            _ => unreachable!("validated native stream snapshot"),
        }
    }
    /// Actual device ordinal from the same existing scalar stream snapshot.
    pub fn device_index(&self) -> i32 { self.value.device_index }

    /// Actual Arc payload layout. The caller must qualify and price its Arc
    /// header/allocation using its host producer contract before realization.
    pub fn shared_body_layout(&self) -> Layout {
        Layout::new::<Body<T>>()
    }
    /// Exact separately allocated Rust retirement node (including T).
    pub fn owner_node_layout(&self) -> Layout {
        Layout::new::<OwnedNode<T>>()
    }
    /// Exact C wrapper request; allocator-private bookkeeping is separate.
    pub fn native_wrapper_bytes(&self) -> usize {
        self.native.wrapper_bytes
    }
    /// Alignment required by the actual C Stream wrapper.
    pub fn native_wrapper_alignment(&self) -> usize {
        self.native.wrapper_alignment
    }
    /// Named capture, constructor, shared clone and final retirement controls.
    pub fn control_bytes(&self) -> Option<usize> {
        Self::control_bytes_for(self.native.controls)
    }
    fn control_bytes_for(native_controls:usize) -> Option<usize> {
        let controls = [
            mem::size_of::<Self>(),
            mem::size_of::<&Stream>(),
            mem::size_of::<Body<T>>(),
            mem::size_of::<Layout>(),
            mem::size_of::<T>(),
            mem::size_of::<Option<Box<OwnedNode<T>>>>(),
            mem::size_of::<*mut OwnedNode<T>>(),
            mem::size_of::<Option<Arc<Body<T>>>>() * 2,
            mem::size_of::<Option<Body<T>>>(),
            mem::size_of::<PreparedStreamCopy<T>>() * 2,
            mem::size_of::<Result<Self, StreamCopyCause>>(),
            mem::size_of::<Result<PreparedStreamCopy<T>, StreamCopyError<T>>>(),
            mem::size_of::<safemlx_sys::mlx_stream>() * 2,
            mem::size_of::<u32>(),
            mem::size_of::<bool>(),
            mem::size_of::<*mut c_void>(),
            mem::size_of::<super::RetirementBatch>(),
            mem::size_of::<*mut RetiredOwner>(),
            mem::size_of::<unsafe fn(*mut RetiredOwner)>(),
        ];
        controls
            .into_iter()
            .try_fold(native_controls, usize::checked_add)?
            .checked_add(mem::size_of_val(&controls))
    }
    /// Pure fixed-layout inspection of this compiled copy worker. These counts
    /// create no plan, stream identity, native context or allocation authority.
    /// Actual realization still requires capture of an existing source stream.
    pub fn capture_control_bytes() -> Result<usize,StreamCopyCause> {
        let native=Self::inspect_copy_layout()?;
        Self::control_bytes_for(native.controls).ok_or(StreamCopyCause::Overflow)
    }
    /// Complete immutable wrapper, shared control and retirement-node storage
    /// used by this exact copy constructor, independent of the stream value.
    pub fn constructor_storage_bytes() -> Result<usize,StreamCopyCause> {
        let native=Self::inspect_copy_layout()?;
        let shared=Layout::new::<[std::sync::atomic::AtomicUsize;2]>()
            .extend(Layout::new::<Body<T>>()).map_err(|_|StreamCopyCause::Overflow)?.0.pad_to_align().size();
        Self::control_bytes_for(native.controls)
            .and_then(|n|n.checked_add(native.wrapper_bytes))
            .and_then(|n|n.checked_add(Layout::new::<OwnedNode<T>>().size()))
            .and_then(|n|n.checked_add(shared)).ok_or(StreamCopyCause::Overflow)
    }
    fn inspect_copy_layout()->Result<safemlx_sys::mlx_stream_copy_layout,StreamCopyCause> {
        let mut native=safemlx_sys::mlx_stream_copy_layout::default();
        // SAFETY: the existing pure layout query writes only this fixed local
        // record and neither reads nor constructs any native stream or owner.
        if unsafe{safemlx_sys::mlx_stream_copy_layout_for(&mut native)}!=0 {
            return Err(StreamCopyCause::Invalid);
        }
        Ok(native)
    }
    /// Allocate this recipe after the caller's exact source comparison. The
    /// supplied owner protects the C block, Rust node and qualified Arc request.
    /// A fixed refusal returns it intact. Arc uses the existing ordinary Rust
    /// global allocation failure policy; no alternate allocator is introduced.
    pub fn realize(self, owner: T) -> Result<PreparedStreamCopy<T>, StreamCopyError<T>> {
        if self.control_bytes().is_none() {
            return Err(StreamCopyError {
                cause: StreamCopyCause::Overflow,
                owner,
            });
        }
        let allocation = Layout::new::<OwnedNode<T>>();
        // SAFETY: exact requested Layout; initialize before adopting as Box.
        let pointer = unsafe { std::alloc::alloc(allocation) }.cast::<OwnedNode<T>>();
        if pointer.is_null() {
            return Err(StreamCopyError {
                cause: StreamCopyCause::AllocationFailed,
                owner,
            });
        }
        let node = unsafe {
            pointer.write(OwnedNode {
                retired: RetiredOwner {
                    next: ptr::null_mut(),
                    destroy: destroy::<T>,
                },
                owner,
            });
            Box::from_raw(pointer)
        };
        let mut raw = safemlx_sys::mlx_stream {
            ctx: ptr::null_mut(),
        };
        // SAFETY: sealed scalar snapshot; empty destination. No runtime effects.
        let status = unsafe { safemlx_sys::mlx_stream_copy_new(&mut raw, self.value) };
        if status != 0 {
            let owner = take_owner(node); // deallocate actual node before custody
            return Err(StreamCopyError {
                cause: if status == 4 {
                    StreamCopyCause::AllocationFailed
                } else {
                    StreamCopyCause::Invalid
                },
                owner,
            });
        }
        Ok(PreparedStreamCopy(Some(Arc::new(Body {
            stream: ManuallyDrop::new(Stream { c_stream: raw }),
            owner: Some(node),
        }))))
    }
}

#[cfg(test)]
mod tests;
