//! Fixed-status construction of source-owned, completed U32/I32/F32/Bool native leaves.
//! The explicit runtime preparation is ordinary; construction never initializes
//! an allocator, enters housekeeping, evaluates, submits, or attaches a grant.
use crate::{
    error::{self, Exception},
    utils::{guard::Guarded, runtime_lock},
    Array, PreparedSubmissionGraphQuota, SubmissionGraphQuota, SubmissionGraphQuotaCause,
    SubmissionGraphQuotaError, SubmissionGraphQuotaLayout,
};
use std::{
    marker::PhantomData,
    mem, ptr,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    },
};

struct AllocatorPlacementSnapshot {
    placement: OnceLock<crate::AllocationPlacement>,
    inconsistent: AtomicBool,
}
impl AllocatorPlacementSnapshot {
    const fn new() -> Self {
        Self {
            placement: OnceLock::new(),
            inconsistent: AtomicBool::new(false),
        }
    }
    fn publish(&self, placement: crate::AllocationPlacement) {
        if *self.placement.get_or_init(|| placement) != placement {
            self.inconsistent.store(true, Ordering::Release);
        }
    }
    fn get(&self) -> Option<crate::AllocationPlacement> {
        let placement = *self.placement.get()?;
        (!self.inconsistent.load(Ordering::Acquire)
            && placement != crate::AllocationPlacement::Unknown)
            .then_some(placement)
    }
}
static ALLOCATOR_PLACEMENT: AllocatorPlacementSnapshot = AllocatorPlacementSnapshot::new();
pub(crate) const fn placement_static_storage_bytes() -> usize {
    mem::size_of::<AllocatorPlacementSnapshot>()
}

/// Fixed refusal without formatting, hidden retry or source-owner transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PreparedInputCause {
    #[error("native prepared input realization is unavailable")]
    /// No recipe for the actual native allocator or ABI.
    Unsupported,
    #[error("invalid native prepared input shape or layout")]
    /// Invalid source count, shape or native representation.
    Invalid,
    #[error("native prepared input runtime is busy")]
    /// The existing native runtime or allocator loan is unavailable.
    RuntimeBusy,
    #[error("native prepared input allocation failed")]
    /// The actual allocation attempt failed.
    AllocationFailed,
    #[error("native prepared input capacity exhausted")]
    /// Original metadata or resource capacity is insufficient.
    Capacity,
    #[error("native prepared input identity supply exhausted")]
    /// Nonrepeating native allocation generations are exhausted.
    IdentityExhausted,
}
fn cause(code: u32) -> PreparedInputCause {
    match code {
        1 => PreparedInputCause::Unsupported,
        3 => PreparedInputCause::RuntimeBusy,
        4 => PreparedInputCause::AllocationFailed,
        5 => PreparedInputCause::Capacity,
        6 => PreparedInputCause::IdentityExhausted,
        _ => PreparedInputCause::Invalid,
    }
}
/// A current-thread ordinary initialization witness. Its native allocator is
/// process-owned. This value exposes no raw context and cannot cross threads.
pub struct PreparedInputRuntime {
    raw: safemlx_sys::mlx_prepared_input_runtime,
    _thread: PhantomData<Rc<()>>,
}
impl std::fmt::Debug for PreparedInputRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedInputRuntime")
            .finish_non_exhaustive()
    }
}
impl PreparedInputRuntime {
    /// Copy this current-thread initialized allocator witness for retained cold
    /// inspection. The allocator is process-owned; this copies only its exact
    /// immutable scalar source. It does not initialize, allocate, reserve, or
    /// grant submission authority, and remains unable to cross threads.
    pub fn inspection_alias(&self) -> Self {
        Self {
            raw: self.raw,
            _thread: PhantomData,
        }
    }
    /// Named source/alias transport controls, with no dynamic destination.
    pub const fn inspection_alias_control_bytes() -> usize {
        std::mem::size_of::<&Self>() + std::mem::size_of::<Self>()
    }
    pub(crate) fn from_initialized(raw: safemlx_sys::mlx_prepared_input_runtime) -> Self {
        ALLOCATOR_PLACEMENT.publish(crate::AllocationPlacement::from_native(raw.placement));
        Self {
            raw,
            _thread: PhantomData,
        }
    }

    /// Placement witnessed by successful preparation of the actual allocator.
    /// This cold query initializes nothing and grants no execution authority.
    /// It remains unavailable before preparation, for unknown placement, or if
    /// later preparation contradicts the process-owned immutable strategy.
    pub fn established_allocation_placement() -> Option<crate::AllocationPlacement> {
        ALLOCATOR_PLACEMENT.get()
    }

    /// Physical placement selected by the actual prepared allocator mechanism.
    pub fn allocation_placement(&self) -> crate::AllocationPlacement {
        crate::AllocationPlacement::from_native(self.raw.placement)
    }

    pub(crate) fn raw(&self) -> safemlx_sys::mlx_prepared_input_runtime {
        self.raw
    }

    /// Ordinary initialization of the actual allocator/mutex/page-size state,
    /// native error handler and this thread's runtime lock. It may allocate and
    /// run ordinary housekeeping. Complete it before original source admission.
    pub fn prepare() -> Result<Self, Exception> {
        let _guard = runtime_lock::enter();
        error::ensure_mlx_error_handler();
        let mut raw = safemlx_sys::mlx_prepared_input_runtime {
            allocator: ptr::null_mut(),
            page_size: 0,
            maximum: 0,
            storage_kind: 0,
            controls: 0,
            placement: safemlx_sys::mlx_memory_placement {
                kind: 0,
                device: -1,
                device_count: 0,
            },
        };
        let status = unsafe { safemlx_sys::mlx_prepared_input_runtime_prepare(&mut raw) };
        match status {
            0 => Ok(Self::from_initialized(raw)),
            2 => Err(error::get_and_clear_last_mlx_error()
                .expect("ordinary native input initialization error")
                .into()),
            _ => Err(Exception::custom(
                "native prepared input realization is unavailable",
            )),
        }
    }
    fn plan<'a>(
        &'a self,
        data: *const std::ffi::c_void,
        elements: usize,
        shape: &'a [usize],
        kind: u32,
    ) -> Result<PreparedInputPlan<'a>, PreparedInputCause> {
        let source = safemlx_sys::mlx_prepared_input_source {
            data,
            shape: shape.as_ptr(),
            rank: shape.len(),
            elements,
            kind,
        };
        let layout = self.source_layout(source)?;
        Ok(PreparedInputPlan {
            runtime: self,
            source,
            layout,
            _source: PhantomData,
        })
    }
    fn source_layout(
        &self,
        source: safemlx_sys::mlx_prepared_input_source,
    ) -> Result<PreparedInputLayout, PreparedInputCause> {
        let mut native = safemlx_sys::mlx_prepared_input_layout {
            metadata_bytes: 0,
            backing_bytes: 0,
            controls: 0,
        };
        // SAFETY: the borrowed shape is live for this synchronous query. The
        // native layout worker validates shape/count/kind and allocator facts;
        // it neither dereferences source.data nor retains any source pointer.
        let status =
            unsafe { safemlx_sys::mlx_prepared_input_layout_for(&mut native, self.raw, source) };
        if status != 0 {
            return Err(PreparedInputCause::Invalid);
        }
        let controls = native
            .controls
            .checked_add(mem::size_of::<PreparedInputPlan<'_>>())
            .and_then(|n| n.checked_add(mem::size_of::<PreparedInputLeaf>()))
            .and_then(|n| {
                n.checked_add(mem::size_of::<Result<PreparedInputLeaf, PreparedInputCause>>())
            })
            .and_then(|n| n.checked_add(mem::size_of::<runtime_lock::RuntimeLockGuard>()))
            .ok_or(PreparedInputCause::Invalid)?;
        Ok(PreparedInputLayout {
            metadata_bytes: native.metadata_bytes,
            backing_bytes: native.backing_bytes,
            controls,
        })
    }
    /// Describe the actual copied-U32 leaf producer before values are present.
    /// This calls the same layout worker as `u32`, without creating a plan,
    /// reserving storage, reading values, or granting construction authority.
    pub fn u32_layout(&self, shape: &[usize]) -> Result<PreparedInputLayout, PreparedInputCause> {
        let elements = shape
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
            .ok_or(PreparedInputCause::Invalid)?;
        self.source_layout(safemlx_sys::mlx_prepared_input_source {
            data: ptr::null(),
            shape: shape.as_ptr(),
            rank: shape.len(),
            elements,
            kind: 0,
        })
    }
    /// Fixed Rust query representations for `u32_layout`. The returned layout's
    /// constructor controls belong to later leaf construction, not this query.
    pub const fn u32_layout_control_bytes() -> usize {
        mem::size_of::<(&Self, &[usize])>()
            + mem::size_of::<usize>() * 3
            + mem::size_of::<safemlx_sys::mlx_prepared_input_source>()
            + mem::size_of::<safemlx_sys::mlx_prepared_input_layout>()
            + mem::size_of::<PreparedInputLayout>()
            + mem::size_of::<Result<PreparedInputLayout, PreparedInputCause>>()
    }
    /// Exact initialized-zero leaf using the same prepared allocator and
    /// descriptor producer. No lazy operation, evaluation or host value vector.
    /// Shape and physical dtype remain explicit, checked source geometry.
    pub fn zeros<'a>(
        &'a self,
        dtype: crate::Dtype,
        shape: &'a [usize],
    ) -> Result<PreparedInputPlan<'a>, PreparedInputCause> {
        let kind = match dtype {
            crate::Dtype::Float16 => 4,
            crate::Dtype::Bfloat16 => 5,
            crate::Dtype::Float32 => 6,
            crate::Dtype::Int32 => 7,
            crate::Dtype::Uint32 => 8,
            crate::Dtype::Bool => 9,
            _ => return Err(PreparedInputCause::Unsupported),
        };
        let elements = shape
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
            .ok_or(PreparedInputCause::Invalid)?;
        self.plan(ptr::null(), elements, shape, kind)
    }
    /// Controls of the zero-source planning call, before the plan is available.
    pub const fn zeros_plan_control_bytes() -> usize {
        mem::size_of::<(&Self, crate::Dtype, &[usize])>()
            + mem::size_of::<u32>()
            + mem::size_of::<usize>() * 3
            + mem::size_of::<PreparedInputPlan<'_>>()
            + mem::size_of::<Result<PreparedInputPlan<'_>, PreparedInputCause>>()
    }
    /// Checked borrowed U32 source recipe; empty shape requires exactly one value.
    pub fn u32<'a>(
        &'a self,
        values: &'a [u32],
        shape: &'a [usize],
    ) -> Result<PreparedInputPlan<'a>, PreparedInputCause> {
        self.plan(values.as_ptr().cast(), values.len(), shape, 0)
    }
    /// Checked borrowed I32 source recipe; empty shape requires exactly one value.
    pub fn i32<'a>(
        &'a self,
        values: &'a [i32],
        shape: &'a [usize],
    ) -> Result<PreparedInputPlan<'a>, PreparedInputCause> {
        self.plan(values.as_ptr().cast(), values.len(), shape, 1)
    }
    /// Checked borrowed Boolean source; the existing native producer copies
    /// canonical Rust Boolean bytes into its ordinary Boolean array storage.
    pub fn boolean<'a>(
        &'a self,
        values: &'a [bool],
        shape: &'a [usize],
    ) -> Result<PreparedInputPlan<'a>, PreparedInputCause> {
        const {
            assert!(mem::size_of::<bool>() == 1);
        }
        self.plan(values.as_ptr().cast(), values.len(), shape, 3)
    }
    /// Checked borrowed F32 source recipe; empty shape requires exactly one value.
    pub fn f32<'a>(
        &'a self,
        values: &'a [f32],
        shape: &'a [usize],
    ) -> Result<PreparedInputPlan<'a>, PreparedInputCause> {
        self.plan(values.as_ptr().cast(), values.len(), shape, 2)
    }
}
/// Allocation facts of the selected prepared input producer. This value owns
/// no allocator, data, arena, source pointers or construction capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedInputLayout {
    metadata_bytes: usize,
    backing_bytes: usize,
    controls: usize,
}
impl PreparedInputLayout {
    /// Source arena capacity, including its exact allocation headers.
    pub const fn metadata_bytes(self) -> usize {
        self.metadata_bytes
    }
    /// Complete page-rounded backing allocation from the selected allocator.
    pub const fn backing_bytes(self) -> usize {
        self.backing_bytes
    }
    /// Native and Rust constructor controls for the actual borrowed leaf plan.
    pub const fn control_bytes(self) -> usize {
        self.controls
    }
}
/// Borrowed exact source and native recipe. No raw array or independent bytes
/// can create this plan; every shape/count product is checked without allocation.
#[derive(Debug)]
pub struct PreparedInputPlan<'a> {
    runtime: &'a PreparedInputRuntime,
    source: safemlx_sys::mlx_prepared_input_source,
    layout: PreparedInputLayout,
    _source: PhantomData<&'a [u8]>,
}
impl PreparedInputPlan<'_> {
    /// Source arena block capacity, including exact allocator headers.
    pub fn metadata_bytes(&self) -> usize {
        self.layout.metadata_bytes()
    }
    /// Complete page-rounded native backing allocation.
    pub fn backing_bytes(&self) -> usize {
        self.layout.backing_bytes()
    }
    /// Named native and Rust constructor/control representations.
    pub fn control_bytes(&self) -> usize {
        self.layout.control_bytes()
    }
    /// Descriptive facts from the same validated borrowed source plan.
    pub fn layout(&self) -> PreparedInputLayout {
        self.layout
    }
    /// Constructs the exact borrowed values in this source-only arena. Refusal
    /// leaves that arena and all earlier leaves with the caller. No retry grant.
    pub fn construct(
        self,
        arena: &PreparedInputArena,
    ) -> Result<PreparedInputLeaf, PreparedInputCause> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PreparedInputCause::RuntimeBusy);
        };
        let mut raw = safemlx_sys::mlx_prepared_input_leaf {
            ctx: ptr::null_mut(),
        };
        let status = unsafe {
            safemlx_sys::mlx_prepared_input_leaf_new(
                &mut raw,
                self.runtime.raw,
                arena.quota.raw(),
                self.source,
            )
        };
        if status == 0 {
            Ok(PreparedInputLeaf {
                raw,
                _thread: PhantomData,
            })
        } else {
            Err(cause(status))
        }
    }
}
/// Source-private metadata resource. It exposes no Scope or general allocator.
pub struct PreparedInputArena {
    quota: SubmissionGraphQuota,
}
impl std::fmt::Debug for PreparedInputArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedInputArena").finish_non_exhaustive()
    }
}
impl PreparedInputArena {
    pub(crate) fn raw(&self) -> safemlx_sys::mlx_submission_graph_quota {
        self.quota.raw()
    }

    /// Actual native resource, Rust retirement node and construction controls.
    pub fn layout<T: Send + 'static>(
        capacity: usize,
    ) -> Result<SubmissionGraphQuotaLayout, SubmissionGraphQuotaCause> {
        let mut layout = PreparedSubmissionGraphQuota::<T>::layout(capacity)?;
        layout.control_bytes = layout
            .control_bytes
            .checked_add(mem::size_of::<Self>())
            .ok_or(SubmissionGraphQuotaCause::InvalidCapacity)?;
        Ok(layout)
    }
    /// Consumes an existing original preparation without housekeeping.
    pub fn try_allocate<T: Send + 'static>(
        prepared: PreparedSubmissionGraphQuota<T>,
    ) -> Result<Self, SubmissionGraphQuotaError<PreparedSubmissionGraphQuota<T>>> {
        prepared.try_allocate().map(|quota| Self { quota })
    }
}
/// A real completed input leaf with no Clone/raw/Weak export. Explicit ordinary
/// C-wrapper cloning shares its completed data; those aliases can outlive this
/// leaf because Data/control/descriptor blocks pin the source arena.
pub struct PreparedInputLeaf {
    raw: safemlx_sys::mlx_prepared_input_leaf,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for PreparedInputLeaf {
    fn drop(&mut self) {
        unsafe { safemlx_sys::mlx_prepared_input_leaf_free(self.raw) };
    }
}
impl std::fmt::Debug for PreparedInputLeaf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedInputLeaf").finish_non_exhaustive()
    }
}
/// Exact source-arena wrapper allocation and fixed construction controls.
/// The enclosing source recipe includes one instance for each constructed view.
#[derive(Clone, Copy, Debug)]
pub struct PreparedInputArrayLayout {
    metadata_bytes: usize,
    control_bytes: usize,
}
impl PreparedInputArrayLayout {
    /// Source-arena capacity including the real block/alignment overhead.
    pub fn metadata_bytes(self) -> usize {
        self.metadata_bytes
    }
    /// Concrete C/Rust constructor, return and retirement controls.
    pub fn control_bytes(self) -> usize {
        self.control_bytes
    }
}
impl PreparedInputLeaf {
    /// Pure concrete wrapper layout; no allocator/runtime initialization.
    pub fn array_layout() -> Result<PreparedInputArrayLayout, PreparedInputCause> {
        let mut metadata_bytes = 0;
        let mut control_bytes = 0;
        let status = unsafe {
            safemlx_sys::mlx_prepared_input_array_layout(&mut metadata_bytes, &mut control_bytes)
        };
        if status != 0 {
            return Err(PreparedInputCause::Unsupported);
        }
        let control_bytes = [
            mem::size_of::<PreparedInputArrayLayout>(),
            mem::size_of::<Array>(),
            mem::size_of::<Result<Array, PreparedInputCause>>(),
            mem::size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            mem::size_of::<u32>(),
        ]
        .into_iter()
        .try_fold(control_bytes, usize::checked_add)
        .ok_or(PreparedInputCause::Invalid)?;
        Ok(PreparedInputArrayLayout {
            metadata_bytes,
            control_bytes,
        })
    }
    /// Constructs a real wrapper from this leaf's already admitted source arena.
    /// No data copy, ordinary allocator, housekeeping, eval or error formatting
    /// occurs. The wrapper's closed native disposition retains that arena until
    /// its own value and allocation have both retired. Capacity refusal never
    /// falls back to ordinary cloning and grants no additional source authority.
    pub fn try_source_array(&self) -> Result<Array, PreparedInputCause> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PreparedInputCause::RuntimeBusy);
        };
        let mut raw = safemlx_sys::mlx_array {
            ctx: ptr::null_mut(),
            prepared_owner: ptr::null_mut(),
        };
        let status = unsafe { safemlx_sys::mlx_prepared_input_leaf_array(&mut raw, self.raw) };
        if status == 0 {
            // SAFETY: the closed native factory has published exactly one owned
            // C handle; mlx_array_free dispatches its actual source disposition.
            Ok(unsafe { Array::from_ptr(raw) })
        } else {
            debug_assert!(raw.ctx.is_null() && raw.prepared_owner.is_null());
            Err(cause(status))
        }
    }
    /// Ordinary C wrapper allocation sharing completed backing. Callers retain
    /// their actual ordinary operation owner; this grants no original control.
    pub fn try_clone_array(&self) -> Result<Array, Exception> {
        let _guard = runtime_lock::enter();
        Array::try_from_op(|out| unsafe {
            safemlx_sys::mlx_prepared_input_leaf_clone_array(out, self.raw)
        })
    }
    /// Immutable allocation facts; no evaluation, housekeeping or source authority.
    pub fn allocation_info(&self) -> Result<crate::AllocationInfo, PreparedInputCause> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(PreparedInputCause::RuntimeBusy);
        };
        let mut identity = 0;
        let mut bytes = 0;
        let status = unsafe {
            safemlx_sys::mlx_prepared_input_leaf_info(&mut identity, &mut bytes, self.raw)
        };
        if status != 0 {
            return Err(PreparedInputCause::Invalid);
        }
        let mut placement = safemlx_sys::mlx_memory_placement {
            kind: 0,
            device: -1,
            device_count: 0,
        };
        if unsafe { safemlx_sys::mlx_prepared_input_leaf_placement(&mut placement, self.raw) } != 0
        {
            return Err(PreparedInputCause::Invalid);
        }
        Ok(crate::AllocationInfo::from_native(
            identity, bytes, placement,
        ))
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests;

#[cfg(test)]
mod placement_snapshot_tests {
    use super::*;

    #[test]
    fn prepared_source_placement_is_independent_of_ordinary_managed_candidates() {
        let source = AllocatorPlacementSnapshot::new();
        assert_eq!(source.get(), None);
        let ordinary = crate::AllocationPlacement::CudaManaged { device_count: 2 };
        source.publish(crate::AllocationPlacement::Host);
        assert_eq!(source.get(), Some(crate::AllocationPlacement::Host));
        assert_ne!(source.get(), Some(ordinary));
        source.publish(crate::AllocationPlacement::Host);
        assert_eq!(source.get(), Some(crate::AllocationPlacement::Host));
        source.publish(ordinary);
        assert_eq!(source.get(), None);
        source.publish(crate::AllocationPlacement::Host);
        assert_eq!(source.get(), None);
    }
}
