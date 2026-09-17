//! Finite row construction using the entered, explicitly authenticated Graph.
use crate::{
    Array, OriginalScopeObserver, Stream,
    error::{Exception, Result},
    utils::{
        guard::{Guard, Guarded, MaybeUninitArray},
        runtime_lock,
    },
};
use std::{mem::size_of, ptr};

/// Cold layout of the actual native row bank. This neither creates a bank nor
/// grants access to a Graph quota. The original constructor rechecks its role.
#[derive(Clone, Copy, Debug)]
pub struct OriginalArrayRowsLayout {
    native: safemlx_sys::mlx_graph_array_rows_layout,
}
impl OriginalArrayRowsLayout {
    /// Inspect the compiled native allocator/vector layout without touching TLS.
    pub fn inspect(capacity: usize, maximum_rank: usize) -> Option<Self> {
        let mut native = safemlx_sys::mlx_graph_array_rows_layout::default();
        // SAFETY: only this scalar output is written, and only on success.
        unsafe { safemlx_sys::mlx_graph_array_rows_inspect(&mut native, capacity, maximum_rank) }
            .then_some(Self { native })
    }
    /// Additional resident constructor envelopes for the actual row header,
    /// element backing and two borrowed-prefix Shape copies. Their unused
    /// descriptor slots are conservative storage, not additional operations.
    pub fn resident_controls(self) -> usize {
        self.native.resident_controls
    }
    /// Largest actual concatenate input population, including the fixed floor.
    pub fn maximum_operands(self) -> usize {
        self.native.maximum_operands
    }
    /// Named native/Rust controls, excluding Graph-paid header and backing.
    pub fn control_bytes(self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<OriginalArrayRows<'static>>(),
            size_of::<Result<OriginalArrayRows<'static>>>(),
            size_of::<Option<Self>>(),
            size_of::<MaybeUninitArray>(),
            size_of::<Result<Array>>(),
            size_of::<Result<()>>(),
            size_of::<Exception>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<&OriginalScopeObserver>(),
            size_of::<&Array>(),
            size_of::<&Stream>(),
            size_of::<safemlx_sys::mlx_graph_array_rows>(),
            2 * size_of::<safemlx_sys::mlx_array>(),
            size_of::<safemlx_sys::mlx_stream>(),
            size_of::<u32>(),
            size_of::<i32>(),
            size_of::<usize>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
}

/// A construction-only native row bank. Both its header and elements retain
/// the accepted Graph allocator through destruction. It borrows the exact role,
/// cannot grow, and concatenates only once. It creates no completion or grant.
pub struct OriginalArrayRows<'a> {
    raw: safemlx_sys::mlx_graph_array_rows,
    observer: &'a OriginalScopeObserver,
}
impl std::fmt::Debug for OriginalArrayRows<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalArrayRows").finish_non_exhaustive()
    }
}
impl<'a> OriginalArrayRows<'a> {
    /// Allocate the inspected fixed capacity from this current role's Graph.
    pub fn new(
        layout: OriginalArrayRowsLayout,
        observer: &'a OriginalScopeObserver,
    ) -> Result<Self> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        let mut raw = safemlx_sys::mlx_graph_array_rows {
            ctx: ptr::null_mut(),
        };
        // SAFETY: native validates the borrowed observer before allocation. It
        // publishes only a complete owner; every refused prefix is destroyed.
        let status = unsafe {
            safemlx_sys::mlx_graph_array_rows_new(&mut raw, observer.raw, layout.native.capacity)
        };
        if status != 0 {
            return Err(observer.error(status));
        }
        if raw.ctx.is_null() {
            return Err(observer.invalid_input_error());
        }
        Ok(Self { raw, observer })
    }
    /// Append one borrowed native array handle without allocating a C wrapper
    /// or growing the fixed backing. Overflow is a fixed refusal.
    pub fn push(&mut self, input: &Array) -> Result<()> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(self.observer.error(10));
        };
        // SAFETY: the bank and input remain live; native authenticates the same
        // current role and checks the remaining capacity before mutation.
        self.check(unsafe {
            safemlx_sys::mlx_graph_array_rows_push(self.raw, self.observer.raw, input.as_ptr())
        })
    }
    /// Consume every declared row through the ordinary native concatenate
    /// equation on the explicitly supplied, authenticated stream.
    pub fn concatenate(self, axis: i32, stream: &Stream) -> Result<Array> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(self.observer.error(10));
        };
        let mut output = MaybeUninitArray::new();
        // SAFETY: output starts empty; native checks fullness and role/stream,
        // moves the input vector once, and publishes the ordinary result.
        self.check(unsafe {
            safemlx_sys::mlx_graph_array_rows_concatenate(
                output.as_mut_raw_ptr(),
                self.raw,
                self.observer.raw,
                axis,
                stream.as_ptr(),
            )
        })?;
        output.set_init_success(true);
        output.try_into_guarded()
    }
    fn check(&self, status: u32) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(self.observer.error(status))
        }
    }
}
impl Drop for OriginalArrayRows<'_> {
    fn drop(&mut self) {
        // SAFETY: unique bank owner; like Array::drop this only releases graph
        // aliases. Native retains the Graph through backing and header free.
        unsafe { safemlx_sys::mlx_graph_array_rows_free(self.raw) };
    }
}

/// Reshape to an existing array's prefix with its last dimension replaced.
/// This shares the ordinary reshape constructor and native Shape allocator;
/// no Rust dimension vector or copied source tensor is constructed.
pub fn reshape_like_prefix(
    input: &Array,
    source: &Array,
    final_dimension: i32,
    stream: &Stream,
) -> Result<Array> {
    Array::try_from_op(|output| unsafe {
        safemlx_sys::mlx_reshape_like_prefix(
            output,
            input.as_ptr(),
            source.as_ptr(),
            final_dimension,
            stream.as_ptr(),
        )
    })
}

/// Rank-dependent Graph metadata for the existing reshape and concatenate
/// workers (including their cast candidates). Descriptor, Data and execution
/// resource populations remain in the ordinary GPU worker layout.
#[derive(Clone, Copy, Debug)]
pub struct OriginalCopyWorkerLayout {
    native: safemlx_sys::mlx_graph_copy_worker_layout,
}
impl OriginalCopyWorkerLayout {
    /// Inspect actual source-rank SmallVector/collapse allocation requests.
    /// This creates no owner, native context or allocation authority.
    pub fn inspect(
        maximum_rank: usize,
        reshapes: usize,
        concatenations: usize,
        concatenate_inputs: usize,
    ) -> Option<Self> {
        let mut native = safemlx_sys::mlx_graph_copy_worker_layout::default();
        // SAFETY: native reads scalar geometry and publishes only this output.
        unsafe {
            safemlx_sys::mlx_graph_copy_worker_inspect(
                &mut native,
                maximum_rank,
                reshapes,
                concatenations,
                concatenate_inputs,
            )
        }
        .then_some(Self { native })
    }
    /// One actual byte reinterpretation, including the same GPU general-copy
    /// branch for noncontiguous input. Dtype/shape/source validation and the
    /// descriptor, backing and completion populations are separate producers.
    pub fn inspect_byte_view(maximum_rank: usize) -> Option<Self> {
        let mut native = safemlx_sys::mlx_graph_copy_worker_layout::default();
        // SAFETY: pure scalar source query writes only the supplied layout.
        unsafe { safemlx_sys::mlx_graph_byte_view_worker_inspect(&mut native, maximum_rank) }
            .then_some(Self { native })
    }
    /// Sum of Graph allocation extents, retaining every possible growth prefix.
    pub fn allocation_extents(self) -> usize {
        self.native.allocation_extents
    }
    /// Actual native and safe-query controls, separately from Graph allocations.
    pub fn control_bytes(self) -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            4 * size_of::<usize>(),
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
}

/// Cold host controls of the shared borrowed-prefix reshape constructor.
/// Its native Shape backing is priced by the caller's source-rank Graph layout.
pub fn reshape_like_prefix_control_bytes() -> Option<usize> {
    let mut native = 0;
    // SAFETY: the native query only writes its fixed constructor frame size.
    if !unsafe { safemlx_sys::mlx_reshape_like_prefix_control_bytes(&mut native) } {
        return None;
    }
    [
        size_of::<Array>(),
        size_of::<MaybeUninitArray>(),
        size_of::<Result<Array>>(),
        size_of::<Exception>(),
        size_of::<runtime_lock::RuntimeLockGuard>(),
        size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
        2 * size_of::<&Array>(),
        size_of::<&Stream>(),
        size_of::<i32>(),
        size_of::<OriginalScopeObserver>(),
        size_of::<Option<OriginalScopeObserver>>(),
        size_of::<Result<Option<OriginalScopeObserver>>>(),
    ]
    .into_iter()
    .try_fold(native, usize::checked_add)
}
