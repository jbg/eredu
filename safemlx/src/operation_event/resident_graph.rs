//! Physical reservation for the closed resident host lowering population.
use super::*;
use std::{ffi::c_void, ptr};

/// Graph-owned descriptor, primitive, input-vector and Array handle envelope.
/// Ordinary axis vectors/sets, error formatting, native backing and worker
/// allocations remain separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentGraphLayout {
    native: safemlx_sys::mlx_resident_graph_layout,
    additional_shells: usize,
}
impl ResidentGraphLayout {
    /// Maximum lowered lazy primitive population, excluding the Eval Synchronizer.
    pub fn primitives(self) -> usize {
        self.native.primitives
    }
    /// Eager scalar/host sources created during this construction prefix.
    pub fn seeds(self) -> usize {
        self.native.seeds
    }
    /// Maximum intermediate rank, including Gather's axis before Squeeze.
    pub fn maximum_rank(self) -> usize {
        self.native.maximum_rank
    }
    /// Largest real operand list, including batched rotary concatenation.
    pub fn maximum_operands(self) -> usize {
        self.native.maximum_operands
    }
    /// Additional semantic or backend handle copies, independent of lazy nodes.
    pub fn additional_shells(self) -> usize {
        self.additional_shells
    }
    /// Requested bytes including the actual Graph-owned header and pointer slots.
    pub fn requested_bytes(self) -> usize {
        self.native.requested_bytes
    }
    /// Maximum sum of block extents; fragmentation still requires physical reservation.
    pub fn allocation_extents(self) -> usize {
        self.native.allocation_extents
    }
    /// Physical constructor block population, excluding header and slots.
    pub fn blocks(self) -> usize {
        self.native.blocks
    }
    /// Actual maximum byte extents/alignment/count for the closed constructor classes.
    pub fn classes(&self) -> impl ExactSizeIterator<Item = (usize, usize, usize)> + '_ {
        self.native
            .request_bytes
            .iter()
            .copied()
            .zip(self.native.request_alignments.iter().copied())
            .zip(self.native.request_counts.iter().copied())
            .map(|((a, b), c)| (a, b, c))
    }
    /// Named C/native/safe construction, destruction and query transports.
    pub fn control_bytes(self) -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_resident_graph_layout>(),
            size_of::<PreparedResidentGraph>(),
            size_of::<Result<PreparedResidentGraph>>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<*mut c_void>(),
            size_of::<u32>(),
            size_of::<bool>(),
            OriginalScopeObserver::control_bytes()?,
        ]
        .into_iter()
        .try_fold(self.native.named_control_bytes, usize::checked_add)
    }
}

/// Native affine graph construction: one primitive and retained fallback,
/// three sibling descriptors and three fixed C result handles. Input custody,
/// physical buffers and Eval storage are separate contributions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineQuantizeConstructionLayout {
    native: safemlx_sys::mlx_affine_quantize_construction_layout,
}
impl AffineQuantizeConstructionLayout {
    /// Graph quota extent bound, including per-block alignment headroom.
    pub fn graph_allocation_extents(self) -> usize { self.native.graph_extents }
    /// Rank used to size the constructor's shape and stride storage.
    pub fn rank(self) -> usize { self.native.rank }
    /// Named native and safe query, constructor and owner transports.
    pub fn control_bytes(self) -> Option<usize> {
        use std::mem::size_of;
        let parts = [size_of::<Self>(), size_of::<Option<Self>>(),
            size_of::<PreparedResidentGraph>(), size_of::<Result<PreparedResidentGraph>>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(), size_of::<*mut c_void>(),
            size_of::<u32>(), OriginalScopeObserver::control_bytes()?,
            size_of::<<crate::ops::QuantizedArrays as crate::utils::guard::Guarded>::Guard>(),
            size_of::<crate::ops::QuantizedArrays>(), size_of::<Result<crate::ops::QuantizedArrays>>(),
            size_of::<crate::ops::QuantizationMode>(), size_of::<[safemlx_sys::mlx_array; 3]>(),
            size_of::<[*mut safemlx_sys::mlx_array; 3]>()];
        parts.into_iter().try_fold(self.native.named_control_bytes.checked_add(std::mem::size_of_val(&parts))?, usize::checked_add)
    }
}

/// CPU MXFP4 group-32/four-bit frontend through the shared resident graph bank.
/// The six eager constant requests need separately admitted physical backing;
/// source custody, Eval and completion resources are not included here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuMxFp4QuantizeConstructionLayout {
    native: safemlx_sys::mlx_cpu_mxfp4_quantize_construction_layout,
}
impl CpuMxFp4QuantizeConstructionLayout {
    /// Bound for potential frontend nodes, including identity candidates.
    pub fn graph(self) -> ResidentGraphLayout {
        ResidentGraphLayout { native: self.native.graph, additional_shells: 0 }
    }
    /// Eager constant payloads in constructor order; apply the actual allocator's
    /// request layout to each entry before creating the bound physical budget.
    pub fn seed_request_bytes(self) -> [usize; 6] { self.native.seed_request_bytes }
    /// Named native, C and safe query/constructor transports, including the
    /// resident graph driver's control bytes. This is not a process memory ceiling.
    pub fn control_bytes(self) -> Option<usize> {
        use std::mem::{size_of,size_of_val};
        let parts=[self.graph().control_bytes()?,size_of::<Self>(),size_of::<Option<Self>>(),
            size_of::<crate::Dtype>(),size_of::<usize>()*3,
            size_of::<<crate::ops::QuantizedArrays as crate::utils::guard::Guarded>::Guard>(),
            size_of::<crate::ops::QuantizedArrays>(),size_of::<Result<crate::ops::QuantizedArrays>>()];
        parts.into_iter().try_fold(self.native.named_control_bytes.checked_add(size_of_val(&parts))?,usize::checked_add)
    }
}

/// Once-owned physical reservation in an existing exact role. Drop frees unused
/// blocks before releasing the retained role; consumed blocks keep their birth.
/// This owner must retire before entering Eval and creates no work authority.
#[derive(Debug)]
pub struct PreparedResidentGraph {
    owner: *mut c_void,
    _observer: OriginalScopeObserver,
}
impl Drop for PreparedResidentGraph {
    fn drop(&mut self) {
        // SAFETY: this unique opaque bank owns only unconsumed Graph storage.
        // The retained observer outlives header/slot retirement on this thread.
        unsafe { safemlx_sys::mlx_operation_event_finish_resident_graph(self.owner) };
    }
}
impl PreparedResidentGraph {
    /// Borrow the exact accepted role retained when this construction bank was
    /// created. Cloning the observer retains that role; it creates no new scope,
    /// quota or source permission. A consumer must still authenticate its active
    /// native context before using the bank's finite completion slots.
    pub fn observer(&self) -> &OriginalScopeObserver {
        &self._observer
    }

    /// Bind a finite nested-completion recipe before any constructor consumes
    /// this bank. It grants no new Scope, storage budget or source authority.
    /// Every nested attempt consumes one slot before native work, including
    /// failure; completed blocks keep their real array/backing ownership.
    pub fn configure_nested_completions(
        &mut self,
        traversal: &OperationEvalTraversalLayout,
        attempts: usize,
    ) -> Result<()> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(self._observer.error(10));
        };
        let limits = traversal.native_limits();
        // SAFETY: unique live bank and its retained same-Scope observer; native
        // validates the untouched bank and recomputes the scalar traversal.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_configure_nested_graph(
                self.owner,
                self._observer.raw,
                &limits,
                attempts,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(self._observer.error(status))
        }
    }
}

impl OperationEvent {
    /// Query the CPU MXFP4 frontend for positive F16/BF16/F32 geometry. Rows is
    /// the product of all leading dimensions; columns must be divisible by 32.
    pub fn cpu_mxfp4_quantize_construction_layout(dtype: crate::Dtype,
        rank: usize, rows: usize, columns: usize) -> Option<CpuMxFp4QuantizeConstructionLayout> {
        let mut native=safemlx_sys::mlx_cpu_mxfp4_quantize_construction_layout::default();
        // SAFETY: scalar-only query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_mxfp4_quantize_construction_layout(
            &mut native,dtype.into(),rank,rows,columns) }
            .then_some(CpuMxFp4QuantizeConstructionLayout { native })
    }

    /// Query the actual fixed affine constructor, including its three results.
    pub fn affine_quantize_construction_layout(rank: usize) -> Option<AffineQuantizeConstructionLayout> {
        let mut native = safemlx_sys::mlx_affine_quantize_construction_layout::default();
        // SAFETY: scalar-only query writes initialized output on success.
        unsafe { safemlx_sys::mlx_operation_event_affine_quantize_construction_layout(&mut native, rank) }
            .then_some(AffineQuantizeConstructionLayout { native })
    }
    /// Reserve the affine constructor bank in the current original role.
    /// Drop this bank before Eval; its consumed blocks retain their owners.
    pub fn prepare_affine_quantize_graph(layout: AffineQuantizeConstructionLayout,
        observer: &OriginalScopeObserver) -> Result<PreparedResidentGraph> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else { return Err(observer.error(10)); };
        let mut owner = ptr::null_mut();
        // SAFETY: live observer and empty unique destination; native rechecks the role.
        let status = unsafe { safemlx_sys::mlx_operation_event_prepare_affine_quantize_graph(
            &mut owner, observer.raw, layout.rank()) };
        if status != 0 { return Err(observer.error(status)); }
        Ok(PreparedResidentGraph { owner, _observer: observer.clone() })
    }
    /// Pure qualified host layout for an actual closed resident lowering recipe.
    /// Scalar counts are diagnostics; preparation still needs the current original role.
    pub fn resident_graph_layout(
        primitives: usize,
        seeds: usize,
        maximum_rank: usize,
    ) -> Option<ResidentGraphLayout> {
        Self::resident_graph_layout_with_operands(primitives, seeds, maximum_rank, 4)
    }
    /// Qualified host recipe with the actual maximum operand-list arity.
    /// The native owner prices geometric vector growth and rechecks this scalar
    /// when reserving physical blocks; it grants no graph or submission authority.
    pub fn resident_graph_layout_with_operands(
        primitives: usize,
        seeds: usize,
        maximum_rank: usize,
        maximum_operands: usize,
    ) -> Option<ResidentGraphLayout> {
        Self::resident_graph_layout_with_shells(
            primitives,
            seeds,
            maximum_rank,
            maximum_operands,
            0,
        )
    }
    /// Reserve independently derived immutable C handle copies in the same
    /// constructor bank. These copies create no lazy primitive or buffer birth.
    pub fn resident_graph_layout_with_shells(
        primitives: usize,
        seeds: usize,
        maximum_rank: usize,
        maximum_operands: usize,
        additional_shells: usize,
    ) -> Option<ResidentGraphLayout> {
        let mut native = safemlx_sys::mlx_resident_graph_layout::default();
        // SAFETY: allocation-free scalar query; only success publishes output.
        unsafe {
            safemlx_sys::mlx_operation_event_resident_graph_layout_with_shells(
                &mut native,
                primitives,
                seeds,
                maximum_rank,
                maximum_operands,
                additional_shells,
            )
        }
        .then_some(ResidentGraphLayout {
            native,
            additional_shells,
        })
    }
    /// Reserve every physical Graph block before host equations start. The
    /// current role is authenticated; no ordinary fallback or second bank exists.
    pub fn prepare_resident_graph(
        layout: ResidentGraphLayout,
        observer: &OriginalScopeObserver,
    ) -> Result<PreparedResidentGraph> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(observer.error(10));
        };
        let mut owner = ptr::null_mut();
        // SAFETY: scalar immutable geometry, live observer and null unique destination.
        let status = unsafe {
            safemlx_sys::mlx_operation_event_prepare_resident_graph_with_shells(
                &mut owner,
                observer.raw,
                layout.primitives(),
                layout.seeds(),
                layout.maximum_rank(),
                layout.maximum_operands(),
                layout.additional_shells(),
            )
        };
        if status != 0 {
            return Err(observer.error(status));
        }
        Ok(PreparedResidentGraph {
            owner,
            _observer: observer.clone(),
        })
    }
}

/// Owning request sum for the selected single-GPU worker profile. Persistent
/// shader source/cache and platform-private Metal storage are separate owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentGpuWorkerLayout {
    native: safemlx_sys::mlx_resident_gpu_worker_layout,
}
impl ResidentGpuWorkerLayout {
    /// Sum of managed Graph request extents, including failed prefixes.
    pub fn allocation_extents(self) -> usize {
        self.native.allocation_extents
    }
    /// Source-derived upper bound on actual native pipeline lookups.
    pub fn kernel_attempts(self) -> usize {
        self.native.kernel_attempts
    }
    /// Named scalar query transports in C, native code and this wrapper.
    pub fn control_bytes(self) -> Option<usize> {
        self.native.named_control_bytes.checked_add(
            std::mem::size_of::<Self>()
                + std::mem::size_of::<Option<Self>>()
                + std::mem::size_of::<safemlx_sys::mlx_resident_gpu_worker_layout>()
                + 10 * std::mem::size_of::<usize>()
                + std::mem::size_of::<Option<usize>>()
                + 3 * std::mem::size_of::<usize>()
                + 2 * std::mem::size_of::<bool>(),
        )
    }
}
impl OperationEvent {
    /// Pure layout for an authenticated no-tracing single-GPU resident recipe.
    /// Unsupported worker/rank profiles remain absent; this creates no work.
    #[allow(clippy::too_many_arguments)]
    pub fn resident_gpu_worker_layout(
        entries: usize,
        input_edges: usize,
        output_slots: usize,
        array_nodes: usize,
        backing_births: usize,
        maximum_rank: usize,
        maximum_operands: usize,
    ) -> Option<ResidentGpuWorkerLayout> {
        let mut native = safemlx_sys::mlx_resident_gpu_worker_layout::default();
        // SAFETY: only scalar inputs; native writes the destination on success.
        unsafe {
            safemlx_sys::mlx_operation_event_resident_gpu_worker_layout(
                &mut native,
                entries,
                input_edges,
                output_slots,
                array_nodes,
                backing_births,
                maximum_rank,
                maximum_operands,
            )
        }
        .then_some(ResidentGpuWorkerLayout { native })
    }
    /// Pure layout for an authenticated no-tracing single-GPU resident recipe.
    /// Includes extra merge/copy kernels from the same native grouped-sort query.
    #[allow(clippy::too_many_arguments)]
    pub fn resident_gpu_worker_layout_with_sorts(
        entries: usize,
        input_edges: usize,
        output_slots: usize,
        array_nodes: usize,
        backing_births: usize,
        maximum_rank: usize,
        maximum_operands: usize,
        additional_sort_kernels: usize,
    ) -> Option<ResidentGpuWorkerLayout> {
        let mut native = safemlx_sys::mlx_resident_gpu_worker_layout::default();
        // SAFETY: only scalar inputs; native writes the destination on success.
        unsafe {
            safemlx_sys::mlx_operation_event_resident_gpu_worker_layout_with_sorts(
                &mut native,
                entries,
                input_edges,
                output_slots,
                array_nodes,
                backing_births,
                maximum_rank,
                maximum_operands,
                additional_sort_kernels,
            )
        }
        .then_some(ResidentGpuWorkerLayout { native })
    }
    /// GPU worker and within-Eval synchronization requests for one selected GPU
    /// plus an admitted rank-two CPU partition stream. GPU counts and backing
    /// births exclude the separately owned CPU partition bank/output births.
    #[allow(clippy::too_many_arguments)]
    pub fn resident_gpu_worker_layout_with_router(
        entries: usize,
        input_edges: usize,
        output_slots: usize,
        array_nodes: usize,
        backing_births: usize,
        maximum_rank: usize,
        maximum_operands: usize,
        additional_sort_kernels: usize,
        cpu_partitions: usize,
    ) -> Option<ResidentGpuWorkerLayout> {
        let mut native = safemlx_sys::mlx_resident_gpu_worker_layout::default();
        // SAFETY: only scalar inputs; native writes the destination on success.
        unsafe {
            safemlx_sys::mlx_operation_event_resident_gpu_worker_layout_with_router(
                &mut native,
                entries,
                input_edges,
                output_slots,
                array_nodes,
                backing_births,
                maximum_rank,
                maximum_operands,
                additional_sort_kernels,
                cpu_partitions,
            )
        }
        .then_some(ResidentGpuWorkerLayout { native })
    }
    /// Cumulative native worker population for the same authenticated descriptor
    /// DAG. Includes every fresh Synchronizer, completion frontier and consumer
    /// wait; it creates no work and permits no replay of a failed primitive.
    #[allow(clippy::too_many_arguments)]
    pub fn resident_gpu_worker_layout_with_frontiers(
        entries: usize,
        input_edges: usize,
        output_slots: usize,
        array_nodes: usize,
        backing_births: usize,
        maximum_rank: usize,
        maximum_operands: usize,
        additional_sort_kernels: usize,
        evaluations: usize,
        consumer_waits: usize,
    ) -> Option<ResidentGpuWorkerLayout> {
        let mut native = safemlx_sys::mlx_resident_gpu_worker_layout::default();
        // SAFETY: scalar query inputs and one writable fixed destination only.
        unsafe {
            safemlx_sys::mlx_operation_event_resident_gpu_worker_layout_with_frontiers(
                &mut native,
                entries,
                input_edges,
                output_slots,
                array_nodes,
                backing_births,
                maximum_rank,
                maximum_operands,
                additional_sort_kernels,
                0,
                evaluations,
                consumer_waits,
            )
        }
        .then_some(ResidentGpuWorkerLayout { native })
    }
    /// Extra GPU merge/copy dispatches after each four-byte grouped/router sort's first kernel.
    /// This shares actual native sort geometry and creates no arrays or work.
    pub fn resident_grouped_sort_additional_kernels(selections: usize) -> Option<usize> {
        let mut value = 0;
        // SAFETY: only a scalar input and writable fixed result are passed.
        unsafe {
            safemlx_sys::mlx_operation_event_resident_grouped_sort_additional_kernels(
                selections, &mut value,
            )
        }
        .then_some(value)
    }
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod grouped_sort_tests {
    use super::OperationEvent;

    #[test]
    fn grouped_sort_merge_boundaries_extend_actual_worker_destinations() {
        let base = OperationEvent::resident_gpu_worker_layout(3, 4, 3, 20, 50, 3, 4);
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_some() {
            assert!(base.is_some(), "selected native worker layout must qualify");
        }
        let Some(base) = base else { return };
        for (selections, extra) in [(0, 0), (1, 0), (2048, 0), (2049, 3), (4097, 5)] {
            assert_eq!(
                OperationEvent::resident_grouped_sort_additional_kernels(selections),
                Some(extra)
            );
            let extended =
                OperationEvent::resident_gpu_worker_layout_with_sorts(3, 4, 3, 20, 50, 3, 4, extra)
                    .unwrap();
            assert_eq!(extended.kernel_attempts(), base.kernel_attempts() + extra);
            assert!(extended.allocation_extents() >= base.allocation_extents());
        }
        assert!(OperationEvent::resident_grouped_sort_additional_kernels(usize::MAX).is_none());
        assert!(OperationEvent::resident_gpu_worker_layout_with_sorts(
            3,
            4,
            3,
            20,
            50,
            3,
            4,
            usize::MAX,
        )
        .is_none());
    }
}

#[cfg(all(
    test,
    feature = "metal",
    target_vendor = "apple",
    not(feature = "cuda")
))]
mod frontier_tests {
    use super::OperationEvent;

    #[test]
    fn cumulative_worker_keeps_frontiers_without_replaying_primitive_population() {
        // One retained GPU descriptor DAG, with substantial nonzero work.
        let query = |evaluations, waits| {
            OperationEvent::resident_gpu_worker_layout_with_frontiers(
                256,
                640,
                256,
                1024,
                512,
                4,
                6,
                0,
                evaluations,
                waits,
            )
        };
        let one = query(1, 0);
        if std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_some() {
            assert!(one.is_some(), "selected worker profile must qualify");
        }
        let Some(one) = one else { return };
        let old = OperationEvent::resident_gpu_worker_layout(256, 640, 256, 1024, 512, 4, 6)
            .expect("same selected worker");
        assert_eq!(one.allocation_extents(), old.allocation_extents());
        assert_eq!(one.kernel_attempts(), old.kernel_attempts());
        let many = query(25, 24).unwrap();
        assert_eq!(many.kernel_attempts(), one.kernel_attempts());
        assert!(many.allocation_extents() > one.allocation_extents());
        assert!(many.allocation_extents() < one.allocation_extents().checked_mul(25).unwrap());
        assert!(query(0, 0).is_none());
        assert!(query(usize::MAX, 0).is_none());
        assert!(query(1, usize::MAX).is_none());
    }
}

#[cfg(test)]
mod affine_construction_tests {
    use super::*;
    #[test]
    fn affine_quantize_construction_layout_tracks_rank_storage() {
        let qualified = OperationEvent::resident_graph_layout(1, 0, 2).is_some();
        let mut previous = 0;
        for rank in [2, 4, 11, 21] {
            let layout = OperationEvent::affine_quantize_construction_layout(rank);
            assert_eq!(layout.is_some(), qualified);
            if let Some(layout) = layout {
                assert_eq!(layout.rank(), rank);
                assert!(layout.graph_allocation_extents() > 0);
                assert!(layout.graph_allocation_extents() >= previous);
                assert!(layout.control_bytes().unwrap() > 0);
                previous = layout.graph_allocation_extents();
            }
        }
        for rank in [0, 1, usize::MAX] {
            assert!(OperationEvent::affine_quantize_construction_layout(rank).is_none());
        }
    }
}

#[cfg(test)]
mod mxfp4_construction_tests {
    use super::*;
    #[test]
    fn cpu_mxfp4_quantize_construction_tracks_seed_precision_and_geometry() {
        let qualified=OperationEvent::resident_graph_layout(1,0,3).is_some();
        for dtype in [crate::Dtype::Float16,crate::Dtype::Bfloat16,crate::Dtype::Float32] {
            for rank in [2,3,4,11,21] {
                let layout=OperationEvent::cpu_mxfp4_quantize_construction_layout(dtype,rank,2,64);
                assert_eq!(layout.is_some(),qualified);
                if let Some(layout)=layout {
                    let scalar=if dtype==crate::Dtype::Float32 {4} else {2};
                    assert_eq!(layout.seed_request_bytes(),[scalar,scalar,scalar,4,64,4]);
                    assert_eq!(layout.graph().seeds(),6);
                    assert!(layout.graph().allocation_extents()>0);
                    assert!(layout.control_bytes().unwrap()>0);
                }
            }
        }
        for (dtype,rank,rows,columns) in [
            (crate::Dtype::Float64,2,2,64),(crate::Dtype::Int32,2,2,64),
            (crate::Dtype::Float32,1,2,64),(crate::Dtype::Float32,2,0,64),
            (crate::Dtype::Float32,2,2,33),(crate::Dtype::Float32,2,usize::MAX,64),
        ] {
            assert!(OperationEvent::cpu_mxfp4_quantize_construction_layout(dtype,rank,rows,columns).is_none());
        }
    }
}
