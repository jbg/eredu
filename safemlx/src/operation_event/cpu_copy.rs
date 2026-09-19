//! Source populations of the actual CPU copy and completion Eval workers.
use super::OperationEvent;
use crate::Dtype;
use std::mem::{size_of, size_of_val};

/// CPU Contiguous, AsType or private completion Eval storage, without physical backing
/// or stream/event/record authority. The native consumer authenticates its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuCopyEvalLayout {
    native: safemlx_sys::mlx_cpu_copy_eval_layout,
}
impl CpuCopyEvalLayout {
    /// Host Eval bank, including cleanup, invocation aliases and actual tasks.
    pub fn graph_allocation_extents(self) -> usize { self.native.graph_extents }
    /// Higher-rank General-copy worker scratch under the same Graph loan.
    pub fn worker_graph_allocation_extents(self) -> usize { self.native.worker_graph_extents }
    /// Actual final CPU Event::signal task, separate from the Eval bank.
    pub fn signal_graph_allocation_extents(self) -> usize { self.native.signal_graph_extents }
    /// Maximum physical output attempts; allocator rounding remains separate.
    pub fn backing_births(self) -> usize { self.native.backing_births }
    /// Complete query and native host/worker control storage.
    pub fn control_bytes(self) -> Option<usize> {
        let parts = [size_of::<Self>(), size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_cpu_copy_eval_layout>(),
            size_of::<*mut safemlx_sys::mlx_cpu_copy_eval_layout>(),
            size_of::<usize>() * 3, size_of::<bool>() * 3, size_of::<Dtype>() * 2];
        parts.into_iter().try_fold(self.native.named_control_bytes.checked_add(size_of_val(&parts))?, usize::checked_add)
    }
    fn inspect(rank: usize, inputs: usize, copy: bool, tracer: bool) -> Option<Self> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: scalar pure query writes the initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_copy_eval_layout(&mut native, rank, inputs, copy, tracer) }
            .then_some(Self { native })
    }
}
impl OperationEvent {
    /// Exact immutable Host load/store Eval through the shared General-copy
    /// worker. The accepted Host owner and stream remain independently checked
    /// by the actual native constructor; this scalar fact grants neither.
    pub fn cpu_host_transfer_layout(dtype:Dtype,rank:usize,store:bool,tracer:bool)->Option<CpuCopyEvalLayout>{
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: scalar-only query, initialized output, no retained pointers.
        if !unsafe{safemlx_sys::mlx_operation_event_cpu_host_transfer_eval_layout(
            &mut native,dtype.into(),rank,store,tracer)} {return None;}
        let parts=[size_of::<(Dtype,usize,bool,bool)>(),size_of_val(&native),size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(parts.into_iter()
            .try_fold(size_of_val(&parts),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }

    /// Bound the existing CPU Contiguous copy branch from its actual input rank.
    /// Alias/empty sources may consume less; no lazy flag or donation is assumed.
    pub fn cpu_contiguous_layout(rank: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        CpuCopyEvalLayout::inspect(rank, 1, true, tracer)
    }
    /// Bound the existing CPU AsType conversion with checked source/output
    /// geometry. This is descriptive storage, not execution or source authority.
    pub fn cpu_cast_layout(source: Dtype, destination: Dtype, rank: usize,
        elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: the pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_cast_eval_layout(&mut native,
            source.into(), destination.into(), rank, elements, tracer) }
            .then_some(CpuCopyEvalLayout { native })
    }
    /// Exact View byte reinterpretation source: an alias or one General-copy
    /// temporary inside the same accepted Eval. Native inspection checks the
    /// actual byte ratio, source span, primitive destination and copy branch.
    pub fn cpu_byte_view_layout(source:Dtype,destination:Dtype,rank:usize,bytes:usize,
        copy:bool,tracer:bool)->Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes only initialized local storage.
        if !unsafe {safemlx_sys::mlx_operation_event_cpu_byte_view_eval_layout(&mut native,
            source.into(),destination.into(),rank,bytes,copy,tracer)} {return None;}
        let controls=[size_of::<(Dtype,Dtype,usize,usize,bool,bool)>(),size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(controls.into_iter()
            .try_fold(size_of_val(&controls),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Exact positive-step CPU Slice at fixed rank 1..4: nonempty alias or
    /// empty zero-byte Data construction with the selected allocator's backing birth.
    /// Native Eval validates the actual normalized source geometry before use.
    pub fn cpu_slice_layout(rank: usize, empty: bool, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: the pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_slice_eval_layout(&mut native, rank, empty, tracer) }
            .then_some(CpuCopyEvalLayout { native })
    }
    /// Exact existing rank-one F32 scalar overwrite worker. The native Eval
    /// authenticates source strides, update coordinates, dtype and reduction.
    pub fn cpu_scalar_update_layout(elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_scalar_update_eval_layout(
            &mut native, elements, tracer) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Exact partial F32/I32 static rectangle through the existing full copy and
    /// shaped overwrite tasks. The native producer authenticates complete row
    /// sources, matching precision, exact coordinates and unit strides.
    pub fn cpu_static_update_layout(rank:usize,elements:usize,update_elements:usize,tracer:bool)->Option<CpuCopyEvalLayout>{
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe {safemlx_sys::mlx_operation_event_cpu_static_update_eval_layout(
            &mut native,rank,elements,update_elements,tracer)}.then_some(CpuCopyEvalLayout{native})
    }
    /// Shared stable ArgSort worker: a single F32 row or compact I32 final-axis
    /// rows at rank one through three. Native Eval authenticates the actual
    /// dtype, shape, strides and source backing independently of this query.
    pub fn cpu_argsort_layout(dtype: Dtype, rank: usize, columns: usize, rows: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_argsort_eval_layout(
            &mut native, dtype.into(), rank, columns, rows, tracer) } { return None; }
        let frames = [size_of::<(Dtype,usize,usize,usize,bool)>(),
            size_of::<safemlx_sys::mlx_dtype>(), size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes = native.named_control_bytes.checked_add(frames.into_iter()
            .try_fold(size_of_val(&frames),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Bound the existing F32 last-axis CPU softmax worker and optional real
    /// contiguous copy. The query grants no input, backing or role authority.
    pub fn cpu_softmax_layout(rank: usize, columns: usize, rows: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_softmax_eval_layout(&mut native,
            rank, columns, rows, tracer) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Exact CPU Softmax task: F32, or F16/BF16 with precise F32 accumulation.
    /// The native descriptor independently validates dtype, precision and the
    /// completed input geometry. This source grants no native role or backing.
    pub fn cpu_typed_softmax_layout(dtype: Dtype, precise: bool, rank: usize,
        columns: usize, rows: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_typed_softmax_eval_layout(
            &mut native, dtype.into(), precise, rank, columns, rows, tracer) } { return None; }
        let controls = [size_of::<Dtype>(),size_of::<safemlx_sys::mlx_dtype>(),
            size_of::<(bool,usize,usize,usize,bool)>(),size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes = native.named_control_bytes.checked_add(controls.into_iter()
            .try_fold(size_of_val(&controls),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Bound the existing F32 ArgReduce worker at fixed rank, independent of
    /// primitive/source authority. Native Eval validates the actual axis/result.
    pub fn cpu_arg_reduce_layout(rank: usize, columns: usize, rows: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_greedy_eval_layout(&mut native,
            rank, columns, rows, true, tracer) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Existing F16/BF16/F32 ArgMin/ArgMax with U32 indices. Native Eval checks
    /// the actual axis, shape and readable strided source; first ties are retained.
    pub fn cpu_typed_arg_reduce_layout(dtype: Dtype, rank: usize, columns: usize,
        rows: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: scalar-only query publishes an initialized layout on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_typed_arg_reduce_eval_layout(
            &mut native, dtype.into(), rank, columns, rows, tracer,
        ) } { return None; }
        let controls = [size_of::<(Dtype, usize, usize, usize, bool)>(),
            size_of::<safemlx_sys::mlx_dtype>(), size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes = native.named_control_bytes.checked_add(
            controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)?,
        )?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Affine conversion Eval with packed weights, scales and biases sharing
    /// one task and cleanup. Input compaction adds one backing and retained
    /// temporary. Physical allocation sizes and graph construction are separate.
    pub fn cpu_affine_quantize_layout(
        dtype: Dtype, rank: usize, rows: usize, columns: usize,
        group_size: i32, bits: i32, copy: bool, tracer: bool,
    ) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: scalar-only query publishes initialized output on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_affine_quantize_eval_layout(
            &mut native, dtype.into(), rank, rows, columns, group_size, bits, copy, tracer,
        ) } { return None; }
        let controls = [size_of::<(Dtype, usize, usize, usize, i32, i32, bool, bool)>(),
            size_of::<safemlx_sys::mlx_dtype>(), size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes = native.named_control_bytes.checked_add(
            controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)?,
        )?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Bound the existing fixed-rank Squeeze alias and its actual cleanup.
    pub fn cpu_squeeze_layout(rank: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_greedy_eval_layout(&mut native,
            rank, 0, 0, false, tracer) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Bound the fixed-rank row-contiguous Reshape alias and its actual cleanup.
    pub fn cpu_reshape_alias_layout(rank: usize, output_rank: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_reshape_alias_eval_layout(&mut native,
            rank, output_rank, tracer) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Price the existing fixed-rank Reshape's possible General-copy branch
    /// when a cold integer layout does not carry stride evidence. Native Eval
    /// still validates the actual complete source geometry and chooses its branch.
    pub fn cpu_reshape_copy_layout(rank: usize, output_rank: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query retains no pointers; initialized output.
        unsafe { safemlx_sys::mlx_operation_event_cpu_reshape_copy_eval_layout(&mut native,
            rank, output_rank, tracer) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Same fixed native reshape planner over proved physical strides. The
    /// returned birth count distinguishes the actual copy from a storage alias.
    pub fn cpu_reshape_layout(source_shape:&[i32],source_strides:&[i64],
        output_shape:&[i32],tracer:bool)->Option<CpuCopyEvalLayout> {
        if source_shape.len()!=source_strides.len(){return None;}
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: exact slice lengths and live immutable scalar buffers; the
        // pure query retains no pointer and writes output only on success.
        unsafe {safemlx_sys::mlx_operation_event_cpu_reshape_eval_layout(&mut native,
            source_shape.as_ptr(),source_strides.as_ptr(),source_shape.len(),
            output_shape.as_ptr(),output_shape.len(),tracer)}.then_some(CpuCopyEvalLayout{native})
    }

    /// Bound the existing fixed-rank Transpose alias and actual cleanup.
    pub fn cpu_transpose_alias_layout(rank: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_shape_alias_layout(0, rank, rank, tracer)
    }
    /// Bound a nonempty fixed-rank Broadcast alias; Eval validates the exact
    /// declaration, source strides and shared backing before using the bank.
    pub fn cpu_broadcast_alias_layout(rank: usize, output_rank: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_shape_alias_layout(1, rank, output_rank, tracer)
    }
    /// Empty Broadcast executes its existing zero-byte Data construction,
    /// without aliasing the source. The selected allocator determines whether
    /// the zero-byte request creates a physical backing owner.
    pub fn cpu_empty_broadcast_layout(rank: usize, output_rank: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_shape_alias_layout(3, rank, output_rank, tracer)
    }
    /// Bound the existing fixed-rank ExpandDims alias worker. Native Eval
    /// validates actual sorted unit axes, dtype, shape and source backing.
    pub fn cpu_expand_dims_alias_layout(rank: usize, output_rank: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_shape_alias_layout(2,rank,output_rank,tracer)
    }
    fn cpu_shape_alias_layout(operation: u32, rank: usize, output_rank: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_alias_eval_layout(
            &mut native, operation, rank, output_rank, tracer,
        ) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Bound actual U32 RandomBits with one explicit two-word key and its raw task.
    pub fn cpu_random_bits_layout(rank: usize, elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_random_bits_eval_layout(&mut native,
            rank, elements, tracer) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Exact plain BF16 CPU Matmul row-worker storage for compact/transposed
    /// fixed-rank matrices. Native execution validates the actual input strides;
    /// this fact neither qualifies BLAS nor grants model/source authority.
    pub fn cpu_bf16_matmul_layout(rank: usize, m: usize, n: usize, k: usize,
        batches: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe { safemlx_sys::mlx_operation_event_cpu_bf16_matmul_eval_layout(&mut native,
            rank, m, n, k, batches, tracer) }.then_some(CpuCopyEvalLayout { native })
    }
    /// Exact BF16 row Matmul plus its actual zero, one or two General copies.
    /// Rank-five batch metadata remains inline; native inspection validates the
    /// selected primitive and actual operand strides before entering this bank.
    pub fn cpu_bf16_matmul_copy_layout(rank: usize, m: usize, n: usize, k: usize,
        batches: usize, copies: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_bf16_matmul_copy_eval_layout(
            &mut native,rank,m,n,k,batches,copies,tracer) } { return None; }
        let controls=[size_of::<(usize,usize,usize,usize,usize,usize,bool)>(),
            size_of_val(&native),size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(controls.into_iter()
            .try_fold(size_of_val(&controls),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Exact selected F16 SIMD Matmul plus its actual zero, one or two General copies.
    /// Rank-five batch metadata remains inline; native inspection validates the
    /// selected primitive and actual operand strides before entering this bank.
    pub fn cpu_f16_matmul_copy_layout(rank: usize, m: usize, n: usize, k: usize,
        batches: usize, copies: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_f16_matmul_copy_eval_layout(
            &mut native,rank,m,n,k,batches,copies,tracer) } { return None; }
        let controls=[size_of::<(usize,usize,usize,usize,usize,usize,bool)>(),
            size_of_val(&native),size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(controls.into_iter()
            .try_fold(size_of_val(&controls),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Bound the explicitly selected F32 tile Matmul source. Actual native
    /// primitive choice, compact layout and input custody are checked at Eval.
    pub fn cpu_tiled_matmul_layout(rank: usize, m: usize, n: usize, k: usize,
        batches: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe {safemlx_sys::mlx_operation_event_cpu_tiled_matmul_eval_layout(&mut native,
            rank,m,n,k,batches,tracer)}.then_some(CpuCopyEvalLayout {native})
    }
    /// Same selected F32 worker with a finite source-derived number of actual
    /// matrix-interior General-copy temporaries (zero, one or two).
    pub fn cpu_tiled_matmul_copy_layout(rank: usize, m: usize, n: usize, k: usize,
        batches: usize, copies: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output only on success.
        unsafe {safemlx_sys::mlx_operation_event_cpu_tiled_matmul_copy_eval_layout(&mut native,
            rank,m,n,k,batches,copies,tracer)}.then_some(CpuCopyEvalLayout {native})
    }
    /// Bound a one-index full-axis CPU Gather using the existing
    /// signed/unsigned index worker over F32, F16, BF16, I32 or U32 sources. The
    /// joined source/index rank is at most five, including the rank-three
    /// centroid selection of an ordered readout. Actual native geometry is
    /// checked at Eval; index validity remains the ordinary Gather precondition.
    /// Empty indices retain the task and Data metadata, plus the selected
    /// allocator's zero-byte backing birth. An empty source is valid only with
    /// empty indices.
    pub fn cpu_gather_layout(source: Dtype, index: Dtype, source_rank: usize,
        index_rank: usize, source_elements: usize, index_elements: usize,
        slice_elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: the pure scalar query writes initialized output only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_gather_eval_layout(
            &mut native, source.into(), index.into(), source_rank, index_rank,
            source_elements, index_elements, slice_elements, tracer,
        ) } { return None; }
        native.named_control_bytes=native.named_control_bytes.checked_add(
            size_of::<(Dtype,Dtype,usize,usize,usize,usize,usize,bool)>(),
        )?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Shared F32/I32 rank-one/two axis-zero Scatter overwrite, including its
    /// initial copy, queued sparse task and inline single-index iterator.
    /// Index values retain the ordinary operation's validity precondition.
    pub fn cpu_scatter_layout(source: Dtype, index: Dtype, rank: usize,
        output_elements: usize, update_elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure query receives initialized scalar output and no native objects.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_scatter_eval_layout(
            &mut native,source.into(),index.into(),rank,output_elements,update_elements,tracer) } { return None; }
        let frames = [size_of::<(Dtype,Dtype,usize,usize,usize,bool)>(),
            size_of::<(safemlx_sys::mlx_dtype,safemlx_sys::mlx_dtype)>(),size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes = native.named_control_bytes.checked_add(frames.into_iter()
            .try_fold(size_of_val(&frames),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Bound the existing F32 final-axis overwrite scatter. One cleanup owns
    /// its three inputs through both the initial copy and the scatter job.
    pub fn cpu_scatter_axis_layout(index: Dtype, rank: usize, output_elements: usize,
        update_elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes initialized output on success.
        if !unsafe {safemlx_sys::mlx_operation_event_cpu_scatter_axis_eval_layout(
            &mut native,index.into(),rank,output_elements,update_elements,tracer)} {return None;}
        native.named_control_bytes=native.named_control_bytes.checked_add(
            size_of::<(Dtype,usize,usize,usize,bool)>())?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Exact rank-two axis-zero F32 ScatterAxis::Sum, including empty update
    /// rows. This is the additive worker; overwrite keeps its separate source.
    pub fn cpu_scatter_add_rows_layout(index:Dtype,output_elements:usize,
        update_elements:usize,tracer:bool)->Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: the checked scalar query initializes output only on success.
        if !unsafe {safemlx_sys::mlx_operation_event_cpu_scatter_add_rows_eval_layout(
            &mut native,index.into(),output_elements,update_elements,tracer)} {return None;}
        let frames=[size_of::<(Dtype,usize,usize,bool)>(),size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(frames.into_iter()
            .try_fold(size_of_val(&frames),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Bound the ordinary boolean all/any worker over every dimension of a
    /// contiguous, nonempty input. Actual axes and backing are checked at Eval.
    pub fn cpu_boolean_reduce_layout(all: bool, rank: usize, elements: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_reduction_layout(if all { 0 } else { 1 }, rank, elements, 1, tracer)
    }
    /// Bound the existing F32 last-axis cascade-sum worker without changing its
    /// rounding tree. Native Eval authenticates actual row-contiguous storage.
    pub fn cpu_row_sum_layout(rank: usize, width: usize, rows: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_reduction_layout(2, rank, width, rows, tracer)
    }
    /// Existing F32 sum over one nonfinal axis of compact rank-two through
    /// rank-four input. Prefix and suffix products describe its exact geometry;
    /// native Eval checks the real axis, dtype, strides and readable backing.
    /// Width one elides Reduce and must use the caller's ordinary alias source.
    pub fn cpu_strided_sum_layout(rank: usize, axis: usize, width: usize,
        outer: usize, inner: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        if !(2..=4).contains(&rank) || axis >= rank - 1 || width <= 1
            || outer == 0 || inner == 0 || (axis == 0 && outer != 1) {
            return None;
        }
        let rows = outer.checked_mul(inner)?;
        let mut value = Self::cpu_reduction_layout(10, rank, width, rows, tracer)?;
        value.native.named_control_bytes = value.native.named_control_bytes.checked_add(
            size_of::<(usize, usize, usize, usize, usize, bool)>()
                + size_of::<usize>() + size_of::<Option<usize>>()
                + size_of::<CpuCopyEvalLayout>() + size_of::<Option<CpuCopyEvalLayout>>(),
        )?;
        Some(value)
    }
    /// Exact ordinary half-precision final-axis SIMD sum on a complete
    /// row-contiguous input. This retains the half accumulator/output rounding;
    /// width one is an alias handled by the caller's existing cast source.
    pub fn cpu_half_row_sum_layout(dtype: Dtype, rank: usize, width: usize,
        rows: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let kind=match dtype {Dtype::Float16=>7,Dtype::Bfloat16=>8,_=>return None};
        let mut value=Self::cpu_reduction_layout(kind,rank,width,rows,tracer)?;
        value.native.named_control_bytes=value.native.named_control_bytes.checked_add(
            size_of::<(Dtype,usize,usize,usize,bool)>()+size_of::<u32>()+
            size_of::<CpuCopyEvalLayout>()+size_of::<Option<CpuCopyEvalLayout>>())?;
        Some(value)
    }
    /// Existing half-precision final-axis maximum over compact rank-one through
    /// rank-four input. Retains the typed SIMD comparison and NaN/tail order.
    pub fn cpu_half_row_max_layout(dtype: Dtype, rank: usize, width: usize,
        rows: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let kind = match dtype { Dtype::Float16 => 12, Dtype::Bfloat16 => 13, _ => return None };
        let mut value = Self::cpu_reduction_layout(kind, rank, width, rows, tracer)?;
        value.native.named_control_bytes = value.native.named_control_bytes.checked_add(
            size_of::<(Dtype, usize, usize, usize, bool)>() + size_of::<u32>()
                + size_of::<CpuCopyEvalLayout>() + size_of::<Option<CpuCopyEvalLayout>>(),
        )?;
        Some(value)
    }
    /// Bound the ordinary contiguous all-axis F32 SIMD sum, at rank 2..=4.
    /// This preserves its distinct SumReduce order rather than selecting the
    /// last-axis cascade. Native Eval validates all actual axes and backing.
    pub fn cpu_all_sum_layout(rank: usize, elements: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_reduction_layout(4, rank, elements, 1, tracer)
    }
    /// Bound the ordinary last-axis F32 minimum, including its unchanged SIMD
    /// NaN/tail comparisons. Width one is an alias, not a native Reduce source.
    pub fn cpu_row_min_layout(rank: usize, width: usize, rows: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_reduction_layout(3, rank, width, rows, tracer)
    }
    /// Bound the ordinary contiguous final-axis F32 maximum with its unchanged
    /// SIMD, NaN and tail comparison order. Native Eval authenticates every row.
    /// Width one is an alias, rather than a native Reduce allocation.
    pub fn cpu_row_max_layout(rank: usize, width: usize, rows: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_reduction_layout(9, rank, width, rows, tracer)
    }
    /// Existing U32 final-axis SIMD sum over compact rank-two through rank-four
    /// input. Native Eval checks the actual dtype, axes, shape and readable span.
    /// Width one is an alias and has no Reduce worker.
    pub fn cpu_u32_row_sum_layout(rank: usize, width: usize, rows: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_reduction_layout(11, rank, width, rows, tracer)
    }
    /// Existing rank-one U32 complete-axis sum. Native Eval checks the exact
    /// contiguous input, dtype, axes and scalar output before admitting its task.
    pub fn cpu_flat_u32_sum_layout(elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_reduction_layout(5, 1, elements, 1, tracer)
    }
    /// Existing rank-one F32 maximum, preserving ordinary SIMD/NaN/tail order.
    pub fn cpu_flat_max_layout(elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_reduction_layout(6, 1, elements, 1, tracer)
    }
    fn cpu_reduction_layout(operation: u32, rank: usize, width: usize, rows: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: the pure scalar query writes initialized output only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_reduction_eval_layout(
            &mut native, operation, rank, width, rows, tracer,
        ) } { return None; }
        native.named_control_bytes=native.named_control_bytes.checked_add(
            size_of::<(u32,usize,usize,usize,bool)>(),
        )?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Bound the shared contiguous three-input Select worker. The native source
    /// checks exact condition/value shape, canonical dtype and compact backing.
    pub fn cpu_select_layout(dtype: Dtype, rank: usize, elements: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_selection_layout(dtype, rank, elements, false, tracer)
    }
    /// Bound the unchanged F32 Select over exact readable stored/broadcast
    /// operands, including the rank-five GQA mask. Its accepted CPU task owns
    /// the four-stride collapse and three prefix iterators through retirement.
    pub fn cpu_select_broadcast_layout(rank: usize, elements: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar query writes only the initialized local on success.
        if !unsafe {safemlx_sys::mlx_operation_event_cpu_select_broadcast_eval_layout(
            &mut native,rank,elements,tracer)} {return None;}
        native.named_control_bytes=native.named_control_bytes.checked_add(
            size_of::<(usize,usize,bool)>())?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Existing F32/F16/BF16 or I32 Select over readable stored/broadcast operands.
    /// The source checks the actual branch dtype, Boolean condition and spans;
    /// the accepted ordinary task retains its collapse and prefix iterators.
    pub fn cpu_typed_select_broadcast_layout(dtype:Dtype,rank:usize,elements:usize,
        tracer:bool)->Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: scalar query only writes the initialized local on success.
        if !unsafe {safemlx_sys::mlx_operation_event_cpu_typed_select_broadcast_eval_layout(
            &mut native,dtype.into(),rank,elements,tracer)} {return None;}
        let controls=[size_of::<(Dtype,usize,usize,bool)>(),size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(controls.into_iter()
            .try_fold(size_of_val(&controls),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Bound Full's scalar or empty copy branch. Native Eval checks the real
    /// one-element or zero-data source; this query grants no source owner.
    pub fn cpu_scalar_full_layout(dtype: Dtype, rank: usize, elements: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        Self::cpu_selection_layout(dtype, rank, elements, true, tracer)
    }
    /// Bound Full's existing copy from one completed F32 value per row. The
    /// actual native source must be the matching last-axis broadcast alias.
    pub fn cpu_row_full_layout(rank: usize, width: usize, rows: usize,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: the pure scalar query writes initialized output on success.
        if !unsafe {safemlx_sys::mlx_operation_event_cpu_row_full_eval_layout(
            &mut native,rank,width,rows,tracer)} {return None;}
        native.named_control_bytes=native.named_control_bytes.checked_add(
            size_of::<(usize,usize,usize,bool)>())?;
        Some(CpuCopyEvalLayout{native})
    }
    fn cpu_selection_layout(dtype: Dtype, rank: usize, elements: usize, full: bool,
        tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: scalar pure query writes initialized output only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_selection_eval_layout(
            &mut native, dtype.into(), rank, elements, full, tracer,
        ) } { return None; }
        native.named_control_bytes=native.named_control_bytes.checked_add(
            size_of::<(Dtype,usize,usize,bool,bool)>(),
        )?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Bound the actual private CPU completion no-op and its input retention.
    pub fn cpu_completion_layout(inputs: usize) -> Option<CpuCopyEvalLayout> {
        CpuCopyEvalLayout::inspect(0, inputs, false, false)
    }
}

impl OperationEvent {
    /// Exact named frames of the existing CPU RMS fallback, separate from its
    /// scalar sources and each ordinary arithmetic/alias producer. This pure
    /// query neither constructs a stream nor authorizes an operation.
    pub fn cpu_rms_fallback_control_bytes(dtype: crate::Dtype, rank: usize,
        width: usize, rows: usize) -> Option<usize> {
        let mut native=0;
        // SAFETY: scalar query writes the initialized destination only on success.
        let present=unsafe { safemlx_sys::mlx_operation_event_cpu_rms_fallback_control_bytes(
            &mut native,dtype.into(),rank,width,rows) };
        if !present {return None;}
        let parts=[std::mem::size_of::<(crate::Dtype,usize,usize,usize)>(),
            std::mem::size_of::<usize>(),std::mem::size_of::<Option<usize>>(),
            std::mem::size_of::<*mut usize>(),std::mem::size_of::<bool>()];
        parts.into_iter().try_fold(native.checked_add(std::mem::size_of_val(&parts))?,usize::checked_add)
    }
}

impl OperationEvent {
    /// Same CPU concatenation over the exact input count: one destination
    /// slice/copy job per input. Empty output includes the selected allocator's
    /// zero-byte backing birth.
    pub fn cpu_concatenate_many_layout(dtype:Dtype,rank:usize,inputs:usize,
        elements:usize,tracer:bool)->Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: scalar source query writes initialized output only on success.
        if !unsafe {safemlx_sys::mlx_operation_event_cpu_concatenate_many_eval_layout(&mut native,
            dtype.into(),rank,inputs,elements,tracer)} {return None;}
        let frames=[size_of::<(Dtype,usize,usize,usize,bool)>(),size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(frames.into_iter()
            .try_fold(size_of_val(&frames),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Source of the existing two-input CPU concatenation: one backing, two
    /// destination views and both queued GeneralGeneral copies. This scalar
    /// query grants no source/stream/operation authority.
    pub fn cpu_concatenate_layout(dtype: Dtype,rank: usize,left_elements: usize,
        right_elements: usize,tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure query initializes the supplied output only on success.
        unsafe{safemlx_sys::mlx_operation_event_cpu_concatenate_eval_layout(&mut native,
            dtype.into(),rank,left_elements,right_elements,tracer)}.then_some(CpuCopyEvalLayout{native})
    }
}

impl OperationEvent {
    /// Actual F32 Arange worker and zero-input cleanup, without source authority.
    /// Native Eval checks the primitive's own finite start/stop/step and extent.
    pub fn cpu_arange_float_layout(elements:usize,tracer:bool)->Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: the pure query writes this fixed output only on success.
        unsafe{safemlx_sys::mlx_operation_event_cpu_arange_float_eval_layout(&mut native,elements,tracer)}
            .then_some(CpuCopyEvalLayout{native})
    }
    /// Exact nonnegative I32/U32 unit-step coordinate source. Native validation
    /// checks both integer endpoints and the final loop increment before Eval.
    pub fn cpu_arange_int_layout(dtype:Dtype,elements:usize,tracer:bool)->Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure scalar source query writes initialized output on success.
        if !unsafe{safemlx_sys::mlx_operation_event_cpu_arange_int_eval_layout(&mut native,dtype.into(),elements,tracer)} {return None;}
        let frames=[size_of::<(Dtype,usize,bool)>(),size_of_val(&native),size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(frames.into_iter()
            .try_fold(size_of_val(&frames),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Selected F32 GatherMM through the same SIMD tile worker as selected
    /// Matmul. Native Eval checks actual matrix spans, index geometry and stream
    /// selection; this query supplies no arrays, index values or authority.
    pub fn cpu_tiled_gather_mm_layout(lhs_rank:usize,rhs_rank:usize,index_rank:usize,
        m:usize,n:usize,k:usize,batches:usize,tracer:bool)->Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: scalar-only query writes initialized output only on success.
        if !unsafe{safemlx_sys::mlx_operation_event_cpu_tiled_gather_mm_eval_layout(
            &mut native,lhs_rank,rhs_rank,index_rank,m,n,k,batches,tracer)} {return None;}
        let frames=[size_of::<(usize,usize,usize,usize,usize,usize,usize,bool)>(),
            size_of_val(&native),size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(frames.into_iter()
            .try_fold(size_of_val(&frames),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
    /// Named controls of the unchanged rank-three/four full-width CPU RoPE
    /// fallback, using default or supplied frequencies. Primitive queries
    /// separately describe the selected frequency and reshape workers.
    pub fn cpu_rope_fallback_control_bytes(rank:usize,dimensions:usize,elements:usize)->Option<usize> {
        let mut native=0;
        // SAFETY: the pure source query retains no pointer or runtime object.
        if !unsafe{safemlx_sys::mlx_operation_event_cpu_rope_fallback_control_bytes(&mut native,rank,dimensions,elements)} {
            return None;
        }
        let frames=[size_of::<usize>()*4,size_of::<*mut usize>(),size_of::<Option<usize>>()];
        frames.into_iter().try_fold(native.checked_add(size_of_val(&frames))?,usize::checked_add)
    }
}

impl OperationEvent {
    /// The same SDPA/GQA worker with an explicit Boolean array mask. These
    /// controls include its borrowed mask frontend and named shared mask worker;
    /// arithmetic, alias, selection and backing sources are priced separately.
    pub fn cpu_sdpa_array_mask_control_bytes(rank:usize,queries:usize,keys:usize,values:usize,scores:usize)->Option<usize> {
        let mut native=0;
        // SAFETY: the pure query writes only the fixed result and retains nothing.
        if !unsafe{safemlx_sys::mlx_operation_event_cpu_sdpa_array_mask_control_bytes(&mut native,rank,queries,keys,values,scores)} {
            return None;
        }
        let parts=[size_of::<usize>()*6,size_of::<*mut usize>(),size_of::<Option<usize>>()];
        parts.into_iter().try_fold(native.checked_add(size_of_val(&parts))?,usize::checked_add)
    }
    /// Exact shared mask-free CPU SDPA/GQA fallback and frontend controls.
    pub fn cpu_sdpa_fallback_control_bytes(rank:usize,queries:usize,keys:usize,values:usize,scores:usize)->Option<usize> {
        let mut native=0;
        // SAFETY: the pure query writes only the fixed result and retains nothing.
        if !unsafe{safemlx_sys::mlx_operation_event_cpu_sdpa_fallback_control_bytes(&mut native,rank,queries,keys,values,scores)} {
            return None;
        }
        let parts=[size_of::<usize>()*6,size_of::<*mut usize>(),size_of::<Option<usize>>()];
        parts.into_iter().try_fold(native.checked_add(size_of_val(&parts))?,usize::checked_add)
    }
}

impl OperationEvent {
    /// Exact F32 single-row partition row source at rank one through three.
    /// Native Eval independently authenticates source, axis, mode and backing.
    pub fn cpu_partition_row_layout(rank: usize, elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure query; initialized destination is published only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_partition_row_eval_layout(
            &mut native, rank, elements, tracer) } { return None; }
        let parts = [size_of::<(usize,usize,bool)>(), size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes = native.named_control_bytes.checked_add(
            parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Exact F32 single-row scan sum row source at rank one through three.
    /// Native Eval independently authenticates source, axis, mode and backing.
    pub fn cpu_scan_sum_row_layout(rank: usize, elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure query; initialized destination is published only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_scan_sum_row_eval_layout(
            &mut native, rank, elements, tracer) } { return None; }
        let parts = [size_of::<(usize,usize,bool)>(), size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes = native.named_control_bytes.checked_add(
            parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout { native })
    }
    /// Exact F32 single-row maximum row source at rank one through three.
    /// Native Eval independently authenticates source, axis, mode and backing.
    pub fn cpu_maximum_row_layout(rank: usize, elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native = safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: pure query; initialized destination is published only on success.
        if !unsafe { safemlx_sys::mlx_operation_event_cpu_maximum_row_eval_layout(
            &mut native, rank, elements, tracer) } { return None; }
        let parts = [size_of::<(usize,usize,bool)>(), size_of_val(&native),
            size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes = native.named_control_bytes.checked_add(
            parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout { native })
    }
}

impl OperationEvent {
    /// Existing same-shape F32 GatherAxis at ranks one to three. Native Eval
    /// checks its compact source and readable U32 index backing, including
    /// broadcast index axes; the queued worker checks values before each read.
    /// Retains the ordinary indexing task and both iterator sources.
    pub fn cpu_gather_axis_row_layout(rank: usize, elements: usize, tracer: bool) -> Option<CpuCopyEvalLayout> {
        let mut native=safemlx_sys::mlx_cpu_copy_eval_layout::default();
        // SAFETY: the pure query only publishes an initialized exact layout.
        if !unsafe {safemlx_sys::mlx_operation_event_cpu_gather_axis_row_eval_layout(
            &mut native,rank,elements,tracer)} {return None;}
        let parts=[size_of::<(usize,usize,bool)>(),size_of_val(&native),size_of::<Option<CpuCopyEvalLayout>>()];
        native.named_control_bytes=native.named_control_bytes.checked_add(
            parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?)?;
        Some(CpuCopyEvalLayout{native})
    }
}

#[cfg(test)]
mod quantization_integer_tests {
    use super::*;

    #[test]
    fn u32_row_sum_layout_checks_the_native_geometry() {
        let qualified = OperationEvent::cpu_flat_u32_sum_layout(7, false).is_some();
        for rank in 2..=4 {
            for rows in [1, 6] {
                let layout = OperationEvent::cpu_u32_row_sum_layout(rank, 7, rows, false);
                assert_eq!(layout.is_some(), qualified);
                if let Some(layout) = layout {
                    assert!(layout.graph_allocation_extents() > 0);
                    assert_eq!(layout.backing_births(), 1);
                    assert_eq!(layout.worker_graph_allocation_extents(), 0);
                    assert!(layout.control_bytes().unwrap() > 0);
                }
            }
            for width in [0, 1, usize::MAX] {
                assert!(OperationEvent::cpu_u32_row_sum_layout(rank, width, 6, false).is_none());
            }
            assert!(OperationEvent::cpu_u32_row_sum_layout(rank, 7, 0, false).is_none());
        }
        for rank in [0, 1, 5, usize::MAX] {
            assert!(OperationEvent::cpu_u32_row_sum_layout(rank, 7, 6, false).is_none());
        }
    }
}

#[cfg(test)]
mod quantization_half_tests {
    use super::*;

    #[test]
    fn typed_reduction_queries_preserve_dtype_and_geometry_checks() {
        let qualified = OperationEvent::cpu_arg_reduce_layout(2, 7, 3, false).is_some();
        for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            for rank in 1..=4 {
                let arg = OperationEvent::cpu_typed_arg_reduce_layout(dtype, rank, 7, 3, false);
                assert_eq!(arg.is_some(), qualified);
                if let Some(arg) = arg {
                    assert_eq!(arg.backing_births(), 1);
                    assert_eq!(arg.worker_graph_allocation_extents(), 0);
                    assert!(arg.graph_allocation_extents() > 0);
                    assert!(arg.control_bytes().unwrap() > 0);
                }
                let max = OperationEvent::cpu_half_row_max_layout(dtype, rank, 7, 3, false);
                assert_eq!(max.is_some(), qualified && dtype != Dtype::Float32);
                if let Some(max) = max {
                    assert_eq!(max.backing_births(), 1);
                    assert_eq!(max.worker_graph_allocation_extents(), 0);
                    assert!(max.control_bytes().unwrap() > 0);
                }
            }
            for rank in [0, 5, usize::MAX] {
                assert!(OperationEvent::cpu_typed_arg_reduce_layout(dtype, rank, 7, 3, false).is_none());
                assert!(OperationEvent::cpu_half_row_max_layout(dtype, rank, 7, 3, false).is_none());
            }
            assert!(OperationEvent::cpu_typed_arg_reduce_layout(dtype, 2, 7, i32::MAX as usize + 2, false).is_none());
            assert!(OperationEvent::cpu_typed_arg_reduce_layout(dtype, 2, 0, 3, false).is_none());
            assert!(OperationEvent::cpu_half_row_max_layout(dtype, 2, 1, 3, false).is_none());
        }
        for dtype in [Dtype::Bool, Dtype::Int32, Dtype::Uint32, Dtype::Float64] {
            assert!(OperationEvent::cpu_typed_arg_reduce_layout(dtype, 2, 7, 3, false).is_none());
            assert!(OperationEvent::cpu_half_row_max_layout(dtype, 2, 7, 3, false).is_none());
        }
    }
}

#[cfg(test)]
mod affine_converter_tests {
    use super::*;
    #[test]
    fn affine_converter_layout_prices_outputs_and_compaction() {
        let qualified = OperationEvent::cpu_arg_reduce_layout(2, 7, 3, false).is_some();
        for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            for rank in [2, 4, 11] {
                for group in [32, 64, 128] {
                    for bits in [2, 3, 4, 5, 6, 8] {
                        for copy in [false, true] {
                            let layout = OperationEvent::cpu_affine_quantize_layout(
                                dtype, rank, 2, group as usize * 2, group, bits, copy, false);
                            assert_eq!(layout.is_some(), qualified);
                            if let Some(layout) = layout {
                                assert_eq!(layout.backing_births(), 3 + usize::from(copy));
                                assert!(layout.graph_allocation_extents() > 0);
                                assert!(layout.control_bytes().unwrap() > 0);
                            }
                        }
                    }
                }
            }
        }
        for (rank, rows, width, group, bits) in [
            (1, 2, 64, 32, 4), (2, 0, 64, 32, 4), (2, 2, 63, 32, 4),
            (2, 2, 64, 16, 4), (2, 2, 64, 32, 7), (2, usize::MAX, 64, 32, 4),
        ] {
            assert!(OperationEvent::cpu_affine_quantize_layout(
                Dtype::Float32, rank, rows, width, group, bits, false, false).is_none());
        }
        assert!(OperationEvent::cpu_affine_quantize_layout(
            Dtype::Int32, 2, 2, 64, 32, 4, false, false).is_none());
    }
}
