//! Native populations of the shared contiguous-then-eager-copy program.
use super::{graph_capacity::ResidentGraphStorage, ResidentDispatchPopulation};
use safemlx::{
    OperationEvalTraversalLayout, OperationEvalTraversalLimits, OperationEvent,
    ResidentGraphLayout, SubmissionGraphQuota, SubmissionRecordQuota,
};

#[derive(Clone, Copy)]
enum CpuCopyProgram {
    Arrays,
    Pending {
        rank: usize,
        elements: usize,
        cast: bool,
    },
}

/// No source or admission authority is created by these derived populations.
/// The caller supplies counts from actual immutable operand descriptor loans.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IsolatedCopyNativeLayout {
    pub(crate) graph: ResidentGraphLayout,
    pub(crate) traversal: OperationEvalTraversalLayout,
    pub(crate) graph_extents: usize,
    pub(crate) record_extents: usize,
    pub(crate) graph_capacity: usize,
    pub(crate) record_capacity: usize,
    pub(crate) kernel_attempts: usize,
    pub(crate) completion_attempts: usize,
    pub(crate) controls: usize,
}
impl IsolatedCopyNativeLayout {
    /// Same array-only copy program on an authenticated CPU execution stream.
    /// Host transfers and pending cast/reshape tails retain their own selection.
    pub(crate) fn cpu(operands: usize, source_clones: usize, rank: usize) -> Option<Self> {
        if operands == 0 {
            return None;
        }
        Self::inspect_program(
            Some(CpuCopyProgram::Arrays),
            operands,
            source_clones,
            rank,
            0,
            0,
            0,
            0,
            0,
        )
    }

    /// Completed media adds no pending producer. An empty state/key source has
    /// zero copy/completion attempts; retained descriptor shells remain exact.
    pub(crate) fn cpu_completed(
        operands: usize,
        source_clones: usize,
        rank: usize,
    ) -> Option<Self> {
        Self::inspect_program(
            Some(CpuCopyProgram::Arrays),
            operands,
            source_clones,
            rank,
            0,
            0,
            0,
            0,
            0,
        )
    }

    /// The same completed isolate followed by the pending worker's actual
    /// reshape and optional signed cast. Complete matrices already have their
    /// target shape, so the ordinary reshape call returns the input directly.
    pub(crate) fn cpu_pending_input(rank: usize, elements: usize, cast: bool) -> Option<Self> {
        Self::cpu_resume(1, 1, rank, rank, elements, cast)
    }

    /// The existing repeated decoder/key isolate worker plus one final pending
    /// input frontier. Its own rank is distinct from the largest state operand.
    pub(crate) fn cpu_resume(
        operands: usize,
        source_clones: usize,
        maximum_rank: usize,
        pending_rank: usize,
        elements: usize,
        cast: bool,
    ) -> Option<Self> {
        if operands == 0
            || source_clones == 0
            || pending_rank > maximum_rank
            || elements == 0
            || (pending_rank != 2 && elements != 1)
        {
            return None;
        }
        Self::inspect_program(
            Some(CpuCopyProgram::Pending {
                rank: pending_rank,
                elements,
                cast,
            }),
            operands,
            source_clones,
            maximum_rank,
            1 + usize::from(cast),
            0,
            0,
            0,
            0,
        )
    }

    pub(crate) fn inspect(operands: usize, source_clones: usize, rank: usize) -> Option<Self> {
        Self::inspect_program(None, operands, source_clones, rank, 0, 0, 0, 0, 0)
    }

    /// One actual pending scalar: the shared isolate leaf followed by reshape
    /// and optional cast, with one additional completed frontier. The initial
    /// source clone is the resume recovery owner's actual retained scalar.
    pub(crate) fn pending_input(rank: usize, cast: bool) -> Option<Self> {
        Self::inspect_program(None, 1, 1, rank, 1 + usize::from(cast), 0, 0, 0, 0)
    }

    pub(crate) fn resume(
        operands: usize,
        source_clones: usize,
        rank: usize,
        cast: bool,
    ) -> Option<Self> {
        Self::inspect_program(
            None,
            operands,
            source_clones,
            rank,
            1 + usize::from(cast),
            0,
            0,
            0,
            0,
        )
    }

    /// Same eager-copy program with source-selected immutable Host loads. The
    /// source builder supplies exact direct native layout sums; these metadata
    /// counts establish neither source identity nor permission to run a copy.
    pub(crate) fn with_host_sources(
        operands: usize,
        source_clones: usize,
        rank: usize,
        pending_tail: usize,
        host_loads: usize,
        host_direct_extents: usize,
        host_controls: usize,
    ) -> Option<Self> {
        if host_loads > operands {
            return None;
        }
        Self::inspect_program(
            None,
            operands,
            source_clones,
            rank,
            pending_tail,
            host_loads,
            host_direct_extents,
            host_controls,
            0,
        )
    }

    pub(crate) fn with_host_transfers(
        operands: usize,
        source_clones: usize,
        rank: usize,
        pending_tail: usize,
        loads: usize,
        load_extents: usize,
        load_controls: usize,
        stores: usize,
        store_extents: usize,
        store_controls: usize,
    ) -> Option<Self> {
        if loads > operands || stores > operands {
            return None;
        }
        Self::inspect_program(
            None,
            operands,
            source_clones,
            rank,
            pending_tail,
            loads,
            load_extents.checked_add(store_extents)?,
            load_controls.checked_add(store_controls)?,
            stores,
        )
    }

    fn inspect_program(
        cpu: Option<CpuCopyProgram>,
        operands: usize,
        source_clones: usize,
        rank: usize,
        pending_tail: usize,
        host_loads: usize,
        host_direct_extents: usize,
        host_controls: usize,
        host_stores: usize,
    ) -> Option<Self> {
        let store = super::host_store_lowering();
        // One possible Contiguous primitive, one eager deep-copy source and
        // two recovery clones and at most one stored result alias per operand. Aliased/contiguous/empty
        // inputs can reduce actual work but cannot increase these populations.
        // The result constructors move ordinary logical values directly; the
        // compressed storage representation additionally aliases each logical
        // copy exactly once. This shared upper bound requires no family branch
        // and does not add a primitive, payload birth or publication charge.
        // The shared native fixed-input ArrayVector constructor reserves four
        // entries even for this one-input operation. Both shared queries retain
        // that actual capacity; one is the semantic arity, not its allocation.
        // Pending source scalars may be rank zero; the real prompt reshape
        // nevertheless constructs rank two before its optional cast.
        let source_rank = rank;
        let has_pending = pending_tail != 0 || matches!(cpu, Some(CpuCopyProgram::Pending { .. }));
        let rank = if has_pending { rank.max(2) } else { rank };
        let shells = source_clones
            .checked_add(operands.checked_mul(3)?)?
            .checked_add(pending_tail)?
            .checked_add(host_loads)?
            .checked_add(host_stores)?;
        // A same-shape reshape still returns/retains descriptor handles, but
        // MLX creates no Reshape primitive. Keep that shell separate from Eval.
        let tail_primitives = if let Some(CpuCopyProgram::Pending { rank, cast, .. }) = cpu {
            usize::from(rank != 2).checked_add(usize::from(cast))?
        } else {
            pending_tail
        };
        let primitives = operands
            .checked_add(tail_primitives)?
            .checked_add(host_loads)?
            .checked_add(host_stores.checked_mul(store.primitives)?)?;
        let graph = OperationEvent::resident_graph_layout_with_shells(
            primitives,
            operands
                .checked_add(host_loads)?
                .checked_add(host_stores.checked_mul(store.seeds)?)?,
            rank,
            4,
            shells,
        )?;
        // Exactly one nested root. The input must already be a completed leaf;
        // its optional Contiguous plus the real Synchronizer make two entries.
        // A pending tail starts from the already-completed eager copy. Its
        // reshape/optional cast plus Synchronizer are a linear prefix with at
        // most three entries. Reuse that finite ceiling for both frontiers.
        let entries = 2usize
            .max(tail_primitives.checked_add(1)?)
            .max(if host_stores == 0 {
                0
            } else {
                store.primitives.checked_add(1)?
            });
        // A Host load completes its CopyFromHostTransfer + Synchronizer before
        // the same contiguous/eager-copy leaf consumes that exact Array.
        let completion_attempts = operands
            .checked_add(usize::from(has_pending))?
            .checked_add(host_loads)?
            .checked_add(host_stores.checked_mul(store.nested_completions)?)?;
        let record = OperationEvent::eval_record_layout(entries, 1, entries)?;
        let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
            roots: 1,
            arrays: entries.checked_add(1)?,
            tape_entries: entries,
            input_edges: entries,
            output_slots: entries,
            streams: 1,
            captures: record.capture_slots().max(1),
        })?;
        // Both devices use the same original constructor, traversal and root
        // worker. Only the native Eval/dispatch source population differs.
        let mut one = ResidentGraphStorage::default();
        let mut tail = ResidentGraphStorage::default();
        let mut repeated = completion_attempts;
        let (kernel_attempts, worker_controls) = if let Some(cpu) = cpu {
            if host_loads != 0 || host_stores != 0 {
                return None;
            }
            // Every occurrence uses the same installed traversal ceiling, but
            // the already-completed isolate is a leaf of the second frontier.
            one.include_cpu_copy_eval(traversal, source_rank)?;
            let tail_controls = if let CpuCopyProgram::Pending {
                rank,
                elements,
                cast,
            } = cpu
            {
                tail.include_cpu_pending_eval(traversal, rank, elements, cast)?;
                tail.add(OperationEvent::root_storage_layout(1)?.graph_request_extent())?;
                repeated = operands;
                usize::try_from(tail.control_bytes()?).ok()?
            } else {
                if has_pending {
                    return None;
                }
                0
            };
            (0, tail_controls)
        } else {
            let worker = OperationEvent::resident_gpu_worker_layout(
                entries,
                entries,
                entries,
                entries.checked_add(1)?,
                1,
                rank,
                4,
            )?;
            let dispatch = ResidentDispatchPopulation {
                cpu_model: None,
                gpu_entries: entries,
                gpu_input_edges: entries,
                gpu_siblings: entries,
                gpu_births: 1,
                additional_sort_kernels: 0,
                cpu_entries: 0,
                cpu_input_edges: 0,
                cpu_siblings: 0,
                parallel_entries: 0,
                parallel_graph_extents: 0,
                worker_graph_extents: worker.allocation_extents(),
                worker_rank: rank,
                copy_rank_extents: 0,
                kernel_attempts: worker.kernel_attempts(),
            };
            one.include_eval(traversal, dispatch)?;
            (worker.kernel_attempts(), worker.control_bytes()?)
        };
        one.add(OperationEvent::root_storage_layout(1)?.graph_request_extent())?;
        let extent = usize::try_from(one.known_constructor_bytes)
            .ok()?
            .checked_mul(repeated)?
            .checked_add(usize::try_from(tail.known_constructor_bytes).ok()?)?
            .checked_add(graph.allocation_extents())?
            .checked_add(host_direct_extents)?;
        let graph_capacity = SubmissionGraphQuota::fresh_capacity_for_extents(extent)?;
        let record_extents = traversal
            .record_allocation_extents()?
            .checked_mul(completion_attempts)?;
        let record_capacity = SubmissionRecordQuota::fresh_capacity_for_extents(record_extents)?;
        let parts = [
            host_controls,
            std::mem::size_of::<(
                Option<CpuCopyProgram>,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
            )>(),
            std::mem::size_of::<CpuCopyProgram>(),
            std::mem::size_of::<(usize, usize, usize, usize, usize, bool)>(),
            std::mem::size_of::<Option<Self>>(),
            std::mem::size_of::<ResidentGraphStorage>(),
            std::mem::size_of::<usize>() * 4,
            std::mem::size_of::<bool>(),
            std::mem::size_of::<(
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
            )>(),
            graph.control_bytes()?,
            traversal.query_control_bytes()?,
            worker_controls,
            std::mem::size_of::<bool>(),
            std::mem::size_of::<(usize, usize)>(),
            usize::try_from(one.control_bytes()?).ok()?,
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Option<Self>>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)?;
        Some(Self {
            graph,
            traversal,
            graph_extents: extent,
            record_extents,
            graph_capacity,
            record_capacity,
            kernel_attempts: kernel_attempts.checked_mul(completion_attempts)?,
            completion_attempts,
            controls,
        })
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;

    #[test]
    fn cpu_pending_input_prices_each_completed_frontier_without_duplicate_copy_workers() {
        for (rank, elements, cast, entries, primitives) in [
            (0, 1, true, 3, 3),
            (1, 1, false, 2, 2),
            (2, 7, true, 2, 2),
            (2, 7, false, 2, 1),
            (6, 1, true, 3, 3),
        ] {
            let layout = IsolatedCopyNativeLayout::cpu_pending_input(rank, elements, cast).unwrap();
            assert_eq!(layout.completion_attempts, 2);
            assert_eq!(layout.kernel_attempts, 0);
            assert_eq!(layout.traversal.limits().tape_entries, entries);
            assert_eq!(layout.graph.primitives(), primitives);
            let isolated = IsolatedCopyNativeLayout::cpu(1, 1, rank).unwrap();
            assert!(layout.graph_extents > isolated.graph_extents);
            assert!(layout.record_extents > isolated.record_extents);
            assert!(layout.controls > 0);
        }
        assert!(IsolatedCopyNativeLayout::cpu_pending_input(1, 7, true).is_none());
        assert!(IsolatedCopyNativeLayout::cpu_pending_input(2, 0, false).is_none());
        assert!(IsolatedCopyNativeLayout::cpu_pending_input(6, 7, true).is_none());
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[test]
fn cpu_saved_resume_keeps_pending_rank_separate_from_state_copy_rank() {
    for (rank, elements, cast, primitives, entries) in
        [(2, 7, false, 6, 2), (2, 7, true, 7, 2), (0, 1, true, 8, 3)]
    {
        let layout = IsolatedCopyNativeLayout::cpu_resume(6, 8, 5, rank, elements, cast).unwrap();
        assert_eq!(layout.graph.primitives(), primitives);
        assert_eq!(layout.traversal.limits().tape_entries, entries);
        assert_eq!(layout.completion_attempts, 7);
        assert_eq!(layout.kernel_attempts, 0);
        let base = IsolatedCopyNativeLayout::cpu(6, 8, 5).unwrap();
        assert!(layout.record_extents > base.record_extents);
        assert!(layout.graph_extents > base.graph_extents);
    }
    assert!(IsolatedCopyNativeLayout::cpu_resume(6, 8, 1, 2, 7, true).is_none());
    assert!(IsolatedCopyNativeLayout::cpu_resume(6, 8, 5, 0, 7, true).is_none());
}
