//! All attempted Graph constructors for the selected resident request.
use super::*;
use eredu_runtime::working_memory::{SamplingWorkspacePhase, WorkingMemoryError};
use safemlx::{OperationEvent, SubmissionGraphQuota};
use std::{
    mem::{size_of, size_of_val},
    num::NonZeroU64,
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ResidentGraphStorage {
    pub(crate) known_constructor_bytes: u64,
    pub(crate) full_capacity: Option<u64>,
    query_controls: usize,
}
impl ResidentGraphStorage {
    pub(super) fn include_source_copies(&mut self, copies: host_copies::HostCopies,
        transfers: host_copies::HostTransfers, dispatch: ResidentDispatchPopulation) -> Option<()> {
        let mut one = ResidentGraphStorage::default();
        one.include_eval(copies.traversal, copies.dispatch)
            ?;
        one.add(
            OperationEvent::root_storage_layout(1)
                ?
                .graph_request_extent(),
        )
        ?;
        self.known_constructor_bytes = self
            .known_constructor_bytes
            .checked_add(
                one.known_constructor_bytes
                    .checked_mul(u64::try_from(copies.per_forward).ok()?)
                    ?,
            )
            ?;
        self
            .add(copies.direct_graph_extents)
            ?;
        self.query_controls = self.query_controls.max(one.query_controls);
        let mut aggregate = ResidentGraphStorage::default();
        aggregate
            .include_eval(copies.aggregate_traversal, copies.aggregate_dispatch)
            ?;
        aggregate
            .add(
                OperationEvent::root_storage_layout(transfers.roots)
                    ?
                    .graph_request_extent(),
            )
            ?;
        self.known_constructor_bytes = self
            .known_constructor_bytes
            .checked_add(
                aggregate
                    .known_constructor_bytes
                    .checked_mul(
                u64::try_from(transfers.per_forward).ok()?,
                    )
                    ?,
            )
            ?;
        // Source waits can close an encoder containing both model and
        // source resources. Price both authentic worker universes.
        self
            .add(
                dispatch.worker_graph_extents
                    .checked_add(copies.aggregate_dispatch.worker_graph_extents)
                    ?
                    .checked_mul(transfers.waits_per_forward)
                    ?,
            )
            ?;
        self.query_controls = self.query_controls.max(aggregate.query_controls);
        self.full_capacity = Some(u64::try_from(SubmissionGraphQuota::fresh_capacity_for_extents(
            usize::try_from(self.known_constructor_bytes).ok()?)?).ok()?);
        Some(())
    }
    /// Same actual DAG and completion population for one non-text numerical cut.
    pub(super) fn for_completion(completion: ResidentCompletionRecipe) -> Option<Self> {
        let mut value = Self::default();
        let roots = safemlx::PrefillRoots::layout(completion.traversal.roots()).ok()?;
        value.query_controls = roots.control_bytes;
        value.add(roots.graph_bytes)?;
        value.add(completion.graph.allocation_extents())?;
        value.include_dag(
            completion.traversal,
            completion.graph,
            completion.dispatch?,
            NestedCompletionRoots::uniform(completion.nested_completions, completion.nested_root_capacity.max(3)),
            0,
        )?;
        value.full_capacity = Some(
            u64::try_from(SubmissionGraphQuota::fresh_capacity_for_extents(
                usize::try_from(value.known_constructor_bytes).ok()?,
            )?)
            .ok()?,
        );
        Some(value)
    }
    /// Existing eager input, fixed-rank view or F32 numerical worker with private CPU completion.
    /// The eager constructor's Graph and backing remain priced by its own source.
    pub(super) fn for_cpu_completion(completion: ResidentCompletionRecipe, cpu_operation: Option<numerical::CpuNumericalOperation>) -> Option<Self> {
        let limits = completion.traversal.limits();
        let roots_count=match cpu_operation {
            Some(numerical::CpuNumericalOperation::SplitKeys {views,..}) if (1..=2).contains(&views)=>views,
            Some(numerical::CpuNumericalOperation::SplitKeys {..})=>return None,
            Some(numerical::CpuNumericalOperation::UniformUnitInterval|numerical::CpuNumericalOperation::Difference{..}|numerical::CpuNumericalOperation::Categorical{..})=>2,
            _=>1,
        };
        if limits.roots != roots_count || limits.streams != 1
            || match cpu_operation { None | Some(numerical::CpuNumericalOperation::EagerKey) => limits.tape_entries != 1,
                Some(numerical::CpuNumericalOperation::Slice {..} | numerical::CpuNumericalOperation::Index {..}) => !(1..=3).contains(&limits.tape_entries),
                Some(numerical::CpuNumericalOperation::Normalize {..}) => !(1..=4).contains(&limits.tape_entries),
                Some(numerical::CpuNumericalOperation::Greedy {..}) => !(1..=6).contains(&limits.tape_entries),
                Some(numerical::CpuNumericalOperation::SplitKeys {views,..}) => !(1..=views.checked_mul(2)?.checked_add(2)?).contains(&limits.tape_entries),
                Some(numerical::CpuNumericalOperation::Difference{..}|numerical::CpuNumericalOperation::Categorical{..})=>limits.tape_entries==0
                    ||limits.tape_entries>completion.graph.primitives().checked_add(completion.graph.seeds())?.checked_add(2)?,
                Some(numerical::CpuNumericalOperation::Logarithm{..})=>!(1..=3).contains(&limits.tape_entries),
                Some(numerical::CpuNumericalOperation::UniformUnitInterval) => limits.tape_entries==0
                    // One existing incoming key is a leaf outside the new constructor bank.
                    || limits.tape_entries>completion.graph.primitives().checked_add(completion.graph.seeds())?.checked_add(1)? }
            || completion.nested_completions != 0 || completion.validation_roots != 0
            || completion.dispatch.is_some() {
            return None;
        }
        let operation = match cpu_operation {
            Some(numerical::CpuNumericalOperation::Slice {rank} | numerical::CpuNumericalOperation::Index {rank,..}) => Some(OperationEvent::cpu_slice_layout(rank,false)?),
            Some(numerical::CpuNumericalOperation::Normalize {rank,columns,rows}) =>
                Some(OperationEvent::cpu_softmax_layout(rank,columns,rows,false)?),
            Some(numerical::CpuNumericalOperation::Greedy {rank,columns}) =>
                Some(OperationEvent::cpu_arg_reduce_layout(rank,columns,1,false)?),
            Some(numerical::CpuNumericalOperation::SplitKeys {count,..}) => Some(OperationEvent::cpu_random_bits_layout(2,count.checked_mul(2)?,false)?),
            None | Some(numerical::CpuNumericalOperation::EagerKey | numerical::CpuNumericalOperation::UniformUnitInterval
                | numerical::CpuNumericalOperation::Difference{..}|numerical::CpuNumericalOperation::Logarithm{..}|numerical::CpuNumericalOperation::Categorical{..}) => None,
        };
        let roots = safemlx::PrefillRoots::layout(roots_count).ok()?;
        let source = OperationEvent::cpu_completion_layout(roots_count)?;
        if source.backing_births() != 0 || source.worker_graph_allocation_extents() != 0 {
            return None;
        }
        let mut value = Self::default();
        value.add(roots.graph_bytes)?;
        value.add(completion.graph.allocation_extents())?;
        let controls = value.include_eval_construction(completion.traversal)?;
        value.add(source.graph_allocation_extents())?;
        value.add(source.signal_graph_allocation_extents())?;
        let operation_controls = if let Some(operation) = operation {
            let expected_births = usize::from(matches!(cpu_operation, Some(numerical::CpuNumericalOperation::Normalize {..} | numerical::CpuNumericalOperation::Greedy {..} | numerical::CpuNumericalOperation::SplitKeys {..})));
            if operation.backing_births()!=expected_births || operation.signal_graph_allocation_extents()!=0 { return None; }
            value.add(operation.graph_allocation_extents())?;
            value.add(operation.worker_graph_allocation_extents())?;
            operation.control_bytes()?
        } else { 0 };
        let alias=match cpu_operation {
            Some(numerical::CpuNumericalOperation::Greedy {rank,..}) => Some(OperationEvent::cpu_squeeze_layout(rank,false)?),
            Some(numerical::CpuNumericalOperation::Index {rank,output_rank}) => Some(OperationEvent::cpu_reshape_alias_layout(rank,output_rank,false)?),
            _ => None,
        };
        let alias_controls=if let Some(alias)=alias {
            if alias.backing_births()!=0 || alias.worker_graph_allocation_extents()!=0
                || alias.signal_graph_allocation_extents()!=0 {return None;}
            value.add(alias.graph_allocation_extents())?;
            alias.control_bytes()?
        } else {0};
        let split_controls=if let Some(numerical::CpuNumericalOperation::SplitKeys {count,views})=cpu_operation {
            let sources=[if count==1 {None} else {Some(OperationEvent::cpu_slice_layout(2,false)?)},
                Some(OperationEvent::cpu_reshape_alias_layout(2,1,false)?)];
            let mut controls=0usize;
            for source in sources.into_iter().flatten() {
                if source.backing_births()!=0 || source.worker_graph_allocation_extents()!=0
                    || source.signal_graph_allocation_extents()!=0 {return None;}
                value.add(source.graph_allocation_extents().checked_mul(views)?)?;
                controls=controls.checked_add(source.control_bytes()?.checked_mul(views)?)?;
            }
            controls
        } else {0};
        let composed=match cpu_operation {
            Some(numerical::CpuNumericalOperation::UniformUnitInterval)=>Some(super::super::cpu::uniform_population()?),
            Some(numerical::CpuNumericalOperation::Categorical{rank,columns})=>Some(super::super::cpu::categorical_population(rank,columns)?),
            Some(numerical::CpuNumericalOperation::Difference{rank,columns,rows})=>Some(super::super::cpu::difference_population(rank,columns,rows)?),
            Some(numerical::CpuNumericalOperation::Logarithm{rank})=>Some(super::super::cpu::logarithm_population(rank)?),
            _=>None,
        };
        let composed_controls=if let Some(population)=composed {
            value.add(population.extents)?;population.controls
        }else{0};
        let parts = [controls, roots.control_bytes, source.control_bytes()?, operation_controls, alias_controls, split_controls, composed_controls,
            size_of::<super::super::cpu::CpuPopulation>(),size_of::<Option<super::super::cpu::CpuPopulation>>(),
            size_of::<[Option<safemlx::CpuCopyEvalLayout>;2]>(),
            size_of::<std::iter::Flatten<std::array::IntoIter<Option<safemlx::CpuCopyEvalLayout>,2>>>(),
            size_of::<usize>()*3,
            size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            size_of::<safemlx::CpuCopyEvalLayout>(),size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            size_of::<Option<numerical::CpuNumericalOperation>>(), size_of::<safemlx::CpuCopyEvalLayout>(),
            size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            completion.graph.control_bytes()?, completion.traversal.query_control_bytes()?,
            size_of::<ResidentCompletionRecipe>(), size_of::<Self>(), size_of::<Option<Self>>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(), size_of::<safemlx::PrefillRootsLayout>(),
            size_of::<safemlx::CpuCopyEvalLayout>(), size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            size_of::<Option<usize>>(), size_of::<usize>()];
        value.query_controls = parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)?;
        value.full_capacity = Some(u64::try_from(SubmissionGraphQuota::fresh_capacity_for_extents(
            usize::try_from(value.known_constructor_bytes).ok()?)?).ok()?);
        Some(value)
    }
    pub(super) fn add(&mut self, extents: usize) -> Option<()> {
        let extents = u64::try_from(extents).ok()?;
        self.known_constructor_bytes = self.known_constructor_bytes.checked_add(extents)?;
        Some(())
    }
    pub(crate) fn select_capacity(
        self,
        ceiling: NonZeroU64,
    ) -> Result<NonZeroU64, WorkingMemoryError> {
        self.select_for_policy(Some(ceiling))
    }
    pub(crate) fn select_for_policy(
        self,
        ceiling: Option<NonZeroU64>,
    ) -> Result<NonZeroU64, WorkingMemoryError> {
        let required_bytes = self.full_capacity.ok_or(WorkingMemoryError::UnknownBound)?;
        if let Some(ceiling) = ceiling.filter(|ceiling| ceiling.get() < required_bytes) {
            return Err(WorkingMemoryError::GraphMetadataCapacity {
                required_bytes,
                configured_bytes: ceiling.get(),
            });
        }
        self.full_capacity
            .and_then(NonZeroU64::new)
            .ok_or(WorkingMemoryError::UnknownBound)
    }
    pub(crate) fn control_bytes(self) -> Option<u64> {
        let fixed = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, crate::backend::error::Error>>(),
            size_of::<(usize, bool)>(),
            size_of::<ResidentDispatchPopulation>(),
            size_of::<safemlx::PrefillRootsLayout>(),
            size_of::<(*mut safemlx::OperationRootStorageLayout, usize, bool)>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(),
            // Shared nested-frontier query: requested roots and added edges.
            size_of::<(usize, usize)>(),
            size_of::<safemlx::OriginalPromptInputFacts>(),
            size_of::<Option<safemlx::OriginalPromptInputFacts>>(),
            size_of::<safemlx::OperationRootStorageLayout>(),
            size_of::<Option<safemlx::OperationRootStorageLayout>>(),
            SubmissionGraphQuota::fit_query_control_bytes()?,
        ]
        .into_iter()
        .try_fold(self.query_controls, usize::checked_add)?;
        u64::try_from(fixed).ok()
    }
    fn include_nested(
        &mut self, traversal: safemlx::OperationEvalTraversalLayout,
        dispatch: ResidentDispatchPopulation, roots: NestedCompletionRoots<'_>,
    ) -> Option<()> {
        for count in roots.iter() {
            let mut single = Self::default();
            let traversal = nested_traversal_with_roots(traversal, count)?;
            single.include_eval(traversal, dispatch)?;
            single.add(OperationEvent::root_storage_layout(count)?.graph_request_extent())?;
            self.known_constructor_bytes = self.known_constructor_bytes
                .checked_add(single.known_constructor_bytes)?;
            self.query_controls = self.query_controls.max(single.query_controls);
        }
        Some(())
    }
    /// The same non-tracing descriptor DAG is dispatched at most once over all
    /// successful nested/final Eval calls. Fresh synchronization and stream
    /// frontier owners remain cumulative, even when the roots were evaluated.
    fn include_dag(
        &mut self,
        traversal: safemlx::OperationEvalTraversalLayout,
        graph: safemlx::ResidentGraphLayout,
        dispatch: ResidentDispatchPopulation,
        roots: NestedCompletionRoots<'_>,
        waits: usize,
    ) -> Option<()> {
        let nested = roots.count()?;
        // Keep the existing mixed-stream population until its within-Eval
        // fences have a separately composed cross-completion proof.
        if dispatch.cpu_entries != 0 || traversal.limits().streams != 1 {
            self.include_eval(traversal, dispatch)?;
            self.include_nested(traversal, dispatch, roots)?;
            self.add(dispatch.worker_graph_extents.checked_mul(waits)?)?;
            return Some(());
        }
        let evaluations = nested.checked_add(1)?;
        let extra_roots = roots.total_roots()?;
        let mut population = dispatch;
        // The original row already includes its final Synchronizer. Each
        // nested producer contributes its quoted root list and one fresh
        // Synchronizer; these are new prologue/worker inputs, not replay.
        population.gpu_entries = population.gpu_entries.checked_add(nested)?;
        population.gpu_input_edges = population.gpu_input_edges.checked_add(extra_roots)?;
        population.gpu_siblings = population.gpu_siblings.checked_add(nested)?;
        let arrays = traversal
            .limits()
            .arrays
            .checked_add(extra_roots)?
            .checked_add(nested)?;
        let worker = OperationEvent::resident_gpu_worker_layout_with_frontiers(
            population.gpu_entries,
            population.gpu_input_edges,
            population.gpu_siblings,
            arrays,
            population.gpu_births,
            population.worker_rank,
            graph.maximum_operands(),
            population.additional_sort_kernels,
            evaluations,
            waits,
        )?;
        population.worker_graph_extents = worker.allocation_extents().checked_add(population.copy_rank_extents)?.checked_add(population.parallel_graph_extents)?;
        self.include_eval(traversal, population)?;
        self.query_controls = self.query_controls.max(worker.control_bytes()?);
        for count in roots.iter() {
            let limits = nested_traversal_with_roots(traversal, count)?.limits();
            // Each actual completion constructs its own synchronization/root
            // storage. Preserve the real distribution instead of max times count.
            let synchronizer = OperationEvent::eval_record_layout(
                limits.tape_entries, limits.streams, limits.output_slots,
            )?;
            let event_roots = OperationEvent::root_storage_layout(count)?;
            self.add(synchronizer.host_graph_allocation_extents())?;
            self.add(event_roots.graph_request_extent())?;
            self.query_controls = self.query_controls.max(synchronizer.query_control_bytes()?);
        }
        let controls = [
            size_of::<ResidentDispatchPopulation>(),
            size_of::<safemlx::ResidentGpuWorkerLayout>(),
            size_of::<Option<safemlx::ResidentGpuWorkerLayout>>(),
            size_of::<safemlx::ResidentGraphLayout>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(),
            8 * size_of::<usize>(),
            size_of::<NestedCompletionRoots<'_>>(),
        ];
        self.query_controls = self.query_controls.checked_add(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)?,
        )?;
        Some(())
    }
    fn include_eval_construction(
        &mut self, traversal: safemlx::OperationEvalTraversalLayout,
    ) -> Option<usize> {
        let limits = traversal.limits();
        let synchronizer = OperationEvent::eval_record_layout(
            limits.tape_entries, limits.streams, limits.output_slots,
        )?;
        self.add(synchronizer.host_graph_allocation_extents())?;
        let parts = [synchronizer.query_control_bytes()?,
            size_of::<(&mut Self, safemlx::OperationEvalTraversalLayout)>(),
            size_of::<Option<usize>>(), size_of::<safemlx::OperationEvalTraversalLimits>()];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// One actual CPU Contiguous and its private completion. A pending copy
    /// installs the shared three-entry traversal ceiling for both frontiers;
    /// the extra entry does not create another copy worker or output birth.
    pub(super) fn include_cpu_copy_eval(
        &mut self, traversal: safemlx::OperationEvalTraversalLayout, rank: usize,
    ) -> Option<()> {
        let copy = OperationEvent::cpu_contiguous_layout(rank, false)?;
        if copy.backing_births() != 1 { return None; }
        self.include_cpu_copy_sources(traversal, [Some(copy), None], 2)?;
        let parts = [size_of::<safemlx::CpuCopyEvalLayout>(),
            size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            size_of::<(&mut Self, safemlx::OperationEvalTraversalLayout, usize)>(),
            size_of::<Option<()>>()];
        self.query_controls = self.query_controls.checked_add(
            parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)?)?;
        Some(())
    }

    /// Actual pending tail after the isolate has completed. The shape contract
    /// gives [1,N] at rank two, otherwise all dimensions are singleton. Reshape
    /// is an alias when present; only the signed cast can produce new backing.
    pub(super) fn include_cpu_pending_eval(
        &mut self, traversal: safemlx::OperationEvalTraversalLayout,
        rank: usize, elements: usize, cast: bool,
    ) -> Option<()> {
        if elements == 0 || (rank != 2 && elements != 1) { return None; }
        let reshape = if rank != 2 {
            let source = OperationEvent::cpu_reshape_alias_layout(rank, 2, false)?;
            if source.backing_births() != 0 || source.worker_graph_allocation_extents() != 0 { return None; }
            Some(source)
        } else { None };
        let conversion = if cast {
            let source = OperationEvent::cpu_cast_layout(safemlx::Dtype::Int32,
                safemlx::Dtype::Uint32, 2, elements, false)?;
            if source.backing_births() != 1 { return None; }
            Some(source)
        } else { None };
        let entries = 1 + usize::from(reshape.is_some()) + usize::from(conversion.is_some());
        self.include_cpu_copy_sources(traversal, [reshape, conversion], entries)?;
        let parts = [size_of::<[Option<safemlx::CpuCopyEvalLayout>; 2]>(),
            size_of::<safemlx::CpuCopyEvalLayout>() * 2,
            size_of::<Option<safemlx::CpuCopyEvalLayout>>() * 2,
            size_of::<(&mut Self, safemlx::OperationEvalTraversalLayout, usize, usize, bool)>(),
            size_of::<usize>(), size_of::<Option<()>>()];
        self.query_controls = self.query_controls.checked_add(
            parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)?)?;
        Some(())
    }

    fn include_cpu_copy_sources(&mut self, traversal: safemlx::OperationEvalTraversalLayout,
        sources: [Option<safemlx::CpuCopyEvalLayout>; 2], entries: usize,
    ) -> Option<()> {
        let limits = traversal.limits();
        if limits.streams != 1 || limits.roots != 1 || !(2..=3).contains(&limits.tape_entries)
            || entries == 0 || entries > limits.tape_entries { return None; }
        let completion = OperationEvent::cpu_completion_layout(1)?;
        if completion.backing_births() != 0 { return None; }
        let mut controls = self.include_eval_construction(traversal)?;
        for source in [sources[0], sources[1], Some(completion)].into_iter().flatten() {
            self.add(source.graph_allocation_extents())?;
            self.add(source.worker_graph_allocation_extents())?;
            self.add(source.signal_graph_allocation_extents())?;
            controls = controls.checked_add(source.control_bytes()?)?;
        }
        let parts = [controls, size_of::<[Option<safemlx::CpuCopyEvalLayout>; 3]>(),
            size_of::<std::iter::Flatten<std::array::IntoIter<Option<safemlx::CpuCopyEvalLayout>, 3>>>(),
            size_of::<std::array::IntoIter<Option<safemlx::CpuCopyEvalLayout>, 3>>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(),
            size_of::<safemlx::CpuCopyEvalLayout>() * 2,
            size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            size_of::<(&mut Self, safemlx::OperationEvalTraversalLayout, [Option<safemlx::CpuCopyEvalLayout>; 2], usize)>(),
            size_of::<Option<()>>(), size_of::<usize>()];
        self.query_controls = self.query_controls.max(
            parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)?,
        );
        Some(())
    }

    pub(super) fn include_eval(
        &mut self,
        traversal: safemlx::OperationEvalTraversalLayout,
        dispatch: ResidentDispatchPopulation,
    ) -> Option<()> {
        if (dispatch.gpu_entries == 0
            && (dispatch.gpu_input_edges != 0 || dispatch.gpu_siblings != 0))
            || (dispatch.cpu_entries == 0
                && (dispatch.cpu_input_edges != 0 || dispatch.cpu_siblings != 0))
            || dispatch.gpu_entries.checked_add(dispatch.cpu_entries)? == 0
        {
            return None;
        }
        if let Some(source) = dispatch.cpu_model {
            let limits=traversal.limits();
            if dispatch.gpu_entries!=0
                || dispatch.worker_graph_extents!=0 || dispatch.copy_rank_extents!=0 || dispatch.kernel_attempts!=0
                || limits.streams!=1 + usize::from(dispatch.parallel_entries != 0)
                || dispatch.cpu_entries!=source.primitives.checked_add(dispatch.parallel_entries)?.checked_add(1)?
                || dispatch.cpu_siblings!=dispatch.cpu_entries || limits.tape_entries!=dispatch.cpu_entries
                || limits.input_edges<source.input_edges.checked_add(dispatch.parallel_entries)?.checked_add(limits.roots)?
                || dispatch.cpu_input_edges<source.input_edges.checked_add(dispatch.parallel_entries)? {
                return None;
            }
            let completion=OperationEvent::cpu_completion_layout(limits.roots)?;
            if completion.backing_births()!=0 || completion.worker_graph_allocation_extents()!=0 {return None;}
            let controls=self.include_eval_construction(traversal)?;
            self.add(source.extents)?;
            self.add(dispatch.parallel_graph_extents)?;
            self.add(completion.graph_allocation_extents())?;
            self.add(completion.signal_graph_allocation_extents())?;
            let parts=[controls,source.controls,completion.control_bytes()?,
                size_of::<super::super::cpu::CpuPopulation>(),size_of::<safemlx::CpuCopyEvalLayout>(),
                size_of::<Option<safemlx::CpuCopyEvalLayout>>(),size_of::<safemlx::OperationEvalTraversalLimits>()];
            self.query_controls=self.query_controls.max(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?);
            return Some(());
        }
        let mut controls = self.include_eval_construction(traversal)?;
        let mut prologue = 0usize;
        if dispatch.gpu_entries != 0 {
            // Original traversal authentication forbids tracing/export. The
            // output vector remains real even for entries with no input edges.
            let layout = OperationEvent::gpu_eval_prologue_layout(
                dispatch.gpu_input_edges,
                dispatch.gpu_siblings,
            )?;
            let population = layout.population(
                dispatch.gpu_entries,
                dispatch.gpu_input_edges,
                dispatch.gpu_siblings,
            )?;
            let requests = layout.requests();
            for (bytes, _) in requests[..2].iter().copied().chain(layout.owner_requests()) {
                prologue = prologue.checked_add(repeated(bytes, dispatch.gpu_entries)?)?;
            }
            prologue = prologue.checked_add(SubmissionGraphQuota::allocation_population_extent(
                population.data_slot_bytes(),
                population.vector_allocations(),
            )?)?;
            prologue = prologue.checked_add(SubmissionGraphQuota::allocation_population_extent(
                population.output_array_bytes(),
                population.evaluations(),
            )?)?;
            if population.tracer_array_bytes() != 0 {
                return None;
            }
            controls = controls.checked_add(layout.control_bytes()?)?;
        }
        let router_entries=dispatch.cpu_entries.checked_sub(dispatch.parallel_entries)?;
        if router_entries != 0 {
            // This profile has only rank-two, one-input/no-sibling CPU
            // ArgPartition. Its owning bank includes outputs(), task weak
            // views, Data/birth and cleanup; do not add generic cleanup again.
            if dispatch.cpu_input_edges != dispatch.cpu_entries
                || dispatch.cpu_siblings != dispatch.cpu_entries
            {
                return None;
            }
            let layout = OperationEvent::cpu_argpartition_layout(false)?;
            prologue = prologue.checked_add(
                layout
                    .allocation_extents()
                    .checked_mul(router_entries)?,
            )?;
            controls = controls.checked_add(layout.control_bytes()?)?;
        }
        // Worker fields exclude every bank above. Their own typed producer
        // covers Data/births, task aliases and encoder resource/fence/temporary/
        // receipt requests, without early retirement or donation credit.
        self.add(dispatch.worker_graph_extents)?;
        self.add(prologue)?;
        self.query_controls = self.query_controls.max(controls);
        Some(())
    }
}
fn repeated(bytes: usize, attempts: usize) -> Option<usize> {
    if bytes == 0 {
        return Some(0);
    }
    SubmissionGraphQuota::allocation_population_extent(bytes.checked_mul(attempts)?, attempts)
}
impl ResidentNativeRecipe {
    pub(crate) fn graph_storage_requirement(
        &self,
        prompt: safemlx::OriginalPromptInputFacts,
    ) -> Result<ResidentGraphStorage, crate::backend::error::Error> {
        self.graph_storage_requirement_with_preparation(prompt.graph_bytes())
    }
    /// The completed original source owns its already-created input Graph.
    /// Only equation/sampling/request controls are constructed by this Q.
    pub(crate) fn completed_input_graph_storage_requirement(
        &self,
        source: &eredu_runtime::input::OriginalPreparedWorkspaceSource,
    ) -> Result<ResidentGraphStorage, crate::backend::error::Error> {
        if source.borrowed_storage().is_none() || self.resume_copy.is_some() {
            return Err(crate::backend::error::Error::PrefillControl(
                WorkingMemoryError::IdentityMismatch,
            ));
        }
        let mut fit = self.graph_storage_requirement_with_preparation(0)?;
        fit.query_controls = fit.query_controls.checked_add(
            std::mem::size_of::<&eredu_runtime::input::OriginalPreparedWorkspaceSource>(),
        ).ok_or(crate::backend::error::Error::PrefillControl(WorkingMemoryError::Overflow))?;
        Ok(fit)
    }
    pub(crate) fn resume_graph_storage_requirement(
        &self,
    ) -> Result<ResidentGraphStorage, crate::backend::error::Error> {
        let copy = self
            .resume_copy
            .ok_or(crate::backend::error::Error::PrefillControl(
                WorkingMemoryError::UnknownBound,
            ))?;
        self.graph_storage_requirement_with_preparation(copy.layout().graph_extents)
    }
    pub(super) fn graph_storage_requirement_with_preparation(
        &self,
        preparation_graph_extents: usize,
    ) -> Result<ResidentGraphStorage, crate::backend::error::Error> {
        self.graph_storage_requirement_with_rows(preparation_graph_extents, self.records())
    }
    pub(super) fn graph_storage_requirement_with_rows(
        &self,
        preparation_graph_extents: usize,
        records: &[ResidentSpanRecipe],
    ) -> Result<ResidentGraphStorage, crate::backend::error::Error> {
        let unknown =
            || crate::backend::error::Error::PrefillControl(WorkingMemoryError::UnknownBound);
        let overflow =
            || crate::backend::error::Error::PrefillControl(WorkingMemoryError::Overflow);
        let mut required = ResidentGraphStorage::default();
        // The eager final-shape prompt producer has one exact descriptor,
        // Data/birth and C shell; its mutable bytes are charged in P separately.
        required
            .add(preparation_graph_extents)
            .ok_or_else(overflow)?;
        let root_capacity = records.iter().try_fold(0usize, |maximum, row| {
            row.traversal.map(|layout| maximum.max(layout.roots()))
        }).ok_or_else(unknown)?;
        let roots = safemlx::PrefillRoots::layout(root_capacity).map_err(|_| unknown())?;
        required.query_controls = roots.control_bytes;
        let mut complete = !records.is_empty();
        for row in records {
            // Every prefill InputTransaction and later ModelExecution owns the
            // actual global-capacity retained/submitted buffers. They are made
            // before dispatch and never replenished after a failed attempt.
            required.add(roots.graph_bytes).ok_or_else(overflow)?;
            match row.graph {
                Some(graph) => required
                    .add(graph.allocation_extents())
                    .ok_or_else(overflow)?,
                None => complete = false,
            }
            match row.traversal.zip(row.dispatch) {
                Some((traversal, dispatch)) => {
                    required
                        .include_dag(
                            traversal,
                            row.graph.ok_or_else(unknown)?,
                            dispatch,
                            NestedCompletionRoots::for_row(row),
                            self.neural_waits_per_forward(),
                        )
                        .ok_or_else(unknown)?;
                }
                None => complete = false,
            }
            if let Some(copies) = self.host_copies {
                required.include_source_copies(copies, self.host_transfers.ok_or_else(unknown)?,
                    row.dispatch.ok_or_else(unknown)?).ok_or_else(unknown)?;
            }
            complete &= row.unqualified_kernel_owner.is_none();
        }
        for row in self.sampling_records() {
            match row.phase() {
                SamplingWorkspacePhase::Preparation => {
                    complete &= row.completion().is_none();
                    match row.preparation {
                        Some(ResidentSamplingPreparation::Empty) => {}
                        Some(ResidentSamplingPreparation::EagerKey(graph)) => {
                            required
                                .add(graph.allocation_extents())
                                .ok_or_else(overflow)?;
                        }
                        None => complete = false,
                    }
                }
                SamplingWorkspacePhase::Step { .. } => match row.completion() {
                    Some(completion) => {
                        required
                            .add(completion.graph.allocation_extents())
                            .ok_or_else(overflow)?;
                        // ReadToken moves this same completion into Submission;
                        // outer scalar observation issues no second Eval/event.
                        let event =
                            OperationEvent::root_storage_layout(completion.traversal.roots())
                                .ok_or_else(unknown)?;
                        required
                            .add(event.graph_request_extent())
                            .ok_or_else(overflow)?;
                        match completion.dispatch {
                            Some(dispatch) => {
                                required
                                    .include_dag(
                                        completion.traversal,
                                        completion.graph,
                                        dispatch,
                                        NestedCompletionRoots::uniform(completion.nested_completions, completion.nested_root_capacity.max(3)),
                                        0,
                                    )
                                    .ok_or_else(unknown)?;
                            }
                            None => complete = false,
                        }
                    }
                    None => complete = false,
                },
            }
        }
        if complete {
            let extents =
                usize::try_from(required.known_constructor_bytes).map_err(|_| overflow())?;
            required.full_capacity = Some(
                u64::try_from(
                    SubmissionGraphQuota::fresh_capacity_for_extents(extents)
                        .ok_or_else(overflow)?,
                )
                .map_err(|_| overflow())?,
            );
        }
        Ok(required)
    }
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod prediction_tests {
    use super::*;
    use eredu_nn::Tensor;
    use eredu_nn::workspace::{WorkspaceContext, WorkspaceTensor, WorkspaceDtype, WorkspaceLayout, WorkspaceOperationKind};

    #[test]
    fn prediction_completion_root_distribution_prices_graph_and_record_frontiers() {
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(mechanism);
        let source = WorkspaceTensor::unloaded_f32(&[1, 5], &context).unwrap();
        context.begin_span();
        let output = context.execute(
            WorkspaceOperationKind::Elementwise("capture_cast_f32"), &[&source],
            vec![WorkspaceLayout::new(&[1, 5], WorkspaceDtype::Float32).unwrap()],
        ).unwrap().remove(0);
        let mut completion = original_component_tests::OriginalComponentTestPlan::from_report(
            context.report(&[output]).unwrap(),
        ).completion;
        let graph = |roots: NestedCompletionRoots<'_>| {
            let mut value = ResidentGraphStorage::default();
            value.include_dag(completion.traversal, completion.graph,
                completion.dispatch.unwrap(), roots, 0).unwrap();
            value.known_constructor_bytes
        };
        let old_fixed = graph(NestedCompletionRoots::uniform(2, 3));
        let actual = graph(NestedCompletionRoots {
            fixed_attempts: 0, fixed_roots: 3, additional: &[1, 17],
        });
        let padded = graph(NestedCompletionRoots::uniform(2, 17));
        assert!(actual > old_fixed, "the second native root list must be funded");
        assert!(actual < padded, "the short first list must retain its actual population");

        completion.nested_completions = 2;
        completion.nested_root_capacity = 3;
        let small = record_capacity::ResidentRecordStorage::for_completion(completion).unwrap();
        completion.nested_root_capacity = 17;
        let large = record_capacity::ResidentRecordStorage::for_completion(completion).unwrap();
        assert!(large.known_constructor_bytes > small.known_constructor_bytes);
        assert!(large.minimum_capacity > small.minimum_capacity);
        assert_eq!(completion.nested_traversal().unwrap().roots(), 17);
    }
}
