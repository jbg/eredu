//! Ordinary Metal and CPU communication controls from retained shared workers.
use super::super::graph_capacity::ResidentGraphStorage;
use super::*;
use safemlx::OperationEvent;

impl ResidentCompletionRecipe {
    /// The ordinary containers differ from fixed Original Eval storage. The
    /// numerical worker, prologue and encoder sources are shared; their unused
    /// conservative dispatch allowance is released by the ordinary account.
    pub(crate) fn ordinary_metal_controls(
        self,
        consumer_waits: usize,
    ) -> Option<OrdinaryNativeControls> {
        let mut query = OrdinaryQueryMetadata::new();
        let controls = self.ordinary_metal_controls_with(
            consumer_waits,
            None,
            std::iter::empty(),
            &mut query,
        )?;
        query.bytes?;
        Some(controls)
    }

    /// Communication constructors and their actual completion frontiers can
    /// reach the enclosing lazy GPU graph. Join that source before querying
    /// Eval, Fence and dispatch controls; a CPU-only Eval quote cannot describe
    /// the GPU inputs of a Ring operation.
    pub(super) fn ordinary_metal_controls_with(
        self,
        consumer_waits: usize,
        extra: Option<OrdinaryCpuPopulation>,
        frontiers: impl IntoIterator<Item = (usize, usize)>,
        query: &mut OrdinaryQueryMetadata,
    ) -> Option<OrdinaryNativeControls> {
        query.include(Some(size_of::<(
            Self,
            &mut OrdinaryQueryMetadata,
            Option<OrdinaryCpuPopulation>,
            [ResidentDispatchPopulation; 2],
            [OrdinaryNativeControls; 3],
            Option<(usize, usize, bool)>,
            Option<OrdinaryNativeControls>,
            safemlx::OperationEvalTraversalLayout,
            safemlx::OperationEvalTraversalLimits,
            safemlx::ResidentGpuWorkerLayout,
            Option<safemlx::ResidentGpuWorkerLayout>,
            ResidentGraphStorage,
            [usize; 10],
            bool,
        )>()));
        query.include(Some(size_of_val(&frontiers)));
        let dispatch = self.dispatch?;
        if dispatch.cpu_model.is_some()
            || dispatch.completion_streams()? != 1 + usize::from(dispatch.cpu_entries != 0)
        {
            return None;
        }
        // A CPU router and a Ring source retain different streams. Their
        // three-stream composition needs its own source; this worker qualifies
        // the existing one-GPU/one-CPU Eval mechanism only.
        if extra.is_some() && dispatch.cpu_entries != dispatch.parallel_entries {
            return None;
        }
        let mut result = OrdinaryNativeControls::default();
        result.include(OperationEvent::ordinary_frontend_control_layout(
            self.graph.primitives(),
            self.graph.seeds(),
            self.graph.maximum_rank(),
            self.graph.maximum_operands().max(4),
        )?)?;
        if let Some(extra) = extra {
            if extra.streams == 0 || extra.streams > 2 {
                return None;
            }
            result.include(OperationEvent::ordinary_frontend_control_layout(
                extra.construction_entries,
                extra.seeds,
                extra.maximum_rank,
                extra.maximum_operands.max(4),
            )?)?;
            result.include(OperationEvent::ordinary_cpu_dispatch_envelope(
                extra.dispatch_graph_extents,
            )?)?;
        }
        let nested = (self.nested_completions != 0)
            .then(|| {
                let traversal = self.nested_traversal()?;
                query.include(traversal.query_control_bytes());
                Some((traversal.roots(), self.nested_completions, false))
            })
            .flatten();
        if self.nested_completions != 0 && nested.is_none() {
            return None;
        }
        let frontiers = std::iter::once((self.traversal.roots(), 1, true))
            .chain(nested)
            .chain(
                frontiers
                    .into_iter()
                    .map(|(roots, count)| (roots, count, false)),
            );
        query.include(Some(size_of_val(&frontiers)));
        for (roots, count, final_frontier) in frontiers {
            if count == 0 {
                continue;
            }
            let traversal = if roots == self.traversal.roots() {
                self.traversal
            } else {
                let traversal = nested_traversal_with_roots(self.traversal, roots)?;
                query.include(traversal.query_control_bytes());
                traversal
            };
            let mut limits = traversal.limits();
            let mut population = dispatch;
            population.gpu_input_edges = population
                .gpu_input_edges
                .checked_sub(self.traversal.roots())?
                .checked_add(roots)?;
            let mut cpu_crossings = population.cpu_entries;
            let mut rank = population.worker_rank;
            let mut operands = self.graph.maximum_operands();
            if let Some(extra) = extra {
                limits.arrays = limits.arrays.checked_add(extra.array_nodes)?;
                limits.tape_entries = limits.tape_entries.checked_add(extra.primitives)?;
                limits.input_edges = limits.input_edges.checked_add(extra.input_edges)?;
                limits.output_slots = limits.output_slots.max(limits.arrays);
                limits.streams = 2;
                limits.captures = limits.captures.max(extra.captures);
                // One CPU primitive can have several inputs. Use the larger
                // source population for synchronization actions; its numerical
                // dispatch remains priced by the exact CPU worker above.
                let crossings = extra.primitives.max(extra.input_edges);
                cpu_crossings = cpu_crossings.checked_add(crossings)?;
                population.gpu_input_edges = population.gpu_input_edges.checked_add(crossings)?;
                rank = rank.max(extra.maximum_rank);
                operands = operands.max(extra.maximum_operands);
            }
            if dispatch.parallel_entries != 0 || extra.is_some() {
                // Physical capacities and their ledger attachments belong to
                // the same collective's scratch source. The GPU worker must
                // also cover resources retained for both fast Fence buffers.
                population.gpu_births = population.gpu_births.checked_add(2)?;
            }
            // All real consumer boundaries close intervals in this same DAG.
            // Price them once with the final frontier; nested Eval calls still
            // have their own Synchronizer and ordinary container lifetimes.
            let waits = if final_frontier { consumer_waits } else { 0 };
            let worker = OperationEvent::resident_gpu_worker_layout_with_router_frontiers(
                population.gpu_entries,
                population.gpu_input_edges,
                population.gpu_siblings,
                limits.arrays,
                population.gpu_births,
                rank,
                operands,
                population.additional_sort_kernels,
                cpu_crossings,
                1,
                waits,
            )?;
            query.include(worker.control_bytes());
            population.worker_graph_extents = worker
                .allocation_extents()
                .checked_add(population.copy_rank_extents)?
                .checked_add(population.parallel_graph_extents)?;
            let mut selected = ResidentGraphStorage::default();
            let qualified = selected.include_ordinary_metal_dispatch(population);
            query.include(
                selected
                    .control_bytes()
                    .and_then(|bytes| usize::try_from(bytes).ok()),
            );
            qualified?;
            let mut controls = OrdinaryNativeControls::default();
            controls.include(OperationEvent::ordinary_dispatch_control_envelope(
                usize::try_from(selected.known_constructor_bytes).ok()?,
            )?)?;
            controls.include(if cpu_crossings == 0 {
                OperationEvent::ordinary_metal_eval_control_layout(limits)?
            } else {
                OperationEvent::ordinary_metal_router_eval_control_layout(limits)?
            })?;
            result = result.append(controls.repeat(count)?)?;
        }
        if consumer_waits != 0 {
            let mut waits = OrdinaryNativeControls::default();
            waits.include(OperationEvent::ordinary_metal_wait_record_control_layout()?)?;
            result = result.append(waits.repeat(consumer_waits)?)?;
        }
        Some(result)
    }
}
