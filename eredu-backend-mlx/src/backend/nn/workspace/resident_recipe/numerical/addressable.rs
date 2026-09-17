//! Composition of already qualified ordinary sources in one original arena.
//! Branch union takes scalar maxima; sequential composition adds the actual
//! occurrences. Neither operation traces or implements a numerical kernel.
use super::*;
use safemlx::{OperationEvalTraversalLimits, OperationEvent};

#[derive(Clone, Copy, Debug)]
pub(crate) struct AddressableNumericalPopulation {
    cpu: Option<super::super::super::cpu::CpuPopulation>,
    primitives: usize,
    seeds: usize,
    rank: usize,
    operands: usize,
    shells: usize,
    arrays: usize,
    edges: usize,
    streams: usize,
    cpu_entries: usize,
    cpu_edges: usize,
    cpu_births: usize,
    gpu_births: usize,
    worker_rank: usize,
    copy_extents: usize,
    sorts: usize,
    validations: usize,
    grouped: GroupedOutputStorage,
    nested: usize,
    nested_roots: usize,
    bytes: u64,
    births: usize,
    controls: u64,
}
impl AddressableNumericalPopulation {
    /// A child's final completion becomes the existing bank.complete call.
    /// The outer parent keeps its final completion as the one role exit.
    pub(crate) fn from_recipe(value: SpeculativeNumericalRecipe, child: bool) -> Option<Self> {
        let completion = value.completion;
        let dispatch = completion.dispatch?;
        if dispatch.parallel_entries != 0 || dispatch.parallel_graph_extents != 0 {
            return None;
        }
        let limits = completion.traversal.limits();
        Some(Self {
            cpu: dispatch.cpu_model,
            primitives: completion.graph.primitives(),
            seeds: completion.graph.seeds(),
            rank: completion.graph.maximum_rank(),
            operands: completion.graph.maximum_operands(),
            shells: completion.graph.additional_shells(),
            // Each source was inspected with its own hypothetical final
            // Synchronizer/root list. The composed scope creates these once.
            arrays: limits.arrays.checked_sub(limits.roots)?.checked_sub(1)?,
            edges: limits.input_edges.checked_sub(limits.roots)?,
            streams: limits.streams,
            cpu_entries: if dispatch.cpu_model.is_some() {
                0
            } else {
                dispatch.cpu_entries
            },
            cpu_edges: if dispatch.cpu_model.is_some() {
                0
            } else {
                dispatch.cpu_input_edges
            },
            cpu_births: if dispatch.cpu_model.is_some() {
                0
            } else {
                value
                    .storage
                    .maximum_births()
                    .checked_sub(dispatch.gpu_births)?
            },
            gpu_births: dispatch.gpu_births,
            worker_rank: dispatch.worker_rank,
            copy_extents: dispatch.copy_rank_extents,
            sorts: dispatch.additional_sort_kernels,
            validations: completion.validation_roots,
            grouped: completion.grouped_outputs,
            nested: completion
                .nested_completions
                .checked_sub(usize::from(!child))?,
            nested_roots: completion.nested_root_capacity,
            bytes: value.storage.mutable_bytes(),
            births: value.storage.maximum_births(),
            controls: value.controls,
        })
    }
    /// Closed alternatives for one actual compact invocation. Output/source
    /// geometry is checked by the caller before these populations are joined.
    pub(crate) fn union(self, other: Self) -> Option<Self> {
        self.combine(other, true)
    }
    /// Two actual source occurrences in the same lifetime and native arena.
    pub(crate) fn append(self, other: Self) -> Option<Self> {
        self.combine(other, false)
    }
    fn combine(mut self, other: Self, alternative: bool) -> Option<Self> {
        let count = |a: usize, b: usize| {
            if alternative {
                Some(a.max(b))
            } else {
                a.checked_add(b)
            }
        };
        let bytes = |a: u64, b: u64| {
            if alternative {
                Some(a.max(b))
            } else {
                a.checked_add(b)
            }
        };
        macro_rules! counts { ($($field:ident),*) => { $(self.$field=count(self.$field,other.$field)?;)* }; }
        counts!(
            primitives,
            seeds,
            shells,
            arrays,
            edges,
            cpu_entries,
            cpu_edges,
            cpu_births,
            gpu_births,
            copy_extents,
            sorts,
            validations,
            nested,
            births
        );
        self.bytes = bytes(self.bytes, other.bytes)?;
        self.controls = bytes(self.controls, other.controls)?;
        self.rank = self.rank.max(other.rank);
        self.operands = self.operands.max(other.operands);
        self.streams = self.streams.max(other.streams);
        self.worker_rank = self.worker_rank.max(other.worker_rank);
        self.nested_roots = self.nested_roots.max(other.nested_roots);
        self.grouped = GroupedOutputStorage {
            calls: count(self.grouped.calls, other.grouped.calls)?,
            chunks: self.grouped.chunks.max(other.grouped.chunks),
            unit_observers: count(self.grouped.unit_observers, other.grouped.unit_observers)?,
            observer_shape_rank: self
                .grouped
                .observer_shape_rank
                .max(other.grouped.observer_shape_rank),
        };
        self.cpu = match (self.cpu, other.cpu) {
            (None, None) => None,
            (Some(a), Some(b)) => Some(super::super::super::cpu::CpuPopulation {
                primitives: count(a.primitives, b.primitives)?,
                input_edges: count(a.input_edges, b.input_edges)?,
                births: count(a.births, b.births)?,
                extents: count(a.extents, b.extents)?,
                controls: count(a.controls, b.controls)?,
            }),
            _ => return None,
        };
        Some(self)
    }
    pub(crate) fn finish_with_residency(
        self, outputs: usize, id_completions: usize,
        residency: &crate::backend::runtime::residency::parameter_bank::IndexedResidencyPlan,
        context: &WorkspaceContext,
    ) -> Result<SpeculativeNumericalRecipe, Error> {
        use super::super::host_copies::PreparedSourceCopies;
        let invalid = || context.metadata_error(format_args!(
            "addressable transfer source differs from its actual numerical population"));
        context.charge_metadata(std::mem::size_of::<(
            PreparedSourceCopies, ResidentCompletionRecipe, ResidentDispatchPopulation,
            OperationEvalTraversalLimits, SpeculativeNumericalRecipe,
            Result<SpeculativeNumericalRecipe, Error>, [usize; 8],
        )>())?;
        context.charge_metadata(PreparedSourceCopies::inspection_control_bytes().ok_or_else(invalid)?)?;
        let source = residency.with_native_copy_source(|source, window, calls|
            PreparedSourceCopies::inspect(source, window, calls))
            .map_err(|cause|context.metadata_source(cause))?;
        self.finish(outputs, id_completions, context)?.with_prepared_source_copies(source, context)
    }

    pub(crate) fn finish(
        self,
        outputs: usize,
        id_completions: usize,
        context: &WorkspaceContext,
    ) -> Result<SpeculativeNumericalRecipe, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "addressable source population cannot form one native role"
            ))
        };
        context.charge_metadata(std::mem::size_of::<(
            Self,
            SpeculativeNumericalRecipe,
            ResidentCompletionRecipe,
            ResidentDispatchPopulation,
            OperationEvalTraversalLimits,
            Result<SpeculativeNumericalRecipe, Error>,
            [usize; 12],
        )>())?;
        let roots = outputs.checked_add(self.validations).ok_or_else(invalid)?;
        let nested = self
            .nested
            .checked_add(id_completions)
            .ok_or_else(invalid)?;
        let root_capacity = roots
            .max(self.nested_roots)
            .max(usize::from(id_completions != 0));
        let tape = self.primitives.checked_add(1).ok_or_else(invalid)?;
        let arrays = self
            .arrays
            .checked_add(root_capacity)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(invalid)?;
        let edges = self.edges.checked_add(root_capacity).ok_or_else(invalid)?;
        let base =
            OperationEvent::eval_record_layout(tape, self.streams, tape).ok_or_else(invalid)?;
        let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
            roots,
            arrays,
            tape_entries: tape,
            input_edges: edges,
            output_slots: tape,
            streams: self.streams,
            captures: base.capture_slots().max(1),
        })
        .ok_or_else(invalid)?;
        let graph = OperationEvent::resident_graph_layout_with_shells(
            self.primitives,
            self.seeds,
            self.rank,
            self.operands,
            self.shells,
        )
        .ok_or_else(invalid)?;
        let dispatch = if let Some(cpu) = self.cpu {
            if self.cpu_entries != 0 || self.sorts != 0 || self.streams != 1 {
                return Err(invalid());
            }
            ResidentDispatchPopulation {
                cpu_model: Some(cpu),
                gpu_entries: 0,
                gpu_input_edges: 0,
                gpu_siblings: 0,
                gpu_births: 0,
                additional_sort_kernels: 0,
                cpu_entries: tape,
                cpu_input_edges: edges,
                cpu_siblings: tape,
                parallel_entries: 0,
                parallel_graph_extents: 0,
                worker_graph_extents: 0,
                worker_rank: self.worker_rank,
                copy_rank_extents: 0,
                kernel_attempts: 0,
            }
        } else {
            let gpu_entries = tape.checked_sub(self.cpu_entries).ok_or_else(invalid)?;
            let gpu_edges = edges.checked_sub(self.cpu_edges).ok_or_else(invalid)?;
            let worker = OperationEvent::resident_gpu_worker_layout_with_router(
                gpu_entries,
                gpu_edges,
                gpu_entries,
                arrays,
                self.gpu_births,
                self.worker_rank,
                self.operands,
                self.sorts,
                self.cpu_entries,
            )
            .ok_or_else(invalid)?;
            ResidentDispatchPopulation {
                cpu_model: None,
                gpu_entries,
                gpu_input_edges: gpu_edges,
                gpu_siblings: gpu_entries,
                gpu_births: self.gpu_births,
                additional_sort_kernels: self.sorts,
                cpu_entries: self.cpu_entries,
                cpu_input_edges: self.cpu_edges,
                cpu_siblings: self.cpu_entries,
                parallel_entries: 0,
                parallel_graph_extents: 0,
                worker_graph_extents: worker
                    .allocation_extents()
                    .checked_add(self.copy_extents)
                    .ok_or_else(invalid)?,
                worker_rank: self.worker_rank,
                copy_rank_extents: self.copy_extents,
                kernel_attempts: worker.kernel_attempts(),
            }
        };
        let completion = ResidentCompletionRecipe {
            validation_roots: self.validations,
            grouped_outputs: self.grouped,
            traversal,
            graph,
            dispatch: Some(dispatch),
            nested_completions: nested,
            nested_root_capacity: root_capacity,
        };
        let graph =
            graph_capacity::ResidentGraphStorage::for_completion(completion).ok_or_else(invalid)?;
        let records = record_capacity::ResidentRecordStorage::for_completion(completion)
            .ok_or_else(invalid)?;
        let controls = self
            .controls
            .checked_add(graph.control_bytes().ok_or_else(invalid)?)
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(invalid)?;
        Ok(SpeculativeNumericalRecipe {
            completion,
            storage: CertifiedSpanStorage {
                mutable_bytes: self.bytes,
                maximum_births: self.births,
            },
            graph_capacity: usize::try_from(graph.full_capacity.ok_or_else(invalid)?)
                .map_err(|_| invalid())?,
            record_capacity: usize::try_from(records.full_capacity.ok_or_else(invalid)?)
                .map_err(|_| invalid())?,
            kernels: dispatch.kernel_attempts,
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
    use eredu_nn::Tensor;
    fn value(context: &WorkspaceContext, width: i32) -> WorkspaceTensor {
        WorkspaceTensor::existing(
            context
                .layout(&[1, width], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            context,
        )
        .unwrap()
    }
    #[test]
    fn addressable_population_composes_one_scope_and_real_nested_frontiers() {
        let metal = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let mechanism = ResidentExecutionMechanisms::Metal(metal);
        for width in [1, 19, 257] {
            let child_context = WorkspaceContext::new(mechanism);
            let input = value(&child_context, width);
            child_context.begin_span();
            let output = input.square(&child_context).unwrap();
            let report = child_context.finish_report(&[output]).unwrap();
            let child = SpeculativeNumericalRecipe::inspect_owned_child(
                &report,
                1,
                mechanism,
                &child_context,
            )
            .unwrap();
            let child_population =
                AddressableNumericalPopulation::from_recipe(child, true).unwrap();
            let branch = child_population.union(child_population).unwrap();

            let parent_context = WorkspaceContext::new(mechanism);
            let inputs = [value(&parent_context, width), value(&parent_context, width)];
            parent_context.begin_span();
            let output = WorkspaceTensor::concatenate(&inputs, 0, &parent_context).unwrap();
            let report = parent_context.finish_report(&[output]).unwrap();
            let parent = SpeculativeNumericalRecipe::inspect_owned_child(
                &report,
                1,
                mechanism,
                &parent_context,
            )
            .unwrap();
            let population = AddressableNumericalPopulation::from_recipe(parent, false)
                .unwrap()
                .append(branch)
                .unwrap()
                .append(branch)
                .unwrap();
            let joined = population.finish(1, 2, &parent_context).unwrap();

            let full_context = WorkspaceContext::new(mechanism);
            let inputs = [value(&full_context, width), value(&full_context, width)];
            full_context.begin_span();
            let inputs = [
                inputs[0].square(&full_context).unwrap(),
                inputs[1].square(&full_context).unwrap(),
            ];
            let output = WorkspaceTensor::concatenate(&inputs, 0, &full_context).unwrap();
            let report = full_context.finish_report(&[output]).unwrap();
            let full = SpeculativeNumericalRecipe::inspect_owned_child(
                &report,
                1,
                mechanism,
                &full_context,
            )
            .unwrap();
            assert_eq!(
                joined.completion.graph.primitives(),
                full.completion.graph.primitives()
            );
            assert_eq!(joined.storage.mutable_bytes(), full.storage.mutable_bytes());
            assert_eq!(
                joined.storage.maximum_births(),
                full.storage.maximum_births()
            );
            assert_eq!(
                joined.completion.traversal.limits().tape_entries,
                joined.completion.graph.primitives() + 1
            );
            assert_eq!(
                joined.completion.nested_completions, 4,
                "two ID reads and two completed chunk outputs"
            );
            assert!(joined.graph_capacity >= full.graph_capacity);
            assert!(joined.record_capacity >= full.record_capacity);
        }
    }
}
