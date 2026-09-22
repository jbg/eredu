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
    captures: usize,
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
/// Structural CPU populations retained from the shared equation reducer.
/// This projection excludes original arena and inspection controls; its counts
/// need the ordinary allocator, evaluation and dispatch source qualifications.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrdinaryCpuPopulation {
    pub(crate) construction_entries: usize,
    pub(crate) primitives: usize,
    pub(crate) seeds: usize,
    pub(crate) maximum_rank: usize,
    pub(crate) maximum_operands: usize,
    pub(crate) parameter_shells: usize,
    pub(crate) array_nodes: usize,
    pub(crate) input_edges: usize,
    pub(crate) streams: usize,
    pub(crate) captures: usize,
    pub(crate) nested_completions: usize,
    pub(crate) nested_roots: usize,
    pub(crate) dispatch_graph_extents: usize,
}
impl OrdinaryCpuPopulation {
    /// Component maxima across alternative executions of the same occurrence.
    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            construction_entries: self.construction_entries.max(other.construction_entries),
            primitives: self.primitives.max(other.primitives),
            seeds: self.seeds.max(other.seeds),
            maximum_rank: self.maximum_rank.max(other.maximum_rank),
            maximum_operands: self.maximum_operands.max(other.maximum_operands),
            parameter_shells: self.parameter_shells.max(other.parameter_shells),
            array_nodes: self.array_nodes.max(other.array_nodes),
            input_edges: self.input_edges.max(other.input_edges),
            streams: self.streams.max(other.streams),
            captures: self.captures.max(other.captures),
            nested_completions: self.nested_completions.max(other.nested_completions),
            nested_roots: self.nested_roots.max(other.nested_roots),
            dispatch_graph_extents: self
                .dispatch_graph_extents
                .max(other.dispatch_graph_extents),
        }
    }
    /// Structural population before its final Synchronizer. Payload capacities
    /// remain in the separately retained numerical recipe and workspace report.
    pub(crate) fn from_completion(source: ResidentCompletionRecipe) -> Option<Self> {
        let dispatch = source.dispatch?;
        let cpu = dispatch.cpu_model?;
        if dispatch.gpu_entries != 0
            || dispatch.gpu_births != 0
            || dispatch.kernel_attempts != 0
            || dispatch.completion_streams()? == 0
        {
            return None;
        }
        let limits = source.traversal.limits();
        Some(Self {
            construction_entries: cpu
                .construction_entries
                .checked_add(dispatch.parallel_entries)?,
            primitives: cpu.primitives.checked_add(dispatch.parallel_entries)?,
            seeds: source.graph.seeds(),
            maximum_rank: source.graph.maximum_rank(),
            maximum_operands: source.graph.maximum_operands(),
            parameter_shells: source.graph.additional_shells(),
            array_nodes: limits.arrays.checked_sub(limits.roots)?.checked_sub(1)?,
            input_edges: limits.input_edges.checked_sub(limits.roots)?,
            streams: limits.streams,
            captures: limits.captures.max(cpu.maximum_captures),
            nested_completions: source.nested_completions,
            nested_roots: source.nested_root_capacity,
            dispatch_graph_extents: cpu.extents.checked_add(dispatch.parallel_graph_extents)?,
        })
    }
    /// Normalize one real completion against the combined reachable graph.
    /// Every primitive output (including siblings) is a reachable array node,
    /// so that population also bounds the ordinary output-pin reserve worker.
    pub(crate) fn traversal(self, roots: usize) -> Option<OperationEvalTraversalLimits> {
        if roots == 0 || self.streams == 0 || self.primitives > self.array_nodes {
            return None;
        }
        let arrays = self.array_nodes.checked_add(roots)?.checked_add(1)?;
        let tape = self.primitives.checked_add(1)?;
        Some(OperationEvalTraversalLimits {
            roots,
            arrays,
            tape_entries: tape,
            input_edges: self.input_edges.checked_add(roots)?,
            output_slots: arrays,
            streams: self.streams,
            captures: self.captures,
        })
    }
    /// Price construction and selected CPU workers once, then the core Eval/
    /// Synchronizer sources for each actual (root count, occurrence count).
    /// A frontier can reach lazy upstream routing; callers join those source
    /// populations before this method. Already completed graph branches leave
    /// unused allowance, never fictitious measured occupancy.
    /// Caller C ArrayVector buffers belong to the concrete safe-call census;
    /// only the distinct core Eval root copy is included here.
    pub(crate) fn controls(
        self,
        frontiers: impl IntoIterator<Item = (usize, usize)>,
    ) -> Option<super::super::OrdinaryNativeControls> {
        use super::super::OrdinaryNativeControls;
        if self.streams == 0 {
            return None;
        }
        let mut result = OrdinaryNativeControls::default();
        result.include(OperationEvent::ordinary_frontend_control_layout(
            self.construction_entries,
            self.seeds,
            self.maximum_rank,
            self.maximum_operands.max(4),
        )?)?;
        result.include(OperationEvent::ordinary_cpu_dispatch_envelope(
            self.dispatch_graph_extents,
        )?)?;
        for (roots, occurrences) in frontiers {
            if occurrences == 0 {
                continue;
            }
            let limits = self.traversal(roots)?;
            let mut completion = OrdinaryNativeControls::default();
            completion.include(OperationEvent::ordinary_cpu_eval_control_layout(limits)?)?;
            let synchronizer = OperationEvent::cpu_completion_layout(roots)?;
            completion.include(OperationEvent::ordinary_cpu_dispatch_envelope(
                synchronizer
                    .graph_allocation_extents()
                    .checked_add(synchronizer.worker_graph_allocation_extents())?,
            )?)?;
            result = result.append(completion.repeat(occurrences)?)?;
        }
        Some(result)
    }
    /// One already qualified CPU worker's structural graph, without Original
    /// arena, routing or inspection controls. Inputs remain real borrowed roots.
    pub(in crate::backend::nn::workspace) fn from_operation(
        source: super::super::super::cpu::OperationPlan,
        inputs: usize,
    ) -> Option<Self> {
        if source.validations != 0 {
            return None;
        }
        let native = source.population;
        Some(Self {
            construction_entries: native.construction_entries,
            primitives: native.primitives,
            seeds: source.seeds,
            maximum_rank: source.rank,
            maximum_operands: native.maximum_operands.max(4),
            parameter_shells: source.parameter_shells,
            array_nodes: native
                .primitives
                .checked_add(inputs)?
                .checked_add(source.seeds)?
                .checked_add(native.hidden_leaves)?,
            input_edges: native.input_edges,
            streams: 1,
            captures: native.maximum_captures,
            nested_completions: 0,
            nested_roots: 0,
            dispatch_graph_extents: native.extents,
        })
    }
    pub(crate) fn append(mut self, other: Self) -> Option<Self> {
        if self.streams == 0 || other.streams == 0 {
            return None;
        }
        macro_rules! add { ($($field:ident),*) => { $(
            self.$field = self.$field.checked_add(other.$field)?;
        )* }; }
        add!(
            construction_entries,
            primitives,
            seeds,
            parameter_shells,
            array_nodes,
            input_edges,
            nested_completions,
            dispatch_graph_extents
        );
        self.maximum_rank = self.maximum_rank.max(other.maximum_rank);
        self.streams = self.streams.max(other.streams);
        self.maximum_operands = self.maximum_operands.max(other.maximum_operands);
        self.captures = self.captures.max(other.captures);
        self.nested_roots = self.nested_roots.max(other.nested_roots);
        Some(self)
    }
    pub(crate) fn repeat(mut self, count: usize) -> Option<Self> {
        macro_rules! multiply { ($($field:ident),*) => { $(
            self.$field = self.$field.checked_mul(count)?;
        )* }; }
        multiply!(
            construction_entries,
            primitives,
            seeds,
            parameter_shells,
            array_nodes,
            input_edges,
            nested_completions,
            dispatch_graph_extents
        );
        Some(self)
    }
}
impl AddressableNumericalPopulation {
    /// Payload capacities retained by the shared numerical reducer, separate
    /// from ordinary native control allowances and transfer staging.
    pub(crate) fn storage(self) -> CertifiedSpanStorage {
        CertifiedSpanStorage {
            mutable_bytes: self.bytes,
            maximum_births: self.births,
        }
    }
    pub(crate) fn ordinary_cpu_missing_source(self) -> Option<&'static str> {
        if self.cpu.is_none() {
            Some("ordinary indexed equation has no retained selected CPU worker population")
        } else if self.cpu_entries != 0 {
            Some("ordinary indexed equation has CPU entries outside its selected worker population")
        } else if self.gpu_births != 0 {
            Some("ordinary indexed equation includes GPU backing births")
        } else if self.sorts != 0 {
            Some("ordinary indexed equation includes a separate sort dispatch population")
        } else if self.streams != 1 {
            Some("ordinary indexed equation is not a single-stream source")
        } else {
            None
        }
    }
    /// Predicates registered by the genuine grouped worker remain in the
    /// enclosing TokenValidationScope. Their DAG is included in this source;
    /// the parent must append these roots and retain the conservative producer
    /// storage through final completion, including subsequent prefill chunks.
    pub(crate) fn ordinary_validation_population(self) -> (usize, CertifiedSpanStorage) {
        (
            self.validations,
            if self.validations == 0 {
                CertifiedSpanStorage {
                    mutable_bytes: 0,
                    maximum_births: 0,
                }
            } else {
                self.storage()
            },
        )
    }
    /// The returned graph includes validation producers. Only a parent that
    /// also consumes ordinary_validation_population may execute this source.
    pub(crate) fn ordinary_cpu_population(self) -> Option<OrdinaryCpuPopulation> {
        if self.ordinary_cpu_missing_source().is_some() {
            return None;
        }
        let cpu = self.cpu?;
        Some(OrdinaryCpuPopulation {
            construction_entries: cpu.construction_entries,
            primitives: cpu.primitives,
            seeds: self.seeds,
            maximum_rank: self.rank,
            maximum_operands: self.operands,
            parameter_shells: self.shells,
            array_nodes: self.arrays,
            input_edges: self.edges,
            streams: self.streams,
            captures: self.captures.max(cpu.maximum_captures),
            nested_completions: self.nested,
            nested_roots: self.nested_roots,
            dispatch_graph_extents: cpu.extents,
        })
    }
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
            captures: limits.captures,
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
        self.captures = self.captures.max(other.captures);
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
                construction_entries: count(a.construction_entries, b.construction_entries)?,
                primitives: count(a.primitives, b.primitives)?,
                input_edges: count(a.input_edges, b.input_edges)?,
                hidden_leaves: count(a.hidden_leaves, b.hidden_leaves)?,
                maximum_operands: a.maximum_operands.max(b.maximum_operands),
                maximum_captures: a.maximum_captures.max(b.maximum_captures),
                births: count(a.births, b.births)?,
                extents: count(a.extents, b.extents)?,
                controls: count(a.controls, b.controls)?,
            }),
            _ => return None,
        };
        Some(self)
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
            &super::super::super::cpu::CpuPopulation,
        )>())?;
        let roots = outputs.checked_add(self.validations).ok_or_else(invalid)?;
        let nested = self
            .nested
            .checked_add(id_completions)
            .ok_or_else(invalid)?;
        let root_capacity = roots
            .max(self.nested_roots)
            .max(usize::from(id_completions != 0));
        let tape = match &self.cpu {
            Some(cpu) => cpu.primitives,
            None => self.primitives,
        }
        .checked_add(1)
        .ok_or_else(invalid)?;
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
            captures: base.capture_slots().max(self.captures).max(1),
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
            ordinary_calls: None,
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
        check_composition(ResidentExecutionMechanisms::Metal(metal));
    }

    #[test]
    fn addressable_cpu_population_composes_one_scope_and_real_nested_frontiers() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        check_composition(ResidentExecutionMechanisms::Cpu {
            ordinary,
            cpu: super::super::super::super::MlxCpuWorkspaceMechanisms::new(
                ordinary.allocation(),
                super::super::super::super::MlxCpuMatmulMechanism::select(
                    eredu_nn::CpuMatmulImplementation::Float32Tiles,
                )
                .unwrap(),
            ),
        });
    }

    fn check_composition(mechanism: ResidentExecutionMechanisms) {
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
            if let Some(cpu) = joined.completion.dispatch.unwrap().cpu_model {
                let tape = joined.completion.traversal.limits().tape_entries;
                assert_eq!(tape, cpu.primitives + 1);
                assert_eq!(joined.completion.dispatch.unwrap().cpu_entries, tape);
            }
            assert!(joined.graph_capacity >= full.graph_capacity);
            assert!(joined.record_capacity >= full.record_capacity);
        }
    }
}
