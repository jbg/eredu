//! Same trace/record/graph construction with actual homogeneous CPU sources.
use super::super::cpu::CpuPopulation;
use super::*;
use std::mem::{size_of, size_of_val};
impl ResidentRecipeRecorder {
    pub(crate) fn with_cpu_context(
        geometry: InferenceGeometry,
        ordinary: MlxMetalWorkspaceMechanisms,
        cpu: MlxCpuWorkspaceMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let frames = [
            size_of::<CpuPopulation>() * 2,
            size_of::<GroupedOutputStorage>() * 2,
            size_of::<Result<GroupedOutputStorage, super::super::MlxWorkspaceFactError>>(),
            size_of::<MlxCpuWorkspaceMechanisms>(),
            size_of::<Option<super::super::cpu::OperationPlan>>(),
            size_of::<super::super::cpu::OperationPlan>(),
            size_of::<Option<safemlx::CpuCopyEvalLayout>>(),
            size_of::<safemlx::CpuCopyEvalLayout>(),
            size_of::<Option<safemlx::ResidentGraphLayout>>(),
            size_of::<Option<safemlx::OperationEvalTraversalLayout>>(),
            size_of::<safemlx::OperationEvalTraversalLimits>(),
            size_of::<usize>() * 29,
            size_of::<Option<&parallel::OriginalParallelInvocation>>(),
            size_of::<u64>(),
            size_of::<(
                &Self,
                &WorkspaceTraceReport,
                Option<usize>,
                usize,
                usize,
                MlxCpuWorkspaceMechanisms,
            )>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<WorkspaceOperation>>>(),
            size_of::<Result<ReducedTrace, Error>>(),
        ];
        let controls = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(controls)?;
        let mut recorder = Self::with_context(geometry, ordinary, context)?;
        recorder.cpu = Some(cpu);
        Ok(recorder)
    }
    fn missing_cpu_source(
        &self, index: usize, operation: &WorkspaceOperation, reason: &'static str,
        missing: &mut Option<usize>, detail: &mut Option<String>,
    ) -> Result<(), Error> {
        if missing.is_some() { return Ok(()); }
        if let Some(context)=&self.context {
            context.charge_metadata(size_of::<(&Self,usize,&WorkspaceOperation,&str,
                &mut Option<usize>,&mut Option<String>,std::fmt::Arguments<'_>,
                Option<String>,Result<(),Error>)>())?;
        }
        let arguments=format_args!("{reason}: {:?}; inputs: {:?}; outputs: {:?}",
            operation.kind,operation.inputs,operation.outputs);
        *detail=Some(match &self.context {
            Some(context)=>context.metadata_string(arguments)?,
            None=>arguments.to_string(),
        });
        *missing=Some(index);
        Ok(())
    }
    pub(super) fn reduce_cpu_trace(
        &self,
        report: &WorkspaceTraceReport,
        input_operation: Option<usize>,
        retained_roots: usize,
        output_roots: usize,
        cpu: MlxCpuWorkspaceMechanisms,
    ) -> Result<ReducedTrace, Error> {
        self.reduce_cpu_trace_with_parallel(report, input_operation, retained_roots, output_roots, cpu, None)
    }
    pub(super) fn reduce_cpu_trace_with_parallel(
        &self,
        report: &WorkspaceTraceReport,
        input_operation: Option<usize>,
        retained_roots: usize,
        output_roots: usize,
        cpu: MlxCpuWorkspaceMechanisms,
        parallel: Option<&parallel::OriginalParallelInvocation>,
    ) -> Result<ReducedTrace, Error> {
        let overflow = || self.metadata_error("CPU source population overflow");
        let mut population = CpuPopulation::default();
        let mut grouped_outputs = GroupedOutputStorage::default();
        let mut maximum_rank = 0usize;
        let mut parallel_entries = 0usize;
        let mut parallel_edges = 0usize;
        let mut parallel_births = 0usize;
        let mut child_births = 0usize;
        let mut parallel_graph_extents = 0usize;
        let mut parallel_controls = 0usize;
        let mut streams = 1usize;
        let mut nested_completions = 0usize;
        let mut nested_root_capacity = report.operations.iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::ValueCompletion|WorkspaceOperationKind::AddressableRegion(_)))
            .map(|op| op.inputs.len()).max().unwrap_or(0);
        let mut arrays = 1usize;
        let mut bytes = 0u64;
        let mut seeds=0usize;
        let mut retained_source_seeds=0usize;
        let mut transient_roots=0usize;
        let mut validations=0usize;
        let mut validation_bytes=0u64;
        let mut validation_births=0usize;
        let mut shells = report.tensor_handle_clones;
        let mut missing = None;
        let mut missing_operation_detail = None;
        for (index, operation) in report.operations.iter().enumerate() {
            maximum_rank = maximum_rank.max(
                operation
                    .inputs
                    .iter()
                    .chain(&operation.outputs)
                    .map(|v| v.shape().len())
                    .max()
                    .unwrap_or(0),
            );
            if Some(index) == input_operation {
                arrays = arrays.checked_add(1).ok_or_else(overflow)?;
                continue;
            }
            if self.layerwise_constructors.is_some()
                && matches!(operation.kind, WorkspaceOperationKind::ParameterPlaceholder) {
                continue;
            }
            if matches!(operation.kind,WorkspaceOperationKind::AddressableRegion(_)) {
                let Some(source)=self.addressable_sources.as_ref()else{self.missing_cpu_source(index,operation,"addressable source",&mut missing,&mut missing_operation_detail)?;continue;};
                let quote=source.quote(operation.as_view())?;
                bytes=bytes.checked_add(u64::try_from(quote.capacity.backing).map_err(|_|overflow())?).ok_or_else(overflow)?;
                child_births=child_births.checked_add(quote.numerical.storage.maximum_births()).ok_or_else(overflow)?;
                arrays=arrays.checked_add(operation.inputs.len()).and_then(|n|n.checked_add(operation.outputs.len())).ok_or_else(overflow)?;
                nested_completions=nested_completions.checked_add(1).ok_or_else(overflow)?;
                population.controls=population.controls.checked_add(crate::backend::runtime::cache::value_completion_control_bytes(4).ok_or_else(overflow)?).ok_or_else(overflow)?;
                continue;
            }
            if matches!(operation.kind,WorkspaceOperationKind::ValueCompletion|WorkspaceOperationKind::ValueRetention) {
                if let Some((controls,nested))=super::super::cpu::value_frontier(operation.as_view()) {
                    population.controls=population.controls.checked_add(controls).ok_or_else(overflow)?;
                    arrays=arrays.checked_add(operation.inputs.len()).ok_or_else(overflow)?;
                    if matches!(operation.kind,WorkspaceOperationKind::ValueRetention) {
                        transient_roots=transient_roots.checked_add(operation.inputs.len()).ok_or_else(overflow)?;
                    }
                    nested_completions=nested_completions.checked_add(nested).ok_or_else(overflow)?;
                    continue;
                }
            }
            if matches!(operation.kind,WorkspaceOperationKind::ExpertProviderWave(_)) {
                let Some(wave)=parallel.and_then(|p|p.expert_provider_wave_occurrence(index))else{
                    self.missing_cpu_source(index,operation,"expert provider occurrence",&mut missing,&mut missing_operation_detail)?;continue;
                };
                bytes=bytes.checked_add(wave.provider.backing).ok_or_else(overflow)?;
                child_births=child_births.checked_add(wave.provider.births).ok_or_else(overflow)?;
                continue;
            }
            if matches!(operation.kind,WorkspaceOperationKind::ExpertRegion(_)|WorkspaceOperationKind::ExpertInactiveWave(_)) {
                let Some(region)=parallel.and_then(|p|p.expert_aggregate(index))else{
                    self.missing_cpu_source(index,operation,"expert aggregate occurrence",&mut missing,&mut missing_operation_detail)?;continue;
                };
                bytes=bytes.checked_add(region.child_bytes).ok_or_else(overflow)?;
                child_births=child_births.checked_add(region.child_births).ok_or_else(overflow)?;
                nested_completions=nested_completions.checked_add(region.parent_completions).ok_or_else(overflow)?;
                if region.indexed_parent {
                    // The addressable wrapper completes all four actual
                    // operands together before opening its native child.
                    nested_root_capacity = nested_root_capacity.max(4);
                    arrays=arrays.checked_add(4).ok_or_else(overflow)?;
                    population.controls=population.controls.checked_add(crate::backend::runtime::cache::value_completion_control_bytes(4).ok_or_else(overflow)?).ok_or_else(overflow)?;
                }
                arrays=arrays.checked_add(operation.outputs.len()).ok_or_else(overflow)?;
                for (slice,n) in std::iter::once((&region.empty_slice,region.empty_slices))
                    .chain(region.extra_parents.iter().map(|operation|(operation,1))){
                let Some(plan)=cpu.plan(slice.as_view()).map_err(|cause|self.metadata_source(cause))?else{self.missing_cpu_source(index,slice,"expert parent source",&mut missing,&mut missing_operation_detail)?;continue;};
                let repeated=|value:usize|value.checked_mul(n).ok_or_else(overflow);
                let source=plan.population;
                population.add(CpuPopulation{construction_entries:repeated(source.construction_entries)?,primitives:repeated(source.primitives)?,input_edges:repeated(source.input_edges)?,
                    hidden_leaves:repeated(source.hidden_leaves)?,
                    maximum_operands:source.maximum_operands,maximum_captures:source.maximum_captures,
                    births:repeated(source.births)?,extents:repeated(source.extents)?,controls:repeated(source.controls)?}).ok_or_else(overflow)?;
                seeds=seeds.checked_add(repeated(plan.seeds)?).ok_or_else(overflow)?;
                arrays=arrays.checked_add(repeated(source.primitives.checked_add(slice.inputs.len()).and_then(|v|v.checked_add(plan.seeds)).and_then(|v|v.checked_add(source.hidden_leaves)).ok_or_else(overflow)?)?).ok_or_else(overflow)?;
                shells=shells.and_then(|v|v.checked_add(plan.parameter_shells.checked_mul(n)?));
                bytes=bytes.checked_add(plan.output_bytes.checked_add(plan.scratch_bytes).and_then(|v|v.checked_mul(n as u64)).ok_or_else(overflow)?).ok_or_else(overflow)?;
                maximum_rank=maximum_rank.max(plan.rank).max(2);
                if plan.validations!=0{self.missing_cpu_source(index,slice,"expert parent validation source",&mut missing,&mut missing_operation_detail)?;}
                }
                continue;
            }
            if matches!(operation.kind, WorkspaceOperationKind::Collective(_)) {
                let Some(profile) = self.parallel_population(index, operation, parallel)? else {
                    self.missing_cpu_source(index,operation,"parallel occurrence source",&mut missing,&mut missing_operation_detail)?;
                    continue;
                };
                bytes = bytes.checked_add(profile.bytes).ok_or_else(overflow)?;
                arrays = arrays.checked_add(profile.arrays).ok_or_else(overflow)?;
                parallel_entries = parallel_entries.checked_add(profile.primitives).ok_or_else(overflow)?;
                parallel_edges = parallel_edges.checked_add(profile.edges).ok_or_else(overflow)?;
                parallel_births = parallel_births.checked_add(profile.births).ok_or_else(overflow)?;
                child_births = child_births.checked_add(profile.child_births).ok_or_else(overflow)?;
                parallel_graph_extents = parallel_graph_extents.checked_add(profile.graph_extents).ok_or_else(overflow)?;
                parallel_controls = parallel_controls.checked_add(profile.controls).ok_or_else(overflow)?;
                nested_completions = nested_completions.checked_add(profile.nested).ok_or_else(overflow)?;
                streams = streams.max(profile.streams);
                continue;
            }
            let Some(plan) = cpu
                .plan(operation.as_view())
                .map_err(|cause| self.metadata_source(cause))?
            else {
                self.missing_cpu_source(index,operation,"equation source",&mut missing,&mut missing_operation_detail)?;
                continue;
            };
            maximum_rank = maximum_rank.max(plan.rank);
            if matches!(operation.kind, WorkspaceOperationKind::Grouped { .. }) {
                grouped_outputs = grouped_outputs.merge(super::super::cpu::grouped::output_storage(operation.as_view())
                    .map_err(|cause| self.metadata_source(cause))?).ok_or_else(overflow)?;
            }
            if matches!(operation.kind,WorkspaceOperationKind::HostStoreFloating(..)) {
                transient_roots=transient_roots.checked_add(1).ok_or_else(overflow)?;
                nested_completions=nested_completions.checked_add(1).ok_or_else(overflow)?;
            } else if super::super::cpu::host_transfer::is_load(operation.as_view().kind) {
                retained_source_seeds=retained_source_seeds.checked_add(1).ok_or_else(overflow)?;
                nested_completions=nested_completions.checked_add(1).ok_or_else(overflow)?;
            }
            nested_completions=nested_completions.checked_add(
                super::super::cpu::blockwise::nested_completions(operation.as_view())).ok_or_else(overflow)?;
            population.add(plan.population).ok_or_else(overflow)?;
            seeds=seeds.checked_add(plan.seeds).ok_or_else(overflow)?;
            validations=validations.checked_add(plan.validations).ok_or_else(overflow)?;
            if plan.validations!=0 {
                validation_bytes=validation_bytes.checked_add(plan.output_bytes)
                    .and_then(|n|n.checked_add(plan.scratch_bytes)).ok_or_else(overflow)?;
                validation_births=validation_births.checked_add(plan.population.births)
                    .and_then(|n|n.checked_add(plan.seeds)).ok_or_else(overflow)?;
            }
            bytes = bytes
                .checked_add(plan.output_bytes)
                .and_then(|n| n.checked_add(plan.scratch_bytes))
                .ok_or_else(overflow)?;
            arrays = arrays
                .checked_add(plan.population.primitives)
                .and_then(|n| n.checked_add(operation.inputs.len()))
                .and_then(|n| n.checked_add(plan.seeds))
                .and_then(|n| n.checked_add(plan.population.hidden_leaves))
                .ok_or_else(overflow)?;
            shells = shells.and_then(|n| n.checked_add(plan.parameter_shells));
        }
        let roots = retained_roots
            .checked_add(output_roots)
            .and_then(|n|n.checked_add(validations))
            .and_then(|n|n.checked_add(transient_roots))
            .ok_or_else(overflow)?;
        let construction_entries = population.construction_entries.checked_add(parallel_entries).ok_or_else(overflow)?;
        let primitives = population.primitives.checked_add(parallel_entries).ok_or_else(overflow)?;
        let tape = primitives.checked_add(1).ok_or_else(overflow)?;
        arrays = arrays.checked_add(roots).ok_or_else(overflow)?;
        let edges = population
            .input_edges
            .checked_add(parallel_edges)
            .and_then(|n| n.checked_add(roots))
            .ok_or_else(overflow)?;
        let completion = if missing.is_none() {
            safemlx::OperationEvent::cpu_completion_layout(roots)
        } else {
            None
        };
        let base = if missing.is_none() {
            safemlx::OperationEvent::eval_record_layout(tape, streams, tape)
        } else {
            None
        };
        let traversal = base.and_then(|base| {
            safemlx::OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
                roots,
                arrays,
                tape_entries: tape,
                input_edges: edges,
                output_slots: tape,
                streams,
                captures: base.capture_slots().max(population.maximum_captures).max(1),
            })
        });
        let graph = if missing.is_none() {
            shells.and_then(|shells| {
                safemlx::OperationEvent::resident_graph_layout_with_shells(
                    construction_entries,
                    seeds.checked_add(retained_source_seeds)?,
                    maximum_rank,
                    // Shared resident constructors include four-entry temporary
                    // operand vectors even when the final CPU primitive is binary.
                    // Eval input edges above still count the actual graph only.
                    population.maximum_operands.max(4),
                    shells,
                )
            })
        } else {
            None
        };
        let query_controls = (|| {
            population
                .controls
                .checked_add(parallel_controls)?
                .checked_add(base?.query_control_bytes()?)?
                .checked_add(traversal?.query_control_bytes()?)?
                .checked_add(graph?.control_bytes()?)?
                .checked_add(completion?.control_bytes()?)
        })();
        let dispatch = (|| {
            let completion = completion?;
            if completion.backing_births() != 0 || completion.worker_graph_allocation_extents() != 0
            {
                return None;
            }
            graph?;
            traversal?;
            query_controls?;
            Some(ResidentDispatchPopulation {
                cpu_model: Some(population),
                gpu_entries: 0,
                gpu_input_edges: 0,
                gpu_siblings: 0,
                gpu_births: 0,
                additional_sort_kernels: 0,
                cpu_entries: tape,
                cpu_input_edges: edges,
                cpu_siblings: tape,
                parallel_entries,
                parallel_graph_extents,
                worker_graph_extents: 0,
                worker_rank: maximum_rank,
                copy_rank_extents: 0,
                kernel_attempts: 0,
            })
        })();
        Ok(ReducedTrace {
            carryover_complete: true,
            validation_producers: missing.is_none().then_some(CertifiedSpanStorage {
                mutable_bytes: validation_bytes,
                maximum_births: validation_births,
            }),
            traversal,
            mutable_storage: missing.is_none().then_some(CertifiedSpanStorage {
                mutable_bytes: bytes,
                maximum_births: population.births.checked_add(seeds)
                    .and_then(|n| n.checked_add(parallel_births))
                    .and_then(|n| n.checked_add(child_births)).ok_or_else(overflow)?,
            }),
            first_missing_operation: missing,
            missing_operation_detail,
            unqualified_kernel_owner: None,
            maximum_rank,
            host_primitive_nodes: primitives,
            validation_roots: validations,
            grouped_outputs,
            query_controls,
            graph,
            dispatch,
            nested_completions,
            nested_root_capacity: nested_root_capacity.max(usize::from(nested_completions!=0)),
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
    use eredu_checkpoint::LinearFormat;
    use eredu_nn::{
        CpuMatmulImplementation, LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend,
        ParameterMetadata, ParameterSpec, ParameterVisitorMut, Parameterized, Tensor,
    };
    fn value(shape: &[i32], context: &WorkspaceContext, represented: bool) -> WorkspaceTensor {
        let layout = context.layout(shape, WorkspaceDtype::Float32).unwrap();
        let layout = if represented {
            layout.with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            )))
        } else {
            layout
        };
        WorkspaceTensor::existing(layout, context).unwrap()
    }
    #[test]
    fn cpu_prepared_token_equation_prices_input_and_host_without_admitting_generic_initialization() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(matmul)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),matmul);
        for dtype in [WorkspaceDtype::Int32,WorkspaceDtype::Uint32] {
            for count in [1,2,5] {
                let context=WorkspaceContext::new(cpu);context.begin_state_span([]).unwrap();
                let input=WorkspaceTensor::prepared_token_input(&[2,count],dtype,&context).unwrap();
                let report=context.report(&[input]).unwrap();
                assert!(report.unpriced_operations.is_empty()&&report.unpriced_host_operations.is_empty());
                assert!(report.inference_transient_bytes().is_some());
                assert_eq!(report.host_workspace_bytes,Some(0));
                // This cold declaration still has no independent native CPU
                // operation plan; only the actual driver ordinal can omit it.
                assert!(cpu.plan(report.operations[0].as_view()).unwrap().is_none());
                let geometry=InferenceGeometry{batch_size:2,cached_positions:0,input_positions:count as u64,
                    max_output_tokens:1,prefill_chunk_positions:count as u64,output:eredu_core::OutputDemand::LastPosition};
                let recorder=ResidentRecipeRecorder::with_cpu_context(geometry,ordinary,cpu,&context).unwrap();
                assert!(recorder.reduce_cpu_trace(&report,Some(0),0,1,cpu).unwrap().first_missing_operation.is_none());
                assert_eq!(recorder.reduce_cpu_trace(&report,None,0,1,cpu).unwrap().first_missing_operation,Some(0));
                let generic=WorkspaceContext::new(cpu);generic.begin_state_span([]).unwrap();
                let input=WorkspaceTensor::initialized(&[2,count],dtype,&generic).unwrap();
                let report=generic.report(&[input]).unwrap();
                assert_eq!(report.unpriced_operations,[0]);assert_eq!(report.unpriced_host_operations,[0]);
                let metal=WorkspaceContext::new(ordinary);metal.begin_state_span([]).unwrap();
                let input=WorkspaceTensor::prepared_token_input(&[2,count],dtype,&metal).unwrap();
                let report=metal.report(&[input]).unwrap();assert!(report.inference_transient_bytes().is_some());
            }
        }
    }
    #[test]
    fn cpu_original_prompt_quote_keeps_full_backing_and_authenticated_input_source() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let matmul=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),matmul);
        let ids=[7;11];let input=eredu_runtime::working_memory::OriginalTokenInputLayout::prepare(
            &eredu_core::TokenIdsInputPlan::new(&ids).unwrap()).unwrap();
        let mut peak=None;
        for chunk in [1,4,11] {
            let geometry=InferenceGeometry{batch_size:1,cached_positions:5,input_positions:11,
                max_output_tokens:9,prefill_chunk_positions:chunk,output:eredu_core::OutputDemand::LastPosition};
            let context=WorkspaceContext::new(cpu);
            let quoted=eredu_runtime::working_memory::quote_original_token_prompt_workspace(geometry,&input,&context).unwrap();
            assert!(quoted.peak().bytes().is_some());
            assert_eq!(quoted.tensor_peak_bytes(),Some(ordinary.allocation().fixed_buffer_capacity(44).unwrap()));
            assert!(quoted.host_peak_bytes().unwrap()>0,"closed text identity remains priced");
            if let Some(prior)=peak {assert_eq!(quoted.peak().bytes(),Some(prior));}
            peak=quoted.peak().bytes();
            let report=context.report(&[]).unwrap();assert_eq!(report.operations.len(),1);
            assert!(super::super::super::basic::is_prepared_token_input(report.operations[0].as_view()));
            assert!(report.unpriced_operations.is_empty()&&report.unpriced_host_operations.is_empty());
            let unsourced=eredu_runtime::working_memory::quote_text_prompt_workspace(geometry,Some(44),&WorkspaceContext::new(cpu)).unwrap();
            assert!(unsourced.peak().bytes().is_none(),"generic native initialization still needs its actual producer");
        }
    }
    struct Bind<'a> {
        weight: &'a WorkspaceTensor,
        bias: &'a WorkspaceTensor,
    }
    impl<'a> ParameterVisitorMut<'a, WorkspaceTensor> for Bind<'_> {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut WorkspaceTensor) {
            *value = if metadata.id().as_str() == "cpu.bias" {
                self.bias
            } else {
                self.weight
            }
            .clone();
        }
    }
    #[test]
    fn selected_cpu_dense_quote_uses_exact_alias_matmul_bias_sources_and_no_gpu_population() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(matmul) = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles)
        else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());
            return;
        };
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), matmul);
        for dtype in [
            WorkspaceFloatingType::Float32,
            WorkspaceFloatingType::Bfloat16,
        ] {
            for (shape, biased) in [
                (&[2, 3][..], false),
                (&[2, 3][..], true),
                (&[2, 2, 3][..], false),
                (&[2, 2, 3][..], true),
            ] {
                let context = WorkspaceContext::new(cpu);
                let represented = |shape: &[i32]| {
                    WorkspaceTensor::existing(
                        context
                            .layout(shape, WorkspaceDtype::Float32)
                            .unwrap()
                            .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
                        &context,
                    )
                    .unwrap()
                };
                let input = represented(shape);
                let weight = represented(&[5, 3]);
                let bias = represented(&[5]);
                let mut projection = WorkspaceBackend::linear(
                    LinearSpec {
                        input: 3,
                        output: 5,
                        weight: ParameterSpec::trainable("cpu.weight").unwrap(),
                        bias: biased.then(|| ParameterSpec::trainable("cpu.bias").unwrap()),
                        format: LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
                    },
                    &context,
                )
                .unwrap();
                projection.visit_parameters_mut(&mut Bind {
                    weight: &weight,
                    bias: &bias,
                });
                context.begin_span();
                let output = projection.forward(&input, &context).unwrap();
                let report = context.report(&[output]).unwrap();
                let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.dtype, dtype);
                assert_eq!(
                    report.operations[0].outputs[0]
                        .representation()
                        .unwrap()
                        .dtype(),
                    dtype
                );
                assert_eq!(
                    plan.population.primitives,
                    2 + usize::from(shape.len() > 2) * 2 + usize::from(biased) * 2
                );
                assert_eq!(plan.population.births, 1 + usize::from(biased));
                let mut recorder = ResidentRecipeRecorder::new(
                    InferenceGeometry {
                        batch_size: 1,
                        cached_positions: 0,
                        input_positions: 2,
                        max_output_tokens: 0,
                        prefill_chunk_positions: 2,
                        output: eredu_core::OutputDemand::Sequence,
                    },
                    ordinary,
                );
                recorder.cpu = Some(cpu);
                let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
                assert!(reduced.first_missing_operation.is_none());
                assert_eq!(reduced.graph.as_ref().expect("qualified shared constructor").maximum_operands(), 4);
                assert!(reduced.traversal.is_some(), "qualified actual CPU traversal");
                let dispatch = reduced.dispatch.unwrap();
                assert_eq!(dispatch.gpu_entries, 0);
                assert_eq!(dispatch.gpu_births, 0);
                assert_eq!(dispatch.kernel_attempts, 0);
                assert_eq!(dispatch.cpu_entries, plan.population.primitives + 1);
                assert_eq!(dispatch.cpu_model.unwrap().extents, plan.population.extents);
                assert_eq!(
                    reduced.mutable_storage.unwrap().maximum_births(),
                    plan.population.births
                );
                assert_eq!(
                    reduced.mutable_storage.unwrap().mutable_bytes(),
                    plan.output_bytes + plan.scratch_bytes
                );
                let completion = ResidentCompletionRecipe {
                    validation_roots: 0,
                    grouped_outputs: GroupedOutputStorage::default(),
                    traversal: reduced.traversal.unwrap(),
                    graph: reduced.graph.unwrap(),
                    dispatch: Some(dispatch),
                    nested_completions: 0,
                    nested_root_capacity: 0,
                };
                let full =
                    graph_capacity::ResidentGraphStorage::for_completion(completion).unwrap();
                assert!(full.full_capacity.unwrap() > plan.population.extents as u64);
                let mut bad = report;
                bad.operations[0].kind =
                    WorkspaceOperationKind::Elementwise("unqualified_cpu_operation");
                let refused = recorder.reduce_trace(&bad, None, 0, 1).unwrap();
                assert_eq!(refused.first_missing_operation, Some(0));
                assert!(refused.dispatch.is_none());
                assert!(refused.mutable_storage.is_none());
            }
        }
    }
    #[test]
    fn cpu_mixed_dense_quotes_each_conversion_at_its_own_extent(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        use WorkspaceFloatingType::{Float32 as F,Bfloat16 as B,Float16 as H};
        // The table cast is much larger than the output. The last pair checks
        // post-Matmul bias promotion separately from the selected BF16 product.
        for (left_dtype,weight_dtype,bias_dtype,casts) in [(F,B,H,2usize),(H,F,B,2),(B,H,F,2),(B,B,F,1)] {
            let context=WorkspaceContext::new(cpu);
            let represented=|shape:&[i32],dtype|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
            let input=represented(&[1,2,257],left_dtype);let weight=represented(&[19,257],weight_dtype);
            let bias=represented(&[19],bias_dtype);
            let mut projection=WorkspaceBackend::linear(LinearSpec{input:257,output:19,
                weight:ParameterSpec::trainable("cpu.weight").unwrap(),bias:Some(ParameterSpec::trainable("cpu.bias").unwrap()),
                format:LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()},&context).unwrap();
            projection.visit_parameters_mut(&mut Bind{weight:&weight,bias:&bias});
            context.begin_span();let output=projection.forward(&input,&context).unwrap();
            assert_eq!(output.layout().representation(),Some(WorkspaceRepresentation::new(F,true)));
            let report=context.report(&[output]).unwrap();let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.population.primitives,6+casts);assert_eq!(plan.population.births,2+casts);
            let matrix_dtype=if left_dtype==weight_dtype {left_dtype}else{F};
            let capacity=|elements:u64|ordinary.allocation().fixed_buffer_capacity(elements*4).unwrap();
            let expected=capacity(38)+if left_dtype!=matrix_dtype {capacity(514)}else{0}
                +if weight_dtype!=matrix_dtype {capacity(4883)}else{0}
                +if matrix_dtype!=F {capacity(38)}else{0}
                +if bias_dtype!=F {capacity(19)}else{0};
            assert_eq!(plan.scratch_bytes,expected);
            let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{batch_size:1,cached_positions:0,input_positions:2,
                max_output_tokens:0,prefill_chunk_positions:2,output:eredu_core::OutputDemand::Sequence},ordinary,cpu,&context).unwrap();
            assert_eq!(recorder.reduce_trace(&report,None,0,1).unwrap().first_missing_operation,None);
            let mut invalid=report;
            invalid.operations[0].inputs[1]=invalid.operations[0].inputs[1].clone().with_representation(None);
            assert!(cpu.plan(invalid.operations[0].as_view()).unwrap().is_none());
            for input in &mut invalid.operations[0].inputs[..2] {
                *input=input.clone().with_representation(Some(WorkspaceRepresentation::new(H,true)));
            }
            let supported = selected.selected().float16_geometry(2,2,19,257,1).is_ok();
            assert_eq!(cpu.plan(invalid.operations[0].as_view()).unwrap().is_some(),supported,
                "F32 bias cannot supply missing F16 Matmul facts; the actual platform SIMD source can");
        }
    }

    #[test]
    fn cpu_selected_f16_dense_joins_matrix_and_post_bias_conversion_sources(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32AndFloat16Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        use WorkspaceFloatingType::{Float32 as F,Float16 as H};
        for bias_dtype in [H,F] {
            let context=WorkspaceContext::new(cpu);
            let represented=|shape:&[i32],dtype|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
            let input=represented(&[2,3,23],H);let weight=represented(&[17,23],H);
            let bias=represented(&[17],bias_dtype);
            let mut projection=WorkspaceBackend::linear(LinearSpec{input:23,output:17,
                weight:ParameterSpec::trainable("cpu.weight").unwrap(),bias:Some(ParameterSpec::trainable("cpu.bias").unwrap()),
                format:LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()},&context).unwrap();
            projection.visit_parameters_mut(&mut Bind{weight:&weight,bias:&bias});
            context.begin_span();let output=projection.forward(&input,&context).unwrap();
            assert_eq!(output.layout().representation(),Some(WorkspaceRepresentation::new(bias_dtype,true)));
            let report=context.report(&[output]).unwrap();let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            let cast=usize::from(bias_dtype==F);
            assert_eq!(plan.population.primitives,6+cast);assert_eq!(plan.population.births,2+cast);
            let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{batch_size:2,cached_positions:0,input_positions:3,
                max_output_tokens:0,prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence},ordinary,cpu,&context).unwrap();
            assert_eq!(recorder.reduce_trace(&report,None,0,1).unwrap().first_missing_operation,None);
            let mut missing=report;
            missing.operations[0].inputs[1]=missing.operations[0].inputs[1].clone().with_representation(None);
            assert!(cpu.plan(missing.operations[0].as_view()).unwrap().is_none());
        }
    }

    #[test]
    fn cpu_mixed_rms_preserves_module_promotion_and_tensor_cast_back(){
        use eredu_nn::{NormalizationConstructionSpec,NormalizationScale,NormalizationOperator};
        use WorkspaceFloatingType::{Float32 as F,Bfloat16 as B,Float16 as H};
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for (input_dtype,gain_dtype) in [(F,B),(F,H),(B,F),(H,F),(B,H)] {
            let context=WorkspaceContext::new(cpu);
            let represented=|shape:&[i32],dtype|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
            let input=represented(&[2,3,17],input_dtype);let gain=represented(&[17],gain_dtype);
            let mut module=WorkspaceBackend::normalization(NormalizationConstructionSpec{groups:None,dimensions:17,epsilon:1e-6,
                scale:NormalizationScale::Learned(ParameterSpec::trainable("cpu.weight").unwrap())},&context).unwrap();
            module.visit_parameters_mut(&mut Bind{weight:&gain,bias:&gain});
            context.begin_span();let module_output=module.forward(&input,&context).unwrap();
            let tensor_output=WorkspaceBackend::rms_norm_with_weight(&input,&gain,1e-6,&context).unwrap();
            assert_eq!(module_output.layout().representation(),Some(WorkspaceRepresentation::new(F,true)));
            assert_eq!(tensor_output.layout().representation(),Some(WorkspaceRepresentation::new(input_dtype,true)));
            let report=context.report(&[module_output,tensor_output]).unwrap();
            let module_plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            let tensor_plan=cpu.plan(report.operations[1].as_view()).unwrap().unwrap();
            assert_eq!(module_plan.population.primitives,26,"mixed input/gain does not create the equal-half GPU probe cast");
            assert_eq!(tensor_plan.population.primitives,module_plan.population.primitives+1);
            assert_eq!(tensor_plan.population.births,module_plan.population.births+1);
            assert!(module_plan.scratch_bytes>module_plan.output_bytes);
            let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{batch_size:2,cached_positions:0,input_positions:3,
                max_output_tokens:0,prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence},ordinary,cpu,&context).unwrap();
            assert_eq!(recorder.reduce_trace(&report,None,0,1).unwrap().first_missing_operation,None);
            let mut missing=report;missing.operations[0].inputs[1]=missing.operations[0].inputs[1].clone().with_representation(None);
            assert!(cpu.plan(missing.operations[0].as_view()).unwrap().is_none());
        }
    }

    #[test]
    fn cpu_dense_source_rejects_unspecified_layout_default_blas_and_distinct_addmm() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected) = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles)
        else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());
            return;
        };
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        let context = WorkspaceContext::new(ordinary);
        let known = value(&[2, 3], &context, true);
        let unknown = value(&[2, 3], &context, false);
        let weight = value(&[5, 3], &context, true);
        let bias = value(&[5], &context, true);
        context.begin_span();
        let output = WorkspaceTensor::linear(&known, &weight, Some(&bias), &context).unwrap();
        let report = context.report(&[output]).unwrap();
        assert!(cpu.plan(report.operations[0].as_view()).unwrap().is_none());
        context.begin_span();
        let output = WorkspaceTensor::linear(&unknown, &weight, None, &context).unwrap();
        let report = context.report(&[output]).unwrap();
        assert!(cpu.plan(report.operations[0].as_view()).unwrap().is_none());
        context.begin_span();
        let output = WorkspaceTensor::linear(&known, &weight, None, &context).unwrap();
        let report = context.report(&[output]).unwrap();
        assert!(cpu.plan(report.operations[0].as_view()).unwrap().is_some());
        let default = MlxCpuWorkspaceMechanisms::new(
            ordinary.allocation(),
            MlxCpuMatmulMechanism::select(CpuMatmulImplementation::PlatformDefault).unwrap(),
        );
        assert!(default
            .plan(report.operations[0].as_view())
            .unwrap()
            .is_none());
    }
    #[test]
    fn cpu_pointwise_residual_trace_composes_exact_sources_and_stored_half_scalar() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected) = MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles)
        else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());
            return;
        };
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        for dtype in [
            WorkspaceFloatingType::Float32,
            WorkspaceFloatingType::Bfloat16,
        ] {
            let context = WorkspaceContext::new(cpu);
            let represented = |shape: &[i32], dtype| {
                WorkspaceTensor::existing(
                    context
                        .layout(shape, WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
                    &context,
                )
                .unwrap()
            };
            let input = represented(&[2, 3], dtype);
            let gain = represented(&[3], dtype);
            context.begin_span();
            let product = input.multiply(&gain, &context).unwrap();
            let square = product.square(&context).unwrap();
            let bounded = square.tanh(&context).unwrap();
            let output = bounded.add(&input, &context).unwrap();
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 4);
            let mut recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: 2,
                    max_output_tokens: 0,
                    prefill_chunk_positions: 2,
                    output: eredu_core::OutputDemand::Sequence,
                },
                ordinary,
            );
            recorder.cpu = Some(cpu);
            let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
            assert_eq!(reduced.first_missing_operation, None);
            assert_eq!(reduced.graph.as_ref().expect("qualified shared constructor").maximum_operands(), 4);
            assert!(reduced.traversal.is_some(), "qualified actual CPU traversal");
            let dispatch = reduced.dispatch.unwrap();
            let source = dispatch.cpu_model.unwrap();
            assert_eq!(source.primitives, 5);
            assert_eq!(source.births, 4);
            assert_eq!(dispatch.gpu_entries, 0);
            assert_eq!(dispatch.kernel_attempts, 0);
            let first = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(first.population.primitives, 2);
            let half_scalar=represented(&[1],WorkspaceFloatingType::Float16);
            context.begin_span();let scaled=input.multiply(&half_scalar,&context).unwrap();
            assert_eq!(scaled.layout().representation(),Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
            let mut mixed=context.report(&[scaled]).unwrap();
            let plan=cpu.plan(mixed.operations[0].as_view()).unwrap().unwrap();
            let casts=1+usize::from(dtype!=WorkspaceFloatingType::Float32);
            assert_eq!(plan.population.primitives,2+casts);assert_eq!(plan.population.births,1+casts);
            let capacity=|elements:u64|ordinary.allocation().fixed_buffer_capacity(elements*4).unwrap();
            assert_eq!(plan.scratch_bytes,capacity(1)+if casts==2 {capacity(6)}else{0});
            assert_eq!(recorder.reduce_trace(&mixed,None,0,1).unwrap().first_missing_operation,None);
            mixed.operations[0].inputs[1]=mixed.operations[0].inputs[1].clone().with_representation(None);
            let refused = recorder.reduce_trace(&mixed, None, 0, 1).unwrap();
            assert_eq!(refused.first_missing_operation, Some(0));
            assert!(refused.dispatch.is_none());
        }
    }
    #[test]
    fn cpu_strict_lookup_quote_keeps_validation_and_five_scalar_sources() {
        strict_lookup_sources(&[WorkspaceFloatingType::Float32,WorkspaceFloatingType::Bfloat16]);
    }
    #[test]
    fn cpu_f16_lookup_preserves_validation_source_and_half_output(){
        strict_lookup_sources(&[WorkspaceFloatingType::Float16]);
    }
    fn strict_lookup_sources(dtypes:&[WorkspaceFloatingType]) {
        lookup_sources(dtypes, eredu_nn::EmbeddingLookupPolicy::Strict);
    }
    #[test]
    fn cpu_sentinel_lookup_keeps_domain_completion_and_physical_output() {
        for sentinel in [-1, -7] {
            lookup_sources(&[WorkspaceFloatingType::Float32,WorkspaceFloatingType::Float16,
                WorkspaceFloatingType::Bfloat16],eredu_nn::EmbeddingLookupPolicy::ZeroSentinel(sentinel));
        }
    }
    fn lookup_sources(dtypes:&[WorkspaceFloatingType],policy:eredu_nn::EmbeddingLookupPolicy) {
        use eredu_nn::{EmbeddingOperator,EmbeddingSpec,EmbeddingLookupPolicy};
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for &dtype in dtypes {
            for ids_shape in [&[][..],&[2,2][..]] {
                let context=WorkspaceContext::new(cpu);
                let weight=WorkspaceTensor::existing(context.layout(&[37,13],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype,false))),&context).unwrap();
                let ids=WorkspaceTensor::existing(context.layout(ids_shape,WorkspaceDtype::Uint32).unwrap(),&context).unwrap();
                let mut lookup=WorkspaceBackend::embedding(EmbeddingSpec {vocabulary:37,dimensions:13,
                    weight:ParameterSpec::trainable("cpu.weight").unwrap(),
                    format:LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()},&context).unwrap();
                lookup.visit_parameters_mut(&mut Bind {weight:&weight,bias:&weight});
                context.begin_span();let output=lookup.lookup(&ids,policy,&context).unwrap();
                let report=context.report(&[output]).unwrap();
                let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.dtype,dtype);assert_eq!(plan.validations,1);
                let sentinel=matches!(policy,EmbeddingLookupPolicy::ZeroSentinel(_));
                let seeds=if sentinel {8}else{5};
                assert_eq!(plan.seeds,seeds);
                assert_eq!(plan.population.primitives,(if ids_shape.is_empty(){48}else{50})
                    + if sentinel {27}else{0});
                let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry {batch_size:1,cached_positions:0,
                    input_positions:2,max_output_tokens:0,prefill_chunk_positions:2,
                    output:eredu_core::OutputDemand::Sequence},ordinary);recorder.cpu=Some(cpu);
                let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
                assert_eq!(reduced.first_missing_operation,None);assert_eq!(reduced.validation_roots,1);
                assert_eq!(reduced.traversal.unwrap().roots(),2);assert_eq!(reduced.graph.unwrap().seeds(),seeds);
                assert_eq!(reduced.mutable_storage.unwrap().maximum_births(),plan.population.births+seeds);
                assert_eq!(reduced.validation_producers.unwrap().mutable_bytes(),plan.output_bytes+plan.scratch_bytes);
                assert_eq!(reduced.validation_producers.unwrap().maximum_births(),plan.population.births+seeds);
                let dispatch=reduced.dispatch.unwrap();assert_eq!(dispatch.gpu_entries,0);
                let completion=ResidentCompletionRecipe {validation_roots:1,grouped_outputs:GroupedOutputStorage::default(),
                    traversal:reduced.traversal.unwrap(),graph:reduced.graph.unwrap(),dispatch:Some(dispatch),
                    nested_completions:0,nested_root_capacity:0};
                assert!(graph_capacity::ResidentGraphStorage::for_completion(completion).unwrap().full_capacity.is_some());
                let mut wrong=report;
                let WorkspaceOperationKind::Embedding(_,policy)=&mut wrong.operations[0].kind else {panic!("lookup lost identity")};
                *policy=EmbeddingLookupPolicy::ZeroSentinel(0);
                assert!(recorder.reduce_trace(&wrong,None,0,1).is_err(),
                    "nonnegative sentinel must retain the ordinary policy rejection");
            }
        }
    }

    #[test]
    fn cpu_rms_quote_preserves_row_reduction_scalars_and_cast_before_gain() {
        rms_quote_sources(&[WorkspaceFloatingType::Float32,WorkspaceFloatingType::Bfloat16]);
    }
    #[test]
    fn cpu_f16_rms_quote_preserves_wide_reduction_and_half_gain_boundary() {
        rms_quote_sources(&[WorkspaceFloatingType::Float16]);
    }
    fn rms_quote_sources(dtypes:&[WorkspaceFloatingType]) {
        use eredu_nn::{NormalizationConstructionSpec,NormalizationScale,NormalizationOperator};
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for &dtype in dtypes {
            for shape in [&[17][..],&[2,3,17][..]] {
                let context=WorkspaceContext::new(cpu);
                let represented=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
                let input=represented(shape);let weight=represented(&[17]);
                let mut normalization=WorkspaceBackend::normalization(NormalizationConstructionSpec {groups:None,
                    dimensions:17,epsilon:1e-6,scale:NormalizationScale::Learned(ParameterSpec::trainable("cpu.weight").unwrap())},&context).unwrap();
                normalization.visit_parameters_mut(&mut Bind{weight:&weight,bias:&weight});
                context.begin_span();let output=normalization.forward(&input,&context).unwrap();
                let report=context.report(&[output]).unwrap();
                let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.dtype,dtype);assert_eq!(plan.seeds,2);assert_eq!(plan.validations,0);
                assert!(plan.scratch_bytes>plan.output_bytes);
                assert_eq!(plan.population.primitives,if dtype==WorkspaceFloatingType::Float32 {26}else{27});
                let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry {batch_size:1,cached_positions:0,
                    input_positions:2,max_output_tokens:0,prefill_chunk_positions:2,output:eredu_core::OutputDemand::Sequence},ordinary);
                recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
                assert_eq!(reduced.first_missing_operation,None);assert_eq!(reduced.graph.unwrap().seeds(),2);
                assert!(reduced.dispatch.unwrap().cpu_model.is_some());
                context.begin_span();let output=WorkspaceBackend::rms_norm_without_weight(&input,1e-6,&context).unwrap();
                let unit=context.report(&[output]).unwrap();
                assert!(cpu.plan(unit.operations[0].as_view()).unwrap().is_some());
                let mut refused=report;
                let WorkspaceOperationKind::ConstructedNormalization(spec)=&mut refused.operations[0].kind else{panic!("normalization identity")};
                spec.scale=NormalizationScale::LearnedOffset{weight:ParameterSpec::trainable("cpu.weight").unwrap(),offset:1.0};
                assert!(cpu.plan(refused.operations[0].as_view()).unwrap().is_none());
            }
        }
    }

    #[test]
    fn cpu_half_unit_rms_quotes_half_mean_then_actual_f32_epsilon_promotion() {
        use eredu_nn::NeuralBackend;
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for dtype in [WorkspaceFloatingType::Float16,WorkspaceFloatingType::Bfloat16] {
            for width in [1,17,1031] {
                let context=WorkspaceContext::new(cpu);
                let input=WorkspaceTensor::existing(context.layout(&[2,3,width],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
                context.begin_span();
                let output=WorkspaceBackend::rms_norm_without_weight(&input,1e-6,&context).unwrap();
                assert_eq!(output.layout().representation(),Some(WorkspaceRepresentation::new(dtype,true)));
                let report=context.report(&[output]).unwrap();
                let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.dtype,dtype);assert_eq!(plan.seeds,2);assert_eq!(plan.validations,0);
                assert_eq!(plan.population.primitives,19);
                assert!(plan.scratch_bytes>plan.output_bytes);
                SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
            }
        }
    }

    #[test]
    fn cpu_activations_preserve_wide_intermediates_and_compose_with_residuals() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Bfloat16] {
            for silu in [true,false] {
                let context=WorkspaceContext::new(cpu);
                let input=WorkspaceTensor::existing(context.layout(&[2,17],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
                context.begin_span();
                let activation=if silu {WorkspaceBackend::silu(input.clone(),&context).unwrap()}
                    else{WorkspaceBackend::sigmoid(input.clone(),&context).unwrap()};
                let output=activation.multiply(&input,&context).unwrap();
                let report=context.report(&[output]).unwrap();
                let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.dtype,dtype);assert_eq!(plan.seeds,usize::from(silu));
                assert_eq!(plan.population.primitives,(if silu {13}else{2})+usize::from(dtype==WorkspaceFloatingType::Bfloat16));
                let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:0,input_positions:2,
                    max_output_tokens:0,prefill_chunk_positions:2,output:eredu_core::OutputDemand::Sequence},ordinary);
                recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
                assert_eq!(reduced.first_missing_operation,None);assert_eq!(reduced.graph.unwrap().seeds(),usize::from(silu));
                assert!(reduced.dispatch.unwrap().cpu_model.is_some());
                let mut unknown=report;unknown.operations[0].inputs[0]=unknown.operations[0].inputs[0].clone()
                    .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float16,true)));
                assert!(cpu.plan(unknown.operations[0].as_view()).unwrap().is_none());
            }
        }
        let context=WorkspaceContext::new(cpu);let input=value(&[2,3],&context,true);
        context.begin_span();let exponential=WorkspaceBackend::exp(input,&context).unwrap();
        let report=context.report(&[exponential]).unwrap();
        assert!(report.operations.iter().all(|operation|cpu.plan(operation.as_view()).unwrap().is_some()));
    }

    #[test]
    fn cpu_target_embedding_scale_keeps_exact_f32_output_and_seed_custody(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else{
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        let context=WorkspaceContext::new(cpu);let input=value(&[1,3,32],&context,true);
        context.begin_span();let scaled=input.multiply_scalar(32.0f32.sqrt(),&context).unwrap();
        assert_eq!(scaled.layout().representation(),Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
        let output=scaled.add(&input,&context).unwrap();let report=context.report(&[output]).unwrap();
        let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!(plan.seeds,1);assert_eq!(plan.population.primitives,5);assert_eq!(plan.population.input_edges,6);
        assert_eq!(plan.population.births,3);assert!(plan.scratch_bytes>plan.output_bytes);
        let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{batch_size:1,cached_positions:0,input_positions:3,
            max_output_tokens:0,prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence},ordinary,cpu,&context).unwrap();
        let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();assert_eq!(reduced.first_missing_operation,None);
        assert!(reduced.dispatch.unwrap().cpu_model.is_some());assert_eq!(reduced.graph.unwrap().seeds(),1);
        let mut unknown=report;
        unknown.operations[0].inputs[0]=unknown.operations[0].inputs[0].clone().with_representation(Some(
            WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false)));
        assert!(cpu.plan(unknown.operations[0].as_view()).unwrap().is_none());
        unknown.operations[0].inputs[0]=unknown.operations[0].inputs[0].clone().with_representation(None);
        assert!(cpu.plan(unknown.operations[0].as_view()).unwrap().is_none());
    }

    #[test]
    fn cpu_half_scalar_and_gelu_sources_preserve_actual_f32_promotion(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else{
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for dtype in [WorkspaceFloatingType::Bfloat16,WorkspaceFloatingType::Float16] {
            let context=WorkspaceContext::new(cpu);
            let input=WorkspaceTensor::existing(context.layout(&[2,3,17],WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
            context.begin_span();
            let scaled=input.multiply_scalar(32.0f32.sqrt(),&context).unwrap();
            let activated=WorkspaceTensor::gelu(&input,&context).unwrap();
            let output=scaled.add(&activated,&context).unwrap();
            for value in [&scaled,&activated,&output] {
                assert_eq!(value.layout().representation(),Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
            }
            assert_eq!(input.layout().representation(),Some(WorkspaceRepresentation::new(dtype,true)));
            let report=context.report(&[output]).unwrap();
            let scale=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!((scale.population.primitives,scale.population.births,scale.seeds),(5,3,1));
            let gelu=cpu.plan(report.operations[1].as_view()).unwrap().unwrap();
            assert_eq!((gelu.population.primitives,gelu.seeds),(22,3));
            assert!(scale.scratch_bytes>scale.output_bytes);assert!(gelu.scratch_bytes>gelu.output_bytes);
            let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{batch_size:2,cached_positions:0,input_positions:3,
                max_output_tokens:0,prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence},ordinary,cpu,&context).unwrap();
            let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
            assert_eq!(reduced.graph.unwrap().seeds(),4);
            let mut missing=report;
            for operation in &mut missing.operations[..2] {
                operation.inputs[0]=operation.inputs[0].clone().with_representation(None);
                assert!(cpu.plan(operation.as_view()).unwrap().is_none());
            }
        }
    }

    #[test]
    fn cpu_singleton_concatenate_preserves_integer_and_strided_source_facts(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else{
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for (dtype,representation) in [(WorkspaceDtype::Int32,None),(WorkspaceDtype::Float32,Some(
            WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false)))] {
            let context=WorkspaceContext::new(cpu);
            let input=WorkspaceTensor::existing(context.layout(&[1,3],dtype).unwrap().with_representation(representation),&context).unwrap();
            context.begin_span();let output=WorkspaceTensor::concatenate(&[input],1,&context).unwrap();
            assert_eq!(output.layout().representation(),representation);
            let report=context.report(&[output]).unwrap();let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.alias_input,Some(0));assert_eq!(plan.population.primitives,0);assert_eq!(plan.population.births,0);
            assert_eq!(plan.parameter_shells,1);assert_eq!(plan.output_bytes+plan.scratch_bytes,0);
            let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{batch_size:1,cached_positions:0,input_positions:3,
                max_output_tokens:0,prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence},ordinary,cpu,&context).unwrap();
            let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();assert_eq!(reduced.first_missing_operation,None);
            assert_eq!(reduced.graph.unwrap().primitives(),0);
        }
    }

    #[test]
    #[ignore="requires CPU native array construction"]
    fn native_cpu_singleton_concatenate_keeps_backing_and_ordinary_axis_semantics(){
        let _pool=crate::tests::support::test_utils::initialize_original_sources();
        let stream=safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
        let source=safemlx::Array::from_slice(&[1i32,2,3,4,5,6],&[2,3]);
        let strided=source.transpose_axes(&[1,0],&stream).unwrap();strided.evaluated().unwrap();
        let before=strided.try_allocation_info().unwrap().unwrap().identity();
        let alias=safemlx::ops::concatenate_axis(&[&strided],99,&stream).unwrap();
        assert_eq!(alias.shape(),&[3,2]);assert_eq!(alias.dtype(),safemlx::Dtype::Int32);
        assert_eq!(alias.try_allocation_info().unwrap().unwrap().identity(),before);
        assert!(!alias.try_descriptor().unwrap().row_contiguous().unwrap());
        drop((source,strided));
        let read=safemlx::ops::contiguous(&alias,false,&stream).unwrap();
        assert_eq!(read.evaluated().unwrap().as_slice::<i32>(),&[1,4,2,5,3,6]);
    }

    #[test]
    fn cpu_attention_head_transpose_keeps_source_and_marks_strided_output(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else{
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        let context=WorkspaceContext::new(cpu);let input=value(&[1,3,4,8],&context,true);
        context.begin_span();let output=input.transpose_axes(&[0,2,1,3],&context).unwrap();
        assert_eq!(output.shape(),&[1,4,3,8]);
        assert!(!output.layout().representation().unwrap().row_contiguous());
        assert!(output.layout().representation().unwrap().last_axis_contiguous());
        let report=context.report(&[output]).unwrap();let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!(plan.alias_input,Some(0));assert_eq!(plan.population.births,0);assert_eq!(plan.population.primitives,1);
        assert_eq!(plan.output_bytes+plan.scratch_bytes,0);
        let mut malformed=report;malformed.operations[0].outputs[0]=context.layout(&[1,2,6,8],WorkspaceDtype::Float32).unwrap();
        assert!(cpu.plan(malformed.operations[0].as_view()).is_err());
        let repeated=value(&[2,2,8],&context,true);context.begin_span();
        let output=repeated.transpose_axes(&[1,0,2],&context).unwrap();
        assert!(!output.layout().representation().unwrap().row_contiguous());
    }

    #[test]
    fn cpu_gated_product_reuses_activation_source_and_retains_its_intermediate() {
        use eredu_nn::GatedProductPolicy;
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Bfloat16] {
            let context=WorkspaceContext::new(cpu);
            let value=||WorkspaceTensor::existing(context.layout(&[2,17],WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
            let gate=value();let up=value();context.begin_span();
            let output=WorkspaceBackend::gated_product(gate.clone(),up.clone(),GatedProductPolicy::ordinary_silu(),&context).unwrap();
            let fused=context.report(&[output]).unwrap();let plan=cpu.plan(fused.operations[0].as_view()).unwrap().unwrap();
            context.begin_span();let activated=WorkspaceBackend::silu(gate.clone(),&context).unwrap();
            let output=activated.multiply(&up,&context).unwrap();let explicit=context.report(&[output]).unwrap();
            let activation=cpu.plan(explicit.operations[0].as_view()).unwrap().unwrap();
            let product=cpu.plan(explicit.operations[1].as_view()).unwrap().unwrap();
            assert_eq!(plan.population.primitives,activation.population.primitives+product.population.primitives);
            assert_eq!(plan.population.births,activation.population.births+product.population.births);
            assert_eq!(plan.output_bytes+plan.scratch_bytes,
                activation.output_bytes+activation.scratch_bytes+product.output_bytes+product.scratch_bytes);
            assert_eq!(plan.seeds,1);assert_eq!(plan.dtype,dtype);
            let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:0,input_positions:2,
                max_output_tokens:0,prefill_chunk_positions:2,output:eredu_core::OutputDemand::Sequence},ordinary);
            recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&fused,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
            let mut bounded=fused;bounded.operations[0].kind=WorkspaceOperationKind::GatedProduct(GatedProductPolicy::bounded_silu(2.0).unwrap());
            assert!(cpu.plan(bounded.operations[0].as_view()).unwrap().is_none());
        }
    }

    #[test]
    fn cpu_row_views_preserve_source_alias_and_allocate_only_the_real_result() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Bfloat16] {
            let context=WorkspaceContext::new(cpu);let input=WorkspaceTensor::existing(context.layout(&[2,3],WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
            context.begin_span();let expanded=input.expand_dims(1,&context).unwrap();
            let flat=expanded.reshape(&[1,6,1],&context).unwrap();
            let squeezed=flat.squeeze_axes(&[0,2],&context).unwrap();
            let output=squeezed.square(&context).unwrap();let report=context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(),4);
            for operation in &report.operations[..3] {
                let plan=cpu.plan(operation.as_view()).unwrap().unwrap();
                assert_eq!(plan.alias_input,Some(0));assert_eq!(plan.output_bytes,0);assert_eq!(plan.population.births,0);
                let bound=cpu.operation_bound(operation).unwrap().unwrap();
                assert!(matches!(bound.outputs.as_slice(),[eredu_nn::workspace::WorkspaceOutputStorage::AliasInput(0)]));
            }
            let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:0,input_positions:2,
                max_output_tokens:0,prefill_chunk_positions:2,output:eredu_core::OutputDemand::Sequence},ordinary);
            recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);assert_eq!(reduced.mutable_storage.unwrap().maximum_births(),1);
            assert_eq!(reduced.dispatch.unwrap().cpu_model.unwrap().births,1);
            let mut unknown=report;unknown.operations[0].inputs[0]=unknown.operations[0].inputs[0].clone()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,false)));
            assert!(cpu.plan(unknown.operations[0].as_view()).unwrap().is_none());
        }
    }

    #[test]
    fn cpu_gemma_step_concatenation_covers_both_sources_and_copy_jobs() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Bfloat16] {
            let context=WorkspaceContext::new(cpu);
            let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
            let embedding=source(&[1,1,32]);let previous_hidden=source(&[1,1,32]);
            context.begin_span();let output=WorkspaceTensor::concatenate(&[embedding,previous_hidden],-1,&context).unwrap();
            assert_eq!(output.shape(),[1,1,64]);let report=context.report(&[output]).unwrap();
            let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.population.primitives,1);assert_eq!(plan.population.births,1);assert_eq!(plan.seeds,0);
            assert_eq!(plan.alias_input,None);
            // Same-type AsType returns each input directly. Only the joined
            // output is allocated; the two copy jobs share this one Eval.
            let output_capacity=ordinary.allocation().fixed_buffer_capacity(64*4).unwrap();
            assert_eq!(plan.scratch_bytes,0);
            assert_eq!(plan.output_bytes,output_capacity);
            let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry {batch_size:1,cached_positions:3,input_positions:1,
                max_output_tokens:0,prefill_chunk_positions:1,output:eredu_core::OutputDemand::Sequence},ordinary);
            recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
        }
    }

    #[test]
    fn cpu_gemma_unit_axis_views_and_exact_gelu_preserve_source_sequence() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        let context=WorkspaceContext::new(cpu);
        let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let input=source(&[1,1,4,8]);
        context.begin_span();let headed=input.transpose_axes(&[0,2,1,3],&context).unwrap();
        let restored=headed.transpose_axes(&[0,2,1,3],&context).unwrap();
        let flat=restored.reshape(&[1,1,32],&context).unwrap();
        let output=WorkspaceTensor::gelu(&flat,&context).unwrap();
        let report=context.report(&[output]).unwrap();
        for operation in &report.operations[..3] {
            let plan=cpu.plan(operation.as_view()).unwrap().unwrap();
            assert_eq!(plan.alias_input,Some(0));assert_eq!(plan.population.births,0);
        }
        let gelu=cpu.plan(report.operations[3].as_view()).unwrap().unwrap();
        assert_eq!(gelu.seeds,3);assert_eq!(gelu.population.primitives,22);
        assert!(gelu.scratch_bytes>gelu.output_bytes);assert_eq!(gelu.dtype,WorkspaceFloatingType::Float32);
        let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:3,input_positions:1,
            max_output_tokens:0,prefill_chunk_positions:1,output:eredu_core::OutputDemand::Sequence},ordinary);
        recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
        assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
        // Shapes alone cannot establish the physical layout of equal nonunit
        // axis exchanges, even though their logical output shape is identical.
        let square=source(&[2,2]);context.begin_span();let transposed=square.transpose_axes(&[1,0],&context).unwrap();
        assert!(!transposed.layout().representation().unwrap().row_contiguous());
        let report=context.report(&[transposed]).unwrap();
        assert_eq!(cpu.plan(report.operations[0].as_view()).unwrap().unwrap().alias_input,Some(0));
    }

    #[test]
    fn cpu_gemma_rotary_quote_retains_frequency_ranges_and_half_products() {
        use eredu_nn::{NeuralBackend,RotaryOperator,RotaryPosition,RotarySpec,RotaryAlgorithm,RotaryArithmetic};
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        let context=WorkspaceContext::new(cpu);
        let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let spec=RotarySpec{arithmetic:RotaryArithmetic::Native,algorithm:RotaryAlgorithm::Default,
            dimensions:8,traditional:false,base:10000.0};
        let input=source(&[1,4,1,8]);let mut rope=eredu_nn::workspace::WorkspaceBackend::rotary(spec,&context).unwrap();
        context.begin_span();let output=rope.forward(&input,RotaryPosition::Offset(2),&context).unwrap();
        let report=context.report(&[output]).unwrap();assert_eq!(report.operations.len(),1);
        let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();assert_eq!(plan.seeds,3);
        assert!(plan.population.primitives>50);assert!(plan.scratch_bytes>plan.output_bytes);
        let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:3,input_positions:1,
            max_output_tokens:0,prefill_chunk_positions:1,output:eredu_core::OutputDemand::Sequence},ordinary);
        recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
        assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
    }

    #[test]
    fn cpu_causal_mask_quote_keeps_integer_coordinates_and_boolean_window_results() {
        use eredu_nn::NeuralBackend;
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for (q,offset,window) in [(3,0,None),(2,3,None),(3,5,Some(0)),(3,5,Some(2))] {
            let context=WorkspaceContext::new(cpu);context.begin_span();
            let output=eredu_nn::workspace::WorkspaceBackend::causal_mask(q,offset,window,&context).unwrap();
            let report=context.report(&[output]).unwrap();assert_eq!(report.operations.len(),1);
            let operation=report.operations[0].as_view();let plan=cpu.plan(operation).unwrap().unwrap();
            assert_eq!(operation.outputs.get(0).unwrap().dtype(),WorkspaceDtype::Bool);
            assert_eq!(operation.outputs.get(0).unwrap().shape(),[q,q+offset]);
            assert_eq!(cpu.output_representation(operation,0),None);
            assert_eq!(plan.seeds,usize::from(window.is_some()));
            assert_eq!(plan.population.primitives,if window.is_some(){24}else{9});
            assert!(plan.scratch_bytes>0);assert!(plan.population.births>=3);
            let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:offset as u64,
                input_positions:q as u64,max_output_tokens:0,prefill_chunk_positions:q as u64,
                output:eredu_core::OutputDemand::Sequence},ordinary);
            recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
        }
    }

    #[test]
    fn cpu_explicit_boolean_attention_quote_counts_mask_frontend_and_gqa_select() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for heads in [2,4] {
            let context=WorkspaceContext::new(cpu);
            let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
            let q=source(&[1,heads,3,8]);let k=source(&[1,2,5,8]);let v=source(&[1,2,5,8]);
            let mask=WorkspaceTensor::existing(context.layout(&[3,5],WorkspaceDtype::Bool).unwrap(),&context).unwrap();
            context.begin_span();
            let output=eredu_nn::workspace::WorkspaceBackend::attention(q,k,v,8f32.sqrt().recip(),Some(&mask),&context).unwrap();
            let report=context.report(&[output]).unwrap();assert_eq!(report.operations.len(),1);
            let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.rank,if heads==2{4}else{5});assert_eq!(plan.seeds,2);
            assert!(plan.scratch_bytes>plan.output_bytes);
            let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:2,input_positions:3,
                max_output_tokens:0,prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence},ordinary);
            recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
            let completion=ResidentCompletionRecipe{validation_roots:0,grouped_outputs:reduced.grouped_outputs,
                traversal:reduced.traversal.unwrap(),graph:reduced.graph.unwrap(),dispatch:reduced.dispatch,
                nested_completions:0,nested_root_capacity:0};
            assert!(graph_capacity::ResidentGraphStorage::for_completion(completion).unwrap().full_capacity.unwrap()>0);
            assert!(record_capacity::ResidentRecordStorage::for_completion(completion).unwrap().full_capacity.unwrap()>0);
        }
    }

    #[test]
    fn cpu_half_attention_quote_joins_precise_dtype_mask_and_strided_value_sources(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32AndFloat16Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        use WorkspaceFloatingType::{Float32 as F,Float16 as H,Bfloat16 as B};
        for (q_type,k_type,v_type,expected) in [(H,H,H,H),(B,B,B,B),(H,B,H,F)] {
            for masked in [false,true] {
                let context=WorkspaceContext::new(cpu);
                let source=|shape:&[i32],dtype|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
                let q=source(&[1,4,3,8],q_type);let k=source(&[1,2,5,8],k_type);
                let value=source(&[1,5,2,8],v_type).transpose_axes(&[0,2,1,3],&context).unwrap();
                assert!(!value.layout().representation().unwrap().row_contiguous());
                let mask=WorkspaceTensor::existing(context.layout(&[3,5],WorkspaceDtype::Bool).unwrap(),&context).unwrap();
                context.begin_span();
                let output=eredu_nn::workspace::WorkspaceBackend::attention(q,k,value,8f32.sqrt().recip(),
                    if masked{Some(&mask)}else{None},&context).unwrap();
                assert_eq!(output.layout().representation(),Some(WorkspaceRepresentation::new(expected,true)));
                let report=context.report(&[output]).unwrap();let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.dtype,expected);assert_eq!(plan.rank,5);assert_eq!(plan.seeds,1+usize::from(masked));
                let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{batch_size:1,cached_positions:2,
                    input_positions:3,max_output_tokens:0,prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence},
                    ordinary,cpu,&context).unwrap();
                let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();assert_eq!(reduced.first_missing_operation,None);
                assert!(reduced.dispatch.unwrap().cpu_model.is_some());
                let mut missing=report;
                missing.operations[0].inputs[1]=missing.operations[0].inputs[1].clone().with_representation(None);
                assert!(cpu.plan(missing.operations[0].as_view()).unwrap().is_none());
            }
        }
    }

    #[test]
    fn cpu_gemma_grouped_attention_quote_retains_rank_five_score_and_value_paths() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        let context=WorkspaceContext::new(cpu);
        let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let q=source(&[1,4,1,8]);let k=source(&[1,2,3,8]);let v=source(&[1,2,3,8]);
        context.begin_span();let output=WorkspaceTensor::scaled_dot_product_attention(&q,&k,&v,
            8f32.sqrt().recip(),eredu_nn::AttentionMask::None,&context).unwrap();
        let report=context.report(&[output]).unwrap();assert_eq!(report.operations.len(),1);
        let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!(plan.rank,5);assert_eq!(plan.seeds,1);assert!(plan.scratch_bytes>plan.output_bytes);
        let mut recorder=ResidentRecipeRecorder::new(InferenceGeometry{batch_size:1,cached_positions:3,input_positions:1,
            max_output_tokens:0,prefill_chunk_positions:1,output:eredu_core::OutputDemand::Sequence},ordinary);
        recorder.cpu=Some(cpu);let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
        assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
        // No opaque platform F32 Matmul source is borrowed from the selected SIMD path.
        let platform=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),
            MlxCpuMatmulMechanism::select(CpuMatmulImplementation::PlatformDefault).unwrap());
        assert!(platform.plan(report.operations[0].as_view()).unwrap().is_none());
    }


    #[test]
    fn cpu_exact_head_permutation_proves_final_axis_for_repeated_prefill_geometry() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for shape in [[1,3,3,8],[1,8,3,8],[1,1,3,8]] {
            let context=WorkspaceContext::new(cpu);
            let source=WorkspaceTensor::existing(context.layout(&shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
            context.begin_span();
            let output=source.transpose_axes(&[0,2,1,3],&context).unwrap();
            let representation=output.layout().representation().unwrap();
            assert!(representation.last_axis_contiguous());
            assert_eq!(representation.row_contiguous(),shape[1]==1);
            let report=context.report(&[output]).unwrap();let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.alias_input,Some(0));assert_eq!(plan.population.births,0);
            let mut missing=report.operations[0].clone();missing.kind=WorkspaceOperationKind::View("transpose");
            assert!(cpu.plan(missing.as_view()).unwrap().is_none());
            assert!(cpu.output_representation(missing.as_view(),0).is_none());
            let mut invalid=report.operations[0].clone();invalid.kind=WorkspaceOperationKind::Transpose(vec![0,2,1,2]);
            assert!(cpu.plan(invalid.as_view()).is_err());
            context.begin_span();
            let changed=source.transpose_axes(&[0,1,3,2],&context).unwrap();
            assert!(!changed.layout().representation().unwrap().last_axis_contiguous());
        }
    }

    #[test]
    fn cpu_strided_head_rms_and_cached_concatenation_share_exact_row_sources() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for positions in [1,3,8] {
            let context=WorkspaceContext::new(cpu);
            let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
            let input=source(&[1,positions,4,8]);let gain=source(&[8]);
            context.begin_span();let headed=input.transpose_axes(&[0,2,1,3],&context).unwrap();
            let normalized=eredu_nn::workspace::WorkspaceBackend::rms_norm_with_weight(&headed,&gain,1e-6,&context).unwrap();
            assert!(normalized.layout().representation().unwrap().row_contiguous());
            let joined=WorkspaceTensor::concatenate(&[headed.clone(),normalized],2,&context).unwrap();
            assert_eq!(joined.shape(),&[1,4,positions*2,8]);
            let report=context.report(&[joined]).unwrap();
            assert!(report.operations.iter().all(|op|cpu.plan(op.as_view()).unwrap().is_some()));
            let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry {
                batch_size:1,cached_positions:0,input_positions:positions as u64,max_output_tokens:0,
                prefill_chunk_positions:positions as u64,output:eredu_core::OutputDemand::Sequence,
            },ordinary,cpu,&context).unwrap();
            let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
            let mut missing=report.operations[1].clone();
            missing.inputs[0]=missing.inputs[0].clone().with_representation(Some(
                WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false)));
            assert!(cpu.plan(missing.as_view()).unwrap().is_none());
            let mut wrong_gain=report.operations[1].clone();
            wrong_gain.inputs[1]=wrong_gain.inputs[1].clone().with_representation(Some(
                WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false).with_last_axis_contiguous(true)));
            assert!(cpu.plan(wrong_gain.as_view()).unwrap().is_none());
        }
    }

    #[test]
    fn cpu_masked_attention_prices_the_actual_strided_value_copy() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        let context=WorkspaceContext::new(cpu);
        let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let q=source(&[1,4,3,8]);let k=source(&[1,2,5,8]);let raw_v=source(&[1,5,2,8]);
        let v=raw_v.transpose_axes(&[0,2,1,3],&context).unwrap();
        let mask=WorkspaceTensor::existing(context.layout(&[3,5],WorkspaceDtype::Bool).unwrap(),&context).unwrap();
        context.begin_span();let output=eredu_nn::workspace::WorkspaceBackend::attention(q,k,v,8f32.sqrt().recip(),Some(&mask),&context).unwrap();
        let report=context.report(&[output]).unwrap();assert_eq!(report.operations.len(),1);
        let strided=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        let mut compact=report.operations[0].clone();compact.inputs[2]=compact.inputs[2].clone().with_representation(
            Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
        let base=cpu.plan(compact.as_view()).unwrap().unwrap();
        assert_eq!(strided.population.births,base.population.births+1);
        assert!(strided.population.extents>base.population.extents);
        assert!(strided.scratch_bytes>base.scratch_bytes);
        let mut missing=report.operations[0].clone();missing.inputs[2]=missing.inputs[2].clone().with_representation(
            Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false)));
        assert!(cpu.plan(missing.as_view()).unwrap().is_none());
        let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry {
            batch_size:1,cached_positions:2,input_positions:3,max_output_tokens:0,
            prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence,
        },ordinary,cpu,&context).unwrap();
        let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
        assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
    }

    #[test]
    fn cpu_head_join_prices_copy_before_row_normalization_and_keeps_alias_refusal() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for positions in [1,3,4] {
            let context=WorkspaceContext::new(cpu);
            let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
            let input=source(&[1,4,positions,8]);let gain=source(&[32]);
            context.begin_span();
            let transposed=input.transpose_axes(&[0,2,1,3],&context).unwrap();
            let joined=transposed.reshape(&[1,positions,32],&context).unwrap();
            assert!(joined.layout().representation().unwrap().row_contiguous());
            let output=eredu_nn::workspace::WorkspaceBackend::rms_norm_with_weight(&joined,&gain,1e-6,&context).unwrap();
            let report=context.report(&[output]).unwrap();
            let plan=cpu.plan(report.operations[1].as_view()).unwrap().unwrap();
            let copied=positions!=1;
            assert_eq!(plan.population.births,usize::from(copied));
            assert_eq!(plan.alias_input,(!copied).then_some(0));
            assert_eq!(plan.output_bytes>0,copied);
            let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry {
                batch_size:1,cached_positions:0,input_positions:positions as u64,max_output_tokens:0,
                prefill_chunk_positions:positions as u64,output:eredu_core::OutputDemand::Sequence,
            },ordinary,cpu,&context).unwrap();
            let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);assert!(reduced.dispatch.unwrap().cpu_model.is_some());
            let mut unknown=report.operations[1].clone();
            unknown.inputs[0]=unknown.inputs[0].clone().with_representation(Some(
                WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false).with_last_axis_contiguous(true)));
            assert!(cpu.plan(unknown.as_view()).unwrap().is_none());
        }
        // Splitting a contiguous inner dimension preserves a strided alias;
        // that result must not acquire a row-major declaration.
        let context=WorkspaceContext::new(cpu);
        let input=WorkspaceTensor::existing(context.layout(&[4,3,8],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        context.begin_span();let headed=input.transpose_axes(&[1,0,2],&context).unwrap();
        let output=headed.reshape(&[3,4,2,4],&context).unwrap();
        assert!(!output.layout().representation().unwrap().row_contiguous());
        let report=context.report(&[output]).unwrap();
        let plan=cpu.plan(report.operations[1].as_view()).unwrap().unwrap();
        assert_eq!(plan.alias_input,Some(0));assert_eq!(plan.population.births,0);assert_eq!(plan.output_bytes,0);
    }

    #[test]
    fn cpu_ordered_readout_composes_integer_sources_and_real_selected_replicas() {
        use eredu_nn::multimodal::MaskedOutputProjectionInput;
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let Some(selected)=MlxCpuMatmulMechanism::select(CpuMatmulImplementation::Float32Tiles) else {
            assert!(std::env::var_os("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").is_none());return;
        };
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for index_dtype in [WorkspaceDtype::Int32,WorkspaceDtype::Uint32] {
            let context=WorkspaceContext::new(cpu);
            let source=|shape:&[i32]|WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
            let hidden=source(&[2,3,32]);let weight=source(&[32,32]);let centroids=source(&[2,3,4]);
            let ordering=WorkspaceTensor::existing(context.layout(&[32],index_dtype).unwrap(),&context).unwrap();
            context.begin_span();
            let output=WorkspaceTensor::masked_output_projection(MaskedOutputProjectionInput {
                hidden:&hidden,output_weight:&weight,centroid_logits:&centroids,token_ordering:&ordering,
                top_centroids:2,mask_margin:1.0,
            },&context).unwrap();
            let report=context.report(&[output]).unwrap();assert_eq!(report.operations.len(),1);
            let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.rank,5);assert_eq!(plan.seeds,1);assert_eq!(plan.alias_input,None);
            assert!(plan.population.input_edges>plan.population.primitives);
            // The real gathered weight contains one selected copy per position.
            let replicas=ordinary.allocation().fixed_buffer_capacity(2*3*16*32*4).unwrap();
            assert!(plan.scratch_bytes>=replicas);
            assert_eq!(report.tensor_buffers.retained_bytes,Some(plan.output_bytes));
            let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry {
                batch_size:2,cached_positions:3,input_positions:3,max_output_tokens:0,
                prefill_chunk_positions:3,output:eredu_core::OutputDemand::Sequence,
            },ordinary,cpu,&context).unwrap();
            let reduced=recorder.reduce_trace(&report,None,0,1).unwrap();
            assert_eq!(reduced.first_missing_operation,None);
            assert!(reduced.dispatch.unwrap().cpu_model.is_some());
            assert!(reduced.mutable_storage.unwrap().maximum_births()>1);
            let mut unknown=report.operations[0].clone();
            unknown.inputs[0]=unknown.inputs[0].clone().with_representation(
                Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float16,true)));
            assert!(cpu.plan(unknown.as_view()).unwrap().is_none());
            let platform=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),
                MlxCpuMatmulMechanism::select(CpuMatmulImplementation::PlatformDefault).unwrap());
            assert!(platform.plan(report.operations[0].as_view()).unwrap().is_none());
        }
    }

}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod invocation_tests;

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod target_trace;

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
#[test]
fn cpu_value_frontiers_quote_exact_borrowed_roots_without_tensor_births(){
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let choice=super::super::MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),choice);
    for count in [1,5] {
        let context=WorkspaceContext::new(cpu);
        let values:Vec<_>=(0..count).map(|_|WorkspaceTensor::existing(context.layout(&[1,2,16],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap()).collect();
        let refs:Vec<_>=values.iter().collect();
        context.begin_state_span(refs.iter().copied()).unwrap();
        context.retain_values(&refs).unwrap();context.complete_values(&refs).unwrap();
        let report=context.report(&values).unwrap();
        assert!(report.unpriced_operations.is_empty()&&report.unpriced_host_operations.is_empty());
        assert_eq!(report.operations.len(),2);assert_eq!(report.tensor_buffers.total_bytes,Some(0));
        let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{batch_size:1,cached_positions:0,input_positions:2,
            max_output_tokens:1,prefill_chunk_positions:2,output:eredu_core::OutputDemand::LastPosition},ordinary,cpu,&context).unwrap();
        let reduced=recorder.reduce_cpu_trace(&report,None,count,count,cpu).unwrap();
        assert_eq!(reduced.first_missing_operation,None);assert_eq!(reduced.nested_completions,1);
        assert_eq!(reduced.nested_root_capacity,count);assert_eq!(reduced.mutable_storage.unwrap().maximum_births(),0);
        assert!(reduced.graph.is_some()&&reduced.dispatch.is_some()&&reduced.query_controls.is_some());
    }
}
