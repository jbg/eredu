//! Output geometry of the selected existing native collective workers.
use super::*;

pub(super) fn control_bytes() -> usize {
    std::mem::size_of::<(&ResidentExecutionMechanisms, WorkspaceOperationView<'_>, usize)>()
        + std::mem::size_of::<Option<WorkspaceRepresentation>>()
        + std::mem::size_of::<WorkspaceRepresentation>()
        + std::mem::size_of::<bool>()
        + std::mem::size_of::<Option<usize>>()
}
pub(super) fn collective(mechanism: &ResidentExecutionMechanisms,
    operation: WorkspaceOperationView<'_>, output: usize) -> Option<WorkspaceRepresentation> {
    if output != 0 || operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return None;
    }
    let input = operation.inputs.get(0)?.representation()?;
    if matches!(operation.kind,WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Boundary {..})) {
        // The sender returns this original value. The receiver's byte payload
        // is either already dense or materialized by the existing zero-add and
        // typed View before reshape. Preserve only common positive row/last-
        // axis facts; never transfer a sender's arbitrary physical strides.
        return Some(WorkspaceRepresentation::new(input.dtype(),input.row_contiguous())
            .with_last_axis_contiguous(input.last_axis_contiguous()));
    }

    // CPU AllReduce donates an already row-contiguous input or makes the
    // existing contiguous copy; AllGather allocates its dense output after
    // ensuring row-contiguous input. Logical Sum/Stack and packed selection
    // use those same complete rows through their quoted arithmetic/slices.
    // Publication settles the root/zero contribution and invokes this same
    // native CPU Sum worker, so its output has the same row guarantee.
    // Preserve the exact input precision. This fact does not provide missing
    // input dtype, worker capability or communication source authority.
    let mut row_contiguous = false;
    // Original publication always binds GroupWorkerOperation::Sum on the
    // retained CPU communication stream, including Metal model execution.
    // Its settled contribution therefore has the same complete output row as
    // the CPU model path; compute-device selection is not its storage policy.
    if matches!(mechanism, ResidentExecutionMechanisms::Cpu { .. })
        || matches!(operation.kind, WorkspaceOperationKindView::Collective(
            WorkspaceCollectiveView::Broadcast { .. }))
    {
        let count = match operation.kind {
            WorkspaceOperationKindView::Collective(
                WorkspaceCollectiveView::Sum { partitions, .. }
                | WorkspaceCollectiveView::Broadcast { partitions, .. }
            ) => Some(partitions),
            WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::GatherFirstAxis { peer_widths, .. }) => Some(peer_widths.len()),
            _ => None,
        };
        match count {
            Some(0) => return None,
            // Native all_sum/all_gather return x before any new primitive for
            // one member. Preserve both actual stride facts of that alias.
            Some(1) => return Some(input),
            Some(_) => row_contiguous = true,
            None => {},
        }
    }
    Some(WorkspaceRepresentation::new(input.dtype(), row_contiguous))
}

#[cfg(all(test, target_vendor="apple", feature="metal", not(feature="cuda")))]
mod tests {
    use super::*;
    use eredu_nn::{NeuralBackend, Tensor};
    /// Cold geometry uses the production collective layout function. Native
    /// execution authority remains absent; this fixture cannot submit a group.
    #[derive(Debug, Clone, Copy)]
    struct CpuCollectiveGeometry {
        ordinary: MlxMetalWorkspaceMechanisms,
        cpu: MlxCpuWorkspaceMechanisms,
    }
    impl WorkspaceMechanisms for CpuCollectiveGeometry {
        fn output_representation(&self, operation: WorkspaceOperationView<'_>, output: usize)
            -> Option<WorkspaceRepresentation> {
            if matches!(operation.kind, WorkspaceOperationKindView::Collective(_)) {
                collective(&ResidentExecutionMechanisms::Cpu {
                    ordinary:self.ordinary, cpu:self.cpu }, operation, output)
            } else { self.cpu.output_representation(operation, output) }
        }
        fn operation_bound(&self, operation: &WorkspaceOperation)
            -> Result<Option<WorkspaceOperationBound>, Error> {
            self.cpu.operation_bound(operation)
        }
        fn host_workspace_bound(&self, operation: &WorkspaceOperation)
            -> Result<Option<WorkspaceHostBound>, Error> {
            self.cpu.host_workspace_bound(operation)
        }
    }
    #[test]
    fn cpu_softplus_gate_supplies_exact_sum_input_representation() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        let mechanism = ResidentExecutionMechanisms::Cpu { ordinary, cpu };
        for dtype in [WorkspaceFloatingType::Float32, WorkspaceFloatingType::Bfloat16, WorkspaceFloatingType::Float16] {
            let context = WorkspaceContext::new(cpu);
            let input = WorkspaceTensor::existing(context.layout(&[1, 2, 8], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype, true))), &context).unwrap();
            context.begin_span();
            let output = WorkspaceBackend::softplus(input, std::f32::consts::LN_2, &context).unwrap();
            // The failing TP attention path applies its gate before the local
            // output projection, then all-sums that projection. Exercise the
            // actual F32 chain rather than assigning a collective input fact.
            let output = if dtype == WorkspaceFloatingType::Float32 {
                let attended = WorkspaceTensor::existing(context.layout(&[1, 2, 8], WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, true))), &context).unwrap();
                let gated = attended.multiply(&output, &context).unwrap();
                let weight = WorkspaceTensor::existing(context.layout(&[8, 8], WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, true))), &context).unwrap();
                WorkspaceTensor::linear(&gated, &weight, None, &context).unwrap()
            } else { output };
            let layouts = [output.layout().as_view()];
            let sum = WorkspaceOperationView {
                kind: WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Sum { partitions: 2, rank: 0 }),
                inputs: WorkspaceLayoutList::Views(&layouts), outputs: WorkspaceLayoutList::Views(&layouts),
            };
            assert_eq!(collective(&mechanism, sum, 0), Some(WorkspaceRepresentation::new(dtype, true)));
            let unknown = [layouts[0].with_representation(None)];
            assert!(collective(&mechanism, WorkspaceOperationView { inputs: WorkspaceLayoutList::Views(&unknown), ..sum }, 0).is_none());
            let report = context.finish_report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty() && report.unpriced_host_operations.is_empty());
            SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context).unwrap();
        }
    }

    #[test]
    fn uneven_vocabulary_gather_keeps_exact_slice_and_zero_fill_sources() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),choice);
        for widths in [[32usize,32usize],[17,15]] {
            for rank in 0..2 {
                let context=WorkspaceContext::new(CpuCollectiveGeometry {ordinary,cpu});
                let input=WorkspaceTensor::existing(context.layout(&[1,1,widths[rank] as i32],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
                context.begin_span();
                let output=input.gather_uneven_axis(2,rank,&widths,&context).unwrap();
                assert_eq!(output.shape(),[1,1,widths.iter().sum::<usize>() as i32]);
                assert_eq!(output.layout().representation(),Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
                let report=context.report(&[output]).unwrap();
                let slices:Vec<_>=report.operations.iter().filter(|op|matches!(op.kind,WorkspaceOperationKind::StaticSlice{..})).collect();
                assert_eq!(slices.len(),2+usize::from(widths[0]!=widths[1]));
                for op in &report.operations {
                    if !matches!(op.kind,WorkspaceOperationKind::Collective(_)) {
                        assert!(cpu.plan(op.as_view()).unwrap().is_some(),"{:?}",op.kind);
                    }
                }
                for op in slices {
                    let plan=cpu.plan(op.as_view()).unwrap().unwrap();
                    assert_eq!(plan.population.births,0);
                    assert_eq!(plan.alias_input,Some(0));
                }
            }
        }
    }
    #[test]
    fn cpu_sum_and_gather_output_rows_feed_actual_rms_and_projection_sources() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        let mechanism = ResidentExecutionMechanisms::Cpu { ordinary, cpu };
        for dtype in [WorkspaceFloatingType::Float32, WorkspaceFloatingType::Bfloat16] {
            for gather in [false, true] {
                let context = WorkspaceContext::new(cpu);
                let inputs = [WorkspaceLayoutView::new(&[1,2,17], WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, false)))];
                let output_shape = if gather { [2,2,17] } else { [1,2,17] };
                let outputs = [WorkspaceLayoutView::new(&output_shape, WorkspaceDtype::Float32).unwrap()];
                let widths = [1,1];
                let operation = WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::Collective(if gather {
                        WorkspaceCollectiveView::GatherFirstAxis { axis:0, rank:0, peer_widths:&widths }
                    } else { WorkspaceCollectiveView::Sum { partitions:2, rank:0 } }),
                    inputs: WorkspaceLayoutList::Views(&inputs), outputs: WorkspaceLayoutList::Views(&outputs),
                };
                let publication = WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Broadcast {
                        group: eredu_core::CollectiveGroupId::new(1), root: 0, rank: 0, partitions: 2,
                    }), outputs: WorkspaceLayoutList::Views(&inputs), ..operation
                };
                assert_eq!(collective(&mechanism, publication, 0),
                    Some(WorkspaceRepresentation::new(dtype, true)));
                assert_eq!(collective(&ResidentExecutionMechanisms::Metal(ordinary), publication, 0),
                    Some(WorkspaceRepresentation::new(dtype, true)));
                let representation = collective(&mechanism, operation, 0).unwrap();
                assert_eq!(representation, WorkspaceRepresentation::new(dtype, true));
                let input = WorkspaceTensor::existing(context.layout(&output_shape, WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(representation)), &context).unwrap();
                let weight = |shape: &[i32]| WorkspaceTensor::existing(context.layout(shape, WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype, true))), &context).unwrap();
                let gain = weight(&[17]); let projection = weight(&[11,17]);
                context.begin_span();
                let normalized = WorkspaceBackend::rms_norm_with_weight(&input, &gain, 1e-6, &context).unwrap();
                let output = WorkspaceTensor::linear(&normalized, &projection, None, &context).unwrap();
                assert_eq!(output.layout().representation(), Some(WorkspaceRepresentation::new(dtype, true)));
                let report = context.report(&[output]).unwrap();
                assert!(report.unpriced_operations.is_empty());
                SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context).unwrap();
                let strided = WorkspaceRepresentation::new(dtype, false).with_last_axis_contiguous(true);
                let singleton_inputs = [inputs[0].with_representation(Some(strided))];
                let singleton_widths = [1];
                let singleton = WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::Collective(if gather {
                        WorkspaceCollectiveView::GatherFirstAxis { axis:0, rank:0, peer_widths:&singleton_widths }
                    } else { WorkspaceCollectiveView::Sum { partitions:1, rank:0 } }),
                    inputs: WorkspaceLayoutList::Views(&singleton_inputs),
                    outputs: WorkspaceLayoutList::Views(&singleton_inputs),
                };
                assert_eq!(collective(&mechanism, singleton, 0), Some(strided));
                let unknown = [inputs[0].with_representation(None)];
                assert!(collective(&mechanism, WorkspaceOperationView {
                    inputs: WorkspaceLayoutList::Views(&unknown), ..operation
                }, 0).is_none());
                assert_eq!(collective(&ResidentExecutionMechanisms::Metal(ordinary), operation, 0),
                    Some(WorkspaceRepresentation::new(dtype, false)));
            }
        }
    }
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
#[test]
fn framed_boundary_preserves_proved_rows_and_not_arbitrary_source_strides(){
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let choice=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),choice);
    for mechanism in [ResidentExecutionMechanisms::Cpu{ordinary,cpu},ResidentExecutionMechanisms::Metal(ordinary)] {
        for dtype in [WorkspaceFloatingType::Float16,WorkspaceFloatingType::Bfloat16,WorkspaceFloatingType::Float32] {
            for rows in [false,true] {
                let known=WorkspaceRepresentation::new(dtype,rows).with_last_axis_contiguous(true);
                let source=known.with_element_strides(&[64,32,1]).unwrap();
                let layouts=[WorkspaceLayoutView::new(&[1,2,16],WorkspaceDtype::Float32).unwrap().with_representation(Some(source))];
                let operation=WorkspaceOperationView{kind:WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Boundary{
                    route:0,ordinal:0,header_bytes:77}),inputs:WorkspaceLayoutList::Views(&layouts),outputs:WorkspaceLayoutList::Views(&layouts)};
                assert_eq!(collective(&mechanism,operation,0),Some(known));
                let unknown=[layouts[0].with_representation(None)];
                assert_eq!(collective(&mechanism,WorkspaceOperationView{inputs:WorkspaceLayoutList::Views(&unknown),..operation},0),None);
            }
        }
    }
}
