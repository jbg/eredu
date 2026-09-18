use super::*;
use crate::backend::nn::logical_collective::{self, packed, routed};
use eredu_nn::Tensor;

fn mechanisms() -> (MlxMetalWorkspaceMechanisms, MlxCpuWorkspaceMechanisms) {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    (ordinary, MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected))
}

fn trace_gather(shape: &[i32], count: usize, packed_input: bool,
    ordinary: MlxMetalWorkspaceMechanisms, cpu: MlxCpuWorkspaceMechanisms) -> SpeculativeNumericalRecipe {
    let context = WorkspaceContext::new(cpu);
    let input = |shape| WorkspaceTensor::existing(
        context.layout(shape, WorkspaceDtype::Int32).unwrap(), &context).unwrap();
    let output = if packed_input {
        let mut world_shape = vec![count as i32];
        world_shape.extend_from_slice(shape);
        let world = input(&world_shape);
        context.begin_state_span([&world]).unwrap();
        let members = (0..count).rev().collect::<Vec<_>>();
        let ops = logical_collective::Workspace(&context);
        let stacked = packed::gather_stacked(&ops, &world, &members).unwrap();
        packed::flatten(&ops, &stacked, shape, count).unwrap()
    } else {
        let inputs = (0..count).map(|_| input(shape)).collect::<Vec<_>>();
        context.begin_state_span(inputs.iter()).unwrap();
        let values = inputs.into_iter().enumerate().rev().collect();
        routed::gather(&logical_collective::Workspace(&context), values, shape).unwrap()
    };
    assert_eq!(output.layout().dtype(), WorkspaceDtype::Int32);
    assert_eq!(output.layout().representation(), None);
    let report = context.finish_report(&[output]).unwrap();
    for operation in &report.operations {
        assert!(cpu.plan(operation.as_view()).unwrap().is_some(), "{operation:?}");
        assert!(operation.outputs.iter().all(|v| v.dtype()==WorkspaceDtype::Int32 && v.representation().is_none()));
    }
    assert!(report.unpriced_operations.is_empty());
    assert!(report.unpriced_host_operations.is_empty());
    assert!(report.inference_transient_bytes().is_some());
    SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context).unwrap()
}

#[test]
fn cpu_integer_collective_views_quote_full_ranked_and_packed_gather() {
    let (ordinary, cpu) = mechanisms();
    for shape in [&[][..], &[18][..], &[2,3][..], &[1,2,3][..]] {
        for count in [2,4] { for packed in [false,true] {
            let recipe = trace_gather(shape,count,packed,ordinary,cpu);
            assert_eq!(recipe.kernels,0);
            assert!(recipe.storage.maximum_births()>=1);
            assert!(recipe.graph_capacity>0 && recipe.record_capacity>0);
        }}
    }
}

#[test]
fn cpu_integer_views_preserve_full_alias_custody_and_possible_reshape_copy() {
    let (ordinary,cpu)=mechanisms();
    for dtype in [WorkspaceDtype::Int32,WorkspaceDtype::Uint32] {
        let context=WorkspaceContext::new(cpu);
        let storage=WorkspaceExistingStorage::try_new(Some(8192),&context).unwrap();
        let input=WorkspaceTensor::existing_with_storage(context.layout(&[2,3],dtype).unwrap(),&storage,&context).unwrap();
        context.begin_state_span([&input]).unwrap();
        let expanded=input.expand_dims(0,&context).unwrap();
        let squeezed=expanded.squeeze_axes(&[0],&context).unwrap();
        let transposed=squeezed.transpose_axes(&[1,0],&context).unwrap();
        let aliases=context.report(std::slice::from_ref(&transposed)).unwrap();
        assert_eq!(aliases.tensor_buffers.total_bytes,Some(0));
        assert_eq!(aliases.state.as_ref().unwrap().retained_bytes,Some(8192));
        for op in &aliases.operations {
            let plan=cpu.plan(op.as_view()).unwrap().unwrap();
            assert_eq!((plan.alias_input,plan.population.births),(Some(0),0));
        }
        let output=transposed.reshape(&[6],&context).unwrap();
        assert_eq!(output.layout().dtype(),dtype);assert_eq!(output.layout().representation(),None);
        let report=context.finish_report(&[output]).unwrap();
        let plan=cpu.plan(report.operations.last().unwrap().as_view()).unwrap().unwrap();
        assert_eq!((plan.alias_input,plan.population.births),(None,1));
        // Actual reshape may alias a larger input or create its own complete
        // row. Both outcomes retain their real owners in the storage report.
        assert!(report.state.as_ref().unwrap().retained_bytes.unwrap()>=8192);
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_outputs(&report,1,ordinary,cpu,&context).unwrap();
        assert_eq!(recipe.storage.maximum_births(),1);
        assert_eq!(recipe.kernels,0);
    }
}

#[test]
fn cpu_structural_views_refuse_mismatched_geometry_and_unproved_precision() {
    let (_,cpu)=mechanisms();
    let inspect=|kind,input:&[i32],output:&[i32],source_dtype,target_dtype| {
        let inputs=[WorkspaceLayoutView::new(input,source_dtype).unwrap()];
        let outputs=[WorkspaceLayoutView::new(output,target_dtype).unwrap()];
        cpu.plan(WorkspaceOperationView {kind,inputs:WorkspaceLayoutList::Views(&inputs),outputs:WorkspaceLayoutList::Views(&outputs)})
    };
    for (kind,input,output) in [
        (WorkspaceOperationKindView::View("expand_dims"),&[2,3][..],&[1,3,2][..]),
        (WorkspaceOperationKindView::View("squeeze"),&[1,2,3][..],&[3,2][..]),
        (WorkspaceOperationKindView::View("reshape"),&[18][..],&[17][..]),
        (WorkspaceOperationKindView::Transpose(&[0,0]),&[2,2][..],&[2,2][..]),
    ] {assert!(inspect(kind,input,output,WorkspaceDtype::Int32,WorkspaceDtype::Int32).is_err());}
    for (input,output,source,target) in [
        (&[18][..],&[1,18][..],WorkspaceDtype::Int32,WorkspaceDtype::Uint32),
        (&[18][..],&[1,18][..],WorkspaceDtype::Float32,WorkspaceDtype::Float32),
        (&[i32::MAX,2][..],&[1,i32::MAX,2][..],WorkspaceDtype::Int32,WorkspaceDtype::Int32),
        (&[1,1,1,1][..],&[1,1,1,1,1][..],WorkspaceDtype::Int32,WorkspaceDtype::Int32),
    ] {assert!(inspect(WorkspaceOperationKindView::View("expand_dims"),input,output,source,target).unwrap().is_none());}
}

#[test]
fn cpu_empty_reshape_keeps_actual_dtype_and_zero_birth_alias() {
    let (ordinary,cpu)=mechanisms();
    for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Float16,WorkspaceFloatingType::Bfloat16] {
        for (shape,target) in [(&[0][..],&[0,8][..]),(&[2,0,8][..],&[0,16][..])] {
            let context=WorkspaceContext::new(cpu);
            let input=WorkspaceTensor::existing(context.layout(shape,WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
            context.begin_state_span([&input]).unwrap();
            let output=input.reshape(target,&context).unwrap();
            assert_eq!(output.layout().representation().unwrap().dtype(),dtype);
            let report=context.finish_report(&[output]).unwrap();
            let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!((plan.alias_input,plan.population.births,plan.output_bytes),(Some(0),0,0));
            assert!(report.unpriced_operations.is_empty());assert!(report.unpriced_host_operations.is_empty());
            let recipe=SpeculativeNumericalRecipe::inspect_cpu_outputs(&report,1,ordinary,cpu,&context).unwrap();
            assert_eq!(recipe.storage.maximum_births(),0);
        }
        let context=WorkspaceContext::new(cpu);
        let input=WorkspaceTensor::existing(context.layout(&[0,1,8],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&context).unwrap();
        context.begin_state_span([&input]).unwrap();
        let output=input.squeeze_axes(&[1],&context).unwrap();
        assert_eq!(output.shape(),[0,8]);
        assert_eq!(output.layout().representation().unwrap().dtype(),dtype);
        let report=context.finish_report(&[output]).unwrap();
        let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!((plan.alias_input,plan.population.births,plan.output_bytes),(Some(0),0,0));
        assert!(report.unpriced_operations.is_empty());
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_outputs(&report,1,ordinary,cpu,&context).unwrap();
        assert_eq!(recipe.storage.maximum_births(),0);
    }
}

#[path = "native.rs"]
mod native;
