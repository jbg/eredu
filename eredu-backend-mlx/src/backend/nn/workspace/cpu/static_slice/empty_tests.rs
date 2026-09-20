use super::*;
use eredu_nn::Tensor;

#[test]
fn cpu_empty_static_slice_prices_allocator_backing_without_retaining_source() {
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    let capacity=ordinary.allocation().buffer_capacity(0).unwrap();
    let births=usize::from(capacity!=0);
    let cases=[
        (WorkspaceDtype::Int32,None),(WorkspaceDtype::Uint32,None),
        (WorkspaceDtype::Uint8,None),(WorkspaceDtype::Bool,None),
        (WorkspaceDtype::Float32,Some(WorkspaceFloatingType::Float32)),
        (WorkspaceDtype::Float32,Some(WorkspaceFloatingType::Float16)),
        (WorkspaceDtype::Float32,Some(WorkspaceFloatingType::Bfloat16)),
    ];
    for (dtype,precision) in cases {
        for shape in [&[3][..],&[1,1][..],&[2,3,4][..],&[2,3,4,5][..]] {
            let context=WorkspaceContext::new(cpu);
            let storage=WorkspaceExistingStorage::try_new(Some(8192),&context).unwrap();
            let input=WorkspaceTensor::existing_with_storage(context.layout(shape,dtype).unwrap()
                .with_representation(precision.map(|dtype|WorkspaceRepresentation::new(dtype,true))),
                &storage,&context).unwrap();
            context.begin_state_span([&input]).unwrap();
            let output=input.narrow_axis(0,0,0,&context).unwrap();
            let report=context.finish_report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty(),"{dtype:?} {shape:?}");
            assert!(report.unpriced_host_operations.is_empty());
            assert_eq!(report.tensor_buffers.total_bytes,Some(capacity));
            assert_eq!(report.state.as_ref().unwrap().retained_bytes,Some(capacity));
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes,Some(8192));
            let operation=report.operations[0].as_view();
            let plan=cpu.plan(operation).unwrap().unwrap();
            assert_eq!((plan.alias_input,plan.population.births,plan.output_bytes,plan.scratch_bytes),
                (None,births,capacity,0));
            assert_eq!((plan.population.primitives,plan.population.maximum_captures),(1,1));
            assert!(plan.population.extents>0&&plan.population.controls>0);
            assert_eq!(cpu.output_representation(operation,0).map(|r|r.dtype()),precision);
            let recipe=SpeculativeNumericalRecipe::inspect_cpu_outputs(&report,1,ordinary,cpu,&context).unwrap();
            assert_eq!(recipe.storage.maximum_births(),births);
            assert_eq!(recipe.kernels,0);
        }
    }
}

#[test]
fn cpu_empty_static_slice_keeps_identity_alias_and_exact_coordinates_distinct() {
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    let context=WorkspaceContext::new(cpu);
    let storage=WorkspaceExistingStorage::try_new(Some(8192),&context).unwrap();
    let input=WorkspaceTensor::existing_with_storage(context.layout(&[0,8],WorkspaceDtype::Int32).unwrap(),
        &storage,&context).unwrap();
    context.begin_state_span([&input]).unwrap();
    let output=input.narrow_axis(0,0,0,&context).unwrap();
    let report=context.finish_report(&[output]).unwrap();
    let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
    assert_eq!((plan.alias_input,plan.population.primitives,plan.parameter_shells),(Some(0),0,1));
    assert_eq!(report.state.as_ref().unwrap().retained_bytes,Some(8192));
    let inputs=[WorkspaceLayoutView::new(&[1,1],WorkspaceDtype::Int32).unwrap()];
    let outputs=[WorkspaceLayoutView::new(&[0,1],WorkspaceDtype::Int32).unwrap()];
    let operation=WorkspaceOperationView{kind:WorkspaceOperationKindView::StaticSlice{
        starts:&[0,0],ends:&[0,1],strides:&[1,1]},
        inputs:WorkspaceLayoutList::Views(&inputs),outputs:WorkspaceLayoutList::Views(&outputs)};
    assert!(cpu.plan(operation).unwrap().is_some());
    for kind in [
        WorkspaceOperationKindView::StaticSlice{starts:&[0,0],ends:&[1,1],strides:&[1,1]},
        WorkspaceOperationKindView::StaticSlice{starts:&[0,0],ends:&[0,1],strides:&[0,1]},
        WorkspaceOperationKindView::StaticSlice{starts:&[2,0],ends:&[2,1],strides:&[1,1]},
    ] {assert!(cpu.plan(WorkspaceOperationView{kind,..operation}).is_err());}
    let float_inputs=[WorkspaceLayoutView::new(&[1,1],WorkspaceDtype::Float32).unwrap()];
    let float_outputs=[WorkspaceLayoutView::new(&[0,1],WorkspaceDtype::Float32).unwrap()];
    assert!(cpu.plan(WorkspaceOperationView{inputs:WorkspaceLayoutList::Views(&float_inputs),
        outputs:WorkspaceLayoutList::Views(&float_outputs),..operation}).unwrap().is_none());
}
