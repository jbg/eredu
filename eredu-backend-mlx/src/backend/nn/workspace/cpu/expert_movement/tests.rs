use super::*;
use crate::backend::nn::expert_movement as movement;
use eredu_nn::Tensor;

fn mechanisms()->(MlxMetalWorkspaceMechanisms,MlxCpuWorkspaceMechanisms) {
    let native=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(native.allocation(),
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap());
    (native,cpu)
}
fn value(context:&WorkspaceContext,shape:&[i32],dtype:WorkspaceDtype,
    physical:Option<WorkspaceFloatingType>)->WorkspaceTensor {
    WorkspaceTensor::existing(context.layout(shape,dtype).unwrap()
        .with_representation(physical.map(|dtype|WorkspaceRepresentation::new(dtype,true))),context).unwrap()
}

#[test]
fn cpu_expert_movement_shared_gathers_keep_empty_singleton_and_maximum_sources() {
    let (native,cpu)=mechanisms();
    for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Float16,WorkspaceFloatingType::Bfloat16] {
        for (source_rows,count) in [(0,0),(7,0),(7,1),(7,7)] {
            for route_values in [false,true] {
                let context=WorkspaceContext::new(cpu);
                let input=value(&context,&[source_rows,3],WorkspaceDtype::Float32,Some(dtype));
                let indices=value(&context,&[count],WorkspaceDtype::Int32,None);
                context.begin_span();
                let output=movement::gather(&movement::Workspace(&context),&input,&indices,route_values).unwrap();
                assert_eq!(output.shape(),[count,if route_values{1}else{3}]);
                assert_eq!(output.layout().representation().unwrap().dtype(),dtype);
                let report=context.finish_report(&[output]).unwrap();
                for operation in &report.operations {
                    assert!(cpu.plan(operation.as_view()).unwrap().is_some(),"{operation:?}");
                }
                let operation=report.operations.iter().find(|operation|
                    matches!(operation.kind,WorkspaceOperationKind::Gather{axis:0})).unwrap();
                let plan=cpu.plan(operation.as_view()).unwrap().unwrap();
                assert_eq!(plan.dtype,dtype);
                assert_eq!(plan.validations,usize::from(count!=0));
                assert_eq!(plan.seeds,if count==0{0}else{3});
                assert_eq!(plan.population.maximum_captures,4);
                assert_eq!(plan.population.construction_entries,if count==0{3}else{32});
                assert_eq!(plan.population.primitives,match count{0=>2,1=>12,_=>13});
                if count==0 {
                    assert_eq!(plan.population.births,0);
                    assert_eq!(plan.output_bytes,0);
                    assert_eq!(plan.scratch_bytes,0);
                    assert_eq!(plan.parameter_shells,1);
                } else {
                    assert!(plan.output_bytes>0);
                    assert!(plan.scratch_bytes>0);
                }
                let recipe=if count==0 {
                    SpeculativeNumericalRecipe::inspect_cpu_outputs(&report,1,native,cpu,&context)
                } else {
                    SpeculativeNumericalRecipe::inspect_owned_child(&report,1,
                        ResidentExecutionMechanisms::Cpu{ordinary:native,cpu},&context)
                }.unwrap();
                assert_eq!(recipe.completion.validation_roots,usize::from(count!=0));
                assert_eq!(recipe.completion.traversal.limits().roots,1+usize::from(count!=0));
                assert_eq!(recipe.kernels,0);
                let mut population=CpuPopulation::default();
                for operation in &report.operations {
                    population.add(cpu.plan(operation.as_view()).unwrap().unwrap().population).unwrap();
                }
                assert_eq!(recipe.completion.graph.primitives(),population.construction_entries);
                assert_eq!(recipe.completion.traversal.limits().tape_entries,population.primitives+1);
            }
        }
    }
}

#[test]
fn cpu_expert_movement_row_add_separates_frontend_bank_from_eval_tape() {
    let (native,cpu)=mechanisms();
    for width in [1,3] {
        for (rows,count) in [(0,0),(7,0),(7,1),(7,7)] {
            for index_dtype in [WorkspaceDtype::Int32,WorkspaceDtype::Uint32] {
                let context=WorkspaceContext::new(cpu);
                let base=value(&context,&[rows,width],WorkspaceDtype::Float32,Some(WorkspaceFloatingType::Float32));
                let updates=value(&context,&[count,width],WorkspaceDtype::Float32,Some(WorkspaceFloatingType::Float32));
                let indices=value(&context,&[count,1],index_dtype,None);
                context.begin_span();
                let output=movement::add(&movement::Workspace(&context),&base,&indices,&updates).unwrap();
                assert_eq!(output.shape(),base.shape());
                let report=context.finish_report(&[output]).unwrap();
                assert_eq!(report.operations.len(),1);
                let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.population.construction_entries,if rows==0{0}else{7});
                assert_eq!(plan.population.primitives,if rows==0{0}else{1+usize::from(width!=1)});
                assert_eq!(plan.population.births,usize::from(rows!=0));
                assert_eq!(plan.population.maximum_captures,if rows==0{0}else{6});
                assert_eq!(plan.alias_input,(rows==0).then_some(0));
                assert_eq!(plan.seeds,0);assert_eq!(plan.validations,0);
                let recipe=SpeculativeNumericalRecipe::inspect_owned_child(&report,1,
                    ResidentExecutionMechanisms::Cpu{ordinary:native,cpu},&context).unwrap();
                assert_eq!(recipe.kernels,0);
                assert_eq!(recipe.completion.graph.primitives(),plan.population.construction_entries);
                assert_eq!(recipe.completion.traversal.limits().tape_entries,plan.population.primitives+1);
                // Addressable branch maxima and sequential composition must
                // preserve the two distinct populations through finalization.
                let source=super::super::super::resident_recipe::AddressableNumericalPopulation::from_recipe(recipe,false).unwrap();
                for (joined,repetitions) in [(source.union(source).unwrap(),1),(source.append(source).unwrap(),2)] {
                    let joined=joined.finish(1,0,&context).unwrap();
                    assert_eq!(joined.completion.graph.primitives(),repetitions*plan.population.construction_entries);
                    assert_eq!(joined.completion.traversal.limits().tape_entries,repetitions*plan.population.primitives+1);
                }
            }
        }
    }
}

#[test]
fn cpu_expert_movement_refuses_unknown_precision_and_malformed_sources() {
    let (_,cpu)=mechanisms();
    let source=WorkspaceLayoutView::new(&[7,3],WorkspaceDtype::Float32).unwrap();
    let ids=WorkspaceLayoutView::new(&[1],WorkspaceDtype::Int32).unwrap();
    let unsigned=WorkspaceLayoutView::new(&[1],WorkspaceDtype::Uint32).unwrap();
    let output=WorkspaceLayoutView::new(&[1,3],WorkspaceDtype::Float32).unwrap();
    for (input,indices) in [(source,ids),(source.with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),unsigned)] {
        let inputs=[input,indices];let outputs=[output];
        let operation=WorkspaceOperationView{kind:WorkspaceOperationKindView::Gather{axis:0},
            inputs:WorkspaceLayoutList::Views(&inputs),outputs:WorkspaceLayoutList::Views(&outputs)};
        assert!(cpu.plan(operation).unwrap().is_none());
    }
    let fp=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true));
    let base=source.with_representation(fp);
    let indices=WorkspaceLayoutView::new(&[1,2],WorkspaceDtype::Int32).unwrap();
    let update=output.with_representation(fp);
    let inputs=[base,indices,update];let outputs=[base];
    let operation=WorkspaceOperationView{kind:WorkspaceOperationKindView::IndexedRowAdd,
        inputs:WorkspaceLayoutList::Views(&inputs),outputs:WorkspaceLayoutList::Views(&outputs)};
    assert!(cpu.plan(operation).is_err());
}

