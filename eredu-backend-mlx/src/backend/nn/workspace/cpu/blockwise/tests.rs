use super::*;
use eredu_nn::{AttentionArithmetic,BlockwiseAttentionBackend,BlockwiseAttentionOptions,BlockwiseAttentionSpec,Tensor};
fn existing(shape:&[i32],dtype:WorkspaceDtype,context:&WorkspaceContext)->WorkspaceTensor {
    WorkspaceTensor::existing(context.layout(shape,dtype).unwrap().with_representation(
        (dtype==WorkspaceDtype::Float32).then_some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),context).unwrap()
}
#[test]
fn cpu_paged_stages_preserve_exact_rows_outputs_and_nested_completions() {
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    for arithmetic in [AttentionArithmetic::Fused,AttentionArithmetic::InputScores] {
      for mask_type in [None,Some(WorkspaceDtype::Bool),Some(WorkspaceDtype::Float32)] {
       for heads in [2,4] {
        for queries in [1,3] {
        let end=queries+4;
        let mut dense_births=Vec::new();
        for strided in [false,true] {
        let context=WorkspaceContext::new(cpu);
        let q=if strided {existing(&[1,queries,heads,4],WorkspaceDtype::Float32,&context)
            .transpose_axes(&[0,2,1,3],&context).unwrap()}else{existing(&[1,heads,queries,4],WorkspaceDtype::Float32,&context)};
        let mask=mask_type.map(|dtype|existing(&[queries,end],dtype,&context));
        let sinks=existing(&[heads],WorkspaceDtype::Float32,&context);
        let spec=BlockwiseAttentionSpec{queries:&q,scale:0.5,mask:mask.as_ref(),query_start:4,
            context_end:i64::from(end),sliding_window:Some(4),prefix_tokens:2,sinks:Some(&sinks)};
        let options=BlockwiseAttentionOptions{arithmetic,softcap:Some(1.75)};
        context.begin_state_span(std::iter::empty::<&WorkspaceTensor>()).unwrap();
        let mut accumulator=WorkspaceBackend::begin_blockwise_attention_with_options(spec,options,&context).unwrap();
        for pass in 0..options.passes() {
            if pass==1 {WorkspaceBackend::begin_blockwise_value_pass(&mut accumulator,&context).unwrap();}
            // Includes singleton aliases, genuine SIMD row maxima, uneven
            // pages, accumulated recurrence and the second probability pass.
            for (start,end) in [(0,1),(2,end-2),(end-2,end)] {
                let k=existing(&[1,2,end-start,4],WorkspaceDtype::Float32,&context);
                let v=existing(&[1,2,end-start,3],WorkspaceDtype::Float32,&context);
                let bias=existing(&[],WorkspaceDtype::Float32,&context);
                WorkspaceBackend::accumulate_blockwise_attention_with_bias(&mut accumulator,
                    i64::from(start),i64::from(end),k,v,Some(&bias),&context).unwrap();
            }
        }
        let result=WorkspaceBackend::finish_blockwise_attention(accumulator,&context).unwrap();
        assert_eq!(result.layout().representation(),Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
        let report=context.finish_report(&[result]).unwrap();
        assert!(report.unpriced_operations.is_empty());assert!(report.unpriced_host_operations.is_empty());
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
        assert_eq!(recipe.completion.nested_completions,3*options.passes());
        let mut accumulation=0;
        for operation in &report.operations {
            if !matches!(operation.kind,WorkspaceOperationKind::BlockwiseAttention{..}) {continue;}
            let plan=cpu.plan(operation.as_view()).unwrap().unwrap();
            let Descriptor{stage,..}=descriptor::decode(operation.as_view()).unwrap().unwrap();
            match stage {
                Stage::Begin{..}=>assert_eq!(plan.output_bytes,0),
                Stage::Accumulate{value_pass,..}=>{
                    assert_eq!(operation.outputs.len(),if value_pass{1}else{3});
                    assert_eq!(plan.rank,if heads==2 {4}else{5});
                    if strided {assert_eq!(plan.population.births,dense_births[accumulation]
                        +usize::from(arithmetic==AttentionArithmetic::InputScores&&queries>1));}
                    else {dense_births.push(plan.population.births);}
                    accumulation+=1;
                    if arithmetic==AttentionArithmetic::InputScores {
                        let mut views=operation.as_view().inputs.iter().collect::<Vec<_>>();
                        views[0]=views[0].with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,false).with_last_axis_contiguous(true)));
                        assert!(cpu.plan(WorkspaceOperationView{inputs:WorkspaceLayoutList::Views(&views),
                            ..operation.as_view()}).unwrap().is_none());
                    }
                }
                Stage::Finish{input_scores,..}=>assert_eq!(plan.alias_input.is_some(),input_scores),
            }
        }
        }
        }
       }
      }
    }
}

#[test]
fn cpu_paged_attention_output_preserves_projection_scalar_source() {
    use eredu_nn::{NeuralBackend,LinearOperator,LinearSpec,LinearFormatSpec,ParameterSpec};
    let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
    for positions in [1,2,3] {
        let context=WorkspaceContext::new(cpu);
        let weight=ParameterSpec::trainable("attention.output.weight").unwrap();
        context.install_parameter_representations(vec![WorkspaceParameterRepresentation::new(
            weight.id.clone(),context.layout(&[8,8],WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))))]).unwrap();
        let mut projection=WorkspaceBackend::linear(LinearSpec{input:8,output:8,weight,bias:None,
            format:LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap()},&context).unwrap();
        let queries=existing(&[1,2,positions,4],WorkspaceDtype::Float32,&context);
        let keys=existing(&[1,1,positions,4],WorkspaceDtype::Float32,&context);
        let values=existing(&[1,1,positions,4],WorkspaceDtype::Float32,&context);
        context.begin_span();
        let mut state=WorkspaceBackend::begin_blockwise_attention_with_options(
            BlockwiseAttentionSpec{queries:&queries,scale:0.5,mask:None,query_start:0,
                context_end:i64::from(positions),sliding_window:None,prefix_tokens:0,sinks:None},
            BlockwiseAttentionOptions{arithmetic:AttentionArithmetic::Fused,softcap:None},&context).unwrap();
        WorkspaceBackend::accumulate_blockwise_attention_with_bias(&mut state,0,
            i64::from(positions),keys,values,None,&context).unwrap();
        let attended=WorkspaceBackend::finish_blockwise_attention(state,&context).unwrap();
        let merged=attended.transpose_axes(&[0,2,1,3],&context).unwrap()
            .reshape(&[1,positions,8],&context).unwrap();
        let output=projection.forward(&merged,&context).unwrap();
        let report=context.finish_report(&[output.clone()]).unwrap();
        assert!(matches!(report.operations.last().unwrap().kind,WorkspaceOperationKind::Projection(_)));
        for operation in &report.operations {
            assert!(cpu.plan(operation.as_view()).unwrap().is_some(),"{operation:?}");
        }
        assert_eq!(output.layout().representation(),Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,true)));
        assert!(report.unpriced_operations.is_empty());
        SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
    }
}
