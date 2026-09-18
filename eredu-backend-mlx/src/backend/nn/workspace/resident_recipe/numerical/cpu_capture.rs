//! Actual CPU raw-capture trace and its borrowed native execution source.
use super::*;
use std::mem::{size_of,size_of_val};
use crate::backend::array_copy::CaptureTensorNativeError;
use safemlx::{OriginalScopeObserver,Stream};

/// Borrow of the exact full numerical recipe already installed by its caller.
/// This lends no new Scope, Record, Graph, Buffer, source or nested occurrence.
pub(crate) struct CpuCaptureLoan<'a> {
    recipe:&'a SpeculativeNumericalRecipe,
    stream:&'a Stream,
    observer:&'a OriginalScopeObserver,
}
impl CpuCaptureLoan<'_> {
    pub(crate) fn validate(&self)->Result<(),CaptureTensorNativeError>{
        let completion=&self.recipe.completion;
        if self.stream.device_type()?!=safemlx::DeviceType::Cpu
            || completion.validation_roots!=0 || self.recipe.kernels!=0
            || completion.dispatch.is_none_or(|dispatch|dispatch.cpu_model.is_none()
                ||dispatch.gpu_entries!=0||dispatch.parallel_entries!=0||dispatch.kernel_attempts!=0) {
            return Err(CaptureTensorNativeError::ClaimMismatch);
        }
        let current=OriginalScopeObserver::require_current()?;
        if !self.observer.same_scope(&current){return Err(CaptureTensorNativeError::ClaimMismatch);}
        Ok(())
    }
    pub(crate) fn stream(&self)->&Stream{self.stream}
    pub(crate) fn observer(&self)->&OriginalScopeObserver{self.observer}
    pub(crate) fn control_bytes()->Option<usize>{
        let parts=[size_of::<Self>(),size_of::<Option<Self>>(),size_of::<&Self>(),
            size_of::<Result<Option<Self>,CaptureTensorNativeError>>(),size_of::<Result<(),CaptureTensorNativeError>>(),
            size_of::<(&SpeculativeNumericalRecipe,&Stream,&OriginalScopeObserver)>(),
            size_of::<Option<ResidentDispatchPopulation>>(),size_of::<OriginalScopeObserver>(),
            OriginalScopeObserver::control_bytes()?,Stream::device_type_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
impl SpeculativeNumericalRecipe {
    pub(crate) fn cpu_capture_loan<'a>(&'a self,stream:&'a Stream,observer:&'a OriginalScopeObserver)
        ->Result<Option<CpuCaptureLoan<'a>>,CaptureTensorNativeError>{
        if stream.device_type()?==safemlx::DeviceType::Gpu{return Ok(None);}
        let loan=CpuCaptureLoan{recipe:self,stream,observer};
        loan.validate()?;
        Ok(Some(loan))
    }

    /// Ordinary composed CPU equations have one final output and no capture
    /// reads. Reuse the same reducer without adding capture roots or events.
    pub(crate) fn inspect_cpu_equations(report:&WorkspaceTraceReport,
        ordinary:MlxMetalWorkspaceMechanisms,cpu:MlxCpuWorkspaceMechanisms,context:&WorkspaceContext)
        ->Result<Self,Error>{
        Self::inspect_cpu_report(report,1,0,0,ordinary,cpu,context)
    }

    /// Every transform and mask is reduced by the same CPU model worker.
    /// Closing aliases extend the actual final frontier; nested reads retain
    /// their separately counted source occurrences in that same host bank.
    pub(crate) fn inspect_cpu_capture(report:&WorkspaceTraceReport,roots:usize,nested:usize,
        ordinary:MlxMetalWorkspaceMechanisms,cpu:MlxCpuWorkspaceMechanisms,context:&WorkspaceContext)
        ->Result<Self,Error>{
        if roots==0{return Err(context.metadata_error(format_args!("CPU numerical capture source is incomplete")));}
        Self::inspect_cpu_report(report,1,roots,nested,ordinary,cpu,context)
    }

    /// Selected deterministic child stages may expose several actual roots
    /// (for example a decoded frame header and payload). They add no capture.
    pub(crate) fn inspect_cpu_outputs(report: &WorkspaceTraceReport, outputs: usize,
        ordinary: MlxMetalWorkspaceMechanisms, cpu: MlxCpuWorkspaceMechanisms,
        context: &WorkspaceContext) -> Result<Self, Error> {
        if outputs == 0 {
            return Err(context.metadata_error(format_args!("CPU numerical stage has no output roots")));
        }
        Self::inspect_cpu_report(report, outputs, 0, 0, ordinary, cpu, context)
    }

    fn inspect_cpu_report(report:&WorkspaceTraceReport,outputs:usize,roots:usize,nested:usize,
        ordinary:MlxMetalWorkspaceMechanisms,cpu:MlxCpuWorkspaceMechanisms,context:&WorkspaceContext)
        ->Result<Self,Error>{
        let invalid=||context.metadata_error(format_args!("CPU numerical equation source is incomplete"));
        let recorder=ResidentRecipeRecorder::with_cpu_context(InferenceGeometry{
            batch_size:1,cached_positions:0,input_positions:1,max_output_tokens:0,
            prefill_chunk_positions:1,output:eredu_core::OutputDemand::Sequence,
        },ordinary,cpu,context)?;
        let reduced=recorder.reduce_trace(report,None,0,outputs)?;
        if reduced.first_missing_operation.is_some()||reduced.unqualified_kernel_owner.is_some()
            ||reduced.validation_roots!=0{
            return Err(context.metadata_error(format_args!(
                "CPU numerical equation source is incomplete: operation {:?}; producer {:?}; kernel {:?}; validation roots {}; nested completions {}",
                reduced.first_missing_operation,reduced.missing_operation_detail,
                reduced.unqualified_kernel_owner,reduced.validation_roots,reduced.nested_completions)));
        }
        let dispatch=reduced.dispatch.ok_or_else(invalid)?;
        if dispatch.cpu_model.is_none()||dispatch.gpu_entries!=0||dispatch.parallel_entries!=0
            ||dispatch.kernel_attempts!=0{return Err(invalid());}
        let graph_source=reduced.graph.ok_or_else(invalid)?;
        let (traversal,dispatch,frontier_controls)=super::super::capture::extend_frontier(
            reduced.traversal.ok_or_else(invalid)?,graph_source,dispatch,roots).ok_or_else(invalid)?;
        // Internal equation completions (including each paged accumulator
        // pass) remain real occurrences beside the caller's capture reads.
        let nested = reduced.nested_completions.checked_add(nested).ok_or_else(invalid)?;
        let completion=ResidentCompletionRecipe{validation_roots:0,grouped_outputs:reduced.grouped_outputs,
            traversal,graph:graph_source,dispatch:Some(dispatch),nested_completions:nested,
            nested_root_capacity:reduced.nested_root_capacity.max(1)};
        let graph=graph_capacity::ResidentGraphStorage::for_completion(completion).ok_or_else(invalid)?;
        let record=record_capacity::ResidentRecordStorage::for_completion(completion).ok_or_else(invalid)?;
        let frames=[size_of::<(&WorkspaceTraceReport,usize,usize,usize,MlxMetalWorkspaceMechanisms,MlxCpuWorkspaceMechanisms,&WorkspaceContext)>(),
            size_of::<InferenceGeometry>(),size_of::<ResidentRecipeRecorder>(),size_of::<ReducedTrace>(),
            size_of::<ResidentDispatchPopulation>()*2,size_of::<safemlx::ResidentGraphLayout>(),
            size_of::<(safemlx::OperationEvalTraversalLayout,ResidentDispatchPopulation,usize)>(),
            size_of::<ResidentCompletionRecipe>(),size_of::<graph_capacity::ResidentGraphStorage>(),
            size_of::<record_capacity::ResidentRecordStorage>(),size_of::<Result<Self,Error>>(),size_of::<Self>(),
            size_of::<Option<usize>>(),size_of::<usize>(),CpuCaptureLoan::control_bytes().ok_or_else(invalid)?,
            // The public constructor's argument/result transport remains live
            // while this shared reducer computes the actual source.
            size_of::<(&WorkspaceTraceReport,usize,usize,usize,MlxMetalWorkspaceMechanisms,MlxCpuWorkspaceMechanisms,&WorkspaceContext)>(),
            size_of::<Result<Self,Error>>()];
        let controls=frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .and_then(|n|n.checked_add(reduced.query_controls?))
            .and_then(|n|n.checked_add(frontier_controls)).ok_or_else(invalid)?;
        let controls=u64::try_from(controls).map_err(|_|invalid())?
            .checked_add(graph.control_bytes().ok_or_else(invalid)?).ok_or_else(invalid)?;
        Ok(Self{completion,storage:reduced.mutable_storage.ok_or_else(invalid)?,
            graph_capacity:usize::try_from(graph.full_capacity.ok_or_else(invalid)?).map_err(|_|invalid())?,
            record_capacity:usize::try_from(record.full_capacity.ok_or_else(invalid)?).map_err(|_|invalid())?,
            kernels:0,controls})
    }
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    #[test]
    fn cpu_raw_capture_joins_mask_sources_and_each_actual_completion_frontier(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation,choice);
        let context=WorkspaceContext::new(cpu);
        let storage=WorkspaceExistingStorage::try_new(Some(4096),&context).unwrap();
        let input=WorkspaceTensor::existing_with_storage(context.layout(&[1,8],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&storage,&context).unwrap();
        context.begin_state_span([&input]).unwrap();
        let row=input.reshape(&[1,1,8],&context).unwrap();
        let full=context.execute(WorkspaceOperationKind::Elementwise("capture_cast_f32"),&[&row],
            vec![row.layout().clone()]).unwrap().remove(0);
        let slice=row.static_slice(&[0,0,2],&[1,1,6],&[1,1,1],&context).unwrap();
        let slice=context.execute(WorkspaceOperationKind::Elementwise("capture_cast_f32"),&[&slice],
            vec![slice.layout().clone()]).unwrap().remove(0);
        let output=context.execute(WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::TokenFilter),
            &[&input],vec![input.layout().clone()]).unwrap().remove(0);
        // These are the same retained source/selection aliases and successful
        // nested reads as the two raw workers; there is one final policy root.
        let retained=[row,full,slice];
        let report=context.finish_report(&[output]).unwrap();
        assert!(report.unpriced_operations.is_empty() && report.unpriced_host_operations.is_empty());
        let ordinary_recipe=SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
        assert_eq!(ordinary_recipe.completion.traversal.limits().roots,1);
        assert_eq!(ordinary_recipe.completion.nested_completions,0);
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_capture(&report,11,2,ordinary,cpu,&context).unwrap();
        assert_eq!(recipe.kernels,0);
        assert_eq!(recipe.completion.nested_completions,2);
        assert_eq!(recipe.completion.traversal.limits().roots,12);
        assert_eq!(recipe.completion.dispatch.unwrap().gpu_entries,0);
        assert!(recipe.completion.dispatch.unwrap().cpu_model.is_some());
        assert!(recipe.graph_capacity>0&&recipe.record_capacity>0&&recipe.controls>0);
        assert_eq!(report.host_workspace_bytes,Some(16));
        // Only the new masked output closes this report; capture source aliases
        // are held separately and extend the native completion frontier above.
        let state=report.state.as_ref().unwrap();
        assert_eq!(state.retained_bytes,report.tensor_buffers.retained_bytes);
        assert!(state.retained_bytes.is_some_and(|bytes|bytes>0));
        assert_eq!(state.displaced_bytes,Some(4096));
        drop(retained);
        let context=WorkspaceContext::new(cpu);
        let row=WorkspaceTensor::unloaded_f32(&[1,1,8],&context).unwrap();
        context.begin_span();
        let stepped=row.static_slice(&[0,0,0],&[1,1,8],&[1,1,2],&context).unwrap();
        let report=context.finish_report(&[stepped]).unwrap();
        assert!(SpeculativeNumericalRecipe::inspect_cpu_capture(&report,6,1,ordinary,cpu,&context).is_err());
    }
    #[test]
    fn cpu_preview_retains_flatten_and_prefix_sources(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),choice);
        let context=WorkspaceContext::new(cpu);
        let storage=WorkspaceExistingStorage::try_new(Some(4096),&context).unwrap();
        let input=WorkspaceTensor::existing_with_storage(context.layout(&[1,1,8],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&storage,&context).unwrap();
        context.begin_state_span([&input]).unwrap();
        let rectangle=input.static_slice(&[0,0,1],&[1,1,7],&[1,1,1],&context).unwrap();
        let flat=rectangle.reshape(&[6],&context).unwrap();
        let prefix=flat.static_slice(&[0],&[3],&[1],&context).unwrap();
        let output=context.execute(WorkspaceOperationKind::Elementwise("capture_cast_f32"),&[&prefix],
            vec![prefix.layout().clone()]).unwrap().remove(0);
        let report=context.finish_report(&[output]).unwrap();
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_capture(&report,6,1,ordinary,cpu,&context).unwrap();
        assert_eq!(recipe.kernels,0);assert_eq!(recipe.completion.nested_completions,1);
        assert_eq!(recipe.completion.dispatch.unwrap().cpu_model.unwrap().primitives,3);
        assert_eq!(report.tensor_buffers.total_bytes,Some(0));
        assert_eq!(report.state.as_ref().unwrap().retained_bytes,Some(4096));
        assert_eq!(prefix.shape(),[3]);assert!(prefix.layout().representation().unwrap().row_contiguous());
        let WorkspaceOperationKindView::StaticSlice{starts,ends,strides}=report.operations[2].as_view().kind
            else{panic!("preview lost exact prefix coordinates")};
        assert_eq!((starts,ends,strides),([0].as_slice(),[3].as_slice(),[1].as_slice()));
    }

}
