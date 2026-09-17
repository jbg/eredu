//! Completed rows bind one source-derived local grouped child. The ordinary
//! architecture/provider callback remains the only numerical execution driver.
use super::*;
use crate::{MlxTensor, backend::{nn::{tensor::PreparedTokenChild, workspace::SpeculativeNumericalRecipe},
    submission_recovery::prefill::nested::NestedRootCompletion}};
use eredu_nn::{Parameterized, workspace::WorkspaceExpertRegionView};
use eredu_runtime::RoutedExpertTensorParallelOutput;
use safemlx::{Array, PreparedArrayClone, PreparedStreamCopy, StreamCopyPlan};
use super::super::super::parallel::{OriginalParallelBinding, RetainedExpertRegion};

struct LocalChild {
    inputs: Vec<Array>,
    completion: RefCell<NestedRootCompletion>,
    collector: PreparedTokenChild,
    quote: RetainedExpertRegion,
    recipe: SpeculativeNumericalRecipe,
    stream: PreparedStreamCopy<Custody>,
    completed: eredu_core::ErasedSharedStorageOwner,
    owner: OriginalParallelControlOwner,
    custody: Custody,
}
impl OriginalParallelControlProjection {
    pub(super) fn expert_binding<'a>(&'a self, context: &'a Group) -> Option<&'a OriginalParallelBinding> {
        context.original_parallel_binding().or_else(|| self.model_roots.as_ref().and_then(|model| model.active_binding()))
    }
    pub(crate) fn with_expert_inactive_wave<E,F>(&self,declaration:eredu_nn::workspace::WorkspaceExpertInactiveWave,
        context:&Group,stream:&Stream,run:F)->Result<Result<(),E>,Error>
    where F:FnOnce()->Result<(),E>{
        self.with_context(context,|bound|{
            let c=&self.custody;
            reserve(&c.funding,&[size_of::<F>(),size_of::<E>(),size_of::<Result<(),E>>(),
                size_of::<eredu_nn::workspace::WorkspaceExpertInactiveWave>(),
                failure_control_bytes().ok_or_else(overflow)?])?;
            let fail=||control_error(ControlCause::Identity,&c.source,&c.funding);
            if bound.is_none(){return Err(fail());}
            let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
            let source=owner.owner().request.source.model().ok_or_else(fail)?;
            let binding=self.expert_binding(context).ok_or_else(fail)?;
            if owner.owner().failed.get(){return Err(fail());}
            let result=binding.with_expert_inactive_wave(source,declaration,stream,||Ok(run()));
            if !matches!(&result,Ok(Ok(_))){owner.owner().failed.set(true);}
            result
        })?
    }
    pub(crate) fn with_expert_provider_wave<E,F>(&self,declaration:eredu_nn::workspace::WorkspaceExpertProviderWave,
        context:&Group,stream:&Stream,run:F)->Result<Result<(),E>,Error>
    where F:FnOnce()->Result<(),E>{
        self.with_context(context,|bound|{
            let c=&self.custody;
            reserve(&c.funding,&[size_of::<F>(),size_of::<E>(),size_of::<Result<(),E>>(),
                size_of::<eredu_nn::workspace::WorkspaceExpertProviderWave>(),
                failure_control_bytes().ok_or_else(overflow)?])?;
            let fail=||control_error(ControlCause::Identity,&c.source,&c.funding);
            if bound.is_none(){return Err(fail());}
            let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
            let source=owner.owner().request.source.model().ok_or_else(fail)?;
            let binding=self.expert_binding(context).ok_or_else(fail)?;
            if owner.owner().failed.get(){return Err(fail());}
            let result=binding.with_expert_provider_wave(source,declaration,stream,||Ok(run()));
            if !matches!(&result,Ok(Ok(_))){owner.owner().failed.set(true);}
            result
        })?
    }
    pub(crate) fn with_expert_region<P,E,F>(&self, declaration: WorkspaceExpertRegionView<'_>,
        bank: &mut P, context: &Group, stream: &Stream, run: F)
        -> Result<Result<RoutedExpertTensorParallelOutput<MlxTensor>,E>,Error>
    where F:FnOnce(&mut P,Option<eredu_runtime::PreparedExpertMovementLoan<'_>>)->Result<RoutedExpertTensorParallelOutput<MlxTensor>,E> {
        self.with_context(context, |bound| {
            let c=&self.custody;
            reserve(&c.funding,&[size_of::<F>(),size_of::<E>(),size_of::<Result<RoutedExpertTensorParallelOutput<MlxTensor>,E>>(),
                failure_control_bytes().ok_or_else(overflow)?])?;
            if bound.is_none(){return Err(control_error(ControlCause::Identity,&c.source,&c.funding));}
            let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
            let source=owner.owner().request.source.model().ok_or_else(||control_error(ControlCause::Identity,&c.source,&c.funding))?;
            let binding=self.expert_binding(context).ok_or_else(||control_error(ControlCause::Identity,&c.source,&c.funding))?;
            if owner.owner().failed.get(){return Err(control_error(ControlCause::Identity,&c.source,&c.funding));}
            let result=binding.with_expert_region(source,declaration,stream,|| {
                let movement=super::expert_movement::OriginalExpertMovementSource::prepare(
                    binding,source,declaration,stream,&owner,c)?;
                let result=run(bank,Some(movement.loan()));
                if result.is_ok(){movement.finish(stream)?;}
                Ok(result)
            });
            if !matches!(&result,Ok(Ok(_))){owner.owner().failed.set(true);}
            result
        })?
    }
    pub(crate) fn with_expert_local<P,E,F>(&self,declaration:WorkspaceExpertRegionView<'_>,bank:&mut P,
        input:&MlxTensor,scores:&MlxTensor,coefficients:&MlxTensor,completed:&eredu_core::ErasedSharedStorageOwner,
        local_rows:&[usize],context:&Group,stream:&Stream,run:F)
        ->Result<Result<RoutedExpertTensorParallelOutput<MlxTensor>,E>,Error>
    where P:Parameterized<MlxTensor>,F:FnOnce(&mut P)->Result<RoutedExpertTensorParallelOutput<MlxTensor>,E> {
        self.with_context(context,|bound| {
            let c=&self.custody;
            reserve(&c.funding,&[size_of::<F>(),size_of::<E>(),size_of::<LocalChild>(),size_of::<AgreementCapacity>(),
                size_of::<Result<LocalChild,Error>>(),size_of::<Result<RoutedExpertTensorParallelOutput<MlxTensor>,E>>(),
                size_of::<Result<Result<RoutedExpertTensorParallelOutput<MlxTensor>,E>,eredu_core::BackendFailure>>(),
                failure_control_bytes().ok_or_else(overflow)?])?;
            let fail=||control_error(ControlCause::Identity,&c.source,&c.funding);
            if bound.is_none(){return Err(fail());}
            let owner=self.upgrade().map_err(|cause|Error::with_original_control_source(cause,false))?;
            if owner.owner().failed.get() || owner.owner().running.replace(true) {
                owner.owner().failed.set(true); return Err(fail());
            }
            let _running=Running{running:&owner.owner().running,failed:&owner.owner().failed};
            let child=self.prepare_expert_local(declaration,bank,input,scores,coefficients,completed,local_rows,context,stream,&owner)?;
            let capacity=child.quote.value().capacity;
            let kernels=child.recipe.kernels;
            let result=run_native_role_with_pipeline(child,AgreementCapacity{graph:capacity.graph,records:capacity.records,backing:capacity.backing},
                Some(safemlx::PreparedPipelineCachePlan::new(kernels)),&owner.owner().bank,&owner.owner().controls,c,
                |child,observer| {
                    let collector=child.collector.enter(observer)?;
                    for input in &child.inputs {safemlx::OperationEvent::validate_traversal_leaf(input,observer)
                        .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;}
                    let mut graph=safemlx::OperationEvent::prepare_resident_graph(child.recipe.completion.graph,observer)
                        .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
                    let traversal=child.recipe.completion.nested_traversal().ok_or_else(fail)?;
                    graph.configure_nested_completions(&traversal,child.recipe.completion.nested_completions)
                        .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
                    let output=match run(bank) {Ok(value)=>value,Err(cause)=>{
                        drop(graph);
                        // Drop restores the parent and retains child roots even
                        // on failure. Preserve the original typed callback error;
                        // the shared native recovery still owns its live scope.
                        drop(collector);return Ok(Err(cause));}};
                    child.completion.try_borrow_mut().map_err(|_|fail())?.complete(|visit| {
                        match &output {RoutedExpertTensorParallelOutput::Complete(value)=>visit(value),
                            RoutedExpertTensorParallelOutput::Partial(value)=>{visit(value.reducible());if let Some(bias)=value.post_reduce(){visit(bias);}}}
                        Ok(())
                    },observer,child.stream.as_stream())?;
                    // Complete and detach under the child that owns these
                    // roots. The next transport may only borrow ready leaves.
                    let publish=|value:&MlxTensor|safemlx::OperationEvent::validate_traversal_leaf(value.as_array(),observer)
                        .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding));
                    match &output {
                        RoutedExpertTensorParallelOutput::Complete(value)=>publish(value)?,
                        RoutedExpertTensorParallelOutput::Partial(value)=>{
                            publish(value.reducible())?;if let Some(bias)=value.post_reduce(){publish(bias)?;}
                        }
                    }
                    drop(graph);
                    collector.finish()?;
                    Ok(Ok(output))
                }).map_err(|cause|Error::with_original_control_source(cause,false));
            if matches!(&result,Ok(Ok(_))) {
                let model=owner.owner().request.source.model().ok_or_else(fail)?;
                self.expert_binding(context).ok_or_else(fail)?.complete_expert_local(model,declaration,stream)?;
            } else {owner.owner().failed.set(true);}
            result
        })?
    }
    /// Prepare the same retained child before entering its native callback.
    /// Every source/shape/custody check stays here; no preparation temporaries
    /// remain live on the sparse observer's nested execution stack.
    #[inline(never)]
    fn prepare_expert_local<P:Parameterized<MlxTensor>>(&self,declaration:WorkspaceExpertRegionView<'_>,bank:&P,
        input:&MlxTensor,scores:&MlxTensor,coefficients:&MlxTensor,
        completed:&eredu_core::ErasedSharedStorageOwner,local_rows:&[usize],context:&Group,stream:&Stream,
        owner:&OriginalParallelControlOwner)->Result<LocalChild,Error>{
            let c=&self.custody;
            let fail=||control_error(ControlCause::Identity,&c.source,&c.funding);
            let model=owner.owner().request.source.model().ok_or_else(fail)?;
            let (quote,parent)=self.expert_binding(context).ok_or_else(fail)?.expert_local_source(model,declaration,stream)?;
            let count=completed.downcast_ref::<super::peer_counts::table::PeerCountTable>()
                .and_then(|table|table.region_received(declaration.group,declaration.rank,declaration.peers,owner.owner(),c)).ok_or_else(fail)?;
            let actual=local_rows.iter().copied().try_fold(0usize,usize::checked_add).ok_or_else(overflow)?;
            if actual!=count || local_rows.len()!=declaration.local_experts()
                || input.as_array().shape()!=[i32::try_from(count).map_err(|_|overflow())?,declaration.kernel.dimensions().0]
                || scores.as_array().shape()!=[i32::try_from(count).map_err(|_|overflow())?,1]
                || coefficients.as_array().shape()!=scores.as_array().shape(){return Err(fail());}
            let slots=bank.retained_value_slot_bound().and_then(|n|n.checked_add(3)).ok_or_else(fail)?;
            reserve(&c.funding,&[Layout::array::<Array>(slots).map_err(|_|overflow())?.size(),
                slots.checked_mul(PreparedArrayClone::control_bytes().and_then(|n|n.checked_add(Array::inspection_clone_handle_bytes())).ok_or_else(overflow)?).ok_or_else(overflow)?,
                size_of::<Option<Error>>(),size_of::<PreparedArrayClone>(),
                size_of::<Result<Array,safemlx::PreparedArrayCloneCause>>()])?;
            let mut inputs=Vec::new();inputs.try_reserve_exact(slots).map_err(|_|fail())?;
            let mut error=None;
            let mut retain=|value:&MlxTensor| {
                if error.is_some(){return;}
                if inputs.len()==slots{error=Some(fail());return;}
                let result=PreparedArrayClone::try_prepare_for_inspection()
                    .and_then(|mut slot|slot.fill_for_inspection(value.as_array()));
                match result {Ok(value)=>inputs.push(value),Err(_)=>error=Some(fail())}
            };
            retain(input);retain(scores);retain(coefficients);
            let complete=bank.visit_retained_values(&mut retain);drop(retain);
            if let Some(error)=error{return Err(error);}
            if !complete||inputs.len()!=quote.value().inputs.len()-1{return Err(fail());}
            for (array,layout) in inputs.iter().skip(3).zip(quote.value().inputs.iter().skip(4)) {
                if array.shape()!=layout.shape() || crate::backend::nn::workspace::byte_view::Dtype::from_layout(layout.as_view())
                    .is_none_or(|dtype|dtype.native()!=array.dtype()){return Err(fail());}
            }
            let recipe=quote.value().actual(count,&inputs).map_err(Error::Neural)?;
            reserve(&c.funding,&[usize::try_from(recipe.controls).map_err(|_|overflow())?])?;
            let roots=1+usize::from(declaration.kernel.separate_bias(declaration.tensor_partitions));
            reserve(&c.funding,&[
                recipe.completion.nested_traversal().and_then(|value|value.query_control_bytes()).ok_or_else(overflow)?,
                roots.checked_mul(safemlx::OperationEvent::traversal_leaf_control_bytes().ok_or_else(overflow)?).ok_or_else(overflow)?])?;
            let completion=NestedRootCompletion::prepare_metadata(roots,recipe.completion.validation_roots,c.raw.clone().into(),&c.funding)?;
            let collector=PreparedTokenChild::prepare(recipe.completion,&owner.owner().controls,&parent,&c.funding)?;
            let plan=StreamCopyPlan::<Custody>::capture(stream).map_err(|_|fail())?;
            reserve(&c.funding,&[plan.control_bytes().ok_or_else(overflow)?,plan.native_wrapper_bytes(),plan.owner_node_layout().size(),
                Layout::new::<[usize;2]>().extend(plan.shared_body_layout()).map_err(|_|overflow())?.0.pad_to_align().size()])?;
            let retained_stream=plan.realize(c.clone()).map_err(|_|fail())?;
            Ok(LocalChild{inputs,completion:RefCell::new(completion),collector,quote,recipe,stream:retained_stream,
                completed:completed.clone(),owner:OriginalParallelControlOwner(owner.0.clone()),custody:c.clone()})
    }

}
