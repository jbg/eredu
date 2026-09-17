//! Each row movement executes and settles in its admitted native child role.
use super::*;
use crate::{MlxTensor,backend::{nn::{expert_movement,tensor::PreparedTokenChild,
    workspace::{ExpertMovementKind,SpeculativeNumericalRecipe}},submission_recovery::prefill::nested::NestedRootCompletion}};
use safemlx::{PreparedArrayClone,PreparedStreamCopy,StreamCopyPlan};
struct Movement {
    inputs:Vec<MlxTensor>,
    completion:RefCell<NestedRootCompletion>,
    collector:PreparedTokenChild,
    source:OriginalExpertMovementSource,
    stream:PreparedStreamCopy<Custody>,
    kind:ExpertMovementKind,
    recipe:SpeculativeNumericalRecipe,
    custody:Custody,
}
impl OriginalExpertMovementSource {
    pub(crate) fn gather(&self,value:&MlxTensor,rows:&[usize],route_values:bool,stream:&Stream)->Result<MlxTensor,Error> {
        let kind=ExpertMovementKind::Gather{route_values};
        self.attempt(kind,stream,|| {
            let indices=MlxTensor::from_array(self.indices(rows,false,stream)?);
            self.run(kind,&[value,&indices],stream)
        })
    }
    pub(crate) fn zeros(&self,value:&MlxTensor,rows:i32,stream:&Stream)->Result<MlxTensor,Error> {
        let kind=ExpertMovementKind::Zero{rows};
        self.attempt(kind,stream,||self.run(kind,&[value],stream))
    }
    pub(crate) fn add(&self,base:&MlxTensor,rows:&[usize],updates:&MlxTensor,stream:&Stream)->Result<MlxTensor,Error> {
        self.attempt(ExpertMovementKind::Add,stream,|| {
            let indices=MlxTensor::from_array(self.indices(rows,true,stream)?);
            self.run(ExpertMovementKind::Add,&[base,&indices,updates],stream)
        })
    }
    fn attempt<T,F>(&self,kind:ExpertMovementKind,stream:&Stream,run:F)->Result<T,Error>
    where F:FnOnce()->Result<T,Error> {
        self.validate(stream)?;
        let source=self.source();let c=&source.custody;
        struct Attempt<'a>{source:&'a Source,complete:bool}
        impl Drop for Attempt<'_>{fn drop(&mut self){if !self.complete{self.source.owner.owner().failed.set(true);}}}
        reserve(&c.funding,&[size_of::<(Attempt<'_>,T,F,Result<T,Error>,[usize;4],usize)>()])?;
        let mut attempt=Attempt{source,complete:false};
        let index=match kind {ExpertMovementKind::Gather{route_values:false}=>0,
            ExpertMovementKind::Gather{route_values:true}=>1,ExpertMovementKind::Zero{..}=>2,ExpertMovementKind::Add=>3};
        let mut attempted=source.attempted.get();
        if attempted!=source.completed.get()||attempted[index]>=source.quote.value().declaration.as_view().movement.counts()[index] {
            return Err(control_error(ControlCause::Identity,&c.source,&c.funding));
        }
        attempted[index]=attempted[index].checked_add(1).ok_or_else(overflow)?;
        source.attempted.set(attempted);
        let value=run()?;
        self.validate(stream)?;
        source.completed.set(attempted);
        attempt.complete=true;
        Ok(value)
    }
    fn run(&self,kind:ExpertMovementKind,values:&[&MlxTensor],stream:&Stream)->Result<MlxTensor,Error> {
        self.validate(stream)?;
        let source=self.source();let c=&source.custody;let owner=source.owner.owner();
        let fail=||control_error(ControlCause::Identity,&c.source,&c.funding);
        if !(1..=3).contains(&values.len()){return Err(fail());}
        reserve(&c.funding,&[size_of::<Movement>(),size_of::<Result<Movement,Error>>(),
            size_of::<ExpertMovementKind>(),size_of::<OriginalExpertMovementSource>(),
            size_of::<[&MlxTensor;3]>(),size_of::<[&Array;1]>(),
            size_of::<Result<MlxTensor,Error>>(),
            size_of::<Result<Result<MlxTensor,Error>,eredu_core::BackendFailure>>(),
            Layout::array::<MlxTensor>(values.len()).map_err(|_|overflow())?.size(),
            values.len().checked_mul(PreparedArrayClone::control_bytes().and_then(|n|n.checked_add(Array::inspection_clone_handle_bytes())).ok_or_else(overflow)?).ok_or_else(overflow)?,
            values.len().checked_mul(crate::backend::runtime::cache::value_completion_control_bytes(1).ok_or_else(overflow)?).ok_or_else(overflow)?,
            // Each input is published under its parent and later validated as
            // a settled leaf under the independently admitted child.
            values.len().checked_mul(2).and_then(|n|n.checked_mul(
                safemlx::OperationEvent::traversal_leaf_control_bytes()?)).ok_or_else(overflow)?,
            expert_movement::control_bytes::<expert_movement::Native<'_>>().ok_or_else(overflow)?])?;
        let mut inputs=Vec::new();inputs.try_reserve_exact(values.len()).map_err(|_|fail())?;
        // A reshape created in the parent may still contain model collectives.
        // Complete each local dependency through the existing parent worker
        // before borrowing it into a distinct native child arena.
        for value in values {
            crate::backend::runtime::cache::complete_values([value.as_array()],stream)
                .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
            // The completed nested Event can remain attached to this exact
            // root. Publish its settled descriptor before the pure backing
            // projection: the existing parent validator detaches only a
            // matching completed event, without another Eval or allocation.
            safemlx::OperationEvent::validate_traversal_leaf(value.as_array(),&source.parent)
                .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
            let mut slot=PreparedArrayClone::try_prepare_for_inspection().map_err(|_|fail())?;
            inputs.push(MlxTensor::from_array(slot.fill_for_inspection(value.as_array()).map_err(|_|fail())?));
        }
        let (recipe,capacity)=source.quote.value().movement.as_ref().ok_or_else(fail)?.actual(kind,&inputs).map_err(Error::Neural)?;
        reserve(&c.funding,&[usize::try_from(recipe.controls).map_err(|_|overflow())?,
            // Configure the already quoted final child completion and publish
            // its one completed output before returning to the parent scope.
            recipe.completion.nested_traversal().and_then(|value|value.query_control_bytes()).ok_or_else(overflow)?,
            safemlx::OperationEvent::traversal_leaf_control_bytes().ok_or_else(overflow)?])?;
        let completion=NestedRootCompletion::prepare_metadata(1,recipe.completion.validation_roots,c.raw.clone().into(),&c.funding)?;
        let collector=PreparedTokenChild::prepare(recipe.completion,&owner.controls,&source.parent,&c.funding)?;
        let plan=StreamCopyPlan::<Custody>::capture(stream).map_err(|_|fail())?;
        reserve(&c.funding,&[plan.control_bytes().ok_or_else(overflow)?,plan.native_wrapper_bytes(),plan.owner_node_layout().size(),
            Layout::new::<[usize;2]>().extend(plan.shared_body_layout()).map_err(|_|overflow())?.0.pad_to_align().size()])?;
        let selected=plan.realize(c.clone()).map_err(|_|fail())?;
        if owner.failed.get()||owner.running.replace(true){owner.failed.set(true);return Err(fail());}
        let _running=Running{running:&owner.running,failed:&owner.failed};
        let child=Movement{inputs,completion:RefCell::new(completion),collector,source:self.clone(),
            stream:selected,kind,recipe,custody:c.clone()};
        let result=run_native_role_with_pipeline(child,AgreementCapacity{graph:capacity.graph,records:capacity.records,backing:capacity.backing},
            Some(safemlx::PreparedPipelineCachePlan::new(recipe.kernels)),&owner.bank,&owner.controls,c,
            |child,observer|Ok(child.execute(observer)))
            .map_err(|cause|Error::with_original_control_source(cause,false))?;
        if result.is_err(){owner.failed.set(true);}
        result
    }
}
impl Movement {
    fn execute(&self,observer:&OriginalScopeObserver)->Result<MlxTensor,Error> {
        let c=&self.custody;
        let collector=self.collector.enter(observer)?;
        for value in &self.inputs {safemlx::OperationEvent::validate_traversal_leaf(value.as_array(),observer)
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;}
        let mut graph=safemlx::OperationEvent::prepare_resident_graph(self.recipe.completion.graph,observer)
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
        // The final completion is one of this exact child's prepaid nested
        // attempts. Retain its bank while the native worker suspends/restores
        // it; dropping here would expose the suspended parent graph instead.
        let traversal=self.recipe.completion.nested_traversal()
            .ok_or_else(||control_error(ControlCause::Identity,&c.source,&c.funding))?;
        graph.configure_nested_completions(&traversal,self.recipe.completion.nested_completions)
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
        let ops=expert_movement::Native(self.stream.as_stream());
        let output=match self.kind {
            ExpertMovementKind::Gather{route_values}=>expert_movement::gather(&ops,&self.inputs[0],&self.inputs[1],route_values),
            ExpertMovementKind::Zero{rows}=>expert_movement::zeros(&ops,&self.inputs[0],rows),
            ExpertMovementKind::Add=>expert_movement::add(&ops,&self.inputs[0],&self.inputs[1],&self.inputs[2]),
        }.map_err(Error::Neural)?;
        self.completion.try_borrow_mut().map_err(|_|control_error(ControlCause::Identity,&c.source,&c.funding))?
            .complete(|visit|{visit(&output);Ok(())},observer,self.stream.as_stream())?;
        safemlx::OperationEvent::validate_traversal_leaf(output.as_array(),observer)
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
        drop(graph);
        collector.finish()?;
        Ok(output)
    }
}
