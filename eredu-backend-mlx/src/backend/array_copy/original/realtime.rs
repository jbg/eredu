//! The shared isolated-copy leaf under one consumed realtime preparation claim.
use super::*;
use crate::backend::{error::Error,submission_recovery::native_role::{self,NativeRoleCapacity}};
use eredu_runtime::working_memory::{OriginalRealtimeNative,RealtimeNativeRequirements,WorkingMemoryError};
use eredu_nn::workspace::{HostMetadataFunding,WorkspaceContext};
use safemlx::PreparedInputRuntime;
use std::{cell::{Cell,RefCell},convert::Infallible,time::Duration};

/// Retains the actual selected stream and source-derived copy population.
/// Source ownership remains with the caller's exact borrowed state program.
pub(crate) struct RealtimeCopyPlan<'a> {
    stream:&'a Stream,layout:IsolatedCopyNativeLayout,capacity:NativeRoleCapacity,
    operands:usize,source_clones:usize,maximum_rank:usize,
}
impl OriginalCopyLayoutBuilder {
    /// Same counters and lowering as the existing isolated-copy preparation.
    /// No new runtime, stream, array or native ownership is created here.
    pub(crate) fn finish_realtime<'a>(self,runtime:&PreparedInputRuntime,stream:&'a Stream)
        ->Result<Option<RealtimeCopyPlan<'a>>,OriginalCopyCause> {
        if self.operands==0 && self.source_clones==0 {return Ok(None);}
        if self.operands==0 || self.host.operands!=0 || !self.host_stores.empty() {
            return Err(OriginalCopyCause::UnknownLayout);
        }
        let selected=StreamCopyPlan::<()>::capture(stream)?;
        let layout=match selected.device_type() {
            DeviceType::Cpu=>IsolatedCopyNativeLayout::cpu(self.operands,self.source_clones,self.maximum_rank),
            DeviceType::Gpu=>self.layout(0),
        }.ok_or(OriginalCopyCause::UnknownLayout)?;
        let population=OriginalBufferBudget::population_layout(runtime,
            self.logical_bytes.checked_mul(2).ok_or(OriginalCopyCause::Overflow)?,
            self.operands.checked_mul(2).ok_or(OriginalCopyCause::Overflow)?)?;
        Ok(Some(RealtimeCopyPlan {stream,layout,
            capacity:NativeRoleCapacity{graph:layout.graph_capacity,records:layout.record_capacity,
                backing:population.capacity()},operands:self.operands,
            source_clones:self.source_clones,maximum_rank:self.maximum_rank}))
    }
}
struct CopyWork<I> { invocation:I,roots:RefCell<Vec<Array>>,funding:HostMetadataFunding }
/// Private-construction lexical loan to the actual configured copy bank.
/// Every attempt consumes its counted slot before invoking the shared worker.
pub(crate) struct RealtimeCopyContext<'a> {
    stream:&'a Stream,roots:&'a RefCell<Vec<Array>>,remaining:Cell<usize>,retained:Cell<usize>,
}
impl RealtimeCopyContext<'_> {
    pub(crate) fn retain_source(&self,source:&Array)->Result<(),Error> {
        let remaining=self.retained.get().checked_sub(1).ok_or_else(mismatch)?;
        self.retained.set(remaining);
        self.roots.borrow_mut().push(source.try_clone_handle()?);Ok(())
    }
    pub(crate) fn copy(&self,source:IsolatedArrayCopy<'_>)->Result<Array,Error> {
        let remaining=self.remaining.get().checked_sub(1).ok_or_else(mismatch)?;
        self.remaining.set(remaining);
        source.copy_retained(self.stream,self.roots).map_err(Into::into)
    }
}
fn mismatch()->Error {Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn overflow()->Error {Error::PrefillControl(WorkingMemoryError::Overflow)}
fn sum(parts:&[usize])->Option<usize> {parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)}
type CopyCallback<'a,I,T>=dyn FnMut(&I,&RealtimeCopyContext<'_>)->Result<T,Error>+'a;
fn callback<'a,I:'static,T>(plan:Option<&'a RealtimeCopyPlan<'a>>,
    mut run:Option<&'a mut CopyCallback<'a,I,T>>)
    ->impl FnMut(&CopyWork<I>,&native_role::realtime::RealtimeRoleContext<'_>)->Result<Result<T,Infallible>,Error>+'a {
    move |work,role| {
        let plan=plan.expect("active retained copy plan");
        let mut graph=OperationEvent::prepare_resident_graph(plan.layout.graph,role.native().observer())?;
        graph.configure_nested_completions(&plan.layout.traversal,plan.layout.completion_attempts)?;
        let context=RealtimeCopyContext{stream:plan.stream,roots:&work.roots,
            remaining:Cell::new(plan.operands),retained:Cell::new(plan.source_clones)};
        let result=run.as_mut().expect("active exact state copier")(&work.invocation,&context);
        // Success and failure both close construction before shared Recovery.
        drop(graph);
        let output=result?;
        if context.remaining.get()!=0||context.retained.get()!=0{return Err(mismatch());}
        Ok(Ok(output))
    }
}
impl RealtimeCopyPlan<'_> {
    pub(crate) fn requirements(&self)->Result<RealtimeNativeRequirements,WorkingMemoryError> {
        RealtimeNativeRequirements::new(u64::try_from(self.capacity.backing).ok(),
            u64::try_from(self.capacity.graph).ok(),u64::try_from(self.capacity.records).ok())
    }
    fn root_count(&self)->Option<usize>{self.operands.checked_mul(2)?.checked_add(self.source_clones)}
    fn worker_controls<I,T>(&self)->Option<usize> {
        let parts=[self.layout.controls,
            OperationEvent::nested_completion_control_bytes::<1>()?,
            safemlx::original_scoped_evaluation_control_bytes()?,
            safemlx::original_scoped_deep_copy_control_bytes(self.maximum_rank)?,
            Array::descriptor_control_bytes()?,
            StreamCopyPlan::<()>::capture_control_bytes().ok()?,
            size_of::<Self>(),size_of::<RealtimeCopyContext<'_>>(),
            size_of::<CopyWork<I>>(),size_of::<Result<T,Error>>(),
            size_of::<Result<Result<T,Infallible>,BackendFailure>>(),
            size_of::<&mut CopyCallback<'_,I,T>>(),size_of::<(usize,usize)>(),
            size_of::<Result<Array,Error>>(),size_of::<super::super::NativeCopy<'_>>(),
            size_of::<IsolatedArrayCopy<'_>>(),
            size_of::<HostMetadataFunding>()];
        sum(&parts)
    }
    pub(crate) fn control_bytes<I:'static,T>(&self,runtime:&PreparedInputRuntime)->Option<usize> {
        let callback=callback::<I,T>(None,None);
        self.worker_controls::<I,T>()?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<Array>(self.root_count()?)?)?
            .checked_add(native_role::realtime::control_bytes::<CopyWork<I>,T,Infallible>(
                runtime,self.capacity,Some(PreparedPipelineCachePlan::new(self.layout.kernel_attempts)),
                size_of_val(&callback)).ok()?)
    }
    /// Completion is established by the existing shared native role before this
    /// returns. No current native scope remains across scheduler preparations.
    pub(crate) fn run<I:'static,T>(self,invocation:I,claim:OriginalRealtimeNative,
        runtime:&PreparedInputRuntime,funding:&HostMetadataFunding,timeout:Option<Duration>,
        run:&mut CopyCallback<'_,I,T>)->Result<T,BackendFailure> {
        funding.reserve_metadata(self.worker_controls::<I,T>().ok_or_else(||overflow().into_backend_failure())?)
            .map_err(|cause|Error::WorkspacePlanning(cause).into_backend_failure())?;
        if claim.physical_bytes()!=self.capacity.backing as u64||claim.graph_bytes()!=self.capacity.graph as u64
            ||claim.record_bytes()!=self.capacity.records as u64 {return Err(mismatch().into_backend_failure());}
        let roots=funding.metadata_vec(self.root_count().ok_or_else(||overflow().into_backend_failure())?).map_err(|cause|Error::Neural(cause).into_backend_failure())?;
        let work=CopyWork{invocation,roots:RefCell::new(roots),funding:funding.clone()};
        let callback=callback(Some(&self),Some(run));
        match native_role::realtime::run(work,claim,runtime,
            Some(PreparedPipelineCachePlan::new(self.layout.kernel_attempts)),funding,timeout,callback)? {
            Ok(value)=>Ok(value),Err(never)=>match never {},
        }
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
