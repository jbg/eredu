//! Completed-source host readback for one admitted realtime frame.
use super::{MlxTensor,Error,CompletedRealtimeFrame,RealtimeOutputFrame,RealtimeDecisionDiagnostics};
use crate::backend::submission_recovery::native_role::realtime::RealtimeRoleContext;
use eredu_core::{BackendFailureKind,SharedBackendFailure};
use eredu_nn::{Tensor,workspace::{WorkspaceContext,WorkspaceTensor,WorkspaceDtype,
    WorkspaceFloatingType,WorkspaceMetadataFunding,WorkspaceMetadataError}};
use eredu_runtime::working_memory::{OriginalRealtimeBudgetCustody,WorkingMemoryError};
use safemlx::{Array,Dtype,OriginalScopeObserver};
use std::{cell::RefCell,rc::Rc,mem::{size_of,size_of_val}};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
enum Kind { Tokens, Diagnostic }
struct Row {shape:Vec<i32>,dtype:Dtype,elements:usize,kind:Kind}
/// Actual output rows from the completed cold coordinator. Shapes and scalar
/// types are descriptive; neither a native observer nor an account is created.
pub(crate) struct RealtimeHostReadPlan {
    rows:Vec<Row>,
    diagnostics:usize,
    aligned:bool,
    _funding:WorkspaceMetadataFunding,
}
#[derive(Debug,thiserror::Error)]
enum Cause {
    #[error("realtime host output differs from its retained source")] Identity,
    #[error(transparent)] Native(#[from] safemlx::error::Exception),
    #[error(transparent)] Slice(#[from] safemlx::error::AsSliceError),
    #[error(transparent)] Integer(#[from] std::num::TryFromIntError),
    #[error(transparent)] Neural(#[from] eredu_nn::Error),
    #[error(transparent)] Funding(#[from] eredu_core::HostMetadataFundingError),
    #[error(transparent)] Observation(#[from] eredu_core::ObservationError),
}
#[derive(Debug,thiserror::Error)]
#[error("realtime completed host observation: {cause}")]
struct Failure {
    #[source] cause:Cause,
    _funding:WorkspaceMetadataFunding,
    _custody:OriginalRealtimeBudgetCustody,
}
struct Owner {
    plan:RealtimeHostReadPlan,
    observer:OriginalScopeObserver,
    attempted:bool,
    funding:WorkspaceMetadataFunding,
    custody:OriginalRealtimeBudgetCustody,
}
/// Clones share the same once-only attempt; no source budget is refunded.
#[derive(Clone)]
pub(crate) struct OriginalRealtimeHostObserver(Option<Rc<RefCell<Owner>>>);
impl Drop for OriginalRealtimeHostObserver {
    fn drop(&mut self){if let Some(value)=self.0.take(){drop(Rc::into_inner(value));}}
}

impl RealtimeHostReadPlan {
    pub(crate) fn inspect(frame:&CompletedRealtimeFrame<WorkspaceTensor,WorkspaceTensor>,
        context:&WorkspaceContext)->Result<Self,Error> {
        let invalid=||Error::Neural(context.metadata_error(format_args!("realtime host read source has no funding or representable row population")));
        let funding=context.metadata_funding().ok_or_else(invalid)?;
        context.charge_metadata(size_of::<Self>()+size_of::<Row>()+size_of::<usize>()+size_of::<Result<Self,Error>>())
            .map_err(|cause|Error::Neural(cause.into()))?;
        let count=3usize.checked_add(usize::from(frame.aligned_audio().is_some()))
            .and_then(|n|n.checked_add(frame.diagnostics().len())).ok_or_else(invalid)?;
        let mut rows=context.metadata_vec(count).map_err(Error::Neural)?;
        let values=[frame.text(),frame.decision_audio(),frame.sampled_audio()].into_iter()
            .chain(frame.aligned_audio()).map(|value|(value,Kind::Tokens))
            .chain(frame.diagnostics().iter().map(|value|(value,Kind::Diagnostic)));
        for (index,(value,kind)) in values.enumerate() {
            let dtype=match (kind,value.layout().dtype()) {
                (Kind::Tokens,WorkspaceDtype::Int32)=>Dtype::Int32,
                (Kind::Tokens,WorkspaceDtype::Uint32)=>Dtype::Uint32,
                (Kind::Diagnostic,WorkspaceDtype::Float32)=>match value.layout().representation().ok_or_else(||Error::Neural(context.metadata_error(format_args!("realtime diagnostic row {index} has no physical representation: {:?}",value.layout()))))?.dtype() {
                    WorkspaceFloatingType::Float32=>Dtype::Float32,
                    WorkspaceFloatingType::Float16=>Dtype::Float16,
                    WorkspaceFloatingType::Bfloat16=>Dtype::Bfloat16,
                },
                _=>return Err(Error::Neural(context.metadata_error(format_args!("realtime host read row {index} has incompatible {kind:?} layout: {:?}",value.layout())))),
            };
            let mut shape=context.metadata_vec(value.shape().len()).map_err(Error::Neural)?;
            shape.extend_from_slice(value.shape());
            let elements=shape.iter().try_fold(1usize,|n,&dim|usize::try_from(dim).ok().and_then(|dim|n.checked_mul(dim)))
                .ok_or_else(invalid)?;
            rows.push(Row{shape,dtype,elements,kind});
        }
        Ok(Self{rows,diagnostics:frame.diagnostics().len(),aligned:frame.aligned_audio().is_some(),_funding:funding})
    }
    /// Same concrete owner, observer, error and destination helpers as prepare
    /// and observe. The payload vectors are cumulatively charged per output.
    pub(crate) fn control_bytes(&self)->Option<usize> { self.control_bytes_for(true) }
    fn control_bytes_for(&self,destinations:bool)->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<Owner>(),size_of::<OriginalRealtimeHostObserver>(),
            size_of::<Rc<RefCell<Owner>>>(),size_of::<std::cell::RefMut<'_,Owner>>(),
            size_of::<Result<RealtimeOutputFrame,Error>>(),size_of::<Failure>(),size_of::<Cause>(),
            size_of::<WorkspaceMetadataFunding>(),size_of::<OriginalRealtimeBudgetCustody>(),
            size_of::<RealtimeOutputFrame>(),size_of::<Option<Vec<i32>>>(),size_of::<usize>()*3];
        let shared=std::alloc::Layout::new::<[std::cell::Cell<usize>;2]>()
            .extend(std::alloc::Layout::new::<RefCell<Owner>>()).ok()?.0.pad_to_align().size();
        let mut total=frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)?
            .checked_add(shared)?.checked_add(SharedBackendFailure::control_bytes::<Failure>()?)?
            .checked_add(OriginalScopeObserver::control_bytes()?.checked_mul(self.rows.len().checked_add(1)?)?)?;
        if destinations {
            total=total.checked_add(RealtimeOutputFrame::host_source_handoff_bytes()?)?
                .checked_add(WorkspaceContext::metadata_vec_bytes::<RealtimeDecisionDiagnostics>(self.diagnostics)?)?;
        }
        for row in &self.rows {
            let fixed=[size_of::<&Row>(),size_of::<&Array>(),size_of::<safemlx::EvaluatedArray<'_>>(),
                size_of::<Result<safemlx::EvaluatedArray<'_>,safemlx::error::Exception>>(),
                size_of::<Result<Vec<i32>,Cause>>(),size_of::<Result<Vec<f32>,Cause>>(),
                size_of::<Result<&[i32],safemlx::error::AsSliceError>>(),size_of::<Result<&[u32],safemlx::error::AsSliceError>>(),
                size_of::<Result<&[f32],safemlx::error::AsSliceError>>(),size_of::<Result<&[half::f16],safemlx::error::AsSliceError>>(),
                size_of::<Result<&[half::bf16],safemlx::error::AsSliceError>>(),size_of::<Vec<usize>>()];
            total=total.checked_add(fixed.into_iter().try_fold(size_of_val(&fixed),usize::checked_add)?)?;
            if destinations { total=total.checked_add(match row.kind {
                Kind::Tokens=>WorkspaceContext::metadata_vec_bytes::<i32>(row.elements)?,
                Kind::Diagnostic=>WorkspaceContext::metadata_vec_bytes::<f32>(row.elements)?
                    .checked_add(WorkspaceContext::metadata_vec_bytes::<usize>(row.shape.len())?)?,
            })?; }
        }
        Some(total)
    }
}
impl OriginalRealtimeHostObserver {
    pub(crate) fn prepare(plan:RealtimeHostReadPlan,role:&RealtimeRoleContext<'_>)->Result<Self,Error> {
        let funding=role.metadata_funding();
        funding.reserve_metadata(plan.control_bytes_for(false).ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        Ok(Self(Some(Rc::new(RefCell::new(Owner{plan,observer:role.native().observer().clone(),
            attempted:false,funding:funding.clone(),custody:role.budget_custody()})))))
    }
    pub(super) fn observe(&self,frame:&CompletedRealtimeFrame<MlxTensor,MlxTensor>)->Result<RealtimeOutputFrame,Error> {
        let mut owner=self.0.as_ref().expect("live original host observer").try_borrow_mut().map_err(|_|Error::PrefillControl(WorkingMemoryError::AlreadyStarted))?;
        if owner.attempted {return Err(Error::PrefillControl(WorkingMemoryError::AlreadyStarted));}
        owner.attempted=true;
        owner.read(frame).map_err(|cause|Error::retained_original(SharedBackendFailure::new(
            BackendFailureKind::Other,Failure{cause,_funding:owner.funding.clone(),_custody:owner.custody.clone()}),false))
    }
}
impl Owner {
    fn validate<'a>(&self,index:usize,array:&'a Array,kind:Kind)->Result<(safemlx::EvaluatedArray<'a>,&Row),Cause> {
        let row=self.plan.rows.get(index).ok_or(Cause::Identity)?;
        if row.kind!=kind || row.shape!=array.shape() || row.dtype!=array.dtype() {return Err(Cause::Identity);}
        Ok((array.completed_in_original_scope(&self.observer)?,row))
    }
    fn tokens(&self,index:usize,array:&Array)->Result<Vec<i32>,Cause> {
        let (source,row)=self.validate(index,array,Kind::Tokens)?;
        let mut output=self.funding.metadata_vec(row.elements)?;
        match row.dtype {
            Dtype::Int32=>output.extend_from_slice(source.try_as_slice::<i32>()?),
            Dtype::Uint32=>for &value in source.try_as_slice::<u32>()? {output.push(i32::try_from(value)?);},
            _=>return Err(Cause::Identity),
        }
        if output.len()!=row.elements {return Err(Cause::Identity);}
        Ok(output)
    }
    fn diagnostic(&self,index:usize,prediction:usize,array:&Array)->Result<RealtimeDecisionDiagnostics,Cause> {
        let (source,row)=self.validate(index,array,Kind::Diagnostic)?;
        let mut values=self.funding.metadata_vec(row.elements)?;
        match row.dtype {
            Dtype::Float32=>values.extend_from_slice(source.try_as_slice::<f32>()?),
            Dtype::Float16=>values.extend(source.try_as_slice::<half::f16>()?.iter().map(|value|value.to_f32())),
            Dtype::Bfloat16=>values.extend(source.try_as_slice::<half::bf16>()?.iter().map(|value|value.to_f32())),
            _=>return Err(Cause::Identity),
        }
        if values.len()!=row.elements {return Err(Cause::Identity);}
        let mut shape=self.funding.metadata_vec(row.shape.len())?;
        for &dim in &row.shape {shape.push(usize::try_from(dim)?);}
        Ok(RealtimeDecisionDiagnostics::new(prediction,shape,values)?)
    }
    fn read(&self,frame:&CompletedRealtimeFrame<MlxTensor,MlxTensor>)->Result<RealtimeOutputFrame,Cause> {
        if frame.diagnostics().len()!=self.plan.diagnostics || frame.aligned_audio().is_some()!=self.plan.aligned {
            return Err(Cause::Identity);
        }
        let batch=usize::try_from(*self.plan.rows.first().and_then(|row|row.shape.first()).ok_or(Cause::Identity)?)?;
        let text=self.tokens(0,frame.text().as_array())?;
        let decision=self.tokens(1,frame.decision_audio().as_array())?;
        let sampled=self.tokens(2,frame.sampled_audio().as_array())?;
        let aligned=frame.aligned_audio().map(|value|self.tokens(3,value.as_array())).transpose()?;
        let first=3+usize::from(self.plan.aligned);
        let mut diagnostics=self.funding.metadata_vec(self.plan.diagnostics)?;
        for (prediction,value) in frame.diagnostics().iter().enumerate() {
            diagnostics.push(self.diagnostic(first+prediction,prediction,value.as_array())?);
        }
        Ok(RealtimeOutputFrame::new(batch,text,decision,sampled,aligned,diagnostics)
            .with_host_source(self.funding.clone().into())?)
    }
}
