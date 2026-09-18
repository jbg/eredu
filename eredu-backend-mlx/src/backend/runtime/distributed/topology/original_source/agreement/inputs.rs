//! Actual immutable status literals, born in the existing source-only arena.
use super::*;
use safemlx::PreparedInputPlan;
/// Both status choices are completed immutable I32 values. Reusing them does
/// not refund or bypass any later per-invocation native operation admission.
pub(crate) struct OriginalAgreementInputs {
    values:[Array;2],
    runtime:PreparedInputRuntime,
    group:CollectiveGroupId,
    source:RetainedCommunicationSource,
    funding:HostMetadataFunding,
}
impl OriginalCommunicationSource<'_> {
    pub(crate) fn prepare_agreement_inputs(&self,group:CollectiveGroupId,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool)->Result<OriginalAgreementInputs,Error> {
        reserve(&self.funding,&[
            size_of::<OriginalAgreementInputs>(),size_of::<RetainedCommunicationSource>(),size_of::<HostMetadataFunding>(),
            size_of::<Result<OriginalAgreementInputs,Error>>(),
            size_of::<(&Self,CollectiveGroupId,&eredu_runtime::working_memory::WorkingMemoryPool)>(),
            size_of::<PreparedInputRuntime>(),
            size_of::<Result<PreparedInputRuntime,eredu_runtime::working_memory::WorkingMemoryError>>(),
            size_of::<[i32;2]>(),size_of::<[usize;1]>(),size_of::<[usize;4]>(),
            size_of::<[PreparedInputPlan<'_>;2]>(),
            size_of::<[Result<PreparedInputPlan<'_>,safemlx::PreparedInputCause>;2]>(),
            safemlx::InitializedInputAllocator::borrow_control_bytes().ok_or_else(overflow)?,
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        self.validate()?;
        let selected=self.source.manifest().select_group_operation(group,CommunicationOperation::FailureAgreement)
            .map_err(|cause|failure(Cause::Rank(cause),&self.source,&self.funding))?;
        if !selected.requirement().exact_completion()
            || i32::try_from(selected.descriptor().members().len()).is_err() {
            return Err(failure(Cause::Resource,&self.source,&self.funding));
        }
        let runtime=crate::backend::managed_memory::input_allocator::borrow_admitted(pool)
            .map_err(|cause|failure(Cause::Allocator(cause),&self.source,&self.funding))?;
        let values=[0_i32,1];let shape=[1_usize];
        let plans=[runtime.i32(&values[..1],&shape),runtime.i32(&values[1..],&shape)];
        let [zero,one]=plans;
        let plans=[zero.map_err(|cause|failure(Cause::Input(cause),&self.source,&self.funding))?,
            one.map_err(|cause|failure(Cause::Input(cause),&self.source,&self.funding))?];
        let values=super::super::inputs::construct(self,plans)?;
        Ok(OriginalAgreementInputs{values,runtime,group,source:self.source.clone(),funding:self.funding.clone()})
    }
}
impl OriginalAgreementInputs {
    pub(super) fn same_source(&self,source:&OriginalCommunicationSource<'_>)->bool {
        self.source.same_source(source.source()) && self.funding.same_account(source.funding())
    }
    pub(crate) fn group(&self)->CollectiveGroupId {self.group}
    pub(super) fn value(&self,success:bool)->&Array {&self.values[usize::from(success)]}
    pub(crate) fn runtime(&self)->&PreparedInputRuntime {&self.runtime}
}
