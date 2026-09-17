//! Source-funded lazy constructor using the same native graph producer.
use super::*;
use safemlx::{Array,OriginalScopeObserver,Stream,distributed::GroupConstructorStorage};

pub(crate) struct OriginalCommunicationConstructor<'a> {
    native:GroupConstructorStorage<'a>,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
/// The lazy output retires before the retained source and query/constructor H.
/// Native primitive/input references retain their actual Graph allocations;
/// this host owner is carried separately into the later completion producer.
pub(crate) struct OriginalCommunicationConstructed {
    value:Array,
    source:RetainedCommunicationSource,
    _funding:WorkspaceMetadataFunding,
}
impl OriginalCommunicationConstructed {
    pub(super) fn from_constructor(value:Array,source:RetainedCommunicationSource,funding:WorkspaceMetadataFunding)->Self {
        Self{value,source,_funding:funding}
    }
    pub(crate) fn value(&self)->&Array{&self.value}
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
    pub(in crate::backend::runtime::distributed) fn into_parts(self)
        ->(Array,RetainedCommunicationSource,WorkspaceMetadataFunding) {
        (self.value,self.source,self._funding)
    }
}
impl<'a> OriginalCommunicationWorkers<'a> {
    pub(crate) fn constructor_storage(&self)->Result<OriginalCommunicationConstructor<'a>,Error> {
        let controls=[size_of::<OriginalCommunicationConstructor<'a>>(),
            size_of::<Result<OriginalCommunicationConstructor<'a>,Error>>(),size_of::<&Self>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            self.native().constructor_storage_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding().reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let native=self.native().constructor_storage().map_err(|_|failure(Cause::Resource,self.source(),self.funding()))?;
        Ok(OriginalCommunicationConstructor{native,source:self.source().clone(),funding:self.funding().clone()})
    }
}
impl OriginalCommunicationConstructor<'_> {
    pub(crate) fn native(&self)->&GroupConstructorStorage<'_>{&self.native}
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
    pub(crate) fn construct(self,source:&OriginalCommunicationSource<'_>,observer:&OriginalScopeObserver,stream:&Stream)
        ->Result<OriginalCommunicationConstructed,Error> {
        let controls=[size_of::<OriginalCommunicationConstructed>(),
            size_of::<Result<OriginalCommunicationConstructed,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&OriginalScopeObserver,&Stream)>(),size_of::<safemlx::error::Exception>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            self.native.construction_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        if !self.source.same_source(&source.source) {return Err(failure(Cause::Identity,&self.source,&self.funding));}
        source.validate()?;
        // Keep query/constructor H with every escaped native failure as well as
        // its original Scope cause. The same prepaid retained-error worker is
        // used; no ordinary formatted string or unpriced source shell is born.
        let value=self.native.construct_original(observer,stream)
            .map_err(|cause|failure(Cause::Native(cause),&self.source,&self.funding))?;
        Ok(OriginalCommunicationConstructed{value,source:self.source,_funding:self.funding})
    }
}

/// The borrowed constructed output retains its exact group/input graph; source
/// and H survive every successful query until this descriptor is released.
pub(crate) struct OriginalCommunicationEvaluation<'a> {
    native:safemlx::distributed::GroupCpuEvaluationStorage<'a>,
    source:RetainedCommunicationSource,
    _funding:WorkspaceMetadataFunding,
}
impl OriginalCommunicationConstructed {
    pub(crate) fn evaluation_storage(&self)->Result<OriginalCommunicationEvaluation<'_>,Error> {
        let controls=[size_of::<OriginalCommunicationEvaluation<'_>>(),
            size_of::<Result<OriginalCommunicationEvaluation<'_>,Error>>(),size_of::<&Self>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            self.value.distributed_cpu_evaluation_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self._funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let native=self.value.distributed_cpu_evaluation_storage()
            .map_err(|_|failure(Cause::Resource,&self.source,&self._funding))?;
        Ok(OriginalCommunicationEvaluation{native,source:self.source.clone(),_funding:self._funding.clone()})
    }
}
impl OriginalCommunicationEvaluation<'_> {
    pub(crate) fn native(&self)->&safemlx::distributed::GroupCpuEvaluationStorage<'_>{&self.native}
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
}
