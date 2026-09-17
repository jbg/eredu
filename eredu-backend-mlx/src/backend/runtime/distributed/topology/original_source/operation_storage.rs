//! Cold exact native operation and retained managed communicator source.
use super::*;
use safemlx::{Array,OriginalScopeObserver,Stream,distributed::{GroupCpuOperationStorage,GroupWorkerOperation}};

/// Native source/inputs remain borrowed until the ordinary constructor is used.
/// This prices managed Graph/H; physical backing and completion remain explicit.
pub(crate) struct OriginalCommunicationOperation<'a> {
    native:GroupCpuOperationStorage<'a>,
    persistent:OriginalCommunicatorPersistent<'a>,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
impl<'input> OriginalCommunicationOperation<'input> {
    pub(super) fn from_layout_binding(native:GroupCpuOperationStorage<'input>,persistent:OriginalCommunicatorPersistent<'input>,
        source:RetainedCommunicationSource,funding:WorkspaceMetadataFunding)->Self {
        Self{native,persistent,source,funding}
    }
    pub(crate) fn native(&self)->&GroupCpuOperationStorage<'_>{&self.native}
    pub(crate) fn backing_storage<'source>(&'source self,runtime:&'source safemlx::PreparedInputRuntime)
        ->Result<OriginalCommunicationBacking<'source,'input>,Error>{
        let frames=[size_of::<OriginalCommunicationBacking<'source,'input>>(),
            size_of::<Result<OriginalCommunicationBacking<'source,'input>,Error>>(),
            size_of::<(&Self,&safemlx::PreparedInputRuntime)>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            self.native.backing_storage_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let native=self.native.backing_storage(runtime)
            .map_err(|cause|failure(Cause::Buffer(cause),&self.source,&self.funding))?;
        Ok(OriginalCommunicationBacking{native,source:self.source.clone(),_funding:self.funding.clone()})
    }
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
    pub(crate) fn persistent(&self)->&OriginalCommunicatorPersistent<'_>{&self.persistent}
    pub(crate) fn construct(self,source:&OriginalCommunicationSource<'_>,observer:&OriginalScopeObserver,stream:&Stream)
        ->Result<OriginalCommunicationConstructed,Error> {
        let frames=[size_of::<Self>(),size_of::<OriginalCommunicationConstructed>(),
            size_of::<Result<OriginalCommunicationConstructed,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&OriginalScopeObserver,&Stream)>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            self.native.constructor().construction_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        if !self.source.same_source(source.source()){
            return Err(failure(Cause::Identity,&self.source,&self.funding));
        }
        source.validate()?;
        let value=self.native.construct_original(observer,stream)
            .map_err(|cause|failure(Cause::Native(cause),&self.source,&self.funding))?;
        // The result's same H account keeps persistent and both worker quotes
        // paid after the borrowed source wrapper itself retires.
        Ok(OriginalCommunicationConstructed::from_constructor(value,self.source,self.funding))
    }
}
impl<'a> OriginalCommunicationSource<'a> {
    fn operation_lookup_controls(&self)->Result<(),Error>{
        let frames=[size_of::<Option<(&Group,&CommunicationGroupDescriptor,bool)>>(),
            size_of::<Result<OriginalCommunicationOperation<'_>,Error>>(),size_of::<usize>(),
            size_of::<(&Self,&Array,GroupWorkerOperation)>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)
    }
    fn operation_storage<'b>(&'b self,group:&'b Group,input:&'b Array,operation:GroupWorkerOperation,
        persistent:OriginalCommunicatorPersistent<'b>)->Result<OriginalCommunicationOperation<'b>,Error>{
        let frames=[size_of::<OriginalCommunicationOperation<'b>>(),size_of::<OriginalCommunicatorPersistent<'b>>(),
            size_of::<Result<OriginalCommunicationOperation<'b>,Error>>(),
            size_of::<(&Self,&Group,&Array,GroupWorkerOperation)>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            group.native_group().cpu_operation_storage_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        self.validate()?;
        // Native full-group equations must never stand in for the existing
        // logical subgroup/world-wave/route driver. Its additional producers
        // need their own selected source quotation before joining this worker.
        if group.is_logical() || persistent.native().has_unqualified_storage() ||
            !persistent.native().is_for(group.native_group()) || !persistent.source().same_source(&self.source) {
            return Err(failure(Cause::Resource,&self.source,&self.funding));
        }
        let native=group.native_group().cpu_operation_storage(input,operation)
            .map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(OriginalCommunicationOperation{native,persistent,source:self.source.clone(),funding:self.funding.clone()})
    }
    /// The complete retained count matrix selects the existing variable worker;
    /// scalar operation tags can never supply its allocation population.
    pub(crate) fn group_variable_cpu_operation_storage<'b>(&'b self, order: usize,
        input: &'b Array, matrix: &'b [usize], transposed: bool)
        -> Result<OriginalCommunicationOperation<'b>, Error> {
        self.operation_lookup_controls()?;
        let (group, _, _) = self.group(order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?;
        self.variable_operation_storage(group, input, matrix, transposed, self.group_persistent(order)?)
    }
    /// The same native worker, selected through the ordinary logical variable
    /// plan and its retained physical-world owner. The complete matrix belongs
    /// to the caller's closed completed count source, never a local zero fill.
    pub(crate) fn world_variable_cpu_operation_storage<'b>(&'b self, order: usize,
        input: &'b Array, matrix: &'b [usize], transposed: bool)
        -> Result<OriginalCommunicationOperation<'b>, Error> {
        self.operation_lookup_controls()?;
        let (group, descriptor, wave) = self.group(order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?;
        let plan = group.logical_variable_world_plan()
            .ok_or_else(|| failure(Cause::LogicalWorldTransport, &self.source, &self.funding))?;
        if !wave || plan.members() != descriptor.members()
            || !std::ptr::eq(plan.group(), group)
            || !group.native_group().shares_native_handle(self.world().native_group()) {
            return Err(failure(Cause::Identity, &self.source, &self.funding));
        }
        self.variable_operation_storage(self.world(), input, matrix, transposed, self.world_persistent()?)
    }
    fn variable_operation_storage<'b>(&'b self, group: &'b Group,
        input: &'b Array, matrix: &'b [usize], transposed: bool,
        persistent: OriginalCommunicatorPersistent<'b>) -> Result<OriginalCommunicationOperation<'b>, Error> {
        let frames = [size_of::<OriginalCommunicationOperation<'b>>(),
            size_of::<OriginalCommunicatorPersistent<'b>>(),
            size_of::<(&Self, usize, &Array, &[usize], bool)>(),
            size_of::<crate::backend::runtime::distributed::group::LogicalVariableWorldPlan<'_>>(),
            size_of::<Option<crate::backend::runtime::distributed::group::LogicalVariableWorldPlan<'_>>>(),
            size_of::<Result<OriginalCommunicationOperation<'b>, Error>>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            group.native_group().variable_cpu_operation_storage_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(frames.into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        self.validate()?;
        if group.is_logical() || persistent.native().has_unqualified_storage()
            || !persistent.native().is_for(group.native_group())
            || !persistent.source().same_source(&self.source) {
            return Err(failure(Cause::Resource, &self.source, &self.funding));
        }
        let native = group.native_group().variable_cpu_operation_storage(input, matrix, transposed)
            .map_err(|_| failure(Cause::Resource, &self.source, &self.funding))?;
        Ok(OriginalCommunicationOperation { native, persistent,
            source: self.source.clone(), funding: self.funding.clone() })
    }
    /// The selected status itinerary borrows its actual physical Ring.
    /// This does not admit a logical group as a native whole-group collective.
    pub(crate) fn status_cpu_operation_storage<'b>(&'b self,order:usize,plan:crate::backend::runtime::distributed::group::StatusPlan<'_>,
        input:&'b Array,operation:GroupWorkerOperation)->Result<OriginalCommunicationOperation<'b>,Error>{
        let frames=[size_of::<crate::backend::runtime::distributed::group::StatusPlan<'_>>(),
            size_of::<(&Self,usize,&Array,GroupWorkerOperation)>(),size_of::<OriginalCommunicatorPersistent<'b>>(),
            size_of::<Result<OriginalCommunicationOperation<'b>,Error>>(),CommunicationManifest::group_operation_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        self.operation_lookup_controls()?;self.validate()?;
        let (group,descriptor,_)=self.group(order).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
        let selected=self.source.manifest().select_group_operation(descriptor.id(),CommunicationOperation::FailureAgreement)
            .map_err(|cause|failure(Cause::Rank(cause),&self.source,&self.funding))?;
        let persistent=self.group_persistent(order)?;
        if selected.order()!=order||!selected.requirement().exact_completion()||group.selected_status_plan().map_err(|_|failure(Cause::LogicalWorldTransport,&self.source,&self.funding))?!=Some(plan)
            ||!plan.accepts(operation)||input.shape()!=[1]||input.dtype()!=safemlx::Dtype::Int32
            ||persistent.native().has_unqualified_storage()||!persistent.native().is_for(group.native_group())
            ||!persistent.source().same_source(&self.source){return Err(failure(Cause::Resource,&self.source,&self.funding));}
        self.funding.reserve_metadata(group.native_group().cpu_operation_storage_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let native=group.native_group().cpu_operation_storage(input,operation).map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(OriginalCommunicationOperation{native,persistent,source:self.source.clone(),funding:self.funding.clone()})
    }
    pub(crate) fn world_cpu_operation_storage<'b>(&'b self,input:&'b Array,operation:GroupWorkerOperation)
        ->Result<OriginalCommunicationOperation<'b>,Error>{
        self.operation_lookup_controls()?;
        self.operation_storage(self.world(),input,operation,self.world_persistent()?)
    }
    pub(crate) fn group_cpu_operation_storage<'b>(&'b self,order:usize,input:&'b Array,operation:GroupWorkerOperation)
        ->Result<OriginalCommunicationOperation<'b>,Error>{
        self.operation_lookup_controls()?;
        let (group,_,_)=self.group(order).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
        self.operation_storage(group,input,operation,self.group_persistent(order)?)
    }
}

/// Same original operation and actual allocator loan plus physical facts/H.
pub(crate) struct OriginalCommunicationBacking<'source,'input>{
    native:safemlx::distributed::GroupCpuBackingStorage<'source,'input>,
    source:RetainedCommunicationSource,
    _funding:WorkspaceMetadataFunding,
}
impl OriginalCommunicationBacking<'_,'_>{
    pub(crate) fn capacity(&self)->usize{self.native.capacity()}
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
}

/// Actual CPU operation plus its finite original completion recipe. The same
/// retained communicator H remains last, including all failed-prefix exits.
pub(crate) struct OriginalCommunicationCompletedOperation<'a> {
    native:safemlx::distributed::GroupCpuCompletionStorage<'a>,
    persistent:OriginalCommunicatorPersistent<'a>,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
impl<'a> OriginalCommunicationOperation<'a> {
    pub(crate) fn with_completion(self)->Result<OriginalCommunicationCompletedOperation<'a>,Error> {
        let parts=[size_of::<OriginalCommunicationCompletedOperation<'a>>(),
            size_of::<Result<OriginalCommunicationCompletedOperation<'a>,Error>>(),size_of::<Self>(),
            self.native.completion_storage_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let Self{native,persistent,source,funding}=self;
        let native=native.with_completion_storage().map_err(|_|failure(Cause::Resource,&source,&funding))?;
        Ok(OriginalCommunicationCompletedOperation{native,persistent,source,funding})
    }
}
impl<'a> OriginalCommunicationCompletedOperation<'a> {
    pub(super) fn from_retained_layout(native:safemlx::distributed::GroupCpuCompletionStorage<'a>,
        persistent:OriginalCommunicatorPersistent<'a>,source:RetainedCommunicationSource,funding:WorkspaceMetadataFunding)->Self {
        Self{native,persistent,source,funding}
    }

    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
    pub(crate) fn graph_capacity(&self)->usize{self.native.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize{self.native.record_capacity()}
    pub(crate) fn native(&self)->&safemlx::distributed::GroupCpuCompletionStorage<'_>{&self.native}
    /// Select and copy the same actual communicator into existing completion
    /// destinations. None names the exact control world; Some names its retained
    /// group order. No caller-provided population or equal-geometry group binds.
    pub(crate) fn prepare_resources(&self,source:&OriginalCommunicationSource<'_>,group:Option<usize>)
        ->Result<crate::backend::runtime::distributed::completion::prepared::ReadyCompletionResources,Error>{
        use crate::backend::runtime::distributed::completion::prepared::{
            PreparedCompletionResources,ReadyCompletionResources,CompletionResourceLayout};
        let parts=[size_of::<(&Self,&OriginalCommunicationSource<'_>,Option<usize>)>(),
            size_of::<Option<(&Group,&CommunicationGroupDescriptor,bool)>>(),
            size_of::<ReadyCompletionResources>(),size_of::<Result<ReadyCompletionResources,Error>>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        let actual=match group {None=>source.world(),Some(order)=>source.group(order)
            .ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?.0};
        if !self.source.same_source(source.source()) || !self.native.is_for_group(actual.native_group()) {
            return Err(failure(Cause::Identity,&self.source,&self.funding));
        }
        // The original native Event owns the actual result/root; its lazy input
        // edge and stream value retain their existing native owners. No second
        // Array/Stream clone is introduced for the host recovery vector.
        let mut prepared=PreparedCompletionResources::prepare_original(source,
            CompletionResourceLayout{arrays:0,counts:&[],groups:1,routes:0,streams:0},self.native.traversal())?;
        match group {None=>prepared.retain_control_world()?,Some(order)=>prepared.retain_group(order)?};
        prepared.finish()
    }
    pub(crate) fn construct(self,source:&OriginalCommunicationSource<'_>,observer:&OriginalScopeObserver,stream:&Stream)
        ->Result<OriginalCommunicationConstructed,Error>{
        let parts=[size_of::<Self>(),size_of::<OriginalCommunicationConstructed>(),
            size_of::<Result<OriginalCommunicationConstructed,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&OriginalScopeObserver,&Stream)>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            self.native.control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        if !self.source.same_source(source.source()) {return Err(failure(Cause::Identity,&self.source,&self.funding));}
        source.validate()?;
        let value=self.native.construct_original(observer,stream)
            .map_err(|cause|failure(Cause::Native(cause),&self.source,&self.funding))?;
        Ok(OriginalCommunicationConstructed::from_constructor(value,self.source,self.funding))
    }
}

mod accepted;
pub(crate) use accepted::AcceptedCommunicationSource;
