//! Closed exact selected group and allocator owner for retained quote mechanisms.
use super::*;
use eredu_nn::workspace::WorkspaceOperationView;
use safemlx::{
    OriginalBufferBudget, PreparedInputRuntime,
    distributed::{GroupCpuLayoutStorage, GroupWorkerOperation},
};
struct StreamOwner {
    _source: RetainedCommunicationSource,
    _funding: HostMetadataFunding,
}
struct Publication { group:Group, descriptor:eredu_runtime::PartitionOutputPublication, root:usize }
// The retained architecture selection distinguishes neural TP from a
// publication-only model source. Sharing the physical Group does not turn on TP.
#[derive(Clone,Copy)]
enum SourceGroup { Neural(CollectiveGroupId), Publication(CollectiveGroupId) }
impl SourceGroup {
    fn id(self)->CollectiveGroupId{match self{Self::Neural(id)|Self::Publication(id)=>id}}
    fn operation(self)->CommunicationOperation{match self{
        Self::Neural(_)=>CommunicationOperation::AllReduceSum,
        Self::Publication(_)=>CommunicationOperation::Broadcast,
    }}
}
struct Body {
    owner: Option<super::super::OriginalCommunicationOwner>,
    control_taken: Cell<bool>,
    agreement:Option<super::super::agreement::OriginalAgreementInputs>,
    publication:Option<Publication>,
    group: Group,
    transport: safemlx::PreparedStreamCopy<StreamOwner>,
    id: CollectiveGroupId,
    kind: SourceGroup,
    authority: PartitionCommunicationAuthority,
    runtime: PreparedInputRuntime,
    source: RetainedCommunicationSource,
    fallback: eredu_nn::Error,
    funding: HostMetadataFunding,
}
pub(crate) struct OriginalParallelSource {
    body: Option<Rc<Body>>,
    funding: HostMetadataFunding,
}
impl Clone for OriginalParallelSource {
    fn clone(&self) -> Self {
        Self {
            body: self.body.clone(),
            funding: self.funding.clone(),
        }
    }
}
impl Drop for OriginalParallelSource {
    fn drop(&mut self) {
        if let Some(body) = self.body.take() {
            drop(Rc::into_inner(body));
        }
    }
}
impl std::fmt::Debug for OriginalParallelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalParallelSource")
            .field("group", &self.body().id)
            .finish_non_exhaustive()
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct ParallelBacking {
    pub(crate) output: u64,
    pub(crate) scratch: u64,
    pub(crate) births: usize,
}
impl OriginalCommunicationSource<'_> {
    /// The actual model source carries its exact session agreement inputs along
    /// with the tensor group. This private construction finishes before aliases
    /// can escape; later operation contexts only borrow immutable status values.
    pub(crate) fn prepare_model_parallel_source(&self,tensor:Option<CollectiveGroupId>,agreement:CollectiveGroupId,
        publication:eredu_runtime::PartitionOutputPublication,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool)->Result<OriginalParallelSource,Error> {
        reserve(&self.funding,&[
            size_of::<OriginalParallelSource>(),size_of::<Result<OriginalParallelSource,Error>>(),
            size_of::<Publication>(),size_of::<Option<Publication>>(),size_of::<eredu_runtime::PartitionOutputPublication>(),
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            size_of::<super::super::agreement::OriginalAgreementInputs>(),
            size_of::<Option<super::super::agreement::OriginalAgreementInputs>>(),
            size_of::<(&Self,Option<CollectiveGroupId>,CollectiveGroupId,&eredu_runtime::working_memory::WorkingMemoryPool)>(),size_of::<SourceGroup>(),
            size_of::<Option<&mut Body>>(),failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        let selected=self.source.manifest().select_group_operation(publication.group,CommunicationOperation::Broadcast)
            .map_err(|cause|failure(Cause::Rank(cause),&self.source,&self.funding))?;
        let root=selected.descriptor().members().iter().position(|&rank|rank==publication.owner_rank)
            .ok_or_else(||failure(Cause::Identity,&self.source,&self.funding))?;
        let group=self.group(selected.order()).ok_or_else(||failure(Cause::Identity,&self.source,&self.funding))?.0;
        let persistent=self.group_persistent(selected.order())?;
        if group.is_logical() || group.has_original_parallel()
            || persistent.native().has_unqualified_storage()
            || !persistent.native().is_for(group.native_group())
            || !persistent.source().same_source(&self.source)
            || group.retained_transport_stream().is_none() {
            return Err(failure(Cause::Resource,&self.source,&self.funding));
        }
        reserve(&self.funding,&[group.retention_copy_bytes().ok_or_else(overflow)?,size_of::<Result<Group,std::collections::TryReserveError>>()])?;
        let group=group.try_copy_for_retention().map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        let selected=tensor.map(SourceGroup::Neural).unwrap_or(SourceGroup::Publication(publication.group));
        let mut source=self.prepare_initialized_source(selected,pool)?;
        let inputs=self.prepare_agreement_inputs(agreement,pool)?;
        let body=Rc::get_mut(source.body.as_mut().expect("fresh source"))
            .expect("private source has not been shared");
        body.agreement=Some(inputs);
        body.publication=Some(Publication {group,descriptor:publication,root});
        Ok(source)
    }

    /// Bind the actual initialized allocator and setup-retained transport.
    /// Neither source can be substituted by equal geometry or initialized here.
    pub(crate) fn prepare_initialized_parallel_source(
        &self, id: CollectiveGroupId,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<OriginalParallelSource, Error> {
        reserve(&self.funding,&[size_of::<(&Self,CollectiveGroupId,&eredu_runtime::working_memory::WorkingMemoryPool)>(),
            size_of::<Result<OriginalParallelSource,Error>>(),size_of::<SourceGroup>()])?;
        self.prepare_initialized_source(SourceGroup::Neural(id),pool)
    }
    fn prepare_initialized_source(&self,kind:SourceGroup,pool:&eredu_runtime::working_memory::WorkingMemoryPool)
        ->Result<OriginalParallelSource,Error>{
        let id=kind.id();
        reserve(&self.funding, &[
            size_of::<(&Self,SourceGroup,&eredu_runtime::working_memory::WorkingMemoryPool)>(),
            size_of::<safemlx::PreparedInputRuntime>(),
            size_of::<Result<safemlx::PreparedInputRuntime,eredu_runtime::working_memory::WorkingMemoryError>>(),
            size_of::<Result<OriginalParallelSource,Error>>(),
            size_of::<Option<&Stream>>(),
            size_of::<Option<(&Group,&CommunicationGroupDescriptor,bool)>>(),
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            safemlx::InitializedInputAllocator::borrow_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        self.validate()?;
        let selected=self.source.manifest().select_group_operation(id,kind.operation())
            .map_err(|cause|failure(Cause::Rank(cause),&self.source,&self.funding))?;
        let group=self.group(selected.order()).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?.0;
        let stream=group.retained_transport_stream()
            .ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
        let runtime=crate::backend::managed_memory::input_allocator::borrow_admitted(pool)
            .map_err(|cause|failure(Cause::Allocator(cause),&self.source,&self.funding))?;
        self.prepare_source(kind,&runtime,stream)
    }

    pub(crate) fn prepare_parallel_source(
        &self,
        id: CollectiveGroupId,
        runtime: &PreparedInputRuntime,
        transport: &Stream,
    ) -> Result<OriginalParallelSource, Error> {
        reserve(&self.funding,&[size_of::<(&Self,CollectiveGroupId,&PreparedInputRuntime,&Stream)>(),
            size_of::<Result<OriginalParallelSource,Error>>(),size_of::<SourceGroup>()])?;
        self.prepare_source(SourceGroup::Neural(id),runtime,transport)
    }
    fn prepare_source(&self,kind:SourceGroup,runtime:&PreparedInputRuntime,transport:&Stream)
        ->Result<OriginalParallelSource,Error>{
        let id=kind.id();
        reserve(
            &self.funding,
            &[
                size_of::<OriginalParallelSource>(),
                size_of::<Body>(),
                size_of::<Rc<Body>>(),
                size_of::<Result<OriginalParallelSource, Error>>(),
                size_of::<(&Self, SourceGroup, &PreparedInputRuntime, &Stream)>(),
                size_of::<StreamCopyPlan<StreamOwner>>(),
                size_of::<Result<StreamCopyPlan<StreamOwner>, safemlx::StreamCopyCause>>(),
                size_of::<
                    Result<
                        safemlx::PreparedStreamCopy<StreamOwner>,
                        safemlx::StreamCopyError<StreamOwner>,
                    >,
                >(),
                CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
                neural_failure_controls().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
                PreparedInputRuntime::inspection_alias_control_bytes(),
            ],
        )?;
        self.validate()?;
        let transport = StreamCopyPlan::<StreamOwner>::capture(transport)
            .map_err(|_| failure(Cause::Resource, &self.source, &self.funding))?;
        if transport.device_type() != safemlx::DeviceType::Cpu {
            return Err(failure(Cause::Resource, &self.source, &self.funding));
        }
        let selected = self
            .source
            .manifest()
            .select_group_operation(id, kind.operation())
            .map_err(|cause| failure(Cause::Rank(cause), &self.source, &self.funding))?;
        let group = self
            .group(selected.order())
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?
            .0;
        let persistent = self.group_persistent(selected.order())?;
        if group.is_logical() && self.group_exchange(selected.order())?.is_none()
            && self.packed_world_plan(selected.order())?.is_none(){
            return Err(failure(Cause::Resource,&self.source,&self.funding));
        }
        if group.has_original_parallel()
            || persistent.native().has_unqualified_storage()
            || !persistent.native().is_for(group.native_group())
            || !persistent.source().same_source(&self.source)
        {
            return Err(failure(Cause::Resource, &self.source, &self.funding));
        }
        reserve(
            &self.funding,
            &[
                Layout::new::<[usize; 2]>()
                    .extend(Layout::new::<Body>())
                    .map_err(|_| overflow())?
                    .0
                    .pad_to_align()
                    .size(),
                group.retention_copy_bytes().ok_or_else(overflow)?,
                transport.control_bytes().ok_or_else(overflow)?,
                transport.native_wrapper_bytes(),
                Layout::new::<[usize; 2]>()
                    .extend(transport.shared_body_layout())
                    .map_err(|_| overflow())?
                    .0
                    .pad_to_align()
                    .size(),
                transport.owner_node_layout().size(),
                size_of::<StreamOwner>(),
                size_of::<Result<Group, std::collections::TryReserveError>>(),
            ],
        )?;
        let group = group
            .try_copy_for_retention()
            .map_err(|_| failure(Cause::Resource, &self.source, &self.funding))?;
        let transport = transport
            .realize(StreamOwner {
                _source: self.source.clone(),
                _funding: self.funding.clone(),
            })
            .map_err(|_| failure(Cause::Resource, &self.source, &self.funding))?;
        let fallback = neural_failure(Cause::Resource, &self.source, &self.funding);
        Ok(OriginalParallelSource {
            body: Some(Rc::new(Body {
                owner:None,
                control_taken:Cell::new(false),
                agreement:None,
                publication:None,
                group,
                transport,
                id,
                kind,
                authority: self.authority.clone(),
                runtime: runtime.inspection_alias(),
                source: self.source.clone(),
                fallback,
                funding: self.funding.clone(),
            })),
            funding: self.funding.clone(),
        })
    }
}
impl OriginalParallelSource {
    fn body(&self) -> &Body {
        self.body.as_deref().expect("live selected source")
    }
    pub(super) fn has_neural_context(&self)->bool {matches!(self.body().kind,SourceGroup::Neural(_))}
    pub(super) fn publication(&self)->Option<(&Group,eredu_runtime::PartitionOutputPublication,usize)> {
        self.body().publication.as_ref().map(|value|(&value.group,value.descriptor,value.root))
    }
    pub(super) fn input_runtime(&self)->&safemlx::PreparedInputRuntime {&self.body().runtime}
    pub(super) fn transport(&self) -> &Stream {
        self.body().transport.as_stream()
    }
    pub(crate) fn agreement_inputs(&self)->Option<&super::super::agreement::OriginalAgreementInputs> {
        self.body().agreement.as_ref()
    }
    pub(in crate::backend::runtime::distributed::topology::original_source) fn take_control_owner(&self)->Result<(),Error> {
        reserve(&self.funding,&[size_of::<&Self>(),size_of::<Result<(),Error>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        if self.body().control_taken.replace(true) || self.body().owner.is_none() || self.body().agreement.is_none() {
            return Err(failure(Cause::Resource,&self.body().source,&self.funding));
        }
        Ok(())
    }

    /// Lend the exact setup table retained by the model source. Standalone
    /// numerical sources intentionally carry no session-control owner.
    pub(crate) fn communication_source(&self)->Result<OriginalCommunicationSource<'_>,Error> {
        reserve(&self.funding,&[
            size_of::<Result<OriginalCommunicationSource<'_>,Error>>(),
            size_of::<Option<&super::super::OriginalCommunicationOwner>>(),
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        self.body().owner.as_ref()
            .ok_or_else(||failure(Cause::Resource,&self.body().source,&self.funding))?
            .borrow()
    }

    pub(in crate::backend::runtime::distributed::topology::original_source) fn retain_communication_owner(&mut self,owner:super::super::OriginalCommunicationOwner)->Result<(),Error> {
        reserve(&self.funding,&[
            size_of::<super::super::OriginalCommunicationOwner>(),
            size_of::<Option<&mut Body>>(),size_of::<Result<(),Error>>(),
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        let body=self.body.as_mut().and_then(Rc::get_mut)
            .ok_or_else(||failure(Cause::Resource,owner.source(),&self.funding))?;
        if body.owner.is_some() || !owner.source().same_source(&body.source)
            || !owner.funding().same_account(&self.funding) {
            return Err(failure(Cause::Identity,&body.source,&self.funding));
        }
        body.owner=Some(owner);
        Ok(())
    }

    pub(crate) fn declaration_source(&self) -> &RetainedCommunicationSource {
        &self.body().source
    }

    pub(crate) fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        Rc::ptr_eq(
            self.body.as_ref().expect("live source"),
            other.body.as_ref().expect("live source"),
        )
    }
    pub(crate) fn neural_error(&self, cause: Error) -> eredu_nn::Error {
        let body = self.body();
        match neural_failure_controls().and_then(|n| self.funding.reserve_metadata(n).ok()) {
            Some(()) => retain_neural(cause, &body.source, &self.funding),
            None => body.fallback.clone(),
        }
    }
    /// The exact neural group still selects a logical pair rather than a
    /// native world collective. Publication always has its separate source.
    pub(crate) fn logical_group_id(&self)->Option<CollectiveGroupId>{
        (self.has_neural_context() && self.body().group.is_logical()).then_some(self.body().id)
    }
    pub(crate) fn logical_collective_layout<'a>(&self,equation:WorkspaceOperationView<'a>)
        ->Result<Option<(usize,eredu_nn::workspace::WorkspaceLayoutView<'a>,safemlx::Dtype,GroupWorkerOperation)>,Error>{
        let Some(id)=self.logical_group_id() else{return Ok(None);};
        if !matches!(equation.kind,eredu_nn::workspace::WorkspaceOperationKindView::Collective(
            eredu_nn::workspace::WorkspaceCollectiveView::Sum{..}|eredu_nn::workspace::WorkspaceCollectiveView::GatherFirstAxis{..})) {
            return Ok(None);
        }
        super::super::workspace_sum::parallel_layout(&self.body().source,&self.funding,id,equation).map(Some)
    }
    pub(crate) fn quote_collective<'a>(
        &'a self,
        equation: WorkspaceOperationView<'a>,
    ) -> Result<GroupCpuLayoutStorage<'a>, Error> {
        let body = self.body();
        reserve(&self.funding,&[size_of::<(&Group,CollectiveGroupId)>(),
            size_of::<Option<&Publication>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let (group,id)=match equation.kind {
            eredu_nn::workspace::WorkspaceOperationKindView::Collective(eredu_nn::workspace::WorkspaceCollectiveView::Broadcast {group,root,..}) => {
                let publication=body.publication.as_ref().ok_or_else(||failure(Cause::Identity,&body.source,&self.funding))?;
                if group!=publication.descriptor.group || root!=publication.root {return Err(failure(Cause::Identity,&body.source,&self.funding));}
                (&publication.group,group)
            }
            _ if self.has_neural_context()=>(&body.group,body.id),
            _=>return Err(failure(Cause::Identity,&body.source,&self.funding)),
        };
        reserve(
            &self.funding,
            &[
                size_of::<(&Self, WorkspaceOperationView<'a>)>(),
                size_of::<Result<GroupCpuLayoutStorage<'a>, Error>>(),
                failure_control_bytes().ok_or_else(overflow)?,
                crate::backend::runtime::distributed::completion::group_source_controls()
                    .ok_or_else(overflow)?,
            ],
        )?;
        if group.is_logical() {return Err(failure(Cause::LogicalWorldTransport,&body.source,&self.funding));}
        if body.authority.ensure_active().is_err()
            || group.native_group().terminal_submission()
            || !crate::backend::runtime::distributed::completion::group_source_available(
                group,
            )
        {
            return Err(failure(Cause::Unavailable, &body.source, &self.funding));
        }
        let (_, input, dtype, operation) = super::super::workspace_sum::parallel_layout(
            &body.source,
            &self.funding,
            id,
            equation,
        )?;
        reserve(
            &self.funding,
            &[group
                .native_group()
                .cpu_layout_storage_control_bytes()
                .ok_or_else(overflow)?],
        )?;
        group
            .native_group()
            .cpu_layout_storage(input.shape(), dtype, operation)
            .map_err(|_| failure(Cause::NativeLayout, &body.source, &self.funding))
    }
    pub(crate) fn backing(
        &self,
        native: &safemlx::distributed::GroupCpuStorageFacts,
    ) -> Result<ParallelBacking, Error> {
        let body = self.body();
        reserve(
            &self.funding,
            &[
                size_of::<ParallelBacking>(),
                size_of::<Result<ParallelBacking, Error>>(),
                OriginalBufferBudget::request_layout_control_bytes()
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let (copy, output) = native.logical_backing_requests();
        let copy = OriginalBufferBudget::request_layout(&body.runtime, copy)
            .map_err(|cause| failure(Cause::Buffer(cause), &body.source, &self.funding))?
            .capacity();
        let output = OriginalBufferBudget::request_layout(&body.runtime, output)
            .map_err(|cause| failure(Cause::Buffer(cause), &body.source, &self.funding))?
            .capacity();
        // Sum may copy a noncontiguous input and return that same allocation,
        // or allocate/donate the result. Gather retains both input copy and output.
        let (births, _) = native.logical_backing_population();
        let (output,scratch)=match births {
            1=>(copy.max(output),0),
            2=>(output,copy),
            _=>return Err(failure(Cause::BackingPopulation(births),&body.source,&self.funding)),
        };
        Ok(ParallelBacking {
            output: u64::try_from(output).map_err(|_| overflow())?,
            scratch: u64::try_from(scratch).map_err(|_| overflow())?,
            births,
        })
    }
}

impl OriginalParallelSource {
    pub(crate) fn prepare_invocation(
        &self,
        operations: &[WorkspaceOperation],
    ) -> Result<OriginalParallelInvocation, Error> {
        self.prepare_invocation_with_boundary(operations,None,None)
    }
    pub(crate) fn prepare_invocation_with_boundary(&self,operations:&[WorkspaceOperation],
        mechanism:Option<crate::backend::nn::workspace::ResidentExecutionMechanisms>,
        addressable:Option<&crate::backend::nn::workspace::AddressableSources>)
        ->Result<OriginalParallelInvocation,Error>{
        let body = self.body();
        reserve(
            &self.funding,
            &[
                size_of::<OriginalParallelInvocation>(),
                size_of::<State>(),
                size_of::<Bound>(),
                size_of::<Result<OriginalParallelInvocation, Error>>(),
                size_of::<(&Self, CollectiveGroupId, &[WorkspaceOperation])>(),
                size_of::<std::iter::Enumerate<std::slice::Iter<'_, WorkspaceOperation>>>(),
                size_of::<(usize, &WorkspaceOperation)>(),
                size_of::<Option<(&Group, &CommunicationGroupDescriptor, bool)>>(),
                size_of::<Vec<Occurrence>>(),size_of::<bool>(),
                size_of::<Result<(), std::collections::TryReserveError>>(),
                CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
                failure_control_bytes().ok_or_else(overflow)?,
                neural_failure_controls().ok_or_else(overflow)?,
                OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,
                Stream::device_type_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        if body.authority.ensure_active().is_err()
            || body.group.native_group().terminal_submission()
            || !crate::backend::runtime::distributed::completion::group_source_available(
                &body.group,
            )
        {
            return Err(failure(Cause::Unavailable, &body.source, &self.funding));
        }
        let group = &body.group;
        if group.has_original_parallel() {
            return Err(failure(Cause::Identity, &body.source, &self.funding));
        }
        let count = operations
            .iter()
            .filter(|op| matches!(op.kind, WorkspaceOperationKind::Collective(
                WorkspaceCollective::Sum{..}|WorkspaceCollective::GatherFirstAxis{..}|WorkspaceCollective::Broadcast{..})))
            .count();
        let rc = Layout::new::<[usize; 2]>()
            .extend(Layout::new::<State>())
            .map_err(|_| overflow())?
            .0
            .pad_to_align();
        reserve(
            &self.funding,
            &[
                rc.size(),
                Layout::array::<Occurrence>(count)
                    .map_err(|_| overflow())?
                    .size(),
                group
                    .retention_copy_bytes()
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(overflow)?,
                size_of::<Rc<State>>(),
                size_of::<Weak<State>>(),
                size_of::<OriginalParallelBinding>(),
                size_of::<PartitionCommunicationAuthority>(),
                size_of::<OnceCell<Bound>>(),
                size_of::<Result<Group, std::collections::TryReserveError>>(),
            ],
        )?;
        let mut occurrences = Vec::new();
        occurrences
            .try_reserve_exact(count)
            .map_err(|_| failure(Cause::Resource, &body.source, &self.funding))?;
        let mut publication_seen=false;
        for (ordinal, operation) in operations.iter().enumerate() {
            match &operation.kind {
                WorkspaceOperationKind::Collective(collective @ (WorkspaceCollective::Sum { .. }
                    | WorkspaceCollective::GatherFirstAxis { .. } | WorkspaceCollective::Broadcast { .. })) => {
                    if publication_seen {
                        return Err(failure(Cause::Identity,&body.source,&self.funding));
                    }
                    publication_seen=matches!(collective,WorkspaceCollective::Broadcast{..});
                    let logical=if self.logical_group_id().is_some() && !publication_seen {
                        let mechanism=mechanism.ok_or_else(||failure(Cause::Resource,&body.source,&self.funding))?;
                        RetainedLogicalCollective::prepare(self,operation.as_view(),mechanism)?
                    }else{None};
                    let (native,backing)=if let Some(quote)=&logical {
                        let quote=quote.value();
                        (None,ParallelBacking{output:quote.output,scratch:quote.scratch,
                            births:quote.maximum_backing_births().ok_or_else(overflow)?})
                    }else{
                        let quoted=self.quote_collective(operation.as_view())?;
                        let backing=self.backing(quoted.evaluation())?;
                        reserve(&self.funding,&[quoted.ownership_control_bytes().ok_or_else(overflow)?])?;
                        let native=quoted.try_into_owned().map_err(|_|failure(Cause::Resource,&body.source,&self.funding))?;
                        (Some(native),backing)
                    };
                    let operation=match collective {
                        WorkspaceCollective::Sum{..}=>InvocationOperation::Sum,
                        WorkspaceCollective::Broadcast{group,root,..}=>InvocationOperation::Broadcast{group:*group,root:*root},
                        WorkspaceCollective::GatherFirstAxis{axis,peer_widths,..}=>{
                            reserve(&self.funding,&[Layout::array::<usize>(peer_widths.len()).map_err(|_|overflow())?.size(),
                                size_of::<Vec<usize>>(),size_of::<Result<(),std::collections::TryReserveError>>()])?;
                            let mut widths=Vec::new();widths.try_reserve_exact(peer_widths.len())
                                .map_err(|_|failure(Cause::Resource,&body.source,&self.funding))?;
                            widths.extend_from_slice(peer_widths);
                            InvocationOperation::Gather{axis:*axis,widths}
                        }
                        _=>unreachable!("matched exact collective"),
                    };
                    occurrences.push(Occurrence { ordinal, native, logical, backing, operation });
                }
                WorkspaceOperationKind::Collective(WorkspaceCollective::Boundary{..}) => {},
                WorkspaceOperationKind::Collective(_) => {
                    return Err(failure(Cause::Resource, &body.source, &self.funding));
                }
                _ => {}
            }
        }
        let boundaries=self.prepare_boundary_occurrences(operations,mechanism)?;
        let expert_regions=self.prepare_expert_occurrences(operations,mechanism,addressable)?;
        let actual = group
            .try_copy_for_retention()
            .map_err(|_| failure(Cause::Resource, &body.source, &self.funding))?;
        let context = group
            .try_copy_for_retention()
            .map_err(|_| failure(Cause::Resource, &body.source, &self.funding))?;
        let fallback = neural_failure(Cause::Resource, &body.source, &self.funding);
        let state = Rc::new(State {
            group: actual,
            quote_source: self.clone(),
            occurrences,
            boundaries,
            next_boundary:Cell::new(0),
            expert_regions, next_expert:Cell::new(0), expert_active:Cell::new(None), expert_local_taken:Cell::new(false), expert_local_complete:Cell::new(false), expert_transport_next:Cell::new(0), expert_transport_pending:Cell::new(false), expert_count_taken:Cell::new(false), expert_count_complete:Cell::new(false), expert_vote_next:Cell::new(0), expert_vote_pending:Cell::new(false),
            bound: OnceCell::new(),
            next: Cell::new(0),
            calling: Cell::new(false),
            forward_done:Cell::new(false),
            publication_started:Cell::new(false),
            model_roots:RefCell::new(None),
            publication_roots:RefCell::new(None),
            closed: Cell::new(false),
            authority: body.authority.clone(),
            source: body.source.clone(),
            funding: self.funding.clone(),
        });
        let context = context.with_original_parallel(OriginalParallelBinding {
            state: Rc::downgrade(&state),
            source: body.source.clone(),
            fallback,
            funding: self.funding.clone(),
        });
        Ok(OriginalParallelInvocation {
            context,
            state: Some(state),
            funding: self.funding.clone(),
        })
    }
}

impl OriginalParallelSource {
    pub(crate) fn initialized_runtime(&self)->&PreparedInputRuntime {&self.body().runtime}
}
