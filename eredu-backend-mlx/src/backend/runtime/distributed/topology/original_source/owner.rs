//! Exact immutable setup owner lent to later original control phases.
use super::*;
use std::rc::Rc;

/// The actual table is retained, never reconstructed from equal descriptions.
/// Each owner alias carries its source and H after all native setup handles.
pub(crate) struct OriginalCommunicationOwner {
    actual: Option<Rc<ParallelCommunicators>>,
    authority: PartitionCommunicationAuthority,
    source: RetainedCommunicationSource,
    registered_buffers: Option<super::registered_buffers::RegisteredBuffers>,
    funding: HostMetadataFunding,
}
impl std::fmt::Debug for OriginalCommunicationOwner {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("OriginalCommunicationOwner")
            .field("session",&self.source.session_identity()).finish_non_exhaustive()
    }
}
impl Drop for OriginalCommunicationOwner {
    fn drop(&mut self) {
        // If this was the final setup alias, retire its Rc allocation before
        // releasing the moved table and, last, this owner's metadata account.
        if let Some(actual)=self.actual.take(){drop(Rc::into_inner(actual));}
    }
}
impl OriginalCommunicationOwner {
    pub(crate) fn bind(
        actual:&Rc<ParallelCommunicators>, selected:&CommunicationManifest,
        world:&NativeGroup, authority:&PartitionCommunicationAuthority,
        funding:&HostMetadataFunding,
    )->Result<Self,Error> {
        let controls=[size_of::<Self>(),size_of::<Result<Self,Error>>(),
            size_of::<(&Rc<ParallelCommunicators>,&CommunicationManifest,&NativeGroup,
                &PartitionCommunicationAuthority,&HostMetadataFunding)>(),
            size_of::<OriginalCommunicationSource<'_>>(),
            size_of::<Result<OriginalCommunicationSource<'_>,Error>>(),
            size_of::<Option<Rc<ParallelCommunicators>>>(),
        ];
        funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        // Share only after the common binder has authenticated every realized
        // group/route, actual native world, selection and live authority.
        let source=actual.bind_original_source(selected,world,authority,funding)?;
        Ok(Self{actual:Some(actual.clone()),authority:authority.clone(),
            source:source.source.clone(),registered_buffers:None,funding:funding.clone()})
    }
    fn actual(&self)->&ParallelCommunicators {
        self.actual.as_deref().expect("live communicator owner")
    }
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
    pub(crate) fn funding(&self)->&HostMetadataFunding{&self.funding}
    pub(crate) fn borrow(&self)->Result<OriginalCommunicationSource<'_>,Error> {self.borrow_funded(&self.funding)}
    pub(crate) fn borrow_funded(&self,funding:&HostMetadataFunding)->Result<OriginalCommunicationSource<'_>,Error> {
        let controls=[size_of::<&Self>(),size_of::<OriginalCommunicationSource<'_>>(),
            size_of::<Result<OriginalCommunicationSource<'_>,Error>>()];
        funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        // Reuse exact validation, including current availability. Accepted
        // completion retains its own source and does not reenter this method.
        let mut loan=self.actual().bind_original_source_retaining(self.source.manifest(),
            self.actual().control_world.native_group(),&self.authority,funding,&self.source)?;
        // The common validation above uses the unchanged immutable table. This
        // alias additionally retains the exact existing buffer pins established
        // before this owner entered model/control source construction.
        loan.registered_buffers=self.registered_buffers;
        Ok(loan)
    }
    pub(crate) fn pin_registered_buffers(mut self,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool)->Result<Self,Error> {
        let controls=[size_of::<Self>(),size_of::<Result<Self,Error>>(),
            size_of::<(&mut Self,&eredu_runtime::working_memory::WorkingMemoryPool)>(),
            size_of::<(RetainedCommunicationSource,super::registered_buffers::RegisteredBuffers)>()];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let (source,proof)=super::registered_buffers::pin(self.actual(),&self.source,
            &self.funding,pool)?;
        self.source=source;
        self.registered_buffers=Some(proof);
        Ok(self)
    }
    /// Retains the exact initialized pure tensor group without inventing a
    /// publication or agreement protocol for a frame that has neither.
    pub(crate) fn prepare_initialized_parallel_source(
        self,id:CollectiveGroupId,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool,
    )->Result<parallel::OriginalParallelSource,Error> {
        let controls=[size_of::<Self>(),size_of::<parallel::OriginalParallelSource>(),
            size_of::<Result<parallel::OriginalParallelSource,Error>>(),
            size_of::<(CollectiveGroupId,&eredu_runtime::working_memory::WorkingMemoryPool)>()];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let owner=self.pin_registered_buffers(pool)?;
        let mut source=owner.borrow()?.prepare_initialized_parallel_source(id,pool)?;
        source.retain_communication_owner(owner)?;
        Ok(source)
    }

    pub(crate) fn prepare_model_parallel_source(
        self,tensor:Option<CollectiveGroupId>,agreement:CollectiveGroupId,
        publication:eredu_runtime::PartitionOutputPublication,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool,
    )->Result<parallel::OriginalParallelSource,Error> {
        let controls=[size_of::<Self>(),size_of::<parallel::OriginalParallelSource>(),
            size_of::<Result<parallel::OriginalParallelSource,Error>>(),
            size_of::<(Option<CollectiveGroupId>,CollectiveGroupId,eredu_runtime::PartitionOutputPublication,&eredu_runtime::working_memory::WorkingMemoryPool)>(),
        ];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        let self_=self.pin_registered_buffers(pool)?;
        let mut source={self_.borrow()?.prepare_model_parallel_source(tensor,agreement,publication,pool)?};
        source.retain_communication_owner(self_)?;
        Ok(source)
    }
}
