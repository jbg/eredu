//! Exact dense child publication, followed by the sole outer prompt completion.
use super::*;
use eredu_runtime::{
    DenseHostSlotInitialization, HostSlotAttachmentError,
    working_memory::{
        FundedDenseHostSlots, InferencePreparationStage, InferencePromptCompletion,
        InitializedDenseDecoderTableGroup, RegisteredDecoderHostCopy,
        RegisteredDenseDecoderTableGroup, WorkingMemoryFundingRun, WorkingMemoryFundingScope,
        WorkingMemoryStorage,
    },
};
use std::fmt;

type LivePlan<'a> = RegisteredDenseDecoderTableGroup<
    'a,
    MlxHybridLayerState,
    MlxHybridLayerState,
    Slot,
    Slot,
    StorageIdentity,
>;
type SavedPlan<'a> = RegisteredDenseDecoderTableGroup<
    'a,
    SavedHybridGroupedLayer,
    MlxHybridLayerState,
    Slot,
    Slot,
    StorageIdentity,
>;
pub(crate) enum PreparedHybridDenseGroup<'a> {
    Live(LivePlan<'a>, Option<super::paged::PagedWork>),
    Saved(SavedPlan<'a>, Option<super::paged::PagedWork>),
}
pub(crate) enum InitializedHybridDenseGroup<'a> {
    Live(
        InitializedDenseDecoderTableGroup<
            'a,
            MlxHybridLayerState,
            MlxHybridLayerState,
            Slot,
            Slot,
            StorageIdentity,
        >,
        Option<super::paged::PagedWork>,
    ),
    Saved(
        InitializedDenseDecoderTableGroup<
            'a,
            SavedHybridGroupedLayer,
            MlxHybridLayerState,
            Slot,
            Slot,
            StorageIdentity,
        >,
        Option<super::paged::PagedWork>,
    ),
}
enum DenseStorage<'a> {
    Live(FundedDenseHostSlots<'a, MlxHybridLayerState, MlxHybridLayerState, StorageIdentity>),
    Saved(FundedDenseHostSlots<'a, SavedHybridGroupedLayer, MlxHybridLayerState, StorageIdentity>),
}
pub(crate) struct PreparedDenseHybridGroupedState<'a> {
    layers: DenseStorage<'a>,
    layout: SharedStateLayout,
    global_layer_start: usize,
    manager: Option<CacheResidencyManager>,
}

impl<'a> PreparedHybridGroupedCopy<'a> {
    /// Reuses the same attention and fixed-slot numerical workers, while each
    /// actual parent/child table is born under this independent host account.
    pub(crate) fn copy_dense_with_preparation(
        self,
        host: &eredu_core::HostPreparationAuthority,
        funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<MlxHybridState, Error> {
        self.validate_fixed()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        if self.is_paged(){return Err(Error::Neural(funding.metadata_source(WorkingMemoryError::UnknownBound)));}
        let layers = match self.source {
            Source::Live(s) => self.copy_prepared_group(
                s.layers
                    .prepare_copy_slots()
                    .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
                    .for_dense_destination()
                    .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
                host,
                funding,
                stream,
                roots,
            )?,
            Source::Saved(s) => self.copy_prepared_group(
                s.layers
                    .prepare_copy_slots()
                    .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?
                    .for_dense_destination()
                    .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
                host,
                funding,
                stream,
                roots,
            )?,
        };
        Ok(MlxHybridState {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
            manager: None,
            inference_retention: eredu_runtime::working_memory::InferenceRetention::new(),
        })
    }
    fn copy_prepared_group<S>(
        &self,
        source: DenseHostSlotInitialization<'_, S, MlxHybridLayerState>,
        host: &eredu_core::HostPreparationAuthority,
        funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<eredu_runtime::HostSlotTable<MlxHybridLayerState>, Error> {
        use super::super::super::resident_copy::host_copy::copy_slots;
        copy_slots(source, host, funding, |i, _| {
            let layer = self
                .layer(i)
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            let child = match layer.fixed {
                Fixed::Live(s) => s
                    .prepare_slots()
                    .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
                Fixed::Saved(s) => s
                    .prepare_copy_slots()
                    .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?,
            }
            .for_dense_destination()
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
            let attention = self.copy_attention_with_metadata(i, stream, roots, Some(funding))?;
            #[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
            super::tests::original_discard::before_child_reservation(roots);
            let fixed = copy_slots(child, host, funding, |_, value| {
                copy_slot_retained(value, stream, roots).map_err(Error::from)
            })?;
            Ok(MlxHybridLayerState {
                attention,
                fixed: FixedStateSlots::from_published_slots(fixed),
                fixed_offset: layer.fixed_offset,
            })
        })
    }

    pub(crate) fn dense_initialization_peak_bytes_fixed(
        &self,
    ) -> Result<u64, ResidentDecoderPreparationError> {
        self.validate_fixed()?;
        let mut bytes = match self.source {
            Source::Live(s) => s
                .layers
                .prepare_copy_slots()?
                .for_dense_destination::<MlxHybridLayerState>()?
                .initialization_peak_bytes(),
            Source::Saved(s) => s
                .layers
                .prepare_copy_slots()?
                .for_dense_destination::<MlxHybridLayerState>()?
                .initialization_peak_bytes(),
        };
        for i in 0..self.len() {
            let child = match self
                .layer(i)
                .ok_or(WorkingMemoryError::IdentityMismatch)?
                .fixed
            {
                Fixed::Live(s) => s.prepare_slots()?,
                Fixed::Saved(s) => s.prepare_copy_slots()?,
            };
            bytes = bytes
                .checked_add(
                    child
                        .for_dense_destination::<Slot>()?
                        .initialization_peak_bytes(),
                )
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        Ok(bytes)
    }

    pub(crate) fn dense_host_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::{
            DecoderHostPreparationError as E, RegisteredDenseDecoderInitialization as Table,
        };
        let nested =
            eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes().ok_or(E::UnknownBound)?;
        let mut bytes =
            match self.source {
                Source::Live(_) => Table::<
                    MlxHybridLayerState,
                    MlxHybridLayerState,
                    StorageIdentity,
                >::preparation_control_bytes(true, nested)?
                .checked_add(LivePlan::preparation_control_bytes(self.len())?)
                .ok_or(E::Overflow)?,
                Source::Saved(_) => Table::<
                    SavedHybridGroupedLayer,
                    MlxHybridLayerState,
                    StorageIdentity,
                >::preparation_control_bytes(false, nested)?
                .checked_add(SavedPlan::preparation_control_bytes(self.len())?)
                .ok_or(E::Overflow)?,
            };
        for i in 0..self.len() {
            let layer = self.layer(i).ok_or(E::UnknownBound)?;
            bytes = bytes
                .checked_add(
                    Table::<Slot, Slot, StorageIdentity>::preparation_control_bytes(
                        matches!(layer.fixed, Fixed::Live(_)),
                        nested,
                    )?,
                )
                .ok_or(E::Overflow)?;
        }
        if let Some(paged)=&self.paged {
            bytes=bytes.checked_add(paged.host_preparation_bytes(self.paged_caller_controls().ok_or(E::Overflow)?)
                .ok_or(E::Overflow)?).ok_or(E::Overflow)?;
        }
        bytes
            .checked_add(std::mem::size_of::<PreparedHybridDenseGroup<'_>>())
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<
                    Result<PreparedHybridDenseGroup<'_>, Error>,
                >())
            })
            .ok_or(E::Overflow)
    }

    pub(crate) fn dense_host_copy(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<PreparedHybridDenseGroup<'a>, Error> {
        self.dense_host_copy_with_preparation(pool, None)
    }
    pub(crate) fn dense_host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<PreparedHybridDenseGroup<'a>, Error> {
        self.validate()?;
        let mut children = Vec::new();
        children.try_reserve_exact(self.len()).map_err(other)?;
        if preparation.is_some() && children.capacity() != self.len() {
            return Err(mismatch());
        }
        for i in 0..self.len() {
            let registered = match self.layer(i).ok_or_else(mismatch)?.fixed {
                Fixed::Live(s) => RegisteredDecoderHostCopy::bind_with_preparation(
                    pool,
                    s.prepare_slots().map_err(other)?,
                    StorageIdentity::HostMetadata(s.metadata().identity().registry_key().clone()),
                    preparation,
                )
                .map_err(other)?,
                Fixed::Saved(s) => s.prepare_copy().map_err(other)?,
            };
            children.push(registered.for_dense_destination::<Slot>().map_err(other)?);
        }
        let paged=self.prepare_paged_work(preparation)?;
        Ok(match self.source {
            Source::Live(s) => {
                let outer = RegisteredDecoderHostCopy::bind_with_preparation(
                    pool,
                    s.layers.prepare_copy_slots().map_err(other)?,
                    StorageIdentity::HostMetadata(
                        s.layers.metadata().identity().registry_key().clone(),
                    ),
                    preparation,
                )
                .map_err(other)?
                .for_dense_destination::<MlxHybridLayerState>()
                .map_err(other)?;
                PreparedHybridDenseGroup::Live(
                    RegisteredDenseDecoderTableGroup::new_with_preparation(
                        outer,
                        children,
                        preparation,
                    )
                    .map_err(other)?,
                    paged,
                )
            }
            Source::Saved(s) => PreparedHybridDenseGroup::Saved(
                RegisteredDenseDecoderTableGroup::new_with_preparation(
                    s.layers
                        .prepare_copy()
                        .map_err(other)?
                        .for_dense_destination::<MlxHybridLayerState>()
                        .map_err(other)?,
                    children,
                    preparation,
                )
                .map_err(other)?,
                paged,
            ),
        })
    }
    pub(crate) fn copy_dense_retained(
        self,slots:InitializedHybridDenseGroup<'a>,stream:&Stream,roots:&RefCell<Vec<Array>>,
    )->Result<PreparedDenseHybridGroupedState<'a>,Error>{
        self.copy_dense_retained_with(slots,stream,roots,&mut |_|Ok(()))
    }
    pub(crate) fn copy_dense_retained_with(
        self,slots:InitializedHybridDenseGroup<'a>,stream:&Stream,roots:&RefCell<Vec<Array>>,
        observe:&mut dyn FnMut(&Array)->Result<(),Error>,
    )->Result<PreparedDenseHybridGroupedState<'a>,Error>{
        self.validate()?;
        let (layers,manager)=match (self.source,slots){
            (Source::Live(s),InitializedHybridDenseGroup::Live(slots,paged))=>{
                let (layers,manager)=self.copy_dense_group(slots,
                    s.layers.prepare_copy_slots().map_err(other)?.for_dense_destination().map_err(other)?,
                    paged,stream,roots,observe)?;
                (DenseStorage::Live(layers),manager)
            }
            (Source::Saved(s),InitializedHybridDenseGroup::Saved(slots,paged))=>{
                let (layers,manager)=self.copy_dense_group(slots,
                    s.layers.prepare_copy_slots().map_err(other)?.for_dense_destination().map_err(other)?,
                    paged,stream,roots,observe)?;
                (DenseStorage::Saved(layers),manager)
            }
            _=>return Err(mismatch()),
        };
        Ok(PreparedDenseHybridGroupedState{layers,layout:self.shared_layout().clone(),
            global_layer_start:self.global_layer_start(),manager})
    }
    fn copy_dense_group<S>(
        &self,
        mut slots: InitializedDenseDecoderTableGroup<
            'a,
            S,
            MlxHybridLayerState,
            Slot,
            Slot,
            StorageIdentity,
        >,
        actual: DenseHostSlotInitialization<'a, S, MlxHybridLayerState>,
        mut paged:Option<super::paged::PagedWork>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        observe:&mut dyn FnMut(&Array)->Result<(),Error>,
    ) -> Result<(FundedDenseHostSlots<'a, S, MlxHybridLayerState, StorageIdentity>,Option<CacheResidencyManager>), Error> {
        if self.is_paged()!=paged.is_some(){return Err(mismatch());}
        let result=(||{
        slots.validate_source(&actual).map_err(other)?;
        let mut manager:Option<CacheResidencyManager>=None;
        if let Some(work)=&mut paged{work.copy_pages(stream,roots,observe,&mut |_|Ok(()),true)?;}
        for i in 0..self.len() {
            let layer = self.layer(i).ok_or_else(mismatch)?;
            let mut child = slots.take_child(i).map_err(other)?;
            let source = match layer.fixed {
                Fixed::Live(s) => s.prepare_slots().map_err(other)?,
                Fixed::Saved(s) => s.prepare_copy_slots().map_err(other)?,
            }
            .for_dense_destination()
            .map_err(other)?;
            child.validate_source(&source).map_err(other)?;
            let attention = self.copy_attention_with_paged(i,&mut paged,stream,roots,observe)?;
            if let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Paged(cache)))=&attention{
                if let Some(manager)=&manager {
                    if !manager.same_catalog(cache.manager()){return Err(mismatch());}
                }else{manager=Some(cache.manager().clone());}
            }
            for j in 0..layer.fixed.len() {
                let value =
                    copy_slot_retained(layer.fixed.slot(j).ok_or_else(mismatch)?, stream, roots)?;
                #[cfg(all(
                    test,
                    target_vendor = "apple",
                    feature = "metal",
                    not(feature = "cuda")
                ))]
                tests::after_fixed_copy()?;
                child.push(value).map_err(|_| mismatch())?;
            }
            let fixed = child.finish().map_err(|_| mismatch())?;
            let key =
                StorageIdentity::HostMetadata(fixed.metadata().identity().registry_key().clone());
            // Publication returns the same actual role table with its attached
            // exact charge, and never a second prompt completion.
            let fixed = match fixed.publish(key) {
                Ok(table) => table,
                Err(error) => {
                    let (owner, cause) = error.into_parts();
                    drop(owner);
                    return Err(other(cause));
                }
            };
            slots
                .push(MlxHybridLayerState {
                    attention,
                    fixed: FixedStateSlots::from_published_slots(fixed),
                    fixed_offset: layer.fixed_offset,
                })
                .map_err(|_| mismatch())?;
        }
        if let Some(work)=&mut paged{work.finish()?;}
        slots.finish().map(|slots|(slots,manager)).map_err(|_| mismatch())
        })();
        match(result,paged){(Err(cause),Some(work))=>Err(work.retain_failure(cause)),(result,_)=>result}
    }
}
impl<'a> PreparedHybridDenseGroup<'a> {
    pub(crate) fn initialization_peak_bytes(&self) -> u64 {
        match self {
            Self::Live(p, _) => p.initialization_peak_bytes(),
            Self::Saved(p, _) => p.initialization_peak_bytes(),
        }
    }
    pub(crate) fn construct(
        self,
        stage: InferencePreparationStage,
        funding: &WorkingMemoryFundingRun,
        complete: WorkingMemoryStorage<StorageIdentity>,
    ) -> Result<(InitializedHybridDenseGroup<'a>, WorkingMemoryFundingScope), Error> {
        match self {
            Self::Live(p,paged) => stage
                .construct_dense_decoder_group(p, funding, complete)
                .map(|(s, n)| (InitializedHybridDenseGroup::Live(s,paged), n))
                .map_err(other),
            Self::Saved(p,paged) => stage
                .construct_dense_decoder_group(p, funding, complete)
                .map(|(s, n)| (InitializedHybridDenseGroup::Saved(s,paged), n))
                .map_err(other),
        }
    }
}
impl PreparedDenseHybridGroupedState<'_> {
    pub(crate) fn slot_metadata(&self) -> &HostSlotMetadata {
        match &self.layers {
            DenseStorage::Live(s) => s.metadata(),
            DenseStorage::Saved(s) => s.metadata(),
        }
    }
    pub(crate) fn visit_operands<'s>(&'s self, visitor: &mut dyn FnMut(&'s Array)) {
        fn visit<'s>(layer: &'s MlxHybridLayerState, visitor: &mut dyn FnMut(&'s Array)) {
            match &layer.attention {
                Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) => {
                    cache.prepare_isolated_copy().visit_operands(visitor)
                }
                Some(MlxHybridAttentionState::Compressed(cache)) => cache
                    .prepare_isolated_copy()
                    .expect("completed copied cache")
                    .visit_operands(visitor),
                _ => {}
            }
            for (_, value) in layer.fixed.iter() {
                if let Some(value) = value {
                    visitor(value.as_array());
                }
            }
        }
        match &self.layers {
            DenseStorage::Live(s) => {
                for i in 0..s.len() {
                    visit(s.get(i).expect("complete outer group"), visitor);
                }
            }
            DenseStorage::Saved(s) => {
                for i in 0..s.len() {
                    visit(s.get(i).expect("complete outer group"), visitor);
                }
            }
        }
    }
}
impl<'a> PreparedDenseHybridGroupedState<'a> {
    pub(crate) fn publish_for_control(
        self,
    ) -> Result<
        (PublishedDenseHybridGroupedState, InferencePromptCompletion),
        HybridGroupedPublishError<'a>,
    > {
        let Self {
            layers,
            layout,
            global_layer_start,
            manager,
        } = self;
        let published = match layers {
            DenseStorage::Live(s) => {
                let key =
                    StorageIdentity::HostMetadata(s.metadata().identity().registry_key().clone());
                match s.publish(key) {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        let (s, e) = e.into_parts();
                        Err((DenseStorage::Live(s), e))
                    }
                }
            }
            DenseStorage::Saved(s) => {
                let key =
                    StorageIdentity::HostMetadata(s.metadata().identity().registry_key().clone());
                match s.publish(key) {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        let (s, e) = e.into_parts();
                        Err((DenseStorage::Saved(s), e))
                    }
                }
            }
        };
        match published {
            Ok((layers, completion)) => Ok((
                PublishedDenseHybridGroupedState {
                    state: MlxHybridState {
                        layers,
                        layout,
                        global_layer_start,
                        manager,
                        inference_retention: InferenceRetention::new(),
                    },
                },
                completion,
            )),
            Err((layers, error)) => Err(HybridGroupedPublishError {
                owner: Self {
                    layers,
                    layout,
                    global_layer_start,
                    manager,
                },
                error,
            }),
        }
    }
}
pub(crate) struct PublishedDenseHybridGroupedState {
    state: MlxHybridState,
}
impl PublishedDenseHybridGroupedState {
    pub(crate) fn into_state(self) -> MlxHybridState {
        self.state
    }
    pub(crate) fn visit_registered_child_metadata(
        &self,
        visitor: &mut dyn FnMut(&HostSlotMetadata) -> Result<(), Error>,
    ) -> Result<(), Error> {
        for layer in self.state.layers.slots() {
            visitor(layer.fixed.metadata())?;
        }
        Ok(())
    }
}
pub(crate) struct HybridGroupedPublishError<'a> {
    owner: PreparedDenseHybridGroupedState<'a>,
    error: HostSlotAttachmentError<WorkingMemoryError>,
}
impl<'a> HybridGroupedPublishError<'a> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        PreparedDenseHybridGroupedState<'a>,
        HostSlotAttachmentError<WorkingMemoryError>,
    ) {
        (self.owner, self.error)
    }
}
impl fmt::Debug for HybridGroupedPublishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl fmt::Display for HybridGroupedPublishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for HybridGroupedPublishError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl super::super::super::key_value::PagedSnapshotState for PreparedDenseHybridGroupedState<'_> {
    fn layout(&self) -> &SharedStateLayout {
        &self.layout
    }
    fn layer_count(&self)->usize {match &self.layers{DenseStorage::Live(s)=>s.len(),DenseStorage::Saved(s)=>s.len()}}
    fn global_start(&self)->usize {self.global_layer_start}
    fn layer(&self,index:usize)->Option<super::super::super::key_value::PagedSnapshotLayer<'_>>{
        let layer=match &self.layers{DenseStorage::Live(s)=>s.get(index),DenseStorage::Saved(s)=>s.get(index)}?;
        Some(super::paged::attention(layer.attention.as_ref()))
    }
}
impl PreparedDenseHybridGroupedState<'_> {
    pub(crate) fn visit_complete_operands(&self,visitor:&mut dyn FnMut(&Array))
        ->Result<(),super::super::super::SnapshotProjectionCause>{
        use super::super::super::{SnapshotArraySources,SnapshotProjectionCause};
        if let Some(manager)=&self.manager {
            let source=super::super::super::key_value::PagedSnapshotSource::prepare(self)?
                .ok_or(SnapshotProjectionCause::Changed)?;
            if !manager.same_catalog(source.manager()){return Err(SnapshotProjectionCause::Changed);}
            source.visit_arrays(&mut|array|{visitor(array);Ok(())})?;
        }
        self.visit_operands(visitor);
        Ok(())
    }
}
