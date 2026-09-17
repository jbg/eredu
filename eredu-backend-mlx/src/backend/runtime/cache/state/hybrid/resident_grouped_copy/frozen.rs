//! One joined account for the actual outer and ordered fixed child tables.
use super::*;
use eredu_runtime::working_memory::{
    AdmittedWorkspaceCopy, FundedSamplerCopy, InitializedDecoderTableGroup,
    RegisteredDecoderHostCopy, RegisteredDecoderTableGroup, RegisteredSamplingCopy,
    WorkingMemoryStorage, WorkspaceCopyLimits,
};

type LiveGroup<'a> = RegisteredDecoderTableGroup<
    'a,
    MlxHybridLayerState,
    SavedHybridGroupedLayer,
    Slot,
    Slot,
    StorageIdentity,
>;
type SavedGroup<'a> = RegisteredDecoderTableGroup<
    'a,
    SavedHybridGroupedLayer,
    SavedHybridGroupedLayer,
    Slot,
    Slot,
    StorageIdentity,
>;
pub(crate) enum PreparedHybridGroupHostCopy<'a> {
    Live(LiveGroup<'a>, Option<super::paged::PagedWork>),
    Saved(SavedGroup<'a>, Option<super::paged::PagedWork>),
}
pub(crate) struct InitializedHybridGroupCopy {
    slots: InitializedDecoderTableGroup<SavedHybridGroupedLayer, Slot>,
    paged: Option<super::paged::PagedWork>,
}

impl<'a> PreparedHybridGroupedCopy<'a> {
    /// The exact same outer destination and child slot extents, without binding
    /// or constructing a child-plan vector just to inspect their aggregate.
    pub(crate) fn host_copy_initialization_peak_bytes(
        &self,
    ) -> Result<u64, ResidentDecoderPreparationError> {
        self.validate_fixed()?;
        let mut bytes = match self.source {
            Source::Live(source) => source
                .layers
                .prepare_copy_slots()?
                .for_destination::<SavedHybridGroupedLayer>()?
                .initialization_peak_bytes(),
            Source::Saved(source) => source
                .layers
                .prepare_copy_slots()?
                .initialization_peak_bytes(),
        };
        for i in 0..self.len() {
            let child = match self
                .layer(i)
                .ok_or(WorkingMemoryError::IdentityMismatch)?
                .fixed
            {
                Fixed::Live(source) => source.prepare_slots()?,
                Fixed::Saved(source) => source.prepare_copy_slots()?,
            };
            bytes = bytes
                .checked_add(child.initialization_peak_bytes())
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        Ok(bytes)
    }

    pub(crate) fn host_copy_preparation_bytes(
        &self,
    ) -> Result<usize, eredu_runtime::working_memory::DecoderHostPreparationError> {
        use eredu_runtime::working_memory::DecoderHostPreparationError as E;
        let mut bytes = match self.source {
            Source::Live(_) => {
                let outer = RegisteredDecoderHostCopy::<
                    MlxHybridLayerState,
                    StorageIdentity,
                    SavedHybridGroupedLayer,
                >::preparation_control_bytes(true)?;
                outer
                    .checked_add(LiveGroup::preparation_control_bytes(self.len())?)
                    .ok_or(E::Overflow)?
            }
            Source::Saved(_) => {
                let outer = RegisteredDecoderHostCopy::<SavedHybridGroupedLayer, StorageIdentity>::preparation_control_bytes(false)?;
                outer
                    .checked_add(SavedGroup::preparation_control_bytes(self.len())?)
                    .ok_or(E::Overflow)?
            }
        };
        for i in 0..self.len() {
            let layer = self.layer(i).ok_or(E::UnknownBound)?;
            let child =
                RegisteredDecoderHostCopy::<Slot, StorageIdentity>::preparation_control_bytes(
                    matches!(layer.fixed, Fixed::Live(_)),
                )?;
            bytes = bytes.checked_add(child).ok_or(E::Overflow)?;
        }
        if let Some(paged)=&self.paged {
            bytes=bytes.checked_add(paged.host_preparation_bytes(self.paged_caller_controls().ok_or(E::Overflow)?)
                .ok_or(E::Overflow)?).ok_or(E::Overflow)?;
        }
        bytes
            .checked_add(std::mem::size_of::<PreparedHybridGroupHostCopy<'_>>())
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<
                    Result<PreparedHybridGroupHostCopy<'_>, Error>,
                >())
            })
            .ok_or(E::Overflow)
    }

    pub(crate) fn host_copy(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<PreparedHybridGroupHostCopy<'a>, Error> {
        self.host_copy_with_preparation(pool, None)
    }

    pub(crate) fn host_copy_with_preparation(
        &self,
        pool: &WorkingMemoryPool,
        preparation: Option<&eredu_core::HostPreparationAuthority>,
    ) -> Result<PreparedHybridGroupHostCopy<'a>, Error> {
        self.validate()?;
        // Plan descriptors are bookkeeping; actual child inline payload and all
        // value temporaries are checked/priced by each closed initializer.
        let mut children = Vec::new();
        children.try_reserve_exact(self.len()).map_err(other)?;
        if preparation.is_some() && children.capacity() != self.len() {
            return Err(mismatch());
        }
        for i in 0..self.len() {
            let plan = match self.layer(i).ok_or_else(mismatch)?.fixed {
                Fixed::Live(s) => RegisteredDecoderHostCopy::bind_with_preparation(
                    pool,
                    s.prepare_slots().map_err(other)?,
                    StorageIdentity::HostMetadata(s.metadata().identity().registry_key().clone()),
                    preparation,
                )
                .map_err(other)?,
                Fixed::Saved(s) => s.prepare_copy().map_err(other)?,
            };
            children.push(plan);
        }
        let paged = self.prepare_paged_work(preparation)?;
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
                .for_destination::<SavedHybridGroupedLayer>()
                .map_err(other)?;
                PreparedHybridGroupHostCopy::Live(
                    RegisteredDecoderTableGroup::new_with_preparation(outer, children, preparation)
                        .map_err(other)?,
                    paged,
                )
            }
            Source::Saved(s) => PreparedHybridGroupHostCopy::Saved(
                RegisteredDecoderTableGroup::new_with_preparation(
                    s.layers.prepare_copy().map_err(other)?,
                    children,
                    preparation,
                )
                .map_err(other)?,
                paged,
            ),
        })
    }
    pub(crate) fn copy_retained(
        self,
        initialized: InitializedHybridGroupCopy,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<SavedHybridGroupedCopy, Error> {
        self.copy_retained_with(initialized,stream,roots,&mut |_|Ok(()),&mut |_|Ok(()))
    }
    pub(crate) fn copy_retained_with(
        self, initialized:InitializedHybridGroupCopy, stream:&Stream, roots:&RefCell<Vec<Array>>,
        observe:&mut dyn FnMut(&Array)->Result<(),Error>,
        observe_host:&mut dyn FnMut(&crate::backend::array_copy::PreparedSavedHostCopy)->Result<(),Error>,
    )->Result<SavedHybridGroupedCopy,Error>{
        self.validate()?;
        let InitializedHybridGroupCopy{mut slots,mut paged}=initialized;
        if self.is_paged()!=paged.is_some(){return Err(mismatch());}
        let result=(||{
        match self.source {
            Source::Live(s) => slots
                .validate_source(
                    &s.layers
                        .prepare_copy_slots()
                        .map_err(other)?
                        .for_destination::<SavedHybridGroupedLayer>()
                        .map_err(other)?,
                )
                .map_err(other)?,
            Source::Saved(s) => slots
                .validate_source(&s.layers.prepare_copy_slots().map_err(other)?)
                .map_err(other)?,
        }
        if let Some(work)=&mut paged{work.copy_pages(stream,roots,observe,observe_host,false)?;}
        for i in 0..self.len() {
            let layer = self.layer(i).ok_or_else(mismatch)?;
            let mut child = slots.take_child(i).map_err(other)?;
            let actual = match layer.fixed {
                Fixed::Live(s) => s.prepare_slots().map_err(other)?,
                Fixed::Saved(s) => s.prepare_copy_slots().map_err(other)?,
            };
            child.validate_source(&actual).map_err(other)?;
            let attention = self.copy_attention_with_paged(i,&mut paged,stream,roots,observe)?;
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
            slots
                .push(SavedHybridGroupedLayer {
                    attention,
                    fixed,
                    fixed_offset: layer.fixed_offset,
                })
                .map_err(|_| mismatch())?;
        }
        if let Some(work)=&mut paged{work.finish()?;}
        let retained = slots.retained_bytes();
        let protected = slots.protected_bytes();
        let layers = slots.finish().map_err(|_| mismatch())?;
        Ok(SavedHybridGroupedCopy {
            layers,
            layout: self.shared_layout().clone(),
            global_layer_start: self.global_layer_start(),
            retained,
            protected,
        })
        })();
        match (result,paged){(Err(cause),Some(work))=>Err(work.retain_failure(cause)),(result,_)=>result}
    }
}
impl InitializedHybridGroupCopy {
    #[cfg(test)]
    pub(super) fn retained_bytes(&self) -> u64 { self.slots.retained_bytes() }
    #[cfg(test)]
    pub(super) fn protected_bytes(&self) -> u64 { self.slots.protected_bytes() }

    pub(crate) fn prepare_host_destinations(&mut self,
        copy:&mut crate::backend::array_copy::PreparedOriginalCopy,
        environment:&crate::backend::OriginalCopyEnvironment<'_>)->Result<(),Error>{
        self.paged.as_mut().map_or(Ok(()),|work|work.prepare_host_destinations(copy,environment))
    }
}
impl<'a> PreparedHybridGroupHostCopy<'a> {
    pub(crate) fn initialization_peak_bytes(&self) -> u64 {
        match self {
            Self::Live(p, _) => p.initialization_peak_bytes(),
            Self::Saved(p, _) => p.initialization_peak_bytes(),
        }
    }
    pub(crate) fn admit(
        self,
        pool: &WorkingMemoryPool,
        sampling: RegisteredSamplingCopy<'a, StorageIdentity>,
        complete: WorkingMemoryStorage<StorageIdentity>,
        limits: WorkspaceCopyLimits,
    ) -> Result<
        (
            FundedSamplerCopy,
            InitializedHybridGroupCopy,
            AdmittedWorkspaceCopy,
        ),
        Error,
    > {
        match self {
            Self::Live(p,paged) => pool
                .copy_text_components_group(
                    sampling.with_decoder_group(p, complete).map_err(other)?,
                    limits,
                )
                .map(|(sampler,slots,native)|(sampler,InitializedHybridGroupCopy{slots,paged},native))
                .map_err(other),
            Self::Saved(p,paged) => pool
                .copy_text_components_group(
                    sampling.with_decoder_group(p, complete).map_err(other)?,
                    limits,
                )
                .map(|(sampler,slots,native)|(sampler,InitializedHybridGroupCopy{slots,paged},native))
                .map_err(other),
        }
    }
}
