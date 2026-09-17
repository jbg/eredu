//! A source-aware outer wrapper; ordinary and parallel facts stay unchanged.
use super::*;
use std::marker::PhantomData;

#[derive(Debug)]
pub(crate) struct MlxAddressableWorkspaceMechanisms<M> {
    base: M,
    source: AddressableSources,
}
impl<M> MlxAddressableWorkspaceMechanisms<M> {
    pub(crate) fn new(base: M, source: AddressableSources) -> Self {
        Self { base, source }
    }
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum AddressableFactError<E> {
    #[error("ordinary addressable parent fact failed: {0}")]
    Base(#[source] E),
    #[error("addressable retained source failed")]
    Source(#[source] Error),
    #[error(transparent)]
    Fixed(#[from] MlxWorkspaceFactError),
    #[error("addressable facts require the original metadata account")]
    Funding,
}
fn is_region(operation: WorkspaceOperationView<'_>) -> bool {
    matches!(
        operation.kind,
        WorkspaceOperationKindView::AddressableRegion(_)
    )
}
impl<M: WorkspaceMechanisms> WorkspaceMechanisms for MlxAddressableWorkspaceMechanisms<M> {
    fn addressable_observation_layout(&self,source:WorkspaceAddressableRegionView<'_>,inputs:&[WorkspaceLayout],
        context:&WorkspaceContext)->Result<Option<WorkspaceAddressableObservationLayout>,Error>{
        self.source.observation_layout(source,inputs,context).map(Some)
    }

    fn prepared_text_input_dtype(&self) -> Option<WorkspaceDtype> {
        self.base.prepared_text_input_dtype()
    }
    fn output_representation(
        &self,
        op: WorkspaceOperationView<'_>,
        index: usize,
    ) -> Option<WorkspaceRepresentation> {
        if is_region(op) {
            self.source
                .quote(op)
                .ok()?
                .outputs
                .get(index)?
                .representation()
        } else {
            self.base.output_representation(op, index)
        }
    }
    fn projection_input_observation_mechanism(
        &self,
        format: &eredu_nn::LinearFormatSpec,
    ) -> Result<Option<eredu_nn::ProjectionInputObservationMechanism>, Error> {
        self.base.projection_input_observation_mechanism(format)
    }
    fn grouped_observation_schedule(
        &self,
        bank: &WorkspaceGroupedBank,
        tokens: u32,
    ) -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        self.base.grouped_observation_schedule(bank, tokens)
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if is_region(op.as_view()) {
            let quote = self.source.quote(op.as_view())?;
            super::super::facts::ordinary(|sink| emit(&quote, sink))
        } else {
            self.base.operation_bound(op)
        }
    }
    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        if is_region(op.as_view()) {
            let quote = self.source.quote(op.as_view())?;
            super::super::facts::ordinary_host(|sink| host(&quote, sink))
        } else {
            self.base.host_workspace_bound(op)
        }
    }
}
impl<M: WorkspaceFactMechanisms> WorkspaceFactMechanisms for MlxAddressableWorkspaceMechanisms<M> {
    type Error = AddressableFactError<M::Error>;
    fn with_prepared_facts<T>(
        &self,
        op: WorkspaceOperationView<'_>,
        funding: Option<&WorkspaceMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        if is_region(op) {
            if funding.is_none_or(|funding| !funding.same_account(self.source.funding())) {
                return Err(AddressableFactError::Funding);
            }
            self.source
                .funding()
                .reserve_metadata(
                    size_of::<(Prepared<M::Error>, Result<T, Self::Error>)>() + size_of_val(&visit),
                )
                .map_err(|cause| {
                    AddressableFactError::Source(WorkspaceMetadataError::Funding(cause).into())
                })?;
            let quote = self
                .source
                .quote(op)
                .map_err(AddressableFactError::Source)?;
            Ok(visit(&Prepared {
                quote,
                _base: PhantomData,
            }))
        } else {
            self.base
                .with_prepared_facts(op, funding, |base| visit(&Delegated(base)))
                .map_err(AddressableFactError::Base)
        }
    }
    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        if is_region(op) {
            Ok(None)
        } else {
            self.base
                .operation_facts(op)
                .map_err(AddressableFactError::Base)
        }
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        if is_region(op) {
            Ok(None)
        } else {
            self.base
                .write_operation_facts(op, destination)
                .map_err(AddressableFactError::Base)
        }
    }
    fn host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        if is_region(op) {
            Ok(None)
        } else {
            self.base.host_facts(op).map_err(AddressableFactError::Base)
        }
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        if is_region(op) {
            Ok(None)
        } else {
            self.base
                .write_host_facts(op, destination)
                .map_err(AddressableFactError::Base)
        }
    }
}
struct Delegated<'a, E>(&'a dyn WorkspaceFactMechanisms<Error = E>);
impl<E> WorkspaceFactMechanisms for Delegated<'_, E> {
    type Error = AddressableFactError<E>;
    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.0
            .operation_facts(op)
            .map_err(AddressableFactError::Base)
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.0
            .write_operation_facts(op, out)
            .map_err(AddressableFactError::Base)
    }
    fn host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.0.host_facts(op).map_err(AddressableFactError::Base)
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.0
            .write_host_facts(op, out)
            .map_err(AddressableFactError::Base)
    }
}
struct Prepared<E> {
    quote: AddressableQuoteRef,
    _base: PhantomData<fn() -> E>,
}
impl<E> Prepared<E> {
    fn validate(&self, op: WorkspaceOperationView<'_>) -> Result<(), AddressableFactError<E>> {
        if !self.quote.matches(op) {
            return Err(MlxWorkspaceFactError::descriptor(
                "addressable fact occurrence differs from original bank source",
            )
            .into());
        }
        Ok(())
    }
}
fn emit(
    quote: &AddressableQuote,
    sink: &mut super::super::facts::Emitter<'_>,
) -> super::super::facts::FactResult<Option<WorkspaceOperationFacts>> {
    let mut outputs = 0u64;
    for layout in &quote.outputs {
        let dtype =
            super::super::byte_view::Dtype::from_layout(layout.as_view()).ok_or_else(|| {
                MlxWorkspaceFactError::descriptor("addressable result has no scalar source")
            })?;
        let bytes = layout
            .as_view()
            .elements()?
            .checked_mul(dtype.bytes())
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        let bytes = quote.allocation().fixed_buffer_capacity(bytes)?;
        outputs = outputs
            .checked_add(bytes)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        sink.output(super::super::facts::Output::Allocate(bytes))?;
    }
    let scratch = quote
        .numerical
        .storage
        .mutable_bytes()
        .checked_sub(outputs)
        .ok_or_else(|| {
            MlxWorkspaceFactError::descriptor(
                "addressable complete output backing exceeds its source population",
            )
        })?;
    sink.finish(
        scratch,
        format_args!(
            "actual indexed bank rows, compact/grouped branches and shared parent numerical role"
        ),
    )
    .map(Some)
}
fn host(
    quote: &AddressableQuote,
    sink: &mut super::super::facts::HostEmitter<'_>,
) -> super::super::facts::FactResult<Option<WorkspaceHostFacts>> {
    sink.finish(quote.host_bytes,format_args!("original indexed discovery/remap and ordinary equation host payloads; accepted constructor/read and native role arenas remain separate sources")).map(Some)
}
impl<E> WorkspaceFactMechanisms for Prepared<E> {
    type Error = AddressableFactError<E>;
    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.validate(op)?;
        emit(&self.quote, &mut super::super::facts::Emitter::count()).map_err(Into::into)
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.validate(op)?;
        super::super::facts::write(|sink| emit(&self.quote, sink), out).map_err(Into::into)
    }
    fn host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.validate(op)?;
        host(&self.quote, &mut super::super::facts::HostEmitter::count()).map_err(Into::into)
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.validate(op)?;
        super::super::facts::write_host(|sink| host(&self.quote, sink), out).map_err(Into::into)
    }
}
