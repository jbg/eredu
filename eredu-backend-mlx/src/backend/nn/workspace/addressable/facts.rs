//! One source-aware fact adapter for retained indexed equations and placements.
use super::*;
use std::marker::PhantomData;

/// A retained source supplies identity and physical facts. Missing finite
/// contributors remain absent in both ordinary and original fact collection.
pub(crate) trait AddressableFactQuote {
    fn matches(&self, operation: WorkspaceOperationView<'_>) -> bool;
    fn outputs(&self) -> &[WorkspaceLayout];
    fn allocation(&self) -> NativeAllocationFacts;
    fn storage_bytes(&self) -> Option<u64>;
    fn host_bytes(&self) -> Option<u64>;
    /// An outer None keeps the original adapter's existing worker. Ordinary
    /// quotes override it; an inner None preserves missing source evidence.
    fn scratch_host_controls(&self) -> Option<Option<u64>> {
        None
    }
    fn host_basis(&self) -> &'static str;
}
pub(crate) trait AddressableFactSource: std::fmt::Debug {
    type Quote: AddressableFactQuote;
    fn funding(&self) -> &HostMetadataFunding;
    fn quote(&self, operation: WorkspaceOperationView<'_>) -> Result<Self::Quote, Error>;
    fn observation_layout(
        &self,
        source: WorkspaceAddressableRegionView<'_>,
        inputs: &[WorkspaceLayout],
        context: &WorkspaceContext,
    ) -> Result<WorkspaceAddressableObservationLayout, Error>;
}
impl AddressableFactQuote for AddressableQuoteRef {
    fn matches(&self, operation: WorkspaceOperationView<'_>) -> bool {
        AddressableQuote::matches(self, operation)
    }
    fn outputs(&self) -> &[WorkspaceLayout] {
        &self.outputs
    }
    fn allocation(&self) -> NativeAllocationFacts {
        AddressableQuote::allocation(self)
    }
    fn storage_bytes(&self) -> Option<u64> {
        Some(self.numerical.storage.mutable_bytes())
    }
    fn host_bytes(&self) -> Option<u64> {
        Some(self.host_bytes)
    }
    fn host_basis(&self) -> &'static str {
        "original indexed discovery/remap and ordinary equation host payloads; accepted constructor/read and native role arenas remain separate sources"
    }
}
impl AddressableFactSource for AddressableSources {
    type Quote = AddressableQuoteRef;
    fn funding(&self) -> &HostMetadataFunding {
        AddressableSources::funding(self)
    }
    fn quote(&self, operation: WorkspaceOperationView<'_>) -> Result<Self::Quote, Error> {
        AddressableSources::quote(self, operation)
    }
    fn observation_layout(
        &self,
        source: WorkspaceAddressableRegionView<'_>,
        inputs: &[WorkspaceLayout],
        context: &WorkspaceContext,
    ) -> Result<WorkspaceAddressableObservationLayout, Error> {
        AddressableSources::observation_layout(self, source, inputs, context)
    }
}

#[derive(Debug)]
pub(crate) struct MlxAddressableWorkspaceMechanisms<M, S = AddressableSources> {
    base: M,
    source: S,
}
impl<M, S> MlxAddressableWorkspaceMechanisms<M, S> {
    pub(crate) fn new(base: M, source: S) -> Self {
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
    #[error("addressable facts require their retained metadata account")]
    Funding,
}
fn is_region(operation: WorkspaceOperationView<'_>) -> bool {
    matches!(
        operation.kind,
        WorkspaceOperationKindView::AddressableRegion(_)
    )
}
impl<M: WorkspaceMechanisms, S: AddressableFactSource> WorkspaceMechanisms
    for MlxAddressableWorkspaceMechanisms<M, S>
{
    fn prepare_allocation_sources(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceOperationAllocationSources>, Error> {
        if is_region(operation) {
            Ok(None)
        } else {
            self.base.prepare_allocation_sources(operation, context)
        }
    }

    fn scratch_allocation_count(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<usize>, Error> {
        self.base.scratch_allocation_count(operation)
    }

    fn completion_strategy(&self) -> WorkspaceCompletionStrategy {
        self.base.completion_strategy()
    }
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        WorkspaceMechanisms::memory_topology(&self.base)
    }
    fn allocation_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<u64> {
        WorkspaceMechanisms::allocation_host_control_bytes(&self.base, op, output)
    }
    fn scratch_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        if is_region(op) {
            if let Some(controls) = self.source.quote(op)?.scratch_host_controls() {
                return Ok(controls);
            }
        }
        WorkspaceMechanisms::scratch_host_control_bytes(&self.base, op)
    }

    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        WorkspaceMechanisms::output_placement(&self.base, operation, output)
    }
    fn scratch_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        WorkspaceMechanisms::scratch_placement(&self.base, operation)
    }
    fn addressable_observation_layout(
        &self,
        source: WorkspaceAddressableRegionView<'_>,
        inputs: &[WorkspaceLayout],
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceAddressableObservationLayout>, Error> {
        self.source
            .observation_layout(source, inputs, context)
            .map(Some)
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
                .outputs()
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
impl<M: WorkspaceFactMechanisms, S: AddressableFactSource> WorkspaceFactMechanisms
    for MlxAddressableWorkspaceMechanisms<M, S>
{
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        WorkspaceFactMechanisms::memory_topology(&self.base)
    }
    fn allocation_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<u64> {
        WorkspaceFactMechanisms::allocation_host_control_bytes(&self.base, op, output)
    }
    fn scratch_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Self::Error> {
        if is_region(op) {
            let quote = self
                .source
                .quote(op)
                .map_err(AddressableFactError::Source)?;
            if let Some(controls) = quote.scratch_host_controls() {
                return Ok(controls);
            }
        }
        WorkspaceFactMechanisms::scratch_host_control_bytes(&self.base, op)
            .map_err(AddressableFactError::Base)
    }

    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        WorkspaceFactMechanisms::output_placement(&self.base, operation, output)
    }
    fn scratch_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        WorkspaceFactMechanisms::scratch_placement(&self.base, operation)
    }
    type Error = AddressableFactError<M::Error>;
    fn with_prepared_facts_context<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        if is_region(operation) {
            self.with_prepared_facts(operation, context.metadata_funding().as_ref(), visit)
        } else {
            let controls = size_of::<(
                Delegated<'_, M::Error>,
                &Self,
                &WorkspaceContext,
                WorkspaceOperationView<'_>,
                Result<T, Self::Error>,
            )>()
            .checked_add(size_of_val(&visit))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            context
                .charge_metadata(controls)
                .map_err(|cause| AddressableFactError::Source(cause.into()))?;
            self.base
                .with_prepared_facts_context(operation, context, |base| visit(&Delegated(base)))
                .map_err(AddressableFactError::Base)
        }
    }
    fn with_prepared_facts<T>(
        &self,
        op: WorkspaceOperationView<'_>,
        funding: Option<&HostMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        if is_region(op) {
            if funding.is_none_or(|funding| !funding.same_account(self.source.funding())) {
                return Err(AddressableFactError::Funding);
            }
            self.source
                .funding()
                .reserve_metadata(
                    size_of::<(Prepared<M::Error, S::Quote>, Result<T, Self::Error>)>()
                        + size_of_val(&visit),
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
    fn scratch_allocations(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<&eredu_nn::workspace::WorkspaceAllocationPopulation>, Self::Error> {
        self.0
            .scratch_allocations(op)
            .map_err(AddressableFactError::Base)
    }
    fn output_allocations(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Result<Option<&eredu_nn::workspace::WorkspaceAllocationPopulation>, Self::Error> {
        self.0
            .output_allocations(op, output)
            .map_err(AddressableFactError::Base)
    }

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
struct Prepared<E, Q> {
    quote: Q,
    _base: PhantomData<fn() -> E>,
}
impl<E, Q: AddressableFactQuote> Prepared<E, Q> {
    fn validate(&self, op: WorkspaceOperationView<'_>) -> Result<(), AddressableFactError<E>> {
        if !self.quote.matches(op) {
            return Err(MlxWorkspaceFactError::descriptor(
                "addressable fact occurrence differs from its retained bank source",
            )
            .into());
        }
        Ok(())
    }
}
fn emit(
    quote: &impl AddressableFactQuote,
    sink: &mut super::super::facts::Emitter<'_>,
) -> super::super::facts::FactResult<Option<WorkspaceOperationFacts>> {
    let Some(storage_bytes) = quote.storage_bytes() else {
        return Ok(None);
    };
    let mut outputs = 0u64;
    for layout in quote.outputs() {
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
    let scratch = storage_bytes.checked_sub(outputs).ok_or_else(|| {
        MlxWorkspaceFactError::descriptor(
            "addressable complete output backing exceeds its source population",
        )
    })?;
    sink.finish(
        scratch,
        format_args!("actual indexed bank rows, compact/grouped equations and parent assembly"),
    )
    .map(Some)
}
fn host(
    quote: &impl AddressableFactQuote,
    sink: &mut super::super::facts::HostEmitter<'_>,
) -> super::super::facts::FactResult<Option<WorkspaceHostFacts>> {
    let Some(bytes) = quote.host_bytes() else {
        return Ok(None);
    };
    sink.finish(bytes, format_args!("{}", quote.host_basis()))
        .map(Some)
}
impl<E, Q: AddressableFactQuote> WorkspaceFactMechanisms for Prepared<E, Q> {
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
