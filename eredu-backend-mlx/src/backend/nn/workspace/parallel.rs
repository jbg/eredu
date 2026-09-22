//! Retained native source preparation followed by the shared pure fact emitter.
use super::*;
mod boundary;
mod expert_region;
pub(crate) use expert_region::{
    ExpertCountQuote, ExpertInactiveWaveQuote, ExpertLocalObservationSource, ExpertLocalQuote,
    ExpertLocalStage, ExpertLocalStageBound, ExpertMovementKind, ExpertProviderQuote,
    ExpertProviderWaveQuote, ExpertRegionAggregate, ExpertReorderEnvelope, ExpertTransferProfile,
    ExpertTransportQuote, GroupedSourceObserver,
};
mod logical;
pub(super) mod ordinary;
pub(crate) use ordinary::OrdinaryParallelControls;
mod representation;
mod scratch;
use crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelSource;
pub(crate) use boundary::{BoundaryStageCapacity, PipelineBoundaryQuote};
pub(crate) use logical::{LogicalCollectiveKind, LogicalCollectiveQuote};

#[derive(Debug)]
pub(crate) struct MlxParallelWorkspaceMechanisms {
    ordinary: ResidentExecutionMechanisms,
    source: OriginalParallelSource,
    metadata_funding: HostMetadataFunding,
    addressable: Option<AddressableSources>,
    ordinary_addressable: Option<OrdinaryAddressableSources>,
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum ParallelFactError {
    #[error(transparent)]
    Fixed(#[from] MlxWorkspaceFactError),
    #[error(transparent)]
    Resident(#[from] super::resident_mechanism::ResidentFactError),
    #[error("selected parallel source preparation failed")]
    Source(#[source] crate::backend::error::Error),
    #[error("prepared pipeline frame source failed")]
    Boundary(#[source] Error),
    #[error("parallel source preparation requires its original funding")]
    Unfunded,
}
impl MlxParallelWorkspaceMechanisms {
    fn local_quote(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<ExpertLocalQuote, Error> {
        if self.ordinary.allocation().original_storage {
            ExpertLocalQuote::prepare(
                &self.source,
                operation,
                self.ordinary,
                self.addressable.as_ref(),
            )
        } else {
            ExpertLocalQuote::prepare_ordinary(
                &self.source,
                operation,
                self.ordinary,
                self.ordinary_addressable.as_ref(),
            )
        }
    }
    pub(crate) fn new(
        ordinary: ResidentExecutionMechanisms,
        source: OriginalParallelSource,
    ) -> Self {
        Self {
            ordinary,
            metadata_funding: source.funding().clone(),
            source,
            addressable: None,
            ordinary_addressable: None,
        }
    }
    pub(crate) fn prepare_workspace(self) -> Result<MlxParallelWorkspace, Error> {
        self.prepare_workspace_source(None, None)
    }
    pub(crate) fn prepare_workspace_with_addressable(
        self,
        addressable: AddressableSources,
    ) -> Result<MlxParallelWorkspace, Error> {
        self.prepare_workspace_source(Some(addressable), None)
    }
    pub(crate) fn prepare_workspace_with_ordinary_addressable(
        self,
        addressable: OrdinaryAddressableSources,
    ) -> Result<MlxParallelWorkspace, Error> {
        self.prepare_workspace_source(None, Some(addressable))
    }
    /// The phase owns trace metadata and addressable descriptors; the retained
    /// parallel source keeps its original native quotation and execution payer.
    pub(crate) fn prepare_workspace_with_metadata_funding(
        mut self,
        funding: HostMetadataFunding,
        addressable: Option<AddressableSources>,
    ) -> Result<MlxParallelWorkspace, Error> {
        self.metadata_funding = funding;
        self.prepare_workspace_source(addressable, None)
    }
    fn prepare_workspace_source(
        mut self,
        addressable: Option<AddressableSources>,
        ordinary_addressable: Option<OrdinaryAddressableSources>,
    ) -> Result<MlxParallelWorkspace, Error> {
        self.addressable = addressable.clone();
        self.ordinary_addressable = ordinary_addressable.clone();
        let source = self.source.clone();
        let ordinary = self.ordinary;
        self.metadata_funding
            .reserve_metadata(
                std::mem::size_of::<MlxParallelWorkspace>()
                    + std::mem::size_of::<Result<MlxParallelWorkspace, Error>>(),
            )
            .map_err(|cause| Error::from(WorkspaceMetadataError::Funding(cause)))?;
        let funding = self.metadata_funding.clone();
        if addressable
            .as_ref()
            .is_some_and(|value| !value.funding().same_account(&funding))
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        if ordinary_addressable
            .as_ref()
            .is_some_and(|value| !value.funding().same_account(&funding))
        {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let context = match (&addressable, &ordinary_addressable) {
            (Some(value), None) => WorkspaceContext::new_with_metadata_funding(
                MlxAddressableWorkspaceMechanisms::new(self, value.clone()),
                funding,
            )?,
            (None, Some(value)) => WorkspaceContext::new_with_metadata_funding(
                MlxAddressableWorkspaceMechanisms::new(self, value.clone()),
                funding,
            )?,
            (None, None) => WorkspaceContext::new_with_metadata_funding(self, funding)?,
            (Some(_), Some(_)) => return Err(WorkspaceMetadataError::Unqualified.into()),
        };
        Ok(MlxParallelWorkspace {
            context,
            ordinary,
            source,
            addressable,
            ordinary_addressable,
        })
    }
}
/// The context and recorder are minted from one exact retained source. A
/// caller cannot substitute another native Group between fact emission and
/// recipe reduction merely because its scalar geometry happens to match.
pub(crate) struct MlxParallelWorkspace {
    context: WorkspaceContext,
    ordinary: ResidentExecutionMechanisms,
    source: OriginalParallelSource,
    addressable: Option<AddressableSources>,
    ordinary_addressable: Option<OrdinaryAddressableSources>,
}
impl MlxParallelWorkspace {
    pub(crate) fn declaration_source(&self) -> &eredu_runtime::RetainedCommunicationSource {
        self.source.declaration_source()
    }
    pub(crate) fn addressable_sources(&self) -> Option<&AddressableSources> {
        self.addressable.as_ref()
    }
    /// One actual frame trace retains its finite native collective occurrences.
    /// This is descriptive preparation; no native invocation is entered here.
    pub(crate) fn prepare_invocation(&self,report:&WorkspaceTraceReport)
    ->Result<crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation,Error>{
        self.source
            .prepare_invocation_with_sources(
                &report.operations,
                Some(self.ordinary),
                self.addressable.as_ref(),
                self.ordinary_addressable.as_ref(),
            )
            .map_err(|cause| self.source.neural_error(cause))
    }
    pub(crate) fn context(&self) -> &WorkspaceContext {
        &self.context
    }
    pub(crate) fn into_context(self) -> WorkspaceContext {
        self.context
    }
    pub(crate) fn recorder(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<resident_recipe::ParallelRecipeRecorder, Error> {
        let mut recorder = resident_recipe::ParallelRecipeRecorder::new(
            geometry,
            self.ordinary,
            &self.context,
            &self.source,
        )?;
        if let Some(source) = &self.addressable {
            recorder.bind_addressable_sources(source.clone())?;
        }
        if let Some(source) = &self.ordinary_addressable {
            recorder.bind_ordinary_addressable_sources(source.clone())?;
        }
        Ok(recorder)
    }
}
impl WorkspaceMechanisms for MlxParallelWorkspaceMechanisms {
    fn prepare_allocation_sources(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceOperationAllocationSources>, Error> {
        self.ordinary.prepare_allocation_sources(operation, context)
    }

    fn scratch_allocation_count(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<usize>, Error> {
        self.ordinary.scratch_allocation_count(operation)
    }

    fn completion_strategy(&self) -> WorkspaceCompletionStrategy {
        if self.ordinary.allocation().original_storage {
            WorkspaceCompletionStrategy::EnclosingSubmission
        } else {
            WorkspaceCompletionStrategy::OperationSubmissions
        }
    }
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        WorkspaceMechanisms::memory_topology(&self.ordinary)
    }
    fn allocation_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<u64> {
        WorkspaceMechanisms::allocation_host_control_bytes(&self.ordinary, op, output)
    }
    fn scratch_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        self.scratch_controls(op)
    }

    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        WorkspaceMechanisms::output_placement(&self.ordinary, operation, output)
    }
    fn scratch_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        WorkspaceMechanisms::scratch_placement(&self.ordinary, operation)
    }
    fn prepared_text_input_dtype(&self) -> Option<WorkspaceDtype> {
        self.ordinary.prepared_text_input_dtype()
    }

    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        if matches!(operation.kind, WorkspaceOperationKindView::ExpertRegion(_)) {
            let quote = self.local_quote(operation).ok()?;
            let dtype = quote.outputs.get(output)?.representation()?.dtype();
            // Final typed zeros/ScatterAxis writes complete contiguous rows.
            return Some(WorkspaceRepresentation::new(dtype, true));
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::Collective(
                WorkspaceCollectiveView::Sum { .. }
                    | WorkspaceCollectiveView::GatherFirstAxis { .. }
                    | WorkspaceCollectiveView::Broadcast { .. }
                    | WorkspaceCollectiveView::Boundary { .. }
            )
        ) {
            return representation::collective(&self.ordinary, operation, output);
        }
        self.ordinary.output_representation(operation, output)
    }
    fn projection_input_observation_mechanism(
        &self,
        format: &eredu_nn::LinearFormatSpec,
    ) -> Result<Option<eredu_nn::ProjectionInputObservationMechanism>, Error> {
        self.ordinary.projection_input_observation_mechanism(format)
    }
    fn grouped_observation_schedule(
        &self,
        bank: &WorkspaceGroupedBank,
        tokens: u32,
    ) -> Result<Option<WorkspaceGroupedObservationSchedule>, Error> {
        self.ordinary.grouped_observation_schedule(bank, tokens)
    }
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        // A legacy caller has no source preparation. It preserves the ordinary
        // unknown result; only the funded shared emitter invokes the hook below.
        self.ordinary.operation_bound(operation)
    }
    fn host_workspace_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        self.ordinary.host_workspace_bound(operation)
    }
}
impl WorkspaceFactMechanisms for MlxParallelWorkspaceMechanisms {
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        WorkspaceFactMechanisms::memory_topology(&self.ordinary)
    }
    fn allocation_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<u64> {
        WorkspaceFactMechanisms::allocation_host_control_bytes(&self.ordinary, op, output)
    }
    fn scratch_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Self::Error> {
        self.scratch_controls(op)
            .map_err(ParallelFactError::Boundary)
    }

    fn output_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<&eredu_core::MemoryPlacement> {
        WorkspaceFactMechanisms::output_placement(&self.ordinary, operation, output)
    }
    fn scratch_placement(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Option<&eredu_core::MemoryPlacement> {
        WorkspaceFactMechanisms::scratch_placement(&self.ordinary, operation)
    }
    type Error = ParallelFactError;
    fn with_prepared_facts_context<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::ExpertInactiveWave(_)
                | WorkspaceOperationKindView::ExpertProviderWave(_)
                | WorkspaceOperationKindView::ExpertRegion(_)
                | WorkspaceOperationKindView::Collective(
                    WorkspaceCollectiveView::Sum { .. }
                        | WorkspaceCollectiveView::GatherFirstAxis { .. }
                        | WorkspaceCollectiveView::Broadcast { .. }
                        | WorkspaceCollectiveView::Boundary { .. }
                )
        ) {
            return self.with_prepared_facts(operation, context.metadata_funding().as_ref(), visit);
        }
        context
            .charge_metadata(
                std::mem::size_of::<(
                    OrdinaryFacts<'_>,
                    &Self,
                    &WorkspaceContext,
                    WorkspaceOperationView<'_>,
                    Result<T, ParallelFactError>,
                )>()
                .checked_add(std::mem::size_of_val(&visit))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            )
            .map_err(|cause| ParallelFactError::Boundary(cause.into()))?;
        self.ordinary
            .with_prepared_facts_context(operation, context, |facts| visit(&OrdinaryFacts(facts)))
            .map_err(Into::into)
    }
    fn with_prepared_facts<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        funding: Option<&HostMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::ExpertInactiveWave(_)
                | WorkspaceOperationKindView::ExpertProviderWave(_)
                | WorkspaceOperationKindView::ExpertRegion(_)
                | WorkspaceOperationKindView::Collective(
                    WorkspaceCollectiveView::Sum { .. }
                        | WorkspaceCollectiveView::GatherFirstAxis { .. }
                        | WorkspaceCollectiveView::Broadcast { .. }
                        | WorkspaceCollectiveView::Boundary { .. }
                )
        ) {
            if let Some(funding) = funding {
                funding
                    .reserve_metadata(std::mem::size_of::<(
                        &Self,
                        WorkspaceOperationView<'_>,
                        Option<&HostMetadataFunding>,
                        Result<T, ParallelFactError>,
                    )>())
                    .map_err(|cause| {
                        ParallelFactError::Source(crate::backend::error::Error::WorkspacePlanning(
                            cause,
                        ))
                    })?;
            }
            return self.with_selected_facts(operation, funding, visit);
        }
        if let Some(funding) = funding {
            funding
                .reserve_metadata(
                    std::mem::size_of::<(
                        OrdinaryFacts<'_>,
                        &Self,
                        WorkspaceOperationView<'_>,
                        Result<T, ParallelFactError>,
                    )>() + std::mem::size_of_val(&visit),
                )
                .map_err(|cause| {
                    ParallelFactError::Source(crate::backend::error::Error::WorkspacePlanning(
                        cause,
                    ))
                })?;
        }
        // CPU fact preparation retains its typed operation source just as
        // resident quotation does; a parallel wrapper must not bypass it.
        return self
            .ordinary
            .with_prepared_facts(operation, funding, |facts| visit(&OrdinaryFacts(facts)))
            .map_err(Into::into);
    }
    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.ordinary.operation_facts(operation).map_err(Into::into)
    }
    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.ordinary
            .write_operation_facts(operation, destination)
            .map_err(Into::into)
    }
    fn host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.ordinary.host_facts(operation).map_err(Into::into)
    }
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.ordinary
            .write_host_facts(operation, destination)
            .map_err(Into::into)
    }
}
impl MlxParallelWorkspaceMechanisms {
    /// The selected child producers already enumerate all native backing births.
    /// Each returned ordinary Array owns one root; the remaining child roots
    /// live through this operation's completion and belong to scratch controls.
    fn scratch_controls(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Error> {
        let allocation = self.ordinary.allocation();
        if allocation.original_storage {
            return Ok(Some(0));
        }
        let overflow = || Error::from(WorkspaceMetadataError::Overflow);
        let births = match operation.kind {
            WorkspaceOperationKindView::ExpertRegion(_) => {
                let quote = self.local_quote(operation)?;
                let Some(aggregate) = quote.aggregate else {
                    return Ok(None);
                };
                // The aggregate reports child scratch separately from the
                // final parent zero/scatter output roots.
                aggregate.child_births
            }
            WorkspaceOperationKindView::ExpertInactiveWave(declaration) => {
                ExpertInactiveWaveQuote::prepare(&self.source, *declaration, self.ordinary)?
                    .aggregate
                    .child_births
            }
            WorkspaceOperationKindView::ExpertProviderWave(declaration) => {
                ExpertProviderWaveQuote::prepare(&self.source, declaration, self.ordinary)?
                    .provider
                    .births
            }
            WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Boundary {
                ..
            }) => {
                let quote = PipelineBoundaryQuote::prepare(&self.source, operation, self.ordinary)?;
                quote
                    .maximum_backing_births()
                    .and_then(|n| n.checked_sub(usize::from(quote.output.is_some())))
                    .ok_or_else(overflow)?
            }
            WorkspaceOperationKindView::Collective(
                WorkspaceCollectiveView::Sum { .. }
                | WorkspaceCollectiveView::GatherFirstAxis { .. }
                | WorkspaceCollectiveView::Broadcast { .. },
            ) => {
                if let Some(quote) =
                    LogicalCollectiveQuote::prepare(&self.source, operation, self.ordinary)?
                {
                    quote
                        .maximum_backing_births()
                        .and_then(|n| n.checked_sub(1))
                        .ok_or_else(overflow)?
                } else {
                    let native = self
                        .source
                        .quote_collective(operation)
                        .map_err(|cause| self.source.neural_error(cause))?;
                    self.source
                        .backing(native.evaluation())
                        .map_err(|cause| self.source.neural_error(cause))?
                        .births
                        .checked_sub(1)
                        .ok_or_else(overflow)?
                }
            }
            _ => return WorkspaceMechanisms::scratch_host_control_bytes(&self.ordinary, operation),
        };
        allocation
            .host_control_bytes()
            .map(|bytes| {
                u64::try_from(births)
                    .ok()
                    .and_then(|count| count.checked_mul(bytes))
                    .ok_or_else(overflow)
            })
            .transpose()
    }
    // Keep unused collective/expert quote values out of ordinary tensor-fact
    // calls. Those calls can occur inside a deep architecture/capture trace.
    #[inline(never)]
    fn with_selected_facts<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        funding: Option<&HostMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = ParallelFactError>) -> T,
    ) -> Result<T, ParallelFactError> {
        if let WorkspaceOperationKindView::ExpertInactiveWave(declaration) = operation.kind {
            let funding = funding.ok_or(ParallelFactError::Unfunded)?;
            if !funding.same_account(&self.metadata_funding) {
                return Err(ParallelFactError::Unfunded);
            }
            funding
                .reserve_metadata(
                    std::mem::size_of::<(PreparedRegion, Result<T, ParallelFactError>)>()
                        + std::mem::size_of_val(&visit),
                )
                .map_err(|cause| {
                    ParallelFactError::Source(crate::backend::error::Error::WorkspacePlanning(
                        cause,
                    ))
                })?;
            let quote = ExpertInactiveWaveQuote::prepare(&self.source, *declaration, self.ordinary)
                .map_err(ParallelFactError::Boundary)?;
            return Ok(visit(&PreparedRegion::Inactive(quote)));
        }
        if let WorkspaceOperationKindView::ExpertProviderWave(declaration) = operation.kind {
            let funding = funding.ok_or(ParallelFactError::Unfunded)?;
            if !funding.same_account(&self.metadata_funding) {
                return Err(ParallelFactError::Unfunded);
            }
            funding
                .reserve_metadata(
                    std::mem::size_of::<(PreparedProviderWave, Result<T, ParallelFactError>)>()
                        + std::mem::size_of_val(&visit),
                )
                .map_err(|cause| {
                    ParallelFactError::Source(crate::backend::error::Error::WorkspacePlanning(
                        cause,
                    ))
                })?;
            let quote = ExpertProviderWaveQuote::prepare(&self.source, declaration, self.ordinary)
                .map_err(ParallelFactError::Boundary)?;
            return Ok(visit(&PreparedProviderWave(quote)));
        }
        if matches!(operation.kind, WorkspaceOperationKindView::ExpertRegion(_)) {
            let funding = funding.ok_or(ParallelFactError::Unfunded)?;
            if !funding.same_account(&self.metadata_funding) {
                return Err(ParallelFactError::Unfunded);
            }
            funding
                .reserve_metadata(
                    std::mem::size_of::<(PreparedRegion, Result<T, ParallelFactError>)>()
                        + std::mem::size_of_val(&visit),
                )
                .map_err(|cause| {
                    ParallelFactError::Source(crate::backend::error::Error::WorkspacePlanning(
                        cause,
                    ))
                })?;
            let quote = self
                .local_quote(operation)
                .map_err(ParallelFactError::Boundary)?;
            return Ok(visit(&PreparedRegion::Local(quote)));
        }
        if funding.is_none() {
            return Err(ParallelFactError::Unfunded);
        }
        let controls = [
            std::mem::size_of::<PreparedCollective<'_>>(),
            representation::control_bytes(),
            std::mem::size_of::<Result<T, ParallelFactError>>(),
            std::mem::size_of::<(
                &Self,
                WorkspaceOperationView<'_>,
                Option<&HostMetadataFunding>,
            )>(),
            std::mem::size_of_val(&visit),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(ParallelFactError::Source(
                crate::backend::error::Error::WorkspacePlanning(HostMetadataFundingError::Overflow),
            ))?;
        self.source.funding().reserve_metadata(bytes).map_err(|e| {
            ParallelFactError::Source(crate::backend::error::Error::WorkspacePlanning(e))
        })?;
        let context = WorkspaceContext::new_with_metadata_funding(
            self.ordinary,
            self.source.funding().clone(),
        )
        .map_err(|cause| ParallelFactError::Boundary(cause.into()))?;
        let unavailable =
            || ParallelFactError::Boundary(WorkspaceMetadataError::Unqualified.into());
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Boundary { .. })
        ) {
            let quote = PipelineBoundaryQuote::prepare(&self.source, operation, self.ordinary)
                .map_err(ParallelFactError::Boundary)?;
            let (scratch_source, output_source) = quote
                .allocation_populations(&context, self.ordinary)
                .map_err(ParallelFactError::Boundary)?
                .ok_or_else(unavailable)?;
            let scratch_source = scratch::with_ring_fences(&context, self.ordinary, scratch_source)
                .map_err(ParallelFactError::Boundary)?;
            let prepared = PreparedCollective::new(operation, scratch_source, output_source)?;
            return Ok(visit(&prepared));
        }
        if let Some(quote) = LogicalCollectiveQuote::prepare(&self.source, operation, self.ordinary)
            .map_err(ParallelFactError::Boundary)?
        {
            let scratch_source = quote
                .operation_scratch(&context, self.ordinary)
                .map_err(ParallelFactError::Boundary)?
                .ok_or_else(unavailable)?;
            let output_source = quote.output_allocations().ok_or_else(unavailable)?.clone();
            let scratch_source = scratch::with_ring_fences(&context, self.ordinary, scratch_source)
                .map_err(ParallelFactError::Boundary)?;
            return Ok(visit(&PreparedCollective::new(
                operation,
                scratch_source,
                Some(output_source),
            )?));
        }
        let native = self
            .source
            .quote_collective(operation)
            .map_err(ParallelFactError::Source)?;
        let backing = self
            .source
            .backing(native.evaluation())
            .map_err(ParallelFactError::Source)?;
        let count = |bytes| {
            usize::try_from(bytes)
                .map_err(|_| ParallelFactError::Boundary(WorkspaceMetadataError::Overflow.into()))
        };
        let scratch_source = scratch::native_cpu(
            &context,
            self.ordinary,
            count(backing.scratch)?,
            backing.births.checked_sub(1).ok_or_else(unavailable)?,
        )
        .map_err(ParallelFactError::Boundary)?;
        let scratch_source = scratch::with_ring_fences(&context, self.ordinary, scratch_source)
            .map_err(ParallelFactError::Boundary)?;
        let output_source = scratch::native_cpu(&context, self.ordinary, count(backing.output)?, 1)
            .map_err(ParallelFactError::Boundary)?;
        let prepared = PreparedCollective::new(operation, scratch_source, Some(output_source))?;
        Ok(visit(&prepared))
    }
}
struct PreparedCollective<'a> {
    operation: WorkspaceOperationView<'a>,
    output: Option<u64>,
    scratch: u64,
    scratch_source: eredu_nn::workspace::WorkspaceAllocationPopulation,
    output_source: Option<eredu_nn::workspace::WorkspaceAllocationPopulation>,
}
impl<'a> PreparedCollective<'a> {
    fn new(
        operation: WorkspaceOperationView<'a>,
        scratch_source: eredu_nn::workspace::WorkspaceAllocationPopulation,
        output_source: Option<eredu_nn::workspace::WorkspaceAllocationPopulation>,
    ) -> Result<Self, ParallelFactError> {
        let overflow = || ParallelFactError::Boundary(WorkspaceMetadataError::Overflow.into());
        let scratch = scratch_source.backing_bytes().ok_or_else(overflow)?;
        let output = output_source
            .as_ref()
            .map(|source| source.backing_bytes().ok_or_else(overflow))
            .transpose()?;
        Ok(Self {
            operation,
            output,
            scratch,
            scratch_source,
            output_source,
        })
    }

    fn validate(&self, actual: WorkspaceOperationView<'_>) -> Result<(), ParallelFactError> {
        let equal = match (self.operation.kind, actual.kind) {
            (
                WorkspaceOperationKindView::Collective(a),
                WorkspaceOperationKindView::Collective(b),
            ) => a == b,
            _ => false,
        };
        if !equal
            || actual.inputs.len() != 1
            || actual.outputs.len() != 1
            || actual.inputs.get(0) != self.operation.inputs.get(0)
            || actual.outputs.get(0) != self.operation.outputs.get(0)
            || actual.inputs.get(0).and_then(|v| v.representation())
                != self
                    .operation
                    .inputs
                    .get(0)
                    .and_then(|v| v.representation())
        {
            return Err(MlxWorkspaceFactError::descriptor(
                "prepared collective facts differ from their operation",
            )
            .into());
        }
        Ok(())
    }
    fn emit(
        &self,
        sink: &mut facts::Emitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
        sink.output(match self.output {
            Some(bytes) => facts::Output::Allocate(bytes),
            None => facts::Output::AliasInput(0),
        })?;
        sink.finish(
            self.scratch,
            format_args!(
                "actual retained collective source; native output and possible copy backing"
            ),
        )
        .map(Some)
    }
    fn host(
        &self,
        sink: &mut facts::HostEmitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceHostFacts>> {
        sink.finish(0,format_args!("Collective has no host numerical payload; retained communicator and native task controls are separately admitted")).map(Some)
    }
}
impl WorkspaceFactMechanisms for PreparedCollective<'_> {
    type Error = ParallelFactError;
    fn scratch_allocations(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<&eredu_nn::workspace::WorkspaceAllocationPopulation>, Self::Error> {
        self.validate(operation)?;
        Ok(Some(&self.scratch_source))
    }
    fn output_allocations(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Result<Option<&eredu_nn::workspace::WorkspaceAllocationPopulation>, Self::Error> {
        self.validate(operation)?;
        if output != 0 {
            return Err(MlxWorkspaceFactError::descriptor(
                "prepared collective output index differs",
            )
            .into());
        }
        Ok(self.output_source.as_ref())
    }

    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.validate(operation)?;
        self.emit(&mut facts::Emitter::count()).map_err(Into::into)
    }
    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.validate(operation)?;
        facts::write(|sink| self.emit(sink), destination).map_err(Into::into)
    }
    fn host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.validate(operation)?;
        self.host(&mut facts::HostEmitter::count())
            .map_err(Into::into)
    }
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.validate(operation)?;
        facts::write_host(|sink| self.host(sink), destination).map_err(Into::into)
    }
}

// Numerical child stages use the implementation retained by the model stream.
pub(crate) fn numerical(
    report: &WorkspaceTraceReport,
    roots: usize,
    mechanism: ResidentExecutionMechanisms,
    context: &WorkspaceContext,
) -> Result<SpeculativeNumericalRecipe, Error> {
    context.charge_metadata(std::mem::size_of::<(
        &WorkspaceTraceReport,
        usize,
        ResidentExecutionMechanisms,
        &WorkspaceContext,
        Result<SpeculativeNumericalRecipe, Error>,
    )>())?;
    let recipe = match mechanism {
        ResidentExecutionMechanisms::Metal(ordinary) => {
            SpeculativeNumericalRecipe::inspect(report, roots, ordinary, context)
        }
        ResidentExecutionMechanisms::Cpu { ordinary, cpu } => {
            SpeculativeNumericalRecipe::inspect_cpu_outputs(report, roots, ordinary, cpu, context)
        }
    }?;
    recipe.with_ordinary_calls(report, mechanism, context)
}

/// Error translation borrows the exact prepared fact source for its lexical
/// visitor. No source is cloned, replaced or allowed to escape its preparation.
struct OrdinaryFacts<'a>(
    &'a dyn WorkspaceFactMechanisms<Error = super::resident_mechanism::ResidentFactError>,
);
impl WorkspaceFactMechanisms for OrdinaryFacts<'_> {
    type Error = ParallelFactError;
    fn output_allocations(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Result<Option<&eredu_nn::workspace::WorkspaceAllocationPopulation>, Self::Error> {
        self.0
            .output_allocations(operation, output)
            .map_err(Into::into)
    }

    fn scratch_allocations(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<&eredu_nn::workspace::WorkspaceAllocationPopulation>, Self::Error> {
        self.0.scratch_allocations(operation).map_err(Into::into)
    }

    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.0.operation_facts(operation).map_err(Into::into)
    }
    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.0
            .write_operation_facts(operation, destination)
            .map_err(Into::into)
    }
    fn host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.0.host_facts(operation).map_err(Into::into)
    }
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.0
            .write_host_facts(operation, destination)
            .map_err(Into::into)
    }
}

/// Exact child source is lent through the ordinary count/write fact emitter.
enum PreparedRegion {
    Local(ExpertLocalQuote),
    Inactive(ExpertInactiveWaveQuote),
}
impl PreparedRegion {
    fn validate(&self, operation: WorkspaceOperationView<'_>) -> Result<(), ParallelFactError> {
        let same = match self {
            Self::Local(source) => {
                matches!(operation.kind,WorkspaceOperationKindView::ExpertRegion(value) if source.matches_operation(value))
                    && operation.inputs.len() == source.inputs.len()
                    && operation.outputs.len() == source.outputs.len()
                    && operation.inputs.iter().zip(&source.inputs).all(|(a, b)| {
                        a.shape() == b.shape()
                            && a.dtype() == b.dtype()
                            && a.representation() == b.representation()
                    })
            }
            Self::Inactive(source) => {
                matches!(operation.kind,WorkspaceOperationKindView::ExpertInactiveWave(value) if *value==source.declaration)
                    && operation.inputs.is_empty()
                    && operation.outputs.is_empty()
            }
        };
        if !same {
            return Err(MlxWorkspaceFactError::descriptor(
                "expert occurrence facts differ from their retained source",
            )
            .into());
        }
        Ok(())
    }
    fn aggregate(&self) -> Option<&ExpertRegionAggregate> {
        match self {
            Self::Local(source) => source.aggregate.as_ref(),
            Self::Inactive(source) => Some(&source.aggregate),
        }
    }
    fn emit(
        &self,
        sink: &mut facts::Emitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
        let Some(source) = self.aggregate() else {
            return Ok(None);
        };
        if source.child_scratch.is_none() {
            return Ok(None);
        }
        for &bytes in &source.output_bytes {
            sink.output(facts::Output::Allocate(bytes))?;
        }
        sink.finish(
            source.child_bytes,
            format_args!("complete retained expert local/movement/transport/count source"),
        )
        .map(Some)
    }
    fn host(
        &self,
        sink: &mut facts::HostEmitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceHostFacts>> {
        let Some(source) = self.aggregate() else {
            return Ok(None);
        };
        sink.finish(source.host_bytes,format_args!("shared sequential order/destination payloads; control/source arenas retain separate exact metadata admissions")).map(Some)
    }
}
impl WorkspaceFactMechanisms for PreparedRegion {
    type Error = ParallelFactError;
    fn scratch_allocations(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<&eredu_nn::workspace::WorkspaceAllocationPopulation>, Self::Error> {
        self.validate(operation)?;
        Ok(self
            .aggregate()
            .and_then(|source| source.child_scratch.as_ref()))
    }

    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.validate(op)?;
        self.emit(&mut facts::Emitter::count()).map_err(Into::into)
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.validate(op)?;
        facts::write(|sink| self.emit(sink), out).map_err(Into::into)
    }
    fn host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.validate(op)?;
        self.host(&mut facts::HostEmitter::count())
            .map_err(Into::into)
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.validate(op)?;
        facts::write_host(|sink| self.host(sink), out).map_err(Into::into)
    }
}

/// An inactive replicated provider has no numerical outputs or local kernel.
struct PreparedProviderWave(ExpertProviderWaveQuote);
impl PreparedProviderWave {
    fn validate(&self, op: WorkspaceOperationView<'_>) -> Result<(), ParallelFactError> {
        if !matches!(op.kind,WorkspaceOperationKindView::ExpertProviderWave(value) if value==self.0.declaration)
            || !op.inputs.is_empty()
            || !op.outputs.is_empty()
        {
            return Err(MlxWorkspaceFactError::descriptor(
                "provider wave differs from its retained source",
            )
            .into());
        }
        Ok(())
    }
    fn emit(
        &self,
        sink: &mut facts::Emitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceOperationFacts>> {
        let Some(source) = &self.0.provider.scratch else {
            return Ok(None);
        };
        let Some(bytes) = source.backing_bytes() else {
            return Ok(None);
        };
        sink.finish(bytes, format_args!("retained control-only provider votes"))
            .map(Some)
    }
    fn host(
        &self,
        sink: &mut facts::HostEmitter<'_>,
    ) -> facts::FactResult<Option<WorkspaceHostFacts>> {
        sink.finish(
            0,
            format_args!("provider source/control metadata retains its separate admission"),
        )
        .map(Some)
    }
}
impl WorkspaceFactMechanisms for PreparedProviderWave {
    type Error = ParallelFactError;
    fn scratch_allocations(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<&eredu_nn::workspace::WorkspaceAllocationPopulation>, Self::Error> {
        self.validate(operation)?;
        Ok(self.0.provider.scratch.as_ref())
    }

    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.validate(op)?;
        self.emit(&mut facts::Emitter::count()).map_err(Into::into)
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.validate(op)?;
        facts::write(|sink| self.emit(sink), out).map_err(Into::into)
    }
    fn host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.validate(op)?;
        self.host(&mut facts::HostEmitter::count())
            .map_err(Into::into)
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        out: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.validate(op)?;
        facts::write_host(|sink| self.host(sink), out).map_err(Into::into)
    }
}
