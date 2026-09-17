//! Retained native source preparation followed by the shared pure fact emitter.
use super::*;
mod boundary;
mod expert_region;
pub(crate) use expert_region::{GroupedSourceObserver,ExpertLocalObservationSource,ExpertLocalQuote,ExpertProviderWaveQuote,ExpertInactiveWaveQuote,ExpertCountQuote,ExpertTransportQuote,ExpertRegionAggregate,ExpertProviderQuote,ExpertMovementKind,ExpertTransferProfile,ExpertReorderEnvelope,ExpertLocalStage,ExpertLocalStageBound};
mod representation;
mod logical;
pub(crate) use logical::{LogicalCollectiveQuote,LogicalCollectiveKind};
pub(crate) use boundary::{PipelineBoundaryQuote,BoundaryStageCapacity};
use crate::backend::runtime::distributed::topology::original_source::parallel::{
    OriginalParallelSource,
};

#[derive(Debug)]
pub(crate) struct MlxParallelWorkspaceMechanisms {
    ordinary: ResidentExecutionMechanisms,
    source: OriginalParallelSource,
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum ParallelFactError {
    #[error(transparent)]
    Fixed(#[from] MlxWorkspaceFactError),
    #[error("selected parallel source preparation failed")]
    Source(#[source] crate::backend::error::Error),
    #[error("prepared pipeline frame source failed")]
    Boundary(#[source] Error),
    #[error("parallel source preparation requires its original funding")]
    Unfunded,
}
impl MlxParallelWorkspaceMechanisms {
    pub(crate) fn new(
        ordinary: ResidentExecutionMechanisms,
        source: OriginalParallelSource,
    ) -> Self {
        Self { ordinary, source }
    }
    pub(crate) fn prepare_workspace(self) -> Result<MlxParallelWorkspace, Error> {
        self.prepare_workspace_source(None)
    }
    pub(crate) fn prepare_workspace_with_addressable(self,addressable:AddressableSources)->Result<MlxParallelWorkspace,Error>{
        self.prepare_workspace_source(Some(addressable))
    }
    fn prepare_workspace_source(self,addressable:Option<AddressableSources>)->Result<MlxParallelWorkspace,Error>{
        let source = self.source.clone();
        let ordinary = self.ordinary;
        source
            .funding()
            .reserve_metadata(
                std::mem::size_of::<MlxParallelWorkspace>()
                    + std::mem::size_of::<Result<MlxParallelWorkspace, Error>>(),
            )
            .map_err(|cause| Error::from(WorkspaceMetadataError::Funding(cause)))?;
        let funding = source.funding().clone();
        if addressable.as_ref().is_some_and(|value|!value.funding().same_account(&funding)) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let context = match &addressable {
            Some(value)=>WorkspaceContext::new_with_metadata_funding(MlxAddressableWorkspaceMechanisms::new(self,value.clone()),funding)?,
            None=>WorkspaceContext::new_with_metadata_funding(self,funding)?,
        };
        Ok(MlxParallelWorkspace {
            context,
            ordinary,
            source,
            addressable,
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
}
impl MlxParallelWorkspace {
    pub(crate) fn declaration_source(&self)->&eredu_runtime::RetainedCommunicationSource {
        self.source.declaration_source()
    }
    pub(crate) fn addressable_sources(&self)->Option<&AddressableSources>{self.addressable.as_ref()}
    /// One actual frame trace retains its finite native collective occurrences.
    /// This is descriptive preparation; no native invocation is entered here.
    pub(crate) fn prepare_invocation(&self,report:&WorkspaceTraceReport)
        ->Result<crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelInvocation,Error> {
        self.source.prepare_invocation_with_boundary(&report.operations,Some(self.ordinary))
            .map_err(|cause|self.source.neural_error(cause))
    }
    pub(crate) fn context(&self) -> &WorkspaceContext {
        &self.context
    }
    pub(crate) fn into_context(self) -> WorkspaceContext { self.context }
    pub(crate) fn recorder(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<resident_recipe::ParallelRecipeRecorder, Error> {
        let mut recorder=resident_recipe::ParallelRecipeRecorder::new(
            geometry,
            self.ordinary,
            &self.context,
            &self.source,
        )?;
        if let Some(source)=&self.addressable{recorder.bind_addressable_sources(source.clone())?;}
        Ok(recorder)
    }
}
impl WorkspaceMechanisms for MlxParallelWorkspaceMechanisms {
    fn prepared_text_input_dtype(&self)->Option<WorkspaceDtype> {
        self.ordinary.prepared_text_input_dtype()
    }

    fn output_representation(
        &self,
        operation: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<WorkspaceRepresentation> {
        if matches!(operation.kind,WorkspaceOperationKindView::ExpertRegion(_)) {
            let quote=ExpertLocalQuote::prepare(&self.source,operation,self.ordinary).ok()?;
            let dtype=quote.outputs.get(output)?.representation()?.dtype();
            // Final typed zeros/ScatterAxis writes complete contiguous rows.
            return Some(WorkspaceRepresentation::new(dtype,true));
        }
        if matches!(
            operation.kind,
            WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Sum { .. } | WorkspaceCollectiveView::GatherFirstAxis { .. } | WorkspaceCollectiveView::Broadcast { .. } | WorkspaceCollectiveView::Boundary { .. })
        ) {
            return representation::collective(&self.ordinary, operation, output);
        }
        let representation = self.ordinary.output_representation(operation, output);
        if representation.is_none()
            && operation.outputs.get(output).is_some_and(|layout| layout.dtype() == WorkspaceDtype::Float32)
            && !matches!(operation.kind, WorkspaceOperationKindView::ParameterPlaceholder)
            && std::env::var_os("EREDU_TRACE_PARALLEL_REPRESENTATION").is_some()
        {
            eprintln!("EREDU_PARALLEL_REPRESENTATION missing producer={:?} output={output} layout={:?}",
                operation.kind, operation.outputs.get(output));
            for (index, input) in operation.inputs.iter().enumerate() {
                eprintln!("EREDU_PARALLEL_REPRESENTATION input={index} shape={:?} dtype={:?} actual={:?}",
                    input.shape(), input.dtype(), input.representation());
            }
        }
        representation
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
    type Error = ParallelFactError;
    fn with_prepared_facts<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        funding: Option<&WorkspaceMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error> {
        if matches!(operation.kind,
            WorkspaceOperationKindView::ExpertInactiveWave(_)
            | WorkspaceOperationKindView::ExpertProviderWave(_)
            | WorkspaceOperationKindView::ExpertRegion(_)
            | WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Sum { .. }
                | WorkspaceCollectiveView::GatherFirstAxis { .. }
                | WorkspaceCollectiveView::Broadcast { .. }
                | WorkspaceCollectiveView::Boundary { .. })) {
            if let Some(funding)=funding {
                funding.reserve_metadata(std::mem::size_of::<(
                    &Self,WorkspaceOperationView<'_>,Option<&WorkspaceMetadataFunding>,
                    Result<T,ParallelFactError>,
                )>()).map_err(|cause|ParallelFactError::Source(
                    crate::backend::error::Error::WorkspacePlanning(cause)))?;
            }
            return self.with_selected_facts(operation,funding,visit);
        }
        if let Some(funding) = funding {
            funding.reserve_metadata(std::mem::size_of::<(
                OrdinaryFacts<'_>, &Self, WorkspaceOperationView<'_>,
                Result<T, ParallelFactError>,
            )>() + std::mem::size_of_val(&visit))
                .map_err(|cause| ParallelFactError::Source(
                    crate::backend::error::Error::WorkspacePlanning(cause)))?;
        }
        // CPU fact preparation retains its typed operation source just as
        // resident quotation does; a parallel wrapper must not bypass it.
        return self.ordinary.with_prepared_facts(operation, funding, |facts| {
            visit(&OrdinaryFacts(facts))
        }).map_err(Into::into);
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
    // Keep unused collective/expert quote values out of ordinary tensor-fact
    // calls. Those calls can occur inside a deep architecture/capture trace.
    #[inline(never)]
    fn with_selected_facts<T>(
        &self,
        operation: WorkspaceOperationView<'_>,
        funding: Option<&WorkspaceMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = ParallelFactError>) -> T,
    ) -> Result<T, ParallelFactError> {
        if let WorkspaceOperationKindView::ExpertInactiveWave(declaration)=operation.kind {
            let funding=funding.ok_or(ParallelFactError::Unfunded)?;
            if !funding.same_account(self.source.funding()){return Err(ParallelFactError::Unfunded);}
            funding.reserve_metadata(std::mem::size_of::<(PreparedRegion,Result<T,ParallelFactError>)>()
                +std::mem::size_of_val(&visit)).map_err(|cause|ParallelFactError::Source(
                    crate::backend::error::Error::WorkspacePlanning(cause)))?;
            let quote=ExpertInactiveWaveQuote::prepare(&self.source,*declaration,self.ordinary)
                .map_err(ParallelFactError::Boundary)?;
            return Ok(visit(&PreparedRegion::Inactive(quote)));
        }
        if let WorkspaceOperationKindView::ExpertProviderWave(declaration)=operation.kind {
            let funding=funding.ok_or(ParallelFactError::Unfunded)?;
            if !funding.same_account(self.source.funding()){return Err(ParallelFactError::Unfunded);}
            funding.reserve_metadata(std::mem::size_of::<(PreparedProviderWave,Result<T,ParallelFactError>)>()
                +std::mem::size_of_val(&visit)).map_err(|cause|ParallelFactError::Source(
                    crate::backend::error::Error::WorkspacePlanning(cause)))?;
            let quote=ExpertProviderWaveQuote::prepare(&self.source,declaration,self.ordinary)
                .map_err(ParallelFactError::Boundary)?;
            return Ok(visit(&PreparedProviderWave(quote)));
        }
        if matches!(operation.kind,WorkspaceOperationKindView::ExpertRegion(_)) {
            let funding=funding.ok_or(ParallelFactError::Unfunded)?;
            if !funding.same_account(self.source.funding()){return Err(ParallelFactError::Unfunded);}
            funding.reserve_metadata(std::mem::size_of::<(PreparedRegion,Result<T,ParallelFactError>)>()
                +std::mem::size_of_val(&visit)).map_err(|cause|ParallelFactError::Source(
                    crate::backend::error::Error::WorkspacePlanning(cause)))?;
            let quote=ExpertLocalQuote::prepare(&self.source,operation,self.ordinary).map_err(ParallelFactError::Boundary)?;
            return Ok(visit(&PreparedRegion::Local(quote)));
        }
        if funding.is_none() {
            return Err(ParallelFactError::Unfunded);
        }
        if std::env::var_os("EREDU_TRACE_PARALLEL_REPRESENTATION").is_some() {
            for (index, input) in operation.inputs.iter().enumerate() {
                if input.dtype() == WorkspaceDtype::Float32 && input.representation().is_none() {
                    eprintln!("EREDU_PARALLEL_REPRESENTATION consuming collective={:?} input={index} shape={:?}",
                        operation.kind, input.shape());
                }
            }
        }
        let controls = [
            std::mem::size_of::<PreparedCollective<'_>>(),
            representation::control_bytes(),
            std::mem::size_of::<Result<T, ParallelFactError>>(),
            std::mem::size_of::<(
                &Self,
                WorkspaceOperationView<'_>,
                Option<&WorkspaceMetadataFunding>,
            )>(),
            std::mem::size_of_val(&visit),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(ParallelFactError::Source(crate::backend::error::Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            )))?;
        self.source
            .funding()
            .reserve_metadata(bytes)
            .map_err(|e| ParallelFactError::Source(crate::backend::error::Error::WorkspacePlanning(e)))?;
        if matches!(operation.kind, WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Boundary{..})) {
            let quote=PipelineBoundaryQuote::prepare(&self.source,operation,self.ordinary)
                .map_err(ParallelFactError::Boundary)?;
            let prepared=PreparedCollective{operation,output:quote.output,scratch:quote.scratch};
            return Ok(visit(&prepared));
        }
        if let Some(quote)=LogicalCollectiveQuote::prepare(&self.source,operation,self.ordinary).map_err(ParallelFactError::Boundary)? {
            return Ok(visit(&PreparedCollective{operation,output:Some(quote.output),scratch:quote.scratch}));
        }
        let native = self
            .source
            .quote_collective(operation)
            .map_err(ParallelFactError::Source)?;
        let backing = self
            .source
            .backing(native.evaluation())
            .map_err(ParallelFactError::Source)?;
        let prepared = PreparedCollective { operation, output:Some(backing.output), scratch:backing.scratch };
        Ok(visit(&prepared))
    }
}
struct PreparedCollective<'a> {
    operation: WorkspaceOperationView<'a>,
    output: Option<u64>,
    scratch: u64,
}
impl PreparedCollective<'_> {
    fn validate(&self, actual: WorkspaceOperationView<'_>) -> Result<(), ParallelFactError> {
        let equal = match(self.operation.kind,actual.kind) {
            (WorkspaceOperationKindView::Collective(a),WorkspaceOperationKindView::Collective(b))=>a==b,
            _=>false,
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
            Some(bytes)=>facts::Output::Allocate(bytes), None=>facts::Output::AliasInput(0),
        })?;
        sink.finish(
            self.scratch,
            format_args!("actual retained collective source; native output and possible copy backing"),
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
        &WorkspaceTraceReport, usize, ResidentExecutionMechanisms, &WorkspaceContext,
        Result<SpeculativeNumericalRecipe, Error>,
    )>())?;
    match mechanism {
        ResidentExecutionMechanisms::Metal(ordinary) =>
            SpeculativeNumericalRecipe::inspect(report, roots, ordinary, context),
        ResidentExecutionMechanisms::Cpu { ordinary, cpu } =>
            SpeculativeNumericalRecipe::inspect_cpu_outputs(report, roots, ordinary, cpu, context),
    }
}

/// Error translation borrows the exact prepared fact source for its lexical
/// visitor. No source is cloned, replaced or allowed to escape its preparation.
struct OrdinaryFacts<'a>(&'a dyn WorkspaceFactMechanisms<Error = MlxWorkspaceFactError>);
impl WorkspaceFactMechanisms for OrdinaryFacts<'_> {
    type Error = ParallelFactError;
    fn operation_facts(&self, operation: WorkspaceOperationView<'_>)
        -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.0.operation_facts(operation).map_err(Into::into)
    }
    fn write_operation_facts(&self, operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>)
        -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.0.write_operation_facts(operation, destination).map_err(Into::into)
    }
    fn host_facts(&self, operation: WorkspaceOperationView<'_>)
        -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.0.host_facts(operation).map_err(Into::into)
    }
    fn write_host_facts(&self, operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>)
        -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.0.write_host_facts(operation, destination).map_err(Into::into)
    }
}

/// Exact child source is lent through the ordinary count/write fact emitter.
enum PreparedRegion{Local(ExpertLocalQuote),Inactive(ExpertInactiveWaveQuote)}
impl PreparedRegion {
    fn validate(&self,operation:WorkspaceOperationView<'_>)->Result<(),ParallelFactError>{
        let same=match self{
            Self::Local(source)=>matches!(operation.kind,WorkspaceOperationKindView::ExpertRegion(value) if source.matches_operation(value))
                &&operation.inputs.len()==source.inputs.len()&&operation.outputs.len()==source.outputs.len()
                &&operation.inputs.iter().zip(&source.inputs).all(|(a,b)|a.shape()==b.shape()&&a.dtype()==b.dtype()&&a.representation()==b.representation()),
            Self::Inactive(source)=>matches!(operation.kind,WorkspaceOperationKindView::ExpertInactiveWave(value) if *value==source.declaration)
                &&operation.inputs.is_empty()&&operation.outputs.is_empty(),
        };
        if !same{return Err(MlxWorkspaceFactError::descriptor("expert occurrence facts differ from their retained source").into());}
        Ok(())
    }
    fn aggregate(&self)->Option<&ExpertRegionAggregate>{
        match self{Self::Local(source)=>source.aggregate.as_ref(),Self::Inactive(source)=>Some(&source.aggregate)}
    }
    fn emit(&self,sink:&mut facts::Emitter<'_>)->facts::FactResult<Option<WorkspaceOperationFacts>>{
        let Some(source)=self.aggregate() else{return Ok(None);};
        for &bytes in &source.output_bytes{sink.output(facts::Output::Allocate(bytes))?;}
        sink.finish(source.child_bytes,format_args!("complete retained expert local/movement/transport/count source")).map(Some)
    }
    fn host(&self,sink:&mut facts::HostEmitter<'_>)->facts::FactResult<Option<WorkspaceHostFacts>>{
        let Some(source)=self.aggregate() else{return Ok(None);};
        sink.finish(source.host_bytes,format_args!("shared sequential order/destination payloads; control/source arenas retain separate exact metadata admissions")).map(Some)
    }
}
impl WorkspaceFactMechanisms for PreparedRegion {
    type Error=ParallelFactError;
    fn operation_facts(&self,op:WorkspaceOperationView<'_>)->Result<Option<WorkspaceOperationFacts>,Self::Error>{
        self.validate(op)?;self.emit(&mut facts::Emitter::count()).map_err(Into::into)
    }
    fn write_operation_facts(&self,op:WorkspaceOperationView<'_>,out:WorkspaceEffectDestination<'_>)->Result<Option<WorkspaceOperationFacts>,Self::Error>{
        self.validate(op)?;facts::write(|sink|self.emit(sink),out).map_err(Into::into)
    }
    fn host_facts(&self,op:WorkspaceOperationView<'_>)->Result<Option<WorkspaceHostFacts>,Self::Error>{
        self.validate(op)?;self.host(&mut facts::HostEmitter::count()).map_err(Into::into)
    }
    fn write_host_facts(&self,op:WorkspaceOperationView<'_>,out:WorkspaceHostDestination<'_>)->Result<Option<WorkspaceHostFacts>,Self::Error>{
        self.validate(op)?;facts::write_host(|sink|self.host(sink),out).map_err(Into::into)
    }
}

/// An inactive replicated provider has no numerical outputs or local kernel.
struct PreparedProviderWave(ExpertProviderWaveQuote);
impl PreparedProviderWave {
    fn validate(&self,op:WorkspaceOperationView<'_>)->Result<(),ParallelFactError>{
        if !matches!(op.kind,WorkspaceOperationKindView::ExpertProviderWave(value) if value==self.0.declaration)
            ||!op.inputs.is_empty()||!op.outputs.is_empty(){
            return Err(MlxWorkspaceFactError::descriptor("provider wave differs from its retained source").into());
        }
        Ok(())
    }
    fn emit(&self,sink:&mut facts::Emitter<'_>)->facts::FactResult<Option<WorkspaceOperationFacts>>{
        sink.finish(self.0.provider.backing,format_args!("retained control-only provider votes")).map(Some)
    }
    fn host(&self,sink:&mut facts::HostEmitter<'_>)->facts::FactResult<Option<WorkspaceHostFacts>>{
        sink.finish(0,format_args!("provider source/control metadata retains its separate admission")).map(Some)
    }
}
impl WorkspaceFactMechanisms for PreparedProviderWave {
    type Error=ParallelFactError;
    fn operation_facts(&self,op:WorkspaceOperationView<'_>)->Result<Option<WorkspaceOperationFacts>,Self::Error>{
        self.validate(op)?;self.emit(&mut facts::Emitter::count()).map_err(Into::into)
    }
    fn write_operation_facts(&self,op:WorkspaceOperationView<'_>,out:WorkspaceEffectDestination<'_>)->Result<Option<WorkspaceOperationFacts>,Self::Error>{
        self.validate(op)?;facts::write(|sink|self.emit(sink),out).map_err(Into::into)
    }
    fn host_facts(&self,op:WorkspaceOperationView<'_>)->Result<Option<WorkspaceHostFacts>,Self::Error>{
        self.validate(op)?;self.host(&mut facts::HostEmitter::count()).map_err(Into::into)
    }
    fn write_host_facts(&self,op:WorkspaceOperationView<'_>,out:WorkspaceHostDestination<'_>)->Result<Option<WorkspaceHostFacts>,Self::Error>{
        self.validate(op)?;facts::write_host(|sink|self.host(sink),out).map_err(Into::into)
    }
}
