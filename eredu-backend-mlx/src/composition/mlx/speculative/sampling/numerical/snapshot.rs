//! Independent key copies through the existing registered-copy worker.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment, RetainedOriginalCopyEnvironment,
    array_copy::{IsolatedArrayCopy, OriginalCopyLayoutBuilder},
    nn::workspace::MlxMetalWorkspaceMechanisms,
};
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_runtime::working_memory::{OriginalSpeculativeSourceIdentity, WorkingMemoryError};

/// No request list or native graph backedge; the source's completed stream is
/// borrowed only during an independently admitted copy.
#[derive(Clone)]
pub(in crate::composition::mlx::speculative::sampling) struct SnapshotContext {
    environment: RetainedOriginalCopyEnvironment,
    draft_environment: Option<RetainedOriginalCopyEnvironment>,
    identity: OriginalSpeculativeSourceIdentity,
    roots: safemlx::PrefillRootsRuntime,
    mechanisms: MlxMetalWorkspaceMechanisms,
    capacity: u64,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Error,
    _host: HostPreparationAuthority,
}
fn missing() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
impl SnapshotContext {
    pub(in crate::composition::mlx::speculative::sampling) fn controls() -> Option<usize> {
        // Exactly one concrete prepared wrapper is lent by the key's owned
        // stream. Price both real generic instantiations without a raw loan.
        let stream_loan = size_of::<(&Self,&safemlx::PreparedStreamCopy<OriginalSpeculativeNumericalBudgetCustody>)>()
            .max(size_of::<(&Self,&safemlx::PreparedStreamCopy<model::StreamOwner>)>());
        let parts = [size_of::<usize>(),
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<OriginalSpeculativeSourceIdentity>(),
            size_of::<RetainedOriginalCopyEnvironment>(),size_of::<(&Self,&Value)>(),
            size_of::<Result<OriginalCopyEnvironment<'_>,crate::backend::OriginalCopyEnvironmentError>>(),
            size_of::<Option<RetainedOriginalCopyEnvironment>>(),
            size_of::<Result<Self,Error>>(),
            stream_loan,stream_loan,
            size_of::<Result<OriginalCopyEnvironment<'_>,crate::backend::OriginalCopyEnvironmentError>>(),
            size_of::<SpeculativeExecutionStreams<'_>>(),
            size_of::<(&OriginalSpeculativeNumericalSources,&OriginalCopyEnvironment<'_>,&OriginalCopyEnvironment<'_>)>(),
            size_of::<Result<OriginalCopyEnvironment<'_>,crate::backend::OriginalCopyEnvironmentError>>(),
            OriginalCopyEnvironment::control_bytes()?,
            size_of::<safemlx::StreamCopyPlan<()>>(),
            size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<
                Option<(
                    &safemlx::OriginalBufferBudget,
                    &OriginalSpeculativeNumericalBudgetCustody,
                )>,
            >(),
            OriginalCopyEnvironment::control_bytes()?,
            safemlx::OriginalBufferBudget::inspection_control_bytes()?,
            OriginalCopyLayoutBuilder::resume_control_bytes()?,
            size_of::<Result<OriginalNumericalKey, Error>>(),
            size_of::<Result<u64, Error>>(),
            size_of::<OriginalNumericalKey>(),
            size_of::<Failure>(),
            BackendFailure::source_retention_peak_bytes::<Failure>()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(in crate::composition::mlx::speculative::sampling) fn key_controls() -> Option<usize> {
        value_control_bytes()?
            .checked_add(size_of::<OriginalNumericalKey>())?
            .checked_add(size_of::<Option<OriginalNumericalKey>>())?
            .checked_add(size_of::<Result<OriginalNumericalKey, Error>>())?
            .checked_add(size_of::<(
                Array,
                crate::backend::array_copy::RegisteredArrayCopyCustody,
            )>())
    }
    pub(in crate::composition::mlx::speculative::sampling) fn prepare(
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Self, Error> {
        Self::prepare_environments(sources,environment,environment)
    }
    pub(in crate::composition::mlx::speculative::sampling) fn prepare_for_context(
        context:SpeculativeExecutionStreams<'_>,
    )->Result<Self,Error>{
        let (sources,target)=context.original_numerical_for(SamplingPlacement::Target).ok_or_else(missing)?;
        let (other,draft)=context.original_numerical_for(SamplingPlacement::Draft).ok_or_else(missing)?;
        if !std::ptr::eq(sources,other) || !target.pool().same_domain(draft.pool()){return Err(missing());}
        Self::prepare_environments(sources,target,draft)
    }
    fn prepare_environments(sources:&OriginalSpeculativeNumericalSources,
        environment:&OriginalCopyEnvironment<'_>,draft:&OriginalCopyEnvironment<'_>,
    )->Result<Self,Error>{
        sources.validate_environment(environment)?;
        sources.validate_environment(draft)?;
        let draft_environment=if std::ptr::eq(environment,draft){None}else{
            let retained=draft.retain_prerequisites().map_err(|cause|sources.retain_startup_error(cause))?;
            sources.metadata_funding().reserve_metadata(retained.control_bytes()
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?)
                .map_err(Error::WorkspacePlanning)?;
            Some(retained)
        };
        // The enclosing sampler owner includes this fixed value and its source
        // snapshots before construction; native copies remain separately admitted.
        // The immutable stream-copy query is consumed before retaining its
        // scalar plan. No new stream/queue/native control owner is constructed.
        let environment = environment
            .retain_prerequisites()
            .map_err(|e| sources.retain_startup_error(e))?;
        let extra = environment
            .control_bytes()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        sources
            .metadata_funding()
            .reserve_metadata(extra)
            .map_err(Error::WorkspacePlanning)?;
        let (roots, mechanisms) = sources.numerical_prerequisites();
        Ok(Self {
            environment,
            draft_environment,
            identity: sources.request().source_identity(),
            roots: roots.clone(),
            mechanisms,
            capacity: sources.request().capacity_bytes(),
        })
    }
    fn validate<'a>(&self, key: &'a OriginalNumericalKey) -> Result<&'a Value, Error> {
        let value = key.value().value();
        if value.meaning != Meaning::RandomKey
            || !value
                .provenance
                .source()
                .belongs_to_identity(&self.identity)
        {
            return Err(missing());
        }
        if value.original_budget.is_none() && value._copy.is_none() {
            return Err(missing());
        }
        Ok(value)
    }
    fn environment<'a>(&'a self, value: &'a Value) -> Result<OriginalCopyEnvironment<'a>, Error> {
        self.try_environment(value)
            .map_err(|cause| retain_planning_error(cause, value.funding.clone()))
    }
    fn try_environment<'a>(&'a self,value:&'a Value)
        ->Result<OriginalCopyEnvironment<'a>,crate::backend::OriginalCopyEnvironmentError>{
        match &value.stream {
            ValueStream::Numerical(stream)=>self.loan_stream(stream),
            ValueStream::Embedded(stream)=>self.loan_stream(stream),
            ValueStream::Model(_)=>Err(WorkingMemoryError::IdentityMismatch.into()),
        }
    }
    fn loan_stream<'a,C:Send+Sync+'static>(&'a self,stream:&'a safemlx::PreparedStreamCopy<C>)
        ->Result<OriginalCopyEnvironment<'a>,crate::backend::OriginalCopyEnvironmentError>{
        // Both markers came from the actual bound context. The key owns the
        // native stream throughout this loan; a scalar descriptor cannot create
        // or replace its stream, allocator, pool or completed source.
        self.environment.loan(stream,self.identity.pool()).or_else(|cause|{
            match &self.draft_environment {
                Some(draft)=>draft.loan(stream,self.identity.pool()),
                None=>Err(cause),
            }
        })
    }
    /// Retained snapshot size uses the actual isolated-copy population, including
    /// the complete backing of a source view. It is not another native grant.
    pub(in crate::composition::mlx::speculative::sampling) fn key_bytes(
        &self,
        key: &OriginalNumericalKey,
    ) -> Option<u64> {
        let value = self.validate(key).ok()?;
        // This query makes no owned diagnostic, handle or tensor.
        let environment = self.try_environment(value).ok()?;
        let mut layout = OriginalCopyLayoutBuilder::new();
        layout.push_retained_source(&value.array).ok()?;
        layout.push_operand(&value.array).ok()?;
        let layout = layout.finish(&environment).ok()??;
        u64::try_from(layout.physical_bytes()).ok()
    }
    pub(in crate::composition::mlx::speculative::sampling) fn validate_key(
        &self,
        key: &OriginalNumericalKey,
    ) -> bool {
        self.key_bytes(key).is_some()
    }
    pub(in crate::composition::mlx::speculative::sampling) fn copy_key(
        &self,
        key: &OriginalNumericalKey,
        host: &HostPreparationAuthority,
    ) -> Result<OriginalNumericalKey, Error> {
        if host.is_unmanaged() {
            return Err(missing());
        }
        let value = self.validate(key)?;
        let environment = self.environment(value)?;
        let original = match (&value.original_budget, &value.provenance) {
            (Some(budget), Provenance::Numerical(custody)) => Some((budget, custody)),
            (None, _) if value._copy.is_some() => None,
            _ => return Err(missing()),
        };
        let copied = IsolatedArrayCopy::new(&value.array).copy_original(
            original,
            &environment,
            &self.roots,
            self.mechanisms,
            &value.funding,
            self.capacity,
        )?;
        let (array, custody) = copied.into_parts();
        // The caller paid the Value Rc and all returned key controls before this
        // constructor. Copy completion/publication precedes the only extraction.
        Ok(OriginalNumericalKey::copied(OriginalNumericalValue(
            Some(Rc::new(Value {
                array,
                original_budget: None,
                stream: value.stream.clone(),
                meaning: Meaning::RandomKey,
                provenance: value.provenance.clone(),
                funding: value.funding.clone(),
                _copy: Some(custody),
                _snapshot_host: Some(host.clone()),
                _readout_source: None,
            })),
            None,
        )))
    }
    pub(in crate::composition::mlx::speculative::sampling) fn failure(
        cause: Error,
        host: &HostPreparationAuthority,
    ) -> eredu_core::speculative::SpeculativeControlError {
        // Even already-retained native errors need independent custody for the
        // snapshot's returned transport. The paid erased shell dies first.
        eredu_core::speculative::SpeculativeControlError::Backend(BackendFailure::from_error(
            Failure {
                cause,
                _host: host.clone(),
            },
        ))
    }
}
