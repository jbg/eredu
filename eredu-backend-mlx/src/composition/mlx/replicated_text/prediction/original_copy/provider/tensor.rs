//! Immutable tensor packets use the same settled source and isolated copy worker.
use super::*;
use crate::{
    backend::array_copy::{IsolatedArrayCopy, RegisteredArrayCopyCustody},
    MlxTensor,
};
use eredu_architectures::speculative_execution::EmbeddedPredictionTensor;

// The tensor retires before this account-only owner. Its Arc shell is paid
// before the native copy and freed before the actual copy and metadata custody.
struct TensorCopyCustody {
    _copy: RegisteredArrayCopyCustody,
    _host: HostPreparationAuthority,
}
fn copy_tensor(
    context: &OriginalPredictionCopyContext,
    source: &MlxTensor,
    completed: Option<&CompletedResidentSource>,
) -> Result<PreparedLane<MlxTensor>, StartupCause> {
    let parts = [
        size_of::<PreparedLane<MlxTensor>>(),
        size_of::<Result<PreparedLane<MlxTensor>, StartupCause>>(),
        size_of::<TensorCopyCustody>(),
        size_of::<(safemlx::Array, RegisteredArrayCopyCustody)>(),
        size_of::<Option<&CompletedResidentSource>>(),
        HostPreparationAuthority::retention_bytes::<TensorCopyCustody>()
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
    ];
    context.reserve(parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
    let environment = context.loan()?;
    let copied = IsolatedArrayCopy::new(source.as_array()).copy_completed(
        completed,
        &environment,
        &context.roots,
        context.mechanisms,
        context.preparation.metadata_funding(),
        context.preparation.limits(),
    )?;
    let (array, custody) = copied.into_parts();
    Ok(PreparedLane {
        value: MlxTensor::from_array(array),
        host: HostPreparationAuthority::retain(TensorCopyCustody {
            _copy: custody,
            _host: context.host.clone(),
        }),
    })
}
use crate::composition::mlx::speculative::{CompletedTensorSource, RegisteredTensorSource};
use eredu_runtime::working_memory::OriginalSpeculativeSourceIdentity;
struct TensorProvider {
    context: OriginalPredictionCopyContext,
    identity: OriginalSpeculativeSourceIdentity,
}
impl PreparedEmbeddedCopyProvider<MlxTensor> for TensorProvider {
    fn copy_state(
        &self,
        source: &MlxTensor,
        evidence: Option<&PreparedEmbeddedEvidence>,
    ) -> Result<PreparedEmbeddedPayload<MlxTensor>, BackendFailure> {
        // The shared dispatcher calls the source-aware method below. Retain
        // the existing callback contract for callers without evidence transport.
        Provider {
            context: self.context.clone(),
            copy: copy_tensor,
        }
        .copy_state(source, evidence)
    }
    fn copy_state_with_owner(
        &self,
        owner: &PreparedEmbeddedCopy<MlxTensor>,
        source: &MlxTensor,
        evidence: Option<&PreparedEmbeddedEvidence>,
    ) -> Result<PreparedEmbeddedPayload<MlxTensor>, BackendFailure> {
        let parts = [
            size_of::<TensorCopyCustody>(),
            size_of::<RegisteredTensorSource>(),
            size_of::<crate::backend::array_copy::RegisteredArrayCopy>(),
            size_of::<Result<PreparedEmbeddedPayload<MlxTensor>, BackendFailure>>(),
            HostPreparationAuthority::retention_bytes::<TensorCopyCustody>()
                .ok_or_else(|| reject(&self.context, PreparedEmbeddedCopyError::Overflow))?,
        ];
        let _controls = pay(
            &self.context,
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| reject(&self.context, PreparedEmbeddedCopyError::Overflow))?,
        )?;
        let result = (|| -> Result<_, StartupCause> {
            let environment = self.context.loan()?;
            let completed = match evidence {
                Some(evidence) if evidence.get::<RegisteredTensorSource>().is_some() => {
                    let registered = evidence
                        .get::<RegisteredTensorSource>()
                        .expect("matched registered tensor");
                    if !registered.identity().same_identity(&self.identity) {
                        return Err(memory(WorkingMemoryError::IdentityMismatch));
                    }
                    registered.validate_stream(
                        environment.stream(),
                        self.context.preparation.metadata_funding(),
                    )?;
                    registered.validate_array(
                        source.as_array(),
                        self.context.preparation.metadata_funding(),
                    )?;
                    None
                }
                other => other
                    .map(|e| {
                        crate::composition::mlx::speculative::completed_tensor_source(e)
                            .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))
                    })
                    .transpose()?,
            };
            let copied = IsolatedArrayCopy::new(source.as_array()).copy_completed(
                completed,
                &environment,
                &self.context.roots,
                self.context.mechanisms,
                self.context.preparation.metadata_funding(),
                self.context.preparation.limits(),
            )?;
            let proof = RegisteredTensorSource::from_copy(
                &copied,
                self.identity.clone(),
                environment.stream(),
                self.context.preparation.metadata_funding(),
            )?;
            Ok((copied, proof))
        })()
        .map_err(|cause| failure(&self.context, cause.into()))?;
        // The still-closed copied owner guards the Array if evidence funding
        // refuses. Only after that final fallible step do its pieces move.
        let (copied, proof) = result;
        let evidence = owner.retain_evidence(proof)?;
        let (array, custody) = copied.into_parts();
        Ok(PreparedEmbeddedPayload::new(
            MlxTensor::from_array(array),
            HostPreparationAuthority::retain(TensorCopyCustody {
                _copy: custody,
                _host: self.context.host.clone(),
            }),
        )
        .with_evidence(evidence))
    }
    fn prepare_host(&self, bytes: usize) -> Result<HostPreparationAuthority, BackendFailure> {
        pay(&self.context, bytes)
    }
    fn reject(&self, cause: PreparedEmbeddedCopyError) -> BackendFailure {
        reject(&self.context, cause)
    }
    fn allocation_failure(&self, cause: TryReserveError) -> BackendFailure {
        failure(&self.context, Cause::Allocation(cause))
    }
}
enum TensorSource {
    Completed(CompletedTensorSource),
    Registered(RegisteredTensorSource),
}
enum TensorGuard {
    Completed(
        Option<(
            safemlx::OriginalBufferBudget,
            eredu_runtime::working_memory::OriginalSpeculativeNumericalBudgetCustody,
        )>,
        safemlx::OriginalBufferBudget,
        eredu_runtime::working_memory::CompletedWorkspaceSourceAccount,
    ),
    Registered(RegisteredTensorSource),
}
// A failed constructor retires the native value before its source inventory,
// parent view, backing guard, and this particular numerical operation's Q/H.
struct TensorInput {
    value: Option<MlxTensor>,
    parents: [Option<EmbeddedPredictionTensor<MlxTensor>>; 2],
    source: Option<TensorSource>,
    guard: Option<TensorGuard>,
    transport: Option<HostPreparationAuthority>,
}
impl OriginalEmbeddedCachePreparation {
    pub(crate) fn tensor_handoff_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<TensorInput>(),
            size_of::<Option<MlxTensor>>(),
            size_of::<Option<TensorSource>>(),
            size_of::<Option<TensorGuard>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn prepare_tensor(
        &self,
        value: MlxTensor,
        completed: CompletedResidentSource,
        sources: &OriginalSpeculativeNumericalSources,
    ) -> Result<EmbeddedPredictionTensor<MlxTensor>, BackendFailure> {
        self.prepare_tensor_with_transport(
            value,
            CompletedTensorSource {
                source: completed,
                operation: None,
            },
            sources,
            self.copy.host.clone(),
            None,
        )
    }
    pub(crate) fn prepare_tensor_with_transport(
        &self,
        value: MlxTensor,
        completed: CompletedTensorSource,
        sources: &OriginalSpeculativeNumericalSources,
        transport: HostPreparationAuthority,
        parent: Option<EmbeddedPredictionTensor<MlxTensor>>,
    ) -> Result<EmbeddedPredictionTensor<MlxTensor>, BackendFailure> {
        self.prepare_tensor_with_parents(value, completed, sources, transport, [parent, None])
    }
    pub(crate) fn prepare_tensor_with_parents(
        &self,
        value: MlxTensor,
        completed: CompletedTensorSource,
        sources: &OriginalSpeculativeNumericalSources,
        transport: HostPreparationAuthority,
        parents: [Option<EmbeddedPredictionTensor<MlxTensor>>; 2],
    ) -> Result<EmbeddedPredictionTensor<MlxTensor>, BackendFailure> {
        self.prepare_tensor_source(
            value,
            TensorSource::Completed(completed),
            sources,
            transport,
            parents,
        )
    }
    pub(crate) fn prepare_registered_tensor(
        &self,
        value: MlxTensor,
        registered: RegisteredTensorSource,
        sources: &OriginalSpeculativeNumericalSources,
        transport: HostPreparationAuthority,
        parent: Option<EmbeddedPredictionTensor<MlxTensor>>,
    ) -> Result<EmbeddedPredictionTensor<MlxTensor>, BackendFailure> {
        self.prepare_tensor_source(
            value,
            TensorSource::Registered(registered),
            sources,
            transport,
            [parent, None],
        )
    }
    fn prepare_tensor_source(
        &self,
        value: MlxTensor,
        source: TensorSource,
        sources: &OriginalSpeculativeNumericalSources,
        transport: HostPreparationAuthority,
        parents: [Option<EmbeddedPredictionTensor<MlxTensor>>; 2],
    ) -> Result<EmbeddedPredictionTensor<MlxTensor>, BackendFailure> {
        let mut input = TensorInput {
            value: Some(value),
            parents,
            source: Some(source),
            guard: None,
            transport: Some(transport),
        };
        self.validate_sources(sources)?;
        let parts = [
            EmbeddedPredictionTensor::<MlxTensor>::retained_state_control_bytes()
                .ok_or_else(|| reject(&self.copy, PreparedEmbeddedCopyError::Overflow))?,
            size_of::<PreparedEmbeddedState<MlxTensor>>(),
            size_of::<PreparedEmbeddedPayload<MlxTensor>>(),
            size_of::<TensorProvider>(),
            size_of::<TensorSource>(),
            size_of::<TensorGuard>(),
            CompletedResidentSource::array_source_control_bytes()
                .ok_or_else(|| reject(&self.copy, PreparedEmbeddedCopyError::Overflow))?,
            size_of::<Result<EmbeddedPredictionTensor<MlxTensor>, BackendFailure>>(),
        ];
        let host = pay(
            &self.copy,
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(|| reject(&self.copy, PreparedEmbeddedCopyError::Overflow))?,
        )?;
        (|| -> Result<(), StartupCause> {
            let environment = self.copy.loan()?;
            let value = input.value.as_ref().expect("unconsumed tensor");
            let funding = self.copy.preparation.metadata_funding();
            input.guard = Some(
                match input.source.as_ref().expect("unconsumed tensor source") {
                    TensorSource::Completed(completed) => {
                        completed.source.validate_request_source(
                            sources.request(),
                            environment.stream(),
                            funding,
                        )?;
                        let (budget, account) = completed
                            .source
                            .array_source_account(value.as_array(), funding)?;
                        TensorGuard::Completed(
                            completed.operation.clone(),
                            budget.clone(),
                            account.clone(),
                        )
                    }
                    TensorSource::Registered(registered) => {
                        registered.validate(sources.request(), environment.stream(), funding)?;
                        registered.validate_array(value.as_array(), funding)?;
                        TensorGuard::Registered(registered.clone())
                    }
                },
            );
            Ok(())
        })()
        .map_err(|cause| failure(&self.copy, cause.into()))?;
        let copy = PreparedEmbeddedCopy::prepare(TensorProvider {
            context: self.copy.clone(),
            identity: sources.request().source_identity(),
        })?;
        let evidence = match input.source.take().expect("unconsumed tensor source") {
            TensorSource::Completed(source) => copy.retain_evidence(source)?,
            TensorSource::Registered(source) => copy.retain_evidence(source)?,
        };
        Ok(EmbeddedPredictionTensor::from_prepared_state_with_sources(
            PreparedEmbeddedState::new(
                PreparedEmbeddedPayload::new(
                    input.value.take().expect("unconsumed tensor"),
                    input.transport.take().expect("unconsumed tensor transport"),
                )
                .with_evidence(evidence),
                copy,
            ),
            host,
            [input.parents[0].take(), input.parents[1].take()],
        ))
    }
}
