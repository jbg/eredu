//! Shared operation-fact construction and its independent metadata allowance.
use super::*;
use facts::owned::{FactEmissionFailure, FiniteWorkspaceFacts, OwnedOperationFacts};

/// Fixed metadata construction refusal. Creating or cloning this error never
/// formats text, allocates an error source, or changes any native authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WorkspaceMetadataError {
    /// The next concrete producer does not fit the admitted metadata remainder.
    #[error("workspace metadata requires {required} bytes, with {available} remaining")]
    Capacity { required: usize, available: usize },
    /// The actual retained account refused the next metadata constructor.
    #[error("{0}")]
    Funding(#[from] WorkspaceMetadataFundingError),
    /// A concrete population or layout cannot be represented.
    #[error("workspace metadata construction layout overflow")]
    Overflow,
    /// The actual metadata constructor lacks a qualified storage layout.
    #[error("workspace metadata constructor has no qualified layout")]
    Unqualified,
    /// A caller requested the legacy diagnostic-cloning report from a finite context.
    #[error("workspace metadata report requires the consuming report producer")]
    ReportClone,
    /// A consumed span must be replaced before it can execute or report again.
    #[error("workspace span report was already consumed")]
    ReportFinished,
    /// Formatting changed between count and bounded emission, or failed.
    #[error("workspace metadata text failed its counted emission")]
    Text,
    /// The shared scalar reducer rejected its supplied graph.
    #[error(transparent)]
    Report(#[from] WorkspaceReportError),
}
impl From<WorkspaceMetadataError> for Error {
    fn from(cause: WorkspaceMetadataError) -> Self {
        Self {
            storage: crate::ErrorStorage::WorkspaceMetadata(cause),
        }
    }
}

impl Error {
    /// Moves only the exact inline host-funding refusal. Every other error is
    /// returned unchanged, preserving its concrete source and retention owner.
    pub fn into_metadata_funding_error(self) -> Result<WorkspaceMetadataFundingError, Self> {
        match self.storage {
            crate::ErrorStorage::WorkspaceMetadata(WorkspaceMetadataError::Funding(cause)) => {
                Ok(cause)
            }
            _ => Err(self),
        }
    }
}

pub(super) enum OperationFacts {
    Ordinary(WorkspaceOperationBound),
    Finite(OwnedOperationFacts),
}
impl OperationFacts {
    pub(super) fn output(&self, index: usize) -> Option<WorkspaceOutputStorageView<'_>> {
        match self {
            Self::Ordinary(bound) => bound
                .outputs
                .get(index)
                .map(WorkspaceOutputStorage::as_view),
            Self::Finite(bound) => bound.output(index),
        }
    }
    pub(super) fn scratch_bytes(&self) -> u64 {
        match self {
            Self::Ordinary(v) => v.scratch_bytes,
            Self::Finite(v) => v.scratch_bytes,
        }
    }
    pub(super) fn into_assumptions(self) -> String {
        match self {
            Self::Ordinary(v) => v.assumptions,
            Self::Finite(v) => v.assumptions,
        }
    }
}

impl WorkspaceContext {
    /// Records exact fact-emission construction populations while executing the
    /// existing equations. This is cold ordinary planning, not admission. Shape,
    /// operation, state and report constructors remain separate populations.
    pub fn new_recording_facts<M>(mechanism: M) -> Self
    where
        M: WorkspaceMechanisms + WorkspaceFactMechanisms + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        Self::new_with_fact_budget(mechanism, usize::MAX)
    }

    /// Constructs the same context with a finite cumulative fact-emission
    /// allowance. Every span shares the remainder; beginning a span never
    /// refunds it. The caller separately admits the context itself and all
    /// other metadata producers. No native storage permission is created.
    pub fn new_with_fact_budget<M>(mechanism: M, bytes: usize) -> Self
    where
        M: WorkspaceMechanisms + WorkspaceFactMechanisms + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        let mechanism = Rc::new(mechanism);
        let identity = WorkspaceIdentity::new();
        identity.fact_bytes.set(Some(0));
        identity.metadata_bytes.set(Some(0));
        identity.metadata_reports.set(Some(0));
        identity.fact_remaining.set(bytes);
        Self {
            identity: Rc::new(identity),
            mechanisms: mechanism.clone(),
            facts: Some(mechanism),
            trace: Rc::new(RefCell::new(Trace::default())),
            borrowed: Rc::new(RefCell::new(None)),
            parameter_representations: Rc::new(RefCell::new(None)),
            tracing_started: Rc::new(report::Lifecycle::new()),
            funding: None,
        }
    }

    /// Constructs the recording context after reserving its actual shared
    /// shells and controls on the supplied host-metadata account. Participating
    /// metadata and fact producers reserve on this same account before each
    /// allocation; starting another span never refunds their reservations.
    ///
    /// Context retains funding only for its own lifetime. Tensors, layouts,
    /// reports and errors can outlive it: callers must retain an independent
    /// funding alias with every such result, including constructor failures.
    pub fn new_with_metadata_funding<M>(
        mechanism: M,
        funding: WorkspaceMetadataFunding,
    ) -> Result<Self, WorkspaceMetadataError>
    where
        M: WorkspaceMechanisms + WorkspaceFactMechanisms + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        let reserved = Self::construction_bytes::<M>()
            .ok_or(WorkspaceMetadataError::Overflow)
            .and_then(|bytes| funding.reserve_metadata(bytes).map_err(Into::into));
        if let Err(cause) = reserved {
            // A caller may supply a mechanism already retaining funded metadata.
            // Retire that value before releasing this last local funding alias.
            drop(mechanism);
            return Err(cause);
        }
        let mut context = Self::new_recording_facts(mechanism);
        context.funding = Some(funding);
        Ok(context)
    }

    /// Retains the same closed account for a result that can outlive Context.
    /// This creates no allowance, execution grant, raw owner or weak reference.
    pub fn metadata_funding(&self) -> Option<WorkspaceMetadataFunding> {
        self.funding.clone()
    }

    /// Supplies independently admitted capacities for the represented context
    /// and fact constructors. Counters are shared by all context clones.
    pub fn new_with_metadata_budget<M>(mechanism: M, envelope: WorkspaceMetadataEnvelope) -> Self
    where
        M: WorkspaceMechanisms + WorkspaceFactMechanisms + 'static,
        M::Error: std::error::Error + Send + Sync + 'static,
    {
        let value = Self::new_with_fact_budget(mechanism, envelope.fact_bytes());
        value
            .identity
            .metadata_remaining
            .set(envelope.context_bytes());
        value
    }

    pub(super) fn emit_operation_facts(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<(Option<OperationFacts>, Option<WorkspaceHostBound>), Error> {
        if let Some(facts) = &self.facts {
            let before = self.identity.fact_remaining.get();
            let mut remaining = before;
            let result = facts.emit(operation.as_view(), &mut remaining, self.funding.as_ref());
            self.identity.fact_remaining.set(remaining);
            self.identity.fact_bytes.set(
                self.identity
                    .fact_bytes
                    .get()
                    .and_then(|v| v.checked_add(before - remaining)),
            );
            return result
                .map(|v| (v.tensor.map(OperationFacts::Finite), v.host))
                .map_err(|cause| match cause {
                    FactEmissionFailure::Capacity {
                        required,
                        available,
                    } => WorkspaceMetadataError::Capacity {
                        required,
                        available,
                    }
                    .into(),
                    FactEmissionFailure::Overflow => WorkspaceMetadataError::Overflow.into(),
                    FactEmissionFailure::Funding(cause) => {
                        WorkspaceMetadataError::Funding(cause).into()
                    }
                    FactEmissionFailure::Retained(cause) => cause,
                });
        }
        // Preserve the legacy provider's invocation and validation order.
        let tensor = self.mechanisms.operation_bound(operation)?;
        if let Some(bound) = &tensor {
            validate_workspace_tensor_declaration(
                bound.outputs.len(),
                operation.outputs.len(),
                &bound.assumptions,
            )
            .map_err(WorkspaceEffectError::into_ordinary)?;
            for (index, (effect, output)) in
                bound.outputs.iter().zip(&operation.outputs).enumerate()
            {
                validate_workspace_output_storage(
                    effect.as_view(),
                    output.as_view(),
                    index,
                    operation.inputs.len(),
                )
                .map_err(WorkspaceEffectError::into_ordinary)?;
            }
        }
        let host = self.mechanisms.host_workspace_bound(operation)?;
        if let Some(host) = &host {
            validate_workspace_host_assumptions(&host.assumptions)
                .map_err(WorkspaceEffectError::into_ordinary)?;
        }
        Ok((tensor.map(OperationFacts::Ordinary), host))
    }
}
