//! Construction destinations for existing workspace diagnostic reports.
use eredu_core::{
    Admission, AdmissionPolicyError, AdmissionRequest, AdmissionRequirements, AdmissionResult,
    AvailableMemory, BorrowedAdmissionResult, CapabilityError, ExecutionWorkspaceEstimate,
    ModelCapabilities, RuntimeStateEstimate, RuntimeStateFacts, SelectedStateBacking,
    StateMemoryAssumptions, WorkspaceBound,
};
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding},
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

/// A fixed composition refusal or an error owned by the participating Context.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceReportError {
    /// The shared geometry/arithmetic worker rejected the report.
    #[error(transparent)]
    Policy(#[from] AdmissionPolicyError),
    /// A metadata destination refused or failed before its allocation.
    #[error(transparent)]
    Metadata(#[from] Error),
}
impl WorkspaceReportError {
    // Ordinary destinations never return Metadata; keep a total adapter for
    // error propagation without introducing a panic on a future producer.
    /// Adapts an ordinary construction result to the existing reporting error.
    pub fn into_capability(self) -> CapabilityError {
        match self {
            Self::Policy(error) => error.into(),
            Self::Metadata(error) => CapabilityError::InvalidConfiguration {
                field: "workspace_report",
                detail: error.to_string(),
            },
        }
    }
}

/// Borrows the exact equation Context for owning diagnostic construction. This
/// carries no native authority and creates no independent allowance or lifetime.
/// The enclosing preparation owner must retain returned reports and failures.
#[derive(Clone, Copy)]
pub struct WorkspaceReportMetadata<'a>(Option<ReportDestination<'a>>);
#[derive(Clone, Copy)]
enum ReportDestination<'a> {
    Context(&'a WorkspaceContext),
    Funding(&'a HostMetadataFunding),
}
impl ReportDestination<'_> {
    fn funding(self) -> Option<HostMetadataFunding> {
        match self {
            Self::Context(context) => context.metadata_funding(),
            Self::Funding(funding) => Some(funding.clone()),
        }
    }
    fn charge_metadata(self, bytes: usize) -> Result<(), WorkspaceMetadataError> {
        match self {
            Self::Context(context) => context.charge_metadata(bytes),
            Self::Funding(funding) => funding.reserve_metadata(bytes).map_err(Into::into),
        }
    }
    fn metadata_source<E: std::error::Error + Send + Sync + 'static>(self, cause: E) -> Error {
        match self {
            Self::Context(context) => context.metadata_source(cause),
            Self::Funding(funding) => funding.metadata_source(cause),
        }
    }
    fn metadata_string(self, arguments: fmt::Arguments<'_>) -> Result<String, Error> {
        match self {
            Self::Context(context) => context.metadata_string(arguments),
            Self::Funding(funding) => funding.metadata_string(arguments),
        }
    }
    fn metadata_vec<T>(self, capacity: usize) -> Result<Vec<T>, Error> {
        match self {
            Self::Context(context) => context.metadata_vec(capacity),
            Self::Funding(funding) => funding.metadata_vec(capacity),
        }
    }
}
impl<'a> WorkspaceReportMetadata<'a> {
    /// Participates in this Context's checked/recording metadata budget. Legacy
    /// contexts keep the same ordinary diagnostic destination behavior.
    pub fn new(context: &'a WorkspaceContext) -> Self {
        Self(
            context
                .uses_checked_metadata()
                .then_some(ReportDestination::Context(context)),
        )
    }
    /// Borrows an existing planning account after its equation Context retires.
    /// No Context, new allowance or source credit is constructed. The caller
    /// must retain this same account through every returned report and error.
    pub fn with_funding(funding: &'a HostMetadataFunding) -> Self {
        Self(Some(ReportDestination::Funding(funding)))
    }
    /// Ordinary diagnostic construction; supplies no finite preparation claim.
    pub const fn ordinary() -> Self {
        Self(None)
    }
    /// Whether report owners participate in a checked metadata producer.
    pub fn is_checked(self) -> bool {
        self.0.is_some()
    }
    /// Closed alias of the current host planning account, without native authority.
    pub fn funding(self) -> Option<eredu_nn::workspace::HostMetadataFunding> {
        self.0.and_then(ReportDestination::funding)
    }
    /// Retains a typed producer failure only after paying its exact error owner.
    pub fn source<E: std::error::Error + Send + Sync + 'static>(self, error: E) -> Error {
        match self.0 {
            Some(context) => context.metadata_source(error),
            None => Error::backend_retained_source(error),
        }
    }
    /// Preserve an already paid error; pay before retaining a fixed policy cause.
    pub fn error(self, error: WorkspaceReportError) -> Error {
        match error {
            WorkspaceReportError::Metadata(error) => error,
            WorkspaceReportError::Policy(error) => match self.0 {
                Some(context) => context.metadata_source(error),
                None => Error::backend_retained_source(CapabilityError::from(error)),
            },
        }
    }
    /// Exact constructor/return transport for one emitted report value.
    pub fn admit<T>(self) -> Result<(), WorkspaceReportError> {
        if let Some(context) = self.0 {
            let parts = [
                size_of::<T>(),
                size_of::<Result<T, WorkspaceReportError>>(),
                size_of::<Self>(),
                size_of::<WorkspaceReportError>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)
                .map_err(Error::from)?;
            context.charge_metadata(bytes).map_err(Error::from)?;
        }
        Ok(())
    }
    /// Debits a separately owned exact constructor layout before invocation.
    pub fn charge(self, bytes: usize) -> Result<(), WorkspaceReportError> {
        if let Some(context) = self.0 {
            context.charge_metadata(bytes).map_err(Error::from)?;
        }
        Ok(())
    }
    /// Writes directly into the final counted text destination.
    pub fn text(self, text: fmt::Arguments<'_>) -> Result<String, WorkspaceReportError> {
        Ok(match self.0 {
            Some(context) => context.metadata_string(text)?,
            None => text.to_string(),
        })
    }
    /// Appends by preparing the complete replacement before retiring old text.
    pub fn append(self, text: &mut String, suffix: &str) -> Result<(), WorkspaceReportError> {
        match self.0 {
            Some(_) => *text = self.text(format_args!("{text}{suffix}"))?,
            None => text.push_str(suffix),
        }
        Ok(())
    }
    /// Constructs the existing known report variant without changing its value.
    pub fn bounded(
        self,
        bytes: u64,
        text: fmt::Arguments<'_>,
    ) -> Result<WorkspaceBound, WorkspaceReportError> {
        self.admit::<WorkspaceBound>()?;
        Ok(WorkspaceBound::bounded(bytes, self.text(text)?))
    }
    /// Constructs the existing missing-bound variant; absence remains explicit.
    pub fn unknown(self, text: fmt::Arguments<'_>) -> Result<WorkspaceBound, WorkspaceReportError> {
        self.admit::<WorkspaceBound>()?;
        Ok(WorkspaceBound::Unknown {
            reason: self.text(text)?,
        })
    }
    /// Copies only diagnostic text, never a source or execution allowance.
    pub fn clone_bound(
        self,
        source: &WorkspaceBound,
    ) -> Result<WorkspaceBound, WorkspaceReportError> {
        if self.0.is_none() {
            return Ok(source.clone());
        }
        match source {
            WorkspaceBound::Bounded { bytes, assumptions } => {
                self.bounded(*bytes, format_args!("{assumptions}"))
            }
            WorkspaceBound::Unknown { reason } => self.unknown(format_args!("{reason}")),
        }
    }
    /// Copies all six existing mandatory domains through counted destinations.
    pub fn clone_execution(
        self,
        source: &ExecutionWorkspaceEstimate,
    ) -> Result<ExecutionWorkspaceEstimate, WorkspaceReportError> {
        if self.0.is_none() {
            return Ok(source.clone());
        }
        self.admit::<ExecutionWorkspaceEstimate>()?;
        Ok(ExecutionWorkspaceEstimate {
            geometry: source.geometry,
            activations: self.clone_bound(&source.activations)?,
            attention: self.clone_bound(&source.attention)?,
            vocabulary: self.clone_bound(&source.vocabulary)?,
            state_update: self.clone_bound(&source.state_update)?,
            materialization: self.clone_bound(&source.materialization)?,
            retained: self.clone_bound(&source.retained)?,
        })
    }
    fn windows(
        self,
        source: impl ExactSizeIterator<Item = u64>,
    ) -> Result<Vec<u64>, WorkspaceReportError> {
        let mut values = match self.0 {
            Some(context) => context.metadata_vec(source.len())?,
            None => Vec::with_capacity(source.len()),
        };
        values.extend(source);
        Ok(values)
    }
    /// Owns the same borrowed logical-state facts without rerunning equations.
    pub fn state_from_facts(
        self,
        facts: RuntimeStateFacts<'_>,
    ) -> Result<RuntimeStateEstimate, WorkspaceReportError> {
        self.admit::<RuntimeStateEstimate>()?;
        let windows = self.windows(facts.sliding_windows.iter())?;
        Ok(facts.into_estimate_with_windows(windows)?)
    }
    /// Retains the existing admitted diagnostic using the same counted state
    /// copier. Scalar admission decisions are preserved without re-evaluation.
    pub fn clone_admission(self, source: &Admission) -> Result<Admission, WorkspaceReportError> {
        if !self.is_checked() {
            return Ok(source.clone());
        }
        self.admit::<Admission>()?;
        Ok(Admission {
            requested_positions: source.requested_positions,
            state: self.clone_state(&source.state)?,
            incremental_required_bytes: source.incremental_required_bytes,
            available_memory_bytes: source.available_memory_bytes,
        })
    }
    /// Runs the shared admission policy over borrowed reports, then constructs
    /// the owned result through this destination. The result is a diagnostic;
    /// runtime source validation and atomic reservation remain separate.
    pub fn apply_admission_with_incremental(
        self,
        capabilities: &ModelCapabilities,
        request: AdmissionRequest,
        state: &RuntimeStateEstimate,
        incremental: &WorkspaceBound,
        available: Option<&AvailableMemory>,
    ) -> Result<AdmissionResult, WorkspaceReportError> {
        self.admit::<AdmissionResult>()?;
        match eredu_core::apply_admission_requirements(
            request,
            AdmissionRequirements::from_reports(capabilities, state, Some(incremental), available),
        )? {
            BorrowedAdmissionResult::Admitted(decision) => {
                self.admit::<Admission>()?;
                Ok(AdmissionResult::Admitted(Admission {
                    requested_positions: decision.requested_positions,
                    state: self.clone_state(state)?,
                    incremental_required_bytes: decision.incremental_required_bytes,
                    available_memory_bytes: decision.available_memory_bytes,
                }))
            }
            BorrowedAdmissionResult::Rejected(rejection) => {
                self.admit::<eredu_core::AdmissionRejection>()?;
                Ok(AdmissionResult::Rejected(
                    rejection.try_into_owned_with(|text| self.text(text))?,
                ))
            }
        }
    }
    /// Clones the complete diagnostic including its exact window list and text.
    pub fn clone_state(
        self,
        source: &RuntimeStateEstimate,
    ) -> Result<RuntimeStateEstimate, WorkspaceReportError> {
        if self.0.is_none() {
            return Ok(source.clone());
        }
        self.admit::<RuntimeStateEstimate>()?;
        Ok(RuntimeStateEstimate {
            fixed_state_bytes: source.fixed_state_bytes,
            bytes_per_position_per_batch: source.bytes_per_position_per_batch,
            context_state_bytes: source.context_state_bytes,
            selected_state_backing: source
                .selected_state_backing
                .as_ref()
                .map(|value| {
                    Ok::<_, WorkspaceReportError>(SelectedStateBacking {
                        geometry: value.geometry,
                        bound: self.clone_bound(&value.bound)?,
                    })
                })
                .transpose()?,
            multimodal_embedding_bytes: source.multimodal_embedding_bytes,
            media_execution_workspace_bytes: source.media_execution_workspace_bytes,
            requested_state_bytes: source.requested_state_bytes,
            execution_workspace: source
                .execution_workspace
                .as_ref()
                .map(|value| self.clone_execution(value))
                .transpose()?,
            persistent_state_completeness: source.persistent_state_completeness,
            assumptions: StateMemoryAssumptions {
                floating_state_dtype_bytes: source.assumptions.floating_state_dtype_bytes,
                batch_size: source.assumptions.batch_size,
                requested_positions: source.assumptions.requested_positions,
                sliding_window_bounds: self
                    .windows(source.assumptions.sliding_window_bounds.iter().copied())?,
                allocation_granularity: source.assumptions.allocation_granularity,
            },
            completeness: source.completeness,
        })
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
