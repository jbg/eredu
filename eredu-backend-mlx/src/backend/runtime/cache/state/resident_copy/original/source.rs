//! Completion-bound native source facts, never registry adoption.
use super::*;
use eredu_nn::workspace::{WorkspaceBorrowedStorage, WorkspaceExistingStorage};
use eredu_runtime::working_memory::{
    AdmittedWorkspaceCopy, CompletedWorkspaceSourceAccount, CompletedWorkspaceSourceLayout,
    CompletedWorkspaceStorage, CompletedWorkspaceStorageLayout, OriginalCompletedWorkspaceCopy,
    OriginalSpeculativeBudgetCustody, OriginalSpeculativeNumericalBudgetCustody,
    OriginalSpeculativeRole, RegisteredWorkspaceStorage,
};
use safemlx::{AllocationIdentity, OriginalBufferBudget, OriginalBufferCause};

#[derive(Clone, Copy, Debug)]
pub(super) struct BindingRows {
    registered: usize,
    completed: usize,
}

#[derive(Clone, Copy)]
pub(super) struct BindingDiagnostics<'a> {
    pub(super) stage: &'a Cell<&'static str>,
    pub(super) rows: &'a Cell<Option<BindingRows>>,
}
fn binding_stage(diagnostic: Option<BindingDiagnostics<'_>>, stage: &'static str) {
    if let Some(diagnostic) = diagnostic {
        diagnostic.stage.set(stage);
    }
}

struct Entry {
    identity: AllocationIdentity,
    bytes: u64,
    budget: OriginalBufferBudget,
    account: CompletedWorkspaceSourceAccount,
}
/// Exact completed role births held by the native state or its retained ingress.
/// It owns no array, plan or session. Existing registered roots stay on their original path.
/// The private state owner replaces this after each terminal successful span.
pub(crate) struct CompletedResidentSource {
    entries: Vec<Entry>,
    completed_stream: Option<safemlx::StreamCopyPlan<()>>,
    _funding: HostMetadataFunding,
}
/// Fixed inspection refusal; the borrowed source and caller keep custody.
#[derive(Debug, thiserror::Error)]
enum CompletedArraySourceError {
    #[error("completed array source does not match its retained allocation")]
    Identity,
    #[error(transparent)]
    Metadata(#[from] safemlx::ArrayMetadataError),
    #[error(transparent)]
    Budget(#[from] OriginalBufferCause),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct CompletedArraySourceFailure {
    #[source]
    cause: CompletedArraySourceError,
    _funding: HostMetadataFunding,
}
impl CompletedResidentSource {
    /// Called only after the role's exact completion, Recovery settlement and
    /// observer retirement. Prior roots reuse their actual budget/account;
    /// unrelated/ordinary roots receive no completed-role credit.
    pub(crate) fn capture(
        state: &PreparedResidentDecoderCopy<'_>,
        prior: Option<&Self>,
        budget: &OriginalBufferBudget,
        role: &OriginalSpeculativeRole,
        funding: &HostMetadataFunding,
        mut additional: impl FnMut(&mut dyn FnMut(&Array)) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        let prior = prior.map(|source| [source]);
        Self::capture_accounts_fallible(
            |visitor| {
                visit_sources(state, visitor).map_err(|cause| paid(funding, cause))?;
                additional(visitor)
            },
            prior.as_ref().map_or(&[], |source| source.as_slice()),
            Some((
                budget,
                CompletedWorkspaceSourceAccount::Model(role.budget_custody()),
            )),
            funding,
        )
    }
    /// Captures an exact current array inventory only after the owning model
    /// invocation has completed and its recovery/observer have retired. The
    /// caller supplies that invocation's account custody, never just an identity.
    /// This same worker covers target and architecture-declared prediction state.
    pub(crate) fn capture_array_sources(
        visit: impl FnMut(&mut dyn FnMut(&Array)),
        prior: Option<&Self>,
        budget: &OriginalBufferBudget,
        custody: &OriginalSpeculativeBudgetCustody,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let prior = prior.map(|source| [source]);
        Self::capture_array_sources_with_priors(
            visit,
            prior.as_ref().map_or(&[], |source| source.as_slice()),
            budget,
            custody,
            funding,
        )
    }
    /// The model phase consumes the exact lexical state and input source loans.
    /// Duplicate prior entries must agree on their physical account and capacity.
    pub(crate) fn capture_array_sources_with_priors(
        visit: impl FnMut(&mut dyn FnMut(&Array)),
        prior: &[&Self],
        budget: &OriginalBufferBudget,
        custody: &OriginalSpeculativeBudgetCustody,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::capture_accounts(
            visit,
            prior,
            Some((
                budget,
                CompletedWorkspaceSourceAccount::Model(custody.clone()),
            )),
            funding,
        )
    }
    /// Called only after the exact numerical phase's completion, Recovery and
    /// observer retirement. Existing view backings retain their original tag.
    pub(crate) fn capture_numerical_array_sources(
        visit: impl FnMut(&mut dyn FnMut(&Array)),
        prior: &[&Self],
        budget: &OriginalBufferBudget,
        custody: &OriginalSpeculativeNumericalBudgetCustody,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::capture_accounts(
            visit,
            prior,
            Some((
                budget,
                CompletedWorkspaceSourceAccount::Numerical(custody.clone()),
            )),
            funding,
        )
    }
    /// Reprojects only accounts already established by completed predecessors.
    /// Unlike model/numerical publication this has no birth account and cannot
    /// grant credit to an ordinary or newly allocated native backing.
    pub(crate) fn project_array_sources(
        visit: impl FnMut(&mut dyn FnMut(&Array)),
        prior: &[&Self],
        request: &eredu_runtime::working_memory::OriginalSpeculativeRequest,
        stream: &safemlx::Stream,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let frames = [
            size_of_val(&visit),
            size_of::<(
                &[&Self],
                &eredu_runtime::working_memory::OriginalSpeculativeRequest,
                &safemlx::Stream,
                &HostMetadataFunding,
            )>(),
            size_of::<Result<Self, Error>>(),
        ];
        reserve(
            funding,
            frames.into_iter().try_fold(size_of_val(&frames), add)?,
        )?;
        for source in prior {
            source.validate_request_source(request, stream, funding)?;
        }
        Self::capture_accounts(visit, prior, None, funding)?.with_completed_stream(stream)
    }

    fn capture_accounts(
        mut visit: impl FnMut(&mut dyn FnMut(&Array)),
        prior: &[&Self],
        birth: Option<(&OriginalBufferBudget, CompletedWorkspaceSourceAccount)>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        Self::capture_accounts_fallible(
            |visitor| {
                visit(visitor);
                Ok(())
            },
            prior,
            birth,
            funding,
        )
    }
    fn capture_accounts_fallible(
        mut visit: impl FnMut(&mut dyn FnMut(&Array)) -> Result<(), Error>,
        prior: &[&Self],
        birth: Option<(&OriginalBufferBudget, CompletedWorkspaceSourceAccount)>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let mut count = 0usize;
        let mut overflow = false;
        visit(&mut |_| match count.checked_add(1) {
            Some(n) => count = n,
            None => overflow = true,
        })?;
        if overflow {
            return Err(memory(WorkingMemoryError::Overflow));
        }
        let frames = [
            size_of::<Self>(),
            size_of_val(&visit),
            size_of::<CompletedWorkspaceSourceAccount>(),
            size_of::<Option<(&OriginalBufferBudget, CompletedWorkspaceSourceAccount)>>(),
            size_of::<(
                &CompletedWorkspaceSourceAccount,
                &CompletedWorkspaceSourceAccount,
            )>(),
            size_of::<bool>(),
            size_of::<&[&Self]>(),
            size_of::<Option<[&Self; 1]>>(),
            size_of::<std::slice::Iter<'_, &Self>>(),
            size_of::<&OriginalSpeculativeBudgetCustody>(),
            size_of::<&OriginalSpeculativeNumericalBudgetCustody>(),
            size_of::<Entry>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Option<&Entry>>(),
            size_of::<Option<Error>>(),
            size_of::<(AllocationIdentity, u64)>(),
            size_of::<Result<Option<safemlx::AllocationInfo>, safemlx::ArrayMetadataError>>(),
            OriginalBufferBudget::inspection_control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?,
            // The fixed cause precedes metadata_source, which pays its actual
            // erased destination before constructing it on a failure path.
            size_of::<safemlx::ArrayMetadataError>(),
            size_of::<OriginalBufferCause>(),
        ];
        reserve(
            funding,
            frames.into_iter().try_fold(size_of_val(&frames), add)?,
        )?;
        let mut entries = funding
            .metadata_vec::<Entry>(count)
            .map_err(|cause| paid(funding, cause))?;
        let mut failure = None;
        visit(&mut |array| {
            if failure.is_some() {
                return;
            }
            failure = (|| {
                let info = array
                    .try_allocation_info()
                    .map_err(|cause| paid(funding, cause))?
                    .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?;
                let identity = info.identity();
                let bytes = u64::try_from(info.bytes())
                    .map_err(|_| memory(WorkingMemoryError::Overflow))?;
                if let Some(old) = entries.iter().find(|entry| entry.identity == identity) {
                    if old.bytes != bytes {
                        return Err(memory(WorkingMemoryError::IdentityMismatch));
                    }
                    return Ok(());
                }
                let mut previous: Option<&Entry> = None;
                for source in prior {
                    if let Some(entry) = source.entry(identity) {
                        if previous.is_some_and(|old| {
                            old.bytes != entry.bytes || !old.account.same_account(&entry.account)
                        }) {
                            return Err(memory(WorkingMemoryError::IdentityMismatch));
                        }
                        previous = Some(entry);
                    }
                }
                let (owner, account) = if let Some(previous) = previous {
                    if previous.bytes != bytes {
                        return Err(memory(WorkingMemoryError::IdentityMismatch));
                    }
                    (&previous.budget, previous.account.clone())
                } else {
                    let (budget, custody) = birth
                        .as_ref()
                        .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
                    (*budget, custody.clone())
                };
                match owner.inspect_array(array) {
                    Ok(Some(witness)) => {
                        if witness.allocation().identity() != identity
                            || witness.allocation().bytes() as u64 != bytes
                        {
                            return Err(memory(WorkingMemoryError::IdentityMismatch));
                        }
                        if entries.len() == count {
                            return Err(memory(WorkingMemoryError::IdentityMismatch));
                        }
                        entries.push(Entry {
                            identity,
                            bytes,
                            budget: owner.clone(),
                            account,
                        });
                    }
                    // No new credit: every omitted root must pass the existing
                    // registry pin validator at the next actual copy boundary.
                    Ok(None) | Err(OriginalBufferCause::ForeignDomain) if previous.is_none() => {}
                    Ok(None) => return Err(memory(WorkingMemoryError::IdentityMismatch)),
                    Err(cause) => return Err(paid(funding, cause)),
                }
                Ok(())
            })()
            .err();
        })?;
        if let Some(cause) = failure {
            return Err(cause);
        }
        Ok(Self {
            entries,
            completed_stream: None,
            _funding: funding.clone(),
        })
    }
    /// Explicitly paid immutable descriptor copy for independently retained
    /// source owners. This copies no native arrays and grants no new birth,
    /// completion or execution authority.
    pub(crate) fn try_clone_for_source(
        &self,
        stream: &safemlx::Stream,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<(&Self, &safemlx::Stream, &HostMetadataFunding)>(),
            size_of::<Entry>(),
            size_of::<Vec<Entry>>(),
            size_of::<std::slice::Iter<'_, Entry>>(),
            size_of::<OriginalBufferBudget>(),
            size_of::<CompletedWorkspaceSourceAccount>(),
            size_of::<Option<safemlx::StreamCopyPlan<()>>>(),
            self.completed_stream_control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?,
        ];
        reserve(
            funding,
            frames.into_iter().try_fold(size_of_val(&frames), add)?,
        )?;
        self.validate_completed_stream(stream)?;
        let mut entries = funding
            .metadata_vec(self.entries.len())
            .map_err(|cause| paid(funding, cause))?;
        for entry in &self.entries {
            entries.push(Entry {
                identity: entry.identity,
                bytes: entry.bytes,
                budget: entry.budget.clone(),
                account: entry.account.clone(),
            });
        }
        Ok(Self {
            entries,
            completed_stream: self.completed_stream,
            _funding: funding.clone(),
        })
    }

    /// Attach the actual stream only at this source's completed publication
    /// boundary. No native stream owner or new queue is created.
    pub(crate) fn with_completed_stream(mut self, stream: &safemlx::Stream) -> Result<Self, Error> {
        let parts = [
            size_of::<safemlx::StreamCopyPlan<()>>(),
            size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<Result<Self, Error>>(),
        ];
        reserve(
            &self._funding,
            parts.into_iter().try_fold(size_of_val(&parts), add)?,
        )?;
        let plan = safemlx::StreamCopyPlan::<()>::capture(stream)
            .map_err(|cause| paid(&self._funding, cause))?;
        reserve(
            &self._funding,
            plan.control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
        )?;
        if self
            .completed_stream
            .as_ref()
            .is_some_and(|old| !old.matches_source(stream))
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        self.completed_stream = Some(plan);
        Ok(self)
    }
    pub(crate) fn completed_stream_control_bytes(&self) -> Option<usize> {
        self.completed_stream
            .as_ref()?
            .source_comparison_control_bytes()
    }
    /// Descriptive comparison with the actual completed publication. This does
    /// not authenticate its request/account or change that publication's stream.
    pub(crate) fn matches_completed_stream(&self, stream: &safemlx::Stream,
        funding: &HostMetadataFunding) -> Result<bool, Error> {
        let parts = [size_of::<(&Self, &safemlx::Stream, &HostMetadataFunding)>(),
            size_of::<Result<bool, Error>>(),
            self.completed_stream.as_ref().map_or(Some(0), |source|
                source.source_comparison_control_bytes()).ok_or_else(|| memory(WorkingMemoryError::Overflow))?];
        reserve(funding, parts.into_iter().try_fold(size_of_val(&parts), add)?)?;
        Ok(self.completed_stream.as_ref().is_some_and(|source| source.matches_source(stream)))
    }
    /// Reuses the actual publication's scalar stream identity. Equal device or
    /// request identity alone cannot substitute for this completed placement.
    pub(crate) fn validate_completed_stream(&self, stream: &safemlx::Stream) -> Result<(), Error> {
        if self
            .completed_stream
            .as_ref()
            .is_some_and(|source| source.matches_source(stream))
        {
            Ok(())
        } else {
            Err(memory(WorkingMemoryError::IdentityMismatch))
        }
    }
    /// Authenticate an immutable lexical input inventory without creating a
    /// source or role. Native consumers still validate each actual array/budget.
    pub(crate) fn validate_request_source(
        &self,
        request: &eredu_runtime::working_memory::OriginalSpeculativeRequest,
        stream: &safemlx::Stream,
        funding: &HostMetadataFunding,
    ) -> Result<(), Error> {
        let parts = [
            size_of::<(
                &Self,
                &eredu_runtime::working_memory::OriginalSpeculativeRequest,
                &safemlx::Stream,
            )>(),
            size_of::<std::slice::Iter<'_, Entry>>(),
            size_of::<eredu_runtime::working_memory::SpeculativeNumericalSource<'_>>(),
            size_of::<Result<(), Error>>(),
            self.completed_stream_control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?,
        ];
        reserve(
            funding,
            parts.into_iter().try_fold(size_of_val(&parts), add)?,
        )?;
        self.validate_completed_stream(stream)?;
        if self
            .entries
            .iter()
            .any(|entry| !entry.account.source().belongs_to_request(request))
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        Ok(())
    }
    /// Controls for the exact borrowed lookup below, paid by its owning caller
    /// before the output destination/native invocation is created.
    pub(crate) fn array_source_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<&Self>(),
            size_of::<&Array>(),
            size_of::<Option<&Entry>>(),
            size_of::<(&Self, &Array)>(),
            size_of::<HostMetadataFunding>(),
            size_of::<CompletedWorkspaceSourceAccount>(),
            size_of::<Result<(&OriginalBufferBudget, &CompletedWorkspaceSourceAccount), Error>>(),
            size_of::<
                Result<
                    (&OriginalBufferBudget, &CompletedWorkspaceSourceAccount),
                    CompletedArraySourceError,
                >,
            >(),
            size_of::<Result<(&OriginalBufferBudget, &OriginalSpeculativeBudgetCustody), Error>>(),
            size_of::<Result<Option<safemlx::AllocationInfo>, safemlx::ArrayMetadataError>>(),
            size_of::<CompletedArraySourceError>(),
            size_of::<CompletedArraySourceFailure>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<CompletedArraySourceFailure>(
            )?,
            size_of::<
                Result<
                    (&OriginalBufferBudget, &OriginalSpeculativeBudgetCustody),
                    CompletedArraySourceError,
                >,
            >(),
            OriginalBufferBudget::inspection_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Resolves only an actual array already present in this completed source.
    /// The budget rechecks native allocation identity/capacity; equal shapes or
    /// a current registry row cannot manufacture completion evidence.
    pub(crate) fn array_source(
        &self,
        array: &Array,
        funding: &HostMetadataFunding,
    ) -> Result<(&OriginalBufferBudget, &OriginalSpeculativeBudgetCustody), Error> {
        let (budget, account) = self.array_source_account(array, funding)?;
        match account {
            CompletedWorkspaceSourceAccount::Model(account) => Ok((budget, account)),
            CompletedWorkspaceSourceAccount::Numerical(_) => {
                Err(memory(WorkingMemoryError::IdentityMismatch))
            }
        }
    }
    /// Exact model/numerical account for an already completed full backing.
    /// The caller retains the supplied source while consuming this lexical loan.
    pub(crate) fn array_source_account(
        &self,
        array: &Array,
        funding: &HostMetadataFunding,
    ) -> Result<(&OriginalBufferBudget, &CompletedWorkspaceSourceAccount), Error> {
        let inspect = || -> Result<_, CompletedArraySourceError> {
            let info = array
                .try_allocation_info()?
                .ok_or(CompletedArraySourceError::Identity)?;
            let entry = self
                .entry(info.identity())
                .ok_or(CompletedArraySourceError::Identity)?;
            let witness = entry
                .budget
                .inspect_array(array)?
                .ok_or(CompletedArraySourceError::Identity)?;
            if u64::try_from(info.bytes()).ok() != Some(entry.bytes)
                || witness.allocation().identity() != entry.identity
                || witness.allocation().bytes() != info.bytes()
            {
                return Err(CompletedArraySourceError::Identity);
            }
            Ok((&entry.budget, &entry.account))
        };
        inspect().map_err(|cause| {
            Error::StorageSource(eredu_core::BackendFailure::from_error(
                CompletedArraySourceFailure {
                    cause,
                    _funding: funding.clone(),
                },
            ))
        })
    }
    fn entry(&self, identity: AllocationIdentity) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.identity == identity)
    }
}

pub(crate) enum SourceBinding {
    Registered(RegisteredWorkspaceStorage<StorageIdentity>),
    Completed(CompletedWorkspaceStorage<StorageIdentity>),
}
impl SourceBinding {
    pub(crate) fn borrowed(&self) -> &WorkspaceBorrowedStorage {
        match self {
            Self::Registered(source) => source.borrowed_storage(),
            Self::Completed(source) => source.borrowed_storage(),
        }
    }
    pub(super) fn admit(
        self,
        plan: WorkspaceIsolatedCopyPlan,
        environment: &OriginalCopyEnvironment<'_>,
        limits: WorkspaceCopyLimits,
        funding: &HostMetadataFunding,
    ) -> Result<AdmittedWorkspaceCopy, Error> {
        match self {
            Self::Registered(source) => environment.pool().admit_workspace_copy(
                RegisteredWorkspaceCopy::bind(plan, source)
                    .map_err(|cause| paid(funding, cause))?,
                limits,
            ),
            Self::Completed(source) => environment.pool().admit_completed_workspace_copy(
                OriginalCompletedWorkspaceCopy::bind(plan, source)
                    .map_err(|cause| paid(funding, cause))?,
                limits,
            ),
        }
        .map_err(|cause| paid(funding, cause))
    }
}
pub(super) fn bind(
    projected: &super::super::source_projection::ProjectedSnapshotInputs,
    completed: Option<&CompletedResidentSource>,
    environment: &OriginalCopyEnvironment<'_>,
    funding: &HostMetadataFunding,
    host: &HostPreparationAuthority,
    diagnostic: BindingDiagnostics<'_>,
) -> Result<SourceBinding, Error> {
    let completed = completed.map(|source| [source]);
    bind_projected_with_priors_selection(
        &projected.context,
        &projected.native,
        completed.as_ref().map_or(&[], |sources| sources.as_slice()),
        environment,
        funding,
        host,
        true,
        Some(diagnostic),
    )
}

/// Same exact completed/registered root binder for state and individual leaves.
/// No source credit is created by this borrowed projection alone.
pub(crate) fn bind_projected(
    context: &eredu_nn::workspace::WorkspaceContext,
    native: &crate::backend::nn::workspace::ProjectedNativeStorage,
    completed: Option<&CompletedResidentSource>,
    environment: &OriginalCopyEnvironment<'_>,
    funding: &HostMetadataFunding,
    host: &HostPreparationAuthority,
) -> Result<SourceBinding, Error> {
    let completed = completed.map(|source| [source]);
    bind_projected_with_priors(
        context,
        native,
        completed.as_ref().map_or(&[], |sources| sources.as_slice()),
        environment,
        funding,
        host,
    )
}

/// Same completed/registered binding over the finite lexical input and branch
/// inventories. Contradictory prior accounts never gain source credit.
pub(crate) fn bind_projected_with_priors(
    context: &eredu_nn::workspace::WorkspaceContext,
    native: &crate::backend::nn::workspace::ProjectedNativeStorage,
    completed: &[&CompletedResidentSource],
    environment: &OriginalCopyEnvironment<'_>,
    funding: &HostMetadataFunding,
    host: &HostPreparationAuthority,
) -> Result<SourceBinding, Error> {
    bind_projected_with_priors_selection(
        context,
        native,
        completed,
        environment,
        funding,
        host,
        true,
        None,
    )
}

pub(crate) fn bind_projected_unselected_with_priors(
    context: &eredu_nn::workspace::WorkspaceContext,
    native: &crate::backend::nn::workspace::ProjectedNativeStorage,
    completed: &[&CompletedResidentSource],
    environment: &OriginalCopyEnvironment<'_>,
    funding: &HostMetadataFunding,
    host: &HostPreparationAuthority,
) -> Result<SourceBinding, Error> {
    bind_projected_with_priors_selection(
        context,
        native,
        completed,
        environment,
        funding,
        host,
        false,
        None,
    )
}

fn bind_projected_with_priors_selection(
    context: &eredu_nn::workspace::WorkspaceContext,
    native: &crate::backend::nn::workspace::ProjectedNativeStorage,
    completed: &[&CompletedResidentSource],
    environment: &OriginalCopyEnvironment<'_>,
    funding: &HostMetadataFunding,
    host: &HostPreparationAuthority,
    select: bool,
    diagnostic: Option<BindingDiagnostics<'_>>,
) -> Result<SourceBinding, Error> {
    if !native.is_complete() {
        return Err(memory(WorkingMemoryError::UnknownBound));
    }
    let count = native.iter().len();
    let accounted = native
        .iter()
        .filter(|(id, _, _)| completed.iter().any(|source| source.entry(*id).is_some()))
        .count();
    if let Some(diagnostic) = diagnostic {
        diagnostic.rows.set(Some(BindingRows {
            registered: count - accounted,
            completed: accounted,
        }));
    }
    let carrier =
        OriginalStorageSourcesLayout::new(0).ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let frames = [
        size_of::<bool>(),
        size_of::<Option<BindingDiagnostics<'_>>>(),
        size_of::<BindingDiagnostics<'_>>(),
        size_of::<BindingRows>(),
        size_of::<Option<BindingRows>>(),
        size_of::<&'static str>(),
        size_of::<SourceBinding>(),
        size_of::<Result<SourceBinding, Error>>(),
        size_of::<(usize, usize)>(),
        size_of::<Option<&Entry>>(),
        size_of::<&[&CompletedResidentSource]>(),
        size_of::<Option<[&CompletedResidentSource; 1]>>(),
        size_of::<std::slice::Iter<'_, &CompletedResidentSource>>(),
        size_of::<Result<Option<&Entry>, Error>>(),
        carrier.requested_bytes(),
        OriginalBufferBudget::inspection_control_bytes()
            .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?,
    ];
    reserve(
        funding,
        frames.into_iter().try_fold(size_of_val(&frames), add)?,
    )?;
    if accounted == 0 {
        binding_stage(diagnostic, "registered source layout");
        let layout =
            RegisteredWorkspaceStorageLayout::<StorageIdentity>::new(count).map_err(memory)?;
        reserve(funding, layout.requested_bytes())?;
        binding_stage(diagnostic, "registered source carrier");
        let mut carrier = carrier
            .construct(environment.pool(), host)
            .map_err(|cause| paid(funding, cause))?;
        let rows = native
            .iter()
            .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone()));
        binding_stage(diagnostic, "registered source rows");
        let registered = if select {
            layout.construct(environment.pool(), context, rows)
        } else {
            layout.construct_unselected(environment.pool(), context, rows)
        }.map_err(memory)?;
        binding_stage(diagnostic, "registered source custody");
        let registered = registered.with_retained_original_sources(&mut carrier).map_err(memory)?;
        return Ok(SourceBinding::Registered(registered));
    }
    binding_stage(diagnostic, "completed source layouts");
    let source_layout = CompletedWorkspaceSourceLayout::new_accounts(accounted).map_err(memory)?;
    let layout =
        CompletedWorkspaceStorageLayout::<StorageIdentity>::new(count - accounted, accounted)
            .map_err(memory)?;
    reserve(
        funding,
        add(
            source_layout.requested_bytes(),
            layout.requested_bytes().map_err(memory)?,
        )?,
    )?;
    let mut entries = funding
        .metadata_vec::<(WorkspaceExistingStorage, CompletedWorkspaceSourceAccount)>(accounted)
        .map_err(|cause| paid(funding, cause))?;
    binding_stage(diagnostic, "completed source witnesses");
    for (id, bytes, root) in native.iter() {
        let mut selected: Option<&Entry> = None;
        for source in completed {
            if let Some(entry) = source.entry(id) {
                if selected.is_some_and(|previous| {
                    previous.bytes != entry.bytes || !previous.account.same_account(&entry.account)
                }) {
                    return Err(memory(WorkingMemoryError::IdentityMismatch));
                }
                selected = Some(entry);
            }
        }
        if let Some(entry) = selected {
            let array = native
                .native_array(id)
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
            let witness = entry
                .budget
                .inspect_array(array)
                .map_err(|cause| paid(funding, cause))?
                .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?;
            if entry.bytes != bytes
                || witness.allocation().identity() != id
                || witness.allocation().bytes() as u64 != bytes
            {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            entries.push((root.clone(), entry.account.clone()));
        }
    }
    binding_stage(diagnostic, "completed source accounts");
    let source = source_layout
        .construct_accounts(context, entries, host)
        .map_err(memory)?;
    binding_stage(diagnostic, "completed source carrier");
    let mut carrier = carrier
        .construct(environment.pool(), host)
        .map_err(|cause| paid(funding, cause))?;
    let rows = native
        .iter()
        .filter(|(id, _, _)| !completed.iter().any(|source| source.entry(*id).is_some()))
        .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone()));
    binding_stage(diagnostic, "completed source rows");
    let registered = if select {
        layout.construct(environment.pool(), context, rows, source, &mut carrier)
    } else {
        layout.construct_unselected(environment.pool(), context, rows, source, &mut carrier)
    }
    .map_err(memory)?;
    Ok(SourceBinding::Completed(registered))
}

fn visit_sources(
    state: &PreparedResidentDecoderCopy<'_>,
    visitor: &mut dyn FnMut(&Array),
) -> Result<(), super::super::SnapshotProjectionCause> {
    state.visit_operands(visitor)?;
    state.visit_retained_arrays(visitor)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
