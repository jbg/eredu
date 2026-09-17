//! Whole pending-input copy through the existing registered-copy account and
//! original native copy/completion engine. Per-span views are separate work.
use super::*;
use crate::backend::{
    OriginalCopyEnvironment,
    array_copy::{OriginalCopyLayoutBuilder, OriginalCopyPlan},
    error::Error,
    nn::workspace::{MlxMetalWorkspaceMechanisms, ProjectionSourceLayout},
    runtime::residency::storage::{CopyPublicationLayout, RetainedStorage, StorageIdentity},
    submission_recovery::{Retention, Status},
};
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceCopyPreparationLayoutBuilder, WorkspaceIsolatedCopyPlan,
    WorkspaceMetadataFunding,
};
use eredu_runtime::working_memory::{
    OriginalStorageSourcesLayout, RegisteredPreparedWorkspaceCopy,
    RegisteredPreparedWorkspaceStorage, RegisteredWorkspaceCopy, RegisteredWorkspaceStorage,
    RegisteredWorkspaceStorageLayout, WorkingMemoryError, WorkingMemoryFundingScope,
    WorkspaceCopyAccountLayout, WorkspaceCopyLimits, WorkspaceCopyRetention,
};
use crate::backend::runtime::cache::state::{
    CompletedResidentSource, OriginalResidentSourceBinding, bind_completed_resident_sources,
};
use eredu_runtime::working_memory::{CompletedWorkspaceStorage, OriginalCompletedWorkspaceCopy};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
    rc::Rc,
};

/// No raw native owner or request role is retained by the accounting token.
/// Arrays and the exact input payload retire before their Q and H custody.
pub(crate) struct RegisteredArrayCopy {
    array: Array,
    completed_stream: safemlx::StreamCopyPlan<()>,
    custody: RegisteredArrayCopyCustody,
}
/// Account-only copy/H custody. Every consuming owner drops its Array first.
#[derive(Clone)]
pub(crate) struct RegisteredArrayCopyCustody {
    _copy: WorkspaceCopyRetention,
    _host: HostPreparationAuthority,
}
impl RegisteredArrayCopyCustody {
    /// Lexical loan of this closed published copy's actual account.
    pub(crate) fn copy_retention(&self) -> &WorkspaceCopyRetention { &self._copy }
}
impl RegisteredArrayCopy {
    pub(crate) fn custody(&self) -> &RegisteredArrayCopyCustody { &self.custody }
    pub(crate) fn copy_retention(&self) -> &WorkspaceCopyRetention {
        self.custody.copy_retention()
    }
    pub(crate) fn validate_completed_stream(
        &self, stream: &Stream, funding: &WorkspaceMetadataFunding,
    ) -> Result<(), Error> {
        reserve(funding, self.completed_stream.source_comparison_control_bytes()
            .ok_or_else(||memory(WorkingMemoryError::Overflow))?)?;
        if self.completed_stream.matches_source(stream) { Ok(()) }
        else { Err(memory(WorkingMemoryError::IdentityMismatch)) }
    }
    pub(crate) fn array(&self) -> &Array {
        &self.array
    }
    pub(crate) fn into_parts(self) -> (Array, RegisteredArrayCopyCustody) {
        (self.array, self.custody)
    }
}
#[derive(Clone, Copy)]
enum Proof<'a> {
    Registered,
    Completed(&'a CompletedResidentSource),
    Prepared(&'a crate::backend::array_copy::OriginalPreparedArrayCopySource<'a>),
    Numerical(
        &'a safemlx::OriginalBufferBudget,
        &'a eredu_runtime::working_memory::OriginalSpeculativeNumericalBudgetCustody,
    ),
}
enum Source {
    Completed(CompletedWorkspaceStorage<StorageIdentity>),
    Registered(RegisteredWorkspaceStorage<StorageIdentity>),
    Prepared(RegisteredPreparedWorkspaceStorage<StorageIdentity>),
    Numerical {
        roots: eredu_nn::workspace::WorkspaceBorrowedStorage,
        custody: eredu_runtime::working_memory::OriginalSpeculativeNumericalBudgetCustody,
        host: HostPreparationAuthority,
    },
}
impl Source {
    fn borrowed(&self) -> &eredu_nn::workspace::WorkspaceBorrowedStorage {
        match self {
            Self::Completed(source) => source.borrowed_storage(),
            Self::Registered(source) => source.borrowed_storage(),
            Self::Prepared(source) => source.borrowed_storage(),
            Self::Numerical { roots, .. } => roots,
        }
    }
    fn bind(
        self,
        plan: WorkspaceIsolatedCopyPlan,
    ) -> Result<Copy, eredu_runtime::working_memory::WorkspaceCopyAdmissionError> {
        match self {
            Self::Completed(source) => OriginalCompletedWorkspaceCopy::bind(plan, source).map(Copy::Completed),
            Self::Numerical {
                roots,
                custody,
                host,
            } => eredu_runtime::working_memory::OriginalNumericalWorkspaceCopy::bind(
                plan, &roots, custody, host,
            )
            .map(Copy::Numerical),
            Self::Registered(source) => {
                RegisteredWorkspaceCopy::bind(plan, source).map(Copy::Registered)
            }
            Self::Prepared(source) => {
                RegisteredPreparedWorkspaceCopy::bind(plan, source).map(Copy::Prepared)
            }
        }
    }
}
enum Copy {
    Completed(OriginalCompletedWorkspaceCopy<StorageIdentity>),
    Registered(RegisteredWorkspaceCopy<StorageIdentity>),
    Prepared(RegisteredPreparedWorkspaceCopy<StorageIdentity>),
    Numerical(eredu_runtime::working_memory::OriginalNumericalWorkspaceCopy),
}
impl Copy {
    fn admit(
        self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        limits: WorkspaceCopyLimits,
    ) -> Result<
        eredu_runtime::working_memory::AdmittedWorkspaceCopy,
        eredu_runtime::working_memory::WorkspaceCopyAdmissionError,
    > {
        match self {
            Self::Completed(copy) => pool.admit_completed_workspace_copy(copy, limits),
            Self::Registered(copy) => pool.admit_workspace_copy(copy, limits),
            Self::Prepared(copy) => pool.admit_prepared_workspace_copy(copy, limits),
            Self::Numerical(copy) => pool.admit_numerical_workspace_copy(copy, limits),
        }
    }
}
struct Work {
    roots: RefCell<Vec<Array>>,
    scope: RefCell<Option<WorkingMemoryFundingScope>>,
    discarded: Cell<bool>,
    healthy: Cell<bool>,
    _copy: WorkspaceCopyRetention,
    _host: HostPreparationAuthority,
}
#[derive(Clone)]
struct Owner(Option<Rc<Work>>);
impl Owner {
    fn work(&self) -> &Work {
        self.0.as_deref().expect("live pending copy")
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
impl Retention for Owner {
    fn observe(&self, status: Status) {
        self.work()
            .healthy
            .set(status.settled && !status.failed && !status.blocked);
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        if self.discarded.get() && self.healthy.get() {
            // The shared native Recovery drops its probe before this payload;
            // the lexical discard guard follows every provisional destination.
            drop(std::mem::take(self.roots.get_mut()));
            if let Some(scope) = self.scope.get_mut().take() {
                let _ = scope.certify();
            }
        }
        // A failed/unsettled prefix remains under the existing quarantine rule.
    }
}
struct Discard(Option<Owner>);
impl Drop for Discard {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.work().discarded.set(true);
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Error,
    _host: HostPreparationAuthority,
}
fn memory(cause: WorkingMemoryError) -> Error {
    Error::PrefillControl(cause)
}
fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
}
fn paid(
    funding: &WorkspaceMetadataFunding,
    cause: impl std::error::Error + Send + Sync + 'static,
) -> Error {
    Error::Neural(funding.metadata_source(cause))
}
fn reserve(funding: &WorkspaceMetadataFunding, bytes: usize) -> Result<(), Error> {
    funding
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)
}
enum Program<'a> {
    Pending(PreparedPendingTokenInput<'a>),
    Isolated(&'a Array),
}
impl Program<'_> {
    fn source(&self) -> &Array {
        match self {
            Self::Pending(input) => input.source(),
            Self::Isolated(source) => source,
        }
    }
    fn retained_descriptor_count(&self) -> usize {
        match self {
            Self::Pending(input) => input.retained_descriptor_count(),
            Self::Isolated(_) => 2,
        }
    }
    fn native<'a>(
        &self,
        environment: &'a OriginalCopyEnvironment<'_>,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<OriginalCopyPlan<'a>, Error> {
        if let Self::Pending(input) = self {
            input.validate_fixed().map_err(|e| paid(funding, e))?;
        }
        let mut native = OriginalCopyLayoutBuilder::new();
        native
            .push_retained_source(self.source())
            .map_err(|e| paid(funding, e))?;
        match self {
            Self::Pending(input) => native
                .finish_pending_input(input, environment)
                .map_err(|e| paid(funding, e)),
            Self::Isolated(source) => {
                native.push_operand(source).map_err(|e| paid(funding, e))?;
                native
                    .finish(environment)
                    .map_err(|e| paid(funding, e))?
                    .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))
            }
        }
    }
    fn copy_retained(
        self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Array, Error> {
        match self {
            Self::Pending(input) => input
                .copy_retained(stream, roots)
                .map(|v| v.into_array())
                .map_err(|e| paid(funding, e)),
            Self::Isolated(source) => IsolatedArrayCopy::new(source)
                .copy_retained(stream, roots)
                .map_err(|e| paid(funding, e)),
        }
    }
}
impl PreparedPendingTokenInput<'_> {
    pub(crate) fn copy_registered(
        self,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        funding: &WorkspaceMetadataFunding,
        capacity: u64,
    ) -> Result<RegisteredArrayCopy, Error> {
        Program::Pending(self).copy_registered(
            Proof::Registered,
            environment,
            initialized,
            mechanisms,
            funding,
            capacity,
        )
    }
    pub(crate) fn copy_registered_with_prepared(
        self,
        prepared: Option<&crate::backend::array_copy::OriginalPreparedArrayCopySource<'_>>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        funding: &WorkspaceMetadataFunding,
        capacity: u64,
    ) -> Result<RegisteredArrayCopy, Error> {
        Program::Pending(self).copy_registered(
            prepared.map_or(Proof::Registered, Proof::Prepared),
            environment,
            initialized,
            mechanisms,
            funding,
            capacity,
        )
    }
}
impl IsolatedArrayCopy<'_> {
    /// Preserve the actual prepared B leaf dtype through the same isolated-copy
    /// worker and original-source binding used by all other completed copies.
    pub(crate) fn copy_prepared(
        self, prepared:&crate::backend::array_copy::OriginalPreparedArrayCopySource<'_>,
        environment:&OriginalCopyEnvironment<'_>,initialized:&safemlx::PrefillRootsRuntime,
        mechanisms:MlxMetalWorkspaceMechanisms,funding:&WorkspaceMetadataFunding,capacity:u64,
    )->Result<RegisteredArrayCopy,Error>{
        Program::Isolated(self.source).copy_registered(Proof::Prepared(prepared),
            environment,initialized,mechanisms,funding,capacity)
    }

    /// Reuses the completed-state root binder for one exact current cache leaf.
    /// Missing completed entries still require real registered source pins.
    pub(crate) fn copy_completed(
        self,
        completed: Option<&CompletedResidentSource>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        funding: &WorkspaceMetadataFunding,
        capacity: u64,
    ) -> Result<RegisteredArrayCopy, Error> {
        Program::Isolated(self.source).copy_registered(
            completed.map_or(Proof::Registered, Proof::Completed),
            environment, initialized, mechanisms, funding, capacity,
        )
    }
    /// A source is either an existing published copy or the exact immutable
    /// completed numerical birth authenticated against its retained native budget.
    pub(crate) fn copy_original(
        self,
        numerical: Option<(
            &safemlx::OriginalBufferBudget,
            &eredu_runtime::working_memory::OriginalSpeculativeNumericalBudgetCustody,
        )>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        funding: &WorkspaceMetadataFunding,
        capacity: u64,
    ) -> Result<RegisteredArrayCopy, Error> {
        Program::Isolated(self.source).copy_registered(
            numerical.map_or(Proof::Registered, |(b, c)| Proof::Numerical(b, c)),
            environment,
            initialized,
            mechanisms,
            funding,
            capacity,
        )
    }
}
impl Program<'_> {
    fn copy_registered(
        self,
        proof: Proof<'_>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        funding: &WorkspaceMetadataFunding,
        capacity: u64,
    ) -> Result<RegisteredArrayCopy, Error> {
        let frames = [
            PreparedPendingTokenInput::source_control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            OriginalCopyLayoutBuilder::resume_control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            BackendFailure::source_retention_peak_bytes::<Failure>()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            size_of::<Self>(),
            size_of::<Proof<'_>>(),
            size_of::<RegisteredArrayCopyCustody>(),
            eredu_runtime::working_memory::OriginalNumericalWorkspaceCopy::control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            safemlx::OriginalBufferBudget::inspection_control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?,
            size_of::<Source>(),
            size_of::<OriginalResidentSourceBinding>(),
            size_of::<Result<OriginalResidentSourceBinding, Error>>(),
            size_of::<Option<&CompletedResidentSource>>(),
            size_of::<Copy>(),
            RegisteredPreparedWorkspaceCopy::<StorageIdentity>::control_bytes()
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            size_of::<Option<&crate::backend::array_copy::OriginalPreparedArrayCopySource<'_>>>(),
            size_of::<Option<eredu_runtime::input::OriginalPreparedWorkspaceSource>>(),
            size_of::<RegisteredArrayCopy>(),
            size_of::<Result<RegisteredArrayCopy, Error>>(),
            size_of::<Failure>(),
            size_of::<Owner>(),
            size_of::<Work>(),
            size_of::<Discard>(),
            size_of::<OriginalCopyLayoutBuilder>(),
            size_of::<OriginalCopyPlan<'_>>(),
            size_of::<Option<WorkingMemoryFundingScope>>(),
            size_of::<std::cell::RefMut<'_, Vec<Array>>>(),
            size_of::<std::cell::RefMut<'_, Option<WorkingMemoryFundingScope>>>(),
            size_of::<Error>(),
            size_of::<Option<Error>>(),
        ];
        reserve(
            funding,
            frames.into_iter().try_fold(size_of_val(&frames), add)?,
        )?;
        let host = HostPreparationAuthority::retain(funding.clone());
        self.copy_registered_inner(
            proof,
            environment,
            initialized,
            mechanisms,
            funding,
            &host,
            capacity,
        )
        .map_err(|cause| {
            Error::StorageSource(BackendFailure::from_error(Failure { cause, _host: host }))
        })
    }
    fn copy_registered_inner(
        self,
        proof: Proof<'_>,
        environment: &OriginalCopyEnvironment<'_>,
        initialized: &safemlx::PrefillRootsRuntime,
        mechanisms: MlxMetalWorkspaceMechanisms,
        funding: &WorkspaceMetadataFunding,
        host: &HostPreparationAuthority,
        capacity: u64,
    ) -> Result<RegisteredArrayCopy, Error> {
        let stream_frames = [
            size_of::<safemlx::StreamCopyPlan<()>>(),
            size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
        ];
        reserve(funding, stream_frames.into_iter().try_fold(size_of_val(&stream_frames), add)?)?;
        let completed_stream=safemlx::StreamCopyPlan::<()>::capture(environment.stream())
            .map_err(|cause|paid(funding,cause))?;
        reserve(funding,completed_stream.control_bytes()
            .ok_or_else(||memory(WorkingMemoryError::Overflow))?)?;
        if let Proof::Prepared(source) = proof {
            if !std::ptr::eq(self.source(), source.array()) {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
        }
        let root_count = self
            .retained_descriptor_count()
            .checked_add(1)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let native = self.native(environment, funding)?;
        let origin = match proof {
            Proof::Numerical(budget, _) => Some(
                budget
                    .inspect_array(self.source())
                    .map_err(|e| paid(funding, e))?
                    .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?
                    .allocation(),
            ),
            _ => None,
        };

        let context = WorkspaceContext::new_with_metadata_funding(mechanisms, funding.clone())
            .map_err(|e| paid(funding, e))?;
        let (source, projected, prepared) = match proof {
            Proof::Prepared(source) => {
                let (value, witness) = source.project(&context).map_err(|e| paid(funding, e))?;
                (value, None, Some(witness))
            }
            _ => {
                let mut projection = ExistingArrayProjection::new(&context);
                let source = projection
                    .project(self.source())
                    .map_err(|e| paid(funding, e))?;
                let projected = projection
                    .try_into_storage()
                    .map_err(|e| paid(funding, e))?;
                if !projected.is_complete() {
                    return Err(memory(WorkingMemoryError::UnknownBound));
                }
                (source, Some(projected), None)
            }
        };
        let mut layout = WorkspaceCopyPreparationLayoutBuilder::new();
        ProjectionSourceLayout::with_layout(self.source(), |source| {
            layout.push_source(source, &mechanisms)
        })
        .map_err(|e| paid(funding, e))?
        .map_err(|e| paid(funding, e))?;
        let layout = layout.finish(1).map_err(|e| paid(funding, e))?;
        let registration = match prepared.as_ref() {
            Some(source) => {
                RegisteredWorkspaceStorageLayout::<StorageIdentity>::new_with_prepared_source(
                    0, source,
                )
            }
            None => RegisteredWorkspaceStorageLayout::<StorageIdentity>::new(1),
        }
        .map_err(memory)?;
        let carrier = OriginalStorageSourcesLayout::new(0)
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        let publication = CopyPublicationLayout::new(1, 0).map_err(memory)?;
        let controls = [
            layout.total_bytes(),
            registration.requested_bytes(),
            carrier.requested_bytes(),
            publication.control_bytes(),
            (if prepared.is_some() {
                WorkspaceCopyAccountLayout::workspace_with_prepared_source()
            } else {
                WorkspaceCopyAccountLayout::workspace()
            })
            .map_err(memory)?
            .requested_bytes(),
            RetainedStorage::original_collector_control_bytes(1)
                .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?,
            native
                .control_bytes::<Owner>()
                .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?,
            Layout::new::<[Cell<usize>; 2]>()
                .extend(Layout::new::<Work>())
                .map_err(|_| memory(WorkingMemoryError::Overflow))?
                .0
                .pad_to_align()
                .size(),
        ];
        reserve(
            funding,
            controls.into_iter().try_fold(size_of_val(&controls), add)?,
        )?;
        let mut carrier = carrier
            .construct(environment.pool(), host)
            .map_err(|e| paid(funding, e))?;
        let registered = if let Proof::Completed(completed) = proof {
            let projected = projected.as_ref().ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
            match bind_completed_resident_sources(&context, projected, Some(completed), environment, funding, host)? {
                OriginalResidentSourceBinding::Registered(source) => Source::Registered(source),
                OriginalResidentSourceBinding::Completed(source) => Source::Completed(source),
            }
        } else if let Proof::Numerical(_, custody) = proof {
            let projected = projected
                .as_ref()
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
            let mut rows = projected.iter();
            let (id, bytes, root) = rows
                .next()
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
            let origin = origin.ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
            if rows.next().is_some() || id != origin.identity() || bytes != origin.bytes() as u64 {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            let roots =
                eredu_nn::workspace::WorkspaceBorrowedStorage::new_finite(&context, [root], 1)
                    .map_err(|e| paid(funding, e))?;
            context
                .set_borrowed_storage_checked(roots.clone())
                .map_err(|e| paid(funding, e))?;
            Source::Numerical {
                roots,
                custody: custody.clone(),
                host: host.clone(),
            }
        } else {
            match prepared {
                Some(source) => Source::Prepared(
                    registration
                        .construct_with_prepared_source(
                            environment.pool(),
                            &context,
                            std::iter::empty(),
                            source,
                        )
                        .and_then(|source| source.with_retained_original_sources(&mut carrier))
                        .map_err(memory)?,
                ),
                None => Source::Registered(
                    registration
                        .construct(
                            environment.pool(),
                            &context,
                            projected
                                .as_ref()
                                .expect("registered projection")
                                .iter()
                                .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone())),
                        )
                        .and_then(|source| source.with_retained_original_sources(&mut carrier))
                        .map_err(memory)?,
                ),
            }
        };
        let program = WorkspaceIsolatedCopyPlan::prepare_finite_with_layout(
            &context,
            registered.borrowed(),
            &[source],
            &mechanisms,
            layout,
        )
        .map_err(|e| paid(funding, e))?
        .construct()
        .map_err(|e| paid(funding, e))?;
        let numerical = program
            .incremental_bytes()
            .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?;
        let copy = registered.bind(program).map_err(|e| paid(funding, e))?;
        let mut limits = WorkspaceCopyLimits::new(capacity);
        // The native population includes the same isolated copy plus its real
        // reshape/optional cast and both nested completions. Charge only the
        // positive physical delta over the registered isolated-copy component.
        limits.safety_reserve_bytes = u64::try_from(native.physical_bytes())
            .map_err(|_| memory(WorkingMemoryError::Overflow))?
            .saturating_sub(numerical);
        let account = copy
            .admit(environment.pool(), limits)
            .map_err(|e| paid(funding, e))?;
        let (custody, scope) = account.into_parts();
        let roots = match funding.metadata_vec(root_count) {
            Ok(value) => value,
            Err(cause) => {
                let _ = scope.certify();
                return Err(paid(funding, cause));
            }
        };
        let prepared = match native.prepare(&custody, initialized) {
            Ok(value) => value,
            Err(cause) => {
                let _ = scope.certify();
                return Err(cause.into());
            }
        };
        let pending = match publication.construct(&scope, &custody, host, Some(prepared.budget())) {
            Ok(value) => value,
            Err(cause) => {
                let _ = scope.certify();
                return Err(cause);
            }
        };
        let inventory =
            match RetainedStorage::prepare_snapshot_publication(1, environment.pool(), host) {
                Ok(value) => value,
                Err(cause) => {
                    let _ = scope.certify();
                    return Err(cause.into());
                }
            };
        let owner = Owner(Some(Rc::new(Work {
            roots: RefCell::new(roots),
            scope: RefCell::new(Some(scope)),
            discarded: Cell::new(false),
            healthy: Cell::new(false),
            _copy: custody.retention(),
            _host: host.clone(),
        })));
        let mut discarded = Discard(Some(owner.clone()));
        let mut pending = pending;
        let mut inventory = inventory;
        let (mut recovery, mut execution) = prepared.begin(owner.clone())?;
        struct Construction<'a>(&'a mut crate::backend::array_copy::OriginalCopyExecution);
        impl Drop for Construction<'_> {
            fn drop(&mut self) {
                self.0.finish_construction();
            }
        }
        let result = {
            let _construction = Construction(&mut execution);
            let roots = &owner.work().roots;
            (|| {
                roots.borrow_mut().push(self.source().try_clone_handle()?);
                let array = self.copy_retained(environment.stream(), roots, funding)?;
                let observer = safemlx::OriginalScopeObserver::require_current()?;
                array.completed_in_original_scope(&observer)?;
                Ok::<_, Error>(array)
            })()
        };
        recovery.seal();
        let array = result?;
        let status = recovery.finish();
        if !status.settled || status.failed || status.blocked {
            return Err(memory(WorkingMemoryError::UnknownBound));
        }
        inventory.include_array(&array).map_err(Error::from)?;
        let published = {
            let scope = owner.work().scope.borrow();
            pending.publish(inventory, scope.as_ref().expect("pending input copy scope"))?
        };
        drop(published);
        owner
            .work()
            .scope
            .borrow_mut()
            .take()
            .expect("single input copy scope")
            .certify()
            .map_err(memory)?;
        drop(execution);
        drop(discarded.0.take());
        Ok(RegisteredArrayCopy {
            array,
            completed_stream,
            custody: RegisteredArrayCopyCustody {
                _copy: custody.retention(),
                _host: host.clone(),
            },
        })
    }
}
