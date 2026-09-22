//! Source-bound host publication; native readiness remains a separate contract.
use super::*;

/// Projection of the complete resident state, never a selected subcomponent.
/// Both methods must return the same whole state and perform no allocation,
/// callback, native work or mutation. Unsupported representations return None.
pub trait ResidentResetProjection<S: ResidentTableResetState> {
    /// Borrows the exact whole state used by immutable source inspection.
    fn resident_reset_ref(&self) -> Option<&S>;
    /// Borrows that same state for an already validated infallible exchange.
    fn resident_reset_mut(&mut self) -> Option<&mut S>;
}

/// Same original reset custody, issued only with its publication profile.
/// No raw owner, clone, amount constructor, registry grant or refill is exposed.
pub struct ResidentResetPublicationCustody(ResetCustody);

/// Actual prepared host retirement owner for this provider's publication.
/// Layout reporting is cold and exact. Preparation runs only after the original
/// comparison; it must retain custody through its complete allocation and Drop,
/// and must never submit native work. Its value is not put in a portable error.
pub trait ResidentResetPublicationProfile: Sized {
    /// Concrete allocation plus named construction/retirement controls. Dynamic
    /// populations, backing and queue infrastructure are not inferred here.
    fn control_bytes() -> Option<u64>;
    /// Allocates one empty retirement owner with the supplied original custody.
    fn prepare(custody: ResidentResetPublicationCustody) -> Self;
}

pub(crate) struct PublicationBinding {
    state: usize,
    selected: usize,
    table: crate::HostMetadataKey,
    layout: crate::HostMetadataKey,
    global_start: usize,
    execution: InferenceExecutionIdentity,
    control: crate::replicated_session::ParameterControlIdentity,
    revision: Option<super::super::InferenceStateRevision>,
}
impl PublicationBinding {
    fn new<S: ResidentTableResetState>(source: &ResidentResetSource<'_, S>) -> Self {
        Self {
            state: std::ptr::from_ref(source.state) as usize,
            selected: std::ptr::from_ref(source.selected) as usize,
            table: source
                .state
                .resident_reset_layers()
                .metadata()
                .identity()
                .registry_key()
                .clone(),
            layout: source
                .state
                .resident_reset_layout()
                .identity()
                .registry_key()
                .clone(),
            global_start: source.state.resident_reset_global_start(),
            execution: source.execution.clone(),
            control: source.control.clone(),
            revision: source.revision.cloned(),
        }
    }
    pub(crate) fn matches<S: ResidentTableResetState>(
        &self,
        source: &ResidentResetSource<'_, S>,
    ) -> bool {
        self.state == std::ptr::from_ref(source.state) as usize
            && self.selected == std::ptr::from_ref(source.selected) as usize
            && &self.table
                == source
                    .state
                    .resident_reset_layers()
                    .metadata()
                    .identity()
                    .registry_key()
            && &self.layout
                == source
                    .state
                    .resident_reset_layout()
                    .identity()
                    .registry_key()
            && self.global_start == source.state.resident_reset_global_start()
            && Arc::ptr_eq(&self.execution.0, &source.execution.0)
            && self.control.matches(source.control)
            && self.revision.as_ref() == source.revision
    }
    pub(crate) fn matches_state<S: ResidentTableResetState>(&self, state: &S) -> bool {
        self.state == std::ptr::from_ref(state) as usize
            && &self.table
                == state
                    .resident_reset_layers()
                    .metadata()
                    .identity()
                    .registry_key()
    }
}

/// A freshly constructed, originally funded state bound to its exact old source.
/// Its only owning exits are checked shared-session installation or an owning
/// construction error. Native readiness is not established by this value.
pub struct ResidentResetInstallation<S: ResidentTableResetState> {
    pub(crate) state: S,
    pub(crate) binding: PublicationBinding,
    pub(crate) custody: ResetCustody,
}
impl<S: ResidentTableResetState> ResidentResetInstallation<S> {
    /// Rejects before publication, keeping the complete destination and same
    /// original allowance in the existing concrete portable error source.
    pub fn into_error(self, cause: WorkingMemoryError) -> ResidentResetError<S> {
        let Self {
            state,
            binding,
            custody,
        } = self;
        drop(binding);
        ResidentResetError {
            cause: ResetCause::Memory(cause),
            partial: Vec::new(),
            state: Some(state),
            entry: None,
            source: None,
            source_children: Vec::new(),
            custody: Some(custody),
        }
    }
}

/// Whole displaced state and prompt source; native callers must put this into
/// their already prepared retirement owner before ending the publication entry.
pub struct ResidentResetDisplaced<S: ResidentTableResetState> {
    pub(crate) state: S,
    pub(crate) prompt: Option<crate::SharedPreparedInputCacheIdentity>,
    pub(crate) binding: PublicationBinding,
    pub(crate) custody: ResetCustody,
}

struct PreparedPublication<P> {
    binding: PublicationBinding,
    owner: P,
    custody: ResetCustody,
}

impl<'a, S: ResidentTableResetState, K: HostSlotStorageKey> PreparedResidentKvReset<'a, S, K> {
    /// Exact constructor plus this provider's single prepared publication owner.
    /// No account, owner or destination allocation is created by reporting it.
    pub fn publication_required_bytes<P: ResidentResetPublicationProfile>(&self) -> Option<u64> {
        let host = [
            size_of::<PublicationBinding>(),
            size_of::<PreparedPublication<P>>(),
            size_of::<ResidentResetPublicationCustody>(),
            size_of::<ResidentResetInstallation<S>>(),
            size_of::<ResidentResetDisplaced<S>>(),
            size_of::<(S, PreparedPublication<P>)>(),
            size_of::<(ResidentResetInstallation<S>, P)>(),
            size_of::<Result<(S, PreparedPublication<P>), ResidentResetError<S>>>(),
            size_of::<Result<(ResidentResetInstallation<S>, P), ResidentResetError<S>>>(),
            size_of::<
                Result<
                    ResidentResetDisplaced<S>,
                    (WorkingMemoryError, ResidentResetInstallation<S>),
                >,
            >(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        self.bytes
            .checked_add(P::control_bytes()?)?
            .checked_add(u64::try_from(host).ok()?)
    }

    /// Complete fixed host demand for the reversible parameter publication slot.
    pub fn parameter_publication_required_bytes<P: ResidentResetPublicationProfile>(
        &self,
    ) -> Option<u64> {
        self.publication_required_bytes::<P>()?
            .checked_add(parameter_exchange_bytes::<S>()?)
    }

    /// Host-only empty-state construction for an enclosing parameter publication.
    /// The backend validates its actual idle session source and selected executable;
    /// this creates no token, native allocation, or submission authority. Existing
    /// table/layout pins and the reset account use the same worker as explicit reset.
    pub fn construct_for_parameter_publication<T, P>(
        mut self,
        session: &T,
        execution: &InferenceExecutionIdentity,
        limits: &eredu_core::MemoryLimits,
        pool: &MemoryLedger,
    ) -> Result<(ResidentResetInstallation<S>, P), ResidentResetError<S>>
    where
        T: ResidentResetSession<S>,
        P: ResidentResetPublicationProfile,
    {
        self.bytes = self
            .parameter_publication_required_bytes::<P>()
            .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        let (state, prepared) = self.construct_prepared(
            session,
            ResetRequest::Parameter { execution, limits },
            pool,
            |source, custody| PreparedPublication {
                binding: PublicationBinding::new(source),
                owner: P::prepare(ResidentResetPublicationCustody(custody.clone())),
                custody: custody.clone(),
            },
        )?;
        Ok((
            ResidentResetInstallation {
                state,
                binding: prepared.binding,
                custody: prepared.custody,
            },
            prepared.owner,
        ))
    }

    /// One original comparison, then preparation of the native host-retirement
    /// slot and the fixed empty destination. Failed construction drops the empty
    /// prepared owner outside Usage while its own same-account custody survives.
    pub fn construct_for_publication<T, P>(
        mut self,
        session: &T,
        claim: SessionResetClaim<'_>,
        pool: &MemoryLedger,
    ) -> Result<(ResidentResetInstallation<S>, P), ResidentResetError<S>>
    where
        T: ResidentResetSession<S>,
        P: ResidentResetPublicationProfile,
    {
        self.bytes = self
            .publication_required_bytes::<P>()
            .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        let (state, prepared) = self.construct_prepared(
            session,
            ResetRequest::Session(claim),
            pool,
            |source, custody| PreparedPublication {
                binding: PublicationBinding::new(source),
                owner: P::prepare(ResidentResetPublicationCustody(custody.clone())),
                custody: custody.clone(),
            },
        )?;
        Ok((
            ResidentResetInstallation {
                state,
                binding: prepared.binding,
                custody: prepared.custody,
            },
            prepared.owner,
        ))
    }
}

/// Move-only reversible empty state for a prepared parameter transaction. Both
/// table identities and revisions are retained before publication; the first
/// exchange preserves the complete displaced state and prompt owner for rollback.
/// This owner grants no native execution or completion permission.
pub struct PreparedParameterStateReset<S: ResidentTableResetState> {
    pub(crate) state: S,
    pub(crate) prompt: Option<crate::SharedPreparedInputCacheIdentity>,
    binding: PublicationBinding,
    destination_table: crate::HostMetadataKey,
    destination_revision: Option<super::super::InferenceStateRevision>,
    pub(crate) exchanged: bool,
    _custody: ResetCustody,
}
impl<S: ResidentTableResetState> PreparedParameterStateReset<S> {
    /// Exact retained prompt owner displaced by the reversible exchange.
    pub fn shared_prompt_input_identity(&self) -> Option<&crate::SharedPreparedInputCacheIdentity> {
        self.prompt.as_ref()
    }
    pub(crate) fn from_installation(value: ResidentResetInstallation<S>) -> Self {
        let destination_table = value
            .state
            .resident_reset_layers()
            .metadata()
            .identity()
            .registry_key()
            .clone();
        let destination_revision = value
            .state
            .inference_retention()
            .initialized_revision()
            .cloned();
        Self {
            state: value.state,
            prompt: None,
            binding: value.binding,
            destination_table,
            destination_revision,
            exchanged: false,
            _custody: value.custody,
        }
    }
    pub(crate) fn matches(&self, source: &ResidentResetSource<'_, S>) -> bool {
        if !self.exchanged {
            return self.binding.matches(source);
        }
        // The enclosing publication separately authenticates its current checked
        // generation. A rollback restores the exact prepared table, never an
        // arbitrary saved state sharing an executable or parameter label.
        self.binding.state == std::ptr::from_ref(source.state).addr()
            && self.binding.selected == std::ptr::from_ref(source.selected).addr()
            && self.binding.layout
                == *source
                    .state
                    .resident_reset_layout()
                    .identity()
                    .registry_key()
            && self.binding.global_start == source.state.resident_reset_global_start()
            && self.binding.execution.same_execution(source.execution)
            && self.binding.control.same_owner(source.control)
            && self.destination_table
                == *source
                    .state
                    .resident_reset_layers()
                    .metadata()
                    .identity()
                    .registry_key()
            && self.destination_revision.as_ref() == source.revision
    }
}
fn parameter_exchange_bytes<S: ResidentTableResetState>() -> Option<u64> {
    let frames = [
        size_of::<PreparedParameterStateReset<S>>(),
        size_of::<
            Result<
                PreparedParameterStateReset<S>,
                (WorkingMemoryError, ResidentResetInstallation<S>),
            >,
        >(),
        size_of::<Option<crate::SharedPreparedInputCacheIdentity>>(),
        size_of::<crate::HostMetadataKey>(),
        size_of::<Option<super::super::InferenceStateRevision>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
        size_of::<(
            &ResidentResetSource<'_, S>,
            &mut PreparedParameterStateReset<S>,
        )>(),
    ];
    u64::try_from(
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)?,
    )
    .ok()
}
