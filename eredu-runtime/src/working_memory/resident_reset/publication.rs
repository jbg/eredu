//! Source-bound host publication; native readiness remains a separate contract.
use super::*;

/// Projection of the complete resident state, never a selected subcomponent.
/// Both methods must return the same whole state and perform no allocation,
/// callback, native work or mutation. Unsupported representations return None.
pub trait ResidentResetProjection<S: ResidentKvResetState> {
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
    control: Arc<()>,
    revision: Option<super::super::InferenceStateRevision>,
}
impl PublicationBinding {
    fn new<S: ResidentKvResetState>(source: &ResidentResetSource<'_, S>) -> Self {
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
            control: Arc::clone(source.control),
            revision: source.revision.cloned(),
        }
    }
    pub(crate) fn matches<S: ResidentKvResetState>(
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
            && Arc::ptr_eq(&self.control, source.control)
            && self.revision.as_ref() == source.revision
    }
    pub(crate) fn matches_state<S: ResidentKvResetState>(&self, state: &S) -> bool {
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
pub struct ResidentResetInstallation<S: ResidentKvResetState> {
    pub(crate) state: S,
    pub(crate) binding: PublicationBinding,
    pub(crate) custody: ResetCustody,
}
impl<S: ResidentKvResetState> ResidentResetInstallation<S> {
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
pub struct ResidentResetDisplaced<S: ResidentKvResetState> {
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

impl<'a, S: ResidentKvResetState, K: HostSlotStorageKey> PreparedResidentKvReset<'a, S, K> {
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

    /// One original comparison, then preparation of the native host-retirement
    /// slot and the fixed empty destination. Failed construction drops the empty
    /// prepared owner outside Usage while its own same-account custody survives.
    pub fn construct_for_publication<T, P>(
        mut self,
        session: &T,
        claim: SessionResetClaim<'_>,
        pool: &WorkingMemoryPool,
    ) -> Result<(ResidentResetInstallation<S>, P), ResidentResetError<S>>
    where
        T: ResidentResetSession<S>,
        P: ResidentResetPublicationProfile,
    {
        self.bytes = self
            .publication_required_bytes::<P>()
            .ok_or_else(|| ResidentResetError::rejected(WorkingMemoryError::Overflow))?;
        let (state, prepared) =
            self.construct_prepared(session, claim, pool, |source, custody| {
                PreparedPublication {
                    binding: PublicationBinding::new(source),
                    owner: P::prepare(ResidentResetPublicationCustody(custody.clone())),
                    custody: custody.clone(),
                }
            })?;
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
