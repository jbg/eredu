//! Finite carriers for existing original storage, separate from copy authority.
use super::*;
use crate::{HostMetadataKey, HostSlotMetadata, SharedPreparedInputCacheIdentity};
use eredu_core::HostPreparationAuthority;
use std::{alloc::Layout, mem::size_of};

#[derive(Debug)]
enum Source {
    Table(super::super::OriginalResidentResetSource),
    Input(SharedPreparedInputCacheIdentity),
}
impl Source {
    fn key(&self) -> &HostMetadataKey {
        match self {
            Self::Table(source) => source.metadata().identity().registry_key(),
            Self::Input(source) => source.identity().registry_key(),
        }
    }
    fn bytes(&self) -> Option<u64> {
        match self {
            Self::Table(source) => source.metadata().capacity_bytes(),
            Self::Input(source) => source.capacity_bytes(),
        }
    }
    fn validate_in(
        &self,
        pool: &WorkingMemoryPool,
        usage: &super::super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Table(source) => source.validate_in(pool, usage),
            Self::Input(source) => source
                .original_residence(pool)
                .ok_or(WorkingMemoryError::IdentityMismatch)?,
        }
    }
}

/// Exact requested carrier extent, with no source, account or execution grant.
#[derive(Debug, Clone, Copy)]
pub struct OriginalStorageSourcesLayout {
    maximum: usize,
    bytes: usize,
}
impl OriginalStorageSourcesLayout {
    /// One inline source per unique original table or prepared-input identity.
    /// The enclosing producer separately prices the registration/shared shell.
    pub fn new(maximum: usize) -> Option<Self> {
        let bytes = [
            Layout::array::<Source>(maximum).ok()?.size(),
            size_of::<Self>(),
            size_of::<RetainedOriginalStorageSources>(),
            size_of::<Payload>(),
            size_of::<Source>(),
            size_of::<Result<Source, WorkingMemoryError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<OriginalStorageSourcesError>(),
            size_of::<Result<RetainedOriginalStorageSources, OriginalStorageSourcesError>>(),
            size_of::<std::collections::TryReserveError>(),
            super::super::qualified_storage::reserve_control_bytes::<Source>().ok()?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        Some(Self { maximum, bytes })
    }
    /// Carrier requests only; already-paid source payloads are not charged again.
    pub fn requested_bytes(self) -> usize {
        self.bytes
    }

    /// Constructs only under the enclosing producer's accepted host preparation.
    /// The authority is lifetime custody, not evidence that these bytes were
    /// admitted. The caller must include this layout in its complete host plan.
    pub fn construct(
        self,
        pool: &WorkingMemoryPool,
        authority: &HostPreparationAuthority,
    ) -> Result<RetainedOriginalStorageSources, OriginalStorageSourcesError> {
        let mut result = RetainedOriginalStorageSources(Some(Payload {
            sources: Vec::new(),
            maximum: self.maximum,
            pool: pool.clone(),
            _authority: authority.clone(),
        }));
        if let Err(error) = super::super::qualified_storage::reserve_empty(
            &mut result.0.as_mut().expect("new carrier").sources,
            self.maximum,
        ) {
            use super::super::qualified_storage::ReserveError;
            let cause = match error {
                ReserveError::Unqualified => WorkingMemoryError::UnknownBound,
                ReserveError::Layout => WorkingMemoryError::Overflow,
                ReserveError::Reserve(cause) => WorkingMemoryError::ControlStorageReserve(cause),
                ReserveError::Contract => WorkingMemoryError::IdentityMismatch,
            };
            return Err(OriginalStorageSourcesError {
                cause,
                _partial: result,
            });
        }
        Ok(result)
    }
}

#[derive(Debug)]
struct Payload {
    // Deallocate source capacity before releasing its enclosing host custody.
    sources: Vec<Source>,
    maximum: usize,
    pool: WorkingMemoryPool,
    _authority: HostPreparationAuthority,
}

/// Original source tokens whose physical charges already exist. This creates
/// neither registration, copy permission, numerical storage nor a new scope.
/// Cloning/extracting a raw source population is deliberately unavailable.
#[derive(Debug)]
pub struct RetainedOriginalStorageSources(Option<Payload>);

/// Construction failure retaining the complete carrier prefix and host custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalStorageSourcesError {
    #[source]
    cause: WorkingMemoryError,
    _partial: RetainedOriginalStorageSources,
}

impl RetainedOriginalStorageSources {
    /// Retains a real completed table only after its original constructor and
    /// domain validate. Duplicate aliases consume no further carrier slot.
    pub fn retain_table(&mut self, metadata: &HostSlotMetadata) -> Result<(), WorkingMemoryError> {
        let source = self.payload()?.pool.pin_original_reset_slots(metadata)?;
        self.push(Source::Table(source))
    }
    /// Retains a genuine original prepared-input description and its source
    /// custody. Ordinary descriptions require their ordinary registered origin.
    pub fn retain_input(
        &mut self,
        input: &SharedPreparedInputCacheIdentity,
    ) -> Result<(), WorkingMemoryError> {
        input
            .original_residence(&self.payload()?.pool)
            .ok_or(WorkingMemoryError::IdentityMismatch)??;
        self.push(Source::Input(input.clone()))
    }
    fn payload(&self) -> Result<&Payload, WorkingMemoryError> {
        self.0.as_ref().ok_or(WorkingMemoryError::IdentityMismatch)
    }
    fn push(&mut self, source: Source) -> Result<(), WorkingMemoryError> {
        let payload = self
            .0
            .as_mut()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let bytes = source.bytes().ok_or(WorkingMemoryError::UnknownBound)?;
        if let Some(prior) = payload
            .sources
            .iter()
            .find(|prior| prior.key() == source.key())
        {
            return same_capacity(
                prior.bytes().ok_or(WorkingMemoryError::UnknownBound)?,
                bytes,
            );
        }
        if payload.sources.len() == payload.maximum {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        payload.sources.push(source);
        Ok(())
    }
    /// Whether this exact original identity is retained; not a capacity credit.
    pub fn contains(&self, key: &HostMetadataKey) -> bool {
        self.0
            .as_ref()
            .is_some_and(|payload| payload.sources.iter().any(|source| source.key() == key))
    }
    pub(super) fn validate_in(
        &self,
        pool: &WorkingMemoryPool,
        usage: &super::super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        let payload = self.payload()?;
        if !payload.pool.same_domain(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        for source in &payload.sources {
            source.validate_in(pool, usage)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) enum OriginalSources {
    Unquoted(super::super::UnquotedOriginalSlotSources),
    Prepared(RetainedOriginalStorageSources),
}
impl OriginalSources {
    pub(super) fn preparation(&self) -> Option<&HostPreparationAuthority> {
        match self {
            Self::Prepared(sources) => sources.0.as_ref().map(|payload| &payload._authority),
            Self::Unquoted(_) => None,
        }
    }
    pub(super) fn validate_in(
        &self,
        pool: &WorkingMemoryPool,
        usage: &super::super::Usage,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Prepared(sources) => sources.validate_in(pool, usage),
            Self::Unquoted(sources) => {
                for source in sources.sources() {
                    source.validate_in(pool, usage)?;
                }
                Ok(())
            }
        }
    }
}

impl<K: Ord + Send + 'static> WorkingMemoryStorage<K> {
    pub(in crate::working_memory) fn source_preparation(
        &self,
    ) -> Option<&HostPreparationAuthority> {
        self.0
            .original_sources
            .as_ref()
            .and_then(OriginalSources::preparation)
    }
}

impl<K: super::super::HostSlotStorageKey> WorkingMemoryStorage<K> {
    /// Associates a finite original source carrier with this unique existing-
    /// only registration. Rejection preserves the carrier and every owner in it.
    /// Source capacities affect diagnostics only; their existing charges remain.
    pub fn with_retained_original_sources(
        mut self,
        sources: &mut RetainedOriginalStorageSources,
    ) -> Result<Self, WorkingMemoryError> {
        let pool = self
            .0
            .pool
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .clone();
        if self.0.original_sources.is_some() || Arc::strong_count(&self.0) != 1 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut bytes = self.0.bytes;
        for source in &sources.payload()?.sources {
            if self
                .0
                .keys
                .iter()
                .any(|key| key.host_slot_identity() == Some(source.key()))
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            bytes = bytes
                .checked_add(source.bytes().ok_or(WorkingMemoryError::UnknownBound)?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        {
            let usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            sources.validate_in(&pool, &usage)?;
        }
        let registration = Arc::get_mut(&mut self.0).expect("unique source registration");
        registration.bytes = bytes;
        registration.original_sources = Some(OriginalSources::Prepared(
            RetainedOriginalStorageSources(sources.0.take()),
        ));
        Ok(self)
    }
}
