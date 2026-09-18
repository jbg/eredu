//! Exact selected Rust queue/source storage. Thread/PAL and read authority are separate.
use super::*;
use eredu_nn::workspace::HostMetadataFunding;
use std::{
    alloc::Layout,
    fmt,
    mem::{size_of, size_of_val},
    ops::Deref,
    sync::atomic::AtomicUsize,
};

pub(super) struct Funded<T> {
    pub(super) value: T,
    // Last: the value and shared allocation retire before this account alias.
    funding: Option<HostMetadataFunding>,
}
pub(super) struct Retained<T> {
    owner: Option<Arc<Funded<T>>>,
}
impl<T> Retained<T> {
    fn new(value: T, funding: Option<HostMetadataFunding>) -> Self {
        Self {
            owner: Some(Arc::new(Funded { value, funding })),
        }
    }
    pub(super) fn funding(&self) -> Option<&HostMetadataFunding> {
        self.owner.as_ref().expect("live storage").funding.as_ref()
    }
    #[cfg(test)]
    pub(super) fn raw_for_test(&self) -> &Arc<Funded<T>> {
        self.owner.as_ref().expect("live storage")
    }
}
impl<T> Clone for Retained<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
        }
    }
}
impl<T> Deref for Retained<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.owner.as_ref().expect("live storage").value
    }
}
impl<T> Drop for Retained<T> {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
fn shared_bytes<T>() -> Option<usize> {
    let payload = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<Funded<T>>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    [
        payload,
        size_of::<Retained<T>>(),
        size_of::<Arc<Funded<T>>>(),
        size_of::<Option<Funded<T>>>(),
    ]
    .into_iter()
    .try_fold(0, usize::checked_add)
}

/// An immutable unit identity from the actual retained worker domain.
/// Clones share its account and source strings; they create no new ID string.
#[derive(Clone)]
pub struct PrefetchUnit {
    source: Retained<Domain>,
    index: usize,
}
impl PrefetchUnit {
    pub(super) fn new(source: Retained<Domain>, index: usize) -> Self {
        Self { source, index }
    }
    /// Borrows the same selected source ID used by the worker operation.
    pub fn id(&self) -> &OffloadUnitId {
        let Domain::Selected(ids) = &*self.source else {
            unreachable!("selected source identity")
        };
        &ids[self.index]
    }
}
impl fmt::Debug for PrefetchUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PrefetchUnit").field(self.id()).finish()
    }
}

pub(super) struct Retirement<E> {
    pub(super) values: Vec<E>,
    _funding: Option<HostMetadataFunding>,
}
impl<E: BackgroundPrefetchFailure> Retirement<E> {
    pub(super) fn bytes(count: usize) -> Option<usize> {
        let parts = [
            size_of::<E>().checked_mul(count)?,
            size_of::<Self>(),
            size_of::<Result<Self, BackgroundPrefetchWorkerError<E>>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Option<usize>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn prepaid(
        count: usize,
        funding: Option<HostMetadataFunding>,
    ) -> Result<Self, std::collections::TryReserveError> {
        let mut values = Vec::new();
        values.try_reserve_exact(count)?;
        Ok(Self {
            values,
            _funding: funding,
        })
    }
    pub(super) fn prepare(
        domain: &Retained<Domain>,
    ) -> Result<Self, BackgroundPrefetchWorkerError<E>> {
        let count = match &**domain {
            Domain::Open => 0,
            Domain::Selected(ids) => ids.len(),
        };
        if let Some(funding) = domain.funding() {
            funding
                .reserve_metadata(Self::bytes(count).ok_or(
                    BackgroundPrefetchWorkerError::HostMetadata(
                        eredu_core::HostMetadataFundingError::Overflow,
                    ),
                )?)
                .map_err(BackgroundPrefetchWorkerError::HostMetadata)?;
        }
        Self::prepaid(count, domain.funding().cloned())
            .map_err(BackgroundPrefetchWorkerError::RetirementStorage)
    }
}

/// A selected domain, fixed FIFO/terminal state and shutdown-retirement destination
/// constructed before worker publication. Its host account follows all shared
/// storage aliases, including a detached worker and escaped failed-unit identity.
/// This is a Rust storage receipt only: no thread/PAL, native read, device budget,
/// execution permission or completion follows from creating it.
pub struct PreparedPrefetchStorage<E: BackgroundPrefetchFailure> {
    pub(super) retirement: Retirement<E>,
    pub(super) domain: Retained<Domain>,
    pub(super) shared: Retained<(Mutex<Shared<E>>, Condvar)>,
}
impl<E: BackgroundPrefetchFailure> fmt::Debug for PreparedPrefetchStorage<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedPrefetchStorage")
            .finish_non_exhaustive()
    }
}

/// A fixed preparation cause retaining every actual constructor prefix and its
/// host account. Refusal performs no queue admission or worker publication.
pub struct PrefetchStoragePreparationError<E> {
    cause: PreparationCause<E>,
    units: Vec<OffloadUnitId>,
    retirement: Vec<E>,
    funding: HostMetadataFunding,
}
enum PreparationCause<E> {
    Funding(eredu_core::HostMetadataFundingError),
    DuplicateUnit,
    Lifecycle(eredu_core::residency::PrefetchStorageError<E>),
    Retirement(std::collections::TryReserveError),
}
impl<E> fmt::Debug for PrefetchStoragePreparationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefetchStoragePreparationError")
            .field("retained_ids", &self.units.len())
            .field("retained_retirement_capacity", &self.retirement.capacity())
            .finish_non_exhaustive()
    }
}
impl<E> fmt::Display for PrefetchStoragePreparationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            PreparationCause::Funding(cause) => cause.fmt(f),
            PreparationCause::DuplicateUnit => {
                f.write_str("prefetch selected domain contains duplicate units")
            }
            PreparationCause::Lifecycle(cause) => cause.fmt(f),
            PreparationCause::Retirement(cause) => cause.fmt(f),
        }
    }
}
impl<E: fmt::Debug + 'static> std::error::Error for PrefetchStoragePreparationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            PreparationCause::Funding(cause) => Some(cause),
            PreparationCause::Lifecycle(cause) => Some(cause),
            PreparationCause::Retirement(cause) => Some(cause),
            PreparationCause::DuplicateUnit => None,
        }
    }
}
impl<E: BackgroundPrefetchFailure> PreparedPrefetchStorage<E> {
    /// Exact requested payload and named Rust constructor/retirement controls.
    /// IDs are borrowed from the source; this query neither copies nor sorts them.
    pub fn metadata_bytes(units: &[OffloadUnitId], queue_capacity: usize) -> Option<usize> {
        Self::metadata_bytes_for_units(units.iter(), queue_capacity)
    }
    /// Counts a finite borrowed source projection without constructing an
    /// intermediate ID vector. The same selected-storage constructor is used.
    pub fn metadata_bytes_for_units<'a>(
        mut units: impl ExactSizeIterator<Item = &'a OffloadUnitId>,
        queue_capacity: usize,
    ) -> Option<usize> {
        let count = units.len();
        let strings = units.try_fold(0usize, |n, id| n.checked_add(id.as_str().len()))?;
        let parts = [
            strings,
            size_of::<OffloadUnitId>().checked_mul(count)?,
            (size_of::<String>() + size_of::<OffloadUnitId>() + size_of::<&OffloadUnitId>())
                .checked_mul(count)?,
            PrefetchExecutionState::<E, usize>::selected_storage_bytes(
                count,
                queue_capacity,
            )?,
            Retirement::<E>::bytes(count)?,
            shared_bytes::<Domain>()?,
            shared_bytes::<(Mutex<Shared<E>>, Condvar)>()?,
            size_of::<Self>(),
            size_of::<PrefetchStoragePreparationError<E>>(),
            size_of::<Result<Self, PrefetchStoragePreparationError<E>>>(),
            size_of::<std::slice::Iter<'_, OffloadUnitId>>(),
            size_of::<Option<usize>>(),
            size_of::<HostMetadataFunding>(),
            size_of::<eredu_core::HostMetadataFundingError>(),
            size_of::<PrefetchUnit>(),
            size_of::<BackgroundPrefetchWorkerError<E>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Copies the actual selected source IDs and reserves all fixed Rust storage
    /// on the supplied cumulative host account before any destination is born.
    pub fn prepare(
        units: &[OffloadUnitId],
        queue_capacity: usize,
        funding: HostMetadataFunding,
    ) -> Result<Self, PrefetchStoragePreparationError<E>> {
        let mut partial = PrefetchStoragePreparationError {
            cause: PreparationCause::Funding(eredu_core::HostMetadataFundingError::Overflow),
            units: Vec::new(),
            retirement: Vec::new(),
            funding,
        };
        let Some(bytes) = Self::metadata_bytes(units, queue_capacity) else {
            return Err(partial);
        };
        if let Err(cause) = partial.funding.reserve_metadata(bytes) {
            partial.cause = PreparationCause::Funding(cause);
            return Err(partial);
        }
        partial.units = units.to_vec();
        partial.units.sort_unstable();
        if partial.units.windows(2).any(|pair| pair[0] == pair[1]) {
            partial.cause = PreparationCause::DuplicateUnit;
            return Err(partial);
        }
        if let Err(cause) = partial.retirement.try_reserve_exact(units.len()) {
            partial.cause = PreparationCause::Retirement(cause);
            return Err(partial);
        }
        let lifecycle = match PrefetchExecutionState::with_unit_count(units.len(), queue_capacity) {
            Ok(value) => value,
            Err(cause) => {
                partial.cause = PreparationCause::Lifecycle(cause);
                return Err(partial);
            }
        };
        let funding = partial.funding;
        Ok(Self {
            retirement: Retirement {
                values: partial.retirement,
                _funding: Some(funding.clone()),
            },
            domain: Retained::new(Domain::Selected(partial.units), Some(funding.clone())),
            shared: Self::shared(Lifecycle::Selected(lifecycle), Some(funding)),
        })
    }
    pub(super) fn ordinary(
        capacity: usize,
        units: Option<Vec<OffloadUnitId>>,
    ) -> Result<Self, BackgroundPrefetchWorkerError<E>> {
        let (domain, lifecycle) = if let Some(mut units) = units {
            units.sort_unstable();
            if units.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(BackgroundPrefetchWorkerError::DuplicateUnit);
            }
            let state = PrefetchExecutionState::with_unit_count(units.len(), capacity)?;
            (Domain::Selected(units), Lifecycle::Selected(state))
        } else {
            (
                Domain::Open,
                Lifecycle::Open(PrefetchExecutionState::new(capacity)?),
            )
        };
        let domain = Retained::new(domain, None);
        let retirement = Retirement::prepare(&domain)?;
        Ok(Self {
            retirement,
            domain,
            shared: Self::shared(lifecycle, None),
        })
    }
    fn shared(
        lifecycle: Lifecycle<E>,
        funding: Option<HostMetadataFunding>,
    ) -> Retained<(Mutex<Shared<E>>, Condvar)> {
        Retained::new(
            (
                Mutex::new(Shared {
                    lifecycle,
                    wake: Default::default(),
                    shutdown: false,
                    cancel_on_shutdown: false,
                    running: true,
                }),
                Condvar::new(),
            ),
            funding,
        )
    }
}

#[cfg(test)]
mod tests;
