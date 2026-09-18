//! Finite source-pin grouping and its owning shared-shell query.
use super::*;
use crate::working_memory::qualified_storage;
use std::mem::size_of;

impl RegisteredStoragePin {
    fn planning_control_bytes() -> Result<usize, WorkingMemoryError> {
        let owner = usize::try_from(qualified_storage::shared_bytes::<PlanningPinGroup>()?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let frames = [
            owner,
            size_of::<PlanningStoragePins>(),
            size_of::<PlanningPinGroup>(),
            size_of::<Option<PlanningPinGroup>>(),
            size_of::<RegisteredStoragePin>(),
            size_of::<Arc<PlanningPinGroup>>(),
            size_of::<Option<eredu_nn::workspace::HostMetadataFunding>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn planned(self, funding: Option<eredu_nn::workspace::HostMetadataFunding>) -> Self {
        match funding {
            Some(funding) => Self::Planned(PlanningStoragePins(Some(Arc::new(PlanningPinGroup {
                pin: self,
                _funding: funding,
            })))),
            None => self,
        }
    }
    pub(in crate::working_memory) fn empty_metadata(
        metadata: crate::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, eredu_nn::Error> {
        if !metadata.is_checked() {
            return Ok(Self::aggregate([]));
        }
        let funding = metadata.funding();
        let allocation = qualified_storage::shared_bytes::<Vec<RegisteredStoragePin>>()
            .map_err(|cause| metadata.source(cause))?;
        let frames = [
            usize::try_from(allocation)
                .map_err(|_| metadata.source(WorkingMemoryError::Overflow))?,
            size_of::<Vec<RegisteredStoragePin>>(),
            size_of::<RegisteredStoragePin>(),
            size_of::<Option<eredu_nn::workspace::HostMetadataFunding>>(),
        ];
        let mut bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| metadata.source(WorkingMemoryError::Overflow))?;
        if funding.is_some() {
            bytes = bytes
                .checked_add(
                    Self::planning_control_bytes().map_err(|cause| metadata.source(cause))?,
                )
                .ok_or_else(|| metadata.source(WorkingMemoryError::Overflow))?;
        }
        metadata
            .charge(bytes)
            .map_err(|cause| metadata.error(cause))?;
        Ok(Self::aggregate([]).planned(funding))
    }
    pub(in crate::working_memory) fn new_metadata<K: Ord + Send + Sync + 'static>(
        storage: WorkingMemoryStorage<K>,
        metadata: crate::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, eredu_nn::Error> {
        if !metadata.is_checked() {
            return Ok(Self::new(storage));
        }
        let funding = metadata.funding();
        let mut bytes = Self::single_control_bytes::<K>(storage.has_source_preparation())
            .map_err(|cause| metadata.source(cause))?;
        if funding.is_some() {
            bytes = bytes
                .checked_add(
                    Self::planning_control_bytes().map_err(|cause| metadata.source(cause))?,
                )
                .ok_or_else(|| metadata.source(WorkingMemoryError::Overflow))?;
        }
        metadata
            .charge(bytes)
            .map_err(|cause| metadata.error(cause))?;
        Ok(Self::new(storage).planned(funding))
    }
    pub(in crate::working_memory) fn pair_metadata(
        first: Self,
        second: Self,
        metadata: crate::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<Self, eredu_nn::Error> {
        if !metadata.is_checked() {
            return Ok(Self::aggregate([first, second]));
        }
        let funding = metadata.funding();
        let prepared = first.preparation().is_some() || second.preparation().is_some();
        let mut bytes =
            Self::pair_control_bytes(prepared).map_err(|cause| metadata.source(cause))?;
        if funding.is_some() {
            bytes = bytes
                .checked_add(
                    Self::planning_control_bytes().map_err(|cause| metadata.source(cause))?,
                )
                .ok_or_else(|| metadata.source(WorkingMemoryError::Overflow))?;
        }
        metadata
            .charge(bytes)
            .map_err(|cause| metadata.error(cause))?;
        Ok(Self::pair(first, second).planned(funding))
    }

    pub(in crate::working_memory) fn pair_control_bytes(
        prepared: bool,
    ) -> Result<usize, WorkingMemoryError> {
        let allocation = if prepared {
            qualified_storage::shared_bytes::<PreparedPinGroup>()
        } else {
            qualified_storage::shared_bytes::<[RegisteredStoragePin; 2]>()
        }?;
        let frames = [
            usize::try_from(allocation).map_err(|_| WorkingMemoryError::Overflow)?,
            size_of::<[RegisteredStoragePin; 2]>(),
            size_of::<PreparedPinGroup>(),
            size_of::<PreparedPinContents>(),
            size_of::<RegisteredStoragePin>(),
            size_of::<Option<PreparedPinGroup>>(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }

    pub(in crate::working_memory) fn single_control_bytes<K: Ord + Send + Sync + 'static>(
        prepared: bool,
    ) -> Result<usize, WorkingMemoryError> {
        let scalar = usize::try_from(qualified_storage::shared_bytes::<WorkingMemoryStorage<K>>()?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let group = if prepared {
            usize::try_from(qualified_storage::shared_bytes::<PreparedPinGroup>()?)
                .map_err(|_| WorkingMemoryError::Overflow)?
        } else {
            0
        };
        let controls = [
            scalar,
            group,
            size_of::<WorkingMemoryStorage<K>>(),
            size_of::<PreparedPinGroup>(),
            size_of::<PreparedPinContents>(),
            size_of::<RegisteredStoragePin>(),
            size_of::<Option<PreparedPinGroup>>(),
            size_of::<Arc<dyn Send + Sync>>(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
        ];
        controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }

    pub(in crate::working_memory) fn aggregate_counted(
        pins: impl IntoIterator<Item = Self>,
        count: usize,
    ) -> Result<Self, WorkingMemoryError> {
        let mut values = qualified_storage::vector(count, true)?;
        for pin in pins {
            if values.len() == count {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            values.push(pin);
        }
        if values.len() != count {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let authority = values
            .iter()
            .find_map(Self::preparation)
            .cloned()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        Ok(Self::prepared(
            PreparedPinContents::Group(values),
            authority,
        ))
    }
}
impl<K: Ord + Send + Sync + 'static> WorkingMemoryStorage<K> {
    /// Whether this existing pin carries the accepted constructor's host lifetime.
    /// This authenticates no amount and grants no additional storage permission.
    pub(crate) fn has_source_preparation(&self) -> bool {
        self.source_preparation().is_some()
    }

    /// One fixed shared pair joining an already-built complete-source pin and
    /// the fresh run's retained residual pin. Both children already exist; no
    /// single-pin allocation or variable-capacity collection is included here.
    /// The source constructor must include this before constructing the pair.
    pub fn copy_source_pair_bytes() -> Result<usize, WorkingMemoryError> {
        let owner = usize::try_from(qualified_storage::shared_bytes::<PreparedPinGroup>()?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        [
            size_of::<[RegisteredStoragePin; 2]>(),
            size_of::<PreparedPinGroup>(),
            size_of::<PreparedStoragePins>(),
            size_of::<PreparedPinContents>(),
            size_of::<RegisteredStoragePin>(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
            size_of::<Arc<PreparedPinGroup>>(),
        ]
        .into_iter()
        .try_fold(owner, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
    }

    /// Requested source-wrapper storage for the actual single-pin population and
    /// one finite aggregate. The source producer supplies its real Registered /
    /// Funded table topology; array counts cannot replace that population.
    /// Quarantine retains these same owners and creates no second source bundle.
    pub fn copy_source_wrapper_bytes(
        ordinary: usize,
        prepared: usize,
    ) -> Result<usize, WorkingMemoryError> {
        let count = ordinary
            .checked_add(prepared)
            .ok_or(WorkingMemoryError::Overflow)?;
        if prepared == 0 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let scalar = usize::try_from(qualified_storage::shared_bytes::<Self>()?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let group = usize::try_from(qualified_storage::shared_bytes::<PreparedPinGroup>()?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        let rows = usize::try_from(qualified_storage::array_bytes::<RegisteredStoragePin>(
            count,
        )?)
        .map_err(|_| WorkingMemoryError::Overflow)?;
        let reserve = usize::try_from(qualified_storage::vector_control_bytes::<
            RegisteredStoragePin,
        >()?)
        .map_err(|_| WorkingMemoryError::Overflow)?;
        let singles = scalar
            .checked_mul(count)
            .ok_or(WorkingMemoryError::Overflow)?;
        let groups = group
            .checked_mul(
                prepared
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?,
            )
            .ok_or(WorkingMemoryError::Overflow)?;
        [
            singles,
            groups,
            rows,
            usize::try_from(qualified_storage::array_bytes::<
                Option<RegisteredStoragePin>,
            >(count)?)
            .map_err(|_| WorkingMemoryError::Overflow)?,
            reserve,
            size_of::<std::vec::IntoIter<RegisteredStoragePin>>(),
            size_of::<(usize, usize, bool)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<PreparedPinGroup>(),
            size_of::<PreparedStoragePins>(),
            size_of::<PreparedPinContents>(),
            size_of::<RegisteredStoragePin>(),
            size_of::<Option<PreparedPinGroup>>(),
            size_of::<Arc<PreparedPinGroup>>(),
            size_of::<Arc<dyn Send + Sync>>(),
            size_of::<Self>(),
            size_of::<Result<RegisteredStoragePin, WorkingMemoryError>>(),
            size_of::<Option<eredu_core::HostPreparationAuthority>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
    }
}
