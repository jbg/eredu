//! The independently retained host allocations of a public parameter result.
use super::*;
use eredu_runtime::working_memory::{
    MemoryLedger, PendingStorageAllocation, StorageAllocation, StoragePublicationLayout,
    WorkingMemoryError,
};
use std::{cmp::Ordering, mem::size_of, sync::Arc};

#[derive(Clone, Debug)]
struct ResultAllocation {
    owner: Arc<u8>,
    allocation: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_parameter_payload_stays_charged_until_its_last_alias() {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let before = pool.snapshot().unwrap();
        let region = ParameterRegion {
            starts: vec![0, 0],
            shape: vec![2, 3],
        };
        let prepared = PreparedResult::query(&pool, "source", "weight", &region).unwrap();
        let granted = pool.snapshot().unwrap();
        let values = prepared
            .finish_query(ParameterValues {
                identity: "source".into(),
                parameter: "weight".into(),
                dtype: InterventionDtype::Float32,
                region,
                values: vec![1., -2., 3., 4., 5., -6.],
                usage: CaptureUsage::default(),
            })
            .unwrap();
        let published = pool.snapshot().unwrap();
        for (grant, published) in granted.domains.iter().zip(&published.domains) {
            assert_eq!(grant.current_charge_bytes, published.current_charge_bytes);
            assert_eq!(grant.historical_peak_bytes, published.historical_peak_bytes);
        }
        let alias = values.clone();
        assert!(alias.same_storage(&values));
        drop(values);
        assert_eq!(alias.values, [1., -2., 3., 4., 5., -6.]);
        assert_eq!(pool.snapshot().unwrap().domains, published.domains);
        drop(alias);
        for (before, after) in before.domains.iter().zip(pool.snapshot().unwrap().domains) {
            assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
        }
    }
}
impl PartialEq for ResultAllocation {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner) && self.allocation == other.allocation
    }
}
impl Eq for ResultAllocation {}
impl PartialOrd for ResultAllocation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ResultAllocation {
    fn cmp(&self, other: &Self) -> Ordering {
        Arc::as_ptr(&self.owner)
            .cmp(&Arc::as_ptr(&other.owner))
            .then_with(|| self.allocation.cmp(&other.allocation))
    }
}
type Custody = PendingStorageAllocation<ResultAllocation>;

pub(super) struct PreparedResult {
    capacities: [usize; 5],
    custody: Custody,
    host: eredu_core::HostPreparationAuthority,
}
fn memory(cause: WorkingMemoryError) -> ParameterError {
    failure(Error::text_admission(cause))
}
fn bytes(capacity: usize, width: usize) -> Result<u64, ParameterError> {
    capacity
        .checked_mul(width)
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(ParameterError::Overflow)
}
impl PreparedResult {
    pub(super) fn host_authority(&self) -> &eredu_core::HostPreparationAuthority {
        &self.host
    }
    pub(super) fn query(
        pool: &MemoryLedger,
        identity: &str,
        parameter: &str,
        region: &ParameterRegion,
    ) -> Result<Self, ParameterError> {
        let count =
            usize::try_from(elements(&region.shape)?).map_err(|_| ParameterError::Overflow)?;
        Self::prepare(
            pool,
            [
                identity.len(),
                parameter.len(),
                region.starts.capacity(),
                region.shape.capacity(),
                count,
            ],
            SharedParameterValues::retained_control_bytes::<Custody>(),
        )
    }
    pub(super) fn projection(
        pool: &MemoryLedger,
        identity: &str,
        parameter: &str,
        shape: &[u64],
        shape_capacity: usize,
    ) -> Result<Self, ParameterError> {
        if shape_capacity < shape.len() {
            return Err(ParameterError::Invalid(
                "projection shape capacity is smaller than its length".into(),
            ));
        }
        let count = usize::try_from(elements(shape)?).map_err(|_| ParameterError::Overflow)?;
        Self::prepare(
            pool,
            [identity.len(), parameter.len(), 0, shape_capacity, count],
            SharedParameterProjectionValues::retained_control_bytes::<Custody>(),
        )
    }
    fn prepare(
        pool: &MemoryLedger,
        capacities: [usize; 5],
        shared_controls: Option<u64>,
    ) -> Result<Self, ParameterError> {
        let key = usize::try_from(
            eredu_runtime::working_memory::OriginalHostMetadataCustody::shared_storage_bytes(
                std::alloc::Layout::new::<u8>(),
            )
            .map_err(memory)?,
        )
        .map_err(|_| ParameterError::Overflow)?;
        let controls = key
            .checked_add(super::completed::control_bytes().ok_or(ParameterError::Overflow)?)
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .and_then(|n| n.checked_add(size_of::<[usize; 5]>()))
            .and_then(|n| n.checked_add(size_of::<[(ResultAllocation, StorageAllocation); 5]>()))
            .and_then(|n| n.checked_add(size_of::<Result<Self, ParameterError>>()))
            .and_then(|n| u64::try_from(n).ok())
            .and_then(|n| n.checked_add(shared_controls?))
            .ok_or(ParameterError::Overflow)?;
        let payload = [
            bytes(capacities[0], 1)?,
            bytes(capacities[1], 1)?,
            bytes(capacities[2], size_of::<u64>())?,
            bytes(capacities[3], size_of::<u64>())?,
            bytes(capacities[4], size_of::<f32>())?,
        ];
        let prepared = StoragePublicationLayout::<ResultAllocation>::pending(payload.len())
            .and_then(|layout| layout.with_additional_host_metadata(controls))
            .and_then(|layout| layout.fund(pool))
            .map_err(memory)?;
        let owner = Arc::new(0);
        let placement = pool.host_placement_handle();
        let host = prepared.host_authority().clone();
        let custody = prepared
            .reserve_storage(payload.into_iter().enumerate().map(|(index, bytes)| {
                (
                    ResultAllocation {
                        owner: Arc::clone(&owner),
                        allocation: index as u8,
                    },
                    StorageAllocation::new(bytes, Arc::clone(&placement)),
                )
            }))
            .map_err(memory)?;
        Ok(Self {
            capacities,
            custody,
            host,
        })
    }
    pub(super) fn finish_query(
        mut self,
        values: ParameterValues,
    ) -> Result<SharedParameterValues, ParameterError> {
        let actual = [
            values.identity.capacity(),
            values.parameter.capacity(),
            values.region.starts.capacity(),
            values.region.shape.capacity(),
            values.values.capacity(),
        ];
        if actual != self.capacities {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        self.custody.publish().map_err(memory)?;
        Ok(SharedParameterValues::retain(values, self.custody))
    }
    pub(super) fn finish_projection(
        mut self,
        values: ParameterProjectionValues,
    ) -> Result<SharedParameterProjectionValues, ParameterError> {
        let actual = [
            values.identity.capacity(),
            values.parameter.capacity(),
            0,
            values.shape.capacity(),
            values.values.capacity(),
        ];
        if actual != self.capacities {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        self.custody.publish().map_err(memory)?;
        Ok(SharedParameterProjectionValues::retain(
            values,
            self.custody,
        ))
    }
}
