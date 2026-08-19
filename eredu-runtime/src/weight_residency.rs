//! Backend-neutral immutable-weight residency policy.

use std::collections::VecDeque;

use eredu_core::{
    residency::{CacheEvictionPolicy, OffloadConfig, OffloadError},
    DEFAULT_MAX_MAPPED_SHARDS,
};

/// Current plus next unit retained by dense streamed execution.
pub const DENSE_TRANSFER_WINDOW: usize = 2;

/// Ordered bounded transfer cursor used by dense streamed execution.
#[derive(Debug)]
pub struct DenseTransferSchedule<T> {
    capacity: usize,
    pending: VecDeque<usize>,
    ready: VecDeque<(usize, T)>,
}

impl<T> DenseTransferSchedule<T> {
    /// Creates a cursor over the remaining unit indices.
    pub fn new(
        pending: impl IntoIterator<Item = usize>,
        capacity: usize,
    ) -> Result<Self, DenseTransferScheduleError> {
        if capacity == 0 {
            return Err(DenseTransferScheduleError::ZeroCapacity);
        }
        let pending = pending.into_iter().collect::<VecDeque<_>>();
        let mut previous = None;
        for &index in &pending {
            if previous.is_some_and(|previous| previous >= index) {
                return Err(DenseTransferScheduleError::UnorderedPending {
                    previous: previous.expect("an invalid pair has a previous index"),
                    actual: index,
                });
            }
            previous = Some(index);
        }
        Ok(Self {
            capacity,
            pending,
            ready: VecDeque::new(),
        })
    }

    /// Returns whether at least one submitted transfer is ready for consumption.
    pub fn has_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    /// Returns whether no ready or pending units remain.
    pub fn is_exhausted(&self) -> bool {
        self.ready.is_empty() && self.pending.is_empty()
    }

    /// Returns whether another transfer may be admitted into the bounded ready window.
    pub fn can_admit(&self) -> bool {
        self.ready.len() < self.capacity && !self.pending.is_empty()
    }

    /// Returns the next unit which must be submitted.
    pub fn next_pending(&self) -> Option<usize> {
        self.pending.front().copied()
    }

    /// Returns the current ready indices followed by future indices, truncated to lookahead.
    pub fn desired_indices(&self, lookahead: usize) -> Vec<usize> {
        self.ready
            .iter()
            .map(|(index, _)| *index)
            .chain(self.pending.iter().copied())
            .take(lookahead)
            .collect()
    }

    /// Commits one successfully submitted transfer in exact pending order.
    pub fn admit(&mut self, index: usize, transfer: T) -> Result<(), DenseTransferScheduleError> {
        if self.ready.len() >= self.capacity {
            return Err(DenseTransferScheduleError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        let expected = self
            .pending
            .front()
            .copied()
            .ok_or(DenseTransferScheduleError::NoPendingUnit)?;
        if expected != index {
            return Err(DenseTransferScheduleError::OutOfOrder {
                expected,
                actual: index,
            });
        }
        self.pending.pop_front();
        self.ready.push_back((index, transfer));
        Ok(())
    }

    /// Removes the oldest ready transfer for execution.
    pub fn pop_ready(&mut self) -> Option<(usize, T)> {
        self.ready.pop_front()
    }
}

/// Invalid transition in a bounded dense transfer schedule.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum DenseTransferScheduleError {
    /// A transfer window cannot have zero capacity.
    #[error("dense transfer window capacity must be nonzero")]
    ZeroCapacity,
    /// Pending units were not supplied in strictly increasing order.
    #[error("dense transfer pending unit {actual} does not follow {previous}")]
    UnorderedPending {
        /// Previous index.
        previous: usize,
        /// Invalid next index.
        actual: usize,
    },
    /// Admission was attempted while the ready window was full.
    #[error("dense transfer window exceeds its capacity of {capacity}")]
    CapacityExceeded {
        /// Configured ready capacity.
        capacity: usize,
    },
    /// Admission was attempted after all pending units were submitted.
    #[error("dense transfer schedule has no pending unit")]
    NoPendingUnit,
    /// A backend submitted transfers in a different order from the architecture sequence.
    #[error("dense transfer schedule expected unit {expected}, received {actual}")]
    OutOfOrder {
        /// Required next index.
        expected: usize,
        /// Submitted index.
        actual: usize,
    },
}

/// Loader controls for a host-backed layerwise execution engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LayerwiseLoadOptions {
    /// Residency budgets and maximum device-unit window.
    pub offload: OffloadConfig,
    /// Maximum number of checkpoint payload shards retained as mappings.
    pub max_mapped_shards: usize,
    /// Reject checkpoint tensors unrelated to the architecture parameter tree.
    pub strict_loading: bool,
    /// Sample backend allocator memory when a forward pass completes.
    pub sample_backend_memory: bool,
    /// Sample process memory metrics when a forward pass completes.
    pub sample_process_memory: bool,
}

impl LayerwiseLoadOptions {
    /// Creates strict options with the default mapped-shard bound.
    pub fn new(offload: OffloadConfig) -> Self {
        Self {
            offload,
            ..Self::default()
        }
    }
}

impl Default for LayerwiseLoadOptions {
    fn default() -> Self {
        Self {
            offload: OffloadConfig::default(),
            max_mapped_shards: DEFAULT_MAX_MAPPED_SHARDS,
            strict_loading: true,
            sample_backend_memory: false,
            sample_process_memory: false,
        }
    }
}

/// Controls for bounded dense-unit streaming from checkpoint storage.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct DenseDiskStreamLoadOptions {
    /// Finite logical device parameter budget, including pinned static weights.
    pub device_budget_bytes: u64,
    /// Finite charged host-allocation budget. Zero selects direct disk-to-device loading.
    pub host_budget_bytes: u64,
    /// Number of current and imminent unit host copies protected from eviction.
    pub host_lookahead: usize,
    /// Maximum number of pending background host materializations.
    pub background_queue_capacity: usize,
    /// Deterministic ordering used when unprotected cached copies must be evicted.
    pub eviction_policy: CacheEvictionPolicy,
    /// Maximum number of checkpoint payload shards retained as mappings.
    pub max_mapped_shards: usize,
    /// Reject checkpoint tensors unrelated to the architecture parameter tree.
    pub strict_loading: bool,
    /// Sample backend allocator memory after a forward pass.
    pub sample_backend_memory: bool,
    /// Sample process memory and page-fault counters after a forward pass.
    pub sample_process_memory: bool,
}

impl DenseDiskStreamLoadOptions {
    /// Creates strict streaming options with finite tier budgets.
    pub fn new(
        device_budget_bytes: u64,
        host_budget_bytes: u64,
        host_lookahead: usize,
        background_queue_capacity: usize,
    ) -> Result<Self, WeightResidencyPolicyError> {
        let options = Self {
            device_budget_bytes,
            host_budget_bytes,
            host_lookahead,
            background_queue_capacity,
            eviction_policy: CacheEvictionPolicy::LeastRecentlyUsed,
            max_mapped_shards: DEFAULT_MAX_MAPPED_SHARDS,
            strict_loading: true,
            sample_backend_memory: false,
            sample_process_memory: false,
        };
        options.validate()?;
        Ok(options)
    }

    /// Revalidates public fields after caller customization.
    pub fn validate(self) -> Result<(), WeightResidencyPolicyError> {
        if self.host_budget_bytes == 0 {
            if self.host_lookahead != 0 || self.background_queue_capacity != 0 {
                return Err(WeightResidencyPolicyError::HostDisabledControls);
            }
        } else {
            if self.host_lookahead == 0 {
                return Err(WeightResidencyPolicyError::ZeroHostLookahead);
            }
            if self.background_queue_capacity == 0 {
                return Err(WeightResidencyPolicyError::ZeroQueueCapacity);
            }
        }
        Ok(())
    }

    /// Selects deterministic cache eviction.
    pub const fn with_eviction_policy(mut self, policy: CacheEvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }
}

impl Default for DenseDiskStreamLoadOptions {
    fn default() -> Self {
        Self::new(4 << 30, 16 << 30, 2, 2).expect("default dense disk streaming controls are valid")
    }
}

/// Placement policy for ordinary architecture execution units.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub enum LayerWeightResidency {
    /// Construct every rank-local module once and retain it on the execution device.
    #[default]
    FullyResident,
    /// Eagerly materialize units on a host stream and use a bounded device window.
    LayerwiseHost(LayerwiseLoadOptions),
    /// Leave units cold on disk and use finite host and device caches.
    DenseDiskStream(DenseDiskStreamLoadOptions),
}

impl LayerWeightResidency {
    /// Returns the checkpoint shard/reader cache bound carried by this policy.
    pub const fn max_mapped_shards(self) -> usize {
        match self {
            Self::FullyResident => DEFAULT_MAX_MAPPED_SHARDS,
            Self::LayerwiseHost(options) => options.max_mapped_shards,
            Self::DenseDiskStream(options) => options.max_mapped_shards,
        }
    }

    /// Returns whether whole-artifact admission rejects unrelated tensors.
    pub const fn strict_loading(self) -> bool {
        match self {
            Self::FullyResident => true,
            Self::LayerwiseHost(options) => options.strict_loading,
            Self::DenseDiskStream(options) => options.strict_loading,
        }
    }

    /// Returns whether backend allocator memory should be sampled.
    pub const fn sample_backend_memory(self) -> bool {
        match self {
            Self::FullyResident => false,
            Self::LayerwiseHost(options) => options.sample_backend_memory,
            Self::DenseDiskStream(options) => options.sample_backend_memory,
        }
    }

    /// Returns whether process memory should be sampled.
    pub const fn sample_process_memory(self) -> bool {
        match self {
            Self::FullyResident => false,
            Self::LayerwiseHost(options) => options.sample_process_memory,
            Self::DenseDiskStream(options) => options.sample_process_memory,
        }
    }

    /// Returns the maximum execution-device unit window.
    pub fn device_depth(self, unit_count: usize) -> usize {
        match self {
            Self::FullyResident => unit_count,
            Self::LayerwiseHost(options) => options.offload.prefetch_depth(),
            Self::DenseDiskStream(_) => unit_count.min(DENSE_TRANSFER_WINDOW),
        }
    }

    /// Resolves this policy into the shared offload configuration.
    pub fn offload(self) -> Result<OffloadConfig, WeightResidencyPolicyError> {
        match self {
            Self::FullyResident => Ok(OffloadConfig::default()),
            Self::LayerwiseHost(options) => Ok(options.offload),
            Self::DenseDiskStream(options) => {
                options.validate()?;
                Ok(OffloadConfig::new(
                    Some(options.device_budget_bytes),
                    Some(options.host_budget_bytes),
                    options.host_lookahead.max(DENSE_TRANSFER_WINDOW),
                )?
                .with_eviction_policy(options.eviction_policy))
            }
        }
    }

    /// Returns dense-stream controls when this policy streams from disk.
    pub const fn dense(self) -> Option<DenseDiskStreamLoadOptions> {
        match self {
            Self::DenseDiskStream(options) => Some(options),
            Self::FullyResident | Self::LayerwiseHost(_) => None,
        }
    }

    /// Returns whether every ordinary execution unit remains device-resident.
    pub const fn is_fully_resident(self) -> bool {
        matches!(self, Self::FullyResident)
    }

    /// Returns the stable physical execution classification.
    pub const fn execution_residency(self) -> ExecutionResidency {
        match self {
            Self::FullyResident => ExecutionResidency::FullyResident,
            Self::LayerwiseHost(_) => ExecutionResidency::LayerwiseHost,
            Self::DenseDiskStream(_) => ExecutionResidency::DenseDiskStream,
        }
    }
}

impl From<LayerwiseLoadOptions> for LayerWeightResidency {
    fn from(value: LayerwiseLoadOptions) -> Self {
        Self::LayerwiseHost(value)
    }
}

impl From<DenseDiskStreamLoadOptions> for LayerWeightResidency {
    fn from(value: DenseDiskStreamLoadOptions) -> Self {
        Self::DenseDiskStream(value)
    }
}

/// Static-parameter placement used by the generalized execution engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExecutionResidency {
    /// Every module is constructed once and all rank-local parameters remain on device.
    FullyResident,
    /// Parameters remain on host behind bounded device windows.
    LayerwiseHost,
    /// Parameters are materialized through bounded disk, host, and device caches.
    DenseDiskStream,
}

/// Invalid backend-neutral immutable-weight residency policy.
#[derive(Debug, thiserror::Error)]
pub enum WeightResidencyPolicyError {
    /// Enabled host caching needs a protected current unit.
    #[error("dense disk streaming host lookahead must be nonzero when the host budget is enabled")]
    ZeroHostLookahead,
    /// Enabled background work requires bounded capacity.
    #[error("dense disk streaming background queue capacity must be nonzero when host caching is enabled")]
    ZeroQueueCapacity,
    /// Direct-to-device mode cannot configure host-only controls.
    #[error("dense disk streaming with a zero host budget requires zero host lookahead and queue capacity")]
    HostDisabledControls,
    /// The derived offload configuration was invalid.
    #[error(transparent)]
    Offload(#[from] OffloadError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_stream_controls_fail_closed_without_a_backend() {
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 1, 0, 1),
            Err(WeightResidencyPolicyError::ZeroHostLookahead)
        ));
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 1, 1, 0),
            Err(WeightResidencyPolicyError::ZeroQueueCapacity)
        ));
        assert!(matches!(
            DenseDiskStreamLoadOptions::new(1, 0, 1, 0),
            Err(WeightResidencyPolicyError::HostDisabledControls)
        ));
        assert!(DenseDiskStreamLoadOptions::new(1, 0, 0, 0).is_ok());
    }

    #[test]
    fn residency_policy_derives_finite_dense_windows() {
        let options = DenseDiskStreamLoadOptions::new(32, 64, 3, 2).unwrap();
        let policy = LayerWeightResidency::DenseDiskStream(options);
        assert_eq!(policy.device_depth(8), DENSE_TRANSFER_WINDOW);
        let offload = policy.offload().unwrap();
        assert_eq!(offload.device_budget_bytes(), Some(32));
        assert_eq!(offload.host_budget_bytes(), Some(64));
        assert_eq!(offload.prefetch_depth(), 3);
    }

    #[test]
    fn dense_transfer_schedule_preserves_order_and_bounded_lookahead() {
        let mut schedule = DenseTransferSchedule::new(3..7, 2).unwrap();
        assert_eq!(schedule.desired_indices(3), vec![3, 4, 5]);
        assert_eq!(
            schedule.admit(4, "wrong"),
            Err(DenseTransferScheduleError::OutOfOrder {
                expected: 3,
                actual: 4,
            })
        );
        schedule.admit(3, "three").unwrap();
        schedule.admit(4, "four").unwrap();
        assert!(!schedule.can_admit());
        assert_eq!(schedule.desired_indices(3), vec![3, 4, 5]);
        assert_eq!(schedule.pop_ready(), Some((3, "three")));
        schedule.admit(5, "five").unwrap();
        assert_eq!(schedule.desired_indices(4), vec![4, 5, 6]);
        assert_eq!(schedule.pop_ready(), Some((4, "four")));
        assert_eq!(schedule.pop_ready(), Some((5, "five")));
        schedule.admit(6, "six").unwrap();
        assert_eq!(schedule.pop_ready(), Some((6, "six")));
        assert!(schedule.is_exhausted());
    }
}
