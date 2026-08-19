//! Backend-neutral dense-stream residency telemetry.

use eredu_core::residency::{
    BackgroundPrefetchReport, MemoryTier, OffloadReport, TransferDirection,
};

use crate::{ResidencyReport, WeightMaterializationReport};

/// Stable dense-stream observations combining residency and worker state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DenseDiskStreamReport {
    planned_layer_count: usize,
    planned_layer_bytes: u64,
    maximum_host_layer_bytes: u64,
    pinned_static_device_bytes: u64,
    transfer_stream_index: i32,
    residency: ResidencyReport,
    background: BackgroundPrefetchReport,
    host_layers: DenseTierResidencyReport,
    device_layers: DenseTierResidencyReport,
    groups: Vec<DenseExecutionGroupReport>,
    prefill: DensePassReport,
    decode: DensePassReport,
}

impl DenseDiskStreamReport {
    /// Creates a complete dense-stream report from a coherent residency snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        planned_layer_count: usize,
        planned_layer_bytes: u64,
        maximum_host_layer_bytes: u64,
        pinned_static_device_bytes: u64,
        transfer_stream_index: i32,
        residency: ResidencyReport,
        background: BackgroundPrefetchReport,
        host_layers: DenseTierResidencyReport,
        device_layers: DenseTierResidencyReport,
        groups: Vec<DenseExecutionGroupReport>,
        prefill: DensePassReport,
        decode: DensePassReport,
    ) -> Self {
        Self {
            planned_layer_count,
            planned_layer_bytes,
            maximum_host_layer_bytes,
            pinned_static_device_bytes,
            transfer_stream_index,
            residency,
            background,
            host_layers,
            device_layers,
            groups,
            prefill,
            decode,
        }
    }

    /// Attaches bounded load-time materialization telemetry.
    pub fn with_materialization(
        mut self,
        materialization: Option<WeightMaterializationReport>,
    ) -> Self {
        self.residency = self.residency.with_materialization(materialization);
        self
    }

    /// Returns the number of disk-planned execution units.
    pub const fn planned_layer_count(&self) -> usize {
        self.planned_layer_count
    }
    /// Returns the logical checkpoint bytes in disk-planned execution units.
    pub const fn planned_layer_bytes(&self) -> u64 {
        self.planned_layer_bytes
    }
    /// Returns the charged host-transfer capacity of the largest execution unit.
    pub const fn maximum_host_layer_bytes(&self) -> u64 {
        self.maximum_host_layer_bytes
    }
    /// Returns pinned static parameter bytes outside the streamed-layer totals.
    pub const fn pinned_static_device_bytes(&self) -> u64 {
        self.pinned_static_device_bytes
    }
    /// Returns the distinct backend stream used for device weight transfers.
    pub const fn transfer_stream_index(&self) -> i32 {
        self.transfer_stream_index
    }
    /// Returns the complete logical tier and checkpoint-store report.
    pub const fn residency(&self) -> &ResidencyReport {
        &self.residency
    }
    /// Returns bounded background worker observations.
    pub const fn background(&self) -> BackgroundPrefetchReport {
        self.background
    }
    /// Returns streamed host-layer occupancy and cache history.
    pub const fn host_layers(&self) -> DenseTierResidencyReport {
        self.host_layers
    }
    /// Returns streamed device-layer occupancy and cache history.
    pub const fn device_layers(&self) -> DenseTierResidencyReport {
        self.device_layers
    }
    /// Returns point-in-time observations for each named execution group.
    pub fn execution_groups(&self) -> &[DenseExecutionGroupReport] {
        &self.groups
    }
    /// Returns completed prefill activity.
    pub const fn prefill(&self) -> DensePassReport {
        self.prefill
    }
    /// Returns completed decode activity.
    pub const fn decode(&self) -> DensePassReport {
        self.decode
    }
    /// Returns completed multi-token forward passes.
    pub const fn prefill_forwards(&self) -> u64 {
        self.prefill.forwards
    }
    /// Returns completed single-token forward passes.
    pub const fn decode_forwards(&self) -> u64 {
        self.decode.forwards
    }
}

/// Cache activity attributed to one logical residency tier.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct DenseCacheMetrics {
    requests: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    evicted_bytes: u64,
}

impl DenseCacheMetrics {
    /// Derives cache counters from one offload snapshot and tier.
    pub fn from_report(report: &OffloadReport, tier: MemoryTier) -> Self {
        let prefetch = report.tier_prefetch(tier);
        let evictions = report.tier_evictions(tier);
        Self {
            requests: prefetch.requests(),
            hits: prefetch.hits(),
            misses: prefetch.misses(),
            evictions: evictions.count(),
            evicted_bytes: evictions.bytes(),
        }
    }

    /// Returns cache requests targeting the tier.
    pub const fn requests(self) -> u64 {
        self.requests
    }
    /// Returns requests served by an existing tier copy.
    pub const fn hits(self) -> u64 {
        self.hits
    }
    /// Returns requests requiring tier materialization.
    pub const fn misses(self) -> u64 {
        self.misses
    }
    /// Returns copies evicted from the tier.
    pub const fn evictions(self) -> u64 {
        self.evictions
    }
    /// Returns logical bytes evicted from the tier.
    pub const fn evicted_bytes(self) -> u64 {
        self.evicted_bytes
    }

    fn saturating_delta(self, earlier: Self) -> Self {
        Self {
            requests: self.requests.saturating_sub(earlier.requests),
            hits: self.hits.saturating_sub(earlier.hits),
            misses: self.misses.saturating_sub(earlier.misses),
            evictions: self.evictions.saturating_sub(earlier.evictions),
            evicted_bytes: self.evicted_bytes.saturating_sub(earlier.evicted_bytes),
        }
    }

    fn saturating_add(&mut self, other: Self) {
        self.requests = self.requests.saturating_add(other.requests);
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.evictions = self.evictions.saturating_add(other.evictions);
        self.evicted_bytes = self.evicted_bytes.saturating_add(other.evicted_bytes);
    }
}

/// Streamed-layer occupancy and cache history for one tier.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct DenseTierResidencyReport {
    current_layer_count: usize,
    peak_layer_count: usize,
    current_layer_bytes: u64,
    peak_layer_bytes: u64,
    cache: DenseCacheMetrics,
}

impl DenseTierResidencyReport {
    /// Creates one logical tier occupancy snapshot.
    pub const fn new(
        current_layer_count: usize,
        peak_layer_count: usize,
        current_layer_bytes: u64,
        peak_layer_bytes: u64,
        cache: DenseCacheMetrics,
    ) -> Self {
        Self {
            current_layer_count,
            peak_layer_count,
            current_layer_bytes,
            peak_layer_bytes,
            cache,
        }
    }

    /// Returns currently resident streamed layers.
    pub const fn current_layer_count(self) -> usize {
        self.current_layer_count
    }
    /// Returns the peak number of simultaneously resident streamed layers.
    pub const fn peak_layer_count(self) -> usize {
        self.peak_layer_count
    }
    /// Returns current streamed-layer bytes in the tier.
    pub const fn current_layer_bytes(self) -> u64 {
        self.current_layer_bytes
    }
    /// Returns peak streamed-layer bytes in the tier.
    pub const fn peak_layer_bytes(self) -> u64 {
        self.peak_layer_bytes
    }
    /// Returns cumulative cache activity for the tier.
    pub const fn cache(self) -> DenseCacheMetrics {
        self.cache
    }
}

/// Point-in-time occupancy for one named execution stack.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DenseExecutionGroupReport {
    id: String,
    planned_layers: usize,
    planned_bytes: u64,
    completed_executions: u64,
    host_layers: usize,
    host_bytes: u64,
    peak_host_layers: usize,
    peak_host_bytes: u64,
    device_layers: usize,
    device_bytes: u64,
    peak_device_layers: usize,
    peak_device_bytes: u64,
}

impl DenseExecutionGroupReport {
    /// Creates one named execution-group report.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: impl Into<String>,
        planned_layers: usize,
        planned_bytes: u64,
        completed_executions: u64,
        host_layers: usize,
        host_bytes: u64,
        peak_host_layers: usize,
        peak_host_bytes: u64,
        device_layers: usize,
        device_bytes: u64,
        peak_device_layers: usize,
        peak_device_bytes: u64,
    ) -> Self {
        Self {
            id: id.into(),
            planned_layers,
            planned_bytes,
            completed_executions,
            host_layers,
            host_bytes,
            peak_host_layers,
            peak_host_bytes,
            device_layers,
            device_bytes,
            peak_device_layers,
            peak_device_bytes,
        }
    }

    /// Returns the stable execution-group identifier.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Returns disk-planned layers in the group.
    pub const fn planned_layers(&self) -> usize {
        self.planned_layers
    }
    /// Returns logical checkpoint bytes in the group.
    pub const fn planned_bytes(&self) -> u64 {
        self.planned_bytes
    }
    /// Returns successfully completed executions of this group.
    pub const fn completed_executions(&self) -> u64 {
        self.completed_executions
    }
    /// Returns current host-resident group layers.
    pub const fn host_layers(&self) -> usize {
        self.host_layers
    }
    /// Returns current host-resident group bytes.
    pub const fn host_bytes(&self) -> u64 {
        self.host_bytes
    }
    /// Returns the peak number of host-resident layers observed for the group.
    pub const fn peak_host_layers(&self) -> usize {
        self.peak_host_layers
    }
    /// Returns peak host-resident layer bytes observed for the group.
    pub const fn peak_host_bytes(&self) -> u64 {
        self.peak_host_bytes
    }
    /// Returns current device-resident group layers.
    pub const fn device_layers(&self) -> usize {
        self.device_layers
    }
    /// Returns current device-resident group bytes.
    pub const fn device_bytes(&self) -> u64 {
        self.device_bytes
    }
    /// Returns the peak number of device-resident layers observed for the group.
    pub const fn peak_device_layers(&self) -> usize {
        self.peak_device_layers
    }
    /// Returns peak device-resident layer bytes observed for the group.
    pub const fn peak_device_bytes(&self) -> u64 {
        self.peak_device_bytes
    }
}

/// Cache and logical transfer activity from completed prefill or decode forwards.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct DensePassReport {
    forwards: u64,
    host_cache: DenseCacheMetrics,
    device_cache: DenseCacheMetrics,
    peak_host_layers: usize,
    peak_host_bytes: u64,
    peak_device_layers: usize,
    peak_device_bytes: u64,
    disk_to_host_bytes: u64,
    disk_to_device_bytes: u64,
    host_to_device_bytes: u64,
}

impl DensePassReport {
    /// Returns completed forwards in this pass category.
    pub const fn forwards(self) -> u64 {
        self.forwards
    }
    /// Returns host-cache activity during completed forwards.
    pub const fn host_cache(self) -> DenseCacheMetrics {
        self.host_cache
    }
    /// Returns device-cache activity during completed forwards.
    pub const fn device_cache(self) -> DenseCacheMetrics {
        self.device_cache
    }
    /// Returns peak host-resident streamed layers observed during these forwards.
    pub const fn peak_host_layers(self) -> usize {
        self.peak_host_layers
    }
    /// Returns peak host-resident streamed-layer bytes during these forwards.
    pub const fn peak_host_bytes(self) -> u64 {
        self.peak_host_bytes
    }
    /// Returns peak device-resident streamed layers observed during these forwards.
    pub const fn peak_device_layers(self) -> usize {
        self.peak_device_layers
    }
    /// Returns peak device-resident streamed-layer bytes during these forwards.
    pub const fn peak_device_bytes(self) -> u64 {
        self.peak_device_bytes
    }
    /// Returns logical disk-to-host bytes during completed forwards.
    pub const fn disk_to_host_bytes(self) -> u64 {
        self.disk_to_host_bytes
    }
    /// Returns logical disk-to-device bytes during completed forwards.
    pub const fn disk_to_device_bytes(self) -> u64 {
        self.disk_to_device_bytes
    }
    /// Returns logical host-to-device bytes during completed forwards.
    pub const fn host_to_device_bytes(self) -> u64 {
        self.host_to_device_bytes
    }

    /// Replaces point-in-time peaks on one completed-pass delta.
    pub fn set_peaks(
        &mut self,
        host_layers: usize,
        host_bytes: u64,
        device_layers: usize,
        device_bytes: u64,
    ) {
        self.peak_host_layers = host_layers;
        self.peak_host_bytes = host_bytes;
        self.peak_device_layers = device_layers;
        self.peak_device_bytes = device_bytes;
    }

    /// Accumulates a completed pass with saturating counters and maximum peaks.
    pub fn accumulate(&mut self, other: Self) {
        self.forwards = self.forwards.saturating_add(other.forwards);
        self.host_cache.saturating_add(other.host_cache);
        self.device_cache.saturating_add(other.device_cache);
        self.peak_host_layers = self.peak_host_layers.max(other.peak_host_layers);
        self.peak_host_bytes = self.peak_host_bytes.max(other.peak_host_bytes);
        self.peak_device_layers = self.peak_device_layers.max(other.peak_device_layers);
        self.peak_device_bytes = self.peak_device_bytes.max(other.peak_device_bytes);
        self.disk_to_host_bytes = self
            .disk_to_host_bytes
            .saturating_add(other.disk_to_host_bytes);
        self.disk_to_device_bytes = self
            .disk_to_device_bytes
            .saturating_add(other.disk_to_device_bytes);
        self.host_to_device_bytes = self
            .host_to_device_bytes
            .saturating_add(other.host_to_device_bytes);
    }
}

/// Counter snapshot used to attribute residency activity to one forward pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct DensePassCounterSnapshot {
    host_cache: DenseCacheMetrics,
    device_cache: DenseCacheMetrics,
    disk_to_host_bytes: u64,
    disk_to_device_bytes: u64,
    host_to_device_bytes: u64,
}

impl DensePassCounterSnapshot {
    /// Captures logical cache and transfer counters from an offload report.
    pub fn from_report(report: &OffloadReport) -> Self {
        Self {
            host_cache: DenseCacheMetrics::from_report(report, MemoryTier::Host),
            device_cache: DenseCacheMetrics::from_report(report, MemoryTier::Device),
            disk_to_host_bytes: report.transfer(TransferDirection::DiskToHost).bytes(),
            disk_to_device_bytes: report.transfer(TransferDirection::DiskToDevice).bytes(),
            host_to_device_bytes: report.transfer(TransferDirection::HostToDevice).bytes(),
        }
    }

    /// Computes the saturating completed-forward delta from an earlier snapshot.
    pub fn delta(self, earlier: Self) -> DensePassReport {
        DensePassReport {
            forwards: 1,
            host_cache: self.host_cache.saturating_delta(earlier.host_cache),
            device_cache: self.device_cache.saturating_delta(earlier.device_cache),
            peak_host_layers: 0,
            peak_host_bytes: 0,
            peak_device_layers: 0,
            peak_device_bytes: 0,
            disk_to_host_bytes: self
                .disk_to_host_bytes
                .saturating_sub(earlier.disk_to_host_bytes),
            disk_to_device_bytes: self
                .disk_to_device_bytes
                .saturating_sub(earlier.disk_to_device_bytes),
            host_to_device_bytes: self
                .host_to_device_bytes
                .saturating_sub(earlier.host_to_device_bytes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_accumulation_preserves_maximum_peaks() {
        let mut total = DensePassReport::default();
        let mut first = DensePassReport::default();
        first.set_peaks(2, 20, 3, 30);
        let mut second = DensePassReport::default();
        second.set_peaks(4, 10, 1, 40);
        total.accumulate(first);
        total.accumulate(second);
        assert_eq!(total.peak_host_layers(), 4);
        assert_eq!(total.peak_host_bytes(), 20);
        assert_eq!(total.peak_device_layers(), 3);
        assert_eq!(total.peak_device_bytes(), 40);
    }
}
