//! Backend-neutral mutable-cache residency telemetry.

use std::{collections::BTreeMap, time::Duration};

/// Maximum number of individually identified layers in a residency report.
///
/// Additional active layers are folded into
/// [`CacheResidencyReport::per_layer_overflow`], so report size is independent
/// of caller-provided layer identifiers and remains bounded.
pub const CACHE_RESIDENCY_LAYER_REPORT_LIMIT: usize = 128;

/// Current residency and cumulative activity attributable to one layer or a
/// bounded overflow group of layers.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheLayerResidencyStats {
    /// Logical cached tokens.
    pub logical_cached_tokens: u64,
    /// Sealed key/value blocks.
    pub key_value_blocks: u64,
    /// Sealed compressed-latent/rotary blocks.
    pub compressed_latent_blocks: u64,
    /// Blocks cataloged on the execution device.
    pub device_blocks: u64,
    /// Blocks cataloged in host memory.
    pub host_blocks: u64,
    /// Blocks cataloged on disk.
    pub disk_blocks: u64,
    /// Current logical device bytes, including mutable tails.
    pub current_device_bytes: u64,
    /// Current physical host allocation capacity, including in-flight ownership.
    pub current_host_bytes: u64,
    /// Current logical disk bytes.
    pub current_disk_bytes: u64,
    /// Current bytes in mutable tails.
    pub mutable_tail_bytes: u64,
    /// Blocks whose host buffers are owned by background disk writes.
    pub in_flight_write_blocks: u64,
    /// Physical host allocation capacity owned by background disk writes.
    pub in_flight_write_bytes: u64,
    /// Blocks retaining both device and host allocations during demotion.
    pub in_flight_host_demotion_blocks: u64,
    /// Physical host allocation capacity charged during device demotion.
    pub in_flight_host_demotion_bytes: u64,
    /// Recent device blocks protected from demotion.
    pub protected_recent_blocks: u64,
    /// Prefix or sink blocks protected for attention.
    pub protected_prefix_blocks: u64,
    /// Cumulative host promotions.
    pub host_promotions: u64,
    /// Cumulative disk promotions.
    pub disk_promotions: u64,
    /// Cumulative device demotions.
    pub host_demotions: u64,
    /// Cumulative demotions to disk.
    pub disk_demotions: u64,
    /// Cumulative logical bytes transferred between tiers.
    pub transfer_bytes: u64,
    /// Cumulative host time at transfer ownership boundaries.
    pub transfer_wait: Duration,
    /// Cumulative demand hits.
    pub demand_hits: u64,
    /// Cumulative demand misses.
    pub demand_misses: u64,
    /// Cumulative in-flight waits.
    pub in_flight_waits: u64,
    /// Cumulative residency or transfer failures.
    pub failures: u64,
    /// Sealed and mutable blocks scanned by full attention during prefill.
    pub prefill_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during prefill.
    pub prefill_full_attention_bytes: u64,
    /// Sealed and mutable blocks scanned by full attention during decode.
    pub decode_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during decode.
    pub decode_full_attention_bytes: u64,
    /// Peak logical scratch bytes used by this layer's attention.
    pub attention_scratch_peak_bytes: u64,
}

impl CacheLayerResidencyStats {
    /// Adds another exact row into this aggregate, preserving peak fields.
    pub fn accumulate(&mut self, other: &Self) {
        self.logical_cached_tokens += other.logical_cached_tokens;
        self.key_value_blocks += other.key_value_blocks;
        self.compressed_latent_blocks += other.compressed_latent_blocks;
        self.device_blocks += other.device_blocks;
        self.host_blocks += other.host_blocks;
        self.disk_blocks += other.disk_blocks;
        self.current_device_bytes += other.current_device_bytes;
        self.current_host_bytes += other.current_host_bytes;
        self.current_disk_bytes += other.current_disk_bytes;
        self.mutable_tail_bytes += other.mutable_tail_bytes;
        self.in_flight_write_blocks += other.in_flight_write_blocks;
        self.in_flight_write_bytes += other.in_flight_write_bytes;
        self.in_flight_host_demotion_blocks += other.in_flight_host_demotion_blocks;
        self.in_flight_host_demotion_bytes += other.in_flight_host_demotion_bytes;
        self.protected_recent_blocks += other.protected_recent_blocks;
        self.protected_prefix_blocks += other.protected_prefix_blocks;
        self.host_promotions += other.host_promotions;
        self.disk_promotions += other.disk_promotions;
        self.host_demotions += other.host_demotions;
        self.disk_demotions += other.disk_demotions;
        self.transfer_bytes += other.transfer_bytes;
        self.transfer_wait += other.transfer_wait;
        self.demand_hits += other.demand_hits;
        self.demand_misses += other.demand_misses;
        self.in_flight_waits += other.in_flight_waits;
        self.failures += other.failures;
        self.prefill_full_attention_blocks += other.prefill_full_attention_blocks;
        self.prefill_full_attention_bytes += other.prefill_full_attention_bytes;
        self.decode_full_attention_blocks += other.decode_full_attention_blocks;
        self.decode_full_attention_bytes += other.decode_full_attention_bytes;
        self.attention_scratch_peak_bytes = self
            .attention_scratch_peak_bytes
            .max(other.attention_scratch_peak_bytes);
    }
}

/// Bounded, individually identified per-layer residency observations.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheLayerResidencyReport {
    /// Global model layer identifier.
    pub global_layer: usize,
    /// Current residency and cumulative activity for this layer.
    pub stats: CacheLayerResidencyStats,
}

/// Aggregated logical device/disk and physical host residency observations.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct CacheResidencyReport {
    /// Absolute token count represented by the longest layer.
    pub logical_cached_tokens: u64,
    /// Sealed key/value blocks.
    pub key_value_blocks: u64,
    /// Sealed compressed-latent/rotary blocks.
    pub compressed_latent_blocks: u64,
    /// Blocks cataloged on the execution device.
    pub device_blocks: u64,
    /// Blocks cataloged in host memory.
    pub host_blocks: u64,
    /// Blocks cataloged on disk.
    pub disk_blocks: u64,
    /// Current logical device bytes, including mutable tails.
    pub current_device_bytes: u64,
    /// Peak successfully admitted logical device bytes.
    pub peak_device_bytes: u64,
    /// Current physical host allocation capacity.
    pub current_host_bytes: u64,
    /// Peak successfully admitted physical host allocation capacity.
    pub peak_host_bytes: u64,
    /// Current logical disk bytes.
    pub current_disk_bytes: u64,
    /// Peak successfully admitted logical disk bytes.
    pub peak_disk_bytes: u64,
    /// Blocks whose host buffers are owned by background disk writes.
    pub in_flight_write_blocks: u64,
    /// Physical host capacity owned by disk writes.
    pub in_flight_write_bytes: u64,
    /// Peak physical host capacity owned by background disk writes.
    pub peak_in_flight_write_bytes: u64,
    /// Blocks retaining both device and host allocations during demotion.
    pub in_flight_host_demotion_blocks: u64,
    /// Physical host capacity charged during device demotion.
    pub in_flight_host_demotion_bytes: u64,
    /// Peak physical host capacity charged during device demotion.
    pub peak_in_flight_host_demotion_bytes: u64,
    /// Current bytes in mutable tails.
    pub mutable_tail_bytes: u64,
    /// Recent blocks protected from device demotion.
    pub protected_recent_blocks: u64,
    /// Prefix or sink blocks protected for attention.
    pub protected_prefix_blocks: u64,
    /// Current per-layer rows, sorted and bounded.
    pub per_layer: Vec<CacheLayerResidencyReport>,
    /// Number of active layers folded into the overflow row.
    pub per_layer_overflow_layers: u64,
    /// Exact aggregate of omitted and unidentifiable layers.
    pub per_layer_overflow: CacheLayerResidencyStats,
    /// Host-to-device promotions.
    pub host_promotions: u64,
    /// Disk-to-device promotions.
    pub disk_promotions: u64,
    /// Device-to-host demotions.
    pub host_demotions: u64,
    /// Resident-to-disk demotions.
    pub disk_demotions: u64,
    /// Logical bytes copied by promotions and demotions.
    pub transfer_bytes: u64,
    /// Host time spent at transfer ownership boundaries.
    pub transfer_wait: Duration,
    /// Blocks evicted after configured tiers were exhausted.
    pub evictions: u64,
    /// Sliding-window blocks discarded as invisible.
    pub discarded_sliding_blocks: u64,
    /// Completed block seals.
    pub block_seals: u64,
    /// Mutable tail allocations.
    pub tail_allocations: u64,
    /// Demand hits.
    pub demand_hits: u64,
    /// Demand misses.
    pub demand_misses: u64,
    /// Requests that joined an existing transfer.
    pub in_flight_waits: u64,
    /// Effective bounded disk request queue capacity.
    pub queue_capacity: usize,
    /// Peak observed queue occupancy.
    pub queue_peak_occupancy: usize,
    /// Requests delayed by queue capacity.
    pub queue_backpressure: u64,
    /// Requests canceled by reset or truncation.
    pub cancellations: u64,
    /// Cache transfer or persistence failures.
    pub failures: u64,
    /// Blocks scanned by full attention during prefill.
    pub prefill_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during prefill.
    pub prefill_full_attention_bytes: u64,
    /// Blocks scanned by full attention during decode.
    pub decode_full_attention_blocks: u64,
    /// Logical bytes scanned by full attention during decode.
    pub decode_full_attention_bytes: u64,
    /// Peak logical scratch bytes used by attention.
    pub attention_scratch_peak_bytes: u64,
    /// Successful prompt-cache saves.
    pub prompt_cache_saves: u64,
    /// Successful prompt-cache loads.
    pub prompt_cache_loads: u64,
    /// Logical bytes written or cataloged for prompt caches.
    pub prompt_cache_bytes: u64,
    /// Imported persistent shard count.
    pub imported_mapped_shards: u64,
    /// Optional peak process resident-set size.
    pub process_rss_bytes: Option<u64>,
    /// Optional cumulative minor page faults.
    pub process_minor_page_faults: Option<u64>,
    /// Optional cumulative major page faults.
    pub process_major_page_faults: Option<u64>,
}

/// Backend-neutral collector for bounded cache activity and snapshot assembly.
///
/// Backends update the aggregate report while inspecting their native storage,
/// then provide exact current per-layer totals to [`Self::finalize_snapshot`].
/// Historical layer identities and overflow accounting remain runtime-owned.
#[derive(Debug, Default)]
pub struct CacheResidencyTelemetry {
    /// Aggregate current and cumulative report fields.
    pub report: CacheResidencyReport,
    layer_activity: BTreeMap<usize, CacheLayerResidencyStats>,
    layer_activity_overflow: CacheLayerResidencyStats,
}

impl CacheResidencyTelemetry {
    /// Creates an empty collector with the effective I/O queue capacity.
    pub fn new(queue_capacity: usize) -> Self {
        let mut telemetry = Self::default();
        telemetry.report.queue_capacity = queue_capacity;
        telemetry
    }

    /// Returns cumulative activity storage for one identified layer.
    ///
    /// The first bounded set of layer identities remains stable for the life of
    /// the collector. Later identities are folded into an exact overflow row.
    pub fn layer_activity_mut(&mut self, global_layer: usize) -> &mut CacheLayerResidencyStats {
        if self.layer_activity.contains_key(&global_layer)
            || self.layer_activity.len() < CACHE_RESIDENCY_LAYER_REPORT_LIMIT
        {
            self.layer_activity.entry(global_layer).or_default()
        } else {
            &mut self.layer_activity_overflow
        }
    }

    /// Returns cumulative activity storage for work without a layer identity.
    pub fn unassigned_activity_mut(&mut self) -> &mut CacheLayerResidencyStats {
        &mut self.layer_activity_overflow
    }

    /// Merges exact current per-layer totals with cumulative runtime activity.
    ///
    /// Peak aggregate fields advance only when current usage is within the
    /// configured limit, matching successful-admission semantics.
    pub fn finalize_snapshot(
        &mut self,
        mut current: BTreeMap<usize, CacheLayerResidencyStats>,
        device_budget_bytes: u64,
        host_budget_bytes: u64,
        disk_budget_bytes: Option<u64>,
    ) {
        self.report.per_layer.clear();
        self.report.per_layer_overflow_layers = 0;
        self.report.per_layer_overflow = CacheLayerResidencyStats::default();

        let mut selected_layers = self.layer_activity.keys().copied().collect::<Vec<_>>();
        for global_layer in current.keys().copied() {
            if selected_layers.len() == CACHE_RESIDENCY_LAYER_REPORT_LIMIT {
                break;
            }
            if !self.layer_activity.contains_key(&global_layer) {
                selected_layers.push(global_layer);
            }
        }
        selected_layers.sort_unstable();
        for global_layer in selected_layers {
            let mut stats = current.remove(&global_layer).unwrap_or_default();
            if let Some(activity) = self.layer_activity.get(&global_layer) {
                apply_activity(activity, &mut stats);
            }
            self.report.per_layer.push(CacheLayerResidencyReport {
                global_layer,
                stats,
            });
        }
        for (_, stats) in current {
            self.report.per_layer_overflow_layers += 1;
            self.report.per_layer_overflow.accumulate(&stats);
        }
        apply_activity(
            &self.layer_activity_overflow,
            &mut self.report.per_layer_overflow,
        );

        if self.report.current_device_bytes <= device_budget_bytes {
            self.report.peak_device_bytes = self
                .report
                .peak_device_bytes
                .max(self.report.current_device_bytes);
        }
        if self.report.current_host_bytes <= host_budget_bytes {
            self.report.peak_host_bytes = self
                .report
                .peak_host_bytes
                .max(self.report.current_host_bytes);
        }
        if disk_budget_bytes.is_none_or(|budget| self.report.current_disk_bytes <= budget) {
            self.report.peak_disk_bytes = self
                .report
                .peak_disk_bytes
                .max(self.report.current_disk_bytes);
        }
        self.report.peak_in_flight_write_bytes = self
            .report
            .peak_in_flight_write_bytes
            .max(self.report.in_flight_write_bytes);
        self.report.peak_in_flight_host_demotion_bytes = self
            .report
            .peak_in_flight_host_demotion_bytes
            .max(self.report.in_flight_host_demotion_bytes);
    }
}

fn apply_activity(activity: &CacheLayerResidencyStats, stats: &mut CacheLayerResidencyStats) {
    stats.host_promotions += activity.host_promotions;
    stats.disk_promotions += activity.disk_promotions;
    stats.host_demotions += activity.host_demotions;
    stats.disk_demotions += activity.disk_demotions;
    stats.transfer_bytes += activity.transfer_bytes;
    stats.transfer_wait += activity.transfer_wait;
    stats.demand_hits += activity.demand_hits;
    stats.demand_misses += activity.demand_misses;
    stats.in_flight_waits += activity.in_flight_waits;
    stats.failures += activity.failures;
    stats.prefill_full_attention_blocks += activity.prefill_full_attention_blocks;
    stats.prefill_full_attention_bytes += activity.prefill_full_attention_bytes;
    stats.decode_full_attention_blocks += activity.decode_full_attention_blocks;
    stats.decode_full_attention_bytes += activity.decode_full_attention_bytes;
    stats.attention_scratch_peak_bytes = stats
        .attention_scratch_peak_bytes
        .max(activity.attention_scratch_peak_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregation_sums_current_and_cumulative_fields_but_preserves_peaks() {
        let mut aggregate = CacheLayerResidencyStats {
            current_device_bytes: 4,
            transfer_bytes: 8,
            attention_scratch_peak_bytes: 16,
            ..Default::default()
        };
        aggregate.accumulate(&CacheLayerResidencyStats {
            current_device_bytes: 5,
            transfer_bytes: 9,
            attention_scratch_peak_bytes: 12,
            ..Default::default()
        });
        assert_eq!(aggregate.current_device_bytes, 9);
        assert_eq!(aggregate.transfer_bytes, 17);
        assert_eq!(aggregate.attention_scratch_peak_bytes, 16);
    }

    #[test]
    fn telemetry_keeps_historical_layers_stable_and_folds_current_overflow() {
        let mut telemetry = CacheResidencyTelemetry::new(3);
        for layer in 0..CACHE_RESIDENCY_LAYER_REPORT_LIMIT {
            telemetry.layer_activity_mut(layer).demand_hits = 1;
        }
        telemetry.unassigned_activity_mut().failures = 2;
        telemetry.report.current_device_bytes = 8;
        telemetry.report.current_host_bytes = 12;
        telemetry.report.current_disk_bytes = 16;
        telemetry.finalize_snapshot(
            BTreeMap::from([(
                CACHE_RESIDENCY_LAYER_REPORT_LIMIT,
                CacheLayerResidencyStats {
                    device_blocks: 1,
                    ..Default::default()
                },
            )]),
            8,
            12,
            Some(16),
        );

        assert_eq!(telemetry.report.queue_capacity, 3);
        assert_eq!(
            telemetry.report.per_layer.len(),
            CACHE_RESIDENCY_LAYER_REPORT_LIMIT
        );
        assert_eq!(telemetry.report.per_layer_overflow_layers, 1);
        assert_eq!(telemetry.report.per_layer_overflow.device_blocks, 1);
        assert_eq!(telemetry.report.per_layer_overflow.failures, 2);
        assert_eq!(telemetry.report.peak_device_bytes, 8);
        assert_eq!(telemetry.report.peak_host_bytes, 12);
        assert_eq!(telemetry.report.peak_disk_bytes, 16);
    }
}
