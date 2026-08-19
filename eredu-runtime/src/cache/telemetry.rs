//! Backend-neutral mutable-cache residency telemetry.

use std::time::Duration;

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
}
