//! Portable accounting for a live catalog and its retained cache operations.

use std::collections::{BTreeMap, HashMap};

use eredu_core::cache::{
    mechanisms::{CacheIoOperationKey, CacheIoOperationKind},
    CacheBlockId, CacheRepresentation, CacheTier,
};

use super::{
    CacheBlockLifecycle, CacheLayerResidencyStats, CachePoolUsage, CacheResidencyTelemetry,
    CacheStoragePhase,
};

/// The exact physical facts of one cataloged cache page.
pub struct CachePageObservation<'a> {
    /// Admitted logical device or backing extent.
    pub bytes: u64,
    /// Current phase of the page's physical resources.
    pub phase: CacheStoragePhase,
    /// Whether the page retains a live spill file charged to this cache.
    pub ephemeral_backing: bool,
    /// Actual host capacity when this page is host resident.
    pub host_capacity: Option<u64>,
    /// Exact device-to-host transfer identity and reserved host capacity.
    pub demotion: Option<(u64, u64)>,
    /// Exact backing I/O and any host capacity reserved by that operation.
    pub disk_io: Option<(&'a CacheIoOperationKey, Option<u64>)>,
}

/// Resources retained by one spill until its worker settles.
pub struct CacheSpillObservation {
    /// Logical extent charged against the disk budget.
    pub logical_bytes: u64,
    /// Physical host allocation retained during the write.
    pub host_capacity: u64,
}

/// Resources retained by one detached device-to-host transfer.
pub struct CacheDemotionObservation<'a> {
    /// Page whose device storage is still retained.
    pub id: &'a CacheBlockId,
    /// Logical device extent held by the transfer.
    pub device_bytes: u64,
    /// Physical host capacity reserved for the transfer.
    pub host_bytes: u64,
}

/// Exact configured limits represented in cache telemetry.
pub struct CacheReportLimits {
    /// Device capacity available to the cache.
    pub device_bytes: u64,
    /// Host capacity available to the cache.
    pub host_bytes: u64,
    /// Live spill capacity, when enabled.
    pub disk_bytes: Option<u64>,
    /// Recent pages protected from eviction on each stream.
    pub recent_device_blocks: usize,
}

/// Aggregates the shared residency report and returns the charge for this pool member.
/// The caller supplies observed allocation facts while holding its catalog access.
pub fn account_cache_residency<B, W, D>(
    telemetry: &mut CacheResidencyTelemetry,
    lifecycle: &CacheBlockLifecycle,
    blocks: &BTreeMap<CacheBlockId, B>,
    spills: &HashMap<CacheIoOperationKey, W>,
    detached_demotions: &HashMap<u64, D>,
    detached_reads: &HashMap<CacheIoOperationKey, (usize, u64)>,
    limits: CacheReportLimits,
    page_facts: fn(&B) -> CachePageObservation<'_>,
    spill_facts: fn(&W) -> CacheSpillObservation,
    demotion_facts: fn(&D) -> CacheDemotionObservation<'_>,
) -> CachePoolUsage {
    let mut pool_disk_bytes = 0u64;
    let mut device_ids = Vec::new();
    let report = &mut telemetry.report;
    report.key_value_blocks = 0;
    report.compressed_latent_blocks = 0;
    report.append_stream_blocks = 0;
    report.device_blocks = 0;
    report.host_blocks = 0;
    report.disk_blocks = 0;
    let tails = lifecycle.tails().collect::<Vec<_>>();
    report.current_device_bytes = tails.iter().map(|(_, tail)| tail.bytes).sum();
    report.mutable_tail_bytes = report.current_device_bytes;
    report.current_host_bytes = 0;
    report.current_disk_bytes = 0;
    report.in_flight_write_blocks = 0;
    report.in_flight_write_bytes = 0;
    report.in_flight_host_demotion_blocks = 0;
    report.in_flight_host_demotion_bytes = 0;
    report.protected_prefix_blocks = 0;
    report.protected_recent_blocks = 0;
    report.logical_cached_tokens = 0;
    let mut per_layer = BTreeMap::<usize, CacheLayerResidencyStats>::new();
    let mut layer_ends: HashMap<usize, i64> = HashMap::new();
    for (stream, tail) in tails {
        let layer = stream.global_layer;
        if !matches!(
            stream.representation,
            CacheRepresentation::AppendStream { .. }
        ) {
            layer_ends
                .entry(layer)
                .and_modify(|end| *end = (*end).max(tail.end))
                .or_insert(tail.end);
        }
        let layer_report = per_layer.entry(layer).or_default();
        layer_report.current_device_bytes += tail.bytes;
        layer_report.mutable_tail_bytes += tail.bytes;
        layer_report.logical_cached_tokens =
            layer_ends.get(&layer).copied().unwrap_or(0).max(0) as u64;
    }
    for (id, record) in blocks {
        let page = page_facts(record);
        if page.ephemeral_backing {
            pool_disk_bytes = pool_disk_bytes.saturating_add(page.bytes);
        }
        if page.phase.tier() == CacheTier::Device {
            device_ids.push(id.clone());
        }
        let layer_report = per_layer.entry(id.global_layer).or_default();
        match id.representation {
            CacheRepresentation::AppendStream { .. } => {
                report.append_stream_blocks += 1;
                layer_report.append_stream_blocks += 1;
            }
            CacheRepresentation::KeyValue => {
                report.key_value_blocks += 1;
                layer_report.key_value_blocks += 1;
            }
            CacheRepresentation::CompressedLatentRotary => {
                report.compressed_latent_blocks += 1;
                layer_report.compressed_latent_blocks += 1;
            }
        }
        if page.disk_io.is_some_and(|(key, _)| {
            key.kind == CacheIoOperationKind::Write && !spills.contains_key(key)
        }) {
            let host_capacity = page
                .host_capacity
                .expect("pending cache write retains inspectable host-transfer storage");
            report.in_flight_write_blocks += 1;
            report.in_flight_write_bytes += host_capacity;
            layer_report.in_flight_write_blocks += 1;
            layer_report.in_flight_write_bytes += host_capacity;
        }
        match page.phase {
            CacheStoragePhase::Device => {
                report.device_blocks += 1;
                report.current_device_bytes += page.bytes;
                layer_report.device_blocks += 1;
                layer_report.current_device_bytes += page.bytes;
            }
            CacheStoragePhase::DemotingToHost => {
                let (_, host_bytes) = page
                    .demotion
                    .expect("demoting phase owns its exact completion");
                report.device_blocks += 1;
                report.host_blocks += 1;
                report.current_device_bytes += page.bytes;
                report.current_host_bytes += host_bytes;
                report.in_flight_host_demotion_blocks += 1;
                report.in_flight_host_demotion_bytes += host_bytes;
                layer_report.device_blocks += 1;
                layer_report.host_blocks += 1;
                layer_report.current_device_bytes += page.bytes;
                layer_report.current_host_bytes += host_bytes;
                layer_report.in_flight_host_demotion_blocks += 1;
                layer_report.in_flight_host_demotion_bytes += host_bytes;
            }
            CacheStoragePhase::HostUnbacked
            | CacheStoragePhase::HostWriting
            | CacheStoragePhase::HostBacked => {
                let host_bytes = page
                    .host_capacity
                    .expect("validated host cache block has inspectable capacity");
                report.host_blocks += 1;
                report.current_host_bytes += host_bytes;
                layer_report.host_blocks += 1;
                layer_report.current_host_bytes += host_bytes;
            }
            CacheStoragePhase::DiskReady | CacheStoragePhase::DiskReading => {
                report.disk_blocks += 1;
                report.current_disk_bytes += page.bytes;
                layer_report.disk_blocks += 1;
                layer_report.current_disk_bytes += page.bytes;
                if let Some((_, Some(reserved_host_bytes))) = page.disk_io {
                    report.current_host_bytes += reserved_host_bytes;
                    layer_report.current_host_bytes += reserved_host_bytes;
                }
            }
        }
        if lifecycle
            .is_protected_prefix(id)
            .expect("storage block has lifecycle state")
        {
            report.protected_prefix_blocks += 1;
            layer_report.protected_prefix_blocks += 1;
        }
        if !matches!(id.representation, CacheRepresentation::AppendStream { .. }) {
            layer_report.logical_cached_tokens =
                layer_report.logical_cached_tokens.max(id.end.max(0) as u64);
            layer_ends
                .entry(id.global_layer)
                .and_modify(|end| *end = (*end).max(id.end))
                .or_insert(id.end);
        }
    }
    for (operation_id, reservation) in detached_demotions {
        let reservation = demotion_facts(reservation);
        let covered_by_demoting_record = blocks.get(reservation.id).is_some_and(|record| {
            page_facts(record)
                .demotion
                .is_some_and(|(id, _)| id == *operation_id)
        });
        if covered_by_demoting_record {
            continue;
        }
        report.device_blocks += 1;
        report.host_blocks += 1;
        report.current_device_bytes += reservation.device_bytes;
        report.current_host_bytes += reservation.host_bytes;
        report.in_flight_host_demotion_blocks += 1;
        report.in_flight_host_demotion_bytes += reservation.host_bytes;
        let layer_report = per_layer.entry(reservation.id.global_layer).or_default();
        layer_report.device_blocks += 1;
        layer_report.host_blocks += 1;
        layer_report.current_device_bytes += reservation.device_bytes;
        layer_report.current_host_bytes += reservation.host_bytes;
        layer_report.in_flight_host_demotion_blocks += 1;
        layer_report.in_flight_host_demotion_bytes += reservation.host_bytes;
    }
    for (key, reservation) in spills {
        let reservation = spill_facts(reservation);
        report.in_flight_write_blocks += 1;
        report.in_flight_write_bytes += reservation.host_capacity;
        let layer_report = per_layer.entry(key.id.global_layer).or_default();
        layer_report.in_flight_write_blocks += 1;
        layer_report.in_flight_write_bytes += reservation.host_capacity;
        let page = blocks.get(&key.id).map(page_facts);
        if !page.as_ref().is_some_and(|page| page.ephemeral_backing) {
            pool_disk_bytes = pool_disk_bytes.saturating_add(reservation.logical_bytes);
        }
        let covered_by_host_record = page.is_some_and(|page| {
            page.phase.tier() == CacheTier::Host
                && page.disk_io.is_some_and(|(pending, _)| pending == key)
        });
        if !covered_by_host_record {
            report.current_host_bytes += reservation.host_capacity;
            layer_report.current_host_bytes += reservation.host_capacity;
        }
    }
    for (key, (global_layer, reserved_host_bytes)) in detached_reads {
        let covered_by_pending_record = blocks.get(&key.id).is_some_and(|record| {
            let page = page_facts(record);
            page.phase == CacheStoragePhase::DiskReading
                && page.disk_io.is_some_and(|(pending, _)| pending == key)
        });
        if !covered_by_pending_record {
            report.current_host_bytes += reserved_host_bytes;
            per_layer
                .entry(*global_layer)
                .or_default()
                .current_host_bytes += reserved_host_bytes;
        }
    }
    let recent = lifecycle
        .recent_protection_counts(device_ids, limits.recent_device_blocks)
        .expect("storage blocks have lifecycle state");
    report.protected_recent_blocks = recent.values().sum();
    for (layer, count) in recent {
        per_layer.entry(layer).or_default().protected_recent_blocks = count;
    }
    report.logical_cached_tokens = layer_ends.values().copied().max().unwrap_or(0).max(0) as u64;
    telemetry.finalize_snapshot(
        per_layer,
        limits.device_bytes,
        limits.host_bytes,
        limits.disk_bytes,
    );
    let report = &telemetry.report;
    CachePoolUsage {
        device_bytes: report.current_device_bytes,
        host_bytes: report.current_host_bytes,
        transfer_in_flight_bytes: report
            .in_flight_write_bytes
            .saturating_add(report.in_flight_host_demotion_bytes),
        // Pending writes reserve their eventual disk extent before the worker
        // owns the request, so aggregate disk admission cannot overcommit.
        disk_bytes: pool_disk_bytes,
    }
}
