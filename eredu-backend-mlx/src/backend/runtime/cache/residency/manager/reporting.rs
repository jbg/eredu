//! Bounded residency telemetry aggregation and attention accounting.

use super::*;

impl CacheResidencyManager {
    /// Returns a bounded aggregate snapshot without retaining per-block history.
    pub fn report(&self) -> Result<CacheResidencyReport, CacheResidencyError> {
        let mut state = self.lock()?;
        update_report_totals(&mut state);
        if self.options().process_sampling_enabled() {
            sample_process(&mut state.telemetry.report);
        }
        Ok(state.telemetry.report.clone())
    }

    /// Records one attention scan and its scratch-memory use in telemetry.
    pub fn record_attention_scan(
        &self,
        global_layer: usize,
        prefill: bool,
        blocks: u64,
        bytes: u64,
        scratch_bytes: u64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        if prefill {
            state.telemetry.report.prefill_full_attention_blocks += blocks;
            state.telemetry.report.prefill_full_attention_bytes += bytes;
            let activity = state.layer_activity_mut(global_layer);
            activity.prefill_full_attention_blocks += blocks;
            activity.prefill_full_attention_bytes += bytes;
        } else {
            state.telemetry.report.decode_full_attention_blocks += blocks;
            state.telemetry.report.decode_full_attention_bytes += bytes;
            let activity = state.layer_activity_mut(global_layer);
            activity.decode_full_attention_blocks += blocks;
            activity.decode_full_attention_bytes += bytes;
        }
        state.telemetry.report.attention_scratch_peak_bytes = state
            .telemetry
            .report
            .attention_scratch_peak_bytes
            .max(scratch_bytes);
        let activity = state.layer_activity_mut(global_layer);
        activity.attention_scratch_peak_bytes =
            activity.attention_scratch_peak_bytes.max(scratch_bytes);
        Ok(())
    }
}

pub(in super::super) fn update_report_totals(state: &mut CacheManagerState) {
    let device_budget_bytes = state.device_budget_bytes;
    let host_budget_bytes = state.host_budget_bytes;
    let disk_budget_bytes = state.disk_budget_bytes;
    let pool_disk_bytes = state
        .blocks
        .values()
        .filter(|record| record.disk().is_some_and(|location| !location.persistent))
        .map(|record| record.bytes)
        .sum::<u64>()
        .saturating_add(
            state
                .host_write_reservations
                .iter()
                .filter(|(key, _)| {
                    !state.blocks.get(&key.id).is_some_and(|record| {
                        record.disk().is_some_and(|location| !location.persistent)
                    })
                })
                .map(|(_, reservation)| reservation.logical_bytes)
                .sum(),
        );
    let report = &mut state.telemetry.report;
    report.key_value_blocks = 0;
    report.compressed_latent_blocks = 0;
    report.device_blocks = 0;
    report.host_blocks = 0;
    report.disk_blocks = 0;
    let tails = state.lifecycle.tails().collect::<Vec<_>>();
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
    for (layer, tail) in tails {
        layer_ends.insert(layer, tail.end);
        let layer_report = per_layer.entry(layer).or_default();
        layer_report.current_device_bytes += tail.bytes;
        layer_report.mutable_tail_bytes += tail.bytes;
        layer_report.logical_cached_tokens = tail.end.max(0) as u64;
    }
    for record in state.blocks.values() {
        let layer_report = per_layer
            .entry(record.physical.id().global_layer)
            .or_default();
        match record.physical.id().representation {
            CacheRepresentation::KeyValue => {
                report.key_value_blocks += 1;
                layer_report.key_value_blocks += 1;
            }
            CacheRepresentation::CompressedLatentRotary => {
                report.compressed_latent_blocks += 1;
                layer_report.compressed_latent_blocks += 1;
            }
        }
        let pending_write = record
            .pending_disk()
            .is_some_and(|pending| pending.ticket.key.kind == CacheIoOperationKind::Write);
        let pending_write_is_reserved = record.pending_disk().is_some_and(|pending| {
            state
                .host_write_reservations
                .contains_key(&pending.ticket.key)
        });
        if pending_write && !pending_write_is_reserved {
            let host_capacity = record
                .host_block()
                .and_then(|block| block.capacity().ok())
                .expect("pending cache write retains inspectable host-transfer storage");
            report.in_flight_write_blocks += 1;
            report.in_flight_write_bytes += host_capacity;
            layer_report.in_flight_write_blocks += 1;
            layer_report.in_flight_write_bytes += host_capacity;
        }
        match record.physical.phase() {
            CacheStoragePhase::Device => {
                report.device_blocks += 1;
                report.current_device_bytes += record.bytes;
                layer_report.device_blocks += 1;
                layer_report.current_device_bytes += record.bytes;
            }
            CacheStoragePhase::DemotingToHost => {
                let ticket = record
                    .physical
                    .host_demotion()
                    .expect("demoting phase owns its exact completion");
                let host_bytes = ticket.reserved_host_bytes;
                report.device_blocks += 1;
                report.host_blocks += 1;
                report.current_device_bytes += record.bytes;
                report.current_host_bytes += host_bytes;
                report.in_flight_host_demotion_blocks += 1;
                report.in_flight_host_demotion_bytes += host_bytes;
                layer_report.device_blocks += 1;
                layer_report.host_blocks += 1;
                layer_report.current_device_bytes += record.bytes;
                layer_report.current_host_bytes += host_bytes;
                layer_report.in_flight_host_demotion_blocks += 1;
                layer_report.in_flight_host_demotion_bytes += host_bytes;
            }
            CacheStoragePhase::HostUnbacked
            | CacheStoragePhase::HostWriting
            | CacheStoragePhase::HostBacked => {
                let block = record
                    .physical
                    .host_resource()
                    .expect("host phase owns host resources");
                let host_bytes = block
                    .capacity()
                    .expect("validated host cache block has inspectable capacity");
                report.host_blocks += 1;
                report.current_host_bytes += host_bytes;
                layer_report.host_blocks += 1;
                layer_report.current_host_bytes += host_bytes;
            }
            CacheStoragePhase::DiskReady | CacheStoragePhase::DiskReading => {
                report.disk_blocks += 1;
                report.current_disk_bytes += record.bytes;
                layer_report.disk_blocks += 1;
                layer_report.current_disk_bytes += record.bytes;
                if let Some(reserved_host_bytes) = record
                    .physical
                    .io()
                    .and_then(|pending| pending.reserved_host_bytes)
                {
                    report.current_host_bytes += reserved_host_bytes;
                    layer_report.current_host_bytes += reserved_host_bytes;
                }
            }
        }
        if state
            .lifecycle
            .is_protected_prefix(record.physical.id())
            .expect("storage block has lifecycle state")
        {
            report.protected_prefix_blocks += 1;
            layer_report.protected_prefix_blocks += 1;
        }
        layer_report.logical_cached_tokens = layer_report
            .logical_cached_tokens
            .max(record.physical.id().end.max(0) as u64);
        layer_ends
            .entry(record.physical.id().global_layer)
            .and_modify(|end| *end = (*end).max(record.physical.id().end))
            .or_insert(record.physical.id().end);
    }
    for (operation_id, reservation) in &state.retiring_host_demotions {
        let covered_by_demoting_record = state.blocks.get(&reservation.id).is_some_and(|record| {
            record
                .host_demotion_ticket()
                .is_some_and(|ticket| ticket.operation_id == *operation_id)
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
    for (key, reservation) in &state.host_write_reservations {
        report.in_flight_write_blocks += 1;
        report.in_flight_write_bytes += reservation.host_capacity;
        let layer_report = per_layer.entry(reservation.global_layer).or_default();
        layer_report.in_flight_write_blocks += 1;
        layer_report.in_flight_write_bytes += reservation.host_capacity;
        let covered_by_host_record = state.blocks.get(&key.id).is_some_and(|record| {
            record.tier() == CacheTier::Host
                && record
                    .pending_disk()
                    .is_some_and(|pending| pending.ticket.key == *key)
        });
        if !covered_by_host_record {
            report.current_host_bytes += reservation.host_capacity;
            layer_report.current_host_bytes += reservation.host_capacity;
        }
    }
    for (key, (global_layer, reserved_host_bytes)) in &state.retiring_disk_reads {
        let covered_by_pending_record = state.blocks.get(&key.id).is_some_and(|record| {
            record.physical.phase() == CacheStoragePhase::DiskReading
                && record.physical.io_matches(key)
        });
        if !covered_by_pending_record {
            report.current_host_bytes += reserved_host_bytes;
            per_layer
                .entry(*global_layer)
                .or_default()
                .current_host_bytes += reserved_host_bytes;
        }
    }
    let device_ids = state
        .blocks
        .values()
        .filter(|record| record.tier() == CacheTier::Device)
        .map(|record| record.physical.id().clone())
        .collect::<Vec<_>>();
    let recent = state
        .lifecycle
        .recent_protection_counts(device_ids, state.recent_device_blocks)
        .expect("storage blocks have lifecycle state");
    report.protected_recent_blocks = recent.values().sum();
    for (layer, count) in recent {
        per_layer.entry(layer).or_default().protected_recent_blocks = count;
    }
    report.logical_cached_tokens = layer_ends.values().copied().max().unwrap_or(0).max(0) as u64;
    state.telemetry.finalize_snapshot(
        per_layer,
        device_budget_bytes,
        host_budget_bytes,
        disk_budget_bytes,
    );
    let report = &state.telemetry.report;
    let pool_usage = CachePoolUsage {
        device_bytes: report.current_device_bytes,
        host_bytes: report.current_host_bytes,
        transfer_in_flight_bytes: report
            .in_flight_write_bytes
            .saturating_add(report.in_flight_host_demotion_bytes),
        // Pending writes reserve their eventual disk extent before the worker
        // owns the request, so aggregate disk admission cannot overcommit.
        disk_bytes: pool_disk_bytes,
    };
    let _ = state.pool.update_manager(state.pool_manager_id, pool_usage);
}

fn sample_process(report: &mut CacheResidencyReport) {
    if let Some(usage) = safemlx::system::process_usage() {
        report.process_rss_bytes = Some(usage.peak_rss);
        report.process_minor_page_faults = Some(usage.minor_page_faults);
        report.process_major_page_faults = Some(usage.major_page_faults);
    }
}
