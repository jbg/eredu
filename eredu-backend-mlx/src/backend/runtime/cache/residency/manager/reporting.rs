//! Bounded residency telemetry aggregation and attention accounting.

use super::*;
use eredu_runtime::cache::{
    CachePoolReservation, CacheRecentCountDestination, CacheTableCapacityError, CacheTelemetryRows,
};

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
        record_scan(
            &mut state,
            global_layer,
            prefill,
            blocks,
            bytes,
            scratch_bytes,
        );
        Ok(())
    }
}

pub(in super::super) fn update_report_totals(state: &mut CacheManagerState) {
    let mut current = state.report_rows.take().unwrap_or_default();
    let _ = update_report_totals_with_rows(state, &mut current, PoolPublication::Manager);
    state.report_rows = Some(current);
}

/// The original caller validates exact row capacity before the same tally can
/// mutate any report/activity destination. This method grants no cache work.
pub(in super::super) fn update_report_totals_prepared(
    state: &mut CacheManagerState,
) -> Result<(), CacheResidencyError> {
    update_report_totals_prepared_with(state, PoolPublication::Manager)
}

/// Copy publication uses the same actual tally, consuming only this destination's
/// admitted reservation instead of briefly charging the copied arrays twice.
pub(super) fn update_report_totals_prepared_copy(
    state: &mut CacheManagerState,
    reservation: &mut CachePoolReservation,
    membership: &CachePoolMembership,
) -> Result<(), CacheResidencyError> {
    update_report_totals_prepared_with(state, PoolPublication::Copy(reservation, membership))
}

/// Promotion keeps the removed host backing in the actual transfer owner.
pub(super) fn update_report_totals_prepared_replacement(
    state: &mut CacheManagerState,
    reservation: &mut CachePoolReservation,
    membership: &CachePoolMembership,
) -> Result<(), CacheResidencyError> {
    update_report_totals_prepared_with(state, PoolPublication::Replacement(reservation, membership))
}

enum PoolPublication<'a> {
    Manager,
    Copy(&'a mut CachePoolReservation, &'a CachePoolMembership),
    Replacement(&'a mut CachePoolReservation, &'a CachePoolMembership),
}

fn update_report_totals_prepared_with(
    state: &mut CacheManagerState,
    publication: PoolPublication<'_>,
) -> Result<(), CacheResidencyError> {
    let mut rows = state
        .report_rows
        .take()
        .ok_or_else(|| CacheLifecycleError::from(CacheTableCapacityError::Unprepared))?;
    rows.clear();
    let checked = (|| {
        for layer in report_layers(state) {
            if !rows.contains_key(&layer) {
                rows.insert_prepared(layer, CacheLayerResidencyStats::default())
                    .map_err(|(cause, _, _)| CacheLifecycleError::from(cause))?;
            }
        }
        state
            .telemetry
            .validate_prepared_layers(rows.iter().map(|(layer, _)| *layer))
            .map_err(CacheLifecycleError::from)?;
        Ok::<_, CacheResidencyError>(())
    })();
    rows.clear();
    let result = checked.and_then(|_| {
        update_report_totals_with_rows(state, &mut rows, publication).map_err(Into::into)
    });
    state.report_rows = Some(rows);
    result
}

pub(super) fn report_layers(state: &CacheManagerState) -> impl Iterator<Item = usize> + Clone + '_ {
    state
        .blocks
        .iter()
        .map(|(id, _)| id.global_layer)
        .chain(state.lifecycle.tail_layers())
        .chain(
            state
                .retiring_host_demotions
                .values()
                .map(|value| value.id.global_layer),
        )
        .chain(
            state
                .host_write_reservations
                .iter()
                .map(|(_, value)| value.global_layer),
        )
        .chain(state.retiring_disk_reads.values().map(|(layer, _)| *layer))
}
struct RecentRows<'a>(&'a mut CacheTelemetryRows);
impl CacheRecentCountDestination for RecentRows<'_> {
    fn recent_count_mut(&mut self, global_layer: usize) -> &mut u64 {
        &mut self.0.get_or_default(global_layer).protected_recent_blocks
    }
}

fn update_report_totals_with_rows(
    state: &mut CacheManagerState,
    per_layer: &mut CacheTelemetryRows,
    publication: PoolPublication<'_>,
) -> Result<(), CachePoolError> {
    let device_budget_bytes = state.device_budget_bytes;
    let host_budget_bytes = state.host_budget_bytes;
    let disk_budget_bytes = state.disk_budget_bytes;
    let pool_disk_bytes = state
        .blocks
        .values()
        .filter(|record| {
            record.disk().is_some_and(|location| {
                !location.persistent
                    && !location
                        .live_source
                        .as_ref()
                        .is_some_and(LiveCacheBlockSource::owns_disk_reservation)
            })
        })
        .map(|record| record.bytes)
        .sum::<u64>()
        .saturating_add(
            state
                .host_write_reservations
                .iter()
                .filter(|(key, reservation)| {
                    if reservation.prepared.is_some() {
                        return false;
                    }
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
    report.current_device_bytes = state.lifecycle.tails().map(|(_, tail)| tail.bytes).sum();
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
    per_layer.clear();
    for (layer, tail) in state.lifecycle.tails() {
        let layer_report = per_layer.get_or_default(layer);
        layer_report.current_device_bytes += tail.bytes;
        layer_report.mutable_tail_bytes += tail.bytes;
        layer_report.logical_cached_tokens = tail.end.max(0) as u64;
    }
    for record in state.blocks.values() {
        let layer_report = per_layer.get_or_default(record.physical.id().global_layer);
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
        let layer_report = per_layer.get_or_default(reservation.id.global_layer);
        layer_report.device_blocks += 1;
        layer_report.host_blocks += 1;
        layer_report.current_device_bytes += reservation.device_bytes;
        layer_report.current_host_bytes += reservation.host_bytes;
        layer_report.in_flight_host_demotion_blocks += 1;
        layer_report.in_flight_host_demotion_bytes += reservation.host_bytes;
    }
    for (key, reservation) in state.host_write_reservations.iter() {
        report.in_flight_write_blocks += 1;
        report.in_flight_write_bytes += reservation.host_capacity;
        let layer_report = per_layer.get_or_default(reservation.global_layer);
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
            per_layer.get_or_default(*global_layer).current_host_bytes += reserved_host_bytes;
        }
    }
    let device_ids = state
        .blocks
        .iter()
        .filter(|(_, record)| record.tier() == CacheTier::Device)
        .map(|(id, _)| id);
    state
        .lifecycle
        .accumulate_recent_protection_counts(
            device_ids,
            state.recent_device_blocks,
            &mut RecentRows(per_layer),
        )
        .expect("canonical storage blocks have lifecycle state");
    report.protected_recent_blocks = per_layer
        .values()
        .map(|row| row.protected_recent_blocks)
        .sum();
    report.logical_cached_tokens = per_layer
        .values()
        .map(|row| row.logical_cached_tokens)
        .max()
        .unwrap_or(0);
    state.telemetry.finalize_snapshot_with_rows(
        per_layer,
        device_budget_bytes,
        host_budget_bytes,
        disk_budget_bytes,
    );
    // Prepared writes own these exact reservations independently. Preserve
    // canonical telemetry while charging each actual backing only once.
    let mut external_transfer = 0u64;
    for (_, reservation) in state.host_write_reservations.iter() {
        let Some(owner) = &reservation.prepared else {
            continue;
        };
        external_transfer = external_transfer.checked_add(owner.host_bytes()).ok_or(
            CachePoolError::AccountingOverflow {
                operation: "prepared write transfer tally",
            },
        )?;
    }
    let mut external_read_host = 0u64;
    for (_, record) in state.blocks.iter() {
        if let Some(owner) = record
            .physical
            .io()
            .and_then(|io| io.prepared_read.as_ref())
        {
            external_read_host = external_read_host.checked_add(owner.host_bytes()).ok_or(
                CachePoolError::AccountingOverflow {
                    operation: "prepared read Host ownership tally",
                },
            )?;
        }
    }
    let report = &state.telemetry.report;
    let pool_usage = CachePoolUsage {
        device_bytes: report.current_device_bytes,
        host_bytes: report
            .current_host_bytes
            .checked_sub(external_read_host)
            .ok_or(CachePoolError::AccountingOverflow {
                operation: "prepared read Host ownership",
            })?,
        transfer_in_flight_bytes: report
            .in_flight_write_bytes
            .checked_add(report.in_flight_host_demotion_bytes)
            .and_then(|n| n.checked_sub(external_transfer))
            .ok_or(CachePoolError::AccountingOverflow {
                operation: "prepared write transfer ownership",
            })?,
        // Pending writes reserve their eventual disk extent before the worker
        // owns the request, so aggregate disk admission cannot overcommit.
        disk_bytes: pool_disk_bytes,
    };
    match publication {
        PoolPublication::Manager => state
            .pool
            .update_manager(state.pool_manager_id, pool_usage)
            .map(drop),
        PoolPublication::Copy(reservation, membership) => {
            reservation.publish_to_manager(membership, pool_usage)
        }
        PoolPublication::Replacement(reservation, membership) => {
            reservation.publish_retaining_replaced_storage(membership, pool_usage)
        }
    }
}

/// Source-side query for the actual shared publication worker, including its
/// concrete iterator/closure frames. It inspects no native array and performs
/// no pool mutation; accepted callbacks must charge these controls per call.
pub(in super::super) fn report_control_bytes(state: &CacheManagerState) -> Option<usize> {
    use std::mem::size_of;
    fn recent<I>(_: &I) -> Option<usize> {
        CacheBlockLifecycle::recent_count_control_bytes::<I, RecentRows<'_>>()
    }
    let candidates = state
        .blocks
        .iter()
        .filter(|(_, record)| record.tier() == CacheTier::Device)
        .map(|(id, _)| id);
    let frames = [
        size_of::<(
            &mut CacheManagerState,
            &mut CacheTelemetryRows,
            PoolPublication<'_>,
        )>(),
        size_of::<PoolPublication<'_>>(),
        size_of::<CacheTelemetryRows>(),
        size_of::<Option<CacheTelemetryRows>>(),
        size_of::<CacheLayerResidencyStats>(),
        size_of::<CachePoolUsage>(),
        size_of::<(
            u64,
            u64,
            Option<u64>,
            u64,
            u64,
            bool,
            bool,
            u64,
            u64,
            Option<&source::DiskReadOccupancy>,
        )>(),
        size_of::<Result<(), CacheResidencyError>>(),
        size_of::<Result<(), CachePoolError>>(),
        size_of::<Result<(), CacheLifecycleError>>(),
        std::mem::size_of_val(&report_layers(state)),
        std::mem::size_of_val(&state.blocks.iter()),
        std::mem::size_of_val(&state.lifecycle.tails()),
        std::mem::size_of_val(&state.host_write_reservations.iter()),
        std::mem::size_of_val(&state.retiring_host_demotions.iter()),
        std::mem::size_of_val(&state.retiring_disk_reads.iter()),
        recent(&candidates)?,
        CacheResidencyTelemetry::snapshot_control_bytes()?,
        CacheTelemetryRows::mutation_control_bytes()?,
        CacheResidencyPool::publication_control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

/// Frames of the source-side size query itself. Iterator adapters here retain
/// no allocation or native object; their closure captures are empty except for
/// the borrowed canonical table. The twenty entries match the publication
/// frame inventory above, not a population allowance.
pub(in super::super) fn report_query_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    let frames = [
        size_of::<[usize; 20]>(),
        size_of::<eredu_runtime::cache::CacheRecordTableIter<'_, CacheBlockId, CacheBlockRecord>>(),
        size_of::<(&CacheManagerState, Option<usize>, usize)>(),
        size_of::<std::array::IntoIter<usize, 20>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

fn sample_process(report: &mut CacheResidencyReport) {
    if let Some(usage) = safemlx::system::process_usage() {
        report.process_rss_bytes = Some(usage.peak_rss);
        report.process_minor_page_faults = Some(usage.minor_page_faults);
        report.process_major_page_faults = Some(usage.major_page_faults);
    }
}

/// Shared actual scan telemetry; no source, allocation or submission authority.
pub(super) fn record_scan(
    state: &mut CacheManagerState,
    global_layer: usize,
    prefill: bool,
    blocks: u64,
    bytes: u64,
    scratch_bytes: u64,
) {
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
}
