//! Bounded residency telemetry aggregation and attention accounting.

use super::*;

impl<M: LiveCacheBackend> CacheResidencyManager<M> {
    /// Records range activity before copies execute; failure and rollback do not refund it.
    pub fn record_append_stream_read(
        &self,
        layer: usize,
        blocks: u64,
        bytes: u64,
        scratch: u64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        state.telemetry.report.append_stream_read_blocks += blocks;
        state.telemetry.report.append_stream_read_bytes += bytes;
        state.telemetry.report.append_stream_scratch_peak_bytes = state
            .telemetry
            .report
            .append_stream_scratch_peak_bytes
            .max(scratch);
        let activity = state.layer_activity_mut(layer);
        activity.append_stream_read_blocks += blocks;
        activity.append_stream_read_bytes += bytes;
        activity.append_stream_scratch_peak_bytes =
            activity.append_stream_scratch_peak_bytes.max(scratch);
        Ok(())
    }

    /// Returns a bounded aggregate snapshot without retaining per-block history.
    pub fn report(&self) -> Result<CacheResidencyReport, CacheResidencyError> {
        let mut state = self.lock()?;
        update_report_totals(&mut state);
        if self.options().process_sampling_enabled() {
            sample_process::<M>(&mut state.telemetry.report);
        }
        Ok(state.telemetry.report.clone())
    }

    /// Records selected reads immediately; later failures do not erase activity.
    pub fn record_selected_attention(
        &self,
        global_layer: usize,
        blocks: u64,
        bytes: u64,
        scratch_bytes: u64,
    ) -> Result<(), CacheResidencyError> {
        let mut state = self.lock()?;
        state.telemetry.report.selected_attention_blocks += blocks;
        state.telemetry.report.selected_attention_bytes += bytes;
        state.telemetry.report.attention_scratch_peak_bytes = state
            .telemetry
            .report
            .attention_scratch_peak_bytes
            .max(scratch_bytes);
        let activity = state.layer_activity_mut(global_layer);
        activity.selected_attention_blocks += blocks;
        activity.selected_attention_bytes += bytes;
        activity.attention_scratch_peak_bytes =
            activity.attention_scratch_peak_bytes.max(scratch_bytes);
        Ok(())
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

pub(super) fn update_report_totals<M: LiveCacheBackend>(state: &mut CacheManagerState<M>) {
    use crate::cache::catalog_telemetry::{
        account_cache_residency, CacheDemotionObservation, CachePageObservation, CacheReportLimits,
        CacheSpillObservation,
    };

    let pool_usage = account_cache_residency(
        &mut state.telemetry,
        &state.lifecycle,
        &state.blocks,
        &state.host_write_reservations,
        &state.retiring_host_demotions,
        &state.retiring_disk_reads,
        CacheReportLimits {
            device_bytes: state.device_budget_bytes,
            host_bytes: state.host_budget_bytes,
            disk_bytes: state.disk_budget_bytes,
            recent_device_blocks: state.recent_device_blocks,
        },
        |record| CachePageObservation {
            bytes: record.bytes,
            phase: record.physical.phase(),
            ephemeral_backing: record.disk().is_some_and(|location| !location.persistent),
            host_capacity: record.host_block().and_then(|block| block.capacity().ok()),
            demotion: record
                .host_demotion_ticket()
                .map(|ticket| (ticket.operation_id, ticket.reserved_host_bytes)),
            disk_io: record
                .pending_disk()
                .map(|operation| (&operation.ticket.key, operation.reserved_host_bytes)),
        },
        |reservation| CacheSpillObservation {
            logical_bytes: reservation.logical_bytes,
            host_capacity: reservation.host_capacity,
        },
        |reservation| CacheDemotionObservation {
            id: &reservation.id,
            device_bytes: reservation.device_bytes,
            host_bytes: reservation.host_bytes,
        },
    );
    let _ = state.pool.update_manager(state.pool_manager_id, pool_usage);
}

fn sample_process<M: LiveCacheBackend>(report: &mut CacheResidencyReport) {
    if let Some(usage) = M::process_usage() {
        report.process_rss_bytes = Some(usage.peak_rss);
        report.process_minor_page_faults = Some(usage.minor_faults);
        report.process_major_page_faults = Some(usage.major_faults);
    }
}
