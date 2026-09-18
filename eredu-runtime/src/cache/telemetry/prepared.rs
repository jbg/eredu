//! Paid storage for the existing bounded activity/report collector.
use super::*;
use crate::cache::table::MetadataSource;
use crate::cache::{CacheTableCapacityError, PreparedCacheTable};
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding},
};
use std::mem::size_of;

/// Empty destinations for the collector's actual finite identity population.
/// These carry metadata storage only, never cache or execution authority.
#[derive(Debug)]
pub struct PreparedCacheTelemetry {
    activity: PreparedCacheTable<usize, CacheLayerResidencyStats>,
    report: Vec<CacheLayerResidencyReport>,
    maximum: usize,
    funding: Option<HostMetadataFunding>,
}

/// Empty predecessor buffers retire only after the backend manager unlocks.
#[derive(Debug)]
pub struct RetiredCacheTelemetryStorage {
    _activity: CacheTelemetryRows,
    _report: Vec<CacheLayerResidencyReport>,
    _funding: Option<HostMetadataFunding>,
}
impl PreparedCacheTelemetry {
    /// Complete storage and shared snapshot/installation control frames.
    pub fn control_bytes(maximum: usize) -> Option<usize> {
        if maximum > CACHE_RESIDENCY_LAYER_REPORT_LIMIT {
            return None;
        }
        Self::fixed_bytes()?
            .checked_add(
                PreparedCacheTable::<usize, CacheLayerResidencyStats>::control_bytes(maximum)?,
            )?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<
                CacheLayerResidencyReport,
            >(maximum)?)
    }
    fn fixed_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<MetadataSource<'_>>(),
            size_of::<RetiredCacheTelemetryStorage>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<RetiredCacheTelemetryStorage, Self>>(),
            size_of::<CacheResidencyTelemetry>(),
            size_of::<CacheLayerResidencyStats>(),
            size_of::<CacheLayerResidencyReport>(),
            size_of::<Option<CacheLayerResidencyStats>>(),
            size_of::<[usize; CACHE_RESIDENCY_LAYER_REPORT_LIMIT]>(),
            size_of::<(usize, usize, usize, u64, u64, Option<u64>)>(),
            size_of::<(&mut CacheResidencyTelemetry, &mut CacheTelemetryRows)>(),
            size_of::<Result<(), CacheTableCapacityError>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Pays final buffers before they can replace any canonical collector.
    pub fn prepare(maximum: usize, context: &WorkspaceContext) -> Result<Self, Error> {
        Self::prepare_from(maximum, MetadataSource::Context(context))
    }
    /// Same collector storage worker without constructing a recording Context.
    pub fn prepare_with_funding(maximum: usize, funding: &HostMetadataFunding) -> Result<Self, Error> {
        Self::prepare_from(maximum, MetadataSource::Funding(funding))
    }
    fn prepare_from(maximum: usize, source: MetadataSource<'_>) -> Result<Self, Error> {
        if maximum > CACHE_RESIDENCY_LAYER_REPORT_LIMIT {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        source.charge(Self::fixed_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        Ok(Self {
            activity: PreparedCacheTable::prepare_from(maximum, source)?,
            report: source.vector(maximum)?,
            maximum,
            funding: source.funding(),
        })
    }
}
impl CacheResidencyTelemetry {
    /// Fixed shared snapshot frames, including the actual bounded selection
    /// array. This excludes caller-owned row/report backing.
    pub fn snapshot_control_bytes() -> Option<usize> {
        PreparedCacheTelemetry::fixed_bytes()
    }

    /// Actual historical identities; later overflow never acquires a new row.
    pub fn activity_layers(&self) -> impl Iterator<Item = usize> + Clone + '_ {
        self.layer_activity.iter().map(|(layer, _)| *layer)
    }

    /// Exact possible bounded report population for these current/future layer
    /// identities plus the collector's retained historical identities.
    pub fn prepared_layer_count<I>(&self, layers: I) -> usize
    where
        I: Iterator<Item = usize> + Clone,
    {
        let additional = layers
            .clone()
            .enumerate()
            .filter(|(index, layer)| {
                !self.layer_activity.contains_key(layer)
                    && !layers.clone().take(*index).any(|earlier| earlier == *layer)
            })
            .count();
        self.layer_activity
            .len()
            .saturating_add(additional)
            .min(CACHE_RESIDENCY_LAYER_REPORT_LIMIT)
            .max(self.report.per_layer.len())
    }

    /// Validates the whole upcoming publication before its shared worker can
    /// mutate report or activity buffers. No fallback allocation is authorized.
    pub fn validate_prepared_layers<I>(&self, layers: I) -> Result<(), CacheTableCapacityError>
    where
        I: Iterator<Item = usize> + Clone,
    {
        let count = self.prepared_layer_count(layers);
        self.layer_activity.validate_prepared_population(count)?;
        if count > self.report.per_layer.capacity() {
            return Err(CacheTableCapacityError::Exhausted);
        }
        Ok(())
    }

    /// Validates both destination populations without moving either one.
    pub fn can_install(&self, prepared: &PreparedCacheTelemetry) -> bool {
        self.layer_activity.len() <= prepared.maximum
            && self.report.per_layer.len() <= prepared.maximum
    }

    /// Atomically replaces only storage, preserving all current and cumulative
    /// rows. The returned empty buffers/account must retire after manager unlock.
    pub fn install_storage(
        &mut self,
        mut prepared: PreparedCacheTelemetry,
    ) -> Result<RetiredCacheTelemetryStorage, PreparedCacheTelemetry> {
        if !self.can_install(&prepared) {
            return Err(prepared);
        }
        let activity = self
            .layer_activity
            .install(prepared.activity)
            .expect("validated collector activity destination");
        prepared.report.append(&mut self.report.per_layer);
        let report = std::mem::replace(&mut self.report.per_layer, prepared.report);
        let funding = std::mem::replace(&mut self.metadata_funding, prepared.funding);
        Ok(RetiredCacheTelemetryStorage {
            _activity: activity,
            _report: report,
            _funding: funding,
        })
    }
}

#[cfg(test)]
mod tests;
