//! Paid destinations bound to one actual canonical manager source.
use super::*;
use eredu_runtime::cache::{
    CacheRecordTable, PreparedCacheLifecycle, PreparedCacheTable, RetiredCacheLifecycleStorage,
    CacheTelemetryRows, PreparedCacheTelemetry, RetiredCacheTelemetryStorage,
};

/// One-use metadata installation. The retained manager/generation comes from
/// its lexical source loan; this is neither a native role nor mutation grant.
pub(crate) struct PreparedManagerCatalog {
    manager: CacheResidencyManager,
    generation: u64,
    publication_controls: usize,
    population: (usize, usize),
    physical: Option<PreparedCacheTable<CacheBlockId, CacheBlockRecord>>,
    logical: Option<PreparedCacheLifecycle>,
    telemetry: Option<PreparedCacheTelemetry>,
    current_rows: Option<CacheTelemetryRows>,
    _funding: Option<HostMetadataFunding>,
}
struct RetiredCatalog {
    _physical: CacheRecordTable<CacheBlockId, CacheBlockRecord>,
    _logical: RetiredCacheLifecycleStorage,
    _telemetry: RetiredCacheTelemetryStorage,
    _current_rows: Option<CacheTelemetryRows>,
}

/// A successful storage installation, still without native execution permission.
#[derive(Clone)]
pub(crate) struct InstalledManagerCatalog {
    manager: CacheResidencyManager,
    generation: u64,
    publication_controls: usize,
    _funding: Option<HostMetadataFunding>,
}
impl InstalledManagerCatalog {
    pub(crate) fn manager(&self) -> &CacheResidencyManager {
        &self.manager
    }
    pub(crate) fn publication_control_bytes(&self) -> usize {
        self.publication_controls
    }
    pub(crate) fn initial_generation(&self) -> u64 {
        self.generation
    }
}

/// Unaccepted destinations stay beside their original typed error until the
/// caller has left all manager loans; their actual host spending is cumulative.
#[derive(thiserror::Error)]
#[error("{cause}")]
pub(crate) struct CatalogInstallFailure {
    #[source]
    cause: CacheSourceFailure,
    _prepared: PreparedManagerCatalog,
}
impl CatalogInstallFailure {
    pub(crate) fn into_parts(self) -> (CacheSourceFailure, PreparedManagerCatalog) {
        (self.cause, self._prepared)
    }
}
impl std::fmt::Debug for CatalogInstallFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatalogInstallFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}

impl PreparedManagerCatalog {
    pub(crate) fn manager(&self) -> &CacheResidencyManager { &self.manager }
    pub(crate) fn publication_control_bytes(&self) -> usize { self.publication_controls }
    pub(crate) fn control_bytes(blocks: usize, tails: usize, current_layers: usize, reported_layers: usize) -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<RetiredCatalog>(),
            size_of::<InstalledManagerCatalog>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<Result<RetiredCatalog, CacheSourceError>>(),
            size_of::<Result<InstalledManagerCatalog, CatalogInstallFailure>>(),
            size_of::<CatalogInstallFailure>(),
            size_of::<CacheSourceFailure>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<(usize, usize, u64)>(),
            size_of::<(&CacheBlockSourceLoan<'_>, usize, usize, &WorkspaceContext)>(),
        ];
        frames.into_iter().try_fold(
            std::mem::size_of_val(&frames)
                .checked_add(
                    PreparedCacheTable::<CacheBlockId, CacheBlockRecord>::control_bytes(blocks)?,
                )?
                .checked_add(PreparedCacheLifecycle::control_bytes(blocks, tails)?)?
                .checked_add(PreparedCacheTelemetry::control_bytes(reported_layers)?)?
                .checked_add(PreparedCacheTable::<usize, CacheLayerResidencyStats>::control_bytes(current_layers)?)?,
            usize::checked_add,
        )
    }

    /// Moves these source-sized empty destinations to the privately constructed
    /// empty copy manager. The source loan supplies this method; neither table
    /// capacity nor an arbitrary caller ID establishes native copy authority.
    pub(super) fn for_empty_manager(mut self, manager: CacheResidencyManager)
        -> Result<Self, CatalogInstallFailure> {
        let checked = (|| {
            let state = manager.inner.state.try_lock().map_err(|cause| match cause {
                TryLockError::WouldBlock => CacheSourceError::Busy,
                TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            })?;
            if !manager.borrowed_storage_complete(&state) || !state.blocks.is_empty()
                || state.lifecycle.catalog_population() != (0,0) || state.generation != 0 {
                return Err(CacheSourceError::Identity);
            }
            Ok(())
        })();
        if let Err(cause) = checked {
            return Err(CatalogInstallFailure {
                cause: CacheSourceFailure { cause: CacheSourceFailureCause::Source(cause),
                    _funding: self._funding.clone() }, _prepared: self,
            });
        }
        self.manager = manager;
        self.generation = 0;
        self.population = (0,0);
        Ok(self)
    }

    /// Installs both catalogs only while this same source is still quiescent.
    /// Empty old backing/account owners retire after the source guard returns.
    pub(crate) fn install(mut self) -> Result<InstalledManagerCatalog, CatalogInstallFailure> {
        match self.install_inner() {
            Ok(retired) => {
                drop(retired);
                Ok(InstalledManagerCatalog {
                    manager: self.manager,
                    generation: self.generation,
                    publication_controls: self.publication_controls,
                    _funding: self._funding,
                })
            }
            Err(cause) => Err(CatalogInstallFailure {
                cause: CacheSourceFailure {
                    cause: CacheSourceFailureCause::Source(cause),
                    _funding: self._funding.clone(),
                },
                _prepared: self,
            }),
        }
    }

    fn install_inner(&mut self) -> Result<RetiredCatalog, CacheSourceError> {
        let mut state = match self.manager.inner.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(CacheSourceError::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(CacheSourceError::Poisoned),
        };
        if !self.manager.borrowed_storage_complete(&state) {
            return Err(CacheSourceError::PendingStorage);
        }
        if state.generation != self.generation
            || state.lifecycle.catalog_population() != self.population
            || state.blocks.len() != self.population.0
            || !state.telemetry.can_install(self.telemetry.as_ref().expect("one telemetry installation"))
        {
            return Err(CacheSourceError::Identity);
        }
        // Counts were validated before these destinations were allocated and
        // are checked again above. Neither installation can now refuse; each
        // moves canonical values without cloning native resources.
        let physical = state
            .blocks
            .install(self.physical.take().expect("one catalog installation"))
            .expect("unchanged source fits its prepared physical destination");
        let logical = state
            .lifecycle
            .install_storage(self.logical.take().expect("one lifecycle installation"))
            .expect("unchanged source fits its prepared logical destinations");
        let telemetry = state.telemetry.install_storage(self.telemetry.take().expect("one telemetry installation"))
            .expect("validated telemetry destination");
        let current_rows = std::mem::replace(&mut state.report_rows, self.current_rows.take());
        Ok(RetiredCatalog {
            _physical: physical,
            _logical: logical,
            _telemetry: telemetry,
            _current_rows: current_rows,
        })
    }
}

impl CacheBlockSourceLoan<'_> {
    fn catalog_shape<I>(&self, future_layers: I) -> Option<(usize, usize, usize)>
    where I: Iterator<Item = usize> + Clone {
        let layers = self.records.iter().map(|(id, _)| id.global_layer)
            .chain(self.lifecycle.tail_layers()).chain(future_layers);
        let current = layers.clone().enumerate().filter(|(index, layer)| {
            !layers.clone().take(*index).any(|earlier| earlier == *layer)
        }).count();
        let reported = self.telemetry.prepared_layer_count(layers.clone());
        let frames = std::mem::size_of_val(&layers).checked_add(size_of::<(usize, usize, I)>())?;
        Some((current, reported, frames))
    }
    pub(super) fn independent_catalog_control_bytes(&self) -> Option<usize> {
        let (current, reported, frames) = self.catalog_shape(std::iter::empty())?;
        let (blocks, tails) = self.catalog_population();
        PreparedManagerCatalog::control_bytes(blocks, tails, current, reported)?.checked_add(frames)
    }
    /// Prepares real destinations for the current manager plus the caller's
    /// exactly quoted maximum population. The maximum itself is not authority.
    pub(crate) fn prepare_catalog(
        &self,
        blocks: usize,
        tails: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedManagerCatalog, CacheSourceFailure> {
        self.prepare_catalog_for_layers(blocks, tails, std::iter::empty(), context)
    }

    pub(crate) fn prepare_catalog_for_layers<I>(
        &self, blocks: usize, tails: usize, future_layers: I, context: &WorkspaceContext,
    ) -> Result<PreparedManagerCatalog, CacheSourceFailure>
    where I: Iterator<Item = usize> + Clone,
    {
        let (current_count, reported_count, traversal) = self.catalog_shape(future_layers)
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        context.charge_metadata(traversal)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let physical = PreparedCacheTable::<CacheBlockId, CacheBlockRecord>::control_bytes(blocks)
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        let logical = PreparedCacheLifecycle::control_bytes(blocks, tails)
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        let telemetry = PreparedCacheTelemetry::control_bytes(reported_count)
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        let current = PreparedCacheTable::<usize, CacheLayerResidencyStats>::control_bytes(current_count)
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        let frames = PreparedManagerCatalog::control_bytes(blocks, tails, current_count, reported_count)
            .and_then(|n| n.checked_sub(physical))
            .and_then(|n| n.checked_sub(logical))
            .and_then(|n| n.checked_sub(telemetry))
            .and_then(|n| n.checked_sub(current))
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?;
        context
            .charge_metadata(frames)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let population = self.lifecycle.catalog_population();
        if population.0 != self.records.len() || blocks < population.0 || tails < population.1 {
            return Err(CacheSourceFailure::source(
                CacheSourceError::Identity,
                context,
            ));
        }
        Ok(PreparedManagerCatalog {
            manager: self.manager.clone(),
            generation: self.generation,
            publication_controls: self.publication_controls,
            population,
            physical: Some(
                PreparedCacheTable::prepare(blocks, context)
                    .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
            ),
            logical: Some(
                PreparedCacheLifecycle::prepare(blocks, tails, context)
                    .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
            ),
            telemetry: Some(PreparedCacheTelemetry::prepare(reported_count, context)
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))?),
            current_rows: Some({
                let prepared = PreparedCacheTable::prepare(current_count, context)
                    .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
                let mut rows = CacheTelemetryRows::new();
                drop(rows.install(prepared).expect("empty current-row destination"));
                rows
            }),
            _funding: context.metadata_funding(),
        })
    }
}

#[cfg(test)]
#[path = "catalog/tests.rs"]
mod tests;
