//! Source-derived reporting storage required by the same checked rollback.
use super::*;
use eredu_runtime::cache::{CacheTelemetryRows,PreparedCacheTable,PreparedCacheTelemetry,RetiredCacheTelemetryStorage};
use sha2::{Digest,Sha256};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) struct RealtimeReportingSource {
    session:u64,generation:u64,current:usize,reported:usize,controls:usize,
    identity:[u8;32],
}
impl RealtimeReportingSource {
    pub(crate) fn identity(self)->[u8;32] {self.identity}
    pub(crate) fn host_bytes(self)->Option<usize> {
        self.controls.checked_add(PreparedCacheTelemetry::control_bytes(self.reported)?)?
            .checked_add(PreparedCacheTable::<usize,CacheLayerResidencyStats>::control_bytes(self.current)?)
    }
}
fn describe<I>(manager:&CacheResidencyManager,state:&CacheManagerState,future:I)
    ->Option<RealtimeReportingSource> where I:Iterator<Item=usize>+Clone {
    let inspection=CacheResidencyManager::realtime_reporting_inspection_controls(&future)?;
    let layers=super::super::reporting::report_layers(state).chain(future);
    let current=layers.clone().enumerate().filter(|(index,layer)|
        !layers.clone().take(*index).any(|prior|prior==*layer)).count();
    let reported=state.telemetry.prepared_layer_count(layers.clone());
    let mut hash=Sha256::new();
    hash.update(manager.session_id.to_ne_bytes());hash.update(state.generation.to_ne_bytes());
    for layer in layers.clone() {hash.update(layer.to_ne_bytes());}
    for layer in state.telemetry.activity_layers() {hash.update(layer.to_ne_bytes());}
    let controls=inspection.checked_add(size_of_val(&layers))?
        .checked_add(size_of::<(PreparedCacheTelemetry,CacheTelemetryRows)>())?
        .checked_add(size_of::<(RetiredCacheTelemetryStorage,Option<CacheTelemetryRows>)>())?
        .checked_add(size_of::<Result<(),Cause>>())?
        .checked_add(size_of::<Result<(),Exception>>())?
        .checked_add(size_of::<Failure>())?
        .checked_add(Exception::retained_source_control_bytes::<Failure>()?)?;
    Some(RealtimeReportingSource {session:manager.session_id,generation:state.generation,
        current,reported,controls,identity:hash.finalize().into()})
}
impl CacheResidencyManager {
    pub(crate) fn realtime_reporting_inspection_controls<I>(_:&I)->Option<usize>
    where I:Iterator<Item=usize>+Clone {
        let parts=[size_of::<(&Self,I)>(),size_of::<I>(),size_of::<RealtimeReportingSource>(),
            size_of::<Sha256>(),size_of::<(usize,usize)>(),
            size_of::<MutexGuard<'_,CacheManagerState>>(),
            size_of::<TryLockError<MutexGuard<'_,CacheManagerState>>>(),
            size_of::<Result<RealtimeReportingSource,CacheSourceError>>(),
            size_of::<Result<MutexGuard<'_,CacheManagerState>,Cause>>(),
            super::super::reporting::report_query_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(crate) fn inspect_realtime_reporting<I>(&self,future:I)->Result<RealtimeReportingSource,CacheSourceError>
    where I:Iterator<Item=usize>+Clone {
        let state=self.inner.state.try_lock().map_err(|error|match error {
            TryLockError::WouldBlock=>CacheSourceError::Busy,
            TryLockError::Poisoned(_)=>CacheSourceError::Poisoned,
        })?;
        if !self.borrowed_storage_complete(&state){return Err(CacheSourceError::PendingStorage);}
        describe(self,&state,future).ok_or(CacheSourceError::Overflow)
    }
    /// Installs only reporting storage. Physical block catalogs, tail slots,
    /// native backing and append permission keep their independent sources.
    /// The canonical collector retains this same Host account until replacement
    /// or manager retirement, including when the attempted frame is discarded.
    pub(crate) fn prepare_realtime_reporting<I>(&self,expected:RealtimeReportingSource,future:I,
        funding:&HostMetadataFunding)->Result<(),Exception>
    where I:Iterator<Item=usize>+Clone {
        let result=(||->Result<(),Cause> {
            funding.reserve_metadata(expected.controls)?;
            let telemetry=PreparedCacheTelemetry::prepare_with_funding(expected.reported,funding)?;
            let prepared=PreparedCacheTable::prepare_with_funding(expected.current,funding)?;
            let mut rows=CacheTelemetryRows::new();
            drop(rows.install(prepared).expect("empty reporting destination"));
            let retired={
                let mut state=self.realtime_transaction_lock()?;
                if describe(self,&state,future)!=Some(expected) || !state.telemetry.can_install(&telemetry) {
                    return Err(source(CacheSourceError::Identity));
                }
                let prior=state.telemetry.install_storage(telemetry).expect("checked reporting source");
                let current=std::mem::replace(&mut state.report_rows,Some(rows));
                (prior,current)
            };
            drop(retired);
            Ok(())
        })();
        result.map_err(|cause|Exception::from_retained_source(Failure{cause,_funding:funding.clone()}))
    }
}
