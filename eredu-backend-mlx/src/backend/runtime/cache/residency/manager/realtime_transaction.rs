//! Paid rollback of one sequential append-only realtime checkpoint.
use super::*;
#[path = "realtime_transaction/reporting.rs"]
mod prepared_reporting;
use eredu_nn::workspace::HostMetadataFunding;
use safemlx::error::Exception;
use std::{mem::{size_of,size_of_val},sync::TryLockError};

#[derive(Debug,thiserror::Error)]
enum Cause {
    #[error(transparent)] Source(#[from] CacheSourceError),
    #[error(transparent)] Residency(#[from] CacheResidencyError),
    #[error(transparent)] Metadata(#[from] eredu_nn::workspace::HostMetadataFundingError),
    #[error(transparent)] Neural(#[from] eredu_nn::Error),
}
#[derive(Debug,thiserror::Error)]
#[error("original cache transaction rollback: {cause}")]
struct Failure {
    #[source] cause:Cause,
    _funding:HostMetadataFunding,
}
fn source(cause:CacheSourceError)->Cause { Cause::Source(cause) }
impl CacheResidencyManager {
    /// Same lease-checked removal as ordinary discard and original append
    /// rollback. No I/O cancellation, eviction, new namespace or native work.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rollback_realtime_checkpoint(&self,session:u64,layer:usize,expected_layer:usize,
        tail:Option<MutableCacheTail>,tail_start:i64,offset:i64,current_offset:i64,original:&[CacheBlockId],
        controls:usize,funding:&HostMetadataFunding)->Result<(),Exception> {
        let result=(|| {
            funding.reserve_metadata(controls)?;
            if session!=self.session_id || layer!=expected_layer || offset<tail_start || tail_start<0 || current_offset<offset
                || tail.is_some_and(|value| value.end!=offset) {
                return Err(source(CacheSourceError::Identity));
            }
            let mut state=self.realtime_transaction_lock()?;
            // Complete preflight before the first mutation. Existing sealed
            // IDs survive; only this append-only layer's new frontier may go.
            for id in original {
                if id.session_id!=session || id.global_layer!=layer
                    || id.representation!=CacheRepresentation::KeyValue || !state.blocks.contains_key(id) {
                    return Err(source(CacheSourceError::Identity));
                }
            }
            let selected=eredu_runtime::CacheBlockSelection::new(layer,CacheRepresentation::KeyValue,0,i64::MAX,0);
            let mut removals=0usize;
            for (id,record) in state.blocks.iter().filter(|(id,_)|selected.includes(id)) {
                if original.contains(id){continue;}
                if id.session_id!=session || id.start<tail_start || id.end<=tail_start || id.end>current_offset || record.imported
                    || state.lifecycle.is_leased(id).map_err(CacheResidencyError::from)? {
                    return Err(source(CacheSourceError::Identity));
                }
                removals=removals.checked_add(1).ok_or_else(||source(CacheSourceError::Overflow))?;
            }
            let old_generation=state.generation;
            let generation=old_generation.checked_add(u64::try_from(removals).map_err(|_|source(CacheSourceError::Overflow))?)
                .ok_or_else(||source(CacheSourceError::Overflow))?;
            // The physical frontier still belongs to this completed branch.
            // An untouched empty branch legitimately has no tail slot.
            match state.lifecycle.tail(layer) {
                Some(value) if value.end==current_offset=>{},
                None if current_offset==offset && removals==0 && tail.is_none()=>{},
                _=>return Err(source(CacheSourceError::Identity)),
            }
            for _ in 0..removals {
                if state.generation!=old_generation {return Err(source(CacheSourceError::Identity));}
                let id=state.blocks.keys().rev().find(|id|selected.includes(id)&&!original.contains(id))
                    .cloned().ok_or_else(||source(CacheSourceError::Identity))?;
                let retired=lifecycle::take_unleased_record(&mut state,&id)?;
                drop(state);
                drop(retired);
                state=self.realtime_transaction_lock()?;
            }
            if state.generation!=old_generation || state.blocks.keys().any(|id|selected.includes(id)&&!original.contains(id)) {
                return Err(source(CacheSourceError::Identity));
            }
            state.lifecycle.restore_tail(layer,tail);
            state.generation=generation;
            reporting::update_report_totals_prepared(&mut state)?;
            Ok(())
        })();
        result.map_err(|cause|Exception::from_retained_source(Failure{cause,_funding:funding.clone()}))
    }
    fn realtime_transaction_lock(&self)->Result<MutexGuard<'_,CacheManagerState>,Cause> {
        let state=self.inner.state.try_lock().map_err(|cause|source(match cause {
            TryLockError::WouldBlock=>CacheSourceError::Busy,TryLockError::Poisoned(_)=>CacheSourceError::Poisoned,
        }))?;
        if !self.borrowed_storage_complete(&state){return Err(source(CacheSourceError::PendingStorage));}
        Ok(state)
    }
    pub(crate) fn realtime_transaction_rollback_control_bytes()->Option<usize> {
        let frames=[
            size_of::<(&Self,u64,usize,usize,Option<MutableCacheTail>,i64,i64,i64,&[CacheBlockId],usize,&HostMetadataFunding)>(),
            size_of::<Failure>(),Exception::retained_source_control_bytes::<Failure>()?,
            size_of::<MutexGuard<'_,CacheManagerState>>(),
            size_of::<Result<MutexGuard<'_,CacheManagerState>,Cause>>(),
            size_of::<TryLockError<MutexGuard<'_,CacheManagerState>>>(),
            size_of::<Option<CacheBlockRecord>>(),size_of::<CacheBlockId>(),
            size_of::<(usize,u64,u64,MutableCacheTail,eredu_runtime::CacheBlockSelection)>(),
            size_of::<Result<(),Cause>>(),size_of::<Result<(),Exception>>(),
            size_of::<Result<(),CacheResidencyError>>(),size_of::<Result<(),CacheLifecycleError>>(),
            eredu_runtime::cache::CacheRecordTable::<CacheBlockId,CacheBlockRecord>::mutation_control_bytes()?,
            CacheBlockLifecycle::prepared_mutation_control_bytes()?,
            eredu_core::HostMetadataFunding::reservation_control_bytes(),
        ];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}
