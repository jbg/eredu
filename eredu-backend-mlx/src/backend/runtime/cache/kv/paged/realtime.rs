//! Same append-only checkpoint, with paid exact catalog and tail owners.
use super::*;
use crate::backend::{error::Error,runtime::cache::residency::CacheSourceError};
use eredu_nn::workspace::{WorkspaceContext,HostMetadataFunding};
use eredu_runtime::{MutableCacheTail,working_memory::WorkingMemoryError};
use sha2::{Digest,Sha256};
use std::{hash::{Hash,Hasher},mem::{size_of,size_of_val}};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(in crate::backend::runtime::cache) struct RealtimePagedSource {
    pub identity:[u8;32],pub arrays:usize,pub blocks:usize,
    pub prepare_controls:usize,pub rollback_controls:usize,
}
/// The checkpoint keeps the frame's original Host account alive through discard.
#[derive(Debug)]
pub(super) struct OriginalRollback {
    tail:Option<MutableCacheTail>,tail_start:i64,controls:usize,
    funding:HostMetadataFunding,
}
struct SourceHash(Sha256);
impl Hasher for SourceHash {
    fn write(&mut self,bytes:&[u8]) {self.0.update(bytes)}
    fn finish(&self)->u64 {unreachable!("source hash is finalized as all 32 bytes")}
}
fn overflow()->Error {Error::PrefillControl(WorkingMemoryError::Overflow)}
fn mismatch()->Error {Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn source_error(cause:CacheSourceError)->Error {
    Error::StorageSource(eredu_core::BackendFailure::from_error(cause))
}
fn describe(source:&PagedKeyValueSource<'_>)->Result<RealtimePagedSource,Error> {
    let manager=source.manager_source();let mut hash=SourceHash(Sha256::new());
    manager.session_id().hash(&mut hash);manager.generation().hash(&mut hash);
    let tail=manager.tail();tail.map(|t|(t.bytes,t.end)).hash(&mut hash);
    let mut blocks=0usize;
    for block in manager.blocks() {
        block.id().hash(&mut hash);blocks=blocks.checked_add(1).ok_or_else(overflow)?;
    }
    let arrays=source.tail_arrays().map_or(0,|_|2);
    let rollback_controls=CacheResidencyManager::realtime_transaction_rollback_control_bytes()
        .and_then(|n|n.checked_add(manager.publication_control_bytes())).ok_or_else(overflow)?;
    Ok(RealtimePagedSource {identity:hash.0.finalize().into(),arrays,blocks,
        prepare_controls:prepare_controls().ok_or_else(overflow)?,rollback_controls})
}
fn inspect(source:PagedKeyValueSource<'_>)->Result<RealtimePagedSource,Error> {describe(&source)}
fn checkpoint_callback<'a>(ids:Option<&'a mut Vec<CacheBlockId>>,expected:Option<RealtimePagedSource>)
    ->impl for<'loan> FnOnce(PagedKeyValueSource<'loan>)->Result<Option<MutableCacheTail>,Error>+'a {
    move |source| {
        let expected=expected.ok_or_else(mismatch)?;
        if describe(&source)?!=expected {return Err(mismatch());}
        let ids=ids.ok_or_else(mismatch)?;
        for block in source.manager_source().blocks() {
            if ids.len()==ids.capacity(){return Err(mismatch());}
            ids.push(block.id().clone());
        }
        Ok(source.manager_source().tail())
    }
}
fn prepare_controls()->Option<usize> {
    let callback=checkpoint_callback(None,None);
    let frames=[size_of::<RealtimePagedSource>(),size_of::<OriginalRollback>(),size_of::<SourceHash>(),
        size_of::<Sha256>(),size_of::<Result<RealtimePagedSource,Error>>(),
        size_of::<Result<Option<MutableCacheTail>,Error>>(),size_of::<[Option<Array>;2]>(),
        size_of::<PagedKeyValueTransactionCheckpoint>(),size_of::<Result<(PagedKeyValueCache,PagedKeyValueTransactionCheckpoint),Error>>(),
        size_of::<(&PagedKeyValueCache,RealtimePagedSource,&HostMetadataFunding,&mut dyn FnMut(&Array)->Result<Array,Error>)>(),
        size_of::<(&PagedKeyValueSource<'_>,usize,usize)>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<CacheSourceError>()?,
        PagedKeyValueCache::workspace_source_control_bytes::<Option<MutableCacheTail>,_>(&callback)?,
    ];frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
impl RealtimePagedSource {
    pub(in crate::backend::runtime::cache) fn host_bytes(self)->Option<usize> {
        self.prepare_controls.checked_add(self.rollback_controls)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<CacheBlockId>(self.blocks)?)
    }
}
impl PagedKeyValueCache {
    pub(in crate::backend::runtime::cache) fn realtime_branch_inspection_controls()->Option<usize> {
        Self::workspace_source_control_bytes::<RealtimePagedSource,_>(&inspect)?
            .checked_add(size_of::<SourceHash>())?
            .checked_add(size_of::<Result<RealtimePagedSource,Error>>())
    }
    pub(in crate::backend::runtime::cache) fn inspect_realtime_branch(&self)->Result<RealtimePagedSource,Error> {
        // Same ordinary checkpoint constraint: discarded sliding history cannot
        // be restored by an append-only transaction receipt.
        if self.sliding_window.is_some() || self.retained_history.is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        self.inspect_copy_source(source_error,inspect)
    }
    pub(in crate::backend::runtime::cache) fn visit_realtime_tail_operands(&self,visit:&mut dyn FnMut(&Array,bool)) {
        if let Some(array)=&self.tail_keys {visit(array,false);}
        if let Some(array)=&self.tail_values {visit(array,false);}
    }
    pub(in crate::backend::runtime::cache) fn prepare_realtime_branch(&self,expected:RealtimePagedSource,
        funding:&HostMetadataFunding,alias:&mut dyn FnMut(&Array)->Result<Array,Error>)
        ->Result<(Self,PagedKeyValueTransactionCheckpoint),Error> {
        funding.reserve_metadata(expected.prepare_controls).map_err(Error::WorkspacePlanning)?;
        let mut ids=funding.metadata_vec(expected.blocks).map_err(Error::Neural)?;
        let tail=self.inspect_copy_source(source_error,checkpoint_callback(Some(&mut ids),Some(expected)))?;
        // Native aliases are constructed after the manager guard has returned.
        let keys=self.tail_keys.as_ref().map(&mut *alias).transpose()?;
        let values=self.tail_values.as_ref().map(&mut *alias).transpose()?;
        let checkpoint=PagedKeyValueTransactionCheckpoint {
            session_id:self.manager.session_id(),global_layer:self.global_layer,
            offset:self.offset,tail_bytes:tail.map_or(0,|value|value.bytes),block_ids:ids,
            original:Some(OriginalRollback {tail,tail_start:self.tail_start,
                controls:expected.rollback_controls,funding:funding.clone()}),
        };
        Ok((self.copied_with_tails(self.manager.clone(),keys,values,None),checkpoint))
    }
    pub(super) fn rollback_original_transaction(&self,checkpoint:&PagedKeyValueTransactionCheckpoint,
        original:&OriginalRollback)->Result<(),Exception> {
        self.manager.rollback_realtime_checkpoint(checkpoint.session_id,self.global_layer,checkpoint.global_layer,
            original.tail,original.tail_start,checkpoint.offset,self.offset,&checkpoint.block_ids,
            original.controls,&original.funding)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
