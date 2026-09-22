//! Source-specific checks around the single canonical publication worker.
use super::*;
use crate::backend::nn::workspace::{OrdinaryPagedAppend, OriginalPagedAppendClaim};
use eredu_runtime::cache::PagedAppendPlan;
use safemlx::{Array, error::Exception};

pub(crate) trait PreparedAppend {
    type Cause: From<CacheSourceError> + From<CacheResidencyError> + From<safemlx::StreamCopyCause>;
    fn error(&self, cause: impl Into<Self::Cause>) -> Exception;
    fn layer(&self) -> usize;
    fn plan(&self) -> PagedAppendPlan;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception>;
    fn validate_retained_storage(
        &self,
        id: &CacheBlockId,
        phase: CacheStoragePhase,
        file: Option<&CacheFileSource>,
    ) -> Result<(), Exception>;
    fn validate_rollback_tail(&self, bytes: u64, end: i64) -> Result<(), Exception>;
    fn validate_tail_update(
        &self,
        bytes: u64,
        end: i64,
        clearing: bool,
        restoring: bool,
    ) -> Result<(), Exception>;
    fn did_update_tail(&mut self, clearing: bool, restoring: bool);
    fn rebalance_host(
        &mut self,
        additional: u64,
        replacement: Option<u64>,
        required: Option<&CacheBlockId>,
        stream: &Stream,
    ) -> Result<(), Exception>;
    fn complete(&self, arrays: [&Array; 2], stream: &Stream) -> Result<(), Exception>;
    fn take_publication(
        &mut self,
        id: &CacheBlockId,
        arrays: &CacheBlockArrays,
    ) -> Result<(CacheBlockMetadata, bool), Exception>;
    fn published(&mut self, id: &CacheBlockId) -> Result<(), Exception>;
    fn publication_id(&self, index: usize) -> Option<&CacheBlockId>;
    fn published_count(&self) -> usize;
}
macro_rules! forward {
    ($ty:ty,$cause:ty) => {
        type Cause = $cause;
        fn error(&self, cause: impl Into<Self::Cause>) -> Exception {
            <$ty>::error(self, cause)
        }
        fn layer(&self) -> usize {
            <$ty>::layer(self)
        }
        fn plan(&self) -> PagedAppendPlan {
            <$ty>::plan(self)
        }
        fn validate_manager(&self, m: &CacheResidencyManager, g: u64) -> Result<(), Exception> {
            <$ty>::validate_manager(self, m, g)
        }
        fn validate_rollback_tail(&self, b: u64, e: i64) -> Result<(), Exception> {
            <$ty>::validate_rollback_tail(self, b, e)
        }
        fn validate_tail_update(&self, b: u64, e: i64, c: bool, r: bool) -> Result<(), Exception> {
            <$ty>::validate_tail_update(self, b, e, c, r)
        }
        fn did_update_tail(&mut self, c: bool, r: bool) {
            <$ty>::did_update_tail(self, c, r)
        }
        fn take_publication(
            &mut self,
            id: &CacheBlockId,
            a: &CacheBlockArrays,
        ) -> Result<(CacheBlockMetadata, bool), Exception> {
            <$ty>::take_publication(self, id, a)
        }
        fn published(&mut self, id: &CacheBlockId) -> Result<(), Exception> {
            <$ty>::published(self, id)
        }
        fn publication_id(&self, i: usize) -> Option<&CacheBlockId> {
            <$ty>::publication_id(self, i)
        }
        fn published_count(&self) -> usize {
            <$ty>::published_count(self)
        }
    };
}
impl PreparedAppend for OriginalPagedAppendClaim<'_> {
    forward!(
        OriginalPagedAppendClaim<'_>,
        crate::backend::nn::workspace::PagedMutationCause
    );
    fn validate_retained_storage(
        &self,
        id: &CacheBlockId,
        p: CacheStoragePhase,
        f: Option<&CacheFileSource>,
    ) -> Result<(), Exception> {
        self.validate_retained_storage(id, p, f)
    }
    fn rebalance_host(
        &mut self,
        a: u64,
        r: Option<u64>,
        id: Option<&CacheBlockId>,
        s: &Stream,
    ) -> Result<(), Exception> {
        self.rebalance_host(a, r, id, s)
    }
    fn complete(&self, arrays: [&Array; 2], stream: &Stream) -> Result<(), Exception> {
        crate::backend::runtime::cache::complete_values(arrays, stream)
    }
}
impl PreparedAppend for OrdinaryPagedAppend {
    forward!(
        OrdinaryPagedAppend,
        crate::backend::nn::workspace::OrdinaryPagedCause
    );
    fn validate_retained_storage(
        &self,
        id: &CacheBlockId,
        phase: CacheStoragePhase,
        file: Option<&CacheFileSource>,
    ) -> Result<(), Exception> {
        self.validate_retained_storage(id, phase, file)
    }
    fn rebalance_host(
        &mut self,
        additional: u64,
        tail: Option<u64>,
        required: Option<&CacheBlockId>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.rebalance_host(additional, tail, required, stream)
    }
    fn complete(&self, arrays: [&Array; 2], _: &Stream) -> Result<(), Exception> {
        evaluate_cache_arrays(arrays)
    }
}
