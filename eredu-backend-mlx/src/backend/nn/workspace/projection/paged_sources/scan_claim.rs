//! Exact accepted source and one-use scan after its append has completed.
use super::*;
use super::{
    append_claim::{Checkout, CurrentRole, PagedMutationCause, current_role, failure},
    scan_program::PreparedPagedScan,
};
use crate::backend::runtime::cache::{
    kv::{BlockwiseAttentionAccumulator, KeyValueAttentionBlock, ProjectedPagedSource},
    residency::{
        CacheBlockMetadata, CacheBlockSourceLoan, CacheResidencyManager, CacheSourceError,
        InstalledManagerCatalog, PinnedCacheBlock,
    },
};
use crate::backend::submission_recovery::prefill::TransientRootsProjection;
use eredu_core::cache::CacheRankIdentity;
use eredu_nn::BlockwiseAttentionOptions;
use safemlx::{Array, Dtype, OriginalScopeObserver, error::Exception};

/// Actual current cache/query facts, never source or execution authority.
pub(crate) struct PagedScanInput<'a> {
    pub(crate) manager: &'a CacheResidencyManager,
    pub(crate) global_layer: usize,
    pub(crate) rank: Option<CacheRankIdentity>,
    pub(crate) queries: [i32; 4],
    pub(crate) dtype: Dtype,
    pub(crate) block_size: i32,
    pub(crate) offset: i64,
    pub(crate) tail_start: i64,
    pub(crate) window: Option<i32>,
    pub(crate) prefix: i32,
    pub(crate) options: BlockwiseAttentionOptions,
}
/// Private construction below retains the exact bank entry and native role.
/// A caller-supplied shape, ID or pool cannot construct this source proof.
pub(crate) struct OriginalPagedScanSource<'a> {
    pub(super) source: &'a ProjectedPagedSource,
    pub(super) installed: &'a InstalledManagerCatalog,
    pub(super) observer: OriginalScopeObserver,
    pub(super) context: &'a WorkspaceContext,
    pub(super) custody: eredu_runtime::working_memory::OriginalHostSourceCustody,
}
impl OriginalPagedScanSource<'_> {
    pub(crate) fn observer(&self) -> &OriginalScopeObserver {
        &self.observer
    }
    pub(crate) fn manager(&self) -> &CacheResidencyManager {
        self.source.manager()
    }
    pub(crate) fn host_source_custody(
        &self,
    ) -> &eredu_runtime::working_memory::OriginalHostSourceCustody {
        &self.custody
    }
    pub(crate) fn context(&self) -> &WorkspaceContext {
        self.context
    }
    pub(crate) fn funding(&self) -> Option<WorkspaceMetadataFunding> {
        self.context.metadata_funding()
    }
    pub(crate) fn publication_controls(&self) -> usize {
        self.installed.publication_control_bytes()
    }
    pub(crate) fn error(&self, cause: impl Into<PagedMutationCause>) -> Exception {
        failure(cause, &self.observer, self.context.metadata_funding())
    }
    pub(crate) fn layer(&self) -> usize {
        self.source.geometry().global_layer
    }
    pub(crate) fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(&self.observer)
            || !manager.same_catalog(self.manager())
            || generation != self.installed.initial_generation()
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn validate_loan(&self, loan: &CacheBlockSourceLoan<'_>) -> Result<(), Exception> {
        self.validate_manager(loan.manager(), loan.generation())
    }
    /// A prepared host transfer must belong to this exact quoted source,
    /// including its physical block ID/rank and construction trace. Shape
    /// agreement alone cannot substitute for the retained manager/source pin.
    pub(crate) fn validate_host_promotion(
        &self,
        id: &eredu_core::cache::CacheBlockId,
        descriptors: &[safemlx::HostTransferDescriptor<4>; 2],
        context: &WorkspaceContext,
    ) -> Result<(), Exception> {
        self.validate_retained_host_geometry(id, descriptors, context, true)
    }
    /// A separately completed read may replace the backing of an exact retained
    /// selected row. Its closed native witness authenticates the actual file and
    /// published Host handles; this shared check authenticates selection geometry.
    pub(crate) fn validate_retained_read_geometry(
        &self,
        id: &eredu_core::cache::CacheBlockId,
        descriptors: &[safemlx::HostTransferDescriptor<4>; 2],
        context: &WorkspaceContext,
    ) -> Result<(), Exception> {
        self.validate_retained_host_geometry(id, descriptors, context, false)
    }
    fn validate_retained_host_geometry(
        &self,
        id: &eredu_core::cache::CacheBlockId,
        descriptors: &[safemlx::HostTransferDescriptor<4>; 2],
        context: &WorkspaceContext,
        retained_host: bool,
    ) -> Result<(), Exception> {
        if !context.shares_trace(self.context) {
            return Err(self.error(CacheSourceError::Identity));
        }
        let block = self
            .source
            .geometry()
            .blocks
            .iter()
            .find(|block| &block.id == id)
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        if retained_host
            && !matches!(
                block.phase,
                eredu_runtime::CacheStoragePhase::HostUnbacked
                    | eredu_runtime::CacheStoragePhase::HostBacked
            )
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        for (actual, expected) in descriptors.iter().zip(&block.arrays) {
            if actual.shape() != expected.shape
                || actual.dtype() != expected.dtype
                || u64::try_from(actual.nbytes()).ok() != Some(expected.logical_bytes)
            {
                return Err(self.error(CacheSourceError::Geometry));
            }
        }
        Ok(())
    }
    pub(crate) fn validate_arrays(
        &self,
        arrays: [&Array; 2],
        length: i64,
    ) -> Result<(), Exception> {
        let [batch, heads, width] = self.source.append_geometry().dimensions;
        let length = i32::try_from(length).map_err(|_| self.error(CacheSourceError::Geometry))?;
        if length <= 0
            || arrays.iter().any(|array| {
                array.shape() != [batch, heads, length, width]
                    || CacheBlockMetadata::floating_dtype_bytes(array.dtype()).is_none()
            })
        {
            return Err(self.error(CacheSourceError::Geometry));
        }
        Ok(())
    }
}
/// A selected immutable row and its paid final destinations. Only the exact
/// claim below can form this borrow; an arbitrary block ID cannot construct it.
pub(crate) struct OriginalPagedBlockSource<'a, 'source> {
    pub(crate) proof: &'a OriginalPagedScanSource<'source>,
    id: &'a eredu_core::cache::CacheBlockId,
    pair: &'a mut super::scan_program::ScanSourcePair,
    pin: &'a mut Option<PinnedCacheBlock>,
}
impl<'a, 'source> OriginalPagedBlockSource<'a, 'source> {
    pub(super) fn new(
        proof: &'a OriginalPagedScanSource<'source>,
        id: &'a eredu_core::cache::CacheBlockId,
        pair: &'a mut super::scan_program::ScanSourcePair,
        pin: &'a mut Option<PinnedCacheBlock>,
    ) -> Self {
        Self {
            proof,
            id,
            pair,
            pin,
        }
    }

    pub(crate) fn id(&self) -> &eredu_core::cache::CacheBlockId {
        self.id
    }
    pub(crate) fn bind(&mut self, mut loan: CacheBlockSourceLoan<'_>) -> Result<(), Exception> {
        self.proof.validate_loan(&loan)?;
        if self.pin.is_some()
            || self.pair.values().is_some()
            || loan.blocks().count() != 1
            || !loan.blocks().any(|block| block.id() == self.id)
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        *self.pin = Some(
            loan.pin_prepared_block(self.id, self.proof.funding())
                .map_err(|cause| self.proof.error(cause))?,
        );
        let source = loan.blocks().next().expect("one exact source");
        let arrays = source
            .device()
            .ok_or_else(|| self.proof.error(CacheSourceError::PromotionRequired))?;
        self.proof
            .validate_arrays(arrays, self.id.end - self.id.start)?;
        // Initial blocks were admitted as actual completed state; appended
        // blocks were settled by that occurrence's exact sealing frontier.
        for array in arrays {
            self.proof.observer.validate_completed_array(array)?;
        }
        self.pair.fill(arrays, &self.proof.observer)
    }
}
/// Actual row/tail values borrowed only from the checked-out immutable source.
/// The private constructor keeps arbitrary arrays out of paged authorization.
pub(crate) struct OriginalPagedAttentionBlock<'a, 'source> {
    block: KeyValueAttentionBlock,
    source: &'a OriginalPagedScanSource<'source>,
    options: BlockwiseAttentionOptions,
    queries: [i32; 4],
    query_start: i64,
    window: Option<i32>,
    prefix: i64,
}
impl OriginalPagedAttentionBlock<'_, '_> {
    pub(crate) fn block(&self) -> &KeyValueAttentionBlock {
        &self.block
    }
    pub(crate) fn validate(
        &self,
        queries: [i32; 4],
        query_start: i64,
        window: Option<i32>,
        prefix: i64,
        causal: bool,
        options: BlockwiseAttentionOptions,
    ) -> Result<(), Exception> {
        let current = OriginalScopeObserver::require_current()?;
        if !current.same_scope(self.source.observer())
            || queries != self.queries
            || query_start != self.query_start
            || window != self.window
            || prefix != self.prefix
            || !causal
            || options.arithmetic != self.options.arithmetic
            || options.softcap.map(f32::to_bits) != self.options.softcap.map(f32::to_bits)
        {
            return Err(self.source.error(CacheSourceError::Identity));
        }
        Ok(())
    }
}
/// Completed exact-page retirement authority, privately issued by the accepted
/// scan or local-update claim. A manager cannot construct it from block IDs.
pub(crate) struct OriginalPagedDiscard<'a, 'source> {
    pub(super) proof: &'a OriginalPagedScanSource<'source>,
    pub(super) ids: &'a [eredu_core::cache::CacheBlockId],
    pub(super) frontier: Option<(i64, i64)>,
}
impl OriginalPagedDiscard<'_, '_> {
    pub(crate) fn proof(&self) -> &OriginalPagedScanSource<'_> {
        self.proof
    }
    pub(crate) fn ids(&self) -> &[eredu_core::cache::CacheBlockId] {
        self.ids
    }
    pub(crate) fn frontier(&self) -> Option<(i64, i64)> {
        self.frontier
    }
}

pub(crate) struct OriginalPagedScanClaim<'a> {
    pub(crate) proof: OriginalPagedScanSource<'a>,
    pub(super) scan: &'a mut PreparedPagedScan,
    host: Option<&'a mut super::host_program::PreparedPagedHostProgram>,
    bank: &'a PagedSourceBank,
    ordinal: usize,
    host_load: Option<usize>,
    roots: TransientRootsProjection,
    passes: usize,
    pass: Option<usize>,
    next: usize,
    tail_consumed: bool,
    retained_output: bool,
    options: BlockwiseAttentionOptions,
    queries: [i32; 4],
    output_dtype: Dtype,
}

/// Borrow of one actual scan row after any active demand lease has finished.
/// Only the claim below can count its canonical source/row pins. A caller's
/// allocation metadata or integer pin count cannot construct this permission.
pub(crate) struct OriginalPagedHostEviction<'a, 'source> {
    pub(super) source: &'a OriginalPagedScanSource<'source>,
    pub(super) id: &'a eredu_core::cache::CacheBlockId,
    pub(super) arrays: [&'a Array; 2],
    pub(super) owned_pins: usize,
    pub(super) roots: &'a TransientRootsProjection,
}
/// Exact request-bank owners of one Host row selected for a durable write.
/// Only the finite itinerary can count and construct this lexical permission.
/// Source pins keep old native resources alive; they are never demand leases.
pub(crate) struct OriginalPagedDiskWriteSource<'a, 'source> {
    pub(super) source: &'a OriginalPagedScanSource<'source>,
    pub(super) id: &'a eredu_core::cache::CacheBlockId,
    pub(super) owned_pins: usize,
}
impl OriginalPagedDiskWriteSource<'_, '_> {
    pub(crate) fn source(&self) -> &OriginalPagedScanSource<'_> {
        self.source
    }
    pub(crate) fn id(&self) -> &eredu_core::cache::CacheBlockId {
        self.id
    }
    pub(crate) fn validate(&self, loan: &CacheBlockSourceLoan<'_>) -> Result<usize, Exception> {
        self.source.validate_loan(loan)?;
        let (pins, demand) = loan
            .source_ownership(self.id)
            .map_err(|cause| self.source.error(cause))?;
        if demand || pins != self.owned_pins {
            return Err(self.source.error(CacheSourceError::Busy));
        }
        Ok(self.owned_pins)
    }
}
/// Exact source-bank return-to-Host claim. Only the ordered itinerary counts
/// the owning pins; the native promotion binds its own completed arrays.
pub(crate) struct OriginalPagedHostReturn<'a, 'source> {
    pub(super) source: &'a OriginalPagedScanSource<'source>,
    pub(super) id: &'a eredu_core::cache::CacheBlockId,
    pub(super) owned_pins: usize,
    pub(super) roots: &'a TransientRootsProjection,
}
impl<'source> OriginalPagedHostReturn<'_, 'source> {
    pub(crate) fn source(&self) -> &OriginalPagedScanSource<'source> {
        self.source
    }
    pub(crate) fn id(&self) -> &eredu_core::cache::CacheBlockId {
        self.id
    }
    pub(crate) fn bind<'a>(
        &'a self,
        arrays: [&'a Array; 2],
    ) -> OriginalPagedHostEviction<'a, 'source> {
        OriginalPagedHostEviction {
            source: self.source,
            id: self.id,
            arrays,
            owned_pins: self.owned_pins,
            roots: self.roots,
        }
    }
}
impl<'source> OriginalPagedHostEviction<'_, 'source> {
    pub(crate) fn source(&self) -> &OriginalPagedScanSource<'source> {
        self.source
    }
    pub(crate) fn id(&self) -> &eredu_core::cache::CacheBlockId {
        self.id
    }
    pub(crate) fn arrays(&self) -> [&Array; 2] {
        self.arrays
    }
    pub(crate) fn context(&self) -> &WorkspaceContext {
        self.source.context
    }
    pub(crate) fn roots(&self) -> &TransientRootsProjection {
        self.roots
    }
    pub(crate) fn validate_pin_counts(
        &self,
        lifecycle: &eredu_runtime::CacheBlockLifecycle,
    ) -> Result<(), Exception> {
        let fail = |cause| self.source.error(CacheSourceError::Lifecycle(cause));
        if lifecycle.is_device_leased(self.id).map_err(fail)?
            || lifecycle.source_pin_count(self.id).map_err(fail)? != self.owned_pins
        {
            return Err(self.source.error(CacheSourceError::Busy));
        }
        Ok(())
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<(), Exception>>(),
            std::mem::size_of::<Result<usize, eredu_runtime::CacheLifecycleError>>(),
            std::mem::size_of::<Result<bool, eredu_runtime::CacheLifecycleError>>(),
            std::mem::size_of::<std::slice::Iter<'_, super::scan_program::ScanSourceRow>>(),
            std::mem::size_of::<std::slice::Iter<'_, eredu_core::cache::CacheBlockId>>(),
            std::mem::size_of::<[&Array; 2]>(),
            std::mem::size_of::<(usize, bool)>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
impl OriginalPagedScanClaim<'_> {
    /// Execute an already prepared store for this exact selected native row.
    /// The enclosing source program retains the mover on success and failure;
    /// its new native roots join the existing final Model completion union.
    pub(crate) fn demote_prepared(
        &mut self,
        mover: crate::backend::runtime::cache::residency::PreparedCacheHostDemotion,
        stream: &safemlx::Stream,
    ) -> Result<(), Exception> {
        if self.scan.active_lease.is_some() || self.proof.source.has_host_promotion(mover.id()) {
            return Err(self.proof.error(CacheSourceError::Busy));
        }
        let row = self
            .scan
            .rows
            .iter_mut()
            .find(|row| &row.id == mover.id())
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?;
        if row.demotion.is_some() {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        row.demotion = Some(mover);
        let arrays = row
            .pair
            .values()
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?;
        let owned_pins = self
            .proof
            .source
            .owned_source_pin_count(&row.id)
            .checked_add(usize::from(row.pin.is_some()))
            .ok_or_else(|| self.proof.error(CacheSourceError::Overflow))?;
        let proof = OriginalPagedHostEviction {
            source: &self.proof,
            id: &row.id,
            arrays,
            owned_pins,
            roots: &self.roots,
        };
        row.demotion
            .as_mut()
            .expect("stored before submission")
            .run(&proof, stream)
    }
}
impl ProjectedPagedSources {
    pub(crate) fn with_current_scan(
        actual: PagedScanInput<'_>,
        run: impl FnOnce(&mut OriginalPagedScanClaim<'_>) -> Result<Array, Exception>,
    ) -> Result<Array, Exception> {
        let CurrentRole {
            source,
            observer,
            ordinal,
            roots,
        } = current_role()?;
        let fail = |cause| failure(cause, &observer, source.context.metadata_funding());
        actual
            .options
            .validate()
            .map_err(|cause| fail(CacheSourceError::Scan(cause.into())))?;
        let mut loan = source
            .catalogs
            .try_borrow_mut()
            .map_err(|_| fail(CacheSourceError::Busy))?;
        let catalogs = loan
            .as_mut()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let index = source
            .sources
            .iter()
            .position(|entry| {
                entry.manager().same_catalog(actual.manager)
                    && entry.geometry().global_layer == actual.global_layer
            })
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let retained = &source.sources[index];
        let geometry = retained.append_geometry();
        let [batch, heads, width] = geometry.dimensions;
        if actual.rank != retained.geometry().rank
            || actual.block_size != geometry.block_size
            || actual.window != geometry.window
            || actual.prefix != geometry.prefix_tokens
            || geometry.key_only
            || CacheBlockMetadata::floating_dtype_bytes(actual.dtype).is_none()
            || actual.queries[0] != batch
            || actual.queries[1] <= 0
            || heads <= 0
            || actual.queries[1] % heads != 0
            || actual.queries[2] <= 0
            || actual.queries[3] != width
        {
            return Err(fail(CacheSourceError::Geometry));
        }
        let installed = catalogs
            .installed
            .iter()
            .find(|entry| entry.manager().same_catalog(actual.manager))
            .ok_or_else(|| fail(CacheSourceError::Identity))?
            .clone();
        let program_index = catalogs
            .programs
            .iter()
            .position(|program| {
                program
                    .as_ref()
                    .is_some_and(|program| program.source == index && program.ordinal == ordinal)
            })
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let program = catalogs.programs[program_index]
            .as_ref()
            .expect("selected program");
        let scan = program
            .scan
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if !program.used
            || !program.completed
            || scan.used
            || scan.context_end != actual.offset
            || scan.tail_start != actual.tail_start
            || actual.offset.checked_sub(i64::from(actual.queries[2])) != Some(scan.query_start)
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let mut program = catalogs.programs[program_index]
            .take()
            .expect("one source checkout");
        program.scan.as_mut().expect("selected scan").used = true;
        source
            .roles
            .borrow_mut()
            .active
            .as_mut()
            .expect("exact active role")
            .append_active = true;
        drop(loan);
        let mut checkout = Checkout {
            source: Rc::clone(&source),
            index: program_index,
            program: Some(program),
            succeeded: false,
        };
        let custody = source
            .roles
            .try_borrow()
            .map_err(|_| fail(CacheSourceError::Busy))?
            .host_source_custody(&observer, ordinal)
            .map_err(fail)?;
        let proof = OriginalPagedScanSource {
            source: retained,
            installed: &installed,
            observer: observer.clone(),
            context: &source.context,
            custody,
        };
        let mut host = super::host_program::HostCheckout::take(&source, &proof)?;
        let mut claim = OriginalPagedScanClaim {
            proof,
            host: host.value.as_mut(),
            bank: &source,
            ordinal,
            host_load: None,
            scan: checkout
                .program
                .as_mut()
                .expect("checked-out program")
                .scan
                .as_mut()
                .expect("prepared scan"),
            roots,
            passes: actual.options.passes(),
            pass: None,
            next: 0,
            tail_consumed: false,
            retained_output: false,
            options: actual.options,
            queries: actual.queries,
            output_dtype: actual.dtype,
        };
        let result = run(&mut claim).and_then(|output| {
            claim.finish()?;
            Ok(output)
        });
        drop(claim);
        checkout.succeeded = result.is_ok();
        result
    }
}
impl OriginalPagedScanClaim<'_> {
    pub(crate) fn validate_readout(
        &self,
        mask: Option<&Array>,
        sinks: Option<&Array>,
    ) -> Result<(), Exception> {
        if self.scan.rows.is_empty() && self.scan.tail.is_none() {
            return Err(self.proof.error(CacheSourceError::MissingHistory));
        }
        if sinks.is_some_and(|value| {
            value.shape() != [self.queries[1]]
                || CacheBlockMetadata::floating_dtype_bytes(value.dtype()).is_none()
        }) {
            return Err(self.proof.error(CacheSourceError::Geometry));
        }
        if let Some(mask) = mask {
            if mask.ndim() == 0
                || mask.ndim() > 4
                || mask.dim(-1) <= 0
                || (mask.dtype() != Dtype::Bool
                    && CacheBlockMetadata::floating_dtype_bytes(mask.dtype()).is_none())
            {
                return Err(self.proof.error(CacheSourceError::Geometry));
            }
            let shape = [
                self.queries[0],
                self.queries[1],
                self.queries[2],
                mask.dim(-1),
            ];
            let broadcast = eredu_nn::workspace::WorkspaceBroadcastShape::new(mask.shape(), &shape)
                .map_err(|cause| self.proof.error(CacheSourceError::Broadcast(cause)))?;
            let first = self
                .scan
                .rows
                .first()
                .map_or(self.scan.tail_start, |row| row.id.start);
            if !broadcast.dimensions().eq(shape)
                || self.scan.context_end - i64::from(mask.dim(-1)) > first
            {
                return Err(self.proof.error(CacheSourceError::Geometry));
            }
        }
        for row in &self.scan.rows {
            self.validate_mask_coordinates(row.id.start, row.id.end)?;
        }
        if self.scan.tail.is_some() {
            self.validate_mask_coordinates(self.scan.tail_start, self.scan.context_end)?;
        }
        Ok(())
    }
    fn validate_mask_coordinates(&self, start: i64, end: i64) -> Result<(), Exception> {
        eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry::new(
            self.scan.query_start,
            self.queries[2],
            start,
            end,
            self.proof.source.append_geometry().window,
            i64::from(self.proof.source.append_geometry().prefix_tokens),
        )
        .map_err(|cause| self.proof.error(CacheSourceError::AbsoluteMask(cause)))?;
        Ok(())
    }
    pub(crate) fn bind_sources(&mut self) -> Result<(), Exception> {
        for row in &mut self.scan.rows {
            if self.host.is_some() {
                continue;
            }
            // Initial Host rows are promoted on their first actual selected
            // pass, after Begin, as in the shared ordinary scan traversal.
            if self.proof.source.has_host_promotion(&row.id) {
                continue;
            }
            let mut target = OriginalPagedBlockSource {
                proof: &self.proof,
                id: &row.id,
                pair: &mut row.pair,
                pin: &mut row.pin,
            };
            self.proof.manager().bind_original_scan_block(&mut target)?;
        }
        if self
            .scan
            .tail
            .as_ref()
            .is_some_and(|pair| pair.values().is_none())
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn begin_pass(&mut self, pass: usize) -> Result<(), Exception> {
        let previous_complete = self.next == self.scan.rows.len()
            && self.tail_consumed
            && self.scan.active_lease.is_none();
        if pass >= self.passes
            || (pass == 0 && self.pass.is_some())
            || (pass != 0 && (self.pass != Some(pass - 1) || !previous_complete))
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        self.pass = Some(pass);
        self.next = 0;
        self.tail_consumed = false;
        Ok(())
    }
    pub(crate) fn acquire_next(
        &mut self,
        stream: &safemlx::Stream,
    ) -> Result<Option<usize>, Exception> {
        if self.pass.is_none() || self.tail_consumed || self.scan.active_lease.is_some() {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        if self.next == self.scan.rows.len() {
            return Ok(None);
        }
        let row = &mut self.scan.rows[self.next];
        if let Some(host) = self.host.as_deref_mut() {
            let (load, lease) = host.acquire(
                self.bank,
                &self.proof,
                self.ordinal,
                self.pass.expect("checked pass"),
                &row.id,
                &self.roots,
                stream,
            )?;
            self.host_load = Some(load);
            self.scan.active_lease = Some(lease);
            let index = self.next;
            self.next += 1;
            return Ok(Some(index));
        }
        if row.pin.is_none() {
            self.proof
                .source
                .promote_host(&row.id, &self.proof, &self.roots, stream)?;
            let mut target = OriginalPagedBlockSource {
                proof: &self.proof,
                id: &row.id,
                pair: &mut row.pair,
                pin: &mut row.pin,
            };
            self.proof.manager().bind_original_scan_block(&mut target)?;
        }
        let pin = row
            .pin
            .as_ref()
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?;
        self.scan.active_lease = Some(pin.acquire().map_err(|cause| self.proof.error(cause))?);
        let index = self.next;
        self.next += 1;
        Ok(Some(index))
    }
    pub(crate) fn block(&self, index: usize) -> Result<(i64, i64, [&Array; 2]), Exception> {
        if self.next
            != index
                .checked_add(1)
                .ok_or_else(|| self.proof.error(CacheSourceError::Overflow))?
            || self.scan.active_lease.is_none()
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        let row = &self.scan.rows[index];
        Ok((
            row.id.start,
            row.id.end,
            match self.host.as_deref() {
                Some(host) => host.values(
                    self.host_load
                        .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?,
                ),
                None => row.pair.values(),
            }
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?,
        ))
    }
    pub(crate) fn release_completed_block(&mut self) -> Result<(), Exception> {
        if !self
            .scan
            .accumulator
            .as_ref()
            .is_some_and(BlockwiseAttentionAccumulator::has_completed_recurrence)
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        let lease = self
            .scan
            .active_lease
            .take()
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?;
        drop(lease);
        Ok(())
    }
    pub(crate) fn tail(&mut self) -> Result<Option<(i64, i64, [&Array; 2])>, Exception> {
        if self.pass.is_none()
            || self.next != self.scan.rows.len()
            || self.tail_consumed
            || self.scan.active_lease.is_some()
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        self.tail_consumed = true;
        self.scan
            .tail
            .as_ref()
            .map(|pair| {
                Ok((
                    self.scan.tail_start,
                    self.scan.context_end,
                    pair.values()
                        .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?,
                ))
            })
            .transpose()
    }
    pub(crate) fn accumulate_block(
        &mut self,
        index: usize,
        stream: &safemlx::Stream,
    ) -> Result<(i64, i64, u64), Exception> {
        let (start, end, [keys, values]) = self.block(index)?;
        let block = KeyValueAttentionBlock::unleased(
            start,
            end,
            keys.try_clone_handle()?,
            values.try_clone_handle()?,
        );
        let bytes = block.bytes;
        self.accumulate(block, stream)?;
        Ok((start, end, bytes))
    }
    pub(crate) fn accumulate_tail(
        &mut self,
        stream: &safemlx::Stream,
    ) -> Result<Option<(i64, i64, u64)>, Exception> {
        let block = self
            .tail()?
            .map(|(start, end, [keys, values])| {
                Ok::<_, Exception>(KeyValueAttentionBlock::unleased(
                    start,
                    end,
                    keys.try_clone_handle()?,
                    values.try_clone_handle()?,
                ))
            })
            .transpose()?;
        let Some(block) = block else {
            return Ok(None);
        };
        let facts = (block.start, block.end, block.bytes);
        self.accumulate(block, stream)?;
        Ok(Some(facts))
    }
    fn accumulate(
        &mut self,
        block: KeyValueAttentionBlock,
        stream: &safemlx::Stream,
    ) -> Result<(), Exception> {
        let source = OriginalPagedAttentionBlock {
            block,
            source: &self.proof,
            options: self.options,
            queries: self.queries,
            query_start: self.scan.query_start,
            window: self.proof.source.append_geometry().window,
            prefix: i64::from(self.proof.source.append_geometry().prefix_tokens),
        };
        self.scan
            .accumulator
            .as_mut()
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?
            .accumulate_original_paged(&source, stream)
    }
    pub(crate) fn retain_output(&mut self, output: Array) -> Result<Array, Exception> {
        if self.retained_output {
            self.scan.failed_root = Some(output);
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        if output.shape() != self.queries || output.dtype() != self.output_dtype {
            self.scan.failed_root = Some(output);
            return Err(self.proof.error(CacheSourceError::Geometry));
        }
        if let Err(cause) = self.roots.append(&output) {
            self.scan.failed_root = Some(output);
            return Err(self.proof.error(cause));
        }
        self.retained_output = true;
        Ok(output)
    }
    fn finish(&mut self) -> Result<(), Exception> {
        if self.pass != self.passes.checked_sub(1)
            || self.next != self.scan.rows.len()
            || !self.tail_consumed
            || self.scan.active_lease.is_some()
            || self.scan.accumulator.is_some()
            || !self.retained_output
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        self.scan.completed = true;
        Ok(())
    }
    /// The returned output is already a retained completion root. Release only
    /// this scan's completed aliases/pins, then request exact logical discard.
    pub(crate) fn discard_completed_output(&mut self, output: &Array) -> Result<(), Exception> {
        if self.proof.source.append_geometry().window.is_none() {
            return Ok(());
        }
        self.proof.observer.validate_completed_array(output)?;
        self.finish()?;
        self.scan.retire_settled();
        let discard = OriginalPagedDiscard {
            proof: &self.proof,
            ids: self.discard_ids(),
            frontier: self.discard_frontier()?,
        };
        self.proof.manager().discard_original(&discard)
    }
    pub(crate) fn discard_frontier(&self) -> Result<Option<(i64, i64)>, Exception> {
        if !self.scan.completed
            || self.scan.accumulator.is_some()
            || self.scan.active_lease.is_some()
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        let geometry = self.proof.source.append_geometry();
        Ok(geometry.window.map(|window| {
            let prefix = i64::from(geometry.prefix_tokens);
            (
                (self.scan.context_end - i64::from(window)).max(prefix),
                prefix,
            )
        }))
    }
    pub(crate) fn discard_ids(&self) -> &[eredu_core::cache::CacheBlockId] {
        &self.scan.discard_ids
    }
    pub(crate) fn accumulator(&mut self) -> &mut Option<BlockwiseAttentionAccumulator> {
        &mut self.scan.accumulator
    }
}

pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    let frames = [
        size_of::<OriginalPagedScanClaim<'_>>(),
        size_of::<OriginalPagedDiscard<'_, '_>>(),
        size_of::<OriginalPagedScanSource<'_>>(),
        size_of::<OriginalPagedBlockSource<'_, '_>>(),
        size_of::<OriginalPagedAttentionBlock<'_, '_>>(),
        size_of::<PagedScanInput<'_>>(),
        size_of::<CurrentRole>(),
        size_of::<Checkout>(),
        size_of::<InstalledManagerCatalog>(),
        OriginalScopeObserver::control_bytes()?,
        size_of::<TransientRootsProjection>(),
        size_of::<(&Rc<PagedSourceBank>, &OriginalScopeObserver)>(),
        size_of::<std::cell::RefMut<'_, Option<super::catalogs::PreparedPagedCatalogs>>>(),
        size_of::<std::cell::Ref<'_, super::roles::RoleState>>(),
        size_of::<std::slice::IterMut<'_, super::scan_program::ScanSourceRow>>(),
        size_of::<std::slice::Iter<'_, Option<super::programs::PagedAppendProgram>>>(),
        size_of::<BlockwiseAttentionOptions>(),
        size_of::<KeyValueAttentionBlock>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<Option<usize>, Exception>>(),
        size_of::<Result<(i64, i64, u64), Exception>>(),
        size_of::<Result<Option<(i64, i64, u64)>, Exception>>(),
        size_of::<Result<(i64, i64, [&Array; 2]), Exception>>(),
        size_of::<[i32; 4]>(),
        size_of::<(Dtype, Option<u64>)>(),
        size_of::<[&Array; 2]>(),
        size_of::<(usize, usize, i64, bool)>(),
        size_of::<(Option<&Array>, Option<&Array>)>(),
        size_of::<eredu_nn::workspace::WorkspaceBroadcastShape<'_>>(),
        size_of::<
            Result<
                eredu_nn::workspace::WorkspaceBroadcastShape<'_>,
                eredu_nn::workspace::WorkspaceShapeError,
            >,
        >(),
        size_of::<
            Result<
                eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry,
                eredu_nn::operation_geometry::AbsoluteAttentionMaskError,
            >,
        >(),
        size_of::<std::slice::Iter<'_, super::scan_program::ScanSourceRow>>(),
        size_of::<Option<(i64, i64)>>(),
        super::append_claim::failure_control_bytes()?,
        ProjectedPagedSource::promotion_access_control_bytes()?,
        size_of::<(&mut OriginalPagedScanClaim<'_>, &safemlx::Stream)>(),
        size_of::<(&mut OriginalPagedScanClaim<'_>, &Array)>(),
        size_of::<Result<Option<(i64, i64)>, Exception>>(),
        size_of::<std::slice::Iter<'_, eredu_core::cache::CacheBlockId>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
