//! Backend-neutral ownership for live cache blocks and mutable tails.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::table::{CacheRecordTable, CacheTableCapacityError, PreparedCacheTable};
use eredu_core::{cache::CacheBlockId, residency::CacheEvictionPolicy};
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};

/// Device-resident mutable state that has not yet become an immutable block.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct MutableCacheTail {
    /// Concrete bytes owned by the backend on the execution device.
    pub bytes: u64,
    /// Exclusive logical token frontier represented by the tail.
    pub end: i64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct BlockLifecycle {
    leases: usize,
    source_pins: usize,
    access_count: u64,
    last_access: u64,
    protected_prefix: bool,
}

/// Canonical logical ownership state for one backend cache session.
///
/// Backends keep concrete arrays, buffers, files, and completion objects in a
/// separate storage map keyed by [`CacheBlockId`]. This catalog is the sole
/// owner of leases, access order, protected-prefix status, and mutable tails.
#[derive(Debug, Default)]
pub struct CacheBlockLifecycle {
    access_clock: u64,
    blocks: CacheRecordTable<CacheBlockId, BlockLifecycle>,
    tails: CacheRecordTable<usize, MutableCacheTail>,
}

/// Paid empty destinations for the existing logical block and tail catalogs.
/// This contains no lease or native permission; installation preserves the
/// actual source identities already in the lifecycle.
#[derive(Debug)]
pub struct PreparedCacheLifecycle {
    blocks: PreparedCacheTable<CacheBlockId, BlockLifecycle>,
    tails: PreparedCacheTable<usize, MutableCacheTail>,
}
/// Empty predecessor storage returned after an atomic installation. Keep this
/// owner until any backend manager loan has ended; dropping it then releases
/// the old inventories before their host accounts.
#[derive(Debug)]
pub struct RetiredCacheLifecycleStorage {
    _blocks: CacheRecordTable<CacheBlockId, BlockLifecycle>,
    _tails: CacheRecordTable<usize, MutableCacheTail>,
}
impl PreparedCacheLifecycle {
    /// Exact storage and constructor controls for both canonical inventories.
    pub fn control_bytes(blocks: usize, tails: usize) -> Option<usize> {
        Self::fixed_bytes()?
            .checked_add(
                PreparedCacheTable::<CacheBlockId, BlockLifecycle>::control_bytes(blocks)?,
            )?
            .checked_add(PreparedCacheTable::<usize, MutableCacheTail>::control_bytes(tails)?)
    }
    fn fixed_bytes() -> Option<usize> {
        let frames = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<Result<RetiredCacheLifecycleStorage, Self>>(),
            std::mem::size_of::<CacheBlockLifecycle>(),
            std::mem::size_of::<RetiredCacheLifecycleStorage>(),
            std::mem::size_of::<(usize, usize)>(),
            std::mem::size_of::<Result<(), CacheLifecycleError>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Reserves both inventories on the same metadata account before installing
    /// either. A partial constructor failure retains its original typed cause.
    pub fn prepare(blocks: usize, tails: usize, context: &WorkspaceContext) -> Result<Self, Error> {
        context.charge_metadata(Self::fixed_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        Ok(Self {
            blocks: PreparedCacheTable::prepare(blocks, context)?,
            tails: PreparedCacheTable::prepare(tails, context)?,
        })
    }
}

impl CacheBlockLifecycle {
    /// Creates an empty cache-session lifecycle.
    pub const fn new() -> Self {
        Self {
            access_clock: 0,
            blocks: CacheRecordTable::new(),
            tails: CacheRecordTable::new(),
        }
    }

    /// Fixed controls of one existing logical mutation/validation worker.
    /// The native call bank must price actual invocations separately from the
    /// storage constructor; these facts supply no spending or source authority.
    pub fn prepared_mutation_control_bytes() -> Option<usize> {
        let frames = [
            std::mem::size_of::<&mut Self>(),
            std::mem::size_of::<CacheBlockId>(),
            std::mem::size_of::<Option<(CacheBlockId, bool)>>(),
            std::mem::size_of::<MutableCacheTail>(),
            std::mem::size_of::<Option<MutableCacheTail>>(),
            std::mem::size_of::<&[(CacheBlockId, usize)]>(),
            std::mem::size_of::<std::slice::Iter<'_, (CacheBlockId, usize)>>(),
            std::mem::size_of::<std::iter::Enumerate<std::slice::Iter<'_, (CacheBlockId, usize)>>>(
            ),
            std::mem::size_of::<CacheLifecycleError>(),
            std::mem::size_of::<Result<(), CacheLifecycleError>>(),
            std::mem::size_of::<Result<Option<MutableCacheTail>, CacheLifecycleError>>(),
            std::mem::size_of::<Result<u64, CacheLifecycleError>>(),
            std::mem::size_of::<[usize; 5]>(),
            std::mem::size_of::<[u64; 2]>(),
            std::mem::size_of::<[bool; 2]>(),
        ];
        frames.into_iter().try_fold(
            std::mem::size_of_val(&frames)
                .checked_add(
                    CacheRecordTable::<CacheBlockId, BlockLifecycle>::mutation_control_bytes()?,
                )?
                .checked_add(
                    CacheRecordTable::<usize, MutableCacheTail>::mutation_control_bytes()?,
                )?,
            usize::checked_add,
        )
    }

    /// Exact current populations of the canonical block and tail inventories.
    /// This descriptive query neither reserves slots nor changes leases.
    pub fn catalog_population(&self) -> (usize, usize) {
        (self.blocks.len(), self.tails.len())
    }

    /// Registers a newly materialized immutable block.
    pub fn insert(
        &mut self,
        id: CacheBlockId,
        protected_prefix: bool,
    ) -> Result<(), CacheLifecycleError> {
        self.insert_impl(id, protected_prefix, false)
    }

    /// Inserts into already prepared canonical storage without allocating.
    pub fn insert_prepared(
        &mut self,
        id: CacheBlockId,
        protected_prefix: bool,
    ) -> Result<(), CacheLifecycleError> {
        self.insert_impl(id, protected_prefix, true)
    }

    fn insert_impl(
        &mut self,
        id: CacheBlockId,
        protected_prefix: bool,
        prepared: bool,
    ) -> Result<(), CacheLifecycleError> {
        if self.blocks.contains_key(&id) {
            return Err(CacheLifecycleError::DuplicateBlock(id));
        }
        if prepared {
            self.blocks.validate_prepared_population(
                self.blocks
                    .len()
                    .checked_add(1)
                    .ok_or(CacheTableCapacityError::Exhausted)?,
            )?;
        }
        let last_access = self.tick()?;
        self.blocks.insert(
            id,
            BlockLifecycle {
                leases: 0,
                source_pins: 0,
                access_count: 0,
                last_access,
                protected_prefix,
            },
        );
        Ok(())
    }

    /// Removes an unleased block from logical ownership.
    pub fn remove(&mut self, id: &CacheBlockId) -> Result<(), CacheLifecycleError> {
        if self.is_leased(id)? {
            return Err(CacheLifecycleError::BlockLeased(id.clone()));
        }
        self.blocks.remove(id);
        Ok(())
    }

    /// Atomically replaces a set of blocks and updates one mutable tail.
    ///
    /// The expected lease count makes a caller-owned truncation lease explicit:
    /// validation completes before any logical state is changed.
    pub fn replace(
        &mut self,
        removals: &[(CacheBlockId, usize)],
        replacement: Option<(CacheBlockId, bool)>,
        tail_layer: usize,
        tail: MutableCacheTail,
    ) -> Result<(), CacheLifecycleError> {
        self.replace_impl(removals, replacement, tail_layer, tail, false)
    }

    /// Applies the same lease-checked replacement using only prepared slots.
    pub fn replace_prepared(
        &mut self,
        removals: &[(CacheBlockId, usize)],
        replacement: Option<(CacheBlockId, bool)>,
        tail_layer: usize,
        tail: MutableCacheTail,
    ) -> Result<(), CacheLifecycleError> {
        self.replace_impl(removals, replacement, tail_layer, tail, true)
    }

    fn replace_impl(
        &mut self,
        removals: &[(CacheBlockId, usize)],
        replacement: Option<(CacheBlockId, bool)>,
        tail_layer: usize,
        tail: MutableCacheTail,
        prepared: bool,
    ) -> Result<(), CacheLifecycleError> {
        for (index, (id, _)) in removals.iter().enumerate() {
            if removals[..index].iter().any(|(prior, _)| prior == id) {
                return Err(CacheLifecycleError::DuplicateRemoval);
            }
        }
        for (id, leases) in removals {
            self.require_lease_count(id, *leases)?;
        }
        if let Some((id, _)) = &replacement {
            if self.blocks.contains_key(id) && !removals.iter().any(|(removed, _)| removed == id) {
                return Err(CacheLifecycleError::DuplicateBlock(id.clone()));
            }
        }
        if prepared {
            let blocks = self
                .blocks
                .len()
                .checked_sub(removals.len())
                .and_then(|n| n.checked_add(usize::from(replacement.is_some())))
                .ok_or(CacheTableCapacityError::Exhausted)?;
            self.blocks.validate_prepared_population(blocks)?;
            self.validate_tail_slot(tail_layer)?;
        }
        let replacement_access = if replacement.is_some() {
            Some(self.tick()?)
        } else {
            None
        };

        for (id, _) in removals {
            self.blocks.remove(id);
        }
        if let Some((id, protected_prefix)) = replacement {
            self.blocks.insert(
                id,
                BlockLifecycle {
                    leases: 0,
                    source_pins: 0,
                    access_count: 0,
                    last_access: replacement_access.expect("replacement access was allocated"),
                    protected_prefix,
                },
            );
        }
        self.tails.insert(tail_layer, tail);
        Ok(())
    }

    /// Acquires one exact logical lease and records demand access.
    pub fn acquire(&mut self, id: &CacheBlockId) -> Result<(), CacheLifecycleError> {
        let leases = self.next_lease_count(id)?;
        let clock = self.tick()?;
        let block = self
            .blocks
            .get_mut(id)
            .expect("cache block was validated before advancing the access clock");
        block.leases = leases;
        block.access_count = block.access_count.saturating_add(1);
        block.last_access = clock;
        Ok(())
    }

    /// Retains immutable block identity without recording Device demand. Both
    /// kinds of ownership block removal/reset; a source pin alone allows an
    /// authenticated tier move that preserves the same immutable contents.
    pub fn pin_source(&mut self, id: &CacheBlockId) -> Result<(), CacheLifecycleError> {
        self.lease_count(id)?
            .checked_add(1)
            .ok_or_else(|| CacheLifecycleError::LeaseOverflow(id.clone()))?;
        self.blocks
            .get_mut(id)
            .expect("source was validated")
            .source_pins += 1;
        Ok(())
    }

    fn next_lease_count(&self, id: &CacheBlockId) -> Result<usize, CacheLifecycleError> {
        self.lease_count(id)?
            .checked_add(1)
            .ok_or_else(|| CacheLifecycleError::LeaseOverflow(id.clone()))?;
        Ok(self.blocks[id].leases + 1)
    }

    /// Releases only an exact source pin. Demand retirement cannot consume it.
    pub fn release_source(&mut self, id: &CacheBlockId) -> Result<(), CacheLifecycleError> {
        let block = self
            .blocks
            .get_mut(id)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
        block.source_pins = block
            .source_pins
            .checked_sub(1)
            .ok_or_else(|| CacheLifecycleError::LeaseUnderflow(id.clone()))?;
        Ok(())
    }

    /// Number of immutable source owners, excluding active Device demand.
    pub fn source_pin_count(&self, id: &CacheBlockId) -> Result<usize, CacheLifecycleError> {
        self.blocks
            .get(id)
            .map(|block| block.source_pins)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))
    }

    /// Whether an active demand lease forbids changing Device residency.
    /// Source pins continue to forbid logical removal and content replacement.
    pub fn is_device_leased(&self, id: &CacheBlockId) -> Result<bool, CacheLifecycleError> {
        self.blocks
            .get(id)
            .map(|block| block.leases != 0)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))
    }

    /// Releases one exact logical lease.
    pub fn release(&mut self, id: &CacheBlockId) -> Result<(), CacheLifecycleError> {
        let block = self
            .blocks
            .get_mut(id)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?;
        block.leases = block
            .leases
            .checked_sub(1)
            .ok_or_else(|| CacheLifecycleError::LeaseUnderflow(id.clone()))?;
        Ok(())
    }

    /// Returns the exact lease count for a block.
    pub fn lease_count(&self, id: &CacheBlockId) -> Result<usize, CacheLifecycleError> {
        self.blocks
            .get(id)
            .map(|block| block.leases + block.source_pins)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))
    }

    /// Returns whether a block currently has any logical owner.
    pub fn is_leased(&self, id: &CacheBlockId) -> Result<bool, CacheLifecycleError> {
        self.lease_count(id).map(|leases| leases != 0)
    }

    /// Returns the first leased block in stable identity order.
    pub fn first_leased(&self) -> Option<&CacheBlockId> {
        self.blocks
            .iter()
            .find_map(|(id, block)| (block.leases != 0 || block.source_pins != 0).then_some(id))
    }

    /// Returns whether the block is protected as an immutable prefix.
    pub fn is_protected_prefix(&self, id: &CacheBlockId) -> Result<bool, CacheLifecycleError> {
        self.blocks
            .get(id)
            .map(|block| block.protected_prefix)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))
    }

    /// Selects one eviction victim from backend-supplied physically eligible IDs.
    pub fn eviction_candidate(
        &self,
        candidates: impl IntoIterator<Item = CacheBlockId>,
        required: Option<&CacheBlockId>,
        recent_per_layer: usize,
        policy: CacheEvictionPolicy,
    ) -> Result<Option<CacheBlockId>, CacheLifecycleError> {
        let candidates = candidates.into_iter().collect::<BTreeSet<_>>();
        self.eviction_candidate_borrowed(candidates.iter(), required, recent_per_layer, policy)
            .map(|value| value.cloned())
    }

    /// Same ordinary eviction predicate over canonical borrowed candidate IDs.
    /// Input must be sorted; repeated adjacent identities are considered once.
    /// No map, ID copy or destination allocation occurs on a successful query.
    pub fn eviction_candidate_borrowed<'a, I>(
        &self,
        candidates: I,
        required: Option<&CacheBlockId>,
        recent_per_layer: usize,
        policy: CacheEvictionPolicy,
    ) -> Result<Option<&'a CacheBlockId>, CacheLifecycleError>
    where
        I: Iterator<Item = &'a CacheBlockId> + Clone,
    {
        self.select_eviction(candidates, required, recent_per_layer, policy, false)
    }

    /// Device-to-Host selection using the same ordering/protection policy.
    /// Immutable source pins permit a tier move but active demand never does.
    /// The backend must still authenticate every source pin, retain replaced
    /// physical occupancy and complete the actual transfer. Ordinary eviction
    /// continues to use the all-owner exclusion above.
    pub fn device_eviction_candidate_borrowed<'a, I>(
        &self,
        candidates: I,
        required: Option<&CacheBlockId>,
        recent_per_layer: usize,
        policy: CacheEvictionPolicy,
    ) -> Result<Option<&'a CacheBlockId>, CacheLifecycleError>
    where
        I: Iterator<Item = &'a CacheBlockId> + Clone,
    {
        self.tier_move_candidate_borrowed(candidates, required, recent_per_layer, policy)
    }

    /// Selects a tier move while retaining immutable source ownership.
    /// Active demand, required blocks, prefix protection and ordinary LRU/LFU
    /// ordering are unchanged. The native mover must authenticate the complete
    /// pin population and retain all physical backing through actual retirement.
    pub fn tier_move_candidate_borrowed<'a, I>(
        &self,
        candidates: I,
        required: Option<&CacheBlockId>,
        recent_per_layer: usize,
        policy: CacheEvictionPolicy,
    ) -> Result<Option<&'a CacheBlockId>, CacheLifecycleError>
    where
        I: Iterator<Item = &'a CacheBlockId> + Clone,
    {
        self.select_eviction(candidates, required, recent_per_layer, policy, true)
    }

    /// Fixed control frames for this exact borrowed iterator, including its
    /// reiteration used to apply the existing per-layer recent-block rule.
    pub fn eviction_control_bytes<I>() -> Option<usize> {
        use std::mem::size_of;
        let frames = [
            size_of::<I>(),
            size_of::<I>(),
            size_of::<&Self>(),
            size_of::<Option<&CacheBlockId>>(),
            size_of::<Option<&CacheBlockId>>(),
            size_of::<Option<&CacheBlockId>>(),
            size_of::<Option<&CacheBlockId>>(),
            size_of::<(&CacheBlockId, &BlockLifecycle)>(),
            size_of::<usize>(),
            size_of::<CacheEvictionPolicy>(),
            size_of::<bool>(),
            size_of::<(&Self, CacheEvictionPolicy)>(),
            size_of::<(u64, u64)>(),
            size_of::<((u64, u64), &CacheBlockId)>(),
            size_of::<std::cmp::Ordering>(),
            size_of::<Result<Option<&CacheBlockId>, CacheLifecycleError>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }

    fn select_eviction<'a, I>(
        &self,
        candidates: I,
        required: Option<&CacheBlockId>,
        recent_per_layer: usize,
        policy: CacheEvictionPolicy,
        move_tier: bool,
    ) -> Result<Option<&'a CacheBlockId>, CacheLifecycleError>
    where
        I: Iterator<Item = &'a CacheBlockId> + Clone,
    {
        let mut previous = None;
        for id in candidates.clone() {
            if !self.blocks.contains_key(id) {
                return Err(CacheLifecycleError::MissingBlock(id.clone()));
            }
            if previous.is_some_and(|earlier| earlier > id) {
                return Err(CacheLifecycleError::UnorderedCandidates);
            }
            previous = Some(id);
        }
        let mut selected: Option<&CacheBlockId> = None;
        let mut previous = None;
        for id in candidates.clone() {
            if previous == Some(id) {
                continue;
            }
            previous = Some(id);
            let block = &self.blocks[id];
            if block.leases != 0
                || (!move_tier && block.source_pins != 0)
                || block.protected_prefix
                || required == Some(id)
            {
                continue;
            }
            if recent_per_layer != 0 {
                let mut newer = 0usize;
                let mut prior = None;
                for candidate in candidates.clone() {
                    if prior == Some(candidate) {
                        continue;
                    }
                    prior = Some(candidate);
                    if candidate.global_layer == id.global_layer
                        && !self.blocks[candidate].protected_prefix
                        && (candidate.start, candidate) > (id.start, id)
                    {
                        newer += 1;
                        if newer == recent_per_layer {
                            break;
                        }
                    }
                }
                if newer < recent_per_layer {
                    continue;
                }
            }
            let key = |candidate: &CacheBlockId| {
                let row = &self.blocks[candidate];
                match policy {
                    CacheEvictionPolicy::LeastRecentlyUsed => (row.last_access, row.access_count),
                    CacheEvictionPolicy::LeastFrequentlyUsed => (row.access_count, row.last_access),
                }
            };
            if selected.is_none_or(|old| (key(id), id) < (key(old), old)) {
                selected = Some(id);
            }
        }
        Ok(selected)
    }

    /// Counts unprotected blocks retained by a per-layer recent window.
    pub fn recent_protection_counts(
        &self,
        candidates: impl IntoIterator<Item = CacheBlockId>,
        recent_per_layer: usize,
    ) -> Result<BTreeMap<usize, u64>, CacheLifecycleError> {
        let candidates = candidates
            .into_iter()
            .filter_map(|id| match self.blocks.get(&id) {
                Some(block) if !block.protected_prefix => Some(Ok(id)),
                Some(_) => None,
                None => Some(Err(CacheLifecycleError::MissingBlock(id))),
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut counts = BTreeMap::new();
        self.accumulate_recent_protection_counts(candidates.iter(), recent_per_layer, &mut counts)?;
        Ok(counts)
    }

    /// Exact fixed worker frames for this borrowed candidate iterator and
    /// destination. Destination backing remains its own preparation obligation.
    pub fn recent_count_control_bytes<I, D>() -> Option<usize> {
        use std::mem::size_of;
        let frames = [
            size_of::<&Self>(),
            size_of::<I>(),
            size_of::<I>(),
            size_of::<&mut D>(),
            size_of::<D>(),
            size_of::<usize>(),
            size_of::<Option<&CacheBlockId>>(),
            size_of::<Option<&CacheBlockId>>(),
            size_of::<&CacheBlockId>(),
            size_of::<&mut u64>(),
            size_of::<Result<(), CacheLifecycleError>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }

    /// Adds the same recent counts into caller-owned destination rows. The
    /// canonical candidate order is validated before any destination mutation;
    /// repeated adjacent identities are counted once and prefix blocks never
    /// consume a recent slot. Neither source ownership nor a lease is granted.
    pub fn accumulate_recent_protection_counts<'a, I, D>(
        &self,
        candidates: I,
        recent_per_layer: usize,
        destination: &mut D,
    ) -> Result<(), CacheLifecycleError>
    where
        I: Iterator<Item = &'a CacheBlockId> + Clone,
        D: CacheRecentCountDestination,
    {
        let mut previous = None;
        for id in candidates.clone() {
            if !self.blocks.contains_key(id) {
                return Err(CacheLifecycleError::MissingBlock(id.clone()));
            }
            if previous.is_some_and(|earlier| earlier > id) {
                return Err(CacheLifecycleError::UnorderedCandidates);
            }
            previous = Some(id);
        }
        if recent_per_layer == 0 {
            return Ok(());
        }
        let mut previous = None;
        for id in candidates {
            if previous == Some(id) {
                continue;
            }
            previous = Some(id);
            if self.blocks[id].protected_prefix {
                continue;
            }
            let count = destination.recent_count_mut(id.global_layer);
            if *count < recent_per_layer as u64 {
                *count += 1;
            }
        }
        Ok(())
    }

    /// Replaces one layer's mutable tail, returning the prior state for rollback.
    pub fn set_tail(&mut self, layer: usize, tail: MutableCacheTail) -> Option<MutableCacheTail> {
        self.tails.insert(layer, tail)
    }

    /// Updates the same tail slot without allocating. A new layer requires an
    /// admitted slot even when the supplied tail has no physical payload.
    pub fn set_tail_prepared(
        &mut self,
        layer: usize,
        tail: MutableCacheTail,
    ) -> Result<Option<MutableCacheTail>, CacheLifecycleError> {
        self.validate_tail_slot(layer)?;
        Ok(self.set_tail(layer, tail))
    }
    fn validate_tail_slot(&self, layer: usize) -> Result<(), CacheLifecycleError> {
        let count = self
            .tails
            .len()
            .checked_add(usize::from(!self.tails.contains_key(&layer)))
            .ok_or(CacheTableCapacityError::Exhausted)?;
        self.tails.validate_prepared_population(count)?;
        Ok(())
    }

    /// Installs both prepared catalogs atomically without changing leases,
    /// access order, protected status, or tail coordinates. An undersized
    /// destination is returned intact; no partial catalog replacement occurs.
    pub fn install_storage(
        &mut self,
        storage: PreparedCacheLifecycle,
    ) -> Result<RetiredCacheLifecycleStorage, PreparedCacheLifecycle> {
        if self.blocks.len() > storage.blocks.maximum()
            || self.tails.len() > storage.tails.maximum()
        {
            return Err(storage);
        }
        let blocks = self
            .blocks
            .install(storage.blocks)
            .expect("validated block destination");
        let tails = self
            .tails
            .install(storage.tails)
            .expect("validated tail destination");
        Ok(RetiredCacheLifecycleStorage {
            _blocks: blocks,
            _tails: tails,
        })
    }

    /// Restores a prior tail after a failed backend admission.
    pub fn restore_tail(&mut self, layer: usize, tail: Option<MutableCacheTail>) {
        match tail {
            Some(tail) => {
                self.tails.insert(layer, tail);
            }
            None => {
                self.tails.remove(&layer);
            }
        }
    }

    /// Returns one layer's mutable tail.
    pub fn tail(&self, layer: usize) -> Option<MutableCacheTail> {
        self.tails.get(&layer).copied()
    }

    /// Reiterable layer identities from the same canonical tail catalog.
    pub fn tail_layers(&self) -> impl Iterator<Item = usize> + Clone + '_ {
        self.tails.iter().map(|(layer, _)| *layer)
    }

    /// Iterates mutable tails in stable layer order.
    pub fn tails(&self) -> impl Iterator<Item = (usize, MutableCacheTail)> + '_ {
        self.tails.iter().map(|(layer, tail)| (*layer, *tail))
    }

    /// Clears all blocks and tails only when no lease is active.
    pub fn clear(&mut self) -> Result<(), CacheLifecycleError> {
        if let Some(id) = self.first_leased() {
            return Err(CacheLifecycleError::BlockLeased(id.clone()));
        }
        self.blocks.clear();
        self.tails.clear();
        Ok(())
    }

    fn require_lease_count(
        &self,
        id: &CacheBlockId,
        expected: usize,
    ) -> Result<(), CacheLifecycleError> {
        let actual = self.lease_count(id)?;
        if actual == expected {
            Ok(())
        } else {
            Err(CacheLifecycleError::UnexpectedLeaseCount {
                id: id.clone(),
                expected,
                actual,
            })
        }
    }

    fn tick(&mut self) -> Result<u64, CacheLifecycleError> {
        self.access_clock = self
            .access_clock
            .checked_add(1)
            .ok_or(CacheLifecycleError::AccessClockOverflow)?;
        Ok(self.access_clock)
    }
}

/// Current recent-protection counters supplied by an ordinary or prepaid
/// report destination. Its rows start at zero for each complete snapshot.
pub trait CacheRecentCountDestination {
    /// Supplies this actual layer's existing or newly initialized counter.
    fn recent_count_mut(&mut self, global_layer: usize) -> &mut u64;
}
impl CacheRecentCountDestination for BTreeMap<usize, u64> {
    fn recent_count_mut(&mut self, global_layer: usize) -> &mut u64 {
        self.entry(global_layer).or_default()
    }
}

/// Invalid backend-neutral cache ownership transition.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CacheLifecycleError {
    /// A borrowed canonical candidate traversal changed stable key order.
    #[error("cache candidates are not in canonical order")]
    UnorderedCandidates,
    /// Exact prepared block/tail storage was absent or exhausted.
    #[error(transparent)]
    Capacity(#[from] CacheTableCapacityError),
    /// A block identity was registered more than once.
    #[error("duplicate cache block {0:?}")]
    DuplicateBlock(CacheBlockId),
    /// A requested block is not registered.
    #[error("missing cache block {0:?}")]
    MissingBlock(CacheBlockId),
    /// A removal list repeated one identity.
    #[error("cache block replacement contains a duplicate removal")]
    DuplicateRemoval,
    /// An operation required an unleased block.
    #[error("cache block is leased by active attention: {0:?}")]
    BlockLeased(CacheBlockId),
    /// A transactional replacement observed a different lease count.
    #[error("cache block {id:?} has lease count {actual}, expected {expected}")]
    UnexpectedLeaseCount {
        /// Stable block identity.
        id: CacheBlockId,
        /// Required exact ownership count.
        expected: usize,
        /// Observed exact ownership count.
        actual: usize,
    },
    /// Lease acquisition exceeded addressable ownership.
    #[error("cache block lease count overflowed: {0:?}")]
    LeaseOverflow(CacheBlockId),
    /// A backend released a lease it did not own.
    #[error("cache block lease count underflowed: {0:?}")]
    LeaseUnderflow(CacheBlockId),
    /// The stable access clock exhausted its range.
    #[error("cache block access clock overflowed")]
    AccessClockOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::cache::CacheRepresentation;

    fn id(layer: usize, start: i64) -> CacheBlockId {
        CacheBlockId {
            session_id: 1,
            global_layer: layer,
            representation: CacheRepresentation::KeyValue,
            start,
            end: start + 1,
            rank: None,
        }
    }

    #[test]
    fn leases_are_exact_and_block_destructive_transitions() {
        let block = id(0, 0);
        let mut lifecycle = CacheBlockLifecycle::new();
        lifecycle.insert(block.clone(), false).unwrap();
        lifecycle.acquire(&block).unwrap();
        assert_eq!(lifecycle.lease_count(&block).unwrap(), 1);
        assert!(matches!(
            lifecycle.remove(&block),
            Err(CacheLifecycleError::BlockLeased(_))
        ));
        lifecycle.release(&block).unwrap();
        lifecycle.remove(&block).unwrap();
        assert_eq!(
            lifecycle.release(&block),
            Err(CacheLifecycleError::MissingBlock(block))
        );
    }

    #[test]
    fn source_pins_share_lease_exclusion_without_changing_demand_order() {
        let first = id(0, 0);
        let second = id(0, 1);
        let mut lifecycle = CacheBlockLifecycle::new();
        lifecycle.insert(first.clone(), false).unwrap();
        lifecycle.insert(second.clone(), false).unwrap();
        let clock = lifecycle.access_clock;
        let prior = lifecycle.blocks[&first];
        lifecycle.pin_source(&first).unwrap();
        lifecycle.pin_source(&first).unwrap();
        assert_eq!(lifecycle.lease_count(&first).unwrap(), 2);
        assert_eq!(lifecycle.access_clock, clock);
        assert_eq!(lifecycle.blocks[&first].access_count, prior.access_count);
        assert_eq!(lifecycle.blocks[&first].last_access, prior.last_access);
        assert!(matches!(
            lifecycle.clear(),
            Err(CacheLifecycleError::BlockLeased(_))
        ));
        lifecycle.acquire(&first).unwrap();
        assert_eq!(lifecycle.lease_count(&first).unwrap(), 3);
        assert_eq!(
            lifecycle.blocks[&first].access_count,
            prior.access_count + 1
        );
        lifecycle.release(&first).unwrap();
        lifecycle.release_source(&first).unwrap();
        lifecycle.release_source(&first).unwrap();
        lifecycle.remove(&first).unwrap();
        assert_eq!(lifecycle.lease_count(&second).unwrap(), 0);
    }

    #[test]
    fn borrowed_eviction_separates_source_retention_from_demand_and_preserves_policy() {
        let a = id(0, 0);
        let b = id(0, 1);
        let recent = id(0, 2);
        let prefix = id(0, 3);
        let ids = [
            a.clone(),
            a.clone(),
            b.clone(),
            recent.clone(),
            prefix.clone(),
        ];
        let mut lifecycle = CacheBlockLifecycle::new();
        for item in [&a, &b, &recent] {
            lifecycle.insert(item.clone(), false).unwrap();
        }
        lifecycle.insert(prefix.clone(), true).unwrap();
        lifecycle.pin_source(&a).unwrap();
        lifecycle.pin_source(&a).unwrap();
        assert!(lifecycle.remove(&a).is_err());
        assert!(lifecycle.clear().is_err());
        assert!(!lifecycle.is_device_leased(&a).unwrap());
        assert!(matches!(
            lifecycle.release(&a),
            Err(CacheLifecycleError::LeaseUnderflow(_))
        ));
        assert_eq!(lifecycle.source_pin_count(&a).unwrap(), 2);
        for policy in [
            CacheEvictionPolicy::LeastRecentlyUsed,
            CacheEvictionPolicy::LeastFrequentlyUsed,
        ] {
            assert_eq!(
                lifecycle
                    .eviction_candidate(ids.clone(), None, 1, policy)
                    .unwrap(),
                Some(b.clone())
            );
            assert_eq!(
                lifecycle
                    .eviction_candidate_borrowed(ids.iter(), None, 1, policy)
                    .unwrap(),
                Some(&b)
            );
            assert_eq!(
                lifecycle
                    .device_eviction_candidate_borrowed(ids.iter(), None, 1, policy)
                    .unwrap(),
                Some(&a)
            );
            assert_eq!(
                lifecycle
                    .device_eviction_candidate_borrowed(ids.iter(), Some(&a), 1, policy)
                    .unwrap(),
                Some(&b)
            );
        }
        lifecycle.acquire(&a).unwrap();
        assert!(lifecycle.is_device_leased(&a).unwrap());
        assert_eq!(
            lifecycle
                .device_eviction_candidate_borrowed(
                    ids.iter(),
                    None,
                    1,
                    CacheEvictionPolicy::LeastRecentlyUsed
                )
                .unwrap(),
            Some(&b)
        );
        lifecycle.release(&a).unwrap();
        // Demand changes its real policy rank; source pins themselves do not.
        assert_eq!(
            lifecycle
                .device_eviction_candidate_borrowed(
                    ids.iter(),
                    None,
                    1,
                    CacheEvictionPolicy::LeastRecentlyUsed
                )
                .unwrap(),
            Some(&b)
        );
        assert!(matches!(
            lifecycle.device_eviction_candidate_borrowed(
                [&b, &a].into_iter(),
                None,
                0,
                CacheEvictionPolicy::LeastRecentlyUsed
            ),
            Err(CacheLifecycleError::UnorderedCandidates)
        ));
        lifecycle.release_source(&a).unwrap();
        assert!(lifecycle.remove(&a).is_err());
        lifecycle.release_source(&a).unwrap();
        lifecycle.remove(&a).unwrap();
        lifecycle.clear().unwrap();
    }

    #[test]
    fn rejected_acquisition_does_not_advance_access_order() {
        let block = id(0, 0);
        let missing = id(0, 1);
        let mut lifecycle = CacheBlockLifecycle::new();
        lifecycle.insert(block.clone(), false).unwrap();
        let access_clock = lifecycle.access_clock;

        assert_eq!(
            lifecycle.acquire(&missing),
            Err(CacheLifecycleError::MissingBlock(missing))
        );
        assert_eq!(lifecycle.access_clock, access_clock);

        lifecycle.blocks.get_mut(&block).unwrap().leases = usize::MAX;
        assert_eq!(
            lifecycle.acquire(&block),
            Err(CacheLifecycleError::LeaseOverflow(block))
        );
        assert_eq!(lifecycle.access_clock, access_clock);
    }

    #[test]
    fn replacement_and_tail_update_are_atomic() {
        let first = id(0, 0);
        let second = id(0, 1);
        let replacement = CacheBlockId {
            end: 1,
            ..first.clone()
        };
        let mut lifecycle = CacheBlockLifecycle::new();
        lifecycle.insert(first.clone(), false).unwrap();
        lifecycle.insert(second.clone(), false).unwrap();
        lifecycle.acquire(&first).unwrap();
        lifecycle
            .replace(
                &[(first.clone(), 1), (second.clone(), 0)],
                Some((replacement.clone(), true)),
                0,
                MutableCacheTail { bytes: 0, end: 1 },
            )
            .unwrap();
        assert_eq!(lifecycle.lease_count(&replacement).unwrap(), 0);
        assert!(lifecycle.is_protected_prefix(&replacement).unwrap());
        assert_eq!(
            lifecycle.tail(0),
            Some(MutableCacheTail { bytes: 0, end: 1 })
        );
    }

    #[test]
    fn eviction_is_deterministic_and_respects_owners_and_windows() {
        let oldest = id(0, 0);
        let leased = id(0, 1);
        let recent = id(0, 2);
        let protected = id(1, 0);
        let mut lifecycle = CacheBlockLifecycle::new();
        for (block, prefix) in [
            (oldest.clone(), false),
            (leased.clone(), false),
            (recent.clone(), false),
            (protected.clone(), true),
        ] {
            lifecycle.insert(block, prefix).unwrap();
        }
        lifecycle.acquire(&leased).unwrap();
        let candidates = [oldest.clone(), leased, recent, protected];
        assert_eq!(
            lifecycle
                .eviction_candidate(candidates, None, 1, CacheEvictionPolicy::LeastRecentlyUsed,)
                .unwrap(),
            Some(oldest)
        );
    }

    #[test]
    fn mutable_tail_rollback_restores_exact_state() {
        let mut lifecycle = CacheBlockLifecycle::new();
        let first = MutableCacheTail { bytes: 8, end: 2 };
        assert_eq!(lifecycle.set_tail(3, first), None);
        let prior = lifecycle.set_tail(3, MutableCacheTail { bytes: 16, end: 4 });
        lifecycle.restore_tail(3, prior);
        assert_eq!(lifecycle.tail(3), Some(first));
    }
}

#[cfg(test)]
#[path = "lifecycle/prepared_tests.rs"]
mod prepared_tests;
