//! Backend-neutral ownership for live cache blocks and mutable tails.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::residency::CacheEvictionPolicy;

use super::CacheBlockId;

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
    blocks: BTreeMap<CacheBlockId, BlockLifecycle>,
    tails: BTreeMap<usize, MutableCacheTail>,
}

impl CacheBlockLifecycle {
    /// Creates an empty cache-session lifecycle.
    pub const fn new() -> Self {
        Self {
            access_clock: 0,
            blocks: BTreeMap::new(),
            tails: BTreeMap::new(),
        }
    }

    /// Registers a newly materialized immutable block.
    pub fn insert(
        &mut self,
        id: CacheBlockId,
        protected_prefix: bool,
    ) -> Result<(), CacheLifecycleError> {
        if self.blocks.contains_key(&id) {
            return Err(CacheLifecycleError::DuplicateBlock(id));
        }
        let last_access = self.tick()?;
        self.blocks.insert(
            id,
            BlockLifecycle {
                leases: 0,
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
        let removal_ids = removals.iter().map(|(id, _)| id).collect::<BTreeSet<_>>();
        if removal_ids.len() != removals.len() {
            return Err(CacheLifecycleError::DuplicateRemoval);
        }
        for (id, leases) in removals {
            self.require_lease_count(id, *leases)?;
        }
        if let Some((id, _)) = &replacement {
            if self.blocks.contains_key(id) && !removal_ids.contains(id) {
                return Err(CacheLifecycleError::DuplicateBlock(id.clone()));
            }
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
        let leases = self
            .blocks
            .get(id)
            .ok_or_else(|| CacheLifecycleError::MissingBlock(id.clone()))?
            .leases
            .checked_add(1)
            .ok_or_else(|| CacheLifecycleError::LeaseOverflow(id.clone()))?;
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
            .map(|block| block.leases)
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
            .find_map(|(id, block)| (block.leases != 0).then_some(id))
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
        for id in &candidates {
            if !self.blocks.contains_key(id) {
                return Err(CacheLifecycleError::MissingBlock(id.clone()));
            }
        }
        let recent = recent_ids(&candidates, recent_per_layer);
        Ok(candidates
            .into_iter()
            .filter(|id| {
                let block = self.blocks.get(id).expect("candidate was validated");
                block.leases == 0
                    && !block.protected_prefix
                    && required != Some(id)
                    && !recent.contains(id)
            })
            .min_by_key(|id| {
                let block = self.blocks.get(id).expect("candidate was validated");
                match policy {
                    CacheEvictionPolicy::LeastRecentlyUsed => {
                        (block.last_access, block.access_count, id.clone())
                    }
                    CacheEvictionPolicy::LeastFrequentlyUsed => {
                        (block.access_count, block.last_access, id.clone())
                    }
                }
            }))
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
        for id in recent_ids(&candidates, recent_per_layer) {
            *counts.entry(id.global_layer).or_default() += 1;
        }
        Ok(counts)
    }

    /// Replaces one layer's mutable tail, returning the prior state for rollback.
    pub fn set_tail(&mut self, layer: usize, tail: MutableCacheTail) -> Option<MutableCacheTail> {
        self.tails.insert(layer, tail)
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

fn recent_ids(candidates: &BTreeSet<CacheBlockId>, limit: usize) -> BTreeSet<CacheBlockId> {
    if limit == 0 {
        return BTreeSet::new();
    }
    let mut by_layer = BTreeMap::<usize, Vec<&CacheBlockId>>::new();
    for id in candidates {
        by_layer.entry(id.global_layer).or_default().push(id);
    }
    by_layer
        .into_values()
        .flat_map(|mut ids| {
            ids.sort_unstable_by(|left, right| right.start.cmp(&left.start).then(right.cmp(left)));
            ids.into_iter().take(limit).cloned().collect::<Vec<_>>()
        })
        .collect()
}

/// Invalid backend-neutral cache ownership transition.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CacheLifecycleError {
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
    use crate::cache::CacheRepresentation;

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
