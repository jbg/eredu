//! Closed completed count source: explicit physical edges, never inferred idle rows.
use super::*;
use eredu_core::{ErasedSharedStorageOwner, SharedStorageOwner, SharedStorageRetirement};
use eredu_runtime::replicated_session::ParallelControlIdentity;
use std::{alloc::Layout, sync::{Arc, atomic::AtomicUsize}};

pub(super) struct WireRow {
    values: Vec<i32>,
    _custody: Custody,
}
impl WireRow {
    pub(super) fn new(local: &[i32], members: &[usize], physical_rank: usize, world: usize, c: &Custody)
        -> Result<Self, Error> {
        if members.len() > world || physical_rank >= world { return Err(fail(PeerCountCause::Identity, c)); }
        let width = width(world)?;
        let mut values = destination::<i32>(width, c)?;
        values.push(i32::try_from(physical_rank.checked_add(1).ok_or_else(overflow)?)
            .map_err(|_| fail(PeerCountCause::Identity, c))?);
        values.push(i32::try_from(members.len()).map_err(|_| fail(PeerCountCause::Identity, c))?);
        for (&member, &count) in members.iter().zip(local) {
            values.push(i32::try_from(member).map_err(|_| fail(PeerCountCause::Identity, c))?);
            values.push(count);
        }
        for _ in members.len()..world { values.push(-1); values.push(0); }
        Ok(Self { values, _custody: c.clone() })
    }
    pub(super) fn values(&self) -> &[i32] { &self.values }
}
fn width(peers: usize) -> Result<usize, Error> {
    OriginalParallelSource::peer_count_wire_words(peers).ok_or_else(overflow)
}
pub(super) fn destination<T>(length: usize, c: &Custody) -> Result<Vec<T>, Error> {
    reserve(&c.funding, &[
        Layout::array::<T>(length).map_err(|_| overflow())?.size(),
        size_of::<Vec<T>>(), size_of::<Result<Vec<T>, Error>>(),
        size_of::<std::collections::TryReserveError>(),
    ])?;
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|cause| fail(PeerCountCause::Allocation(cause), c))?;
    Ok(values)
}
/// Each row identifies its actual source and every destination. A row of all
/// zero counts remains distinguishable from an absent participant.
pub(super) struct WorldTable {
    counts: Vec<usize>,
    world: usize,
    _custody: Custody,
}
impl WorldTable {
    pub(super) fn decode(words: &[i32], peers: usize, world: usize, c: &Custody)
        -> Result<Self, Error> {
        let width = width(peers)?;
        if world == 0 || peers == 0 || peers > world
            || words.len() != world.checked_mul(width).ok_or_else(overflow)? {
            return Err(fail(PeerCountCause::Identity, c));
        }
        let mut counts = destination::<usize>(world.checked_mul(world).ok_or_else(overflow)?, c)?;
        counts.resize(world * world, 0);
        for (rank, row) in words.chunks_exact(width).enumerate() {
            validate_row(row, peers, rank, world, c)?;
            for pair in row[2..2 + row[1] as usize * 2].chunks_exact(2) {
                counts[rank * world + pair[0] as usize] = pair[1] as usize;
            }
        }
        Ok(Self { counts, world, _custody: c.clone() })
    }
}
fn validate_row(row: &[i32], peers: usize, rank: usize, world: usize, c: &Custody)
    -> Result<(), Error> {
    if usize::try_from(row[0]).ok() != rank.checked_add(1)
        || usize::try_from(row[1]).ok().is_none_or(|actual| actual == 0 || actual > peers) {
        return Err(fail(PeerCountCause::Identity, c));
    }
    let actual = row[1] as usize;
    if row[2 + actual * 2..].chunks_exact(2).any(|pair| pair != [-1, 0]) {
        return Err(fail(PeerCountCause::Identity, c));
    }
    let mut self_seen = false;
    for (index, pair) in row[2..2 + actual * 2].chunks_exact(2).enumerate() {
        let destination = usize::try_from(pair[0]).map_err(|_| fail(PeerCountCause::Identity, c))?;
        if destination >= world || pair[1] < 0
            || row[2..2 + index * 2].chunks_exact(2).any(|prior| prior[0] == pair[0]) {
            return Err(fail(PeerCountCause::Identity, c));
        }
        self_seen |= destination == rank;
    }
    if !self_seen { return Err(fail(PeerCountCause::Identity, c)); }
    Ok(())
}
/// The source retains no native array or thread-local projection. Its closed
/// owner retires the Arc allocation before any source/funding custody.
pub(in super::super) struct PeerCountTable {
    local: Vec<i32>,
    members: Vec<usize>,
    world: Option<WorldTable>,
    order: usize,
    identity: ParallelControlIdentity,
    ordinal: u64,
    custody: Custody,
}
impl SharedStorageRetirement for PeerCountTable {
    fn retire(self: Arc<Self>) { drop(Arc::into_inner(self)); }
}
impl PeerCountTable {
    /// The physical gather already returned one exact source-major matrix.
    /// Retain it under the same completed control claim; no world rows or
    /// absent participant data are synthesized for this local physical group.
    pub(super) fn from_completed_physical(words: &[i32], members: &[usize],
        order: usize, identity: ParallelControlIdentity, ordinal: u64, c: &Custody)
        -> Result<SharedStorageOwner<Self>, Error> {
        Self::prepare_controls(c)?;
        if members.is_empty() || words.len() != members.len().checked_mul(members.len()).ok_or_else(overflow)?
            || words.iter().any(|value| *value < 0) {
            return Err(fail(PeerCountCause::Identity, c));
        }
        let mut local = destination::<i32>(words.len(), c)?;
        local.extend_from_slice(words);
        let mut owned_members = destination::<usize>(members.len(), c)?;
        owned_members.extend_from_slice(members);
        Ok(SharedStorageOwner::new(Self { local, members: owned_members, world: None,
            order, identity, ordinal, custody: c.clone() }))
    }
    pub(super) fn prepare_controls(c: &Custody) -> Result<(), Error> {
        reserve(&c.funding, &[
            Layout::new::<[AtomicUsize; 2]>().extend(Layout::new::<Self>())
                .map_err(|_| overflow())?.0.pad_to_align().size(),
            size_of::<Self>(), size_of::<Option<Self>>(), size_of::<WorldTable>(),
            size_of::<Option<WorldTable>>(), size_of::<WireRow>(),
            size_of::<SharedStorageOwner<Self>>(), size_of::<ErasedSharedStorageOwner>(),
            size_of::<Arc<Self>>(), size_of::<Arc<dyn std::any::Any + Send + Sync>>(),
            size_of::<Option<Arc<dyn std::any::Any + Send + Sync>>>(),
            size_of::<Result<Arc<Self>, Arc<dyn std::any::Any + Send + Sync>>>(),
            size_of::<Result<SharedStorageOwner<Self>, Error>>(),
            size_of::<RefCell<Option<WorldTable>>>(),
            size_of::<Result<WorldTable, Error>>(),
            size_of::<[usize; 2]>(), size_of::<[usize; 1]>(),
        ])
    }
    pub(super) fn new(words: &[i32], members: &[usize], world_size: usize,
        world: Option<WorldTable>, order: usize, identity: ParallelControlIdentity,
        ordinal: u64, c: &Custody) -> Result<SharedStorageOwner<Self>, Error> {
        let peers = members.len();
        let width = width(world_size)?;
        if peers == 0 || words.len() != peers.checked_mul(width).ok_or_else(overflow)? {
            return Err(fail(PeerCountCause::Identity, c));
        }
        let mut local = destination::<i32>(peers.checked_mul(peers).ok_or_else(overflow)?, c)?;
        for (source, row) in words.chunks_exact(width).enumerate() {
            validate_row(row, world_size, members[source], world_size, c)?;
            if usize::try_from(row[1]).ok() != Some(peers) { return Err(fail(PeerCountCause::Identity, c)); }
            for (destination, pair) in row[2..2 + peers * 2].chunks_exact(2).enumerate() {
                if usize::try_from(pair[0]).ok() != Some(members[destination]) {
                    return Err(fail(PeerCountCause::Identity, c));
                }
                if world.as_ref().is_some_and(|table| table.world != world_size
                    || table.counts[members[source] * world_size + members[destination]] != pair[1] as usize) {
                    return Err(fail(PeerCountCause::Identity, c));
                }
                local.push(pair[1]);
            }
        }
        let mut owned_members = destination::<usize>(peers, c)?;
        owned_members.extend_from_slice(members);
        Ok(SharedStorageOwner::new(Self { local, members: owned_members, world, order,
            identity, ordinal, custody: c.clone() }))
    }
    pub(super) fn with_world(&self, world: WorldTable, c: &Custody) -> Result<SharedStorageOwner<Self>, Error> {
        Self::prepare_controls(c)?;
        if self.world.is_some() || !self.custody.source.same_source(&c.source)
            || !self.custody.funding.same_account(&c.funding) {
            return Err(fail(PeerCountCause::Identity, c));
        }
        for (source, &physical_source) in self.members.iter().enumerate() {
            for (destination, &physical_destination) in self.members.iter().enumerate() {
                if physical_source >= world.world || physical_destination >= world.world
                    || world.counts[physical_source * world.world + physical_destination]
                        != self.local[source * self.members.len() + destination] as usize {
                    return Err(fail(PeerCountCause::Identity, c));
                }
            }
        }
        let mut local = destination::<i32>(self.local.len(), c)?; local.extend_from_slice(&self.local);
        let mut members = destination::<usize>(self.members.len(), c)?; members.extend_from_slice(&self.members);
        Ok(SharedStorageOwner::new(Self { local, members, world: Some(world), order: self.order,
            identity: self.identity, ordinal: self.ordinal, custody: c.clone() }))
    }
    pub(super) fn local(&self) -> &[i32] { &self.local }
    /// Consumer authentication includes the issuing request and selected source;
    /// equal matrix values alone never authorize a foreign or restored request.
    pub(in super::super) fn region_received(&self, group: eredu_core::CollectiveGroupId,
        rank: usize, peers: usize, owner: &Owner, c: &Custody) -> Option<usize> {
        if !self.belongs_to(owner,c) || peers.checked_mul(peers)? != self.local.len() || rank >= peers { return None; }
        let source = owner.request.source.communication_source().ok()?;
        let (_, descriptor, _) = source.group(self.order)?;
        if descriptor.id() != group || descriptor.local_index() != Some(rank) || descriptor.members().len() != peers { return None; }
        (0..peers).try_fold(0usize, |sum, peer| sum.checked_add(usize::try_from(self.local[peer*peers+rank]).ok()?))
    }
    pub(in super::super) fn belongs_to(&self, owner: &Owner, c: &Custody) -> bool {
        self.identity == owner.request.cursor.borrow().identity()
            && self.ordinal < owner.request.cursor.borrow().attempted()
            && self.custody.source.same_source(&c.source)
            && self.custody.funding.same_account(&c.funding)
    }
    pub(in super::super) fn matches(&self, matrix: &eredu_runtime::CommunicationPeerMatrix<'_>,
        order: usize, owner: &Owner, c: &Custody) -> bool {
        self.order == order && self.members == matrix.descriptor().members()
            && self.identity == owner.request.cursor.borrow().identity()
            && self.ordinal < owner.request.cursor.borrow().attempted()
            && self.custody.source.same_source(&c.source)
            && self.custody.funding.same_account(&c.funding)
            && self.local.len() == matrix.forward_counts().len()
            && self.local.iter().zip(matrix.forward_counts()).all(|(&a, &b)| usize::try_from(a).ok() == Some(b))
    }
    pub(in super::super) fn world(&self) -> Option<&[usize]> {
        self.world.as_ref().map(|table| table.counts.as_slice())
    }
}
