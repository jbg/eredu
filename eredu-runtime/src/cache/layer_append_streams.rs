//! Admission, retention, and checkpointing of named streams in a layer.

use super::{live::CacheResidencyManager, PagedAppendStream};
use crate::ResidentAppendStream;
use eredu_core::{
    cache::{
        append_stream::{
            AppendOnlyStream, AppendStreamBacking, AppendStreamBinding, AppendStreamError,
            AppendStreamSpec,
        },
        mechanisms::{CacheTensorStorage, LiveCacheSelectedReadBackend},
        CacheRankIdentity,
    },
    BackendFailure,
};
use std::ops::Range;

/// Runtime policy and mechanism failures while composing layer streams.
#[derive(Debug, thiserror::Error)]
pub enum LayerAppendStreamError {
    /// Paged lanes require the selected page catalog.
    #[error("paged append streams require a shared residency manager")]
    MissingManager,
    /// Checkpoint lane identity or placement differs.
    #[error("append-stream checkpoint bindings, lane identity or placement differ")]
    Checkpoint,
    /// Only paged state supports prompt-cache publication.
    #[error("prompt persistence requires selected paged append streams")]
    ResidentPersistence,
    /// The active batch exceeds the lane capacity.
    #[error("prompt-cache batch exceeds append-stream lane capacity")]
    BatchCapacity,
    /// A smaller batch would discard retained history.
    #[error("prompt-cache batch would omit nonempty append-stream history")]
    OmittedLane,
    /// The selected lane failed admission or execution.
    #[error(transparent)]
    Stream(#[from] AppendStreamError),
    /// A native snapshot copy failed.
    #[error(transparent)]
    Copy(#[from] BackendFailure),
}

/// Record stream selected by an exact component placement and finite binding.
#[derive(Debug, Clone)]
pub enum LayerAppendStream<M: CacheTensorStorage + LiveCacheSelectedReadBackend<Event: 'static>> {
    /// Resident immutable chunks with bounded compact reads.
    Resident(ResidentAppendStream<M::Value>),
    /// Pages sharing the model's ordinary cache residency manager.
    Paged(PagedAppendStream<M>),
}
impl<M: CacheTensorStorage + LiveCacheSelectedReadBackend<Event: 'static>>
    AppendOnlyStream<M::Value> for LayerAppendStream<M>
{
    fn specification(&self) -> &AppendStreamSpec {
        match self {
            Self::Resident(s) => s.specification(),
            Self::Paged(s) => s.specification(),
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Resident(s) => s.len(),
            Self::Paged(s) => s.len(),
        }
    }
    fn append(
        &mut self,
        frontier: usize,
        values: M::Value,
        stream: &M::Queue,
    ) -> Result<(), AppendStreamError> {
        match self {
            Self::Resident(s) => s.append(frontier, values, stream),
            Self::Paged(s) => s.append(frontier, values, stream),
        }
    }
    fn read(
        &mut self,
        range: Range<usize>,
        stream: &M::Queue,
    ) -> Result<M::Value, AppendStreamError> {
        match self {
            Self::Resident(s) => s.read(range, stream),
            Self::Paged(s) => s.read(range, stream),
        }
    }
}

/// All admitted named lanes in one heterogeneous state layer.
#[derive(Debug, Clone)]
pub struct LayerAppendStreams<M: CacheTensorStorage + LiveCacheSelectedReadBackend<Event: 'static>>
{
    bindings: Vec<AppendStreamBinding>,
    values: Vec<(u32, u32, LayerAppendStream<M>)>,
}
impl<M: CacheTensorStorage + LiveCacheSelectedReadBackend<Event: 'static>> LayerAppendStreams<M> {
    /// Realizes each admitted lane with its selected backing.
    pub fn new(
        layer: usize,
        allocations: &[(AppendStreamBinding, AppendStreamBacking)],
        manager: Option<&CacheResidencyManager<M>>,
        rank: Option<CacheRankIdentity>,
    ) -> Result<Self, LayerAppendStreamError> {
        let mut values = Vec::new();
        for (binding, selected) in allocations {
            for lane in 0..binding.lanes {
                let value = match selected {
                    AppendStreamBacking::Resident => {
                        LayerAppendStream::Resident(ResidentAppendStream::new(
                            binding.spec.clone(),
                            binding.limits,
                            binding.payload_bytes,
                            binding.scratch_bytes,
                            binding.catalog_bytes,
                        )?)
                    }
                    AppendStreamBacking::Paged => LayerAppendStream::Paged(PagedAppendStream::new(
                        binding.spec.clone(),
                        binding.limits,
                        manager
                            .ok_or(LayerAppendStreamError::MissingManager)?
                            .clone(),
                        layer,
                        lane,
                        rank,
                        binding.scratch_bytes,
                    )?),
                };
                values.push((binding.spec.slot, lane, value));
            }
        }
        Ok(Self {
            bindings: allocations
                .iter()
                .map(|(binding, _)| binding.clone())
                .collect(),
            values,
        })
    }
    /// Returns native roots owned by the current lane frontiers.
    pub fn retained_values(&self) -> Vec<&M::Value> {
        self.values
            .iter()
            .flat_map(|(_, _, value)| match value {
                LayerAppendStream::Resident(s) => s.retained_values().collect::<Vec<_>>(),
                LayerAppendStream::Paged(s) => s.retained_values().collect(),
            })
            .collect()
    }
    /// Restores lane frontiers using their shared page catalog.
    pub fn restore(
        &mut self,
        previous: &Self,
        stream: &M::Queue,
    ) -> Result<(), LayerAppendStreamError> {
        if self.bindings != previous.bindings || self.values.len() != previous.values.len() {
            return Err(LayerAppendStreamError::Checkpoint);
        }
        for ((slot, lane, current), (old_slot, old_lane, old)) in
            self.values.iter_mut().zip(&previous.values)
        {
            if slot != old_slot || lane != old_lane {
                return Err(LayerAppendStreamError::Checkpoint);
            }
            match (current, old) {
                (LayerAppendStream::Resident(current), LayerAppendStream::Resident(old)) => {
                    current.clone_from(old)
                }
                (LayerAppendStream::Paged(current), LayerAppendStream::Paged(old)) => {
                    current.restore_checkpoint(old, stream)?
                }
                _ => return Err(LayerAppendStreamError::Checkpoint),
            }
        }
        Ok(())
    }
    /// Copies each lane for an isolated branch.
    pub fn fork(
        &self,
        manager: Option<&CacheResidencyManager<M>>,
        stream: &M::Queue,
    ) -> Result<Self, LayerAppendStreamError> {
        self.copy_candidate(manager, stream, true)
    }

    /// Builds a transaction candidate with shared immutable resident roots.
    pub fn transaction_candidate(
        &self,
        manager: Option<&CacheResidencyManager<M>>,
        stream: &M::Queue,
    ) -> Result<Self, LayerAppendStreamError> {
        self.copy_candidate(manager, stream, false)
    }

    fn copy_candidate(
        &self,
        manager: Option<&CacheResidencyManager<M>>,
        stream: &M::Queue,
        isolated: bool,
    ) -> Result<Self, LayerAppendStreamError> {
        Ok(Self {
            bindings: self.bindings.clone(),
            values: self
                .values
                .iter()
                .map(|(slot, lane, value)| {
                    Ok((
                        *slot,
                        *lane,
                        match value {
                            LayerAppendStream::Resident(s) if !isolated => {
                                LayerAppendStream::Resident(s.clone())
                            }
                            LayerAppendStream::Resident(s) => {
                                LayerAppendStream::Resident(s.try_clone_with(|value| {
                                    M::copy_tensor(M::storage_ref(value), stream).map(M::into_value)
                                })?)
                            }
                            LayerAppendStream::Paged(s) => LayerAppendStream::Paged(
                                s.fork_into(
                                    manager
                                        .ok_or(LayerAppendStreamError::MissingManager)?
                                        .clone(),
                                    stream,
                                )?,
                            ),
                        },
                    ))
                })
                .collect::<Result<_, LayerAppendStreamError>>()?,
        })
    }
    /// Clears all retained lanes.
    pub fn clear(&mut self) -> Result<(), LayerAppendStreamError> {
        for (_, _, value) in &mut self.values {
            match value {
                LayerAppendStream::Resident(s) => s.clear(),
                LayerAppendStream::Paged(s) => s.clear()?,
            }
        }
        Ok(())
    }
    /// Seals paged lanes for publication.
    pub fn finalize(&mut self) -> Result<(), LayerAppendStreamError> {
        for (_, _, value) in &mut self.values {
            match value {
                LayerAppendStream::Paged(s) => s.finalize()?,
                LayerAppendStream::Resident(_) => {
                    return Err(LayerAppendStreamError::ResidentPersistence);
                }
            }
        }
        Ok(())
    }
    /// Returns the number of paged lanes.
    pub fn paged_count(&self) -> usize {
        self.values
            .iter()
            .filter(|(_, _, s)| matches!(s, LayerAppendStream::Paged(_)))
            .count()
    }
    /// Reports descriptor and resident catalog bytes.
    pub fn metadata_bytes(&self) -> Option<u64> {
        let bindings = (self.bindings.capacity() as u64)
            .checked_mul(std::mem::size_of::<AppendStreamBinding>() as u64)?;
        let values = (self.values.capacity() as u64).checked_mul(std::mem::size_of::<(
            u32,
            u32,
            LayerAppendStream<M>,
        )>() as u64)?;
        self.values
            .iter()
            .try_fold(bindings.checked_add(values)?, |bytes, (_, _, value)| {
                bytes.checked_add(match value {
                    LayerAppendStream::Resident(s) => s.catalog_bytes()?,
                    LayerAppendStream::Paged(_) => 0, // The shared manager charges its page catalog.
                })
            })
    }
    /// Checks that a batch can be persisted without omitting retained history.
    pub fn validate_batch(&self, batch: usize) -> Result<(), LayerAppendStreamError> {
        if batch == 0 || self.bindings.iter().any(|b| (b.lanes as usize) < batch) {
            return Err(LayerAppendStreamError::BatchCapacity);
        }
        // Bindings reserve maximum batch capacity. Manifests describe active
        // lanes; omitting unused capacity must never discard stored history.
        if self
            .values
            .iter()
            .any(|(_, lane, stream)| (*lane as usize) >= batch && stream.len() != 0)
        {
            return Err(LayerAppendStreamError::OmittedLane);
        }
        Ok(())
    }
    /// Checks that paged lanes belong to the supplied catalog.
    pub fn consistent_manager(&self, manager: Option<&CacheResidencyManager<M>>) -> bool {
        self.values.iter().all(|(_, _, s)| match s {
            LayerAppendStream::Resident(_) => true,
            LayerAppendStream::Paged(s) => {
                manager.is_some_and(|m| m.session_id() == s.manager().session_id())
            }
        })
    }
    // Includes every bound lane and future chunk descriptor. Ordinary facade
    // geometry estimates one text lane; this extra allowance remains conservative
    // for batched low-level callers and physically isolated stream snapshots.
    /// Bounds lane storage after a continuation.
    pub fn snapshot_growth(&self, additional: u64) -> Option<u64> {
        self.bindings.iter().try_fold(0_u64, |bytes, binding| {
            let current = self
                .values
                .iter()
                .filter(|(slot, _, _)| *slot == binding.spec.slot)
                .map(|(_, _, value)| value.len() as u64)
                .max()
                .unwrap_or(0);
            let records = current
                .checked_add(additional)?
                .min(binding.limits.entries as u64);
            let chunks = records.div_ceil(binding.limits.page_entries as u64);
            let payload = records
                .checked_mul(binding.spec.record_bytes().ok()?)?
                .checked_mul(2)?;
            bytes.checked_add(
                payload
                    .checked_add(chunks.checked_mul(4096)?)?
                    .checked_mul(u64::from(binding.lanes))?,
            )
        })
    }
    /// Borrows the selected slot and lane.
    pub fn append_stream(
        &mut self,
        slot: u32,
        lane: u32,
    ) -> Result<&mut LayerAppendStream<M>, AppendStreamError> {
        self.values
            .iter_mut()
            .find(|(s, l, _)| *s == slot && *l == lane)
            .map(|(_, _, v)| v)
            .ok_or(AppendStreamError::Geometry)
    }
    /// Borrows two distinct slots in the supplied lane.
    pub fn append_stream_pair(
        &mut self,
        first: u32,
        second: u32,
        lane: u32,
    ) -> Result<(&mut LayerAppendStream<M>, &mut LayerAppendStream<M>), AppendStreamError> {
        let position = |slot| {
            self.values
                .iter()
                .position(|(s, l, _)| *s == slot && *l == lane)
                .ok_or(AppendStreamError::Geometry)
        };
        let lanes = [position(first)?, position(second)?];
        let [first, second] = self
            .values
            .get_disjoint_mut(lanes)
            .map_err(|_| AppendStreamError::Geometry)?;
        Ok((&mut first.2, &mut second.2))
    }
}

impl<M: CacheTensorStorage + LiveCacheSelectedReadBackend<Event: 'static>> Default
    for LayerAppendStreams<M>
{
    fn default() -> Self {
        Self {
            bindings: Vec::new(),
            values: Vec::new(),
        }
    }
}
