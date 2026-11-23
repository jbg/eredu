//! Page-shaped persistence for contiguous execution state.

use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

use eredu_core::{
    cache::{
        mechanisms::{
            PromptCacheBlockRestore, PromptCacheExport, PromptCacheExportBlock,
            PromptCacheExportInventory, PromptCacheExportState, PromptCacheShardPayload,
            PromptCacheStateRestore, ResidentKeyValueStorage,
        },
        CacheBlockId, CacheRepresentation, PromptCacheBlock, PromptCacheStateTensor,
        StateTensorOwner, StateTensorRole,
    },
    BackendFailure, BackendFailureKind,
};

use super::{
    live::prompt_cache::{
        import_prompt_block, native_prompt_layout, transfer_prompt_state, CachePayload,
    },
    ResidentKeyValueCache,
};

const BLOCK_TOKENS: i32 = 128;

fn invalid(detail: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::InvalidInput,
        std::io::Error::other(detail),
    )
}

pub(crate) struct ResidentExport<M: ResidentKeyValueStorage> {
    inventory: PromptCacheExportInventory,
    blocks: Vec<[M::Tensor; 2]>,
    tensors: Vec<M::Tensor>,
    queue: M::Queue,
}

impl<M: ResidentKeyValueStorage> ResidentExport<M> {
    pub(crate) fn new(queue: &M::Queue) -> Self {
        Self {
            inventory: PromptCacheExportInventory {
                block_size_tokens: BLOCK_TOKENS,
                blocks: Vec::new(),
                states: Vec::new(),
                unsealed_streams: Vec::new(),
                transfer_bytes: 0,
            },
            blocks: Vec::new(),
            tensors: Vec::new(),
            queue: queue.clone(),
        }
    }

    fn account(&mut self, bytes: u64) -> Result<(), BackendFailure> {
        let capacity = M::host_capacity_bound(bytes)?
            .checked_mul(2)
            .ok_or_else(|| invalid("resident prompt-cache transfer capacity overflows"))?;
        self.inventory.transfer_bytes = self.inventory.transfer_bytes.max(capacity);
        Ok(())
    }

    pub(crate) fn add_attention(
        &mut self,
        layer: usize,
        frontier: i32,
        cache: &mut ResidentKeyValueCache<M>,
    ) -> Result<(), BackendFailure> {
        if cache.offset() != frontier {
            return Err(invalid(
                "resident attention differs from the prefix frontier",
            ));
        }
        let Some((keys, values)) = cache.snapshot_arrays(&self.queue)? else {
            return Ok(());
        };
        let origin = frontier - cache.length();
        for start in (0..cache.length()).step_by(BLOCK_TOKENS as usize) {
            let stop = start.saturating_add(BLOCK_TOKENS).min(cache.length());
            let arrays = [
                M::slice_tokens(&keys, 2, start..stop, &self.queue)?,
                M::slice_tokens(&values, 2, start..stop, &self.queue)?,
            ];
            let layouts = [
                native_prompt_layout::<M>(&arrays[0]).map_err(BackendFailure::from_error)?,
                native_prompt_layout::<M>(&arrays[1]).map_err(BackendFailure::from_error)?,
            ];
            self.account(
                layouts[0]
                    .bytes
                    .checked_add(layouts[1].bytes)
                    .ok_or_else(|| invalid("resident prompt-cache block bytes overflow"))?,
            )?;
            self.inventory.blocks.push(PromptCacheExportBlock {
                id: CacheBlockId {
                    session_id: 0,
                    global_layer: layer,
                    representation: CacheRepresentation::KeyValue,
                    start: i64::from(origin + start),
                    end: i64::from(origin + stop),
                    rank: None,
                },
                tensors: layouts,
            });
            self.blocks.push(arrays);
        }
        Ok(())
    }

    pub(crate) fn add_state(
        &mut self,
        layer: usize,
        role: StateTensorRole,
        value: &M::Value,
    ) -> Result<(), BackendFailure> {
        let array = M::storage_ref(value).clone();
        let tensor = native_prompt_layout::<M>(&array).map_err(BackendFailure::from_error)?;
        self.account(tensor.bytes)?;
        self.inventory.states.push(PromptCacheExportState {
            owner: StateTensorOwner::Layer(layer),
            role,
            tensor,
        });
        self.tensors.push(array);
        Ok(())
    }
}

impl<M: ResidentKeyValueStorage> PromptCacheExport for ResidentExport<M> {
    type Payload = CachePayload<M>;

    fn inventory(&self) -> &PromptCacheExportInventory {
        &self.inventory
    }

    fn transfer_block(&self, index: usize) -> Result<Self::Payload, BackendFailure> {
        CachePayload::transfer_block(self.blocks[index].each_ref(), &self.queue)
    }

    fn transfer_state(&self, index: usize) -> Result<Self::Payload, BackendFailure> {
        CachePayload::transfer_state(&self.tensors[index], &self.queue)
    }
}

/// Imports one attention layer or fixed tensor at a time into a provisional owner.
pub(crate) struct ResidentRestore<'a, M: ResidentKeyValueStorage> {
    restore: &'a super::PromptCacheRestore<'a>,
    blocks: BTreeMap<usize, Vec<&'a PromptCacheBlock>>,
    states: BTreeMap<(StateTensorOwner, StateTensorRole), &'a PromptCacheStateTensor>,
    queue: &'a M::Queue,
}

impl<'a, M: ResidentKeyValueStorage> ResidentRestore<'a, M> {
    pub(crate) fn new(restore: &'a super::PromptCacheRestore<'a>, queue: &'a M::Queue) -> Self {
        let mut blocks = BTreeMap::<usize, Vec<_>>::new();
        for block in &restore.manifest().blocks {
            blocks.entry(block.global_layer).or_default().push(block);
        }
        let states = restore
            .manifest()
            .state_tensors
            .iter()
            .map(|state| ((state.owner, state.role), state))
            .collect();
        Self {
            restore,
            blocks,
            states,
            queue,
        }
    }

    pub(crate) fn install_attention(
        &mut self,
        layer: usize,
        frontier: i32,
        cache: &mut ResidentKeyValueCache<M>,
    ) -> Result<(), BackendFailure> {
        let mut blocks = self
            .blocks
            .remove(&layer)
            .ok_or_else(|| invalid("missing resident cache pages"))?;
        blocks.sort_by_key(|block| block.start);
        let mut target = ResidentTransfer::<M> {
            queue: self.queue,
            pages: Vec::with_capacity(blocks.len()),
            fixed: None,
        };
        for block in blocks {
            self.restore
                .restore_block(block, &mut target)
                .map_err(BackendFailure::from_error)?;
        }
        let combine = |slot: usize| {
            if let [page] = target.pages.as_slice() {
                return Ok(page[slot].clone());
            }
            let arrays = target
                .pages
                .iter()
                .map(|page| &page[slot])
                .collect::<Vec<_>>();
            M::concatenate_tokens(&arrays, 2, self.queue)
        };
        let (keys, values) = (combine(0)?, combine(1)?);
        M::settle_tensors(&[&keys, &values])?;
        cache.restore_resident(keys, values, frontier)
    }

    pub(crate) fn install_state(
        &mut self,
        layer: usize,
        role: StateTensorRole,
    ) -> Result<M::Value, BackendFailure> {
        let state = self
            .states
            .remove(&(StateTensorOwner::Layer(layer), role))
            .ok_or_else(|| invalid("missing resident fixed tensor"))?;
        let mut target = ResidentTransfer::<M> {
            queue: self.queue,
            pages: Vec::new(),
            fixed: None,
        };
        self.restore
            .restore_state(state, &mut target)
            .map_err(BackendFailure::from_error)?;
        target
            .fixed
            .map(M::into_value)
            .ok_or_else(|| invalid("missing resident fixed tensor"))
    }

    pub(crate) fn finish(self) -> Result<(), BackendFailure> {
        if !self.blocks.is_empty() || !self.states.is_empty() {
            return Err(invalid("resident cache has unowned tensors"));
        }
        Ok(())
    }
}

struct ResidentTransfer<'a, M: ResidentKeyValueStorage> {
    queue: &'a M::Queue,
    pages: Vec<[M::Tensor; 2]>,
    fixed: Option<M::Tensor>,
}

impl<M: ResidentKeyValueStorage> PromptCacheBlockRestore for ResidentTransfer<'_, M> {
    fn retain_block(
        &mut self,
        block: &PromptCacheBlock,
        pinned: bool,
        source: Arc<dyn PromptCacheShardPayload>,
        _: PathBuf,
    ) -> Result<(), BackendFailure> {
        self.transfer_block(block, pinned, source.as_ref())
    }

    fn transfer_block(
        &mut self,
        block: &PromptCacheBlock,
        _: bool,
        source: &dyn PromptCacheShardPayload,
    ) -> Result<(), BackendFailure> {
        let arrays = import_prompt_block::<M>(block, source, self.queue)
            .map_err(BackendFailure::from_error)?;
        self.pages.push(arrays);
        Ok(())
    }
}

impl<M: ResidentKeyValueStorage> PromptCacheStateRestore for ResidentTransfer<'_, M> {
    fn transfer_state(
        &mut self,
        state: &PromptCacheStateTensor,
        source: &dyn PromptCacheShardPayload,
    ) -> Result<(), BackendFailure> {
        self.fixed = Some(
            transfer_prompt_state::<M>(state, source, self.queue)
                .map_err(BackendFailure::from_error)?,
        );
        Ok(())
    }
}
