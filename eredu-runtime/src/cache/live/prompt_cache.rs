//! Native execution-state inventory, allocation and completed byte transfers.
use super::*;
use eredu_core::cache::mechanisms::{
    PromptCacheBlockRestore, PromptCacheExport, PromptCacheExportBlock, PromptCacheExportInventory,
    PromptCacheExportState, PromptCachePayload, PromptCacheShardPayload, PromptCacheStateRestore,
    PromptCacheTensorLayout,
};
use eredu_core::BackendFailure;

/// Native binding of an architecture-declared fixed state tensor.
pub struct PromptCacheStateArray<T> {
    /// Architecture-declared state owner.
    pub owner: StateTensorOwner,
    /// Architecture-declared tensor role.
    pub role: StateTensorRole,
    /// Native tensor held by the completed execution state.
    pub array: T,
}
pub(super) fn prompt_shape(shape: &[i32]) -> Result<Vec<usize>, CacheResidencyError> {
    shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                CacheResidencyError::ArrayMismatch("negative cache tensor dimension".into())
            })
        })
        .collect()
}

pub(super) fn transfer_prompt_block<M: LiveCacheBackend>(
    block: &PromptCacheBlock,
    source: &dyn PromptCacheShardPayload,
) -> Result<HostCacheBlock<M>, CacheResidencyError> {
    let mut first = M::allocate(
        &prompt_shape(&block.first_shape)?,
        stored_dtype(&block.first_dtype)?,
    )?;
    let mut second = M::allocate(
        &prompt_shape(&block.second_shape)?,
        stored_dtype(&block.second_dtype)?,
    )?;
    source.read_into(&mut [
        (&block.first_array, M::bytes_mut(&mut first)?),
        (&block.second_array, M::bytes_mut(&mut second)?),
    ])?;
    Ok(HostCacheBlock {
        representation: block.representation,
        buffers: [M::freeze_host(first), M::freeze_host(second)],
    })
}

fn transfer_prompt_state<M: LiveCacheBackend>(
    state: &PromptCacheStateTensor,
    source: &dyn PromptCacheShardPayload,
    stream: &Queue<M>,
) -> Result<M::Tensor, CacheResidencyError> {
    let mut buffer = M::allocate(&prompt_shape(&state.shape)?, stored_dtype(&state.dtype)?)?;
    source.read_into(&mut [(&state.array, M::bytes_mut(&mut buffer)?)])?;
    let [(tensor, event)] = M::copy_to_device([&M::freeze_host(buffer)], stream)?;
    M::synchronize(event)?;
    Ok(tensor)
}

/// Factory and finalization used by the portable prompt-cache restoration driver.
pub trait RestoreCache<C: ?Sized>:
    PromptCacheBlockRestore + PromptCacheStateRestore + Sized
{
    /// Restored resources ready to install into the retained model state.
    type Restored;

    /// Creates an empty destination for the selected paged cache.
    fn begin(options: &PagedCacheOptions, queue: &C) -> Result<Self, CacheResidencyError>;

    /// Finalizes the complete validated restore.
    fn complete(
        self,
        manifest: &PromptCacheManifest,
    ) -> Result<Self::Restored, CacheResidencyError>;
}

/// A restored live cache and its separately transferred fixed state.
pub type RestoredCache<M> = (
    CacheResidencyManager<M>,
    Vec<LoadedPromptCacheStateTensor<<M as LiveCacheBackend>::Tensor>>,
);

/// Owns a portable cache and fixed-state destinations during a validated restore.
pub struct CacheRestore<M: LiveCacheBackend> {
    manager: CacheResidencyManager<M>,
    stream: Queue<M>,
    tensors: Vec<LoadedPromptCacheStateTensor<M::Tensor>>,
}
impl<M: LiveCacheBackend> CacheRestore<M> {
    /// Starts constructing a cache for restored pages.
    pub fn new(options: PagedCacheOptions, stream: &Queue<M>) -> Result<Self, CacheResidencyError> {
        Ok(Self {
            manager: CacheResidencyManager::new(options)?,
            stream: stream.clone(),
            tensors: Vec::new(),
        })
    }

    /// Publishes the restored cache and the transferred fixed-state tensors.
    pub fn finish(
        self,
        manifest: &PromptCacheManifest,
    ) -> Result<
        (
            CacheResidencyManager<M>,
            Vec<LoadedPromptCacheStateTensor<M::Tensor>>,
        ),
        CacheResidencyError,
    > {
        {
            let mut state = self.manager.lock()?;
            state.telemetry.report.prompt_cache_loads += 1;
            state.telemetry.report.prompt_cache_bytes +=
                manifest.blocks.iter().map(|b| b.logical_bytes).sum::<u64>();
            state.telemetry.report.imported_retained_shards += manifest.blocks.len() as u64;
            update_report_totals(&mut state);
        }
        Ok((self.manager, self.tensors))
    }
}
impl<M: LiveCacheBackend> RestoreCache<Queue<M>> for CacheRestore<M> {
    type Restored = RestoredCache<M>;

    fn begin(options: &PagedCacheOptions, queue: &Queue<M>) -> Result<Self, CacheResidencyError> {
        Self::new(options.clone(), queue)
    }

    fn complete(
        self,
        manifest: &PromptCacheManifest,
    ) -> Result<Self::Restored, CacheResidencyError> {
        self.finish(manifest)
    }
}

impl<M: LiveCacheBackend> PromptCacheBlockRestore for CacheRestore<M> {
    fn transfer_block(
        &mut self,
        block: &PromptCacheBlock,
        pinned: bool,
        source: &dyn PromptCacheShardPayload,
    ) -> Result<(), BackendFailure> {
        (|| -> Result<(), CacheResidencyError> {
            let host = transfer_prompt_block::<M>(block, source)?;
            let (arrays, events) = host.copy_to_device(&self.stream)?;
            for event in events {
                M::synchronize(event)?;
            }
            self.manager.seal_block(
                block.global_layer,
                block.start,
                block.end,
                block.rank,
                arrays,
                pinned,
            )?;

            Ok(())
        })()
        .map_err(BackendFailure::from_error)
    }
    fn retain_block(
        &mut self,
        block: &PromptCacheBlock,
        pinned: bool,
        source: Arc<dyn PromptCacheShardPayload>,
        shard: PathBuf,
    ) -> Result<(), BackendFailure> {
        (|| -> Result<(), CacheResidencyError> {
            let id = CacheBlockId {
                session_id: self.manager.session_id,
                global_layer: block.global_layer,
                representation: block.representation,
                start: block.start,
                end: block.end,
                rank: block.rank,
            };
            let mut state = self.manager.lock()?;
            let record = CacheBlockRecord {
                physical: LiveCacheBlockStorage::disk(
                    id.clone(),
                    DiskLocation {
                        path: shard,
                        first_name: block.first_array.clone(),
                        second_name: block.second_array.clone(),
                        persistent: true,
                        source: Some((source, block.clone())),
                        logical_bytes: block.logical_bytes,
                    },
                ),
                bytes: block.logical_bytes,
                shapes: [block.first_shape.clone(), block.second_shape.clone()],
                dtypes: [block.first_dtype.clone(), block.second_dtype.clone()],
                imported: true,
            };
            state.lifecycle.insert(id.clone(), pinned)?;
            if state.blocks.insert(id.clone(), record).is_some() {
                return Err(CacheLifecycleError::DuplicateBlock(id).into());
            }

            Ok(())
        })()
        .map_err(BackendFailure::from_error)
    }
}
/// Native binding produced by a completed state restore.
pub struct LoadedPromptCacheStateTensor<T> {
    /// Architecture-declared state owner.
    pub owner: StateTensorOwner,
    /// Architecture-declared tensor role.
    pub role: StateTensorRole,
    /// Restored native tensor.
    pub array: T,
}
impl<M: LiveCacheBackend> PromptCacheStateRestore for CacheRestore<M> {
    fn transfer_state(
        &mut self,
        state: &PromptCacheStateTensor,
        source: &dyn PromptCacheShardPayload,
    ) -> Result<(), BackendFailure> {
        let array = transfer_prompt_state::<M>(state, source, &self.stream)
            .map_err(BackendFailure::from_error)?;
        self.tensors.push(LoadedPromptCacheStateTensor {
            owner: state.owner,
            role: state.role,
            array,
        });
        Ok(())
    }
}
enum CacheBuffers<M: LiveCacheBackend> {
    Block(HostCacheBlock<M>),
    State(M::Host),
}

/// Completed host-transfer buffers in portable inventory order.
pub struct CachePayload<M: LiveCacheBackend>(CacheBuffers<M>);

impl<M: LiveCacheBackend> PromptCachePayload for CachePayload<M> {
    fn bytes(&self) -> Result<Vec<&[u8]>, BackendFailure> {
        match &self.0 {
            CacheBuffers::Block(block) => {
                let [first, second] = block.buffers();
                Ok(vec![
                    M::host_payload(first)?.bytes,
                    M::host_payload(second)?.bytes,
                ])
            }
            CacheBuffers::State(buffer) => Ok(vec![M::host_payload(buffer)?.bytes]),
        }
    }
}

/// Native handles for the admitted block and fixed-state inventory.
pub struct CacheExport<M: LiveCacheBackend> {
    manager: CacheResidencyManager<M>,
    inventory: PromptCacheExportInventory,
    records: Vec<CacheBlockRecord<M>>,
    states: Vec<PromptCacheStateArray<M::Tensor>>,
    stream: Queue<M>,
}
impl<M: LiveCacheBackend> PromptCacheExport for CacheExport<M> {
    type Payload = CachePayload<M>;
    fn inventory(&self) -> &PromptCacheExportInventory {
        &self.inventory
    }
    fn transfer_block(&self, index: usize) -> Result<CachePayload<M>, BackendFailure> {
        let record = &self.records[index];
        (|| -> Result<_, CacheResidencyError> {
            let host = if let Some(arrays) = record.physical.device_resource() {
                HostCacheBlock::from_device_arrays(arrays, &self.stream)?
            } else if let Some(host) = record.physical.host_resource() {
                host.clone()
            } else {
                load_host_cache_block_direct::<M>(
                    record
                        .physical
                        .backing()
                        .expect("backing-only cache record"),
                    record.physical.id().representation,
                    &record.shapes,
                    &record.dtypes,
                )?
            };
            Ok(CachePayload(CacheBuffers::Block(host)))
        })()
        .map_err(BackendFailure::from_error)
    }
    fn transfer_state(&self, index: usize) -> Result<CachePayload<M>, BackendFailure> {
        let [host] = M::copy_to_host([&self.states[index].array], &self.stream)?;
        Ok(CachePayload(CacheBuffers::State(host)))
    }
    fn record_publication(&self, logical_bytes: u64) {
        if let Ok(mut state) = self.manager.lock() {
            let report = &mut state.telemetry.report;
            report.prompt_cache_saves = report.prompt_cache_saves.saturating_add(1);
            report.prompt_cache_bytes = report.prompt_cache_bytes.saturating_add(logical_bytes);
        }
    }
}
impl<M: LiveCacheBackend> CacheResidencyManager<M> {
    /// Exposes native block inventory and completed transfers to the runtime publisher.
    pub fn cache_export(
        &self,
        descriptor: &PromptCacheDescriptor,
        state_arrays: Vec<PromptCacheStateArray<M::Tensor>>,
    ) -> Result<CacheExport<M>, CacheResidencyError> {
        let records = {
            let state = self.lock()?;
            let mut records = Vec::new();
            for record in state.blocks.values().filter(|record| {
                (descriptor.global_layer_start()..descriptor.global_layer_end())
                    .contains(&record.physical.id().global_layer)
            }) {
                if state.lifecycle.is_leased(record.physical.id())? {
                    return Err(
                        CacheLifecycleError::BlockLeased(record.physical.id().clone()).into(),
                    );
                }
                records.push(record.clone());
            }
            records
        };
        let blocks = records
            .iter()
            .map(|record| {
                Ok(PromptCacheExportBlock {
                    id: record.physical.id().clone(),
                    tensors: [
                        PromptCacheTensorLayout::new(
                            record.shapes[0].clone(),
                            record.dtypes[0].clone(),
                        )?,
                        PromptCacheTensorLayout::new(
                            record.shapes[1].clone(),
                            record.dtypes[1].clone(),
                        )?,
                    ],
                })
            })
            .collect::<Result<Vec<_>, PromptCacheStoreError>>()?;
        let states = state_arrays
            .iter()
            .map(|state| {
                Ok(PromptCacheExportState {
                    owner: state.owner,
                    role: state.role,
                    tensor: PromptCacheTensorLayout::new(
                        M::shape(&state.array).to_vec(),
                        dtype_name(M::dtype(&state.array))?.to_owned(),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, CacheResidencyError>>()?;
        let mut transfer_bytes = 0;
        for bytes in records
            .iter()
            .map(|r| r.bytes)
            .chain(state_arrays.iter().map(|s| M::bytes(&s.array)))
        {
            let capacity = M::host_capacity_bound(bytes)?;
            transfer_bytes = transfer_bytes.max(capacity.checked_mul(2).ok_or_else(|| {
                CacheResidencyError::Runtime("native transfer capacity overflow".into())
            })?);
        }
        let unsealed_streams = self
            .lock()?
            .lifecycle
            .tails()
            .filter(|(_, tail)| tail.bytes != 0)
            .map(|(stream, _)| stream)
            .collect();
        Ok(CacheExport {
            manager: self.clone(),
            inventory: PromptCacheExportInventory {
                block_size_tokens: self.options().block_size_tokens(),
                blocks,
                states,
                unsealed_streams,
                transfer_bytes,
            },
            records,
            states: state_arrays,
            stream: M::transfer_queue(M::host_device(), CacheTransferLane::Persistence)?,
        })
    }
}
