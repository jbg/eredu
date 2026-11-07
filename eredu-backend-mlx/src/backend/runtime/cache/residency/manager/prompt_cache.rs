//! Native execution-state inventory, allocation and completed byte transfers.
use super::*;
use eredu_core::BackendFailure;
use eredu_runtime::{
    PromptCacheBlockRestore, PromptCacheExport, PromptCacheExportBlock, PromptCacheExportInventory,
    PromptCacheExportState, PromptCachePayload, PromptCacheStateRestore, PromptCacheTensorLayout,
};

/// Native binding of an architecture-declared fixed state tensor.
pub struct PromptCacheStateArray<'a> {
    /// Architecture-declared state owner.
    pub owner: StateTensorOwner,
    /// Architecture-declared tensor role.
    pub role: StateTensorRole,
    /// Native tensor held by the completed execution state.
    pub array: &'a Array,
}
fn native_cache_dtype(name: &str) -> Result<Dtype, CacheResidencyError> {
    Ok(match name {
        "Bool" => Dtype::Bool,
        "Uint8" => Dtype::Uint8,
        "Uint16" => Dtype::Uint16,
        "Uint32" => Dtype::Uint32,
        "Uint64" => Dtype::Uint64,
        "Int8" => Dtype::Int8,
        "Int16" => Dtype::Int16,
        "Int32" => Dtype::Int32,
        "Int64" => Dtype::Int64,
        "Float16" => Dtype::Float16,
        "Bfloat16" => Dtype::Bfloat16,
        "Float32" => Dtype::Float32,
        "Float64" => Dtype::Float64,
        _ => {
            return Err(CacheResidencyError::ArrayMismatch(
                "unsupported native state dtype".into(),
            ))
        }
    })
}
fn transfer_prompt_block(
    block: &PromptCacheBlock,
    source: &eredu_runtime::RetainedCacheShard,
) -> Result<HostCacheBlock, CacheResidencyError> {
    let mut first = HostTransferBuffer::new(
        &block.first_shape,
        native_cache_dtype(&block.first_dtype)?,
        HostTransferPolicy::Transfer,
    )
    .map_err(|e| transfer_error("allocate first restore buffer", e))?;
    let mut second = HostTransferBuffer::new(
        &block.second_shape,
        native_cache_dtype(&block.second_dtype)?,
        HostTransferPolicy::Transfer,
    )
    .map_err(|e| transfer_error("allocate second restore buffer", e))?;
    source.read_into(
        &mut [
            (
                &block.first_array,
                first
                    .as_bytes_mut()
                    .map_err(|e| transfer_error("bind first restore bytes", e))?,
            ),
            (
                &block.second_array,
                second
                    .as_bytes_mut()
                    .map_err(|e| transfer_error("bind second restore bytes", e))?,
            ),
        ],
        Some(&block.payload_sha256),
    )?;
    Ok(HostCacheBlock::from_buffers(
        block.representation,
        first.freeze(),
        second.freeze(),
    ))
}
fn transfer_prompt_state(
    state: &PromptCacheStateTensor,
    source: &eredu_runtime::RetainedCacheShard,
    stream: &Stream,
) -> Result<Array, CacheResidencyError> {
    let mut host = HostTransferBuffer::new(
        &state.shape,
        native_cache_dtype(&state.dtype)?,
        HostTransferPolicy::Transfer,
    )
    .map_err(|e| transfer_error("allocate state restore buffer", e))?;
    source.read_into(
        &mut [(
            &state.array,
            host.as_bytes_mut()
                .map_err(|e| transfer_error("bind state restore bytes", e))?,
        )],
        Some(&state.payload_sha256),
    )?;
    host.copy_to_array(stream)
        .and_then(|p| p.synchronize())
        .map(|(array, _)| array)
        .map_err(|e| transfer_error("complete state restore", e))
}
struct NativeBlockRestore<'a> {
    manager: &'a CacheResidencyManager,
    stream: &'a Stream,
    sink_tokens: usize,
}
impl PromptCacheBlockRestore for NativeBlockRestore<'_> {
    fn transfer_block(
        &mut self,
        block: &PromptCacheBlock,
        source: &eredu_runtime::RetainedCacheShard,
    ) -> Result<(), BackendFailure> {
        (|| -> Result<(), CacheResidencyError> {
            let host = transfer_prompt_block(block, source)?;
            let [first, second] = host.buffers();
            let (first, completion) = first
                .copy_to_array(self.stream)
                .map_err(|e| transfer_error("restore cache block", e))?
                .into_parts();
            completion
                .synchronize()
                .map_err(|e| transfer_error("complete cache restore", e))?;
            let (second, completion) = second
                .copy_to_array(self.stream)
                .map_err(|e| transfer_error("restore cache block", e))?
                .into_parts();
            completion
                .synchronize()
                .map_err(|e| transfer_error("complete cache restore", e))?;
            let arrays = match block.representation {
                CacheRepresentation::KeyValue => CacheBlockArrays::KeyValue {
                    keys: first,
                    values: second,
                },
                CacheRepresentation::CompressedLatentRotary => {
                    CacheBlockArrays::CompressedLatentRotary {
                        latent: first,
                        rotary_key: second,
                    }
                }
                CacheRepresentation::AppendStream { slot, lane } => {
                    CacheBlockArrays::AppendStream {
                        slot,
                        lane,
                        records: first,
                        reserved: second,
                    }
                }
            };
            self.manager.seal_block(
                block.global_layer,
                block.start,
                block.end,
                block.rank,
                arrays,
                !matches!(
                    block.representation,
                    CacheRepresentation::AppendStream { .. }
                ) && block.end <= self.sink_tokens as i64,
            )?;

            Ok(())
        })()
        .map_err(BackendFailure::from_error)
    }
    fn retain_block(
        &mut self,
        block: &PromptCacheBlock,
        source: Arc<eredu_runtime::RetainedCacheShard>,
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
                physical: MlxCacheBlockStorage::disk(
                    id.clone(),
                    DiskLocation {
                        path: shard,
                        first_name: block.first_array.clone(),
                        second_name: block.second_array.clone(),
                        persistent: true,
                        source: Some(source),
                        logical_bytes: block.logical_bytes,
                        payload_sha256: Some(block.payload_sha256.clone()),
                    },
                ),
                bytes: block.logical_bytes,
                shapes: [block.first_shape.clone(), block.second_shape.clone()],
                dtypes: [block.first_dtype.clone(), block.second_dtype.clone()],
                imported: true,
            };
            state.lifecycle.insert(
                id.clone(),
                !matches!(
                    block.representation,
                    CacheRepresentation::AppendStream { .. }
                ) && block.end <= self.sink_tokens as i64,
            )?;
            if state.blocks.insert(id.clone(), record).is_some() {
                return Err(CacheLifecycleError::DuplicateBlock(id).into());
            }

            Ok(())
        })()
        .map_err(BackendFailure::from_error)
    }
}
/// Creates native paging storage from the runtime's validated restore plan.
pub fn open_prompt_cache(
    read: &dyn PromptCacheRead,
    expected: &PromptCacheDescriptor,
    model: &PromptCacheModelIdentity,
    prefix: &[u32],
    options: PagedCacheOptions,
    stream: &Stream,
) -> Result<(CacheResidencyManager, PromptCacheManifest), CacheResidencyError> {
    let restore = eredu_runtime::PromptCacheRestore::new(
        read,
        expected,
        model,
        prefix,
        options.block_size_tokens(),
    )?;
    let manifest = restore.manifest();
    let manager = CacheResidencyManager::new(options)?;
    restore.restore_blocks(&mut NativeBlockRestore {
        manager: &manager,
        stream,
        sink_tokens: manifest.sink_tokens,
    })?;
    {
        let mut state = manager.lock()?;
        state.telemetry.report.prompt_cache_loads += 1;
        state.telemetry.report.prompt_cache_bytes +=
            manifest.blocks.iter().map(|b| b.logical_bytes).sum::<u64>();
        state.telemetry.report.imported_retained_shards += manifest.blocks.len() as u64;
        update_report_totals(&mut state);
    }
    Ok((manager, manifest.clone()))
}
/// Native binding produced by a completed state restore.
pub struct LoadedPromptCacheStateTensor {
    /// Architecture-declared state owner.
    pub owner: StateTensorOwner,
    /// Architecture-declared tensor role.
    pub role: StateTensorRole,
    /// Restored native tensor.
    pub array: Array,
}
struct NativeStateRestore<'a> {
    stream: &'a Stream,
    tensors: Vec<LoadedPromptCacheStateTensor>,
}
impl PromptCacheStateRestore for NativeStateRestore<'_> {
    fn transfer_state(
        &mut self,
        state: &PromptCacheStateTensor,
        source: &eredu_runtime::RetainedCacheShard,
    ) -> Result<(), BackendFailure> {
        let array = transfer_prompt_state(state, source, self.stream)
            .map_err(BackendFailure::from_error)?;
        self.tensors.push(LoadedPromptCacheStateTensor {
            owner: state.owner,
            role: state.role,
            array,
        });
        Ok(())
    }
}
/// Completes native fixed-state transfers selected by the portable driver.
pub fn load_prompt_cache_state_tensors(
    read: &dyn PromptCacheRead,
    manifest: &PromptCacheManifest,
    stream: &Stream,
) -> Result<Vec<LoadedPromptCacheStateTensor>, CacheResidencyError> {
    let mut target = NativeStateRestore {
        stream,
        tensors: Vec::new(),
    };
    eredu_runtime::restore_prompt_cache_states(read, manifest, &mut target)?;
    Ok(target.tensors)
}
enum NativeCachePayload {
    Block(HostCacheBlock),
    State(ImmutableHostTransferBuffer),
}
impl PromptCachePayload for NativeCachePayload {
    fn bytes(&self) -> Result<Vec<&[u8]>, BackendFailure> {
        match self {
            Self::Block(block) => block
                .buffers()
                .into_iter()
                .map(|buffer| buffer.as_bytes().map_err(BackendFailure::from_error))
                .collect(),
            Self::State(buffer) => Ok(vec![buffer
                .as_bytes()
                .map_err(BackendFailure::from_error)?]),
        }
    }
}
struct NativeCacheExport<'a> {
    inventory: PromptCacheExportInventory,
    records: Vec<CacheBlockRecord>,
    states: &'a [PromptCacheStateArray<'a>],
    stream: Stream,
}
impl PromptCacheExport for NativeCacheExport<'_> {
    type Payload = NativeCachePayload;
    fn inventory(&self) -> &PromptCacheExportInventory {
        &self.inventory
    }
    fn transfer_block(&self, index: usize) -> Result<NativeCachePayload, BackendFailure> {
        let record = &self.records[index];
        (|| -> Result<_, CacheResidencyError> {
            let host = if let Some(arrays) = record.physical.device_resource() {
                HostCacheBlock::from_device_arrays(arrays, &self.stream)?
            } else if let Some(host) = record.physical.host_resource() {
                host.clone()
            } else {
                load_host_cache_block_direct(
                    record
                        .physical
                        .backing()
                        .expect("backing-only cache record"),
                    record.physical.id().representation,
                )?
            };
            Ok(NativeCachePayload::Block(host))
        })()
        .map_err(BackendFailure::from_error)
    }
    fn transfer_state(&self, index: usize) -> Result<NativeCachePayload, BackendFailure> {
        HostTransferBuffer::copy_from_array(
            self.states[index].array,
            HostTransferPolicy::Transfer,
            &self.stream,
        )
        .and_then(|p| p.synchronize())
        .map(|host| NativeCachePayload::State(host.freeze()))
        .map_err(BackendFailure::from_error)
    }
}
impl CacheResidencyManager {
    /// Exposes native block inventory and completed transfers to the runtime publisher.
    pub fn save_prompt_cache(
        &self,
        store: &dyn PromptCacheStore,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        state_arrays: &[PromptCacheStateArray<'_>],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, CacheResidencyError> {
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
                        state.array.shape().to_vec(),
                        dtype_name(state.array.dtype()),
                    )?,
                })
            })
            .collect::<Result<Vec<_>, PromptCacheStoreError>>()?;
        let mut transfer_bytes = 0;
        for bytes in records
            .iter()
            .map(|r| r.bytes)
            .chain(state_arrays.iter().map(|s| s.array.nbytes() as u64))
        {
            let capacity = host_transfer_capacity_upper_bound(
                usize::try_from(bytes).map_err(|e| CacheResidencyError::Runtime(e.to_string()))?,
                HostTransferPolicy::Transfer,
            )
            .map_err(|e| transfer_error("price native cache transfer", e))?
                as u64;
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
        let source = NativeCacheExport {
            inventory: PromptCacheExportInventory {
                block_size_tokens: self.options().block_size_tokens(),
                blocks,
                states,
                unsealed_streams,
                transfer_bytes,
            },
            records,
            states: state_arrays,
            stream: Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
        };
        let manifest = eredu_runtime::publish_prompt_cache(
            &source,
            store,
            descriptor,
            prefix_token_ids,
            options,
        )?;
        let mut state = self.lock()?;
        state.telemetry.report.prompt_cache_saves += 1;
        state.telemetry.report.prompt_cache_bytes += manifest
            .blocks
            .iter()
            .map(|b| b.logical_bytes)
            .chain(manifest.state_tensors.iter().map(|s| s.logical_bytes))
            .sum::<u64>();
        Ok(manifest)
    }
}
