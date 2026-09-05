//! Prompt-cache catalog loading and fixed-state restoration.

use super::*;

/// In-memory fixed state supplied when saving a cache snapshot.
pub struct PromptCacheStateArray<'a> {
    /// Layer or model-global owner.
    pub owner: StateTensorOwner,
    /// Semantic role declared by the canonical layout.
    pub role: StateTensorRole,
    /// Array to persist.
    pub array: &'a Array,
}

/// Catalogs a compatible prompt prefix lazily as read-only disk-backed blocks.
pub fn open_prompt_cache(
    directory: impl AsRef<Path>,
    expected: &PromptCacheDescriptor,
    model: &PromptCacheModelIdentity,
    prefix_token_ids: &[u32],
    options: PagedCacheOptions,
) -> Result<(CacheResidencyManager, PromptCacheManifest), CacheResidencyError> {
    validate_prompt_cache_model_identity(expected, model)?;
    let directory = directory.as_ref();
    let cache_root = resolve_prompt_cache_root(directory)?;
    let manifest = inspect_prompt_cache(directory)?;
    manifest.validate_compatibility(expected, prefix_token_ids)?;
    if manifest.block_size_tokens != options.block_size_tokens() {
        return Err(CacheResidencyError::PromptCache(
            PromptCacheError::Incompatible(format!(
                "block size {} does not match requested {}",
                manifest.block_size_tokens,
                options.block_size_tokens()
            )),
        ));
    }
    let manager = CacheResidencyManager::new(options)?;
    {
        let mut state = manager.lock()?;
        for block in &manifest.blocks {
            let id = CacheBlockId {
                session_id: manager.session_id,
                global_layer: block.global_layer,
                representation: block.representation,
                start: block.start,
                end: block.end,
                rank: block.rank,
            };
            let shard = safe_prompt_cache_shard_path(&cache_root, &block.shard)?;
            let buffered = buffer_prompt_cache_shard(&shard)?;
            let record = CacheBlockRecord {
                physical: MlxCacheBlockStorage::disk(
                    id.clone(),
                    DiskLocation {
                        path: shard,
                        first_name: block.first_array.clone(),
                        second_name: block.second_array.clone(),
                        persistent: true,
                        buffered: Some(buffered),
                        payload_sha256: Some(block.payload_sha256.clone()),
                        payload_verification: Arc::new(OnceLock::new()),
                    },
                ),
                bytes: block.logical_bytes,
                shapes: [block.first_shape.clone(), block.second_shape.clone()],
                dtypes: [block.first_dtype.clone(), block.second_dtype.clone()],
                imported: true,
            };
            state
                .lifecycle
                .insert(id.clone(), block.end <= manifest.sink_tokens as i64)?;
            if state.blocks.insert(id.clone(), record).is_some() {
                return Err(CacheLifecycleError::DuplicateBlock(id).into());
            }
        }
        state.telemetry.report.prompt_cache_loads += 1;
        state.telemetry.report.prompt_cache_bytes += manifest
            .blocks
            .iter()
            .map(|block| block.logical_bytes)
            .sum::<u64>();
        state.telemetry.report.imported_buffered_shards += manifest.blocks.len() as u64;
        update_report_totals(&mut state);
    }
    Ok((manager, manifest))
}

/// Materialized non-attention state tensor from a validated prompt cache.
pub struct LoadedPromptCacheStateTensor {
    /// Architecture state owner identified by the prompt-cache manifest.
    pub owner: StateTensorOwner,
    /// Semantic role of the state tensor.
    pub role: StateTensorRole,
    /// Materialized MLX state array.
    pub array: Array,
}

/// Loads all fixed-state tensors after manifest and model compatibility validation.
pub fn load_prompt_cache_state_tensors(
    directory: impl AsRef<Path>,
    manifest: &PromptCacheManifest,
    stream: &Stream,
) -> Result<Vec<LoadedPromptCacheStateTensor>, CacheResidencyError> {
    let root = resolve_prompt_cache_root(directory.as_ref())?;
    let host_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut loaded = Vec::with_capacity(manifest.state_tensors.len());
    for state in &manifest.state_tensors {
        let path = safe_prompt_cache_shard_path(&root, &state.shard)?;
        let actual_hash = hash_prompt_cache_shard_payload(&path)?;
        if actual_hash != state.payload_sha256 {
            return Err(CacheResidencyError::MalformedShard {
                path,
                reason: "fixed-state payload digest does not match the manifest".into(),
            });
        }
        let mut arrays = Array::load_safetensors(&path, &host_stream).map_err(|source| {
            CacheResidencyError::Runtime(format!("load {}: {source}", path.display()))
        })?;
        let array =
            arrays
                .remove(&state.array)
                .ok_or_else(|| CacheResidencyError::MalformedShard {
                    path: path.clone(),
                    reason: format!("missing state array {}", state.array),
                })?;
        if !arrays.is_empty() {
            return Err(CacheResidencyError::MalformedShard {
                path,
                reason: "fixed-state shard contains unexpected arrays".into(),
            });
        }
        let array = array.copy(stream).map_err(|source| {
            CacheResidencyError::Runtime(format!(
                "copy {} to execution stream: {source}",
                path.display()
            ))
        })?;
        eval([&array]).map_err(|source| {
            CacheResidencyError::Runtime(format!(
                "evaluate {} on execution stream: {source}",
                path.display()
            ))
        })?;
        loaded.push(LoadedPromptCacheStateTensor {
            owner: state.owner,
            role: state.role,
            array,
        });
    }
    Ok(loaded)
}

impl CacheResidencyManager {
    /// Writes a completed immutable prefix atomically to a persistent directory.
    pub fn save_prompt_cache(
        &self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        state_arrays: &[PromptCacheStateArray<'_>],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, CacheResidencyError> {
        let destination = destination.as_ref();
        descriptor.validate()?;
        let publication = PromptCachePublication::begin(destination, options.replace_existing())?;
        let temporary = publication.staging_directory().to_path_buf();

        let result = (|| {
            let records = {
                let state = self.lock()?;
                let owned_layers = descriptor.global_layer_start()..descriptor.global_layer_end();
                let mut records = Vec::new();
                for record in state
                    .blocks
                    .values()
                    .filter(|record| owned_layers.contains(&record.physical.id().global_layer))
                {
                    let id = record.physical.id();
                    if state.lifecycle.is_leased(id)? {
                        return Err(CacheLifecycleError::BlockLeased(id.clone()).into());
                    }
                    records.push(record.clone());
                }
                records
            };
            let mut manifest_blocks = Vec::with_capacity(records.len());
            let mut manifest_state = Vec::with_capacity(state_arrays.len());
            let mut logical_bytes = 0u64;
            for (index, record) in records.iter().enumerate() {
                let shard = format!("block-{index:08}.safetensors");
                let shard_path = temporary.join(&shard);
                if let Some(arrays) = record.physical.device_resource() {
                    save_block_arrays(&shard_path, arrays)?;
                } else if let Some(block) = record.physical.host_resource() {
                    save_host_cache_block(&shard_path, block)?;
                } else {
                    let location = record
                        .physical
                        .backing()
                        .expect("backing-only phase owns its disk location");
                    let block = load_host_cache_block_direct(
                        location,
                        record.physical.id().representation,
                    )?;
                    save_host_cache_block(&shard_path, &block)?;
                }
                let payload_sha256 = finalize_prompt_cache_shard(&shard_path)?;
                logical_bytes += record.bytes;
                let names = array_names(record.physical.id().representation);
                manifest_blocks.push(PromptCacheBlock {
                    global_layer: record.physical.id().global_layer,
                    representation: record.physical.id().representation,
                    start: record.physical.id().start,
                    end: record.physical.id().end,
                    rank: record.physical.id().rank,
                    shard,
                    first_array: names.0.into(),
                    second_array: names.1.into(),
                    first_shape: record.shapes[0].clone(),
                    second_shape: record.shapes[1].clone(),
                    first_dtype: record.dtypes[0].clone(),
                    second_dtype: record.dtypes[1].clone(),
                    logical_bytes: record.bytes,
                    payload_sha256,
                });
            }
            for (index, state) in state_arrays.iter().enumerate() {
                let shard = format!("state-{index:08}.safetensors");
                let shard_path = temporary.join(&shard);
                let array_name = "state";
                Array::save_safetensors([(array_name, state.array)], None, &shard_path).map_err(
                    |source| {
                        CacheResidencyError::Runtime(format!(
                            "save {}: {source}",
                            shard_path.display()
                        ))
                    },
                )?;
                let payload_sha256 = finalize_prompt_cache_shard(&shard_path)?;
                let state_bytes = state.array.nbytes() as u64;
                logical_bytes = logical_bytes.checked_add(state_bytes).ok_or_else(|| {
                    CacheResidencyError::PromptCache(PromptCacheError::Malformed(
                        "prompt-cache state byte count overflow".into(),
                    ))
                })?;
                manifest_state.push(PromptCacheStateTensor {
                    owner: state.owner,
                    role: state.role,
                    shard,
                    array: array_name.into(),
                    shape: state.array.shape().to_vec(),
                    dtype: dtype_name(state.array.dtype()),
                    logical_bytes: state_bytes,
                    payload_sha256,
                });
            }
            let manifest = PromptCacheManifest {
                schema_version: PROMPT_CACHE_SCHEMA_VERSION,
                model_family: descriptor.model_family().to_owned(),
                effective_model_type: descriptor.effective_model_type().to_owned(),
                checkpoint_fingerprint: descriptor.checkpoint_fingerprint().to_owned(),
                prefix_content_fingerprint: descriptor.prefix_content_fingerprint().to_owned(),
                architecture_fingerprint: descriptor.architecture_fingerprint().to_owned(),
                layer_count: descriptor.layer_count(),
                global_layer_start: descriptor.global_layer_start(),
                global_layer_end: descriptor.global_layer_end(),
                block_size_tokens: self.options().block_size_tokens(),
                batch_size: descriptor.batch_size(),
                total_prefix_tokens: prefix_token_ids.len(),
                prefix_sha256: prompt_cache_token_fingerprint(prefix_token_ids),
                layer_layout: descriptor.layer_layout().clone(),
                layer_prefix_offsets: descriptor.layer_prefix_offsets().to_vec(),
                state_segments: descriptor.state_segments().to_vec(),
                sink_tokens: descriptor.sink_tokens(),
                topology: descriptor.topology().clone(),
                distributed_commit: descriptor.distributed_commit(),
                application_namespace: options.application_namespace().map(str::to_owned),
                blocks: manifest_blocks,
                state_tensors: manifest_state,
            };
            publication.commit(&manifest)?;
            let mut state = self.lock()?;
            state.telemetry.report.prompt_cache_saves += 1;
            state.telemetry.report.prompt_cache_bytes += logical_bytes;
            Ok(manifest)
        })();

        result
    }
}
