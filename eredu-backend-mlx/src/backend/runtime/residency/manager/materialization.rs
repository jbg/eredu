//! Checkpoint materialization, host transfer, and capacity accounting.

use super::*;

pub(super) struct PreparedResidentArrays {
    pub(super) arrays: BTreeMap<String, Array>,
    pub(super) pending_sources: Vec<PendingWeightMaterialization>,
    pub(super) retained_arrays: Vec<Array>,
    pub(super) retained_host: Option<Arc<ResidentHostBuffers>>,
    pub(super) retained_events: Vec<Event>,
    pub(super) direction: TransferDirection,
}

pub(super) fn shared_arrays_for_unit(
    state: &ManagerState,
    _store: &dyn eredu_checkpoint::store::CheckpointSource,
    id: &OffloadUnitId,
) -> Result<BTreeMap<String, Array>, ResidencyError> {
    let shared = state
        .control
        .unit(id)
        .ok_or(ResidencyError::StatePoisoned)?
        .bindings()
        .iter()
        .filter_map(|binding| {
            state
                .control
                .binding_owner(id, binding)
                .map(|(owner_unit, owner)| {
                    (
                        binding.name().to_owned(),
                        (owner_unit.clone(), owner.name().to_owned()),
                        owner.name().to_owned(),
                    )
                })
        })
        .collect::<Vec<_>>();
    let mut arrays = BTreeMap::new();
    for (target, (owner_unit, owner_name), _) in shared {
        if owner_unit == *id {
            continue;
        }
        let array = state
            .storage
            .get(&owner_unit)
            .and_then(|storage| storage.device.as_ref())
            .and_then(|owner| owner.arrays.get(&owner_name))
            .ok_or(ResidencyError::StatePoisoned)?;
        arrays.insert(target, array.clone());
    }
    Ok(arrays)
}

pub(super) fn shared_host_buffers_for_unit(
    state: &ManagerState,
    id: &OffloadUnitId,
) -> Result<BTreeMap<String, Arc<ImmutableHostTransferBuffer>>, ResidencyError> {
    let unit = state
        .control
        .unit(id)
        .ok_or(ResidencyError::StatePoisoned)?;
    let mut shared = BTreeMap::new();
    for binding in unit.bindings().iter().filter(|binding| binding.is_alias()) {
        let (owner_unit, owner) = state
            .control
            .binding_owner(id, binding)
            .ok_or(ResidencyError::StatePoisoned)?;
        if owner_unit == id {
            continue;
        }
        let buffer = state
            .storage
            .get(owner_unit)
            .and_then(|storage| storage.host.as_ref())
            .and_then(|buffers| buffers.buffers.get(owner.name()))
            .ok_or(ResidencyError::StatePoisoned)?;
        shared.insert(binding.name().to_owned(), Arc::clone(buffer));
    }
    Ok(shared)
}

pub(super) fn materialize_host_buffers(
    id: &OffloadUnitId,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    context: &MlxParameterMaterializationContext,
    shared: &BTreeMap<String, Arc<ImmutableHostTransferBuffer>>,
) -> Result<ResidentHostBuffers, ResidencyError> {
    let mut buffers = shared.clone();
    for binding in bindings {
        if buffers.contains_key(binding.name()) {
            continue;
        }
        let (array, sources) = match binding.recipe() {
            Some(recipe) => {
                let pending = recipe
                    .prepare_materialization(store, context)
                    .map_err(|source| ResidencyError::Recipe {
                        binding: binding.name().to_owned(),
                        source,
                    })?;
                pending.into_parts()
            }
            None => {
                let lease = store.acquire_lease(eredu_checkpoint::store::TensorReadRequest {
                    key: binding.checkpoint_key().to_owned(),
                    selection: binding.selection().clone(),
                    policy: eredu_checkpoint::store::ReadPolicy::RequireBounded,
                })?;
                let lease = context.weight_lease(lease)?;
                let pending = lease.prepare_materialization(source_stream, source_stream)?;
                (pending.output().clone(), vec![pending])
            }
        };
        let buffer = HostTransferBuffer::copy_from_array(
            &array,
            HostTransferPolicy::Transfer,
            source_stream,
        )
        .map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "weight array-to-host-buffer submission",
            source,
        })?
        .synchronize()
        .map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "weight array-to-host-buffer completion",
            source,
        })?;
        for source in sources {
            source.complete();
        }
        let actual = u64::try_from(buffer.nbytes().map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "host-buffer byte inspection",
            source,
        })?)
        .map_err(|_| ResidencyError::ArithmeticOverflow {
            context: "host-buffer byte conversion",
        })?;
        if actual != binding.expected_bytes() {
            return Err(ResidencyError::BindingByteMismatch {
                id: id.clone(),
                binding: binding.name().to_owned(),
                expected_bytes: binding.expected_bytes(),
                actual_bytes: actual,
            });
        }
        buffers.insert(binding.name().to_owned(), Arc::new(buffer.freeze()));
    }
    Ok(ResidentHostBuffers { buffers })
}

pub(super) fn prepare_from_disk(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
    context: &MlxParameterMaterializationContext,
    direction: TransferDirection,
    shared: &BTreeMap<String, Array>,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let mut arrays = shared.clone();
    let mut pending_sources = Vec::new();
    let mut retained_arrays = Vec::new();
    for binding in bindings {
        if arrays.contains_key(binding.name()) {
            continue;
        }
        let mut retried_after_capacity = false;
        loop {
            let prepared = (|| match binding.recipe() {
                Some(recipe) => {
                    let pending =
                        recipe
                            .prepare_materialization(store, context)
                            .map_err(|source| ResidencyError::Recipe {
                                binding: binding.name().to_owned(),
                                source,
                            })?;
                    let (host, sources) = pending.into_parts();
                    if execution_stream == source_stream {
                        Ok((host, sources, None))
                    } else {
                        let output = host.copy(execution_stream).map_err(|source| {
                            ResidencyError::Recipe {
                                binding: binding.name().to_owned(),
                                source: WeightRecipeError::Mlx(source),
                            }
                        })?;
                        Ok((output, sources, Some(host)))
                    }
                }
                None => {
                    let lease =
                        store.acquire_lease(eredu_checkpoint::store::TensorReadRequest {
                            key: binding.checkpoint_key().to_owned(),
                            selection: binding.selection().clone(),
                            policy: eredu_checkpoint::store::ReadPolicy::RequireBounded,
                        })?;
                    let lease = context.weight_lease(lease)?;
                    let pending = lease.prepare_materialization(source_stream, execution_stream)?;
                    let output = pending.output().clone();
                    Ok((output, vec![pending], None))
                }
            })();
            match prepared {
                Ok((output, sources, retained)) => {
                    pending_sources.extend(sources);
                    retained_arrays.extend(retained);
                    arrays.insert(binding.name().to_owned(), output);
                    break;
                }
                Err(error)
                    if !retried_after_capacity
                        && !pending_sources.is_empty()
                        && is_shard_cache_capacity_error(&error) =>
                {
                    eval(arrays.values()).map_err(|source| ResidencyError::Mlx {
                        id: internal_id(),
                        operation: "shard-cache-capacity residency evaluation",
                        source,
                    })?;
                    for source in pending_sources.drain(..) {
                        source.complete();
                    }
                    retained_arrays.clear();
                    retried_after_capacity = true;
                }
                Err(error) => return Err(error),
            }
        }
    }
    Ok(PreparedResidentArrays {
        arrays,
        pending_sources,
        retained_arrays,
        retained_host: None,
        retained_events: Vec::new(),
        direction,
    })
}

pub(super) fn is_shard_cache_capacity_error(error: &ResidencyError) -> bool {
    matches!(
        error,
        ResidencyError::CheckpointStore(
            eredu_checkpoint::store::StoreError::CapacityExhausted { .. }
        ) | ResidencyError::Recipe {
            source: WeightRecipeError::CheckpointStore(
                eredu_checkpoint::store::StoreError::CapacityExhausted { .. }
            ),
            ..
        }
    )
}

pub(super) fn prepare_copy_to_device(
    id: &OffloadUnitId,
    host: Arc<ResidentHostBuffers>,
    device_stream: &Stream,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let mut arrays = BTreeMap::new();
    let mut retained_events = Vec::new();
    for (name, buffer) in &host.buffers {
        let submitted =
            buffer
                .copy_to_array(device_stream)
                .map_err(|source| ResidencyError::Mlx {
                    id: id.clone(),
                    operation: "host-buffer-to-device copy",
                    source,
                })?;
        let (array, completion) = submitted.into_parts();
        arrays.insert(name.clone(), array);
        retained_events.push(completion);
    }
    Ok(PreparedResidentArrays {
        arrays,
        pending_sources: Vec::new(),
        retained_arrays: Vec::new(),
        retained_host: Some(host),
        retained_events,
        direction: TransferDirection::HostToDevice,
    })
}

pub(super) fn arrays_nbytes(
    arrays: &BTreeMap<String, Array>,
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            let array = arrays
                .get(binding.name())
                .ok_or(ResidencyError::StatePoisoned)?;
            let bytes =
                u64::try_from(array.nbytes()).map_err(|_| ResidencyError::ArithmeticOverflow {
                    context: "array byte conversion",
                })?;
            total
                .checked_add(bytes)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "resident array byte total",
                })
        })
}

pub(super) fn resident_capacity_requirement(
    bindings: &[WeightBinding],
    planned_bytes: u64,
    tier: MemoryTier,
) -> Result<u64, ResidencyError> {
    if tier != MemoryTier::Host {
        return Ok(planned_bytes);
    }
    host_capacity_upper_bound_for_bindings(bindings)
}

/// Returns the complete charged host-transfer capacity for one atomic unit.
pub fn host_capacity_upper_bound_for_bindings(
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            let logical = usize::try_from(binding.expected_bytes()).map_err(|_| {
                ResidencyError::ArithmeticOverflow {
                    context: "host capacity-bound input conversion",
                }
            })?;
            let capacity =
                host_transfer_capacity_upper_bound(logical, HostTransferPolicy::Transfer).map_err(
                    |source| ResidencyError::Mlx {
                        id: internal_id(),
                        operation: "host-transfer capacity-bound query",
                        source,
                    },
                )?;
            let capacity =
                u64::try_from(capacity).map_err(|_| ResidencyError::ArithmeticOverflow {
                    context: "host capacity-bound output conversion",
                })?;
            total
                .checked_add(capacity)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "host unit capacity bound",
                })
        })
}

pub(super) fn host_buffers_nbytes(
    buffers: &ResidentHostBuffers,
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            let buffer = buffers
                .buffers
                .get(binding.name())
                .ok_or(ResidencyError::StatePoisoned)?;
            let bytes = u64::try_from(buffer.nbytes().map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "host-buffer byte inspection",
                source,
            })?)
            .map_err(|_| ResidencyError::ArithmeticOverflow {
                context: "host-buffer byte conversion",
            })?;
            total
                .checked_add(bytes)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "resident host-buffer byte total",
                })
        })
}

pub(super) fn host_buffers_capacity(
    buffers: &ResidentHostBuffers,
    bindings: &[WeightBinding],
) -> Result<u64, ResidencyError> {
    bindings
        .iter()
        .filter(|binding| !binding.is_alias())
        .try_fold(0u64, |total, binding| {
            let buffer = buffers
                .buffers
                .get(binding.name())
                .ok_or(ResidencyError::StatePoisoned)?;
            let bytes = u64::try_from(buffer.capacity().map_err(|source| ResidencyError::Mlx {
                id: internal_id(),
                operation: "host-buffer capacity inspection",
                source,
            })?)
            .map_err(|_| ResidencyError::ArithmeticOverflow {
                context: "host-buffer capacity conversion",
            })?;
            total
                .checked_add(bytes)
                .ok_or(ResidencyError::ArithmeticOverflow {
                    context: "resident host-buffer capacity total",
                })
        })
}
