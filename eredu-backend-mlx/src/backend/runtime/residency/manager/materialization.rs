//! Checkpoint materialization, host transfer, and capacity accounting.

use super::*;
mod foreground_disk;
use crate::backend::runtime::checkpoint::recipe::{
    plan_binding_reads, BindingReadBatch, DirectRecipeRead,
};
use crate::backend::submission_recovery::{Recovery, Retention, Status};
pub(crate) use foreground_disk::ForegroundDiskMaterializationError;
pub(super) use foreground_disk::{
    control_bytes as foreground_disk_control_bytes, prepare_background_disk_into,
    prepare_foreground_disk_into,
};
use safemlx::OperationEvent as Event;

struct HostMaterialization {
    array: Array,
    _sources: Vec<PendingWeightMaterialization>,
    buffer: Option<HostTransferBuffer>,
    event: Option<Event>,
}

impl Retention for HostMaterialization {
    fn observe(&self, _: Status) {}
}

pub(super) struct PreparedResidentArrays {
    pub(super) arrays: PreparedArrayValues,
    pub(super) direction: TransferDirection,
}

/// Ordinary map ownership and the original preallocated final Arc share the
/// same borrowed value interface. An original destination never degrades into
/// the ordinary map branch on error.
pub(super) enum PreparedArrayValues {
    Ordinary(NamedArrays),
    Original(super::named_arrays::PreparedNamedArrays),
}
impl From<BTreeMap<String, Array>> for PreparedArrayValues {
    fn from(values: BTreeMap<String, Array>) -> Self {
        Self::Ordinary(values.into())
    }
}
impl std::ops::Deref for PreparedArrayValues {
    type Target = NamedArrays;
    fn deref(&self) -> &NamedArrays {
        match self {
            Self::Ordinary(values) => values,
            Self::Original(ready) => &ready.arrays,
        }
    }
}
impl std::ops::DerefMut for PreparedArrayValues {
    fn deref_mut(&mut self) -> &mut NamedArrays {
        match self {
            Self::Ordinary(values) => values,
            Self::Original(ready) => &mut ready.arrays,
        }
    }
}
impl PreparedArrayValues {
    pub(super) fn validate_publication(
        &mut self,
        manager: Option<&ManagerOwner>,
        unit: &OffloadUnit,
    ) -> Result<(), ResidencyError> {
        match self {
            Self::Ordinary(_) => Ok(()),
            Self::Original(ready) => ready
                .validate_publication(manager.ok_or(NamedArrayError::ForeignManager)?, unit)
                .map_err(Into::into),
        }
    }
    pub(super) fn publish(
        self,
        manager: Option<&ManagerOwner>,
        unit: &OffloadUnit,
    ) -> Result<ResidentArraysOwner, ResidencyError> {
        match self {
            Self::Ordinary(arrays) => Ok(Arc::new(ResidentArrays { arrays }).into()),
            Self::Original(ready) => ready
                .publish(manager.ok_or(NamedArrayError::ForeignManager)?, unit)
                .map_err(|(cause, _retained_destination)| cause.into()),
        }
    }
}

fn store_output(
    arrays: &mut NamedArrays,
    name: &str,
    output: Array,
) -> Result<(), NamedArrayError> {
    // Every caller has retained this same output before a fallible destination
    // write, so rejection cannot release the last pending native descriptor.
    arrays.put(name, output).map_err(|(cause, _output)| cause)
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
) -> Result<BTreeMap<String, RetainedHostBuffer>, ResidencyError> {
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
        shared.insert(binding.name().to_owned(), buffer.clone());
    }
    Ok(shared)
}

// Canonical resolution is supplied by the already validated global controller.
// Restrict this publication map to the current unit; external owners continue
// through the existing shared-array/host-buffer maps and owner pins.
pub(super) fn local_aliases_for_unit(
    state: &ManagerState,
    id: &OffloadUnitId,
) -> Result<BTreeMap<String, String>, ResidencyError> {
    let unit = state
        .control
        .unit(id)
        .ok_or(ResidencyError::StatePoisoned)?;
    let mut aliases = BTreeMap::new();
    for binding in unit.bindings().iter().filter(|binding| binding.is_alias()) {
        let (owner_unit, owner) = state
            .control
            .binding_owner(id, binding)
            .ok_or(ResidencyError::StatePoisoned)?;
        if owner_unit == id {
            aliases.insert(binding.name().to_owned(), owner.name().to_owned());
        }
    }
    Ok(aliases)
}

fn publish_local_aliases<T: Clone>(
    values: &mut BTreeMap<String, T>,
    aliases: &BTreeMap<String, String>,
) -> Result<(), ResidencyError> {
    for (alias, owner) in aliases {
        let value = values
            .get(owner)
            .ok_or(ResidencyError::StatePoisoned)?
            .clone();
        if values.insert(alias.clone(), value).is_some() {
            return Err(ResidencyError::StatePoisoned);
        }
    }
    Ok(())
}

pub(super) fn materialize_host_buffers<'context>(
    id: &OffloadUnitId,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
    shared: &BTreeMap<String, RetainedHostBuffer>,
    local_aliases: &BTreeMap<String, String>,
) -> Result<ResidentHostBuffers, ResidencyError> {
    let context = context.into();
    let mut buffers = shared.clone();
    let batches = plan_binding_reads(
        store,
        bindings
            .iter()
            .filter(|binding| !binding.is_alias() && !shared.contains_key(binding.name())),
    )
    .map_err(|source| ResidencyError::Recipe {
        binding: "<parameter batch>".into(),
        source,
    })?;
    for batch in batches {
        let binding = match batch {
            BindingReadBatch::Direct { bindings, reads } => {
                let outputs = DirectRecipeRead::materialize_host_many(reads).map_err(|source| {
                    ResidencyError::Recipe {
                        binding: "<parameter batch>".into(),
                        source,
                    }
                })?;
                for (binding, buffer) in bindings.into_iter().zip(outputs) {
                    validate_host_buffer_bytes(id, binding, &buffer)?;
                    buffers.insert(binding.name().to_owned(), Arc::new(buffer.freeze()).into());
                }
                continue;
            }
            BindingReadBatch::Ordinary(binding) => binding,
        };
        let (array, sources) = match binding.recipe() {
            Some(recipe) => {
                let pending = crate::backend::runtime::checkpoint::recipe::prepare_ordinary_recipe(
                    recipe, store, context, false,
                )
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
        let mut retained = Recovery::begin(HostMaterialization {
            array,
            _sources: sources,
            buffer: None,
            event: None,
        })
        .map_err(|source| ResidencyError::Mlx {
            id: id.clone(),
            operation: "prepare host materialization recovery",
            source,
        })?;
        let result = HostTransferBuffer::copy_from_array(
            &retained.retention().array,
            HostTransferPolicy::Transfer,
            source_stream,
        );
        retained.seal();
        let (buffer, event) = result
            .map_err(|source| ResidencyError::Mlx {
                id: id.clone(),
                operation: "weight array-to-host-buffer submission",
                source,
            })?
            .into_parts();
        retained.retention_mut().buffer = Some(buffer);
        retained.retention_mut().event = Some(event.into());
        loop {
            let status = retained.progress();
            if status.failed || status.blocked {
                return Err(ResidencyError::Mlx {
                    id: id.clone(),
                    operation: "weight array-to-host-buffer completion",
                    source: safemlx::error::Exception::custom(
                        "native host transfer failed; unresolved resources remain retained",
                    ),
                });
            }
            if status.settled {
                break;
            }
            std::thread::yield_now();
        }
        let buffer = retained
            .retention_mut()
            .buffer
            .take()
            .expect("completed host buffer");
        validate_host_buffer_bytes(id, binding, &buffer)?;
        buffers.insert(binding.name().to_owned(), Arc::new(buffer.freeze()).into());
        let status = retained
            .finish()
            .map_err(ResidencyError::OriginalRetirement)?;
        if status.failed || status.blocked {
            return Err(ResidencyError::Mlx {
                id: id.clone(),
                operation: "host materialization retirement",
                source: safemlx::error::Exception::custom("native host materialization failed"),
            });
        }
    }
    publish_local_aliases(&mut buffers, local_aliases)?;
    Ok(ResidentHostBuffers {
        buffers: buffers.into_iter().collect(),
    })
}

fn validate_host_buffer_bytes(
    id: &OffloadUnitId,
    binding: &WeightBinding,
    buffer: &HostTransferBuffer,
) -> Result<(), ResidencyError> {
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
    Ok(())
}

pub(super) fn prepare_from_disk<'context>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
    context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
    direction: TransferDirection,
    shared: &BTreeMap<String, Array>,
    local_aliases: &BTreeMap<String, String>,
    retained: &mut ResidentTransferResources,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let context = context.into();
    prepare_from_disk_with_operations(
        store,
        bindings,
        source_stream,
        execution_stream,
        context,
        direction,
        shared,
        local_aliases,
        retained,
        None,
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_from_disk_with_operations<'context>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
    context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
    direction: TransferDirection,
    shared: &BTreeMap<String, Array>,
    local_aliases: &BTreeMap<String, String>,
    retained: &mut ResidentTransferResources,
    original: Option<(
        &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
        &safemlx::OriginalScopeObserver,
    )>,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let context = context.into();
    let mut arrays = NamedArrays::Ordinary(shared.clone());
    retained.retained_arrays.extend(shared.values().cloned());
    prepare_from_disk_into(
        store,
        bindings,
        source_stream,
        execution_stream,
        context,
        &mut arrays,
        retained,
        original,
    )?;
    let NamedArrays::Ordinary(mut arrays) = arrays else {
        unreachable!("ordinary adapter")
    };
    publish_local_aliases(&mut arrays, local_aliases)?;
    Ok(PreparedResidentArrays {
        arrays: arrays.into(),
        direction,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_from_disk_into<'context>(
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    bindings: &[WeightBinding],
    source_stream: &Stream,
    execution_stream: &Stream,
    context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
    arrays: &mut NamedArrays,
    retained: &mut ResidentTransferResources,
    mut original: Option<(
        &mut crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots<'_>,
        &safemlx::OriginalScopeObserver,
    )>,
) -> Result<(), ResidencyError> {
    let context = context.into();
    let batches = plan_binding_reads(
        store,
        bindings
            .iter()
            .filter(|binding| !binding.is_alias() && !arrays.contains_key(binding.name())),
    )
    .map_err(|source| ResidencyError::Recipe {
        binding: "<parameter batch>".into(),
        source,
    })?;
    for batch in batches {
        let binding = match batch {
            BindingReadBatch::Direct { bindings, reads } => {
                let inputs = DirectRecipeRead::materialize_many(reads).map_err(|source| {
                    ResidencyError::Recipe {
                        binding: "<parameter batch>".into(),
                        source,
                    }
                })?;
                // Arm the enclosing recovery owner before any stream operation.
                retained.retained_arrays.extend(inputs.iter().cloned());
                for (binding, input) in bindings.into_iter().zip(inputs) {
                    let output = if source_stream == execution_stream {
                        input
                    } else {
                        input
                            .copy(execution_stream)
                            .map_err(|source| ResidencyError::Recipe {
                                binding: binding.name().to_owned(),
                                source: WeightRecipeError::Mlx(source),
                            })?
                    };
                    retained.retained_arrays.push(output.clone());
                    store_output(arrays, binding.name(), output)?;
                }
                continue;
            }
            BindingReadBatch::Ordinary(binding) => binding,
        };
        let mut retried_after_capacity = false;
        loop {
            let prepared = (|| match binding.recipe() {
                Some(recipe) => {
                    let pending =
                        crate::backend::runtime::checkpoint::recipe::prepare_ordinary_recipe_with_operations(
                            recipe, store, context, false,
                            original.as_mut().map(|(slots, observer)| (&mut **slots, *observer)),
                        )
                        .map_err(|source| ResidencyError::Recipe {
                            binding: binding.name().to_owned(),
                            source,
                        })?;
                    let (host, sources) = pending.into_parts();
                    retained.sources.extend(sources);
                    retained.retained_arrays.push(host.clone());
                    if execution_stream == source_stream {
                        Ok(host)
                    } else {
                        let output = host.copy(execution_stream).map_err(|source| {
                            ResidencyError::Recipe {
                                binding: binding.name().to_owned(),
                                source: WeightRecipeError::Mlx(source),
                            }
                        })?;
                        Ok(output)
                    }
                }
                None => {
                    let pending = match original.as_mut() {
                        Some((slots, observer)) => slots
                            .acquire_lease(
                                store,
                                binding.checkpoint_key(),
                                binding.selection(),
                                observer,
                            )?
                            .prepare(
                                context,
                                source_stream,
                                execution_stream,
                                slots,
                                observer,
                                false,
                            )?,
                        None => {
                            let lease = store.acquire_lease(
                                eredu_checkpoint::store::TensorReadRequest {
                                    key: binding.checkpoint_key().to_owned(),
                                    selection: binding.selection().clone(),
                                    policy: eredu_checkpoint::store::ReadPolicy::RequireBounded,
                                },
                            )?;
                            context
                                .weight_lease(lease)?
                                .prepare_materialization(source_stream, execution_stream)?
                        }
                    };
                    let output = pending.output().clone();
                    retained.sources.push(pending);
                    Ok(output)
                }
            })();
            match prepared {
                Ok(output) => {
                    retained.retained_arrays.push(output.clone());
                    store_output(arrays, binding.name(), output)?;
                    break;
                }
                Err(error)
                    if !retried_after_capacity
                        && !retained.sources.is_empty()
                        && is_shard_cache_capacity_error(&error) =>
                {
                    // Original errors can retain a failed cache lease. Release
                    // that exact owner before detaching prior successes/retrying.
                    let _ordinary_error = original.is_none().then_some(error);
                    let prior = match original.as_mut() {
                        Some((slots, observer)) => WeightMaterialization::detach_with_operations(
                            &retained.retained_arrays,
                            &mut retained.sources,
                            slots,
                            observer,
                        )?,
                        None => WeightMaterialization::prepare_retained(
                            Vec::new(),
                            std::mem::take(&mut retained.sources),
                        )?
                        .submit_outputs(retained.retained_arrays.clone())?,
                    };
                    prior.wait()?;
                    prior.finish()?;
                    retried_after_capacity = true;
                }
                Err(error) => return Err(error),
            }
        }
    }
    Ok(())
}

pub(super) fn is_shard_cache_capacity_error(error: &ResidencyError) -> bool {
    let prepared = match error {
        ResidencyError::CheckpointMaterialization(
            CheckpointMaterializationError::PreparedAcquisition(error),
        ) => error.store_error(),
        ResidencyError::Recipe {
            source:
                WeightRecipeError::CheckpointMaterialization(
                    CheckpointMaterializationError::PreparedAcquisition(error),
                ),
            ..
        } => error.store_error(),
        _ => None,
    };
    matches!(
        prepared,
        Some(eredu_checkpoint::store::StoreError::CapacityExhausted { .. })
    ) || matches!(
        error,
        ResidencyError::CheckpointStore(
            eredu_checkpoint::store::StoreError::CapacityExhausted { .. }
        ) | ResidencyError::Recipe {
            source: WeightRecipeError::CheckpointStore(
                eredu_checkpoint::store::StoreError::CapacityExhausted { .. }
            ),
            ..
        } | ResidencyError::Recipe {
            source: WeightRecipeError::Neutral(eredu_checkpoint::recipe::RecipeError::Store(
                eredu_checkpoint::store::StoreError::CapacityExhausted { .. }
            )),
            ..
        }
    )
}

pub(super) fn prepare_copy_to_device(
    id: &OffloadUnitId,
    bindings: &[WeightBinding],
    host: ResidentHostOwner,
    device_stream: &Stream,
    shared: &BTreeMap<String, Array>,
    local_aliases: &BTreeMap<String, String>,
    retained: &mut ResidentTransferResources,
    original: Option<&safemlx::OriginalScopeObserver>,
) -> Result<PreparedResidentArrays, ResidencyError> {
    let mut arrays = NamedArrays::Ordinary(shared.clone());
    retained.retained_arrays.extend(shared.values().cloned());
    prepare_copy_to_device_into(
        id,
        bindings,
        host,
        device_stream,
        &mut arrays,
        retained,
        original,
    )?;
    let NamedArrays::Ordinary(mut arrays) = arrays else {
        unreachable!("ordinary adapter")
    };
    publish_local_aliases(&mut arrays, local_aliases)?;
    Ok(PreparedResidentArrays {
        arrays: arrays.into(),
        direction: TransferDirection::HostToDevice,
    })
}

/// The shared original copy worker uses fixed typed failures; no error embeds
/// ResidencyError recursively and every submitted prefix stays in retention.
#[derive(Debug, thiserror::Error)]
pub(super) enum OriginalHostCopyCause {
    #[error("original host copy native failure: {0}")]
    Native(#[from] safemlx::error::Exception),
    #[error("original host copy destination: {0}")]
    Named(#[from] NamedArrayError),
    #[error("original host copy scope or prepared capacity mismatch")]
    Domain,
}
impl From<OriginalHostCopyCause> for ResidencyError {
    fn from(cause: OriginalHostCopyCause) -> Self {
        match cause {
            OriginalHostCopyCause::Native(cause) => Self::OriginalNative(cause),
            OriginalHostCopyCause::Named(cause) => Self::OriginalNamedDestination(cause),
            OriginalHostCopyCause::Domain => Self::OriginalOperationDomain,
        }
    }
}
pub(crate) fn original_host_copy_control_bytes() -> Option<usize> {
    [
        NamedArrays::host_source_control_bytes()?,
        std::mem::size_of::<OriginalHostCopyCause>(),
        std::mem::size_of::<Result<(), OriginalHostCopyCause>>(),
        std::mem::size_of::<Result<(), ResidencyError>>(),
        std::mem::size_of::<ResidentHostOwner>(),
        std::mem::size_of::<&[WeightBinding]>(),
        std::mem::size_of::<usize>(),
        std::mem::size_of::<&mut ResidentTransferResources>(),
        std::mem::size_of::<&Stream>(),
        std::mem::size_of::<&mut NamedArrays>(),
        std::mem::size_of::<&safemlx::OriginalScopeObserver>(),
        std::mem::size_of::<safemlx::OriginalScopeObserver>(),
        std::mem::size_of::<Result<safemlx::OriginalScopeObserver, safemlx::error::Exception>>(),
        std::mem::size_of::<std::slice::Iter<'_, WeightBinding>>(),
        std::mem::size_of::<&WeightBinding>(),
        std::mem::size_of::<&str>(),
        std::mem::size_of::<&RetainedHostBuffer>(),
        std::mem::size_of::<Array>(),
        std::mem::size_of::<Event>(),
        std::mem::size_of::<Result<(Array, Event), safemlx::error::Exception>>(),
        std::mem::size_of::<Result<Array, safemlx::error::Exception>>(),
        std::mem::size_of::<Result<(), NamedArrayError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
fn prepare_original_host_copy(
    bindings: &[WeightBinding],
    host: ResidentHostOwner,
    stream: &Stream,
    arrays: &mut NamedArrays,
    retained: &mut ResidentTransferResources,
    observer: &safemlx::OriginalScopeObserver,
) -> Result<(), OriginalHostCopyCause> {
    let current = safemlx::OriginalScopeObserver::require_current()?;
    let count = bindings
        .iter()
        .filter(|binding| !binding.is_alias() && !arrays.contains_key(binding.name()))
        .count();
    if !current.same_scope(observer)
        || retained.retained_host.len() == retained.retained_host.capacity()
        || retained.retained_arrays.capacity() - retained.retained_arrays.len() < count
        || retained.retained_events.capacity() - retained.retained_events.len() < count
        || bindings
            .iter()
            .filter(|binding| !binding.is_alias() && !arrays.contains_key(binding.name()))
            .any(|binding| host.buffers.get(binding.name()).is_none())
    {
        return Err(OriginalHostCopyCause::Domain);
    }
    // Preflight every source cell before the first source alias or native copy.
    for binding in bindings.iter().filter(|binding| !binding.is_alias()) {
        if !arrays.contains_key(binding.name()) {
            arrays.validate_host_source(binding.name())?;
        }
    }
    retained.retained_host.push(host.clone());
    for binding in bindings.iter().filter(|binding| !binding.is_alias()) {
        let name = binding.name();
        if arrays.contains_key(name) {
            continue;
        }
        let buffer = host
            .buffers
            .get(name)
            .expect("validated complete host result");
        arrays.retain_host_source(name, buffer)?;
        let (array, event) = buffer.copy_to_array_in_original_scope(stream, observer)?;
        // Arm both owners before a fallible C-shell clone or destination write.
        retained.retained_arrays.push(array);
        retained.retained_events.push(event);
        let output = retained
            .retained_arrays
            .last()
            .expect("retained output")
            .try_clone_handle()?;
        store_output(arrays, name, output)?;
    }
    Ok(())
}

pub(super) fn prepare_copy_to_device_into(
    id: &OffloadUnitId,
    bindings: &[WeightBinding],
    host: ResidentHostOwner,
    device_stream: &Stream,
    arrays: &mut NamedArrays,
    retained: &mut ResidentTransferResources,
    original: Option<&safemlx::OriginalScopeObserver>,
) -> Result<(), ResidencyError> {
    if let Some(observer) = original {
        return prepare_original_host_copy(
            bindings,
            host,
            device_stream,
            arrays,
            retained,
            observer,
        )
        .map_err(ResidencyError::from);
    }
    retained.retained_host.push(host.clone());
    for binding in bindings.iter().filter(|binding| !binding.is_alias()) {
        let name = binding.name();
        if arrays.contains_key(name) {
            continue;
        }
        let buffer = host
            .buffers
            .get(name)
            .ok_or(ResidencyError::StatePoisoned)?;
        if arrays.is_prepared() {
            arrays.retain_host_source(name, buffer)?;
        }
        let submitted = match original {
            Some(observer) => buffer
                .copy_to_array_in_original_scope(device_stream, observer)
                .map_err(ResidencyError::OriginalNative),
            None => buffer
                .copy_to_array(device_stream)
                .map(|value| {
                    let (array, event) = value.into_parts();
                    (array, event.into())
                })
                .map_err(|source| ResidencyError::Mlx {
                    id: id.clone(),
                    operation: "host-buffer-to-device copy",
                    source,
                }),
        }?;
        let (array, completion) = submitted;
        retained.retained_arrays.push(array.clone());
        retained.retained_events.push(completion);
        store_output(arrays, name, array)?;
    }
    Ok(())
}

pub(super) fn arrays_nbytes(
    arrays: &NamedArrays,
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

mod operation_slots;
pub(crate) use operation_slots::PreparedHostMaterialization;
