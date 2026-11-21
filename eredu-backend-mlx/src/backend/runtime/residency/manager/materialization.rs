//! Checkpoint materialization, host transfer, and capacity accounting.

use super::*;
use crate::backend::submission_recovery::{Recovery, Retention, Status};
use eredu_core::checkpoint_mechanisms::ParameterReadInput;

struct HostMaterialization {
    array: Array,
    _sources: Vec<PendingWeightMaterialization>,
    buffer: Option<HostTransferBuffer>,
    event: Option<Event>,
}

impl Retention for HostMaterialization {
    fn observe(&self, _: Status) {}
}

pub(super) fn materialize_host_buffer(
    id: &OffloadUnitId,
    binding: &WeightBinding,
    input: ParameterReadInput<'_>,
    context: &MlxParameterMaterializationContext,
) -> Result<Arc<ResidentHostAllocation>, ResidencyError> {
    let source_stream = context.source_stream();
    let (array, sources) = match input {
        ParameterReadInput::Recipe(recipe, source) => recipe
            .prepare_materialization(source, context)
            .map_err(|source| ResidencyError::Recipe {
                binding: binding.name().to_owned(),
                source,
            })?
            .into_parts(),
        ParameterReadInput::Lease(lease) => {
            let pending = context
                .weight_lease(lease)?
                .prepare_materialization(source_stream, source_stream)?;
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
    retained.retention_mut().event = Some(event);
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
    let allocation = Arc::new(ResidentHostAllocation::new(buffer.freeze())?);
    let status = retained.finish();
    if status.failed || status.blocked {
        return Err(ResidencyError::Mlx {
            id: id.clone(),
            operation: "host materialization retirement",
            source: safemlx::error::Exception::custom("native host materialization failed"),
        });
    }
    Ok(allocation)
}

pub(super) fn prepare_copy_to_device(
    id: &OffloadUnitId,
    names: &[&str],
    host: Arc<ResidentHostBuffers>,
    device_stream: &Stream,
    retained: &mut ResidentTransferResources,
) -> Result<BTreeMap<String, Array>, ResidencyError> {
    let mut arrays = BTreeMap::new();
    retained.retained_host.push(Arc::clone(&host));
    for &name in names {
        let buffer = host
            .buffers
            .get(name)
            .ok_or(ResidencyError::StatePoisoned)?;
        let submitted = buffer
            .buffer
            .copy_to_array(device_stream)
            .map_err(|source| ResidencyError::Mlx {
                id: id.clone(),
                operation: "host-buffer-to-device copy",
                source,
            })?;
        let (array, completion) = submitted.into_parts();
        retained.retained_arrays.push(array.clone());
        arrays.insert(name.to_owned(), array);
        retained.retained_events.push(completion);
    }
    Ok(arrays)
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
            let bytes =
                u64::try_from(
                    buffer
                        .buffer
                        .nbytes()
                        .map_err(|source| ResidencyError::Mlx {
                            id: internal_id(),
                            operation: "host-buffer byte inspection",
                            source,
                        })?,
                )
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
            let bytes =
                u64::try_from(
                    buffer
                        .buffer
                        .capacity()
                        .map_err(|source| ResidencyError::Mlx {
                            id: internal_id(),
                            operation: "host-buffer capacity inspection",
                            source,
                        })?,
                )
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
