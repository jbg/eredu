//! Separate physical and metadata contributions of the actual prepared-leaf
//! worker. Completed leaf copies, the equation and final detachment each have
//! their own real completion frontier; the final source buffer is quoted by
//! the enclosing static Host constructor.
use super::*;
use crate::backend::{
    managed_memory,
    nn::workspace::{NativeAllocationFacts, OrdinaryNativeControls},
};
use eredu_core::DomainMemoryRequirements;
use eredu_runtime::working_memory::{StorageMetadataFunding, WorkingMemoryFundingScope};
use safemlx::PreparedInputRuntime;
use std::mem::size_of_val;

pub(in crate::backend::runtime::residency::manager) struct Quotation {
    pub(in crate::backend::runtime::residency::manager) requirements: DomainMemoryRequirements,
    pub(in crate::backend::runtime::residency::manager) metadata: usize,
}

fn known<T>(value: Option<T>) -> Result<T, WorkingMemoryError> {
    value.ok_or(WorkingMemoryError::UnknownBound)
}
fn add(value: &mut u64, bytes: u64) -> Result<(), WorkingMemoryError> {
    *value = value
        .checked_add(bytes)
        .ok_or(WorkingMemoryError::Overflow)?;
    Ok(())
}
fn n(value: usize) -> Result<u64, WorkingMemoryError> {
    u64::try_from(value).map_err(|_| WorkingMemoryError::Overflow)
}

pub(in crate::backend::runtime::residency::manager::construction) fn quote(
    plan: &read_source_plan::MaterializedReadPlan,
    pool: &MemoryLedger,
    runtime: &PreparedInputRuntime,
    allocation: NativeAllocationFacts,
    source_stream: &Stream,
    funding: &eredu_nn::workspace::HostMetadataFunding,
) -> Result<Quotation, WorkingMemoryError> {
    let stream = safemlx::StreamCopyPlan::<()>::capture(source_stream)
        .map_err(|_| WorkingMemoryError::UnknownBound)?;
    if stream.device_type() != DeviceType::Cpu {
        return Err(WorkingMemoryError::UnknownBound);
    }
    let placement = known(managed_memory::cold_copy_placement(
        stream.device_type(),
        stream.device_index(),
    ))?;
    let root_metadata = managed_memory::ordinary_root_metadata_bytes()?;
    let backing_controls = n(safemlx::physical_backing_control_bytes())?
        .checked_add(root_metadata)
        .ok_or(WorkingMemoryError::Overflow)?;
    let (_, host_placement) = HostTransferBuffer::ordinary_capacity(runtime, 0)
        .map_err(|_| WorkingMemoryError::UnknownBound)?;
    let host_placement = known(managed_memory::cold_placement_fact(host_placement))?;
    let managed = |placement: &eredu_core::MemoryPlacement| {
        usize::from(matches!(
            placement.kind(),
            eredu_core::MemoryPlacementKind::Possible { .. }
        ))
    };
    let host_count = plan
        .leaves
        .len()
        .checked_add(1)
        .ok_or(WorkingMemoryError::Overflow)?;
    let allowance_count = managed(host_placement)
        .checked_mul(host_count)
        .and_then(|count| count.checked_add(managed(placement)))
        .ok_or(WorkingMemoryError::Overflow)?;
    let descriptors =
        DomainMemoryRequirements::construction_backing_bytes(pool.topology(), allowance_count)?
            .checked_add(
                host_placement
                    .backing_bytes()?
                    .checked_mul(n(managed(host_placement)
                        .checked_mul(host_count)
                        .ok_or(WorkingMemoryError::Overflow)?)?)
                    .ok_or(WorkingMemoryError::Overflow)?,
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    placement
                        .backing_bytes()
                        .ok()?
                        .checked_mul(n(managed(placement)).ok()?)?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    n(size_of::<(Quotation, Result<Quotation, WorkingMemoryError>)>()).ok()?,
                )
            })
            .ok_or(WorkingMemoryError::Overflow)?;
    funding
        .reserve_metadata(usize::try_from(descriptors).map_err(|_| WorkingMemoryError::Overflow)?)
        .map_err(|cause| {
            WorkingMemoryError::MetadataConstruction(
                eredu_nn::workspace::WorkspaceMetadataError::Funding(cause),
            )
        })?;
    let mut requirements =
        DomainMemoryRequirements::zero_with_allowance_capacity(pool.topology(), allowance_count);
    let mut native = plan.numerical.storage.mutable_bytes();
    let mut host = backing_controls
        .checked_mul(n(plan.numerical.storage.maximum_births())?)
        .ok_or(WorkingMemoryError::Overflow)?;
    let mut observed = known(plan.numerical.ordinary_cpu_controls())?
        .append(plan.wrapper_observed)
        .ok_or(WorkingMemoryError::Overflow)?;
    // The equation has one explicit asynchronous caller with one final root.
    // Scalar validation uses Array::evaluated and has no C ArrayVector; leaf
    // load and final Host detach callers are included by their transfer source.
    known(observed.include(known(
        OperationEvent::ordinary_array_vector_control_layout(1),
    )?))?;
    let mut metadata = plan.wrapper_metadata;
    add(
        &mut metadata,
        n(known(
            crate::backend::runtime::checkpoint::recipe::prepared_recipe_entry_control_bytes(),
        )?)?,
    )?;
    add(&mut metadata, n(known(crate::backend::runtime::checkpoint::store::WeightMaterialization::prepared_validation_control_bytes())?)?
        .checked_mul(n(plan.validations)?).ok_or(WorkingMemoryError::Overflow)?)?;
    add(
        &mut metadata,
        known(PreparedRecovery::<Resources, HostPreparationAuthority>::control_bytes())?,
    )?;
    let capacity = plan
        .leaves
        .len()
        .checked_add(2)
        .ok_or(WorkingMemoryError::Overflow)?;
    for layout in [
        Layout::array::<ImmutableHostTransferBuffer>(plan.leaves.len()),
        Layout::array::<Array>(capacity),
        Layout::array::<OperationEvent>(capacity),
    ] {
        add(
            &mut metadata,
            n(layout.map_err(|_| WorkingMemoryError::Overflow)?.size())?,
        )?;
    }
    for leaf in &plan.leaves {
        host_buffer(
            leaf.read.encoded().output().byte_len(),
            leaf.read.shape().len(),
            runtime,
            pool,
            &mut requirements,
            &mut host,
            &mut metadata,
        )?;
        add(
            &mut native,
            allocation
                .fixed_buffer_capacity(leaf.read.encoded().output().byte_len())
                .map_err(|_| WorkingMemoryError::UnknownBound)?,
        )?;
        add(&mut host, backing_controls)?;
        observed = observed
            .append(known(ResidencyManager::ordinary_host_transfer_controls(
                leaf.read.dtype(),
                leaf.read.shape().len(),
                false,
            ))?)
            .ok_or(WorkingMemoryError::Overflow)?;
        add(
            &mut metadata,
            n(known(
                ImmutableHostTransferBuffer::ordinary_copy_wrapper_control_bytes(),
            )?)?,
        )?;
        add(
            &mut metadata,
            n(known(Array::ordinary_clone_control_bytes())?)?,
        )?;
    }
    host_buffer(
        plan.output.byte_len(),
        plan.shape.len(),
        runtime,
        pool,
        &mut requirements,
        &mut host,
        &mut metadata,
    )?;
    observed = observed
        .append(known(ResidencyManager::ordinary_host_transfer_controls(
            plan.dtype,
            plan.shape.len(),
            true,
        ))?)
        .ok_or(WorkingMemoryError::Overflow)?;
    add(&mut host, observed.host_allowance_with_ledger_metadata()?)?;
    add(
        &mut metadata,
        n(known(
            HostTransferBuffer::ordinary_detach_wrapper_control_bytes(),
        )?)?,
    )?;
    add(
        &mut metadata,
        n(known(
            OperationEvent::ordinary_submission_wrapper_control_bytes(),
        )?)?,
    )?;
    let frames = [
        size_of::<Resources>(),
        size_of::<Failure>(),
        size_of::<Result<(), Failure>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<Option<Failure>>(),
        size_of::<Option<eredu_checkpoint::store::DetachedReadFailure<ManagerCustody>>>(),
        size_of::<MaterializationView<'_>>(),
        size_of::<(&str, &eredu_checkpoint::store::TensorSelection, &Stream)>(),
        size_of::<Result<Array, WeightRecipeError>>(),
        size_of::<HostTransferBuffer>(),
        size_of::<[&mut [u8]; 1]>(),
        size_of::<(&[u8], &mut [u8])>(),
        size_of::<Result<&[u8], safemlx::error::Exception>>(),
        size_of::<Result<&mut [u8], safemlx::error::Exception>>(),
    ];
    add(
        &mut metadata,
        n(frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?)?,
    )?;
    requirements.add_allocation(native, placement)?;
    requirements.add_allocation(host, pool.host_placement())?;
    Ok(Quotation {
        requirements,
        metadata: usize::try_from(metadata).map_err(|_| WorkingMemoryError::Overflow)?,
    })
}

fn host_buffer(
    bytes: u64,
    rank: usize,
    runtime: &PreparedInputRuntime,
    pool: &MemoryLedger,
    requirements: &mut DomainMemoryRequirements,
    host: &mut u64,
    metadata: &mut u64,
) -> Result<(), WorkingMemoryError> {
    let (capacity, placement) = HostTransferBuffer::ordinary_capacity(
        runtime,
        usize::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)?,
    )
    .map_err(|_| WorkingMemoryError::UnknownBound)?;
    requirements.add_allocation(
        n(capacity)?,
        known(managed_memory::cold_placement_fact(placement))?,
    )?;
    add(
        host,
        n(known(HostTransferBuffer::ordinary_observed_control_bytes(
            runtime, rank,
        ))?)?,
    )?;
    add(host, managed_memory::ordinary_root_metadata_bytes()?)?;
    add(
        metadata,
        n(known(
            HostTransferBuffer::ordinary_constructor_control_bytes(rank),
        )?)?,
    )?;
    let _ = pool;
    Ok(())
}

/// One observer and one prepaid Host owner serve the complete constructor.
pub(in crate::backend::runtime::residency::manager) fn with_owner(
    requirements: &mut DomainMemoryRequirements,
    pool: &MemoryLedger,
    metadata: usize,
) -> Result<usize, WorkingMemoryError> {
    let metadata = metadata
        .checked_add(
            usize::try_from(managed_memory::scoped_observer_bytes()?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
        )
        .ok_or(WorkingMemoryError::Overflow)?;
    let owner = StorageMetadataFunding::host_owner_bytes(metadata).map_err(|cause| {
        WorkingMemoryError::MetadataConstruction(
            eredu_nn::workspace::WorkspaceMetadataError::Funding(cause),
        )
    })?;
    let controls = n(owner)?
        .checked_add(MemoryLedger::storage_metadata_control_bytes()?)
        .and_then(|bytes| {
            bytes.checked_add(WorkingMemoryFundingScope::allocation_funding_control_bytes().ok()?)
        })
        .ok_or(WorkingMemoryError::Overflow)?;
    requirements.add_allocation(controls, pool.host_placement())?;
    Ok(metadata)
}
