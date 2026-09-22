//! Physical populations retained from the selected child allocation workers.
use super::*;
use eredu_nn::workspace::{WorkspaceAllocationPopulation, WorkspaceScratchAllocation};

/// Native communication workers execute on CPU and use one established allocator
/// route. Only this homogeneous source may be represented by a single row;
/// numerical children retain their individual allocation descriptors instead.
pub(super) fn native_cpu(
    context: &WorkspaceContext,
    mechanism: ResidentExecutionMechanisms,
    bytes: usize,
    births: usize,
) -> Result<WorkspaceAllocationPopulation, Error> {
    context.charge_metadata(std::mem::size_of::<(
        WorkspaceScratchAllocation,
        Result<WorkspaceAllocationPopulation, Error>,
    )>())?;
    let allocation = mechanism.allocation();
    let placement = if allocation.original_storage {
        crate::backend::managed_memory::cold_original_allocator_placement()
    } else {
        crate::backend::managed_memory::cold_default_placement()
    };
    let placement_metadata = match placement.map(eredu_core::MemoryPlacement::kind) {
        Some(eredu_core::MemoryPlacementKind::Possible { domains, basis }) => {
            std::alloc::Layout::array::<eredu_core::MemoryDomainId>(domains.len())
                .map_err(|_| WorkspaceMetadataError::Overflow)?
                .size()
                .checked_add(basis.len())
                .ok_or(WorkspaceMetadataError::Overflow)?
        }
        _ => 0,
    };
    context.charge_metadata(placement_metadata)?;
    let controls = allocation
        .host_control_bytes()
        .map(|bytes| {
            u64::try_from(births)
                .ok()
                .and_then(|count| count.checked_mul(bytes))
                .ok_or(WorkspaceMetadataError::Overflow)
        })
        .transpose()?;
    context.source_scratch_population(&[WorkspaceScratchAllocation {
        bytes: u64::try_from(bytes).map_err(|_| WorkspaceMetadataError::Overflow)?,
        maximum_allocations: Some(births),
        host_control_bytes: controls,
        placement: placement.cloned(),
    }])
}

/// An ordinary lazy Ring contribution can cross between the selected Metal
/// model stream and its CPU communication stream in the same Eval. Each
/// producing stream owns at most one fast Fence U32 backing; the slow Event
/// alternative has no tensor backing. Both use the backend's default allocator,
/// independently of the stream which consumes the value.
pub(super) fn ring_fences(
    context: &WorkspaceContext,
    mechanism: ResidentExecutionMechanisms,
) -> Result<Option<WorkspaceAllocationPopulation>, Error> {
    context.charge_metadata(std::mem::size_of::<(
        &WorkspaceContext,
        ResidentExecutionMechanisms,
        u64,
        usize,
        Option<WorkspaceAllocationPopulation>,
        Result<Option<WorkspaceAllocationPopulation>, Error>,
    )>())?;
    if !matches!(mechanism, ResidentExecutionMechanisms::Metal(_))
        || mechanism.allocation().original_storage
    {
        return Ok(None);
    }
    let bytes = mechanism
        .allocation()
        .buffer_capacity(std::mem::size_of::<u32>() as u64)?
        .checked_mul(2)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(WorkspaceMetadataError::Overflow)?;
    native_cpu(context, mechanism, bytes, 2).map(Some)
}

pub(super) fn with_ring_fences(
    context: &WorkspaceContext,
    mechanism: ResidentExecutionMechanisms,
    scratch: WorkspaceAllocationPopulation,
) -> Result<WorkspaceAllocationPopulation, Error> {
    context.charge_metadata(std::mem::size_of::<(
        &WorkspaceContext,
        ResidentExecutionMechanisms,
        WorkspaceAllocationPopulation,
        Option<WorkspaceAllocationPopulation>,
        [(&WorkspaceAllocationPopulation, usize); 2],
        Result<WorkspaceAllocationPopulation, Error>,
    )>())?;
    match ring_fences(context, mechanism)? {
        Some(fences) => simultaneous(context, &[(&scratch, 1), (&fences, 1)]),
        None => Ok(scratch),
    }
}

pub(super) fn alternatives(
    context: &WorkspaceContext,
    alternatives: &[Option<WorkspaceAllocationPopulation>],
) -> Result<Option<WorkspaceAllocationPopulation>, Error> {
    let mut sources = context.metadata_vec(alternatives.len())?;
    for source in alternatives {
        let Some(source) = source else {
            return Ok(None);
        };
        sources.push(source);
    }
    context.peak_scratch_populations(&sources).map(Some)
}

pub(super) fn simultaneous(
    context: &WorkspaceContext,
    sources: &[(&WorkspaceAllocationPopulation, usize)],
) -> Result<WorkspaceAllocationPopulation, Error> {
    context.combine_scratch_populations(sources)
}

pub(super) fn itinerary(
    context: &WorkspaceContext,
    mechanism: ResidentExecutionMechanisms,
    counts: &ExpertCountQuote,
    transport: &ExpertTransportQuote,
    provider: &ExpertProviderQuote,
) -> Result<Option<WorkspaceAllocationPopulation>, Error> {
    let Some(counts) = counts.allocation_scratch(context, mechanism)? else {
        return Ok(None);
    };
    let Some(provider) = provider.allocation_scratch(context, mechanism)? else {
        return Ok(None);
    };
    let mut sources = context.metadata_vec(
        transport
            .scratch()
            .len()
            .checked_add(2)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    sources.push((&counts, 1));
    sources.push((&provider, 1));
    for source in transport.scratch() {
        let Some(source) = source else {
            return Ok(None);
        };
        sources.push((source, 1));
    }
    simultaneous(context, &sources).map(Some)
}

pub(super) fn region(
    context: &WorkspaceContext,
    local: &ExpertLocalQuote,
) -> Result<Option<WorkspaceAllocationPopulation>, Error> {
    let Some(child) = &local.local_scratch else {
        return Ok(None);
    };
    let (Some(counts), Some(transport), Some(provider), Some(movement)) = (
        &local.counts,
        &local.transport,
        &local.provider,
        &local.movement,
    ) else {
        return Ok(None);
    };
    let Some(itinerary) = itinerary(context, local.mechanism, counts, transport, provider)? else {
        return Ok(None);
    };
    let mut populations = context.metadata_vec(4)?;
    let counts = local.declaration.as_view().movement.counts();
    for (kind, occurrences) in counts.into_iter().enumerate() {
        if occurrences == 0 {
            continue;
        }
        let count = movement
            .per_kind_scratch(kind)
            .try_fold(0usize, |n, rows| n.checked_add(rows.len()))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut alternatives = context.metadata_vec(count)?;
        for rows in movement.per_kind_scratch(kind) {
            for source in rows {
                let Some(source) = source else {
                    return Ok(None);
                };
                alternatives.push(source);
            }
        }
        populations.push((
            context.peak_scratch_populations(&alternatives)?,
            occurrences,
        ));
    }
    let mut sources = context.metadata_vec(
        populations
            .len()
            .checked_add(2)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    sources.push((child, 1));
    sources.push((&itinerary, 1));
    sources.extend(populations.iter().map(|(source, count)| (source, *count)));
    simultaneous(context, &sources).map(Some)
}
