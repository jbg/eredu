//! Source-tagged child allocations from the selected native fact emitters.
use super::*;
use crate::backend::nn::workspace::routing::selector::{self, AllocationSource};
use eredu_core::{MemoryPlacement, MemoryPlacementKind};
use eredu_nn::workspace::*;

pub(super) enum Description {
    Router(WorkspaceOperationFacts, selector::AllocationSources),
    Default(
        WorkspaceOperationFacts,
        super::super::facts::DefaultScratchSources,
    ),
}

pub(super) struct AllocationPopulations {
    scratch: WorkspaceAllocationPopulation,
    generic: Option<super::super::facts::DefaultScratchSources>,
    outputs: [Option<WorkspaceAllocationPopulation>; 6],
}
impl AllocationPopulations {
    pub(super) fn into_operation_sources(
        self,
        output_count: usize,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceOperationAllocationSources, Error> {
        let mut outputs = context.metadata_vec(output_count.min(self.outputs.len()))?;
        outputs.extend(self.outputs.into_iter().take(output_count));
        Ok(WorkspaceOperationAllocationSources {
            scratch: Some(self.scratch),
            outputs,
        })
    }

    pub(super) fn prepare(
        mechanism: ResidentExecutionMechanisms,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
    ) -> Result<Option<Self>, ResidentFactError> {
        let funding = context.metadata_funding();
        let ResidentExecutionMechanisms::Metal(native) = mechanism else {
            return Ok(None);
        };
        let description = if matches!(
            operation.kind,
            WorkspaceOperationKindView::GroupSelection { .. }
        ) {
            let Some((facts, source)) =
                selector::allocation_sources(operation, native.allocation())?
            else {
                return Ok(None);
            };
            Description::Router(facts, source)
        } else {
            let mut emitter = super::super::facts::Emitter::count();
            let Some(facts) = native.emit(operation, &mut emitter)? else {
                return Ok(None);
            };
            let Some(source) = emitter.default_scratch_sources() else {
                return Ok(None);
            };
            Description::Default(facts, source)
        };
        // A cold descriptive context can precede publication of the native
        // topology. Keep its validated numerical facts without inventing a
        // placement or turning missing physical evidence into an operation error.
        // The report remains incomplete and cannot admit native execution.
        if context.memory_topology().is_none() {
            return Ok(None);
        }
        let result = match description {
            Description::Router(facts, source) => {
                Self::from_source(native, operation, facts, source, context)
            }
            Description::Default(facts, source) => {
                Self::from_default_sources(native, operation, facts, source, context)
            }
        };
        result
            .map(Some)
            .map_err(|cause| ResidentFactError::source(cause, funding.as_ref()))
    }
    fn from_default_sources(
        native: MlxMetalWorkspaceMechanisms,
        operation: WorkspaceOperationView<'_>,
        facts: WorkspaceOperationFacts,
        source: super::super::facts::DefaultScratchSources,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || Error::from(WorkspaceMetadataError::Unqualified);
        context.charge_metadata(std::mem::size_of::<(
            Self,
            super::super::facts::DefaultScratchSources,
            WorkspaceOperationFacts,
            [WorkspaceScratchAllocation; 2],
            Result<Self, Error>,
        )>())?;
        let (default, execution) = if native.allocation().original_storage {
            let original = crate::backend::managed_memory::cold_original_allocator_placement();
            (original, original)
        } else {
            (
                crate::backend::managed_memory::cold_default_placement(),
                crate::backend::managed_memory::cold_gpu_allocator_placement(),
            )
        };
        let default = default.ok_or_else(invalid)?;
        let execution = execution.ok_or_else(invalid)?;
        let births = native
            .scratch_births(operation)
            .map_err(Error::backend_retained_source)?;
        let controls = native.allocation().host_control_bytes();
        let mut alternatives = context.metadata_vec(source.alternatives().len())?;
        let mut peak = 0;
        for branch in source.alternatives() {
            let total = branch.scratch_bytes.unwrap_or(facts.scratch_bytes);
            if total > facts.scratch_bytes {
                return Err(invalid());
            }
            let device_bytes = total
                .checked_sub(branch.default_bytes)
                .ok_or_else(invalid)?;
            let device_births = births
                .map(|n| n.checked_sub(branch.default_births).ok_or_else(invalid))
                .transpose()?;
            let rows = [
                row(
                    branch.default_bytes,
                    Some(branch.default_births),
                    controls,
                    default,
                    context,
                )?,
                row(device_bytes, device_births, controls, execution, context)?,
            ];
            alternatives.push(context.source_scratch_population(&rows)?);
            peak = peak.max(total);
        }
        if peak != facts.scratch_bytes {
            return Err(invalid());
        }
        let scratch = if alternatives.len() == 1 {
            alternatives.pop().expect("one source branch")
        } else {
            let mut borrowed = context.metadata_vec(alternatives.len())?;
            borrowed.extend(alternatives.iter());
            context.peak_scratch_populations(&borrowed)?
        };
        Ok(Self {
            scratch,
            outputs: std::array::from_fn(|_| None),
            generic: Some(source),
        })
    }

    fn from_source(
        native: MlxMetalWorkspaceMechanisms,
        operation: WorkspaceOperationView<'_>,
        facts: WorkspaceOperationFacts,
        source: selector::AllocationSources,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let invalid = || Error::from(WorkspaceMetadataError::Unqualified);
        context.charge_metadata(std::mem::size_of::<(
            Self,
            selector::AllocationSources,
            WorkspaceOperationFacts,
            [WorkspaceScratchAllocation; 2],
            Result<Self, Error>,
        )>())?;
        let ordinary = !native.allocation().original_storage;
        let default = if ordinary {
            crate::backend::managed_memory::cold_default_placement()
        } else {
            crate::backend::managed_memory::cold_original_allocator_placement()
        }
        .ok_or_else(invalid)?;
        let execution = if ordinary {
            crate::backend::managed_memory::cold_gpu_allocator_placement()
        } else {
            crate::backend::managed_memory::cold_original_allocator_placement()
        }
        .ok_or_else(invalid)?;
        let scratch_births = native
            .scratch_births(operation)
            .map_err(Error::backend_retained_source)?;
        let controls = native.allocation().host_control_bytes();
        let output_births = source.output_births[..source.outputs]
            .iter()
            .filter(|&&birth| birth)
            .count();
        let total_births = scratch_births
            .map(|n| {
                n.checked_add(output_births)
                    .ok_or(WorkspaceMetadataError::Overflow)
            })
            .transpose()?;
        let choices = source.output_births[..source.outputs]
            .iter()
            .zip(&source.output_sources)
            .filter(|(birth, placement)| **birth && **placement == AllocationSource::CutoffChoice)
            .count();
        let branches = 1usize
            .checked_shl(u32::try_from(choices).map_err(|_| WorkspaceMetadataError::Overflow)?)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let fallback = [super::super::facts::DefaultScratchAlternative::default()];
        let projection = source
            .projection_sources
            .as_ref()
            .map_or(&fallback[..], |p| p.alternatives());
        let mut alternatives = context.metadata_vec(
            branches
                .checked_mul(projection.len())
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let mut scratch_peak = 0;
        for child in projection {
            let child_total = child.scratch_bytes.unwrap_or(source.projection_scratch);
            if child_total > source.projection_scratch {
                return Err(invalid());
            }
            let replaced = source
                .projection_scratch
                .checked_sub(child_total)
                .and_then(|bytes| bytes.checked_mul(source.projection_calls as u64))
                .ok_or(WorkspaceMetadataError::Overflow)?;
            let total = source
                .total_bytes
                .checked_sub(replaced)
                .ok_or_else(invalid)?;
            let default_bytes = child
                .default_bytes
                .checked_mul(source.projection_calls as u64)
                .and_then(|bytes| bytes.checked_add(source.default_bytes))
                .ok_or(WorkspaceMetadataError::Overflow)?;
            let default_births = child
                .default_births
                .checked_mul(source.projection_calls)
                .and_then(|births| births.checked_add(source.default_births))
                .ok_or(WorkspaceMetadataError::Overflow)?;
            let total_execution = total.checked_sub(default_bytes).ok_or_else(invalid)?;
            let execution_births = total_births
                .map(|n| n.checked_sub(default_births).ok_or_else(invalid))
                .transpose()?;
            for branch in 0..branches {
                let mut bytes = [default_bytes, total_execution];
                let mut births = [Some(default_births), execution_births];
                let mut choice = 0;
                for (index, (&output, &placement)) in source.output_bytes[..source.outputs]
                    .iter()
                    .zip(&source.output_sources)
                    .enumerate()
                {
                    if !source.output_births[index] {
                        continue;
                    }
                    let slot = match placement {
                        AllocationSource::Default => 0,
                        AllocationSource::Execution => 1,
                        AllocationSource::CutoffChoice => {
                            let slot = usize::from(branch & (1 << choice) == 0);
                            choice += 1;
                            slot
                        }
                    };
                    bytes[slot] = bytes[slot].checked_sub(output).ok_or_else(invalid)?;
                    births[slot] = births[slot]
                        .map(|n| n.checked_sub(1).ok_or_else(invalid))
                        .transpose()?;
                }
                let scratch = bytes[0]
                    .checked_add(bytes[1])
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                if scratch > facts.scratch_bytes
                    || births[0].zip(births[1]).and_then(|(a, b)| a.checked_add(b))
                        != scratch_births
                {
                    return Err(invalid());
                }
                let rows = [
                    row(bytes[0], births[0], controls, default, context)?,
                    row(bytes[1], births[1], controls, execution, context)?,
                ];
                alternatives.push(context.source_scratch_population(&rows)?);
                scratch_peak = scratch_peak.max(scratch);
            }
        }
        if scratch_peak != facts.scratch_bytes {
            return Err(invalid());
        }
        let scratch = if alternatives.len() == 1 {
            alternatives.pop().expect("one source alternative")
        } else {
            let mut borrowed = context.metadata_vec(alternatives.len())?;
            borrowed.extend(alternatives.iter());
            // Each branch contains simultaneous source rows. Physical-domain
            // maxima follow their resolution, never a split of a scalar peak.
            context.peak_scratch_populations(&borrowed)?
        };
        let mut outputs = std::array::from_fn(|_| None);
        for (index, (&bytes, &placement)) in source.output_bytes[..source.outputs]
            .iter()
            .zip(&source.output_sources)
            .enumerate()
        {
            if !source.output_births[index] {
                continue;
            }
            let output = match placement {
                AllocationSource::Default => clone_placement(default, context)?,
                AllocationSource::Execution => clone_placement(execution, context)?,
                AllocationSource::CutoffChoice => {
                    alternatives_placement(default, execution, context)?
                }
            };
            outputs[index] =
                Some(
                    context.source_scratch_population(&[WorkspaceScratchAllocation {
                        bytes,
                        maximum_allocations: Some(1),
                        host_control_bytes: controls,
                        placement: Some(output),
                    }])?,
                );
        }
        Ok(Self {
            scratch,
            outputs,
            generic: None,
        })
    }
}
fn row(
    bytes: u64,
    births: Option<usize>,
    controls: Option<u64>,
    placement: &MemoryPlacement,
    context: &WorkspaceContext,
) -> Result<WorkspaceScratchAllocation, Error> {
    Ok(WorkspaceScratchAllocation {
        bytes,
        maximum_allocations: births,
        host_control_bytes: match (births, controls) {
            (Some(0), _) => Some(0),
            (Some(births), Some(controls)) => Some(
                controls
                    .checked_mul(
                        u64::try_from(births).map_err(|_| WorkspaceMetadataError::Overflow)?,
                    )
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            ),
            _ => None,
        },
        placement: Some(clone_placement(placement, context)?),
    })
}
fn clone_placement(
    source: &MemoryPlacement,
    context: &WorkspaceContext,
) -> Result<MemoryPlacement, Error> {
    let bytes = source
        .clone_backing_bytes()
        .map_err(Error::backend_retained_source)?;
    context
        .charge_metadata(usize::try_from(bytes).map_err(|_| WorkspaceMetadataError::Overflow)?)?;
    Ok(source.clone())
}
fn alternatives_placement(
    default: &MemoryPlacement,
    execution: &MemoryPlacement,
    context: &WorkspaceContext,
) -> Result<MemoryPlacement, Error> {
    if default == execution {
        return clone_placement(default, context);
    }
    let topology = context
        .memory_topology()
        .ok_or(WorkspaceMetadataError::Unqualified)?;
    let count = default
        .domains()
        .len()
        .checked_add(execution.domains().len())
        .ok_or(WorkspaceMetadataError::Overflow)?;
    let mut domains = context.metadata_vec(count)?;
    domains.extend_from_slice(default.domains());
    domains.extend_from_slice(execution.domains());
    fn basis(source: &MemoryPlacement) -> &str {
        match source.kind() {
            MemoryPlacementKind::Fixed(_) => "exact selected allocator",
            MemoryPlacementKind::Possible { basis, .. } => basis.as_str(),
        }
    }
    let label=context.metadata_string(format_args!("router returned backing: eager/CPU source ({}) or execution source ({}); escaping output and completion scratch retain independent conservative placement allowances",basis(default),basis(execution)))?;
    MemoryPlacement::possible(topology, domains, label).map_err(Error::backend_retained_source)
}

pub(super) struct PreparedAllocationFacts<'a> {
    pub(super) base: &'a ResidentExecutionMechanisms,
    pub(super) source: &'a AllocationPopulations,
    pub(super) operation: WorkspaceOperationView<'a>,
}
impl PreparedAllocationFacts<'_> {
    fn authenticate(&self, operation: WorkspaceOperationView<'_>) -> Result<(), ResidentFactError> {
        let same_kind = if let Some(expected) = self.source.generic {
            let ResidentExecutionMechanisms::Metal(native) = *self.base else {
                return Err(MlxWorkspaceFactError::descriptor(
                    "allocation source mechanism differs",
                )
                .into());
            };
            let mut emitter = super::super::facts::Emitter::count();
            let actual = native.emit(operation, &mut emitter)?;
            std::mem::discriminant(&self.operation.kind) == std::mem::discriminant(&operation.kind)
                && emitter.default_scratch_sources() == Some(expected)
                && actual.is_some_and(|fact| {
                    Some(fact.scratch_bytes) == self.source.scratch.backing_bytes()
                })
        } else {
            match (self.operation.kind, operation.kind) {
                (
                    WorkspaceOperationKindView::GroupSelection {
                        spec: a,
                        supplied_indices: ai,
                        control: ac,
                    },
                    WorkspaceOperationKindView::GroupSelection {
                        spec: b,
                        supplied_indices: bi,
                        control: bc,
                    },
                ) => {
                    std::ptr::eq(a, b)
                        && ai == bi
                        && match (ac, bc) {
                            (None, None) => true,
                            (Some(a), Some(b)) => std::ptr::eq(a, b),
                            _ => false,
                        }
                }
                _ => false,
            }
        };
        let same_layouts = |a: WorkspaceLayoutList<'_>, b: WorkspaceLayoutList<'_>| {
            a.len() == b.len()
                && a.iter().zip(b.iter()).all(|(a, b)| {
                    a.shape() == b.shape()
                        && a.dtype() == b.dtype()
                        && a.representation() == b.representation()
                })
        };
        if same_kind
            && same_layouts(self.operation.inputs, operation.inputs)
            && same_layouts(self.operation.outputs, operation.outputs)
        {
            Ok(())
        } else {
            Err(
                MlxWorkspaceFactError::descriptor("allocation source belongs to another operation")
                    .into(),
            )
        }
    }
}
impl WorkspaceFactMechanisms for PreparedAllocationFacts<'_> {
    type Error = ResidentFactError;
    fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        WorkspaceFactMechanisms::memory_topology(self.base)
    }
    fn output_placement(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<&MemoryPlacement> {
        WorkspaceFactMechanisms::output_placement(self.base, op, output)
    }
    fn scratch_placement(&self, op: WorkspaceOperationView<'_>) -> Option<&MemoryPlacement> {
        WorkspaceFactMechanisms::scratch_placement(self.base, op)
    }
    fn allocation_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Option<u64> {
        WorkspaceFactMechanisms::allocation_host_control_bytes(self.base, op, output)
    }
    fn scratch_host_control_bytes(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<u64>, Self::Error> {
        WorkspaceFactMechanisms::scratch_host_control_bytes(self.base, op)
    }
    fn scratch_allocations(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Self::Error> {
        self.authenticate(op)?;
        Ok(Some(&self.source.scratch))
    }
    fn output_allocations(
        &self,
        op: WorkspaceOperationView<'_>,
        output: usize,
    ) -> Result<Option<&WorkspaceAllocationPopulation>, Self::Error> {
        self.authenticate(op)?;
        Ok(self.source.outputs.get(output).and_then(Option::as_ref))
    }
    fn operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.base.operation_facts(op)
    }
    fn write_operation_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        dst: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        self.base.write_operation_facts(op, dst)
    }
    fn host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.base.host_facts(op)
    }
    fn write_host_facts(
        &self,
        op: WorkspaceOperationView<'_>,
        dst: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        self.base.write_host_facts(op, dst)
    }
}
