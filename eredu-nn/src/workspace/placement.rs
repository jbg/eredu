//! Physical attribution attached to the ordinary workspace allocation trace.
use super::*;
use eredu_core::{MemoryPlacement, MemoryPlacementKind};
mod composite;
mod output;
use composite::ScratchDomains;

/// One operation's temporary native allocation allowance. This keeps the
/// selected placement before simultaneous lifetimes are reduced.
#[derive(Debug, Clone)]
pub struct WorkspaceScratchAllocation {
    /// Complete native scratch capacity retained to the completion boundary.
    pub bytes: u64,
    /// Maximum independent backings represented by this exact source row.
    /// An unknown count cannot qualify this row as a retained output population.
    pub maximum_allocations: Option<usize>,
    /// Host allocation controls live until these scratch backings retire.
    pub host_control_bytes: Option<u64>,
    /// Selected mechanism fact; absence cannot be used for admission.
    pub placement: Option<MemoryPlacement>,
}

/// Missing physical attribution is independent of finite or unlimited limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WorkspacePlacementError {
    /// The selected mechanism has no established physical topology.
    #[error("workspace mechanism has no physical memory topology")]
    MissingTopology,
    /// An attribution descriptor is malformed or belongs to another topology.
    #[error(transparent)]
    Domain(#[from] eredu_core::MemoryDomainError),
}

impl WorkspaceContext {
    /// The immutable backend-established topology used by this trace.
    pub fn memory_topology(&self) -> Option<&eredu_core::MemoryTopology> {
        self.mechanisms.memory_topology()
    }

    pub(super) fn copy_placement(
        &self,
        placement: Option<&MemoryPlacement>,
    ) -> Result<Option<MemoryPlacement>, Error> {
        let Some(placement) = placement else {
            return Ok(None);
        };
        let topology = self.memory_topology().ok_or_else(|| {
            Error::backend_retained_source(WorkspacePlacementError::MissingTopology)
        })?;
        placement
            .validate(topology)
            .map_err(Error::backend_retained_source)?;
        // The copied descriptor requests exactly its logical lengths. Its
        // owner shell is included in Storage or WorkspaceScratchAllocation.
        let bytes = match placement.kind() {
            MemoryPlacementKind::Fixed(_) => 0,
            MemoryPlacementKind::Possible { domains, basis } => {
                std::alloc::Layout::array::<eredu_core::MemoryDomainId>(domains.len())
                    .map_err(|_| WorkspaceMetadataError::Overflow)?
                    .size()
                    .checked_add(basis.len())
                    .ok_or(WorkspaceMetadataError::Overflow)?
            }
        };
        self.charge_metadata(bytes)?;
        Ok(Some(placement.clone()))
    }
}

impl WorkspaceExistingStorage {
    /// Imports one homogeneous source envelope with its complete backing count
    /// and the placement established by the selected source producer.
    pub fn try_new_population_placed(
        population: WorkspaceStoragePopulation,
        placement: &MemoryPlacement,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let placement = context.copy_placement(Some(placement))?;
        let mut value = Self::try_new_population(population, context)?;
        Rc::get_mut(&mut value.storage)
            .expect("unpublished existing backing")
            .placement = placement;
        Ok(value)
    }
    /// Imports actual backing placement together with its complete capacity.
    /// Views constructed from this owner retain this entire charge.
    pub fn try_new_placed(
        bytes: Option<u64>,
        placement: &MemoryPlacement,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let placement = context.copy_placement(Some(placement))?;
        let mut value = Self::try_new(bytes, context)?;
        Rc::get_mut(&mut value.storage)
            .expect("unpublished existing backing")
            .placement = placement;
        Ok(value)
    }

    /// Imports the actual backing and its independently allocated host controls.
    pub fn try_new_placed_with_host_controls(
        bytes: Option<u64>,
        placement: &MemoryPlacement,
        host_control_bytes: Option<u64>,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let mut value = Self::try_new_placed(bytes, placement, context)?;
        Rc::get_mut(&mut value.storage)
            .expect("unpublished existing backing")
            .host_control_bytes = host_control_bytes;
        Ok(value)
    }
    /// Host controls retained by this backing, including through every view.
    pub fn host_control_bytes(&self) -> Option<u64> {
        self.storage.host_control_bytes
    }

    /// The original backing's physical attribution, never a view's stream.
    pub fn placement(&self) -> Option<&MemoryPlacement> {
        self.storage.placement.as_ref()
    }
}

impl WorkspaceContext {
    pub(super) fn copy_scratch_allocations(
        &self,
        rows: &[WorkspaceScratchAllocation],
        expected: u64,
    ) -> Result<Vec<WorkspaceScratchAllocation>, Error> {
        let total = rows
            .iter()
            .try_fold(0u64, |sum, row| sum.checked_add(row.bytes))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        if total != expected {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        let mut owned = self.metadata_vec(rows.len())?;
        for row in rows {
            owned.push(WorkspaceScratchAllocation {
                bytes: row.bytes,
                maximum_allocations: row.maximum_allocations,
                host_control_bytes: row.host_control_bytes,
                placement: self.copy_placement(row.placement.as_ref())?,
            });
        }
        Ok(owned)
    }
}

/// Source-derived new native backing and scratch, kept with original metadata
/// identities until the enclosing operation's physical-domain reduction. This
/// is a descriptive population, not native ownership or source storage credit.
#[derive(Clone, Debug)]
pub struct WorkspaceAllocationPopulation(Rc<ScratchPopulation>);
#[derive(Debug)]
struct ScratchPopulation {
    allocations: Vec<WorkspaceScratchAllocation>,
    roots: Vec<Rc<Storage>>,
    children: Vec<WorkspaceAllocationPopulation>,
    funding: Option<HostMetadataFunding>,
    domains: Option<ScratchDomains>,
}
impl WorkspaceAllocationPopulation {
    /// Raw source allocations, before any physical-domain maxima. Resolved
    /// composite populations expose an empty directory; `backing_bytes` reports
    /// their diagnostic capacity. Repetition always declares independent births.
    pub fn allocations(&self) -> &[WorkspaceScratchAllocation] {
        &self.0.allocations
    }
}
impl WorkspaceContext {
    /// Retains a backend producer's homogeneous allocation descriptors before
    /// lifetime reduction. The producer supplies actual allocator placements;
    /// these descriptive rows do not authorize allocation or credit inputs.
    pub fn source_scratch_population(
        &self,
        allocations: &[WorkspaceScratchAllocation],
    ) -> Result<WorkspaceAllocationPopulation, Error> {
        self.charge_metadata(std::mem::size_of::<(
            ScratchPopulation,
            WorkspaceAllocationPopulation,
            Result<WorkspaceAllocationPopulation, Error>,
        )>())?;
        let bytes = allocations
            .iter()
            .try_fold(0u64, |sum, row| sum.checked_add(row.bytes))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let allocations = self.copy_scratch_allocations(allocations, bytes)?;
        Ok(WorkspaceAllocationPopulation(self.metadata_rc(
            ScratchPopulation {
                allocations,
                roots: Vec::new(),
                children: Vec::new(),
                funding: self.metadata_funding(),
                domains: None,
            },
        )?))
    }

    /// Retains new-allocation identities and their selected physical placement
    /// before this child trace is reduced. Existing inputs and their aliases do
    /// not enter the new-allocation directory; scratch shares the same completion
    /// lifetime. Missing facts remain unavailable, never an empty population.
    pub fn new_allocation_scratch(&self) -> Result<Option<WorkspaceAllocationPopulation>, Error> {
        let trace = self.trace.borrow();
        if trace.report_finished {
            return Err(WorkspaceMetadataError::ReportFinished.into());
        }
        if !trace.missing.is_empty() {
            return Ok(None);
        }
        let count = trace
            .allocations
            .len()
            .checked_add(trace.placed_scratch.len())
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.charge_metadata(std::mem::size_of::<(
            WorkspaceAllocationPopulation,
            Result<Option<WorkspaceAllocationPopulation>, Error>,
            ScratchPopulation,
        )>())?;
        let mut allocations = self.metadata_vec(count)?;
        let mut roots = self.metadata_vec(trace.allocations.len())?;
        for root in &trace.allocations {
            let Some(bytes) = root.bytes else {
                return Ok(None);
            };
            // Allocate-or-alias metadata records its prospective new backing
            // once; the borrowed alternative remains in the original DAG.
            allocations.push(WorkspaceScratchAllocation {
                bytes,
                maximum_allocations: Some(root.maximum_allocations),
                host_control_bytes: root.host_control_bytes,
                placement: self.copy_placement(root.placement.as_ref())?,
            });
            roots.push(root.clone());
        }
        for row in &trace.placed_scratch {
            allocations.push(WorkspaceScratchAllocation {
                bytes: row.bytes,
                maximum_allocations: row.maximum_allocations,
                host_control_bytes: row.host_control_bytes,
                placement: self.copy_placement(row.placement.as_ref())?,
            });
        }
        let mut children = self.metadata_vec(trace.scratch_sources.len())?;
        children.extend(trace.scratch_sources.iter().cloned());
        let ordinary = WorkspaceAllocationPopulation(self.metadata_rc(ScratchPopulation {
            allocations,
            roots,
            children,
            funding: self.metadata_funding(),
            domains: None,
        })?);
        let groups = trace
            .scratch_sources
            .iter()
            .filter(|source| source.domain_population().is_some())
            .count();
        if groups == 0 {
            return Ok(Some(ordinary));
        }
        let mut sources = self.metadata_vec(
            groups
                .checked_add(1)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        sources.push((&ordinary, 1));
        sources.extend(
            trace
                .scratch_sources
                .iter()
                .filter(|source| source.domain_population().is_some())
                .map(|source| (source, 1)),
        );
        Ok(Some(self.combine_scratch_populations(&sources)?))
    }
}

impl WorkspaceContext {
    /// Combines independent child occurrences without deduplicating their new
    /// allocations. Original source identities and placement bases remain held;
    /// this operation does not select a branch or calculate domain maxima.
    pub fn combine_scratch_populations(
        &self,
        sources: &[(&WorkspaceAllocationPopulation, usize)],
    ) -> Result<WorkspaceAllocationPopulation, Error> {
        if sources
            .iter()
            .any(|(source, _)| source.domain_population().is_some())
        {
            return self.compose_scratch_domains(sources, false);
        }
        self.charge_metadata(std::mem::size_of::<(
            WorkspaceAllocationPopulation,
            ScratchPopulation,
            Result<WorkspaceAllocationPopulation, Error>,
            &[(&WorkspaceAllocationPopulation, usize)],
        )>())?;
        let count = sources
            .iter()
            .try_fold(0usize, |total, (source, repeat)| {
                source
                    .allocations()
                    .len()
                    .checked_mul(*repeat)
                    .and_then(|n| total.checked_add(n))
            })
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut allocations = self.metadata_vec(count)?;
        let mut children = self.metadata_vec(sources.len())?;
        for (source, repeat) in sources {
            if *repeat == 0 {
                continue;
            }
            for _ in 0..*repeat {
                for row in source.allocations() {
                    allocations.push(WorkspaceScratchAllocation {
                        bytes: row.bytes,
                        maximum_allocations: row.maximum_allocations,
                        host_control_bytes: row.host_control_bytes,
                        placement: self.copy_placement(row.placement.as_ref())?,
                    });
                }
            }
            children.push((*source).clone());
        }
        Ok(WorkspaceAllocationPopulation(self.metadata_rc(
            ScratchPopulation {
                allocations,
                roots: Vec::new(),
                children,
                funding: self.metadata_funding(),
                domains: None,
            },
        )?))
    }
}

impl WorkspaceContext {
    pub(super) fn copy_scratch_population(
        &self,
        source: &WorkspaceAllocationPopulation,
        expected: u64,
    ) -> Result<Vec<WorkspaceScratchAllocation>, Error> {
        if source.backing_bytes() != Some(expected) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        if let Some(topology) = self.memory_topology() {
            source.validate_domains(topology)?;
        }
        let raw = source
            .allocations()
            .iter()
            .try_fold(0u64, |n, row| n.checked_add(row.bytes))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.copy_scratch_allocations(source.allocations(), raw)
    }
}
