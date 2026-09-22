//! Dense physical-domain destinations for the shared allocation-lifetime walk.
use super::*;
use eredu_core::{DomainMemoryCharge, MemoryDomainId, MemoryPlacement, MemoryTopology};

/// A backing population attributed to one physical domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceDomainStoragePopulation {
    /// Fixed charges and conservative candidate-placement allowances.
    pub charge: DomainMemoryCharge,
    /// Distinct positive backings that may occupy this domain.
    pub maximum_allocations: usize,
}

/// Exact state lifetimes in one physical domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceDomainStateReport {
    /// All closing state backing, including unchanged aliases.
    pub retained: DomainMemoryCharge,
    /// Opening backing no longer present in the closing state.
    pub displaced: DomainMemoryCharge,
    /// New transients plus displaced opening backing.
    pub transient: DomainMemoryCharge,
}

/// Native allocation allowance independent of disjoint host workspace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceDomainTensorBufferReport {
    /// New native backings and temporary native scratch in this domain.
    pub total: DomainMemoryCharge,
    /// New native backing retained in the closing roots.
    pub retained: DomainMemoryCharge,
    /// Native allocations retained only through the completion boundary.
    pub transient: DomainMemoryCharge,
}

/// Source-credit reduction uses original backing identities before summation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceDomainResidualReport {
    /// Unborrowed opening backing union.
    pub opening: WorkspaceDomainStoragePopulation,
    /// Unborrowed closing backing union.
    pub closing: WorkspaceDomainStoragePopulation,
    /// Unborrowed simultaneous union, including scratch and host workspace.
    pub total: DomainMemoryCharge,
    /// Unborrowed closing backing.
    pub retained: DomainMemoryCharge,
    /// Unborrowed opening backing absent from closing state.
    pub displaced: DomainMemoryCharge,
    /// Unborrowed union absent from closing state plus scratch and host.
    pub transient: DomainMemoryCharge,
    /// Selected native backings and scratch before disjoint host workspace.
    pub tensor_buffers: WorkspaceDomainTensorBufferReport,
    /// Operation-owned host workspace attributed to the host physical domain.
    pub host_workspace_bytes: u64,
}

/// One complete physical-domain result; there is no aggregate memory ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceDomainReportEntry {
    /// The topology-scoped domain represented by this destination cell.
    pub domain: MemoryDomainId,
    /// Full closing union, preserving complete backing capacity for views.
    pub closing: WorkspaceDomainStoragePopulation,
    /// Full opening union, or absent if no state span was seeded.
    pub opening: Option<WorkspaceDomainStoragePopulation>,
    /// New allocations, native scratch and disjoint host workspace.
    pub total: DomainMemoryCharge,
    /// New allocations surviving in the supplied closing roots.
    pub retained: DomainMemoryCharge,
    /// New allocations not surviving, native scratch and host workspace.
    pub transient: DomainMemoryCharge,
    /// Selected native backings and scratch before disjoint host workspace.
    pub tensor_buffers: WorkspaceDomainTensorBufferReport,
    /// Operation-owned host workspace attributed to the host physical domain.
    pub host_workspace_bytes: u64,
    /// Opening/closing lifetime accounting, when seeded.
    pub state: Option<WorkspaceDomainStateReport>,
    /// Original backing exclusion, when an exact borrowed selection is installed.
    pub residual: Option<WorkspaceDomainResidualReport>,
}
impl WorkspaceDomainReportEntry {
    /// Initializes one caller-owned destination without granting any authority.
    pub fn empty(domain: MemoryDomainId) -> Self {
        Self {
            domain,
            closing: Default::default(),
            opening: None,
            total: Default::default(),
            retained: Default::default(),
            transient: Default::default(),
            tensor_buffers: Default::default(),
            host_workspace_bytes: 0,
            state: None,
            residual: None,
        }
    }
}

/// Attribution for the exact original nodes, never reconstructed from totals.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceReportPlacements<'a> {
    /// Immutable physical mapping supplied by the selected backend.
    pub topology: &'a MemoryTopology,
    /// One placement per graph node; zero-capacity view nodes need no placement.
    pub backings: &'a [Option<MemoryPlacement>],
    /// Per-backing host control allocations; absent means no additional controls.
    pub host_controls: Option<&'a [Option<u64>]>,
    /// Native scratch retained through the same completion boundary.
    pub scratch: &'a [WorkspaceScratchAllocation],
}

/// Exact source roots with scratch supplied separately by physical placement.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceDomainReportInputs<'a> {
    /// Absent and explicitly empty state are distinct.
    pub opening: Option<&'a [usize]>,
    /// One ordered root per new allocation.
    pub allocations: &'a [usize],
    /// Retained closing roots, including views and aliases.
    pub closing: &'a [usize],
    /// Authenticated source selection's corresponding metadata roots.
    pub borrowed: Option<&'a [usize]>,
    /// Disjoint operation-owned host workspace.
    pub host_workspace: Option<u64>,
    /// Every executed operation must have complete tensor facts.
    pub tensor_complete: bool,
}

/// Owning result from an ordinary workspace context. Candidate allowance
/// descriptors retain their estimation bases; their sum is not residency.
#[derive(Clone, Debug)]
pub struct WorkspaceDomainReport {
    /// Dense topology-order results.
    pub domains: Vec<WorkspaceDomainReportEntry>,
    /// Candidate placements from the distinct backing union and scratch facts.
    pub placement_allowances: Vec<eredu_core::PlacementAllowance>,
    /// New backing, native scratch and disjoint host workspace.
    pub new_allocations: eredu_core::DomainMemoryRequirements,
    /// New backing and native scratch without disjoint host workspace.
    pub native_allocations: eredu_core::DomainMemoryRequirements,
    /// New transients and displaced opening state, when a state span is seeded.
    pub state_transient: Option<eredu_core::DomainMemoryRequirements>,
    /// Complete retained state backing, when a state span is seeded.
    pub retained_state: Option<eredu_core::DomainMemoryRequirements>,
    /// Actual unborrowed backing union with scratch and host workspace.
    pub residual: Option<eredu_core::DomainMemoryRequirements>,
}
impl WorkspaceDomainReport {
    /// Adds separately proved operation-owned host overlap to this same report.
    /// Native buffers and retained decoder backing remain independently charged.
    pub fn add_host_workspace(
        &mut self,
        topology: &MemoryTopology,
        bytes: u64,
    ) -> Result<(), WorkspaceReportError> {
        let host = topology.host_domain();
        let placement = MemoryPlacement::fixed(topology, host)?;
        let amount = DomainMemoryCharge {
            accounted_bytes: bytes,
            ..Default::default()
        };
        for requirements in std::iter::once(&self.new_allocations)
            .chain(self.state_transient.iter())
            .chain(self.residual.iter())
        {
            requirements.validate(topology)?;
            requirements.get(host)?.checked_add(amount)?;
        }
        let entry = self
            .domains
            .iter_mut()
            .find(|entry| entry.domain == host)
            .ok_or(WorkspaceReportError::Source)?;
        let mut next = *entry;
        next.total = next.total.checked_add(amount)?;
        next.transient = next.transient.checked_add(amount)?;
        next.host_workspace_bytes = next
            .host_workspace_bytes
            .checked_add(bytes)
            .ok_or(WorkspaceReportError::Overflow)?;
        if let Some(state) = &mut next.state {
            state.transient = state.transient.checked_add(amount)?;
        }
        if let Some(residual) = &mut next.residual {
            residual.total = residual.total.checked_add(amount)?;
            residual.transient = residual.transient.checked_add(amount)?;
            residual.host_workspace_bytes = residual
                .host_workspace_bytes
                .checked_add(bytes)
                .ok_or(WorkspaceReportError::Overflow)?;
        }
        for requirements in std::iter::once(&mut self.new_allocations)
            .chain(self.state_transient.iter_mut())
            .chain(self.residual.iter_mut())
        {
            requirements
                .add_allocation(bytes, &placement)
                .expect("validated host requirement addition");
        }
        *entry = next;
        Ok(())
    }
}

impl WorkspaceReportWorkspace {
    /// Uses the same identity union and opening/closing traversal as `report`.
    /// No callback, native work or destination allocation occurs. Every domain
    /// is validated before any supplied result cell changes.
    pub fn report_domains(
        &mut self,
        graph: WorkspaceReportGraph<'_>,
        input: WorkspaceDomainReportInputs<'_>,
        placements: WorkspaceReportPlacements<'_>,
        destination: &mut [WorkspaceDomainReportEntry],
    ) -> Result<(), WorkspaceReportError> {
        let source = source::Flat::new(
            graph,
            WorkspaceReportInputs {
                opening: input.opening,
                allocations: input.allocations,
                closing: input.closing,
                borrowed: input.borrowed,
                scratch: 0,
                host_workspace: input.host_workspace,
                tensor_complete: input.tensor_complete,
            },
        )?
        .with_placements(placements.backings, placements.host_controls)?;
        self.scratch.report_domains(
            &source,
            placements.topology,
            placements.scratch,
            destination,
        )
    }
}

impl WorkspaceContext {
    /// Produces the same physical report when every backing has attribution.
    /// Missing placement or capacity remains an absent report; malformed facts,
    /// arithmetic overflow and metadata funding failures remain typed errors.
    pub fn complete_domain_report(
        &self,
        retained: &[WorkspaceTensor],
    ) -> Result<Option<WorkspaceDomainReport>, Error> {
        if self.memory_topology().is_none() {
            return Ok(None);
        }
        match self.report_domains(retained) {
            Ok(report) => Ok(Some(report)),
            Err(cause)
                if matches!(
                    &cause.storage,
                    crate::ErrorStorage::WorkspaceMetadata(WorkspaceMetadataError::Report(
                        WorkspaceReportError::IncompletePlacement
                    ))
                ) =>
            {
                Ok(None)
            }
            Err(cause) => Err(cause),
        }
    }
    /// Reduces the ordinary trace directly into physical domains. The same
    /// metadata account funds all graph buffers, dense result cells and copied
    /// allowance descriptions before allocation. This creates no authority.
    pub fn report_domains(
        &self,
        retained: &[WorkspaceTensor],
    ) -> Result<WorkspaceDomainReport, Error> {
        if self.trace.borrow().report_finished {
            return Err(WorkspaceMetadataError::ReportFinished.into());
        }
        for value in retained {
            value.validate_context(self)?;
        }
        let topology = self.memory_topology().ok_or_else(|| {
            Error::backend_retained_source(WorkspacePlacementError::MissingTopology)
        })?;
        let layout = self
            .tracing_started
            .layout()
            .map_err(WorkspaceMetadataError::from)?;
        let controls = Self::report_metadata_bytes(layout.nodes(), layout.edges())?
            .checked_add(std::mem::size_of::<WorkspaceDomainReport>())
            .and_then(|v| {
                v.checked_add(std::mem::size_of::<Result<WorkspaceDomainReport, Error>>())
            })
            .and_then(|v| v.checked_add(std::mem::size_of::<WorkspaceDomainReportEntry>()))
            .and_then(|v| v.checked_add(std::mem::size_of::<WorkspaceReportPlacements<'_>>()))
            .ok_or(WorkspaceMetadataError::Overflow)?;
        self.charge_metadata(controls)?;
        let mut scratch =
            Scratch::<Rc<Storage>>::new(layout, None).map_err(|(prefix, cause)| {
                drop(prefix);
                Error::backend_retained_source(cause)
            })?;
        let trace = self.trace.borrow();
        let borrowed = self.borrowed.borrow();
        let source = Ordinary {
            trace: &trace,
            retained,
            borrowed: borrowed.as_ref(),
        };
        let mut report =
            owned_report(&mut scratch, &source, topology, &trace.placed_scratch, self)?;
        for source in &trace.scratch_sources {
            report.append_scratch_population(source, self)?;
        }
        Ok(report)
    }
}

fn owned_report<R: Clone, G: Graph<Root = R>>(
    scratch: &mut Scratch<R>,
    source: &G,
    topology: &MemoryTopology,
    temporaries: &[WorkspaceScratchAllocation],
    context: &WorkspaceContext,
) -> Result<WorkspaceDomainReport, Error> {
    let mut domains = context.metadata_vec(topology.len())?;
    domains.extend(
        topology
            .domains()
            .map(|(domain, _)| WorkspaceDomainReportEntry::empty(domain)),
    );
    scratch
        .report_domains(source, topology, temporaries, &mut domains)
        .map_err(WorkspaceMetadataError::from)?;
    let scratch_candidates = || {
        temporaries.iter().filter_map(|allowance| {
            let placement = allowance.placement.as_ref()?;
            matches!(
                placement.kind(),
                eredu_core::MemoryPlacementKind::Possible { .. }
            )
            .then_some((allowance.bytes, placement))
        })
    };
    let count = scratch
        .candidate_placements(source)
        .count()
        .checked_add(scratch_candidates().count())
        .ok_or(WorkspaceMetadataError::Overflow)?;
    let mut placement_allowances = context.metadata_vec(count)?;
    for (backing_bytes, placement) in scratch
        .candidate_placements(source)
        .chain(scratch_candidates())
    {
        placement_allowances.push(eredu_core::PlacementAllowance {
            backing_bytes,
            placement: context
                .copy_placement(Some(placement))?
                .expect("provided placement"),
        });
    }
    if context.facts.is_some() {
        context.identity.metadata_reports.set(Some(
            context
                .identity
                .metadata_reports
                .get()
                .and_then(|v| v.checked_add(1))
                .ok_or(WorkspaceMetadataError::Overflow)?,
        ));
    }
    use reduce::domains::RequirementSelection;
    let requirement =
        |selection| scratch.requirements(source, topology, temporaries, selection, context);
    let new_allocations = requirement(RequirementSelection::New)?;
    let native_allocations = requirement(RequirementSelection::Native)?;
    let state_transient = source
        .facts()
        .seeded
        .then(|| requirement(RequirementSelection::StateTransient))
        .transpose()?;
    let retained_state = source
        .facts()
        .seeded
        .then(|| requirement(RequirementSelection::Closing))
        .transpose()?;
    let residual = source
        .facts()
        .residual
        .then(|| requirement(RequirementSelection::Residual))
        .transpose()?;
    Ok(WorkspaceDomainReport {
        domains,
        placement_allowances,
        new_allocations,
        native_allocations,
        state_transient,
        retained_state,
        residual,
    })
}

impl WorkspaceReportWorkspace {
    /// Emits owning physical descriptors through the supplied metadata account,
    /// using this already allocated flat graph workspace and its same reducer.
    pub fn report_domains_owned(
        &mut self,
        graph: WorkspaceReportGraph<'_>,
        input: WorkspaceDomainReportInputs<'_>,
        placements: WorkspaceReportPlacements<'_>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceDomainReport, Error> {
        let source = source::Flat::new(
            graph,
            WorkspaceReportInputs {
                opening: input.opening,
                allocations: input.allocations,
                closing: input.closing,
                borrowed: input.borrowed,
                scratch: 0,
                host_workspace: input.host_workspace,
                tensor_complete: input.tensor_complete,
            },
        )
        .map_err(WorkspaceMetadataError::from)?
        .with_placements(placements.backings, placements.host_controls)
        .map_err(WorkspaceMetadataError::from)?;
        context.charge_metadata(
            std::mem::size_of::<WorkspaceDomainReport>()
                + std::mem::size_of::<WorkspaceDomainReportEntry>()
                + std::mem::size_of::<Result<WorkspaceDomainReport, Error>>(),
        )?;
        owned_report(
            &mut self.scratch,
            &source,
            placements.topology,
            placements.scratch,
            context,
        )
    }

    pub(in crate::workspace) fn report_domain_diagnostics(
        &mut self,
        graph: WorkspaceReportGraph<'_>,
        input: WorkspaceReportInputs<'_>,
        controls: &[Option<u64>],
        scratch: &[WorkspaceScratchAllocation],
        complete_domains: bool,
    ) -> Result<WorkspaceReportScalars, WorkspaceReportError> {
        self.scratch.report_with_diagnostics(
            &source::Flat::new(graph, input)?.with_controls(Some(controls), scratch)?,
            complete_domains,
        )
    }
}

#[cfg(test)]
mod tests;

impl WorkspaceDomainReport {
    /// Add a retained alternative/simultaneous scratch source after the same
    /// domain reduction. No retained output or borrowed input credit is created.
    pub(in crate::workspace) fn append_scratch_population(
        &mut self,
        source: &WorkspaceAllocationPopulation,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        let Some(population) = source.domain_population() else {
            return Ok(());
        };
        let topology = context.memory_topology().ok_or_else(|| {
            Error::backend_retained_source(WorkspacePlacementError::MissingTopology)
        })?;
        source.validate_domains(topology)?;
        let all = population
            .all
            .as_ref()
            .ok_or(WorkspaceMetadataError::Report(
                WorkspaceReportError::IncompletePlacement,
            ))?;
        let add = |left: &eredu_core::DomainMemoryRequirements,
                   right: &eredu_core::DomainMemoryRequirements|
         -> Result<eredu_core::DomainMemoryRequirements, Error> {
            context.charge_requirements(&[left, right])?;
            left.checked_add(right)
                .map_err(Error::backend_retained_source)
        };
        let new_allocations = add(&self.new_allocations, all)?;
        let native_allocations = add(&self.native_allocations, &population.native)?;
        let state_transient = self
            .state_transient
            .as_ref()
            .map(|source| add(source, all))
            .transpose()?;
        let residual = self
            .residual
            .as_ref()
            .map(|source| add(source, all))
            .transpose()?;
        let next =
            |entry: &WorkspaceDomainReportEntry| -> Result<WorkspaceDomainReportEntry, Error> {
                let mut entry = *entry;
                let all = all
                    .get(entry.domain)
                    .map_err(Error::backend_retained_source)?;
                let native = population
                    .native
                    .get(entry.domain)
                    .map_err(Error::backend_retained_source)?;
                let sum = |a: DomainMemoryCharge, b: DomainMemoryCharge| {
                    a.checked_add(b).map_err(Error::backend_retained_source)
                };
                entry.total = sum(entry.total, all)?;
                entry.transient = sum(entry.transient, all)?;
                entry.tensor_buffers.total = sum(entry.tensor_buffers.total, native)?;
                entry.tensor_buffers.transient = sum(entry.tensor_buffers.transient, native)?;
                if let Some(state) = &mut entry.state {
                    state.transient = sum(state.transient, all)?;
                }
                if let Some(residual) = &mut entry.residual {
                    residual.total = sum(residual.total, all)?;
                    residual.transient = sum(residual.transient, all)?;
                    residual.tensor_buffers.total = sum(residual.tensor_buffers.total, native)?;
                    residual.tensor_buffers.transient =
                        sum(residual.tensor_buffers.transient, native)?;
                }
                Ok(entry)
            };
        // Check the entire destination before publishing any domain.
        for entry in &self.domains {
            next(entry)?;
        }
        let mut allowances = context.metadata_vec(
            self.placement_allowances
                .len()
                .checked_add(population.native.placement_allowances().len())
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        for allowance in self
            .placement_allowances
            .iter()
            .chain(population.native.placement_allowances())
        {
            allowances.push(eredu_core::PlacementAllowance {
                backing_bytes: allowance.backing_bytes,
                placement: context
                    .copy_placement(Some(&allowance.placement))?
                    .expect("provided placement"),
            });
        }
        for entry in &mut self.domains {
            *entry = next(entry)?;
        }
        self.new_allocations = new_allocations;
        self.native_allocations = native_allocations;
        self.state_transient = state_transient;
        self.residual = residual;
        self.placement_allowances = allowances;
        Ok(())
    }
}
