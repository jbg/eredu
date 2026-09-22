use super::*;
use crate::workspace::report::domains::*;
use eredu_core::{DomainMemoryCharge, MemoryPlacement, MemoryPlacementKind, MemoryTopology};

const N: u8 = 8;

#[derive(Clone, Copy)]
pub(in crate::workspace::report) enum RequirementSelection {
    New,
    Native,
    StateTransient,
    Closing,
    Residual,
}
impl RequirementSelection {
    fn includes(self, marks: u8) -> bool {
        match self {
            Self::New | Self::Native => marks & N != 0,
            Self::StateTransient => marks & (N | O) != 0 && marks & C == 0,
            Self::Closing => marks & C != 0,
            Self::Residual => marks & B == 0,
        }
    }
    fn scratch(self) -> bool {
        !matches!(self, Self::Closing)
    }
    fn host(self) -> bool {
        !matches!(self, Self::Closing | Self::Native)
    }
}

fn charge(bytes: u64, placement: &MemoryPlacement) -> DomainMemoryCharge {
    match placement.kind() {
        MemoryPlacementKind::Fixed(_) => DomainMemoryCharge {
            accounted_bytes: bytes,
            ..Default::default()
        },
        MemoryPlacementKind::Possible { .. } => DomainMemoryCharge {
            placement_allowance_bytes: bytes,
            ..Default::default()
        },
    }
}
fn add_to(
    destination: &mut DomainMemoryCharge,
    amount: DomainMemoryCharge,
) -> Result<(), WorkspaceReportError> {
    *destination = destination.checked_add(amount)?;
    Ok(())
}
fn add_population(
    destination: &mut WorkspaceDomainStoragePopulation,
    amount: DomainMemoryCharge,
    allocations: usize,
) -> Result<(), WorkspaceReportError> {
    add_to(&mut destination.charge, amount)?;
    destination.maximum_allocations = destination
        .maximum_allocations
        .checked_add(allocations)
        .ok_or(WorkspaceReportError::Overflow)?;
    Ok(())
}

impl<R: Clone> Scratch<R> {
    pub(in crate::workspace::report) fn requirements<G: Graph<Root = R>>(
        &self,
        source: &G,
        topology: &MemoryTopology,
        scratch: &[WorkspaceScratchAllocation],
        selection: RequirementSelection,
        context: &WorkspaceContext,
    ) -> Result<eredu_core::DomainMemoryRequirements, Error> {
        let nodes = || {
            self.nodes
                .iter()
                .zip(&self.marks)
                .filter_map(|(node, &marks)| {
                    if !selection.includes(marks) || node.bytes == Some(0) {
                        return None;
                    }
                    Some((
                        node.bytes.expect("validated capacity"),
                        source
                            .placement(&node.identity)
                            .expect("validated placement"),
                    ))
                })
        };
        let temporaries = || {
            scratch.iter().filter_map(|allowance| {
                (selection.scratch() && allowance.bytes != 0).then(|| {
                    (
                        allowance.bytes,
                        allowance.placement.as_ref().expect("validated scratch"),
                    )
                })
            })
        };
        let mut count = 0usize;
        let mut descriptions = 0u64;
        for (_, placement) in nodes().chain(temporaries()) {
            if matches!(placement.kind(), MemoryPlacementKind::Possible { .. }) {
                count = count
                    .checked_add(1)
                    .ok_or(WorkspaceMetadataError::Overflow)?;
                descriptions = descriptions
                    .checked_add(
                        placement
                            .backing_bytes()
                            .map_err(Error::backend_retained_source)?,
                    )
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
        }
        let bytes =
            eredu_core::DomainMemoryRequirements::construction_backing_bytes(topology, count)
                .map_err(Error::backend_retained_source)?
                .checked_add(descriptions)
                .ok_or(WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(
            usize::try_from(bytes).map_err(|_| WorkspaceMetadataError::Overflow)?,
        )?;
        let mut result =
            eredu_core::DomainMemoryRequirements::zero_with_allowance_capacity(topology, count);
        for (bytes, placement) in nodes().chain(temporaries()) {
            result
                .add_allocation(bytes, placement)
                .map_err(Error::backend_retained_source)?;
        }
        if !matches!(selection, RequirementSelection::Native) {
            let mut controls = 0u64;
            for (node, &marks) in self.nodes.iter().zip(&self.marks) {
                if selection.includes(marks) {
                    controls = controls
                        .checked_add(
                            source
                                .host_control_bytes(&node.identity)
                                .expect("validated backing controls"),
                        )
                        .ok_or(WorkspaceMetadataError::Overflow)?;
                }
            }
            if selection.scratch() {
                for allowance in scratch {
                    controls = controls
                        .checked_add(
                            allowance
                                .host_control_bytes
                                .expect("validated scratch controls"),
                        )
                        .ok_or(WorkspaceMetadataError::Overflow)?;
                }
            }
            result
                .add_allocation(
                    controls,
                    &MemoryPlacement::fixed(topology, topology.host_domain())
                        .map_err(Error::backend_retained_source)?,
                )
                .map_err(Error::backend_retained_source)?;
        }
        if selection.host() {
            result
                .add_allocation(
                    source.facts().host.expect("validated host"),
                    &MemoryPlacement::fixed(topology, topology.host_domain())
                        .map_err(Error::backend_retained_source)?,
                )
                .map_err(Error::backend_retained_source)?;
        }
        Ok(result)
    }
    pub(in crate::workspace::report) fn report_domains<G: Graph<Root = R>>(
        &mut self,
        source: &G,
        topology: &MemoryTopology,
        scratch: &[WorkspaceScratchAllocation],
        destination: &mut [WorkspaceDomainReportEntry],
    ) -> Result<(), WorkspaceReportError> {
        if destination.len() != topology.len() {
            return Err(WorkspaceReportError::Capacity);
        }
        for ((domain, _), entry) in topology.domains().zip(destination.iter()) {
            if domain != entry.domain {
                return Err(WorkspaceReportError::Source);
            }
        }
        let facts = source.facts();
        if !facts.complete || facts.host.is_none() {
            return Err(WorkspaceReportError::IncompletePlacement);
        }
        if facts.residual && !facts.seeded {
            return Err(WorkspaceReportError::IncompletePlacement);
        }
        self.fill(source)?;
        self.walk(source, CLOSE, C, false)?;
        if facts.seeded {
            self.walk(source, OPEN, O, false)?;
        }
        for i in 0..source.roots(NEW) {
            let root = source.root(NEW, i);
            let index = self
                .find(source, &root)
                .ok_or(WorkspaceReportError::Source)?;
            self.marks[index] |= N;
        }
        // Reject incomplete or foreign facts before reducing any destination.
        for node in &self.nodes {
            let bytes = node
                .bytes
                .ok_or(WorkspaceReportError::IncompletePlacement)?;
            source
                .host_control_bytes(&node.identity)
                .ok_or(WorkspaceReportError::IncompletePlacement)?;
            if bytes != 0 {
                source
                    .placement(&node.identity)
                    .ok_or(WorkspaceReportError::IncompletePlacement)?
                    .validate(topology)?;
            }
        }
        for allowance in scratch {
            allowance
                .host_control_bytes
                .ok_or(WorkspaceReportError::IncompletePlacement)?;
            if allowance.bytes != 0 {
                allowance
                    .placement
                    .as_ref()
                    .ok_or(WorkspaceReportError::IncompletePlacement)?
                    .validate(topology)?;
            }
        }
        // A stack-local complete domain result validates every checked counter.
        // Repeating this pure reduction publishes only after every domain fits.
        for entry in destination.iter() {
            self.reduce_domain(source, topology, scratch, entry.domain)?;
        }
        for entry in destination {
            *entry = self
                .reduce_domain(source, topology, scratch, entry.domain)
                .expect("all domain reductions validated");
        }
        Ok(())
    }

    fn reduce_domain<G: Graph<Root = R>>(
        &self,
        source: &G,
        topology: &MemoryTopology,
        scratch: &[WorkspaceScratchAllocation],
        domain: eredu_core::MemoryDomainId,
    ) -> Result<WorkspaceDomainReportEntry, WorkspaceReportError> {
        let facts = source.facts();
        let mut result = WorkspaceDomainReportEntry::empty(domain);
        result.opening = facts.seeded.then(Default::default);
        result.state = facts.seeded.then(Default::default);
        result.residual = facts.residual.then(Default::default);
        let mut temporary = DomainMemoryCharge::default();
        if domain == topology.host_domain() {
            temporary.accounted_bytes = facts.host.expect("validated host workspace");
            result.host_workspace_bytes = temporary.accounted_bytes;
        }
        let mut native_temporary = DomainMemoryCharge::default();
        for allowance in scratch {
            if domain == topology.host_domain() {
                temporary.accounted_bytes = temporary
                    .accounted_bytes
                    .checked_add(
                        allowance
                            .host_control_bytes
                            .expect("validated scratch controls"),
                    )
                    .ok_or(WorkspaceReportError::Overflow)?;
            }
            if allowance.bytes == 0 {
                continue;
            }
            let placement = allowance
                .placement
                .as_ref()
                .expect("validated scratch placement");
            if placement.domains().contains(&domain) {
                let amount = charge(allowance.bytes, placement);
                add_to(&mut temporary, amount)?;
                add_to(&mut native_temporary, amount)?;
            }
        }
        result.tensor_buffers.total = native_temporary;
        result.tensor_buffers.transient = native_temporary;
        if let Some(residual) = &mut result.residual {
            residual.tensor_buffers = result.tensor_buffers;
        }
        result.total = temporary;
        result.transient = temporary;
        if let Some(residual) = &mut result.residual {
            residual.total = temporary;
            residual.transient = temporary;
        }
        for &i in &self.index {
            let node = &self.nodes[i];
            let bytes = node.bytes.expect("validated backing capacity");
            let native_amount = if bytes == 0 {
                DomainMemoryCharge::default()
            } else {
                let placement = source
                    .placement(&node.identity)
                    .expect("validated backing placement");
                if placement.domains().contains(&domain) {
                    charge(bytes, placement)
                } else {
                    DomainMemoryCharge::default()
                }
            };
            let mut amount = native_amount;
            if domain == topology.host_domain() {
                amount.accounted_bytes = amount
                    .accounted_bytes
                    .checked_add(
                        source
                            .host_control_bytes(&node.identity)
                            .expect("validated backing controls"),
                    )
                    .ok_or(WorkspaceReportError::Overflow)?;
            }
            if amount == DomainMemoryCharge::default() {
                continue;
            }
            let marks = self.marks[i];
            if marks & N != 0 {
                add_to(&mut result.tensor_buffers.total, native_amount)?;
                if marks & C != 0 {
                    add_to(&mut result.tensor_buffers.retained, native_amount)?;
                } else {
                    add_to(&mut result.tensor_buffers.transient, native_amount)?;
                }
            }
            if marks & B == 0 {
                if let Some(residual) = &mut result.residual {
                    add_to(&mut residual.tensor_buffers.total, native_amount)?;
                    if marks & C != 0 {
                        add_to(&mut residual.tensor_buffers.retained, native_amount)?;
                    } else {
                        add_to(&mut residual.tensor_buffers.transient, native_amount)?;
                    }
                }
            }
            if marks & C != 0 {
                add_population(&mut result.closing, amount, node.maximum_allocations)?;
            }
            if marks & O != 0 {
                add_population(
                    result.opening.as_mut().expect("seeded opening"),
                    amount,
                    node.maximum_allocations,
                )?;
            }
            if marks & N != 0 {
                add_to(&mut result.total, amount)?;
                if marks & C != 0 {
                    add_to(&mut result.retained, amount)?;
                } else {
                    add_to(&mut result.transient, amount)?;
                }
            }
            if marks & O != 0 && marks & C == 0 {
                add_to(
                    &mut result.state.as_mut().expect("seeded state").displaced,
                    amount,
                )?;
            }
            if marks & B == 0 {
                if let Some(residual) = &mut result.residual {
                    add_to(&mut residual.total, amount)?;
                    if marks & O != 0 {
                        add_population(&mut residual.opening, amount, node.maximum_allocations)?;
                    }
                    if marks & C != 0 {
                        add_population(&mut residual.closing, amount, node.maximum_allocations)?;
                        add_to(&mut residual.retained, amount)?;
                    } else {
                        add_to(&mut residual.transient, amount)?;
                        if marks & O != 0 {
                            add_to(&mut residual.displaced, amount)?;
                        }
                    }
                }
            }
        }
        if let Some(state) = &mut result.state {
            state.retained = result.closing.charge;
            state.transient = result.transient.checked_add(state.displaced)?;
        }
        if let Some(residual) = &mut result.residual {
            residual.host_workspace_bytes = result.host_workspace_bytes;
        }
        Ok(result)
    }

    pub(in crate::workspace::report) fn candidate_placements<'a, G: Graph<Root = R>>(
        &'a self,
        source: &'a G,
    ) -> impl Iterator<Item = (u64, &'a MemoryPlacement)> + 'a {
        self.nodes.iter().filter_map(move |node| {
            let placement = source.placement(&node.identity)?;
            matches!(placement.kind(), MemoryPlacementKind::Possible { .. })
                .then_some((node.bytes?, placement))
        })
    }
}
