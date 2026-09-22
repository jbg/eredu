//! Branch reduction over retained allocation sources, after domain resolution.
use super::*;
use eredu_core::{DomainMemoryRequirements, MemoryTopology};

#[derive(Debug)]
pub(in crate::workspace) struct ScratchDomains {
    pub(in crate::workspace) native: DomainMemoryRequirements,
    /// Absent while any retained source lacks its required host controls.
    pub(in crate::workspace) all: Option<DomainMemoryRequirements>,
    pub(in crate::workspace) bytes: u64,
    pub(in crate::workspace) controls: Option<u64>,
}
impl WorkspaceAllocationPopulation {
    /// Native backing capacity before domain placement. Alternative occurrences
    /// take a scalar peak only for diagnostics; admission uses resolved domains.
    pub fn backing_bytes(&self) -> Option<u64> {
        match &self.0.domains {
            Some(source) => Some(source.bytes),
            None => self
                .allocations()
                .iter()
                .try_fold(0u64, |n, row| n.checked_add(row.bytes)),
        }
    }
    pub(in crate::workspace) fn host_control_bytes(&self) -> Option<u64> {
        match &self.0.domains {
            Some(source) => source.controls,
            None => self
                .allocations()
                .iter()
                .try_fold(0u64, |n, row| n.checked_add(row.host_control_bytes?)),
        }
    }
    pub(in crate::workspace) fn domain_population(&self) -> Option<&ScratchDomains> {
        self.0.domains.as_ref()
    }
    pub(in crate::workspace) fn validate_domains(
        &self,
        topology: &MemoryTopology,
    ) -> Result<(), Error> {
        if let Some(source) = self.domain_population() {
            source
                .native
                .validate(topology)
                .map_err(Error::backend_retained_source)?;
            if let Some(all) = &source.all {
                all.validate(topology)
                    .map_err(Error::backend_retained_source)?;
            }
        }
        for row in self.allocations() {
            if let Some(placement) = &row.placement {
                placement
                    .validate(topology)
                    .map_err(Error::backend_retained_source)?;
            }
        }
        Ok(())
    }
    fn resolved(&self, context: &WorkspaceContext) -> Result<ScratchDomains, Error> {
        let topology = context.memory_topology().ok_or_else(|| {
            Error::backend_retained_source(WorkspacePlacementError::MissingTopology)
        })?;
        self.validate_domains(topology)?;
        if let Some(source) = self.domain_population() {
            match &source.all {
                Some(all) => context.charge_requirements(&[&source.native, all])?,
                None => context.charge_requirements(&[&source.native])?,
            }
            return Ok(ScratchDomains {
                native: source.native.clone(),
                all: source.all.clone(),
                bytes: source.bytes,
                controls: source.controls,
            });
        }
        let count = self
            .allocations()
            .iter()
            .filter(|row| {
                row.placement
                    .as_ref()
                    .is_some_and(|p| matches!(p.kind(), MemoryPlacementKind::Possible { .. }))
            })
            .count();
        let mut bytes = DomainMemoryRequirements::construction_backing_bytes(topology, count)
            .map_err(Error::backend_retained_source)?;
        for row in self.allocations() {
            if let Some(placement) = &row.placement {
                bytes = bytes
                    .checked_add(
                        placement
                            .clone_backing_bytes()
                            .map_err(Error::backend_retained_source)?,
                    )
                    .ok_or(WorkspaceMetadataError::Overflow)?;
            }
        }
        context.charge_metadata(
            usize::try_from(bytes).map_err(|_| WorkspaceMetadataError::Overflow)?,
        )?;
        let mut native = DomainMemoryRequirements::zero_with_allowance_capacity(topology, count);
        for row in self.allocations() {
            if row.bytes == 0 {
                continue;
            }
            let placement = row
                .placement
                .as_ref()
                .ok_or(WorkspaceMetadataError::Report(
                    WorkspaceReportError::IncompletePlacement,
                ))?;
            native
                .add_allocation(row.bytes, placement)
                .map_err(Error::backend_retained_source)?;
        }
        let controls = self.host_control_bytes();
        let all = match controls {
            Some(controls) => {
                context.charge_requirements(&[&native])?;
                let mut all = native.clone();
                all.add_allocation(
                    controls,
                    &MemoryPlacement::fixed(topology, topology.host_domain())
                        .map_err(Error::backend_retained_source)?,
                )
                .map_err(Error::backend_retained_source)?;
                Some(all)
            }
            // Retain the known backing attribution and unknown control fact.
            // A later complete-domain report must refuse this contribution.
            None => None,
        };
        Ok(ScratchDomains {
            native,
            all,
            bytes: self
                .backing_bytes()
                .ok_or(WorkspaceMetadataError::Overflow)?,
            controls,
        })
    }
}
impl WorkspaceContext {
    pub(in crate::workspace) fn charge_requirements(
        &self,
        parts: &[&DomainMemoryRequirements],
    ) -> Result<(), Error> {
        let mut bytes = std::mem::size_of::<(
            DomainMemoryRequirements,
            Result<DomainMemoryRequirements, eredu_core::MemoryDomainError>,
            &[&DomainMemoryRequirements],
        )>();
        for part in parts {
            bytes = bytes
                .checked_add(
                    usize::try_from(
                        part.clone_backing_bytes()
                            .map_err(Error::backend_retained_source)?,
                    )
                    .map_err(|_| WorkspaceMetadataError::Overflow)?,
                )
                .ok_or(WorkspaceMetadataError::Overflow)?;
        }
        Ok(self.charge_metadata(bytes)?)
    }
    /// Retain alternative completed child paths, comparing physical domains only
    /// after their actual allocation descriptors have been resolved. Every
    /// alternative retains its source identities and candidate-placement basis.
    pub fn peak_scratch_populations(
        &self,
        sources: &[&WorkspaceAllocationPopulation],
    ) -> Result<WorkspaceAllocationPopulation, Error> {
        let mut repeated = self.metadata_vec(sources.len())?;
        repeated.extend(sources.iter().map(|source| (*source, 1)));
        self.compose_scratch_domains(&repeated, true)
    }
    pub(super) fn compose_scratch_domains(
        &self,
        sources: &[(&WorkspaceAllocationPopulation, usize)],
        peak: bool,
    ) -> Result<WorkspaceAllocationPopulation, Error> {
        self.charge_metadata(std::mem::size_of::<(
            ScratchDomains,
            ScratchDomains,
            Option<ScratchDomains>,
            ScratchPopulation,
            WorkspaceAllocationPopulation,
            bool,
        )>())?;
        let topology = self.memory_topology().ok_or_else(|| {
            Error::backend_retained_source(WorkspacePlacementError::MissingTopology)
        })?;
        let mut result: Option<ScratchDomains> = None;
        let mut children = self.metadata_vec(sources.len())?;
        for (source, repeat) in sources {
            if *repeat == 0 {
                continue;
            }
            children.push((*source).clone());
            let part = source.resolved(self)?;
            for _ in 0..*repeat {
                result = Some(match result {
                    None => {
                        match &part.all {
                            Some(all) => self.charge_requirements(&[&part.native, all])?,
                            None => self.charge_requirements(&[&part.native])?,
                        }
                        ScratchDomains {
                            native: part.native.clone(),
                            all: part.all.clone(),
                            bytes: part.bytes,
                            controls: part.controls,
                        }
                    }
                    Some(prior) => {
                        match (&prior.all, &part.all) {
                            (Some(a), Some(b)) => {
                                self.charge_requirements(&[&prior.native, &part.native, a, b])?
                            }
                            _ => self.charge_requirements(&[&prior.native, &part.native])?,
                        }
                        let native = if peak {
                            prior.native.checked_peak(&part.native)
                        } else {
                            prior.native.checked_add(&part.native)
                        }
                        .map_err(Error::backend_retained_source)?;
                        let all = match (&prior.all, &part.all) {
                            (Some(a), Some(b)) => Some(
                                if peak {
                                    a.checked_peak(b)
                                } else {
                                    a.checked_add(b)
                                }
                                .map_err(Error::backend_retained_source)?,
                            ),
                            _ => None,
                        };
                        let bytes = if peak {
                            prior.bytes.max(part.bytes)
                        } else {
                            prior
                                .bytes
                                .checked_add(part.bytes)
                                .ok_or(WorkspaceMetadataError::Overflow)?
                        };
                        let controls = match (prior.controls, part.controls) {
                            (Some(a), Some(b)) => Some(if peak {
                                a.max(b)
                            } else {
                                a.checked_add(b).ok_or(WorkspaceMetadataError::Overflow)?
                            }),
                            _ => None,
                        };
                        ScratchDomains {
                            native,
                            all,
                            bytes,
                            controls,
                        }
                    }
                });
            }
        }
        let domains = match result {
            Some(result) => result,
            None => {
                let bytes = DomainMemoryRequirements::construction_backing_bytes(topology, 0)
                    .map_err(Error::backend_retained_source)?;
                self.charge_metadata(
                    usize::try_from(
                        bytes
                            .checked_mul(2)
                            .ok_or(WorkspaceMetadataError::Overflow)?,
                    )
                    .map_err(|_| WorkspaceMetadataError::Overflow)?,
                )?;
                ScratchDomains {
                    native: DomainMemoryRequirements::zero(topology),
                    all: Some(DomainMemoryRequirements::zero(topology)),
                    bytes: 0,
                    controls: Some(0),
                }
            }
        };
        Ok(WorkspaceAllocationPopulation(self.metadata_rc(
            ScratchPopulation {
                allocations: Vec::new(),
                roots: Vec::new(),
                children,
                domains: Some(domains),
                funding: self.metadata_funding(),
            },
        )?))
    }
}
