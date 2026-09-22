//! Retained ordinary bank equations and their authenticated request occurrences.
use super::super::resident_recipe::{CertifiedSpanStorage, OrdinaryCpuPopulation};
use super::*;
use crate::backend::runtime::residency::parameter_bank::{
    MlxIndexedMovement, OrdinaryBankHostSource, OrdinaryIndexedOccurrence,
    OrdinaryIndexedRequestOwner, OrdinaryIndexedRequestProgram, OrdinaryIndexedResidencyFacts,
};
use std::{cell::RefCell, rc::Rc};

struct Directory {
    source: AddressableSources,
    quotes: RefCell<Vec<OrdinaryEquation>>,
}
#[derive(Clone)]
pub(crate) struct OrdinaryAddressableSources(Rc<Directory>);
impl std::fmt::Debug for OrdinaryAddressableSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryAddressableSources")
            .field("source", &self.0.source)
            .finish_non_exhaustive()
    }
}
#[derive(Clone)]
pub(crate) struct OrdinaryEquation(Rc<Equation>);
struct Equation {
    source: AddressableEquationSource,
    residency: OrdinaryIndexedResidencyFacts,
    discovery: Option<super::super::cpu::OrdinaryIndexedNumericalFacts>,
    allocation: NativeAllocationFacts,
    facts: Option<OrdinaryFacts>,
}
impl std::fmt::Debug for OrdinaryEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryEquation")
            .field("declaration", &self.0.source.declaration)
            .finish_non_exhaustive()
    }
}
impl OrdinaryAddressableSources {
    pub(crate) fn new(source: AddressableSources) -> Result<Self, Error> {
        source
            .funding()
            .reserve_metadata(
                super::sources::shared_bytes::<Directory>()
                    .and_then(|n| n.checked_add(size_of::<(Self, Result<Self, Error>)>()))
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )
            .map_err(WorkspaceMetadataError::Funding)?;
        Ok(Self(Rc::new(Directory {
            source,
            quotes: RefCell::new(Vec::new()),
        })))
    }
    fn mechanism(&self) -> ResidentExecutionMechanisms {
        self.0.source.mechanism().ordinary_storage()
    }
    pub(crate) fn funding(&self) -> &HostMetadataFunding {
        self.0.source.funding()
    }
    pub(crate) fn equation(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<OrdinaryEquation, Error> {
        let context =
            WorkspaceContext::new_with_metadata_funding(self.mechanism(), self.funding().clone())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "ordinary indexed occurrence differs from its retained bank"
            ))
        };
        context.charge_metadata(size_of::<(
            Self,
            OrdinaryEquation,
            Result<OrdinaryEquation, Error>,
            WorkspaceOperationView<'_>,
        )>())?;
        if let Some(quote) = self
            .0
            .quotes
            .try_borrow()
            .map_err(|_| invalid())?
            .iter()
            .find(|quote| quote.0.source.matches(operation))
        {
            return Ok(quote.clone());
        }
        let WorkspaceOperationKindView::AddressableRegion(region) = operation.kind else {
            return Err(invalid());
        };
        let bank = self
            .0
            .source
            .bank(region.as_view().bank)
            .ok_or_else(invalid)?;
        let equation = AddressableEquationSource::prepare(
            bank,
            operation,
            self.mechanism(),
            self.funding(),
            |identity, spec| {
                let host = OrdinaryBankHostSource::new(
                    identity.source(),
                    identity.first().ok_or_else(invalid)?,
                    self.funding(),
                )
                .map_err(|cause| context.metadata_source(cause))?;
                let bytes = match spec {
                    WorkspaceGroupedBank::Linear(spec) => host.linear_metadata_bytes(spec),
                    WorkspaceGroupedBank::GatedProduct(spec) => {
                        host.gated_product_metadata_bytes(spec)
                    }
                    WorkspaceGroupedBank::Relu2(spec) => host.relu2_metadata_bytes(spec),
                }
                .map_err(|cause| context.metadata_source(cause))?;
                u64::try_from(bytes).map_err(|_| invalid())
            },
        )?;
        context.charge_metadata(super::sources::shared_bytes::<Equation>().ok_or_else(invalid)?)?;
        let first = equation.identity().first().ok_or_else(invalid)?;
        let residency = bank
            .inspect_ordinary_residency(
                first,
                self.0.source.runtime(),
                self.mechanism().allocation(),
                self.funding(),
            )
            .map_err(|cause| context.metadata_source(cause))?;
        residency
            .validate(bank, first)
            .map_err(|cause| context.metadata_source(cause))?;
        let discovery = match self.mechanism() {
            ResidentExecutionMechanisms::Cpu { cpu, .. } => Some(
                cpu.ordinary_indexed_numerical_facts(first, equation.inputs[1].dtype())
                    .map_err(|cause| context.metadata_source(cause))?,
            ),
            ResidentExecutionMechanisms::Metal(_) => None,
        };
        let facts = OrdinaryFacts::prepare(
            &equation,
            &residency,
            discovery,
            &context,
            self.funding(),
            self.mechanism().allocation(),
        )?;
        let quote = OrdinaryEquation(Rc::new(Equation {
            source: equation,
            residency,
            discovery,
            allocation: self.mechanism().allocation(),
            facts,
        }));
        let mut quotes = self.0.quotes.try_borrow_mut().map_err(|_| invalid())?;
        context.reserve_metadata_vec(&mut quotes, 1)?;
        quotes.push(quote.clone());
        Ok(quote)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

impl super::facts::AddressableFactQuote for OrdinaryEquation {
    fn matches(&self, operation: WorkspaceOperationView<'_>) -> bool {
        self.0.source.matches(operation)
    }
    fn outputs(&self) -> &[WorkspaceLayout] {
        &self.0.source.outputs
    }
    fn allocation(&self) -> NativeAllocationFacts {
        self.0.allocation
    }
    fn storage_bytes(&self) -> Option<u64> {
        Some(self.storage()?.mutable_bytes())
    }
    fn host_bytes(&self) -> Option<u64> {
        OrdinaryEquation::host_bytes(self)
    }
    fn scratch_host_controls(&self) -> Option<Option<u64>> {
        Some(
            self.0
                .facts
                .as_ref()
                .map(|facts| facts.scratch_host_controls),
        )
    }
    fn host_basis(&self) -> &'static str {
        "ordinary indexed equation, acquisition and completed transfer host sources; reachable numerical Eval and transfer payload placement are composed by the enclosing request"
    }
}
impl super::facts::AddressableFactSource for OrdinaryAddressableSources {
    type Quote = OrdinaryEquation;
    fn funding(&self) -> &HostMetadataFunding {
        self.funding()
    }
    fn quote(&self, operation: WorkspaceOperationView<'_>) -> Result<Self::Quote, Error> {
        self.equation(operation)
    }
    fn observation_layout(
        &self,
        source: WorkspaceAddressableRegionView<'_>,
        inputs: &[WorkspaceLayout],
        context: &WorkspaceContext,
    ) -> Result<WorkspaceAddressableObservationLayout, Error> {
        self.0.source.observation_layout(source, inputs, context)
    }
}

struct Occurrence {
    maximum: OrdinaryEquation,
    alternatives: Vec<OrdinaryEquation>,
    local: bool,
}
/// Ordered descriptions retain the exact source identities and every local
/// row class. Installation still requires the admitted ordinary Work owner.
pub(crate) struct OrdinaryAddressableProgram {
    occurrences: Vec<Occurrence>,
    source: OrdinaryAddressableSources,
    facts: Option<OrdinaryFacts>,
    installation_host_bytes: u64,
}
impl std::fmt::Debug for OrdinaryAddressableProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdinaryAddressableProgram")
            .field("occurrences", &self.occurrences.len())
            .finish_non_exhaustive()
    }
}
impl OrdinaryIndexedRequestProgram for OrdinaryAddressableProgram {
    fn len(&self) -> usize {
        self.occurrences.len()
    }
    fn occurrence(&self, index: usize) -> Option<OrdinaryIndexedOccurrence<'_>> {
        let value = self.occurrences.get(index)?;
        Some(OrdinaryIndexedOccurrence {
            identity: value.maximum.0.source.identity(),
            residency: &value.maximum.0.residency,
            declaration: value.maximum.0.source.declaration.as_view(),
            local: value.local,
        })
    }
}
impl OrdinaryAddressableSources {
    pub(crate) fn program(
        &self,
        operations: &[WorkspaceOperation],
    ) -> Result<Rc<OrdinaryAddressableProgram>, Error> {
        let context = WorkspaceContext::new_with_metadata_funding(
            MlxAddressableWorkspaceMechanisms::new(self.mechanism(), self.clone()),
            self.funding().clone(),
        )?;
        let invalid = || {
            context.metadata_error(format_args!(
                "ordinary indexed program lacks its exact source rows"
            ))
        };
        context.charge_metadata(
            super::sources::shared_bytes::<OrdinaryAddressableProgram>()
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        Occurrence,
                        Self,
                        Result<Rc<OrdinaryAddressableProgram>, Error>,
                    )>())
                })
                .ok_or_else(invalid)?,
        )?;
        let count = operations.iter().filter(|operation| {
            matches!(operation.kind, WorkspaceOperationKind::AddressableRegion(_))
                || matches!(&operation.kind, WorkspaceOperationKind::ExpertRegion(region) if region.as_view().addressable.is_some())
        }).count();
        let mut occurrences = context.metadata_vec(count)?;
        for operation in operations {
            if matches!(operation.kind, WorkspaceOperationKind::AddressableRegion(_)) {
                occurrences.push(Occurrence {
                    maximum: self.equation(operation.as_view())?,
                    alternatives: context.metadata_vec(0)?,
                    local: false,
                });
            } else if matches!(&operation.kind, WorkspaceOperationKind::ExpertRegion(region) if region.as_view().addressable.is_some())
            {
                occurrences.push(self.local_occurrence(operation.as_view(), &context)?);
            }
        }
        self.finish_program(occurrences, &context)
    }

    /// The same local-row alternatives used by the enclosing ordered program.
    /// This is a retained source quotation, not another request installation.
    pub(crate) fn local_program(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Rc<OrdinaryAddressableProgram>, Error> {
        let context = WorkspaceContext::new_with_metadata_funding(
            MlxAddressableWorkspaceMechanisms::new(self.mechanism(), self.clone()),
            self.funding().clone(),
        )?;
        context.charge_metadata(
            super::sources::shared_bytes::<OrdinaryAddressableProgram>()
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        Self,
                        Occurrence,
                        Result<Rc<OrdinaryAddressableProgram>, Error>,
                    )>())
                })
                .ok_or_else(ordinary_overflow)?,
        )?;
        let mut occurrences = context.metadata_vec(1)?;
        occurrences.push(self.local_occurrence(operation, &context)?);
        self.finish_program(occurrences, &context)
    }
    fn local_occurrence(
        &self,
        operation: WorkspaceOperationView<'_>,
        context: &WorkspaceContext,
    ) -> Result<Occurrence, Error> {
        let invalid = || {
            context.metadata_error(format_args!(
                "ordinary local indexed program differs from its actual row source"
            ))
        };
        let local = super::local::LocalAddressableSource::prepare(operation, context)?;
        let report = local.report(local.maximum, context)?;
        if report.operations.len() != 1 {
            return Err(invalid());
        }
        let maximum = self.equation(report.operations[0].as_view())?;
        let rows = super::row_candidates::local_row_candidates(
            local.source.kernel,
            local.maximum,
            local.source.chunks.chunk_rows,
            context,
        )?;
        let mut alternatives = context.metadata_vec(rows.len())?;
        for rows in rows {
            let report = local.report(rows, context)?;
            if report.operations.len() != 1 {
                return Err(invalid());
            }
            alternatives.push(self.equation(report.operations[0].as_view())?);
        }
        Ok(Occurrence {
            maximum,
            alternatives,
            local: true,
        })
    }
    fn finish_program(
        &self,
        occurrences: Vec<Occurrence>,
        context: &WorkspaceContext,
    ) -> Result<Rc<OrdinaryAddressableProgram>, Error> {
        let mut aggregate: Option<OrdinaryFacts> = None;
        let mut complete = true;
        for occurrence in &occurrences {
            let Some(maximum) = &occurrence.maximum.0.facts else {
                complete = false;
                continue;
            };
            let mut branch = maximum.copy(&context)?;
            for alternative in &occurrence.alternatives {
                let Some(facts) = &alternative.0.facts else {
                    complete = false;
                    continue;
                };
                branch = branch.combine(facts, false, &context)?;
            }
            aggregate = Some(match aggregate {
                None => branch,
                Some(prior) => prior.combine(&branch, true, &context)?,
            });
        }
        let mut program = OrdinaryAddressableProgram {
            occurrences,
            source: self.clone(),
            facts: if complete { aggregate } else { None },
            installation_host_bytes: 0,
        };
        program.installation_host_bytes = u64::try_from(
            OrdinaryIndexedRequestOwner::runtime_control_bytes(&program)
                .ok_or_else(ordinary_overflow)?,
        )
        .map_err(|_| ordinary_overflow())?;
        Ok(Rc::new(program))
    }
}

/// Complete source contributions for one ordinary occurrence. Numerical Eval
/// controls are deliberately absent: its lazy inputs belong to the parent DAG.
struct OrdinaryFacts {
    storage: CertifiedSpanStorage,
    calls: Option<super::super::OrdinaryCallControls>,
    scratch_host_controls: u64,
    raw: OrdinaryCpuPopulation,
    host_bytes: u64,
    frontiers: [(usize, usize); 2],
    transfers: eredu_core::DomainMemoryRequirements,
    transfer_allocations: usize,
    platform_events: usize,
    capture_publications: usize,
    validation_roots: usize,
    validation_storage: CertifiedSpanStorage,
}
fn ordinary_overflow() -> Error {
    WorkspaceMetadataError::Overflow.into()
}
impl OrdinaryFacts {
    fn prepare(
        source: &AddressableEquationSource,
        residency: &OrdinaryIndexedResidencyFacts,
        discovery: Option<super::super::cpu::OrdinaryIndexedNumericalFacts>,
        context: &WorkspaceContext,
        funding: &HostMetadataFunding,
        allocation: NativeAllocationFacts,
    ) -> Result<Option<Self>, Error> {
        let Some(discovery) = discovery else {
            return Ok(None);
        };
        let (
            Some(equation),
            Some(indexed),
            Some(staging),
            Some(destination),
            Some(metadata),
            Some(observed),
            Some(transfer),
        ) = (
            source.population.ordinary_cpu_population(),
            discovery.raw_population,
            residency.host_staging(),
            residency.destination(),
            residency.host_control_bytes(),
            residency.observed_host_control_bytes(),
            residency.transfer_native_controls(),
        )
        else {
            return Ok(None);
        };
        let first = residency.first();
        let chunks = first.plan().len();
        let repetitions = u64::try_from(chunks).map_err(|_| ordinary_overflow())?;
        let storage = source
            .population
            .storage()
            .checked_add(discovery.backing_bytes, discovery.backing_births)
            .ok_or_else(ordinary_overflow)?;
        let scratch_host_controls = storage
            .maximum_births()
            .checked_sub(source.outputs.len())
            .and_then(|n| u64::try_from(n).ok())
            .and_then(|n| n.checked_mul(allocation.host_control_bytes()?))
            .ok_or_else(ordinary_overflow)?;
        let raw = equation.append(indexed).ok_or_else(ordinary_overflow)?;
        let root_metadata = crate::backend::managed_memory::ordinary_root_metadata_bytes()
            .map_err(|e| context.metadata_source(e))?;
        let destination_controls = u64::try_from(safemlx::physical_backing_control_bytes())
            .map_err(|_| ordinary_overflow())?
            .checked_add(root_metadata)
            .ok_or_else(ordinary_overflow)?;
        let staging_controls = root_metadata
            .checked_mul(u64::try_from(staging.allocations).map_err(|_| ordinary_overflow())?)
            .ok_or_else(ordinary_overflow)?;
        let destination_controls = destination_controls
            .checked_mul(u64::try_from(destination.allocations).map_err(|_| ordinary_overflow())?)
            .ok_or_else(ordinary_overflow)?;
        let per_chunk = [
            u64::try_from(metadata).map_err(|_| ordinary_overflow())?,
            u64::try_from(observed).map_err(|_| ordinary_overflow())?,
            transfer
                .host_allowance_with_ledger_metadata()
                .map_err(|e| context.metadata_source(e))?,
            staging_controls,
            destination_controls,
        ]
        .into_iter()
        .try_fold(0u64, |sum, bytes| {
            sum.checked_add(bytes).ok_or_else(ordinary_overflow)
        })?;
        let calls = source
            .ordinary_calls
            .map(|calls| {
                calls
                    .append(super::super::OrdinaryCallControls {
                        metadata_bytes: discovery.fixed_host_controls,
                        observed: discovery.caller_native_controls,
                    })
                    .ok_or_else(ordinary_overflow)
            })
            .transpose()?;
        let mut host_bytes = source
            .host_bytes
            .checked_add(
                per_chunk
                    .checked_mul(repetitions)
                    .ok_or_else(ordinary_overflow)?,
            )
            .and_then(|n| n.checked_add(residency.source_pin_control_bytes().unwrap_or(0)))
            .and_then(|n| {
                n.checked_add(
                    u64::try_from(MlxIndexedMovement::ordinary_invocation_control_bytes()?).ok()?,
                )
            })
            .ok_or_else(ordinary_overflow)?;
        for ordinal in 0..chunks {
            let census = eredu_runtime::expert::AddressableChunkCensus::new(
                first.bank(),
                first.unit(),
                first.plan(),
                ordinal,
                first.access(),
            )
            .ok_or_else(ordinary_overflow)?;
            let host = OrdinaryBankHostSource::new(source.identity().source(), census, funding)
                .map_err(|e| context.metadata_source(e))?;
            let acquisition = host
                .acquisition_metadata_bytes()
                .map_err(|e| context.metadata_source(e))?;
            let bytes = MlxIndexedMovement::ordinary_chunk_control_bytes(census)
                .and_then(|n| n.checked_add(acquisition))
                .ok_or_else(ordinary_overflow)?;
            host_bytes = host_bytes
                .checked_add(u64::try_from(bytes).map_err(|_| ordinary_overflow())?)
                .ok_or_else(ordinary_overflow)?;
        }
        let topology = crate::backend::managed_memory::cold_topology().ok_or_else(|| {
            context.metadata_error(format_args!(
                "ordinary source has no retained physical topology"
            ))
        })?;
        let reports = eredu_runtime::working_memory::WorkspaceReportMetadata::new(context);
        let placed = |part: crate::backend::runtime::residency::parameter_bank::OrdinaryResidencyPhysical| {
            reports.placed_requirements(
                topology, part.bytes.checked_mul(repetitions).ok_or_else(ordinary_overflow)?, part.placement,
            ).map_err(|e| context.metadata_source(e))
        };
        let transfers = reports
            .combine_domain_requirements(&placed(staging)?, &placed(destination)?, true)
            .map_err(|e| context.metadata_source(e))?;
        Ok(Some(Self {
            storage,
            calls,
            scratch_host_controls,
            raw,
            host_bytes,
            frontiers: [
                (equation.nested_roots, equation.nested_completions),
                (2, discovery.discovery_completions),
            ],
            transfers,
            transfer_allocations: staging
                .allocations
                .checked_add(destination.allocations)
                .and_then(|n| n.checked_mul(chunks))
                .ok_or_else(ordinary_overflow)?,
            platform_events: transfer
                .platform_events
                .checked_mul(chunks)
                .ok_or_else(ordinary_overflow)?,
            capture_publications: source.capture_publications,
            validation_roots: source.population.ordinary_validation_population().0,
            validation_storage: source.population.ordinary_validation_population().1,
        }))
    }
    fn combine(
        &self,
        other: &Self,
        simultaneous: bool,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let combine = |a: usize, b: usize| {
            if simultaneous {
                a.checked_add(b).ok_or_else(ordinary_overflow)
            } else {
                Ok(a.max(b))
            }
        };
        let storage = if simultaneous {
            self.storage
                .checked_add(
                    other.storage.mutable_bytes(),
                    other.storage.maximum_births(),
                )
                .ok_or_else(ordinary_overflow)?
        } else {
            self.storage.union(other.storage)
        };
        let raw = if simultaneous {
            self.raw.append(other.raw).ok_or_else(ordinary_overflow)?
        } else {
            self.raw.union(other.raw)
        };
        let mut frontiers = self.frontiers;
        for (out, other) in frontiers.iter_mut().zip(other.frontiers) {
            out.0 = out.0.max(other.0);
            out.1 = combine(out.1, other.1)?;
        }
        let calls = self
            .calls
            .zip(other.calls)
            .map(|(left, right)| {
                if simultaneous {
                    left.append(right).ok_or_else(ordinary_overflow)
                } else {
                    Ok(left.union(right))
                }
            })
            .transpose()?;
        Ok(Self {
            storage,
            calls,
            raw,
            frontiers,
            // This aggregate is consumed by the parent, never emitted as one
            // operation; retain the independently priced scratch controls.
            scratch_host_controls: if simultaneous {
                self.scratch_host_controls
                    .checked_add(other.scratch_host_controls)
                    .ok_or_else(ordinary_overflow)?
            } else {
                self.scratch_host_controls.max(other.scratch_host_controls)
            },
            host_bytes: if simultaneous {
                self.host_bytes
                    .checked_add(other.host_bytes)
                    .ok_or_else(ordinary_overflow)?
            } else {
                self.host_bytes.max(other.host_bytes)
            },
            transfers: eredu_runtime::working_memory::WorkspaceReportMetadata::new(context)
                .combine_domain_requirements(&self.transfers, &other.transfers, simultaneous)
                .map_err(|e| context.metadata_source(e))?,
            transfer_allocations: combine(self.transfer_allocations, other.transfer_allocations)?,
            platform_events: combine(self.platform_events, other.platform_events)?,
            capture_publications: combine(self.capture_publications, other.capture_publications)?,
            validation_roots: combine(self.validation_roots, other.validation_roots)?,
            validation_storage: if simultaneous {
                self.validation_storage
                    .checked_add(
                        other.validation_storage.mutable_bytes(),
                        other.validation_storage.maximum_births(),
                    )
                    .ok_or_else(ordinary_overflow)?
            } else {
                self.validation_storage.union(other.validation_storage)
            },
        })
    }
    fn copy(&self, context: &WorkspaceContext) -> Result<Self, Error> {
        Ok(Self {
            storage: self.storage,
            calls: self.calls,
            scratch_host_controls: self.scratch_host_controls,
            raw: self.raw,
            host_bytes: self.host_bytes,
            frontiers: self.frontiers,
            transfers: eredu_runtime::working_memory::WorkspaceReportMetadata::new(context)
                .clone_domain_requirements(&self.transfers)
                .map_err(|e| context.metadata_source(e))?,
            transfer_allocations: self.transfer_allocations,
            platform_events: self.platform_events,
            capture_publications: self.capture_publications,
            validation_roots: self.validation_roots,
            validation_storage: self.validation_storage,
        })
    }
}
impl OrdinaryEquation {
    /// First absent producer in the same completeness gate used by this quote.
    /// The static diagnostic identifies no storage and grants no execution.
    pub(crate) fn missing_source(&self) -> Option<&'static str> {
        if self.0.facts.is_some() {
            return None;
        }
        let source = &self.0;
        let Some(discovery) = source.discovery else {
            return Some("ordinary indexed discovery has no selected native source");
        };
        if let Some(missing) = source.source.population.ordinary_cpu_missing_source() {
            return Some(missing);
        }
        if discovery.raw_population.is_none() {
            return Some("ordinary indexed discovery has no retained reachable population");
        }
        if source.residency.host_staging().is_none() {
            return Some("ordinary indexed host staging has no physical source placement");
        }
        if source.residency.destination().is_none() {
            return Some("ordinary indexed copy destination has no physical source placement");
        }
        if source.residency.foreground_read_control_bytes().is_none() {
            return Some("ordinary indexed retained read or materialization has no host source");
        }
        if source.residency.transfer_native_controls().is_none() {
            return Some("ordinary indexed transfer has no qualified native completion source");
        }
        if source.residency.observed_host_control_bytes().is_none() {
            return Some("ordinary indexed host staging has no observed constructor source");
        }
        Some("ordinary indexed source composition is incomplete")
    }
    /// Additional actual grouped/parent/replacement Slice and discovery/remap
    /// safe calls, plus caller root-vector backing. Provider, acquisition,
    /// transfer and synchronous completion shells are already in host_bytes;
    /// primitive frontend/Eval controls belong to the enclosing raw DAG.
    pub(crate) fn ordinary_calls(&self) -> Option<super::super::OrdinaryCallControls> {
        self.0.facts.as_ref()?.calls
    }
    /// Numerical payloads only; separately placed source transfers are below.
    pub(crate) fn storage(&self) -> Option<CertifiedSpanStorage> {
        Some(self.0.facts.as_ref()?.storage)
    }
    /// Actual predicates retained by the enclosing TokenValidationScope, and
    /// their already included conservative producer storage. Add the roots to
    /// the parent completion; carry storage across prefill, never debit twice.
    pub(crate) fn validation_population(&self) -> Option<(usize, CertifiedSpanStorage)> {
        let facts = self.0.facts.as_ref()?;
        Some((facts.validation_roots, facts.validation_storage))
    }
    /// Construction/worker populations to join the enclosing lazy routing DAG.
    pub(crate) fn raw_cpu_population(&self) -> Option<OrdinaryCpuPopulation> {
        Some(self.0.facts.as_ref()?.raw)
    }
    /// Actual child and discovery completion populations, excluding parent exit.
    pub(crate) fn completion_frontiers(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.0
            .facts
            .iter()
            .flat_map(|facts| facts.frontiers)
            .filter(|&(_, n)| n != 0)
    }
    /// Source and equation metadata already emitted in the operation host fact;
    /// numerical backing controls and combined graph/Eval remain separate.
    pub(crate) fn host_bytes(&self) -> Option<u64> {
        Some(self.0.facts.as_ref()?.host_bytes)
    }
    /// Simultaneous per-domain transfer payloads only, across actual chunks.
    pub(crate) fn transfer_requirements(&self) -> Option<&eredu_core::DomainMemoryRequirements> {
        Some(&self.0.facts.as_ref()?.transfers)
    }
}

impl OrdinaryAddressableProgram {
    /// Final logical layouts of one local occurrence at its maximum row count.
    pub(crate) fn local_outputs(&self) -> Option<&[WorkspaceLayout]> {
        let [occurrence] = self.occurrences.as_slice() else {
            return None;
        };
        occurrence
            .local
            .then_some(&occurrence.maximum.0.source.outputs)
            .map(Vec::as_slice)
    }
    pub(crate) fn missing_source(&self) -> Option<&'static str> {
        self.occurrences.iter().find_map(|occurrence| {
            occurrence.maximum.missing_source().or_else(|| {
                occurrence
                    .alternatives
                    .iter()
                    .find_map(OrdinaryEquation::missing_source)
            })
        })
    }
    /// Same additional safe-call source as OrdinaryEquation::ordinary_calls,
    /// unioned over local row alternatives and added over ordered occurrences.
    /// These returned controls are excluded from host_bytes and charged once
    /// by the enclosing request alongside its direct operator callers.
    pub(crate) fn ordinary_calls(&self) -> Option<super::super::OrdinaryCallControls> {
        self.facts.as_ref()?.calls
    }
    /// Union of local row alternatives, summed across actual occurrences.
    /// These payloads already appear in the operation facts, not transfer rows.
    pub(crate) fn storage(&self) -> Option<CertifiedSpanStorage> {
        Some(self.facts.as_ref()?.storage)
    }
    pub(crate) fn raw_cpu_population(&self) -> Option<OrdinaryCpuPopulation> {
        Some(self.facts.as_ref()?.raw)
    }
    /// Branch maxima and occurrence sums of real deferred grouped predicates.
    pub(crate) fn validation_population(&self) -> Option<(usize, CertifiedSpanStorage)> {
        let facts = self.facts.as_ref()?;
        Some((facts.validation_roots, facts.validation_storage))
    }
    /// Child and discovery frontiers for the parent's combined reachable DAG.
    pub(crate) fn completion_frontiers(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.facts
            .iter()
            .flat_map(|facts| facts.frontiers)
            .filter(|&(_, n)| n != 0)
    }
    /// Already present in operation host facts; do not charge twice.
    pub(crate) fn host_bytes(&self) -> Option<u64> {
        Some(self.facts.as_ref()?.host_bytes)
    }
    /// One actual request provider/channel installation, outside operation facts.
    pub(crate) fn installation_host_bytes(&self) -> u64 {
        self.installation_host_bytes
    }
    /// Incremental transfer payload only; each source keeps its own placement.
    pub(crate) fn transfer_requirements(&self) -> Option<&eredu_core::DomainMemoryRequirements> {
        Some(&self.facts.as_ref()?.transfers)
    }
    pub(crate) fn transfer_allocations(&self) -> Option<usize> {
        Some(self.facts.as_ref()?.transfer_allocations)
    }
    /// Opaque transfer-platform events remain distinct from observed controls.
    pub(crate) fn transfer_platform_events(&self) -> Option<usize> {
        Some(self.facts.as_ref()?.platform_events)
    }
    pub(crate) fn capture_publications(&self) -> Option<usize> {
        Some(self.facts.as_ref()?.capture_publications)
    }
}
