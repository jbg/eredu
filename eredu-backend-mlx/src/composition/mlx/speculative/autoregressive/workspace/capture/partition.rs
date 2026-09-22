//! One exact partition frame, quoted and consumed under its actual AR role.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::{
    agreement::AgreementCapacity,
    control::{
        OriginalCaptureTransport,
        speculative::{PreparedSpeculativeControl, SpeculativeCaptureOwner},
    },
};
use crate::composition::mlx::session::{
    OriginalModelPartitionSource,
    bounded_capture::{
        funded_model,
        partition::{self, ModelEvidenceHostSource, PartitionCaptureInvocationSource},
    },
};
use eredu_nn::workspace::{WorkspaceFloatingType, WorkspaceMetadataAllocation};
use eredu_runtime::{
    capture::{
        FundedAutoregressiveCaptureInvocation,
        partition::{
            PartitionCaptureTransportDemand as Demand, partition_capture_coordination_demands,
        },
    },
    working_memory::{OriginalSpeculativeRole, OwnedPartitionFragmentHostPlan},
};

type Transport = OriginalCaptureTransport<SpeculativeCaptureOwner>;
/// Immutable native capacities and exact architecture receipt rows. It owns no
/// producing budget; construction receives the consumed role's Host loan.
pub(in crate::composition::mlx::speculative::autoregressive::workspace) struct PartitionQuote {
    source: PartitionCaptureInvocationSource,
    scalars: Vec<Option<WorkspaceFloatingType>>,
    capacity: AgreementCapacity,
    metadata: usize,
    prediction: u64,
    has_interventions: bool,
    evidence: Option<ModelEvidenceHostSource>,
}
fn overflow() -> Error {
    Error::PrefillControl(WorkingMemoryError::Overflow)
}
impl PartitionQuote {
    pub(in crate::composition::mlx::speculative::autoregressive::workspace) fn prepare(
        source: PartitionCaptureInvocationSource,
        native: &OriginalModelPartitionSource,
        descriptor: OriginalSpeculativeCaptureInvocation<'_>,
        fragments: &[OwnedPartitionFragmentHostPlan],
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Demand>(),
            size_of::<AgreementCapacity>(),
            size_of::<(
                &OriginalModelPartitionSource,
                OriginalSpeculativeCaptureInvocation<'_>,
                &[OwnedPartitionFragmentHostPlan],
                &WorkspaceContext,
            )>(),
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        let mut value = Self {
            source,
            scalars: Vec::new(),
            has_interventions: descriptor.interventions().is_some(),
            evidence: None,
            capacity: AgreementCapacity {
                graph: 0,
                records: 0,
                backing: 0,
            },
            prediction: descriptor.origin().prediction as u64,
            metadata: PreparedSpeculativeControl::capture_constructor_control_bytes()
                .and_then(|n| {
                    n.checked_add(Transport::preparation_metadata_control_bytes()?.checked_mul(2)?)
                })
                .ok_or_else(overflow)?,
        };
        value.metadata = value
            .metadata
            .checked_add(
                value
                    .source
                    .protocol_metadata_bytes::<Transport>(fragments, value.has_interventions)?,
            )
            .ok_or_else(overflow)?;
        if !descriptor.source().plan().admission().is_empty() || value.has_interventions {
            for demand in partition_capture_coordination_demands(
                descriptor.source().plan(),
                descriptor.capture_phase(),
                descriptor.origin().prediction as u64,
            ) {
                Self::add(&mut value.capacity, &mut value.metadata, demand, native)?;
            }
        }
        for receipt in value.source.complete_receipts() {
            for demand in receipt
                .transport_demands()
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
            {
                Self::add(&mut value.capacity, &mut value.metadata, demand, native)?;
            }
            value.metadata = value
                .metadata
                .checked_add(partition::native::model_control_bytes().ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        for fragment in fragments {
            for demand in fragment
                .transport_demands()
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
            {
                Self::add(&mut value.capacity, &mut value.metadata, demand, native)?;
            }
            value.metadata = value
                .metadata
                .checked_add(partition::native::model_control_bytes().ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        Ok(value)
    }
    fn add(
        capacity: &mut AgreementCapacity,
        total: &mut usize,
        demand: Demand,
        native: &OriginalModelPartitionSource,
    ) -> Result<(), Error> {
        let metadata = match demand {
            Demand::Active => Transport::active_control_bytes().ok_or_else(overflow)?,
            Demand::Words { maximum_elements } => {
                Transport::word_destination_control_bytes(maximum_elements).ok_or_else(overflow)?
            }
            Demand::Bytes { maximum_elements } => {
                Transport::byte_destination_control_bytes(maximum_elements).ok_or_else(overflow)?
            }
            Demand::Gather { maximum_words, .. } => {
                let quote = native
                    .control()
                    .capture_gather_requirements(maximum_words)?;
                *capacity = AgreementCapacity {
                    graph: capacity
                        .graph
                        .checked_add(quote.capacity.graph)
                        .ok_or_else(overflow)?,
                    records: capacity
                        .records
                        .checked_add(quote.capacity.records)
                        .ok_or_else(overflow)?,
                    backing: capacity
                        .backing
                        .checked_add(quote.capacity.backing)
                        .ok_or_else(overflow)?,
                };
                quote.metadata
            }
        };
        *total = total.checked_add(metadata).ok_or_else(overflow)?;
        Ok(())
    }
    pub(in crate::composition::mlx::speculative::autoregressive::workspace) fn finish(
        &mut self,
        cells: &[Cell<Option<WorkspaceFloatingType>>],
        context: &WorkspaceContext,
        native: &OriginalModelPartitionSource,
        edits: Option<&PreparedModelInterventions>,
    ) -> Result<(usize, Vec<(usize, [OwnedPartitionFragmentHostPlan; 2])>), Error> {
        let mut scalars = context
            .metadata_vec(cells.len())
            .map_err(|cause| Error::Neural(cause.into()))?;
        scalars.extend(cells.iter().map(Cell::get));
        let constructor = self
            .source
            .execution_metadata_bytes::<Transport, FundedAutoregressiveCaptureInvocation>(
                &scalars,
                self.has_interventions,
            )?;
        let hooks = self
            .source
            .protocol_hook_metadata_bytes::<Transport>(&scalars)?;
        self.scalars = scalars;
        if edits.is_some() != self.has_interventions || self.evidence.is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let evidence = match edits {
            Some(edits) => self.prepare_interventions(native, edits, context)?,
            None => Vec::new(),
        };
        let bytes = self
            .metadata
            .checked_add(constructor)
            .and_then(|bytes| bytes.checked_add(hooks))
            .ok_or_else(overflow)?;
        Ok((bytes, evidence))
    }
    pub(in crate::composition::mlx::speculative::autoregressive::workspace) fn backing(
        &self,
    ) -> usize {
        self.capacity.backing
    }
    pub(in crate::composition::mlx::speculative::autoregressive::workspace) fn construct(
        self,
        native: &OriginalModelPartitionSource,
        request: &eredu_runtime::working_memory::OriginalSpeculativeRequest,
        role: &OriginalSpeculativeRole,
        value: &mut FundedAutoregressiveCaptureInvocation,
        attempt: u64,
    ) -> Result<funded_model::Partition, Error> {
        let funding = value
            .take_partition_metadata()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let hosts = value
            .take_partition_fragments()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let metadata = funding.clone();
        let (owner, transport) = native.control().capture_transport(
            request,
            role,
            funding,
            self.capacity,
            native.transport(),
            attempt,
        )?;
        let mut frame = self.source.bind(&self.scalars, hosts, transport)?;
        if let Some(evidence) = self.evidence {
            let hosts = value
                .take_partition_evidence_fragments()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            frame.install_model_evidence(evidence, hosts, &metadata)?;
        } else if value.take_partition_evidence_fragments().is_some() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        Ok(funded_model::Partition::new(frame, owner, self.prediction))
    }
}

mod interventions;
