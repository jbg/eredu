//! Funded per-role collector around the existing native span/recovery worker.
use super::*;
use crate::composition::mlx::session::bounded_capture::funded_model;
use eredu_runtime::{
    ActivationObserver,
    capture::{FundedAutoregressiveCaptureInvocation, OriginalSpeculativeCaptureInvocation},
    working_memory::{AutoregressiveCaptureHostPlan, FundedAutoregressiveCaptureBank},
};

pub(super) type Capture = funded_model::Capture<OriginalSpeculativeRole>;
pub(super) struct Frame {
    pub value: Rc<Capture>,
    owner: funded_model::Owner<OriginalSpeculativeRole>,
    identity: String,
}
impl Frame {
    pub(super) fn prepare(
        mut value: FundedAutoregressiveCaptureInvocation,
        roots: usize,
        edits: Option<crate::composition::mlx::session::intervention::PreparedModelInterventions>,
        role: &OriginalSpeculativeRole,
        funding: &HostMetadataFunding,
        descriptor: OriginalSpeculativeCaptureInvocation<'_>,
        sources: &AutoregressiveSourcePair,
        partition: Option<super::super::super::capture::PartitionQuote>,
        ordinal: usize,
    ) -> Result<Self, Error> {
        if !role.same_role(value.role()) {
            return Err(failure(Cause::Source, role));
        }
        value
            .prepare_envelope(descriptor)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let identity = descriptor.admission_identity().to_owned();
        let partition = match partition {
            Some(partition) => {
                let native = sources
                    .partition_source(role.invocation().source())
                    .ok_or_else(|| failure(Cause::Source, role))?;
                Some(partition.construct(
                    native,
                    sources.request(),
                    role,
                    &mut value,
                    u64::try_from(ordinal).map_err(|_| failure(Cause::Overflow, role))?,
                )?)
            }
            None => None,
        };
        let owner = funded_model::Owner::prepare(value);
        let value =
            Rc::new(Capture::prepare(owner.clone(), roots, role, edits)?.with_partition(partition));
        Ok(Self {
            value,
            owner,
            identity,
        })
    }
    pub(super) fn deliver(
        self,
        receiver: &mut dyn ActivationObserver<MlxTensor, Error>,
        success: bool,
    ) -> Result<(), Error> {
        let invocation = receiver
            .original_speculative_capture()
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?;
        if let Some(envelope) = self.owner.finish(invocation, self.identity, success)? {
            self.owner.receive(receiver, envelope)?;
        }
        Ok(())
    }
}

impl super::super::super::capture::Quote<'_> {
    pub(super) fn frame_controls(
        &self,
        index: usize,
        span: &eredu_runtime::working_memory::InferenceWorkspaceSpan,
        invocation: AutoregressiveInvocation,
        geometry: eredu_core::InferenceGeometry,
    ) -> Option<u64> {
        let population = self.populations.get(index)?;
        let descriptor = super::super::super::capture::descriptor(
            self.source,
            invocation,
            geometry,
            span,
            index,
        )
        .ok()?;
        let parts = [
            Capture::control_bytes(population.retained_roots)?,
            descriptor.envelope_control_bytes()?,
            rc_bytes::<Capture>()?,
            size_of::<Frame>(),
            size_of::<Option<Frame>>(),
            size_of::<Option<Rc<Capture>>>(),
            size_of::<Result<Frame, Error>>(),
            size_of::<Result<(), Error>>(),
            size_of::<(
                FundedAutoregressiveCaptureInvocation,
                usize,
                Option<crate::composition::mlx::session::intervention::PreparedModelInterventions>,
                &OriginalSpeculativeRole,
                &HostMetadataFunding,
                OriginalSpeculativeCaptureInvocation<'_>,
                &AutoregressiveSourcePair,
                Option<super::super::super::capture::PartitionQuote>,
                usize,
            )>(),
        ];
        u64::try_from(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)?,
        )
        .ok()
    }
    pub(super) fn partition_backing(&self, index: usize) -> Result<usize, Error> {
        let slot = self.partitions.get(index).ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))?;
        let value = slot
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        Ok(value.as_ref().map_or(0, |value| value.backing()))
    }
    pub(super) fn bank(&self) -> Result<AutoregressiveCaptureHostPlan<'_>, Error> {
        AutoregressiveCaptureHostPlan::prepare(&self.hosts)
            .map_err(|cause| Error::Neural(eredu_nn::Error::backend_retained_source(cause)))
    }
}
