//! Actual retained communication facts for one independently admitted frame.
use super::super::super::role::OriginalCaptureTransport;
use super::super::super::role::{CaptureTransportBinding, GatherRequirements};
use super::*;

impl PreparedSpeculativeControl {
    /// Borrow the retained descriptive communication source of this exact owner.
    pub(crate) fn capture_retained_source(&self) -> Result<&RetainedCommunicationSource, Error> {
        if self.body().failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        Ok(&self.body().request.retained)
    }
    pub(crate) fn capture_rank(&self) -> usize {
        self.body().request.retained.manifest().rank()
    }

    pub(crate) fn capture_gather_requirements(
        &self,
        words: usize,
    ) -> Result<GatherRequirements, Error> {
        let body = self.body();
        if body.failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let actual = body
            .request
            .source
            .model()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let source = actual.communication_source()?;
        let inputs = actual
            .agreement_inputs()
            .ok_or(Error::PrefillScopeUnavailable)?;
        OriginalCaptureTransport::<SpeculativeCaptureOwner>::gather_requirements(
            &source,
            inputs.runtime(),
            words,
            OriginalParallelSource::communication_source_funded_control_bytes()
                .ok_or_else(overflow)?,
        )
    }
    pub(crate) fn capture_constructor_control_bytes() -> Option<usize> {
        SpeculativeCaptureOwner::control_bytes()?
            .checked_add(OriginalCaptureTransport::<SpeculativeCaptureOwner>::source_construction_control_bytes()?)
            .and_then(|n|n.checked_add(OriginalParallelSource::communication_source_funded_control_bytes()?))
            .and_then(|n|n.checked_add(OriginalCommunicationSource::validation_control_bytes()?))
            .and_then(|n|n.checked_add(Self::capture_factory_control_bytes()?))
    }
    fn capture_factory_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<(
                &Self,
                &OriginalSpeculativeRequest,
                &OriginalSpeculativeRole,
                HostMetadataFunding,
                AgreementCapacity,
                &CaptureTransportBinding,
                u64,
            )>(),
            size_of::<(
                SpeculativeCaptureOwner,
                OriginalCaptureTransport<SpeculativeCaptureOwner>,
            )>(),
            size_of::<
                Result<
                    (
                        SpeculativeCaptureOwner,
                        OriginalCaptureTransport<SpeculativeCaptureOwner>,
                    ),
                    Error,
                >,
            >(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn capture_transport(
        &self,
        request: &OriginalSpeculativeRequest,
        role: &OriginalSpeculativeRole,
        funding: HostMetadataFunding,
        capacity: AgreementCapacity,
        binding: &CaptureTransportBinding,
        attempt: u64,
    ) -> Result<
        (
            SpeculativeCaptureOwner,
            OriginalCaptureTransport<SpeculativeCaptureOwner>,
        ),
        Error,
    > {
        funding.reserve_metadata(Self::capture_factory_control_bytes().ok_or_else(overflow)?)?;
        let owner = SpeculativeCaptureOwner::new(self, request, role, funding, capacity)?;
        let transport = OriginalCaptureTransport::from_binding(owner.retained(), binding, attempt)?;
        Ok((owner, transport))
    }
}

impl PreparedSpeculativeControl {
    pub(crate) fn capture_member_vote_requirements(
        &self,
        members: &[usize],
    ) -> Result<GatherRequirements, Error> {
        let body = self.body();
        if body.failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let actual = body
            .request
            .source
            .model()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let source = actual.communication_source()?;
        let inputs = actual
            .agreement_inputs()
            .ok_or(Error::PrefillScopeUnavailable)?;
        OriginalCaptureTransport::<SpeculativeCaptureOwner>::member_vote_requirements(
            &source,
            inputs,
            members,
            OriginalParallelSource::communication_source_funded_control_bytes()
                .ok_or_else(overflow)?,
        )
    }
}
