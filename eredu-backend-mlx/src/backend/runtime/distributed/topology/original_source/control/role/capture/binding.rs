//! Closed identity of a real loaded transport, without a native handle clone.
use super::*;

pub(crate) struct CaptureTransportBinding {
    source: RetainedCommunicationSource,
    authority: PartitionCommunicationAuthority,
    _funding: HostMetadataFunding,
}
impl CaptureTransportBinding {
    pub(crate) fn prepare(
        base: &MlxDistributedSession,
        source: &OriginalCommunicationSource<'_>,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<(
                &MlxDistributedSession,
                &OriginalCommunicationSource<'_>,
                &HostMetadataFunding,
            )>(),
            failure_control_bytes().ok_or_else(overflow)?,
        ];
        funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(overflow)?,
        )?;
        source.validate()?;
        if !base.matches_capture_source(source.source(), source.authority)
            || source.authority.completion_policy().is_none()
            || source.world().retained_transport_stream().is_none()
        {
            return Err(failure(Cause::Identity, source.source(), funding));
        }
        Ok(Self {
            source: source.source().clone(),
            authority: source.authority.clone(),
            _funding: funding.clone(),
        })
    }
    pub(super) fn matches(&self, source: &OriginalCommunicationSource<'_>) -> bool {
        self.source.same_source(source.source()) && self.authority.same_authority(source.authority)
    }
}
/// Each variant is a borrowed exact owner. The closed variant can be retained
/// before a model loan without retaining another native Stream or Group handle.
pub(super) enum ExpectedCaptureSource<'a> {
    Loaded(&'a MlxDistributedSession),
    Closed(&'a CaptureTransportBinding),
}
impl ExpectedCaptureSource<'_> {
    pub(super) fn matches(&self, source: &OriginalCommunicationSource<'_>) -> bool {
        match self {
            Self::Loaded(base) => base.matches_capture_source(source.source(), source.authority),
            Self::Closed(binding) => binding.matches(source),
        }
    }
}
