//! Synchronous original miss storage, before any native group construction.
use super::super::leases::WeightLeaseSource;
use super::*;
use eredu_checkpoint::gguf_store::PreparedGgufBoxedFailure;
use eredu_gguf::ConvertedCheckpointTensor;

/// The exact prepared failure/source retires before its accepted original custody.
/// This owner grants neither payload storage nor native execution authority.
#[derive(Debug)]
pub struct PreparedGgufMaterializationFailure {
    cause: Box<PreparedGgufBoxedFailure>,
    _custody: OriginalTextControlGuard,
    _acquisition_metadata: Option<eredu_runtime::working_memory::OriginalHostMetadataCustody>,
}
impl std::fmt::Display for PreparedGgufMaterializationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for PreparedGgufMaterializationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}
impl PreparedGgufMaterializationFailure {
    /// Actual neutral failure, including the original box and partial destinations.
    pub fn cause(&self) -> &PreparedGgufBoxedFailure {
        &self.cause
    }
    pub(super) fn control_bytes() -> Option<u64> {
        let bytes = std::mem::size_of::<Self>()
            // Actual error allocation; the lease inside it remains the same
            // existing Box. Allocator charge is a separate pending fit term.
            .checked_add(std::mem::size_of::<PreparedGgufBoxedFailure>())?
            .checked_add(std::mem::size_of::<Result<ConvertedCheckpointTensor, Self>>())?
            .checked_add(std::mem::size_of::<
                Result<
                    (ConvertedCheckpointTensor, Box<NeutralGgufLease>),
                    PreparedGgufBoxedFailure,
                >,
            >())?;
        u64::try_from(bytes).ok()
    }
}
impl PendingWeightMaterialization {
    pub(in crate::backend::runtime::checkpoint::store) fn gguf_lease(&self) -> &NeutralGgufLease {
        let WeightLeaseSource::Gguf(source) = &self.lease().source else {
            unreachable!("closed GGUF pending source")
        };
        source.lease.as_ref()
    }

    pub(in crate::backend::runtime::checkpoint::store) fn materialize_gguf_prepared(
        &mut self,
    ) -> Result<ConvertedCheckpointTensor, PreparedGgufMaterializationFailure> {
        let value = self.retained.retention_mut();
        assert!(
            value.output.is_none() && value.source.is_none() && value.group.is_none(),
            "only unpublished pre-native GGUF preparation can move its lease"
        );
        let custody = value
            .gguf_custody
            .take()
            .expect("actual original slot custody");
        let WeightLeaseSource::Gguf(source) = &mut value.lease.source else {
            unreachable!("closed GGUF pending source")
        };
        let lease = source.lease.take();
        let address = std::ptr::from_ref(lease.as_ref());
        #[cfg(test)]
        tests::inject_pre_native_unwind();
        // No caller callback or native producer runs in this source-owned call.
        // On unwind its destinations/box retire before the local custody; the
        // same pending recovery node still owns its independent original guard.
        match NeutralGgufLease::materialize_prepared_boxed(lease) {
            Ok((portable, lease)) => {
                assert_eq!(std::ptr::from_ref(lease.as_ref()), address);
                source.lease.restore(lease);
                value.gguf_custody = Some(custody);
                Ok(portable)
            }
            Err(cause) => Err(PreparedGgufMaterializationFailure {
                // Compact transport prevents the whole partial destination
                // from inflating every recursive recipe error frame. Custody
                // is already live across this allocation and retires last.
                cause: Box::new(cause),
                _custody: custody,
                _acquisition_metadata: value.acquisition_metadata.clone(),
            }),
        }
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::OriginalGgufMissFixture;

mod admitted;
pub use admitted::PreparedGgufAdmittedFailure;
