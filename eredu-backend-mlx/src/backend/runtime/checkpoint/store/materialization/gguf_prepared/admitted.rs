//! The actual G1 producer consumes the separately admitted current-miss bank.
use super::*;
use crate::backend::runtime::execution::generic::gguf_host_typed::supplied::{
    self, AdmittedFamily, AdmittedStorage, StorageCause,
};
use eredu_checkpoint::gguf_store::{GgufRawStorageRequest, GgufStorageProvider, StoredGgufFailure};
use eredu_gguf::{StorageProvider, StoredCheckpointTensor, SuppliedStorageError};
use eredu_runtime::working_memory::{HostDestinationCause, OriginalHostDestinationBank};
struct Provider<'a> {
    bank: &'a mut OriginalHostDestinationBank,
    source: usize,
}
impl StorageProvider for Provider<'_> {
    type Family = AdmittedFamily;
    fn prepare<T: Copy + std::fmt::Debug + 'static>(
        &mut self,
        count: usize,
        initializer: T,
    ) -> Result<AdmittedStorage<T>, (StorageCause, Option<AdmittedStorage<T>>)> {
        supplied::prepare(self.bank, count, initializer)
    }
}
impl GgufStorageProvider for Provider<'_> {
    fn prepare_raw(
        &mut self,
        request: GgufRawStorageRequest<'_>,
    ) -> Result<AdmittedStorage<u8>, (StorageCause, Option<AdmittedStorage<u8>>)> {
        if std::ptr::from_ref(request.lease()) as usize != self.source {
            return Err((StorageCause::Source, None));
        }
        self.prepare(request.layout().size(), 0)
    }
}
type ActualFailure = StoredGgufFailure<AdmittedFamily>;
#[derive(Debug)]
struct FailureBody {
    cause: Option<ActualFailure>,
    // After all source/destination members; the enclosing allocation is freed
    // by the closed extraction below before any of these fields is destroyed.
    _custody: OriginalTextControlGuard,
}

/// Actual same-source failure; its final shell is prepared before role execution.
#[derive(Debug)]
pub struct PreparedGgufAdmittedFailure {
    body: Option<Box<FailureBody>>,
    acquisition_metadata: Option<eredu_runtime::working_memory::OriginalHostMetadataCustody>,
}
impl PreparedGgufAdmittedFailure {
    pub(in crate::backend::runtime::checkpoint::store::materialization) fn prepare(
        custody: OriginalTextControlGuard,
    ) -> Self {
        Self {
            acquisition_metadata: None,
            body: Some(Box::new(FailureBody {
                cause: None,
                _custody: custody,
            })),
        }
    }
    fn publish(
        mut self,
        cause: ActualFailure,
        metadata: Option<eredu_runtime::working_memory::OriginalHostMetadataCustody>,
    ) -> Self {
        self.acquisition_metadata = metadata;
        self.body.as_mut().expect("prepared failure body").cause = Some(cause);
        self
    }
    /// Same retained source, including provider refusal before any I/O.
    pub fn lease(&self) -> &NeutralGgufLease {
        self.actual().lease()
    }
    /// Actual runtime admission/fill cause when the provider refused G1.
    pub fn host_destination_cause(&self) -> Option<&HostDestinationCause> {
        match self.actual().storage_error() {
            Some(SuppliedStorageError::Provider(StorageCause::Host(cause))) => Some(cause),
            _ => None,
        }
    }

    fn actual(&self) -> &ActualFailure {
        self.body
            .as_ref()
            .expect("live failure body")
            .cause
            .as_ref()
            .expect("published failure")
    }
    pub(in crate::backend::runtime::checkpoint::store::materialization) fn control_bytes(
    ) -> Option<u64> {
        let bytes = std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<FailureBody>())?
            .checked_add(std::mem::size_of::<ActualFailure>())?
            .checked_add(std::mem::size_of::<AdmittedStorage<u8>>())?
            .checked_add(std::mem::size_of::<Provider<'static>>())?
            .checked_add(std::mem::size_of::<eredu_gguf::SharedTensorMetadataSource>())?
            .checked_add(std::mem::size_of::<
                Result<Option<eredu_gguf::SharedTensorMetadataSource>, eredu_gguf::Error>,
            >())?
            .checked_add(std::mem::size_of::<
                Result<StoredCheckpointTensor<AdmittedFamily>, Self>,
            >())?
            .checked_add(std::mem::size_of::<
                Result<
                    (
                        StoredCheckpointTensor<AdmittedFamily>,
                        Box<NeutralGgufLease>,
                    ),
                    ActualFailure,
                >,
            >())?;
        u64::try_from(bytes).ok()
    }
}
impl Drop for PreparedGgufAdmittedFailure {
    fn drop(&mut self) {
        if let Some(shell) = self.body.take() {
            // Moving from Box deallocates its shell now; then the actual partial
            // source/destination retires, finally the original accepted hold.
            let body = *shell;
            drop(body);
        }
    }
}
impl std::fmt::Display for PreparedGgufAdmittedFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.actual().fmt(f)
    }
}
impl std::error::Error for PreparedGgufAdmittedFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.actual())
    }
}

impl PendingWeightMaterialization {
    pub(in crate::backend::runtime::checkpoint::store) fn has_admitted_gguf_storage(&self) -> bool {
        self.retained
            .retention()
            .gguf_host
            .as_ref()
            .is_some_and(|host| host.host_destinations.is_some())
    }
    pub(in crate::backend::runtime::checkpoint::store) fn materialize_gguf_admitted(
        &mut self,
    ) -> Result<StoredCheckpointTensor<AdmittedFamily>, PreparedGgufAdmittedFailure> {
        let value = self.retained.retention_mut();
        assert!(
            value.output.is_none() && value.source.is_none() && value.group.is_none(),
            "only unpublished pre-native GGUF preparation can move its lease"
        );
        let host = value
            .gguf_host
            .as_mut()
            .expect("admitted original host slot");
        assert!(
            !host.raw_attempted && host.allocation_source.is_none(),
            "raw destination is once-only"
        );
        let bank = host.host_destinations.as_mut().expect("admitted bank");
        let failure = host.raw_failure.take().expect("preallocated raw failure");
        let custody = value
            .gguf_custody
            .take()
            .expect("actual original slot custody");
        let WeightLeaseSource::Gguf(source) = &mut value.lease.source else {
            unreachable!("closed GGUF pending source")
        };
        let lease = source.lease.take();
        let address = std::ptr::from_ref(lease.as_ref()) as usize;
        host.raw_attempted = true;
        host.allocation_source = Some(address);
        match NeutralGgufLease::materialize_prepared_boxed_with_destinations(
            lease,
            Provider {
                bank,
                source: address,
            },
        ) {
            Ok((portable, lease)) => {
                assert_eq!(std::ptr::from_ref(lease.as_ref()) as usize, address);
                source.lease.restore(lease);
                value.gguf_custody = Some(custody);
                drop(failure);
                Ok(portable)
            }
            Err(cause) => {
                let failure = failure.publish(cause, value.acquisition_metadata.clone());
                drop(custody);
                Err(failure)
            }
        }
    }
}
