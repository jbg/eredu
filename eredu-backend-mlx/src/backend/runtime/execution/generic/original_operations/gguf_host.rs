//! Remaining GGUF host obligations, independent of neural fit.
use super::*;
pub(crate) mod source_arenas;
pub(crate) mod typed;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GgufHostFitContribution {
    // Ordinary cold allocator/platform setup is not a per-request payload.
    PersistentRuntimeInitializationOwners,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingGgufHostFit {
    pub(crate) pending_source_slots: Option<usize>,
    pub(crate) maximum_physical_outputs_per_group: usize,
    // Charged only by the disjoint accepted host component, not again below.
    pub(crate) source_constructions:
        Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
    pub(crate) destinations: Option<eredu_runtime::working_memory::HostDestinationFacts>,
    pub(crate) contributions: [GgufHostFitContribution; 1],
}
impl PendingGgufHostFit {
    pub(super) fn new(
        pending_source_slots: Option<usize>,
        source_constructions: Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
        destinations: Option<eredu_runtime::working_memory::HostDestinationFacts>,
    ) -> Self {
        Self {
            pending_source_slots,
            maximum_physical_outputs_per_group: 3,
            source_constructions,
            destinations,
            contributions: [GgufHostFitContribution::PersistentRuntimeInitializationOwners],
        }
    }
    pub(super) fn additional_control_bytes(self) -> Option<u64> {
        // No fit certificate is constructed by a shape, a recovery slot or an
        // observed completion. The source layout is already included in Work;
        // Selected copies include C_new+B and prepaid-H alias publication.
        // Cold source/converted-cache owners remain separately unknown in
        // ResidencyPopulation::prepared_payload_control_bytes; missing selected
        // facts or runtime initialization owners also keep the whole fit unknown.
        self.source_constructions?;
        self.destinations?;
        None
    }
}
pub(super) fn runtime_control_bytes() -> Option<u64> {
    u64::try_from(
        rc_layout::<safemlx::PreparedInputRuntime>()?
            .checked_add(eredu_core::SharedBackendFailure::control_bytes::<
                safemlx::error::Exception,
            >()?)?
            .checked_add(size_of::<
                Result<Rc<safemlx::PreparedInputRuntime>, eredu_core::SharedBackendFailure>,
            >())?
            .checked_add(size_of::<Result<Rc<safemlx::PreparedInputRuntime>, Error>>())?,
    )
    .ok()
}

pub(super) fn project_runtime(
    prepared: &Result<Rc<safemlx::PreparedInputRuntime>, eredu_core::SharedBackendFailure>,
) -> Result<Rc<safemlx::PreparedInputRuntime>, Error> {
    prepared
        .as_ref()
        .map(Rc::clone)
        .map_err(|cause| Error::retained_original(cause.retained(), true))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_runtime_projection_preserves_the_actual_cold_failure_source() {
        #[derive(Debug)]
        struct ColdCause(Arc<()>);
        impl std::fmt::Display for ColdCause {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("cold host-copy initialization cause")
            }
        }
        impl std::error::Error for ColdCause {}
        let owner = Arc::new(());
        let cause = safemlx::error::Exception::from_source(ColdCause(Arc::clone(&owner)));
        let pointer = cause.what().as_ptr();
        let prepared = Err(eredu_core::SharedBackendFailure::new(
            eredu_core::BackendFailureKind::Other,
            cause,
        ));
        let first = project_runtime(&prepared).unwrap_err();
        assert!(first.model_state_preserved());
        let first = first.into_backend_failure();
        let escaped = project_runtime(&prepared)
            .unwrap_err()
            .into_backend_failure();
        let first_source = std::error::Error::source(&first)
            .unwrap()
            .downcast_ref::<safemlx::error::Exception>()
            .unwrap();
        let source_pointer = std::ptr::from_ref(first_source);
        assert_eq!(first_source.what().as_ptr(), pointer);
        assert_eq!(first.kind(), eredu_core::BackendFailureKind::Other);
        assert_eq!(Arc::strong_count(&owner), 2);
        // The first public error can retire before the cold preparation owner.
        drop(first);
        drop(prepared);
        let exception = std::error::Error::source(&escaped)
            .unwrap()
            .downcast_ref::<safemlx::error::Exception>()
            .unwrap();
        assert_eq!(std::ptr::from_ref(exception), source_pointer);
        assert_eq!(exception.what().as_ptr(), pointer);
        let source = std::error::Error::source(exception)
            .unwrap()
            .downcast_ref::<ColdCause>()
            .unwrap();
        assert!(Arc::ptr_eq(&source.0, &owner));
        assert_eq!(Arc::strong_count(&owner), 2);
        drop(escaped);
        assert_eq!(Arc::strong_count(&owner), 1);
    }
}
