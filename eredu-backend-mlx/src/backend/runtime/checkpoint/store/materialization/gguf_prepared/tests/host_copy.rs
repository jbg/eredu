use super::*;
use crate::backend::runtime::checkpoint::store::{GgufHostCopyCause, PreparedGgufHostCopyFailure};
use safemlx::OwnedHostCopyCause;

impl OriginalGgufMissFixture {
    pub(crate) fn host_transform_refusal(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        wrong_binding: bool,
        error_first: bool,
    ) -> Option<PreparedGgufHostCopyFailure> {
        let lease = self.lease();
        let address = Self::address(&lease);
        let mut pending = PendingWeightMaterialization::begin_with_original(
            lease,
            &self.stream,
            &self.stream,
            Some((self.ready(controls), observer.clone())),
        )
        .unwrap();
        let portable = pending.materialize_gguf_prepared().unwrap();
        let (descriptor, _, converted) = portable.into_parts();
        let eredu_gguf::ConvertedTensor::Dense(mut dense) = converted else {
            panic!("dense source fixture");
        };
        if !wrong_binding {
            dense.data.push(0x31);
        }
        let host = pending.bind_original_gguf_host();
        let error = host.test_transform_refusal(&descriptor, dense.data, wrong_binding);
        let weak = host.source_owner();
        assert_eq!(Self::address(pending.lease()), address);
        assert!(observer.status().is_settled());
        assert!(!observer.status().failed());
        if error_first {
            drop(error);
            assert!(weak.upgrade().is_some());
            Self::finish(pending);
            safemlx::reclaim_allocation_owners();
            assert!(weak.upgrade().is_none());
            None
        } else {
            Self::finish(pending);
            safemlx::reclaim_allocation_owners();
            assert!(weak.upgrade().is_some());
            Some(error)
        }
    }
    pub(crate) fn host_copy_failure(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        foreign: &OriginalScopeObserver,
        output: usize,
        error_first: bool,
    ) -> Option<PreparedGgufHostCopyFailure> {
        let before = self.source.diagnostics().unwrap().physical_reads;
        let lease = self.lease();
        let pointer = Self::address(&lease);
        let mut pending = PendingWeightMaterialization::begin_with_original(
            lease,
            &self.stream,
            &self.stream,
            Some((self.ready(controls), observer.clone())),
        )
        .unwrap();
        let portable = pending.materialize_gguf_prepared().unwrap();
        pending
            .retained
            .retention_mut()
            .gguf_host
            .as_mut()
            .unwrap()
            .fail_using_observer(output, foreign);
        let error = pending.convert_original_gguf(portable).unwrap_err();
        assert_eq!(error.output_ordinal(), output);
        let GgufHostCopyCause::Copy {
            cause: OwnedHostCopyCause::Domain,
            native: Some(native),
        } = error.cause()
        else {
            panic!("real wrong-owner copy refusal: {error:?}");
        };
        assert!(native.scoped_evaluation_cause().is_some());
        let cause = std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<GgufHostCopyCause>()
            .unwrap();
        assert!(std::ptr::eq(cause, error.cause()));
        assert!(std::error::Error::source(cause)
            .unwrap()
            .downcast_ref::<safemlx::error::Exception>()
            .is_some());
        assert_eq!(Self::address(pending.lease()), pointer);
        let host = pending.retained.retention().gguf_host.as_ref().unwrap();
        host.assert_failed_input_unchanged();
        host.assert_bound_source_plan();
        assert_eq!(host.transform_facts().iter().flatten().count(), output + 1);
        for facts in host.transform_facts().iter().flatten() {
            assert!(facts.input_capacity >= facts.input_elements);
            assert!(facts.output_capacity >= facts.output_elements);
            assert!(facts.requested_output_elements >= facts.output_elements);
            assert!(facts.output_elements > 0);
            // The scalar fixture is dense F32; the affine fixture's first
            // output moves U32 words and its subsequent F16 outputs use a new Vec.
            if facts.output_element_bytes == 2 || facts.input_element_bytes == 1 {
                assert!(facts.separate_destination);
            } else {
                assert!(!facts.separate_destination);
                assert_eq!(facts.input_capacity, facts.output_capacity);
            }
        }

        assert_eq!(host.facts().iter().flatten().count(), output + 1);
        assert!(host
            .facts()
            .iter()
            .flatten()
            .all(|facts| facts.copy_bytes() > 0 && facts.backing_bytes() >= facts.copy_bytes()));
        let weak = host.source_owner();
        assert_eq!(
            self.source.diagnostics().unwrap().physical_reads,
            before + 1
        );
        assert!(self
            .context
            .converted_groups
            .lock()
            .unwrap()
            .all_values(|value| value.upgrade().is_none()));
        assert!(pending.retained.retention().group.is_none());
        let status = observer.status();
        assert!(status.is_settled() && !status.failed() && !status.blocked());
        if error_first {
            drop(error);
            assert!(
                weak.upgrade().is_some(),
                "same pending owner still holds failed input and source"
            );
            Self::finish(pending);
            safemlx::reclaim_allocation_owners();
            assert!(weak.upgrade().is_none());
            None
        } else {
            Self::finish(pending);
            safemlx::reclaim_allocation_owners();
            assert!(
                weak.upgrade().is_some(),
                "escaping typed error keeps actual source custody"
            );
            Some(error)
        }
    }

    pub(crate) fn host_copy_busy(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) -> PreparedGgufHostCopyFailure {
        let mut pending = PendingWeightMaterialization::begin_with_original(
            self.lease(),
            &self.stream,
            &self.stream,
            Some((self.ready(controls), observer.clone())),
        )
        .unwrap();
        let portable = pending.materialize_gguf_prepared().unwrap();
        let (entered, ready) = std::sync::mpsc::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || loop {
            if safemlx::try_with_submission_retirement(|| {
                entered.send(()).unwrap();
                wait.recv_timeout(std::time::Duration::from_secs(10))
                    .unwrap();
            })
            .is_some()
            {
                break;
            }
            std::thread::yield_now();
        });
        ready
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        let started = std::time::Instant::now();
        let error = pending.convert_original_gguf(portable).unwrap_err();
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        assert!(matches!(
            error.cause(),
            GgufHostCopyCause::Copy {
                cause: OwnedHostCopyCause::Busy,
                native: None
            }
        ));
        pending
            .retained
            .retention()
            .gguf_host
            .as_ref()
            .unwrap()
            .assert_failed_input_unchanged();
        release.send(()).unwrap();
        holder.join().unwrap();
        assert!(!observer.status().failed());
        Self::finish(pending);
        safemlx::reclaim_allocation_owners();
        error
    }

    pub(crate) fn host_copy_escaping_alias(
        &self,
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
    ) -> Array {
        let pending = self
            .lease()
            .prepare_original_gguf_fixture(&self.stream, self.ready(controls), observer.clone())
            .unwrap();
        self.settle(&pending, observer);
        let weak = pending
            .retained
            .retention()
            .gguf_host
            .as_ref()
            .unwrap()
            .source_owner();
        let value = pending.output().clone();
        Self::finish(pending);
        safemlx::reclaim_allocation_owners();
        assert!(
            weak.upgrade().is_some(),
            "actual immutable descriptor retains source custody"
        );
        value
    }
}

#[test]
fn host_copy_error_transport_is_compact_and_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<PreparedGgufHostCopyFailure>();
    assert!(std::mem::size_of::<PreparedGgufHostCopyFailure>() < 128);
    assert!(std::mem::size_of::<CheckpointMaterializationError>() < 256);
}
