use super::*;
use crate::{
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionScopeOwner,
    ScopedSubmissionProgress,
};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
struct Custody(Arc<AtomicUsize>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn retained_root_exception_outlives_native_roots_and_keeps_actual_source_and_custody() {
    let count = Arc::new(AtomicUsize::new(0));
    let error = {
        // Fixture-only ordinary runtime/arena setup. The exact original admission
        // test lives in the backend and obtains its claim from the real core.
        let guard = runtime_lock::enter();
        let stream = crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let runtime = PrefillRootsRuntime::prepare();
        let failure = PreparedPrefillFailure::try_new(Custody(count.clone()))
            .unwrap()
            .try_allocate()
            .unwrap();
        let arena = PreparedSubmissionGraphQuota::try_new(64 * 1024, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let mut roots = PrefillRoots::new_retained(&runtime, 1, &arena, &failure).unwrap();
        let mut scope = SubmissionScope::try_begin_retaining(
            PreparedSubmissionScopeOwner::try_new(())
                .unwrap()
                .with_graph_quota(arena.clone()),
        )
        .unwrap();
        scope.enable_scoped_observation().unwrap();
        roots.bind_scope(&scope).unwrap();
        let source = Array::try_from_slice(&[3.0f32, 9.0], &[2]).unwrap();
        let output = source.add(&source, &stream).unwrap();
        roots.append(&output).unwrap();
        // Keep every constructed descriptor live until the actual arena refuses.
        // This is an exhaustion fixture, not a guessed production demand.
        let mut filler = Vec::new();
        loop {
            match source.add(&source, &stream) {
                Ok(value) => filler.push(value),
                Err(error) => {
                    assert_eq!(
                        std::error::Error::source(&error)
                            .unwrap()
                            .downcast_ref::<crate::error::GraphMetadataFailure>(),
                        Some(&crate::error::GraphMetadataFailure::Exhausted)
                    );
                    break;
                }
            }
        }
        loop {
            match Array::try_from_scalar(1u32) {
                Ok(value) => filler.push(value),
                Err(error) => {
                    assert_eq!(
                        std::error::Error::source(&error)
                            .unwrap()
                            .downcast_ref::<crate::error::GraphMetadataFailure>(),
                        Some(&crate::error::GraphMetadataFailure::Exhausted)
                    );
                    break;
                }
            }
        }
        let error = match roots.submit_scoped(&scope).unwrap_err() {
            PrefillRootsError::Retained(cause) => cause,
            other => panic!("lost original retained exception: {other}"),
        };
        assert_eq!(error.kind(), crate::PrefillNativeFailureKind::Exception);
        assert!(!error.message_bytes().unwrap().is_empty());
        assert!(!error.source_type_bytes().unwrap().is_empty());
        for result in [roots.wait_scoped(&scope), roots.validate_scoped(&scope)] {
            assert!(matches!(
                result,
                Err(PrefillRootsError::Refused(PrefillRootsCause::Spent))
            ));
        }
        scope.seal();
        drop(filler);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !scope.status().is_settled() {
            let (progress, _) = scope.progress_scoped();
            assert_eq!(progress, ScopedSubmissionProgress::Observed);
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        drop((roots, output, source, scope, arena, failure));
        drop(guard);
        // Scope settlement is not registry destruction. This explicit ordinary
        // fixture cleanup retires the completed native record/header before the
        // escaped error becomes the final carrier owner. No scoped wait gains a
        // global cleanup fallback from this test-only boundary.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match crate::try_retire_completed_submissions().unwrap() {
                crate::SubmissionRetirement::CompleteSnapshot => break,
                crate::SubmissionRetirement::Busy => {
                    assert!(Instant::now() < deadline);
                    std::thread::yield_now();
                }
            }
        }
        crate::reclaim_allocation_owners();
        assert_eq!(count.load(Ordering::SeqCst), 0);
        error
    };
    let message = error.message_bytes().unwrap().to_vec();
    let native_type = error.source_type_bytes().unwrap().to_vec();
    let alias = error.clone();
    drop(error);
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    for _ in 0..3 {
        assert_eq!(alias.message_bytes().unwrap(), message);
        assert_eq!(alias.source_type_bytes().unwrap(), native_type);
        assert!(!alias.to_string().is_empty());
    }
    drop(alias);
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn prepared_roots_keep_filling_after_refusal_and_close_selected_stream() {
    use crate::{
        Device, DeviceType, OriginalScopeObserver, PreparedSubmissionRecordQuota, Stream,
        SubmissionRetirement,
    };
    let device = Device::new(DeviceType::Cpu, 0);
    let selected = Stream::new_with_device(&device);
    let producer = Stream::new_with_device(&device);
    let missing = Stream::new_with_device(&device);
    let runtime = PrefillRootsRuntime::prepare_for_stream(&selected, &producer).unwrap();
    let identity = Array::try_from_slice(&[2.0f32, -3.0, 7.0], &[3]).unwrap();
    for mode in 0..3 {
        let failure = PreparedPrefillFailure::try_new(())
            .unwrap()
            .try_allocate()
            .unwrap();
        let graph = PreparedSubmissionGraphQuota::try_new(1 << 20, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let records = PreparedSubmissionRecordQuota::try_new(1 << 20, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let mut roots = PrefillRoots::new_retained(&runtime, 1, &graph, &failure).unwrap();
        let mut scope = SubmissionScope::try_begin_retaining(
            PreparedSubmissionScopeOwner::try_new(())
                .unwrap()
                .with_record_quota(records.clone())
                .with_graph_quota(graph.clone()),
        )
        .unwrap();
        scope.enable_scoped_observation().unwrap();
        scope.require_original_native_controls().unwrap();
        roots.bind_scope(&scope).unwrap();
        scope.enable_original_native_controls().unwrap();
        let observer = OriginalScopeObserver::require_current().unwrap();
        let parent = crate::transforms::async_eval_with_original_operation_event_on_stream(
            [&identity],
            &observer,
            &producer,
        )
        .unwrap();
        parent.synchronize().unwrap();
        let refused = roots
            .complete_current_scope_on_stream(&missing)
            .unwrap_err();
        assert!(matches!(
            refused,
            PrefillRootsError::Refused(PrefillRootsCause::Domain)
        ));
        assert!(roots.is_empty());
        {
            let mut child = SubmissionScope::try_begin().unwrap();
            assert!(matches!(
                roots.complete_current_scope_on_stream(&selected),
                Err(PrefillRootsError::Refused(PrefillRootsCause::Domain))
            ));
            child.seal();
        }
        let pending = (mode == 2).then(|| identity.add(&identity, &selected).unwrap());
        if mode == 1 {
            roots.append(&identity).unwrap();
        } else if let Some(pending) = &pending {
            roots.append(pending).unwrap();
        }
        parent.wait_on(&selected).unwrap();
        roots.complete_current_scope_on_stream(&selected).unwrap();
        parent.synchronize().unwrap();
        assert!(matches!(
            roots.complete_current_scope_on_stream(&selected),
            Err(PrefillRootsError::Refused(PrefillRootsCause::Spent))
        ));
        if let Some(pending) = &pending {
            assert_eq!(
                pending
                    .completed_in_original_scope(&observer)
                    .unwrap()
                    .as_slice::<f32>(),
                &[4.0, -6.0, 14.0]
            );
        }
        assert!(observer.belongs_to(&scope));
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let (outcome, status) = observer.progress().unwrap();
            assert_eq!(outcome, ScopedSubmissionProgress::Observed);
            if status.is_settled() {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        drop((parent, pending, roots));
        assert_eq!(
            observer.retire_completed_records().unwrap(),
            SubmissionRetirement::CompleteSnapshot
        );
        scope.seal();
        drop((observer, scope, failure, graph, records));
        while crate::try_retire_completed_submissions().unwrap() == SubmissionRetirement::Busy {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    }
}
