use super::super::{prediction, PredictionRole, SamplingRoots};
use super::*;
use crate::backend::{MlxBackend, MlxCompletion};
use eredu_core::{BackendFailure, BackendProvider};
use safemlx::{Array, OriginalScopeObserver, ScopedSubmissionProgress, Stream};
use std::error::Error as _;

fn published(error: &safemlx::error::Exception) -> bool {
    error.source().and_then(|source| source.source()).is_some()
}
fn leaf(error: &BackendFailure) -> &SamplingEventFailure {
    error
        .source()
        .unwrap()
        .downcast_ref::<SamplingEventFailure>()
        .unwrap()
}
impl MlxCompletion {
    // Called only by the genuine composition fixture. The callback holds a
    // real foreign runtime lock; no native publication/status is fabricated.
    pub(crate) fn exercise_sampling_snapshot_publication(
        role: PredictionRole,
        root: &Array,
        stream: &Stream,
        root_count: usize,
        with_busy: impl FnOnce(&mut dyn FnMut()),
    ) -> [BackendFailure; 5] {
        let controls = role.control_guard().clone();
        let mut scope = prediction::begin(
            Some(role),
            SamplingRoots {
                event: None,
                roots: [None, None],
                witness: None,
                cleanup: None,
                original: None,
            },
        )
        .unwrap();
        let observer = OriginalScopeObserver::require_current().unwrap();
        let poll = PollOwner::new();
        let mut before = None;
        with_busy(&mut || {
            let (outcome, _) = observer.progress().unwrap();
            assert_eq!(outcome, ScopedSubmissionProgress::Busy);
            let cause = observer.observation_error(outcome).unwrap();
            assert!(!published(&cause));
            let first = MlxBackend::into_backend_failure(poll.failure(
                Origin::Observer,
                cause.into(),
                controls.clone(),
            ));
            let repeated = MlxBackend::into_backend_failure(poll.failure(
                Origin::Observer,
                observer.observation_error(outcome).unwrap().into(),
                controls.clone(),
            ));
            assert!(std::ptr::eq(leaf(&first), leaf(&repeated)));
            before = Some([first, repeated]);
        });
        // count derives from the genuine configured Graph byte ceiling plus
        // one. Every C++ array object occupies at least one byte, so this root
        // vector cannot fit; the repeated borrowed iterator itself allocates none.
        let cause = match safemlx::transforms::async_eval_with_original_operation_event_on_stream(
            std::iter::repeat(root).take(root_count),
            &observer,
            stream,
        ) {
            Err(error) => error,
            Ok(_) => panic!("finite graph root vector unexpectedly fit"),
        };
        assert_eq!(cause.scoped_evaluation_cause(), Some(Code::Failed));
        assert!(
            published(&cause),
            "must reach append/capture, not fail event allocation first"
        );
        let published_cause = cause
            .source()
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<safemlx::PrefillNativeError>()
            .unwrap();
        assert_eq!(
            published_cause.kind(),
            safemlx::PrefillNativeFailureKind::Exception
        );
        assert_eq!(
            published_cause.message_bytes(),
            Some(b"graph metadata capacity exhausted".as_slice())
        );
        let native = MlxBackend::into_backend_failure(poll.failure(
            Origin::Event,
            cause.into(),
            controls.clone(),
        ));
        // Publication alone need not create a failed Eval record. Reuse the
        // actual snapshot constructor; never invent a status.failed transition.
        let later = observer
            .observation_error(ScopedSubmissionProgress::Busy)
            .unwrap();
        assert!(published(&later));
        let after = MlxBackend::into_backend_failure(poll.failure(
            Origin::Observer,
            later.into(),
            controls.clone(),
        ));
        let repeated = MlxBackend::into_backend_failure(
            poll.failure(
                Origin::Observer,
                observer
                    .observation_error(ScopedSubmissionProgress::Busy)
                    .unwrap()
                    .into(),
                controls.clone(),
            ),
        );
        assert!(std::ptr::eq(leaf(&after), leaf(&repeated)));
        let [first, old_repeat] = before.unwrap();
        let SamplingEventCause::Native(old) = &leaf(&first).cause else {
            panic!("native snapshot")
        };
        assert!(!published(old));
        assert!(!std::ptr::eq(leaf(&first), leaf(&after)));
        assert!(poll.terminal().is_none());
        scope.seal();
        drop(scope);
        drop(observer);
        drop(poll);
        drop(controls);
        [first, old_repeat, native, after, repeated]
    }

    pub(crate) fn exercise_sampling_terminal_contract(&self) -> BackendFailure {
        let super::super::super::CompletionKind::Original(value) = &self.inner else {
            panic!("original completion")
        };
        // Safe unclassified concrete input exercises the same closed terminal
        // branch as an unexpected future ABI status. No invalid ABI is forged.
        let first = MlxBackend::into_backend_failure(value.poll.failure(
            Origin::Event,
            safemlx::error::Exception::custom("first offending source").into(),
            value.controls.clone(),
        ));
        let second = MlxBackend::into_backend_failure(value.poll.failure(
            Origin::Event,
            safemlx::error::Exception::custom("later source must not replace first").into(),
            value.controls.clone(),
        ));
        assert_eq!(first.kind(), BackendFailureKind::InvalidSession);
        let a = first
            .source()
            .unwrap()
            .downcast_ref::<ContractFailure>()
            .unwrap();
        let b = second
            .source()
            .unwrap()
            .downcast_ref::<ContractFailure>()
            .unwrap();
        assert!(std::ptr::eq(a, b));
        assert!(matches!(a.reason, ContractReason::Unclassified));
        assert!(a.offending.to_string().contains("first offending source"));
        let terminal =
            MlxBackend::into_backend_failure(value.source().ensure_usable().unwrap_err());
        assert!(std::ptr::eq(
            a,
            terminal
                .source()
                .unwrap()
                .downcast_ref::<ContractFailure>()
                .unwrap()
        ));
        first
    }
    pub(crate) fn exercise_sampling_conflicting_snapshot(
        &self,
        foreign: &OriginalScopeObserver,
    ) -> [BackendFailure; 2] {
        let super::super::super::CompletionKind::Original(value) = &self.inner else {
            panic!("original completion")
        };
        assert!(!foreign.same_scope(value.retained.observer()));
        let old = value
            .retained
            .observer()
            .observation_error(ScopedSubmissionProgress::Busy)
            .unwrap();
        let old = MlxBackend::into_backend_failure(value.poll.failure(
            Origin::Observer,
            old.into(),
            value.controls.clone(),
        ));
        let offending = foreign
            .observation_error(ScopedSubmissionProgress::Busy)
            .unwrap();
        let conflict = MlxBackend::into_backend_failure(value.poll.failure(
            Origin::Observer,
            offending.into(),
            value.controls.clone(),
        ));
        assert_eq!(conflict.kind(), BackendFailureKind::InvalidSession);
        let source = conflict
            .source()
            .unwrap()
            .downcast_ref::<ContractFailure>()
            .unwrap();
        assert!(matches!(source.reason, ContractReason::Changed(_)));
        let SamplingEventCause::Native(actual) = &source.offending else {
            panic!("native offending cause")
        };
        assert_eq!(
            actual,
            &foreign
                .observation_error(ScopedSubmissionProgress::Busy)
                .unwrap()
        );
        let next = MlxBackend::into_backend_failure(value.poll.terminal().unwrap());
        assert!(std::ptr::eq(
            source,
            next.source()
                .unwrap()
                .downcast_ref::<ContractFailure>()
                .unwrap()
        ));
        [old, conflict]
    }
}
