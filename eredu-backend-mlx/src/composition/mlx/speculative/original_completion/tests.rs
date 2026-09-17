use super::*;
use crate::backend::nn::shared::{OriginalObservationFailure, OriginalObservationSite};
use eredu_core::speculative::SpeculativeControlError;
use eredu_runtime::capture::{CaptureExecutionError, SpeculativeCaptureErrorTransport};
use safemlx::ScopedSubmissionProgress;
use std::error::Error as _;

fn pair(error: Error) -> (Error, SpeculativeControlError) {
    let transport = super::super::capture_error_transport::RetainedCaptureTransport::new(
        |_: &CaptureExecutionError<Error>| -> Error { panic!("retained source formatted") },
    );
    match transport.take_retained(CaptureExecutionError::Backend(error)) {
        Ok(pair) => pair,
        Err(_) => panic!("original failure was not retained"),
    }
}

impl PreparedOriginalSpeculativeCompletion {
    pub(in crate::composition::mlx) fn exercise_deadline_overflow(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        root: &Array,
        stream: &Stream,
    ) -> Error {
        assert!(std::time::Instant::now()
            .checked_add(std::time::Duration::MAX)
            .is_none());
        let completion = Self::try_new(1, controls.clone())
            .unwrap()
            .submit([root], observer.clone(), stream)
            .unwrap();
        let policy = BoundedCompletionWait::new(
            std::time::Duration::MAX,
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let error = completion.wait_bounded(policy).unwrap_err();
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<OnceFailure>()
            .unwrap();
        assert!(matches!(&source.cause, OnceCause::DeadlineOverflow));
        assert!(!OriginalSpeculativeCompletion::supports_cancellation(
            CompletionCancellationMode::NativeCancel
        ));
        error
    }

    pub(in crate::composition::mlx) fn exercise_wrong_owner(
        controls: &OriginalTextControlGuard,
        foreign: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Error {
        let error = match Self::try_new(1, controls.clone()).unwrap().submit(
            std::iter::empty(),
            foreign.clone(),
            stream,
        ) {
            Ok(_) => panic!("foreign original role accepted"),
            Err(error) => error,
        };
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<OnceFailure>()
            .unwrap();
        let OnceCause::Submission(OriginalArraySubmissionCause::Native(cause)) = &source.cause
        else {
            panic!("identity must precede iterator extent failure");
        };
        assert_eq!(
            cause.scoped_evaluation_cause(),
            Some(safemlx::error::ScopedEvaluationCause::Unobservable)
        );
        error
    }

    pub(in crate::composition::mlx) fn exercise_terminal_source(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        offending: Exception,
    ) -> [eredu_core::BackendFailure; 2] {
        let sources = Sources::new();
        let first = sources.failure(
            OriginalObservationFailure {
                site: OriginalObservationSite::Event,
                cause: offending,
            },
            controls.clone(),
        );
        let later = sources.failure(
            OriginalObservationFailure {
                site: OriginalObservationSite::Observer,
                cause: observer
                    .observation_error(ScopedSubmissionProgress::Busy)
                    .unwrap(),
            },
            controls.clone(),
        );
        assert!(std::ptr::addr_eq(
            first.source().unwrap(),
            later.source().unwrap()
        ));
        assert!(sources.terminal().is_some());
        let [first, later] = [first, later].map(Error::into_backend_failure);
        assert_eq!(first.kind(), BackendFailureKind::InvalidSession);
        assert!(first
            .to_string()
            .contains("retained first speculative contract source"));
        [first, later]
    }

    pub(in crate::composition::mlx) fn exercise_roots(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        inputs: &[Array; 2],
        stream: &Stream,
    ) {
        let empty = Self::try_new(0, controls.clone())
            .unwrap()
            .submit(std::iter::empty(), observer.clone(), stream)
            .unwrap();
        empty.finish().unwrap();
        let sum = inputs[0].add(&inputs[1], stream).unwrap();
        let completion = Self::try_new(3, controls.clone())
            .unwrap()
            .submit([&sum, &inputs[0], &sum], observer.clone(), stream)
            .unwrap();
        drop(sum);
        completion.wait().unwrap();
        completion.retained.with_retained_arrays_for_test(|roots| {
            assert_eq!(roots.len(), 3);
            for (root, expected) in roots.iter().zip([4.0_f32, 1.0, 4.0]) {
                let ready = root.completed_in_original_scope(observer).unwrap();
                assert_eq!(ready.try_as_slice::<f32>().unwrap(), &[expected]);
            }
        });
        completion.finish().unwrap();
    }

    pub(in crate::composition::mlx) fn exercise_busy_pair(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        root: &Array,
        stream: &Stream,
        with_busy: impl FnOnce(&mut dyn FnMut()),
    ) -> [(Error, SpeculativeControlError); 2] {
        let completion = Self::try_new(1, controls.clone())
            .unwrap()
            .submit([root], observer.clone(), stream)
            .unwrap();
        let mut retained = None;
        with_busy(&mut || {
            let first = completion.is_complete().unwrap_err();
            let second = completion.is_complete().unwrap_err();
            assert!(std::ptr::addr_eq(
                first.source().unwrap(),
                second.source().unwrap()
            ));
            retained = Some([pair(first), pair(second)]);
        });
        completion.finish().unwrap();
        retained.unwrap()
    }

    pub(in crate::composition::mlx) fn exercise_publication(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        root: &Array,
        stream: &Stream,
        count: usize,
    ) -> [eredu_core::BackendFailure; 4] {
        let sources = Sources::new();
        let make = |site, cause| {
            sources.failure(OriginalObservationFailure { site, cause }, controls.clone())
        };
        let before = make(
            OriginalObservationSite::Observer,
            observer
                .observation_error(ScopedSubmissionProgress::Busy)
                .unwrap(),
        );
        let cause = match safemlx::transforms::async_eval_with_original_operation_event_on_stream(
            std::iter::repeat(root).take(count),
            observer,
            stream,
        ) {
            Err(cause) => cause,
            Ok(_) => panic!("root vector exceeded actual graph ceiling but fit"),
        };
        assert_eq!(
            cause.scoped_evaluation_cause(),
            Some(safemlx::error::ScopedEvaluationCause::Failed)
        );
        let native = cause
            .source()
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<safemlx::PrefillNativeError>()
            .unwrap();
        assert_eq!(
            native.message_bytes(),
            Some(b"graph metadata capacity exhausted".as_slice())
        );
        let published = make(OriginalObservationSite::Event, cause);
        let after = make(
            OriginalObservationSite::Observer,
            observer
                .observation_error(ScopedSubmissionProgress::Busy)
                .unwrap(),
        );
        let validation = make(
            OriginalObservationSite::Validation,
            observer
                .observation_error(ScopedSubmissionProgress::Busy)
                .unwrap(),
        );
        assert!(!std::ptr::addr_eq(
            before.source().unwrap(),
            after.source().unwrap()
        ));
        assert!(!std::ptr::addr_eq(
            after.source().unwrap(),
            validation.source().unwrap()
        ));
        drop(sources);
        [before, published, after, validation].map(Error::into_backend_failure)
    }
}
