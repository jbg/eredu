//! Invoked by the genuine prefill Source fixture, not a synthetic scope issuer.
use super::*;

impl PreparedNeuralSubmission {
    pub(crate) fn exercise_original_identity_join(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        input: &MlxTensor,
        streams: &[Stream; 4],
    ) {
        // This fixture explicitly constructs finite test storage under borrowed
        // real custody. It does not certify these extra slots against Q.
        let shape = NeuralSubmissionShape::new(1, 2).unwrap();
        let [initial, left, right, joined, readout] =
            std::array::from_fn(|_| Self::try_new(shape, controls.clone()).unwrap());
        let initial = initial
            .submit(&[input], observer.clone(), &streams[0])
            .unwrap();

        initial.order_after(&streams[1]).unwrap();
        let left = left
            .submit(&[input], observer.clone(), &streams[1])
            .unwrap();
        initial.order_after(&streams[2]).unwrap();
        let right = right
            .submit(&[input], observer.clone(), &streams[2])
            .unwrap();
        assert_eq!(initial.next_consumer.get(), 2);

        // These are actual native waits into one consumer stream. The second
        // predecessor must be allowed before any consumer work is submitted.
        left.order_after(&streams[3]).unwrap();
        let pending = left.wait().unwrap_err();
        assert_eq!(
            pending.scoped_evaluation_cause(),
            Some(safemlx::error::ScopedEvaluationCause::NeedsFundedProgress)
        );
        assert!(!left.retained.retention().shared.failed.get());
        right.order_after(&streams[3]).unwrap();
        assert_eq!(left.next_consumer.get(), 1);
        assert_eq!(right.next_consumer.get(), 1);
        drop(pending);

        // Input was already evaluated before Source entry. No intervening
        // neural operation can accidentally flush these identity boundaries.
        let joined = joined
            .submit(&[input], observer.clone(), &streams[3])
            .unwrap();
        joined.order_after(&streams[0]).unwrap();
        let readout = readout
            .submit(&[input], observer.clone(), &streams[0])
            .unwrap();
        readout.finish().unwrap();
        for completion in [left, right, joined, initial] {
            assert!(!completion.retained.retention().shared.failed.get());
            completion.finish().unwrap();
        }
        let (outcome, status) = observer.progress().unwrap();
        assert_eq!(outcome, ScopedSubmissionProgress::Observed);
        assert!(!status.failed() && !status.blocked());
        assert!(status.is_settled());
    }
}
