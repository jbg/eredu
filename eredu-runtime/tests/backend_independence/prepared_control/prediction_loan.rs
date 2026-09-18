use super::*;
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::PredictionStateLoanError;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Account {
    refuse: Arc<AtomicBool>,
    used: Arc<AtomicUsize>,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        if self.refuse.load(Ordering::SeqCst) {
            return Err(HostMetadataFundingError::Unavailable);
        }
        self.used.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn prediction_loan_preserves_authenticated_source_but_persistent_exchange_invalidates() {
    let (mut session, counters) = session();
    let mut lane = state(7, 1, 1);
    let target = session
        .inspect_runtime_state(|state| Ok(state.inference_retention().revision().clone()))
        .unwrap();
    let source = lane.inference_retention().revision().clone();
    let before = counters.snapshot();
    let (bound, returned) = session
        .with_prediction_target_state(&mut lane, &(), None, |session| {
            assert_eq!(session.report().unwrap().state_report(), &[7]);
            session.inspect_runtime_state(|state| {
                state
                    .inference_retention()
                    .validate_revision(&source)
                    .unwrap();
                Ok(state.inference_retention().revision().clone())
            })
        })
        .unwrap();
    returned.unwrap();
    let bound = bound.unwrap();
    lane.inference_retention()
        .validate_revision(&bound)
        .unwrap();
    session
        .inspect_runtime_state(|state| {
            state
                .inference_retention()
                .validate_revision(&target)
                .unwrap();
            Ok(())
        })
        .unwrap();
    assert_eq!(counters.snapshot().forward_calls, before.forward_calls);
    assert_eq!(
        counters.snapshot().completion_attempts,
        before.completion_attempts
    );
    session
        .exchange_prediction_target_state(&mut lane, &())
        .unwrap();
    assert!(
        lane.inference_retention()
            .validate_revision(&target)
            .is_err()
    );
    session
        .inspect_runtime_state(|state| {
            assert!(
                state
                    .inference_retention()
                    .validate_revision(&bound)
                    .is_err()
            );
            Ok(())
        })
        .unwrap();
}

#[test]
fn prediction_loan_rejects_wrong_layout_and_metadata_refusal_before_callback() {
    let (mut session, _) = session();
    let target = session
        .inspect_runtime_state(|state| Ok(state.inference_retention().revision().clone()))
        .unwrap();
    let mut lane = state(0, 2, 1);
    let source = lane.inference_retention().revision().clone();
    let error = session
        .with_prediction_target_state(&mut lane, &(), None, |_| {
            panic!("wrong source must not enter loan");
            #[allow(unreachable_code)]
            Ok::<(), ()>(())
        })
        .unwrap_err();
    assert!(matches!(error, PredictionStateLoanError::Layout));
    lane.inference_retention()
        .validate_revision(&source)
        .unwrap();
    session
        .inspect_runtime_state(|state| {
            state
                .inference_retention()
                .validate_revision(&target)
                .unwrap();
            Ok(())
        })
        .unwrap();

    let refuse = Arc::new(AtomicBool::new(false));
    let used = Arc::new(AtomicUsize::new(0));
    let funding = HostMetadataFunding::new(Account {
        refuse: refuse.clone(),
        used: used.clone(),
    })
    .unwrap();
    let mut lane = state(0, 1, 1);
    let source = lane.inference_retention().revision().clone();
    refuse.store(true, Ordering::SeqCst);
    let error = session
        .with_prediction_target_state(&mut lane, &(), Some(&funding), |_| {
            panic!("unfunded source must not enter loan");
            #[allow(unreachable_code)]
            Ok::<(), ()>(())
        })
        .unwrap_err();
    assert!(matches!(
        error,
        PredictionStateLoanError::Metadata(HostMetadataFundingError::Unavailable)
    ));
    lane.inference_retention()
        .validate_revision(&source)
        .unwrap();
    let setup = used.load(Ordering::SeqCst);
    refuse.store(false, Ordering::SeqCst);
    let (result, returned) = session
        .with_prediction_target_state(&mut lane, &(), Some(&funding), |_| Ok::<_, ()>(()))
        .unwrap();
    result.unwrap();
    returned.unwrap();
    assert!(used.load(Ordering::SeqCst) > setup);
    session
        .inspect_runtime_state(|state| {
            state
                .inference_retention()
                .validate_revision(&target)
                .unwrap();
            Ok(())
        })
        .unwrap();
}

#[test]
fn prediction_loan_real_publication_advances_lane_and_keeps_target_unchanged() {
    let (mut session, counters) = session_with_checkpoint_failure(false);
    let mut lane = state(0, 1, 1);
    let source = lane.inference_retention().revision().clone();
    let target = session
        .inspect_runtime_state(|state| Ok(state.inference_retention().revision().clone()))
        .unwrap();
    let before = counters.snapshot();
    let (result, returned) = session
        .with_prediction_target_state(&mut lane, &(), None, |session| {
            let result = session.prefill(&FakeTensor(vec![1, 2]), None, &())?;
            let published = session.inspect_runtime_state(|state| {
                assert!(
                    state
                        .inference_retention()
                        .validate_revision(&source)
                        .is_err()
                );
                Ok(state.inference_retention().revision().clone())
            })?;
            Ok::<_, eredu_runtime::ReplicatedTextSessionError<Error, &'static str, &'static str>>((
                result, published,
            ))
        })
        .unwrap();
    returned.unwrap();
    let (output, published) = result.unwrap();
    // Ordinary prefill requests the final readout position from the same
    // marker traversal; the prediction-capture fixture returns every marker.
    assert_eq!(output, FakeTensor(vec![5]));
    lane.inference_retention()
        .validate_revision(&published)
        .unwrap();
    assert!(
        lane.inference_retention()
            .validate_revision(&source)
            .is_err()
    );
    assert_eq!(lane.as_ref()[0].0, 1);
    session
        .inspect_runtime_state(|state| {
            state
                .inference_retention()
                .validate_revision(&target)
                .unwrap();
            Ok(())
        })
        .unwrap();
    assert_eq!(counters.snapshot().forward_calls, before.forward_calls + 1);
    assert_eq!(
        counters.snapshot().completion_attempts,
        before.completion_attempts + 1
    );
    assert_eq!(counters.snapshot().publications, before.publications + 1);
}
