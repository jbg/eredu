use super::*;
use eredu_core::{HostPreparationAuthority, capture::*};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn facade_shared_success_and_failure_keep_entire_ordinary_frame_custody() {
    for failed in [false, true] {
        let retired = Arc::new(AtomicUsize::new(0));
        let host = HostPreparationAuthority::retain(Retired(retired.clone()));
        let control = PreparedCapturedStep::retain(host);
        let frame = control.finish(CapturedStep {
            outcome: if failed {
                CaptureStepOutcome::Aborted
            } else {
                CaptureStepOutcome::Committed
            },
            phase: CapturePhase::Prefill,
            invocation: None,
            prediction_index: 0,
            records: vec![CaptureRecord {
                schema_version: 1,
                selection_id: "summary".into(),
                path: "model.logits".into(),
                node_id: "model".into(),
                position: eredu_core::ObservationPosition::BeforeIntervention,
                source_shape: Some(vec![1, 5, 64]),
                source_dtype: Some(eredu_core::checkpoint::TensorDtype::F32),
                selected_shape: None,
                outcome: if failed {
                    CaptureOutcome::Failed {
                        reason: CaptureFailureReason::Native,
                        message: "original diagnostic".into(),
                    }
                } else {
                    CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::Schedule,
                    }
                },
                payload: None,
                charged: CaptureUsage::default(),
            }],
            partitions: vec![],
            interventions: vec![],
            step_usage: CaptureUsage::default(),
            cumulative_usage: CaptureUsage::default(),
            capture_seconds: 0.0,
        });
        let records = frame.records().as_ptr();
        let delivery = frame;
        let event = if failed {
            ObservedGenerationEvent::from_failed_delivery(0, [0, 5], delivery, 0.0)
        } else {
            ObservedGenerationEvent::from_token_delivery(3, false, 0, [0, 5], Some(delivery), 0.0)
        };
        assert_eq!(event.captures().unwrap().records.as_ptr(), records);
        let alias = event.shared_captures().unwrap().clone();
        if failed {
            assert!(matches!(
                &event,
                ObservedGenerationEvent::CaptureFailure { .. }
            ))
        } else {
            assert!(matches!(&event, ObservedGenerationEvent::Token { .. }))
        }
        drop(event);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert!(!alias.records()[0].path.is_empty());
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
