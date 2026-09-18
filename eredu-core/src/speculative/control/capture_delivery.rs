//! Retained speculative capture custody and diagnostic wire conformance.
use super::{SpeculativeActivationCapture, SpeculativePredictionCapture};
use crate::capture::SharedCapturedStep;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HostPreparationAuthority, ObservationPosition, SpeculativeBuffer,
        capture::{
            CAPTURE_SCHEMA_VERSION, CaptureOutcome, CapturePhase, CaptureRecord,
            CaptureStepOutcome, CaptureUsage, CapturedStep,
        },
        speculative::SpeculativeCaptureRole,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Retires(Arc<AtomicUsize>);
    impl Drop for Retires {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn frame() -> CapturedStep {
        CapturedStep {
            outcome: CaptureStepOutcome::Untracked,
            phase: CapturePhase::Decode,
            invocation: None,
            prediction_index: 3,
            records: vec![CaptureRecord {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selection_id: "raw-row".into(),
                path: crate::MODEL_LOGITS_OBSERVATION_PATH.into(),
                node_id: "output".into(),
                position: ObservationPosition::BeforeIntervention,
                source_shape: Some(vec![1, 1, 4]),
                source_dtype: None,
                selected_shape: None,
                outcome: CaptureOutcome::Missing,
                payload: None,
                charged: CaptureUsage::default(),
            }],
            partitions: Vec::new(),
            interventions: Vec::new(),
            step_usage: CaptureUsage::default(),
            cumulative_usage: CaptureUsage {
                captures: 7,
                ..Default::default()
            },
            capture_seconds: 0.0,
        }
    }

    #[test]
    fn prediction_capture_preserves_wire_tentative_state_and_independent_custody() {
        let frame = frame();
        let legacy = SpeculativePredictionCapture {
            role: SpeculativeCaptureRole::Draft,
            position: 3,
            capture: crate::capture::SharedCapturedStep::retain(
                frame.clone(),
                (),
            ),
        };
        let frame_retired = Arc::new(AtomicUsize::new(0));
        let buffer_retired = Arc::new(AtomicUsize::new(0));
        let shared = SharedCapturedStep::retain(frame, Retires(frame_retired.clone()));
        let source = shared.clone();
        let record = SpeculativePredictionCapture {
            role: legacy.role,
            position: legacy.position,
            capture: shared,
        };
        let wire = serde_json::to_value(&legacy).unwrap();
        assert_eq!(serde_json::to_value(&record).unwrap(), wire);
        let decoded: SpeculativePredictionCapture = serde_json::from_value(wire).unwrap();
        assert_eq!(record, decoded);

        let alias = record.clone();
        assert!(source.same_storage(&alias.capture));
        assert!(std::ptr::eq(
            source.records()[0].selection_id.as_ptr(),
            alias.capture.as_step().records[0].selection_id.as_ptr(),
        ));
        let mut rows = SpeculativeBuffer::try_new_retained(
            1,
            HostPreparationAuthority::retain(Retires(buffer_retired.clone())),
        )
        .unwrap();
        rows.try_push(record).unwrap();
        drop(source);
        let mut drained = rows.into_iter();
        let escaped = drained.next().unwrap();
        assert_eq!(buffer_retired.load(Ordering::SeqCst), 0);
        drop(drained);
        assert_eq!(buffer_retired.load(Ordering::SeqCst), 1);
        assert_eq!(frame_retired.load(Ordering::SeqCst), 0);
        assert_eq!(
            escaped.capture.as_step().outcome,
            CaptureStepOutcome::Untracked
        );
        assert_eq!(escaped.capture.as_step().cumulative_usage.captures, 7);
        drop(escaped);
        assert_eq!(frame_retired.load(Ordering::SeqCst), 0);
        drop(alias);
        assert_eq!(frame_retired.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn activation_capture_keeps_shared_frame_and_wire_after_buffer_retirement() {
        use crate::speculative::{SpeculativeActivationOrigin, SpeculativeActivationPhase};
        let retired = Arc::new(AtomicUsize::new(0));
        let buffer_retired = Arc::new(AtomicUsize::new(0));
        let shared = SharedCapturedStep::retain(frame(), Retires(retired.clone()));
        let source = shared.clone();
        let record = SpeculativeActivationCapture {
            admission_identity: Some("admitted-internal-plan".into()),
            invocation: 17,
            origin: SpeculativeActivationOrigin {
                request: crate::SpeculativeRequestId::new(0),
                committed_tokens: 1,
                prediction: 3,
                prefix_digest: [7; 32],
                optimistic: true,
            },
            phase: SpeculativeActivationPhase::Proposal { depth: 2 },
            prefill_span: None,
            completed: false,
            captures: shared,
            prefill_reductions: None,
        };
        let wire = serde_json::to_value(&record).unwrap();
        let decoded: SpeculativeActivationCapture = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), wire);
        assert_eq!(record, decoded);
        let alias = record.clone();
        assert!(source.same_storage(&alias.captures));
        assert_eq!(
            source.records()[0].selection_id.as_ptr(),
            alias.captures.as_step().records[0].selection_id.as_ptr()
        );
        let mut rows = SpeculativeBuffer::try_new_retained(
            1,
            HostPreparationAuthority::retain(Retires(buffer_retired.clone())),
        )
        .unwrap();
        rows.try_push(record).unwrap();
        let mut drained = rows.into_iter();
        let escaped = drained.next().unwrap();
        drop((drained, source));
        assert_eq!(buffer_retired.load(Ordering::SeqCst), 1);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert_eq!(escaped.captures.as_step().cumulative_usage.captures, 7);
        assert!(!escaped.completed);
        drop(escaped);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
