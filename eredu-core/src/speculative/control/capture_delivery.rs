//! The existing capture delivery owner inside a speculative prediction record.
use super::{SpeculativePredictionCapture, SpeculativeActivationCapture};
use crate::capture::{CapturedStep, CapturedStepDelivery};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl Clone for SpeculativePredictionCapture {
    fn clone(&self) -> Self {
        Self {
            role: self.role,
            position: self.position,
            capture: match &self.capture {
                CapturedStepDelivery::Legacy(frame) => CapturedStepDelivery::Legacy(frame.clone()),
                CapturedStepDelivery::Shared(frame) => CapturedStepDelivery::Shared(frame.clone()),
            },
        }
    }
}
impl PartialEq for SpeculativePredictionCapture {
    fn eq(&self, other: &Self) -> bool {
        self.role == other.role
            && self.position == other.position
            && self.capture.as_step() == other.capture.as_step()
    }
}

impl Clone for SpeculativeActivationCapture {
    fn clone(&self) -> Self {
        Self {
            admission_identity: self.admission_identity.clone(),
            invocation: self.invocation,
            origin: self.origin,
            phase: self.phase,
            prefill_span: self.prefill_span,
            completed: self.completed,
            captures: match &self.captures {
                CapturedStepDelivery::Legacy(frame) => CapturedStepDelivery::Legacy(frame.clone()),
                CapturedStepDelivery::Shared(frame) => CapturedStepDelivery::Shared(frame.clone()),
            },
            prefill_reductions: self.prefill_reductions.clone(),
        }
    }
}
impl PartialEq for SpeculativeActivationCapture {
    fn eq(&self, other: &Self) -> bool {
        self.admission_identity == other.admission_identity
            && self.invocation == other.invocation
            && self.origin == other.origin
            && self.phase == other.phase
            && self.prefill_span == other.prefill_span
            && self.completed == other.completed
            && self.captures.as_step() == other.captures.as_step()
            && self.prefill_reductions == other.prefill_reductions
    }
}

pub(super) fn serialize<S: Serializer>(
    value: &CapturedStepDelivery,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    value.as_step().serialize(serializer)
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<CapturedStepDelivery, D::Error> {
    // Decoding wire records creates ordinary caller-owned data, never funding
    // or an original source/transaction witness.
    CapturedStep::deserialize(deserializer).map(CapturedStepDelivery::Legacy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HostPreparationAuthority, ObservationPosition, SpeculativeBuffer,
        capture::{
            CAPTURE_SCHEMA_VERSION, CaptureOutcome, CapturePhase, CaptureRecord,
            CaptureStepOutcome, CaptureUsage, SharedCapturedStep,
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
            capture: CapturedStepDelivery::Legacy(frame.clone()),
        };
        let frame_retired = Arc::new(AtomicUsize::new(0));
        let buffer_retired = Arc::new(AtomicUsize::new(0));
        let shared = SharedCapturedStep::retain(frame, Retires(frame_retired.clone()));
        let source = shared.clone();
        let record = SpeculativePredictionCapture {
            role: legacy.role,
            position: legacy.position,
            capture: CapturedStepDelivery::Shared(shared),
        };
        let wire = serde_json::to_value(&legacy).unwrap();
        assert_eq!(serde_json::to_value(&record).unwrap(), wire);
        let decoded: SpeculativePredictionCapture = serde_json::from_value(wire).unwrap();
        assert!(matches!(&decoded.capture, CapturedStepDelivery::Legacy(_)));
        assert_eq!(record, decoded);

        let alias = record.clone();
        assert!(source.same_storage(alias.capture.shared().unwrap()));
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
            captures: CapturedStepDelivery::Shared(shared),
            prefill_reductions: None,
        };
        let wire = serde_json::to_value(&record).unwrap();
        let decoded: SpeculativeActivationCapture = serde_json::from_value(wire.clone()).unwrap();
        assert!(matches!(&decoded.captures, CapturedStepDelivery::Legacy(_)));
        assert_eq!(serde_json::to_value(&decoded).unwrap(), wire);
        assert_eq!(record, decoded);
        let alias = record.clone();
        assert!(source.same_storage(alias.captures.shared().unwrap()));
        assert_eq!(source.records()[0].selection_id.as_ptr(),
            alias.captures.as_step().records[0].selection_id.as_ptr());
        let mut rows = SpeculativeBuffer::try_new_retained(
            1, HostPreparationAuthority::retain(Retires(buffer_retired.clone())),
        ).unwrap();
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
