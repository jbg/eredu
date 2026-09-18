//! Full raw capture through the source-explicit managed facade and shared cursor.
use super::*;
use eredu_core::observation::TensorObservationData;

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_managed_observed_plain_delivers_full_prompt_and_retained_decode_frames() {
    let root = managed_fixture(fixture(false));
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let discovery = model.capture_discovery().unwrap();
    let usage = CaptureUsage {
        captures: 4,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    let mut plan = CapturePlan::none();
    plan.selections.push(CaptureSelection {
        id: "full logits".into(),
        path: "model.logits".into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform: CaptureTransform::FullTensor,
    });
    plan.limits.per_step = usage;
    plan.limits.cumulative = usage;
    let capture = SharedCapturePlan::new(
        plan.admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 5,
                max_predictions: 4,
            },
        )
        .unwrap(),
    );
    let mut frames = Vec::new();
    let mut tokens = Vec::new();
    let mut observer = |token: Option<u32>, frame: Option<SharedCapturedStep>, seconds: f64| {
        assert!(seconds >= 0.0);
        tokens.push(token.expect("successful committed token"));
        let Some(frame) = frame else {
            panic!("original capture returns its shared paid owner")
        };
        assert_eq!(frame.prediction_index() as usize, frames.len());
        assert_eq!(frame.records().len(), 1);
        frames.push(frame);
    };
    let cancellation = GenerationCancellationToken::new();
    let mut visible = String::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| {
        if let GenerationPlainTextEvent::TextDelta(delta) = event {
            visible.push_str(delta);
        }
    };
    let mut session = model
        .start_observed_managed_plain_text(
            &source,
            ManagedPlainTextRequest::new(PROMPT, settings(0.0)),
            capture,
            &cancellation,
            &mut observer,
        )
        .unwrap_or_else(report_failure)
        .expect("live observed request");
    while session.finish_reason().is_none() {
        session = session
            .advance(&cancellation, &mut emit)
            .unwrap_or_else(report_failure);
    }
    let output = session
        .into_output()
        .unwrap_or_else(|_| panic!("completed session"));
    drop(observer);
    assert_eq!(tokens, output.token_ids.as_ref());
    assert_eq!(
        frames.len(),
        4,
        "exactly one delivery for each committed prediction"
    );
    assert_eq!(output.text.as_str(), visible);
    drop((source, model, root));
    // Shared frame and output remain readable after every model/source owner retires.
    for (prediction, frame) in frames.iter().enumerate() {
        assert_eq!(frame.outcome(), CaptureStepOutcome::Committed);
        let record = &frame.records()[0];
        assert_eq!(
            record.selected_shape.as_deref(),
            Some([1, if prediction == 0 { 5 } else { 1 }, 64].as_slice())
        );
        let TensorObservationData::F32(values) =
            record.payload.as_ref().unwrap().as_tensor().unwrap().data()
        else {
            panic!("floating host representation")
        };
        assert_eq!(values.len(), if prediction == 0 { 5 * 64 } else { 64 });
        assert!(values.iter().all(|v| v.is_finite()));
        assert!(values.iter().any(|v| v.abs() > 1e-6));
        let alias = frame.clone();
        assert!(alias.same_storage(frame));
    }
    assert_eq!(output.token_ids.as_ref().len(), 4);
}

#[path = "capture/saved.rs"]
mod saved;
pub(super) use saved::{check_saved_capture_loaded, check_saved_prepared_capture_loaded,
    check_saved_capture_loaded_with_plan};

#[path = "capture/token_scores.rs"]
mod token_scores;

#[path = "capture/candidates.rs"]
mod candidates;

#[path = "capture/summary.rs"]
mod summary;

#[path = "capture/histogram.rs"]
mod histogram;

#[path = "capture/routed.rs"]
mod routed;

#[path = "capture/interventions.rs"]
mod interventions;

pub(super) use interventions::{run_intervention_loaded,run_intervention_evidence_loaded};
