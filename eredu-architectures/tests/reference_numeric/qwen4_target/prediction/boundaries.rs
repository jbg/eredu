//! Traversal and publication edits must reach the next operator and sampler.
use super::*;
const UNITS: [&str; 5] = [
    "model.layers.0",
    "model.layers.1.ple",
    "model.layers.1",
    "model.layers.2",
    "model.layers.3",
];
fn tensor<'a>(
    invocation: &'a SpeculativeActivationCapture,
    path: &str,
) -> &'a eredu_core::TensorObservation {
    let record = invocation
        .captures
        .records
        .iter()
        .find(|r| r.path == path)
        .unwrap();
    assert_eq!(record.outcome, CaptureOutcome::Captured, "{path}");
    let Some(CapturePayload::Tensor(value)) = &record.payload else {
        panic!("missing {path}")
    };
    value
}
fn values<'a>(invocation: &'a SpeculativeActivationCapture, path: &str) -> &'a [f32] {
    let eredu_core::TensorObservationData::F32(values) = tensor(invocation, path).data() else {
        panic!()
    };
    values
}
pub(super) fn validate(invocation: &SpeculativeActivationCapture) {
    if SpeculativeCaptureScope::Target.applies(invocation.phase) {
        let mut previous = "readout.embedding".to_owned();
        for unit in UNITS {
            assert_eq!(
                tensor(invocation, &previous),
                tensor(invocation, &format!("{unit}.input"))
            );
            previous = format!("{unit}.output");
        }
        assert_eq!(
            tensor(invocation, &previous),
            tensor(invocation, "readout.residual")
        );
        assert_eq!(
            tensor(invocation, "readout.linear"),
            tensor(invocation, eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
        );
    }
    for depth in 0..2 {
        if (SpeculativeCaptureScope::Prediction { depth }).applies(invocation.phase) {
            let root = format!("mtp.layers.{depth}.prediction");
            assert_eq!(
                tensor(invocation, &format!("{root}.readout.linear")),
                tensor(invocation, &format!("{root}.logits"))
            );
        }
    }
}
fn operation(path: &str, factor: f32) -> InterventionOperation {
    InterventionOperation {
        id: format!("edit-{path}"),
        target: path.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        action: if factor == 0. {
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            }
        } else {
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor,
            }
        },
        evidence: InterventionEvidence::None,
    }
}
pub(super) fn checks(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    discovery: &SpeculativeActivationDiscovery,
    plain: &Outcome,
) {
    let mut request = plan(discovery);
    for unit in UNITS {
        for seam in ["input", "output"] {
            request
                .interventions
                .operations
                .push(operation(&format!("{unit}.{seam}"), 1.));
        }
    }
    request
        .interventions
        .operations
        .push(operation(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, 1.));
    for depth in 0..2 {
        for seam in ["input", "embedding", "logits"] {
            request.interventions.operations.push(operation(
                &format!("mtp.layers.{depth}.prediction.{seam}"),
                1.,
            ));
        }
    }
    let mut options = super::super::options();
    options.activations = Some(request.admit(discovery).unwrap());
    let identity = run(selected, prepared, ctx, options, |session| {
        while let Some(step) = session.step()? {
            for invocation in &step.activations {
                validate(invocation);
                for edit in &invocation.captures.interventions {
                    let point = discovery
                        .interventions
                        .points
                        .iter()
                        .find(|p| p.path == edit.target)
                        .unwrap();
                    let scope = discovery
                        .bindings
                        .iter()
                        .find(|b| b.node_id == point.node_id)
                        .unwrap()
                        .scope;
                    assert_eq!(
                        edit.outcome,
                        if scope.applies(invocation.phase) {
                            InterventionOutcome::Applied
                        } else {
                            InterventionOutcome::Inactive
                        }
                    );
                }
            }
        }
        Ok(())
    });
    assert_eq!(identity.tokens, plain.tokens);
    exact(&identity.target, &plain.target, "boundary identity target");
    exact(
        &identity.prediction,
        &plain.prediction,
        "boundary identity prediction",
    );
    for (target, consumed) in [
        ("model.layers.0.input", "model.layers.0.mixer.input"),
        ("model.layers.0.output", "model.layers.1.ple.input"),
        ("model.layers.1.ple.input", "model.layers.1.lexical.input"),
        ("model.layers.1.ple.output", "model.layers.1.input"),
        ("model.layers.3.output", "readout.residual"),
    ] {
        let mut request = plan(discovery);
        request.interventions.operations.push(operation(target, 0.));
        let mut options = super::super::options();
        options.activations = Some(request.admit(discovery).unwrap());
        run(selected, prepared, ctx, options, |session| {
            let step = session.step()?.unwrap();
            let invocation = step
                .activations
                .iter()
                .find(|a| a.phase == SpeculativeActivationPhase::TargetPrefill)
                .unwrap();
            assert!(values(invocation, target).iter().any(|v| *v != 0.));
            assert!(
                values(invocation, consumed).iter().all(|v| *v == 0.),
                "replacement reaches {consumed}"
            );
            Ok(())
        });
    }
    for depth in 0..2 {
        let root = format!("mtp.layers.{depth}.prediction");
        let mut request = plan(discovery);
        for seam in ["input", "embedding"] {
            request
                .interventions
                .operations
                .push(operation(&format!("{root}.{seam}"), 0.));
        }
        let mut options = super::super::options();
        options.activations = Some(request.admit(discovery).unwrap());
        run(selected, prepared, ctx, options, |session| {
            let step = session.step()?.unwrap();
            let invocation = step
                .activations
                .iter()
                .find(|a| a.phase == SpeculativeActivationPhase::PredictionPrefill)
                .unwrap();
            for seam in ["input", "embedding"] {
                assert!(values(invocation, &format!("{root}.{seam}"))
                    .iter()
                    .any(|v| *v != 0.));
            }
            assert!(
                values(invocation, &format!("{root}.fusion"))
                    .iter()
                    .all(|v| *v == 0.),
                "fusion consumes edited inputs"
            );
            Ok(())
        });
    }
    let mut mask = operation(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, 1.);
    mask.action = InterventionAction::MaskLogits {
        dtype: InterventionDtype::Float32,
        token_ids: vec![plain.tokens[0]],
    };
    let mut request = plan(discovery);
    request.interventions.operations.push(mask);
    let mut wrong_stage = request.clone();
    wrong_stage.interventions.operations[0].target = "readout.linear".into();
    assert!(wrong_stage.admit(discovery).is_err());
    let mut out_of_range = request.clone();
    out_of_range.interventions.operations[0].action = InterventionAction::MaskLogits {
        dtype: InterventionDtype::Float32,
        token_ids: vec![32],
    };
    assert!(out_of_range.admit(discovery).is_err());
    let mut options = super::super::options();
    options.activations = Some(request.admit(discovery).unwrap());
    let mut expected = None;
    let masked = run(selected, prepared, ctx, options, |session| {
        let step = session.step()?.unwrap();
        let invocation = step
            .activations
            .iter()
            .find(|a| a.phase == SpeculativeActivationPhase::TargetPrefill)
            .unwrap();
        let row = values(invocation, eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
            .chunks(32)
            .last()
            .unwrap();
        expected = row
            .iter()
            .enumerate()
            .filter(|(id, _)| *id != plain.tokens[0] as usize)
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .map(|(id, _)| id as u32);
        Ok(())
    });
    assert_eq!(masked.tokens, [expected.unwrap()]);
    assert_ne!(masked.tokens, plain.tokens[..1]);
    // The fixture's stable greedy tie break chooses vocabulary ID zero.
    let mut request = plan(discovery);
    request
        .interventions
        .operations
        .push(operation(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, 0.));
    let mut options = super::super::options();
    options.activations = Some(request.admit(discovery).unwrap());
    let target = run(selected, prepared, ctx, options, |session| {
        let step = session.step()?.unwrap();
        let invocation = step
            .activations
            .iter()
            .find(|a| a.phase == SpeculativeActivationPhase::TargetPrefill)
            .unwrap();
        assert!(
            values(invocation, eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
                .iter()
                .any(|v| *v != 0.)
        );
        Ok(())
    });
    assert_eq!(target.tokens, [0]);
    assert_ne!(
        target.tokens,
        plain.tokens[..1],
        "publication edit changes the sampled token"
    );
    let mut request = plan(discovery);
    for depth in 0..2 {
        let path = format!("mtp.layers.{depth}.prediction.logits");
        assert_eq!(
            discovery
                .interventions
                .points
                .iter()
                .find(|p| p.path == path)
                .unwrap()
                .stage,
            InterventionStage::LogitsBeforeSampling
        );
        let mut mask = operation(&path, 1.);
        mask.action = InterventionAction::MaskLogits {
            dtype: InterventionDtype::Float32,
            token_ids: (1..32).collect(),
        };
        request.interventions.operations.push(mask);
    }
    let mut options = super::super::options();
    options.activations = Some(request.admit(discovery).unwrap());
    let prediction = run(selected, prepared, ctx, options, |session| {
        let mut proposals = 0;
        while let Some(step) = session.step()? {
            if let Some(drafted) = step.drafted {
                assert!(!drafted.token_ids.is_empty());
                assert!(drafted.token_ids.iter().all(|token| *token == 0));
                proposals += 1;
            }
        }
        assert!(proposals > 0);
        Ok(())
    });
    // Draft logits affect proposals, while target verification remains canonical.
    assert_eq!(prediction.tokens, plain.tokens);
    let (_, frontiers) = oracle(selected, prepared, ctx, None);
    let frontier = prediction.target.clone().layer(0).unwrap().position() as usize - 3;
    exact(
        &prediction.target,
        &frontiers[frontier].0,
        "publication-edited canonical target",
    );
    exact(
        &prediction.prediction,
        &frontiers[frontier].1,
        "publication-edited canonical prediction",
    );
}
