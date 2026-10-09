//! Independent NVIDIA MoG residual-stack, guidance and selected Gaussian outputs.
use super::*;
use eredu_architectures::voicechat::tts_mog::{MogDimensions, MogHead};
use serde_json::Value;

fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    )
}
fn dimensions(value: &Value) -> MogDimensions {
    let i = |name: &str| value[name].as_i64().unwrap() as i32;
    MogDimensions {
        hidden: i("hidden"),
        intermediate: i("intermediate"),
        latent: i("latent"),
        mixtures: i("mixtures"),
        rank: i("low_rank"),
        layers: i("mlp_layers") as usize,
        epsilon: value["eps"].as_f64().unwrap() as f32,
        min_log_std: value["min_log_std"].as_f64().unwrap() as f32,
    }
}
struct Bind<'a>(&'a Value);
impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, target: &'a mut NumericTensor) {
        *target = tensor(&self.0[metadata.id.as_str()]);
    }
}
fn compare(actual: &NumericTensor, expected: &Value, fixture: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let tolerance = &fixture["comparison"];
    let absolute = tolerance["absolute_tolerance"].as_f64().unwrap() as f32;
    let relative = tolerance["relative_tolerance"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual.data.iter().zip(expected.data).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "element {index}: {actual} versus NVIDIA {expected}"
        );
    }
}

#[test]
fn voicechat_mog_matches_nvidia_guidance_low_rank_means_and_gaussian_draws() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let mut head =
        MogHead::<NumericBackend>::new(dimensions(&fixture["dimensions"]), &context).unwrap();
    head.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    for case in ["released_policy", "top_k_without_guidance"] {
        let case = &fixture[case];
        let noise = case["policy"]["noise_scale"].as_f64().unwrap() as f32;
        for step in case["steps"].as_array().unwrap() {
            let prediction = head
                .predict(
                    &tensor(&step["input"]),
                    step["guidance_scale"].as_f64().unwrap() as f32,
                    &context,
                )
                .unwrap();
            compare(&prediction.hidden, &step["guided_hidden"], &fixture);
            compare(&prediction.logits, &step["logits"], &fixture);
            let decisions = step["mixture_indices"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            let decisions = NumericTensor::from_i32_slice(
                &decisions,
                &prediction.hidden.shape()[..2],
                &context,
            )
            .unwrap();
            let sample = head
                .selected(
                    &prediction,
                    &decisions,
                    &tensor(&step["gaussian_draw"]),
                    noise,
                    &context,
                )
                .unwrap();
            compare(&sample.mean, &step["mu"], &fixture);
            compare(&sample.log_std, &step["logs"], &fixture);
            compare(&sample.latent, &step["latent"], &fixture);
        }
    }
}

#[test]
fn voicechat_rvq_refinement_matches_nvidia_and_replays_the_same_transition() {
    use eredu_architectures::voicechat::tts_rvq::{RefinementPolicy, RvqRefinement};
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let case = &fixture["released_policy"];
    let steps = case["steps"].as_array().unwrap();
    let context = NumericContext::default();
    let mut head =
        MogHead::<NumericBackend>::new(dimensions(&fixture["dimensions"]), &context).unwrap();
    head.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    let mut refinement = RvqRefinement::<NumericBackend>::new(5, 4, 4, &context).unwrap();
    refinement.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    let policy = RefinementPolicy {
        guidance_scale: 0.2,
        noise_scale: 0.001,
        top_p: 0.95,
    };
    let hidden = tensor(&case["hidden"]);
    let conditional = hidden.axis_slice(0, 0, 1);
    let unconditional = hidden.axis_slice(0, 1, 2);
    let initial = refinement.start(1, 1, &context).unwrap();
    assert_eq!(initial.next_slot(), Some(3));
    assert_eq!(initial.completed_depth(), 0);
    assert_eq!(
        refinement.depthsum(initial.codes(), &context).unwrap().data,
        vec![0.; 4]
    );
    let mut state = initial.clone();
    for step in steps {
        let snapshot = state.clone();
        assert_eq!(
            state.next_slot(),
            Some(step["slot"].as_u64().unwrap() as usize)
        );
        let input = refinement
            .input(&state, &conditional, Some(&unconditional), policy, &context)
            .unwrap();
        compare(&input, &step["input"], &fixture);
        // A malformed sampled payload cannot modify a retained candidate.
        assert!(refinement
            .advance(&state, &NumericTensor::zeros(vec![1, 1, 3]), &context)
            .is_err());
        let latent = tensor(&step["latent"]);
        state = refinement.advance(&state, &latent, &context).unwrap();
        assert_eq!(snapshot.codes().data, tensor(&step["codes_before"]).data);
        assert_eq!(state.codes().data, tensor(&step["codes_after"]).data);
        compare(
            &refinement.depthsum(state.codes(), &context).unwrap(),
            &step["summed_feedback"],
            &fixture,
        );
        let fork = refinement.advance(&snapshot, &latent, &context).unwrap();
        assert_eq!(fork.codes().data, state.codes().data);
    }
    assert_eq!(state.completed_depth(), 31);
    assert_eq!(state.next_slot(), None);
    assert_eq!(initial.codes().data, vec![5.; 31]);
    let mut calls = 0;
    let generated = refinement
        .generate(
            &conditional,
            Some(&unconditional),
            policy,
            &mut head,
            |slot, logits: &NumericTensor, top_p| {
                let step = &steps[calls];
                calls += 1;
                assert_eq!(slot, step["slot"].as_u64().unwrap() as usize);
                assert_eq!(top_p, 0.95);
                compare(logits, &step["logits"], &fixture);
                let values = step["mixture_indices"]["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_i64().unwrap() as i32)
                    .collect::<Vec<_>>();
                Ok((
                    NumericTensor::from_i32_slice(&values, &[1, 1], &context).unwrap(),
                    tensor(&step["gaussian_draw"]),
                ))
            },
            &context,
        )
        .unwrap();
    assert_eq!(calls, 5, "zero-depth slots must consume no draws");
    assert_eq!(generated.codes().data, tensor(&case["codes"]).data);
    assert_eq!(generated.codes().data, state.codes().data);
    compare(
        &refinement.depthsum(generated.codes(), &context).unwrap(),
        &case["summed_feedback"],
        &fixture,
    );
}

#[test]
fn voicechat_rvq_parameter_rebinding_invalidates_retained_codebook_norms() {
    use eredu_architectures::voicechat::tts_rvq::RvqRefinement;
    let context = NumericContext::default();
    let mut refinement = RvqRefinement::<NumericBackend>::new(2, 1, 1, &context).unwrap();
    let mut parameters = BTreeMap::from([
        (
            "tts_model.tts_model.rvq_embs".into(),
            NumericTensor::new(vec![31, 2, 1], vec![0., 10.].repeat(31)),
        ),
        (
            "tts_model.tts_model.embed_code.weight".into(),
            NumericTensor::new(vec![1, 1], vec![1.]),
        ),
    ]);
    replace_numeric_parameters(&mut refinement, &parameters);
    let state = refinement.start(1, 1, &context).unwrap();
    let latent = NumericTensor::new(vec![1, 1, 1], vec![1.]);
    let first = refinement.advance(&state, &latent, &context).unwrap();
    assert_eq!(first.codes().data[0], 0.);
    // New means reverse the nearest mean. Retaining old [0,100] norms chooses 0.
    parameters.insert(
        "tts_model.tts_model.rvq_embs".into(),
        NumericTensor::new(vec![31, 2, 1], vec![10., 1.].repeat(31)),
    );
    replace_numeric_parameters(&mut refinement, &parameters);
    let rebound = refinement.advance(&state, &latent, &context).unwrap();
    assert_eq!(rebound.codes().data[0], 1.);
    assert_eq!(first.codes().data[0], 0.);
}
