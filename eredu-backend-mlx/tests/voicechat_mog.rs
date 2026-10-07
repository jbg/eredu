//! Pinned NVIDIA MoG predictions and selected Gaussian samples on MLX CPU.
use eredu_architectures::voicechat::tts_mog::{MogDimensions, MogHead};
use eredu_backend_mlx::backend::runtime::generation::MlxSamplingBackend;
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor, TensorElementType};
use eredu_runtime::{DenseSamplingBackend, GumbelTruncation};
use safemlx::{Array, Device, DeviceType, Stream};
use serde_json::Value;

fn tensor(value: &Value) -> MlxTensor {
    let shape = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let values = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    MlxTensor::from_array(Array::from_slice(&values, &shape))
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
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, target: &'a mut MlxTensor) {
        *target = tensor(&self.0[metadata.id.as_str()]);
    }
}
fn compare(actual: &MlxTensor, expected: &Value, fixture: &Value, stream: &Stream) {
    assert_eq!(actual.shape(), tensor(expected).shape());
    let tolerance = &fixture["comparison"];
    let absolute = tolerance["absolute_tolerance"].as_f64().unwrap() as f32;
    let relative = tolerance["relative_tolerance"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual
        .to_f32_vec(stream)
        .unwrap()
        .iter()
        .zip(expected["values"].as_array().unwrap())
        .enumerate()
    {
        let expected = expected.as_f64().unwrap() as f32;
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "element {index}: {actual} versus NVIDIA {expected}"
        );
    }
}

#[test]
fn cpu_mog_matches_nvidia_guidance_low_rank_means_and_gaussian_draws() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut head = MogHead::<Backend>::new(dimensions(&fixture["dimensions"]), &stream).unwrap();
    head.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    for case in ["released_policy", "top_k_without_guidance"] {
        let case = &fixture[case];
        let noise = case["policy"]["noise_scale"].as_f64().unwrap() as f32;
        for step in case["steps"].as_array().unwrap() {
            let prediction = head
                .predict(
                    &tensor(&step["input"]),
                    step["guidance_scale"].as_f64().unwrap() as f32,
                    &stream,
                )
                .unwrap();
            compare(
                &prediction.hidden,
                &step["guided_hidden"],
                &fixture,
                &stream,
            );
            compare(&prediction.logits, &step["logits"], &fixture, &stream);
            let decisions = step["mixture_indices"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            let decision_policy = if step["top_p_or_k"].is_u64() {
                GumbelTruncation::TopK(step["top_p_or_k"].as_u64().unwrap() as usize)
            } else {
                GumbelTruncation::TopP(step["top_p_or_k"].as_f64().unwrap())
            };
            let sampled_decisions = MlxSamplingBackend::gumbel_max(
                &prediction.logits,
                &tensor(&step["uniform_draw"]),
                decision_policy,
                1e-8,
                &stream,
            )
            .unwrap();
            assert_eq!(sampled_decisions.to_i32_vec(&stream).unwrap(), decisions);
            let decisions = sampled_decisions;
            let sample = head
                .selected(
                    &prediction,
                    &decisions,
                    &tensor(&step["gaussian_draw"]),
                    noise,
                    &stream,
                )
                .unwrap();
            compare(&sample.mean, &step["mu"], &fixture, &stream);
            compare(&sample.log_std, &step["logs"], &fixture, &stream);
            compare(&sample.latent, &step["latent"], &fixture, &stream);
        }
    }
    // A low-rank table cannot silently narrow an otherwise F32 speech stack.
    struct Narrow<'a>(&'a Stream);
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for Narrow<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut MlxTensor) {
            if metadata.id.as_str() == "tts_model.tts_model.mog_head.low_mat" {
                *value = value.cast_float(TensorElementType::F16, self.0).unwrap();
            }
        }
    }
    head.visit_parameters_mut(&mut Narrow(&stream));
    assert!(head
        .predict(
            &tensor(&fixture["released_policy"]["steps"][0]["input"]),
            0.2,
            &stream
        )
        .is_err());
}

#[test]
fn cpu_rvq_refinement_matches_nvidia_controlled_steps_and_replay() {
    use eredu_architectures::voicechat::tts_rvq::{RefinementPolicy, RvqRefinement};
    use eredu_nn::Index;
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let case = &fixture["released_policy"];
    let steps = case["steps"].as_array().unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut head = MogHead::<Backend>::new(dimensions(&fixture["dimensions"]), &stream).unwrap();
    head.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    let mut refinement = RvqRefinement::<Backend>::new(5, 4, 4, &stream).unwrap();
    refinement.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    let policy = RefinementPolicy {
        guidance_scale: 0.2,
        noise_scale: 0.001,
        top_p: 0.95,
    };
    let hidden = tensor(&case["hidden"]);
    let conditional = hidden.index(&[Index::Range(0, 1)], &stream).unwrap();
    let unconditional = hidden.index(&[Index::Range(1, 2)], &stream).unwrap();
    let initial = refinement.start(1, 1, &stream).unwrap();
    assert_eq!(initial.next_slot(), Some(3));
    assert_eq!(
        refinement
            .depthsum(initial.codes(), &stream)
            .unwrap()
            .to_f32_vec(&stream)
            .unwrap(),
        vec![0.; 4]
    );
    let mut state = initial.clone();
    let codes = |value: &MlxTensor| value.to_i32_vec(&stream).unwrap();
    let expected_codes = |value: &Value| {
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect::<Vec<_>>()
    };
    for step in steps {
        let snapshot = state.clone();
        assert_eq!(
            state.next_slot(),
            Some(step["slot"].as_u64().unwrap() as usize)
        );
        let input = refinement
            .input(&state, &conditional, Some(&unconditional), policy, &stream)
            .unwrap();
        compare(&input, &step["input"], &fixture, &stream);
        let invalid = MlxTensor::full_f32(0., &[1, 1, 3], &stream).unwrap();
        assert!(refinement.advance(&state, &invalid, &stream).is_err());
        let latent = tensor(&step["latent"]);
        state = refinement.advance(&state, &latent, &stream).unwrap();
        assert_eq!(
            codes(snapshot.codes()),
            expected_codes(&step["codes_before"])
        );
        assert_eq!(codes(state.codes()), expected_codes(&step["codes_after"]));
        compare(
            &refinement.depthsum(state.codes(), &stream).unwrap(),
            &step["summed_feedback"],
            &fixture,
            &stream,
        );
        let fork = refinement.advance(&snapshot, &latent, &stream).unwrap();
        assert_eq!(codes(fork.codes()), codes(state.codes()));
    }
    assert_eq!(state.completed_depth(), 31);
    assert_eq!(state.next_slot(), None);
    assert_eq!(codes(initial.codes()), vec![5; 31]);
    let mut calls = 0;
    let generated = refinement
        .generate(
            &conditional,
            Some(&unconditional),
            policy,
            &mut head,
            |slot, logits: &MlxTensor, top_p| {
                let step = &steps[calls];
                calls += 1;
                assert_eq!(slot, step["slot"].as_u64().unwrap() as usize);
                assert_eq!(top_p, 0.95);
                compare(logits, &step["logits"], &fixture, &stream);
                let decisions = MlxSamplingBackend::gumbel_max(
                    logits,
                    &tensor(&step["uniform_draw"]),
                    GumbelTruncation::TopP(top_p),
                    1e-8,
                    &stream,
                )
                .unwrap();
                assert_eq!(codes(&decisions), expected_codes(&step["mixture_indices"]));
                Ok((decisions, tensor(&step["gaussian_draw"])))
            },
            &stream,
        )
        .unwrap();
    assert_eq!(calls, 5, "zero-depth slots must consume no draws");
    assert_eq!(codes(generated.codes()), expected_codes(&case["codes"]));
    assert_eq!(codes(generated.codes()), codes(state.codes()));
    compare(
        &refinement.depthsum(generated.codes(), &stream).unwrap(),
        &case["summed_feedback"],
        &fixture,
        &stream,
    );
}
