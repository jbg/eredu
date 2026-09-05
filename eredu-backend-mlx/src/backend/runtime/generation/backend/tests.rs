use super::{effective_allowed_mask, MlxSamplingBackend};
use crate::MlxTensor;
use eredu_core::TokenFilter;
use eredu_runtime::{GenerationSampler, MirostatV2Sampler, Sampler, SamplingBackend, TokenDomain};
use safemlx::{random, transforms::async_eval_with_event, Array, Device, DeviceType};

use crate::backend::{nn::tensor::TokenValidationScope, random::RandomState, ExecutionContext};

#[test]
fn token_filter_accepts_a_truncated_output_vocabulary_prefix() {
    assert_eq!(
        effective_allowed_mask(&[false, true, false, true], 3).unwrap(),
        &[false, true, false]
    );
    assert!(effective_allowed_mask(&[false, false, true], 2).is_err());
    assert!(effective_allowed_mask(&[true], 2).is_err());
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_token_filter_precedes_sampling_policy() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let raw = MlxTensor::from_array(Array::from_slice(&[100.0f32, 10.0], &[1, 2]));
    let filter = TokenFilter::allowed(vec![false, true]).unwrap();
    let masked = MlxSamplingBackend::apply_token_filter(&raw, &filter, stream).unwrap();
    let mut sampler = GenerationSampler::new().top_k(1).top_p(1.0).min_p(0.0);

    let selected =
        Sampler::<MlxSamplingBackend>::sample(&mut sampler, &masked, 0.0, None, stream).unwrap();

    assert_eq!(selected.as_array().clone().item::<u32>(stream), 1);
    assert_eq!(sampler.generated_tokens(), &[1]);
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_mirostat_v2_samples_and_updates_mu() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let logits = MlxTensor::from_array(Array::from_slice(&[0.0f32, -100.0, -100.0], &[1, 3]));
    let mut state = RandomState::from_key(random::key(0).unwrap());
    let mut sampler = MirostatV2Sampler::new(5.0, 0.1).unwrap();

    let token =
        Sampler::<MlxSamplingBackend>::sample(&mut sampler, &logits, 1.0, Some(&mut state), stream)
            .unwrap();

    assert_eq!(token.as_array().clone().item::<u32>(stream), 0);
    assert!(sampler.mu() > 10.0);
    assert_eq!(sampler.generated_tokens(), &[0]);
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn mlx_token_domain_validation_is_deferred_to_completion() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = execution.stream();
    let scope = TokenValidationScope::begin().unwrap();
    let valid = MlxSamplingBackend::validate_token(
        &MlxTensor::from_array(Array::from_slice(&[0_u32, 4], &[2])),
        TokenDomain::new(5),
        stream,
    )
    .unwrap();
    assert_eq!(valid.as_array().dtype(), safemlx::Dtype::Int32);
    let validations = scope.finish();
    let event =
        async_eval_with_event(std::iter::once(valid.as_array()).chain(validations.arrays()))
            .unwrap();
    event.synchronize().unwrap();
    validations.validate_completed().unwrap();

    for invalid in [-1_i32, 5] {
        let scope = TokenValidationScope::begin().unwrap();
        let token = MlxSamplingBackend::validate_token(
            &MlxTensor::from_array(Array::from_slice(&[invalid], &[1])),
            TokenDomain::new(5),
            stream,
        )
        .expect("lazy device validation must not synchronize while building the graph");
        let validations = scope.finish();
        let event =
            async_eval_with_event(std::iter::once(token.as_array()).chain(validations.arrays()))
                .unwrap();
        event.synchronize().unwrap();
        assert!(validations.validate_completed().is_err());
    }
}
