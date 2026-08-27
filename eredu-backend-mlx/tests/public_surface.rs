use eredu_backend_mlx::backend::{
    config::ModelLoadOptions,
    error::Error,
    nn::generation::sample as backend_sample,
    nn::shared::MlxNeuralBackend,
    runtime::{
        cache::state::MlxKeyValueState,
        checkpoint::quantization::{CheckpointQuantizationOptions, CheckpointQuantizationReport},
        checkpoint::store::WeightStoreError,
        generation::sampler::{MlxSamplingBackend, Sampler as BackendSampler, SpeculativeSampler},
    },
    topology::{DeviceAssignment, MlxParallelContext},
    MlxBackend, MlxCompletion, MlxModel, MlxModelConfig,
};
use eredu_backend_mlx::native::{
    error::Exception, random::RandomState, Array, MlxDrafter, MlxInspectionOptions, MlxModelInput,
    MlxModelOutput, MlxModelSession, MlxRealtimeModel, MlxSessionCompletion, Stream,
};
use eredu_backend_mlx::MlxTensor;

fn assert_public_type<T: ?Sized>() {}

#[test]
fn reusable_backend_modules_are_rooted_directly_under_backend() {
    assert_public_type::<Error>();
    assert_public_type::<MlxBackend<'static>>();
    assert_public_type::<MlxNeuralBackend>();
    assert_public_type::<MlxKeyValueState>();
    assert_public_type::<WeightStoreError>();
}

#[test]
fn reusable_sampling_api_is_rooted_under_backend() {
    let _: fn(&Array, f32, Option<&mut RandomState>, &Stream) -> Result<Array, Exception> =
        backend_sample;
    assert_public_type::<MlxSamplingBackend>();
    assert_public_type::<dyn BackendSampler>();
    assert_public_type::<dyn SpeculativeSampler>();
}

#[test]
fn backend_owned_types_are_rooted_under_backend_ownership_modules() {
    assert_public_type::<CheckpointQuantizationOptions>();
    assert_public_type::<CheckpointQuantizationReport>();
    assert_public_type::<DeviceAssignment>();
    assert_public_type::<MlxCompletion>();
    assert_public_type::<MlxModel>();
    assert_public_type::<MlxModelConfig>();
    assert_public_type::<MlxParallelContext>();
    assert_public_type::<ModelLoadOptions>();
}

#[test]
fn composition_owned_native_types_have_one_public_native_path() {
    assert_public_type::<MlxDrafter>();
    assert_public_type::<MlxInspectionOptions>();
    assert_public_type::<MlxModelInput>();
    assert_public_type::<MlxModelOutput>();
    assert_public_type::<MlxModelSession<'static>>();
    assert_public_type::<MlxSessionCompletion>();
}

#[test]
fn native_realtime_adapter_exposes_architecture_owned_identity() {
    let _: fn(&MlxRealtimeModel) -> eredu_architectures::moshi::EffectiveModelType =
        MlxRealtimeModel::effective_model_type;
}

#[test]
fn model_output_exposes_backend_owned_tensors() {
    let _: fn(&MlxModelOutput) -> Option<&MlxTensor> = MlxModelOutput::logits;
    let _: fn(MlxModelOutput) -> Option<MlxTensor> = MlxModelOutput::into_logits;
}
