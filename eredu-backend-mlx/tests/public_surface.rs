use eredu_backend_mlx::backend::{
    error::Error,
    nn::MlxNeuralBackend,
    runtime::{cache::state::MlxKeyValueState, checkpoint::store::WeightStoreError},
    MlxBackend,
};
use eredu_backend_mlx::native::{
    error::Exception, random::RandomState, sample, Array, DeviceAssignment, MlxCompletion,
    MlxDrafter, MlxModelInput, MlxModelSession, MlxParallelContext, MlxRealtimeModel,
    MlxSessionCompletion, Sampler, Stream,
};
use eredu_backend_mlx::{MlxModelOutput, MlxTensor};

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
fn raw_sampling_api_is_rooted_under_native() {
    let _: fn(&Array, f32, Option<&mut RandomState>, &Stream) -> Result<Array, Exception> = sample;
    assert_public_type::<dyn Sampler>();
}

#[test]
fn native_model_execution_types_are_rooted_under_native() {
    assert_public_type::<DeviceAssignment>();
    assert_public_type::<MlxCompletion>();
    assert_public_type::<MlxDrafter>();
    assert_public_type::<MlxModelInput>();
    assert_public_type::<MlxModelSession<'static>>();
    assert_public_type::<MlxParallelContext>();
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
