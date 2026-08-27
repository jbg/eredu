use eredu_backend_mlx::backend::{
    error::Error,
    nn::MlxNeuralBackend,
    runtime::{cache::state::MlxKeyValueState, checkpoint::store::WeightStoreError},
    MlxBackend,
};
use eredu_backend_mlx::native::{
    error::Exception, random::RandomState, sample, Array, CheckpointQuantizationOptions,
    CheckpointQuantizationReport, DeviceAssignment, MlxCompletion, MlxDrafter, MlxError,
    MlxInspectionOptions, MlxModel, MlxModelConfig, MlxModelInput, MlxModelOutput, MlxModelSession,
    MlxParallelContext, MlxRealtimeModel, MlxSessionCompletion, ModelLoadOptions, Sampler, Stream,
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
fn raw_sampling_api_is_rooted_under_native() {
    let _: fn(&Array, f32, Option<&mut RandomState>, &Stream) -> Result<Array, Exception> = sample;
    assert_public_type::<dyn Sampler>();
}

#[test]
fn native_model_execution_types_are_rooted_under_native() {
    assert_public_type::<CheckpointQuantizationOptions>();
    assert_public_type::<CheckpointQuantizationReport>();
    assert_public_type::<DeviceAssignment>();
    assert_public_type::<MlxCompletion>();
    assert_public_type::<MlxDrafter>();
    assert_public_type::<MlxError>();
    assert_public_type::<MlxInspectionOptions>();
    assert_public_type::<MlxModel>();
    assert_public_type::<MlxModelConfig>();
    assert_public_type::<MlxModelInput>();
    assert_public_type::<MlxModelOutput>();
    assert_public_type::<MlxModelSession<'static>>();
    assert_public_type::<MlxParallelContext>();
    assert_public_type::<MlxSessionCompletion>();
    assert_public_type::<ModelLoadOptions>();
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
