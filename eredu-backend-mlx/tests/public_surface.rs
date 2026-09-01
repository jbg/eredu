use eredu_backend_mlx::backend::{
    error::Error,
    nn::shared::MlxNeuralBackend,
    runtime::{
        cache::state::MlxKeyValueState,
        checkpoint::quantization::{CheckpointQuantizationOptions, CheckpointQuantizationReport},
        checkpoint::store::CheckpointMaterializationError,
        generation::MlxSamplingBackend,
    },
    MlxBackend, MlxCompletion, MlxModel,
};
use eredu_backend_mlx::native::{
    DeviceAssignment, MlxDrafter, MlxInspectionOptions, MlxModelInput, MlxModelOutput,
    MlxModelSession, MlxParallelPlan, MlxRealtimeModel, MlxSessionCompletion,
};
use eredu_backend_mlx::{MlxLoadRequest, MlxModelConfig, MlxTensor};
use eredu_core::InspectableBackendSession;

fn assert_public_type<T: ?Sized>() {}

#[test]
fn reusable_backend_modules_are_rooted_directly_under_backend() {
    assert_public_type::<Error>();
    assert_public_type::<MlxBackend<'static>>();
    assert_public_type::<MlxNeuralBackend>();
    assert_public_type::<MlxKeyValueState>();
    assert_public_type::<CheckpointMaterializationError>();
}

#[test]
fn reusable_sampling_primitives_are_rooted_under_backend() {
    assert_public_type::<MlxSamplingBackend>();
}

#[test]
fn backend_and_composition_types_are_rooted_under_their_ownership_modules() {
    assert_public_type::<CheckpointQuantizationOptions>();
    assert_public_type::<CheckpointQuantizationReport>();
    assert_public_type::<MlxCompletion>();
    assert_public_type::<MlxModel>();
    assert_public_type::<MlxModelConfig>();
    assert_public_type::<MlxLoadRequest>();
}

#[test]
fn composition_owned_native_types_have_one_public_native_path() {
    assert_public_type::<DeviceAssignment>();
    assert_public_type::<MlxDrafter>();
    assert_public_type::<MlxInspectionOptions>();
    assert_public_type::<MlxModelInput>();
    assert_public_type::<MlxModelOutput>();
    assert_public_type::<MlxModelSession<'static>>();
    assert_public_type::<MlxParallelPlan>();
    assert_public_type::<MlxSessionCompletion>();
}

#[test]
fn model_session_exposes_the_neutral_inspection_contract() {
    fn assert_inspectable<T: InspectableBackendSession<MlxBackend<'static>>>() {}

    assert_inspectable::<MlxModelSession<'static>>();
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
