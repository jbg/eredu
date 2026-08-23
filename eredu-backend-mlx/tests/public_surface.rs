use eredu_backend_mlx::backend::{
    error::Error,
    nn::MlxNeuralBackend,
    runtime::{cache::state::MlxKeyValueState, checkpoint::store::WeightStoreError},
    MlxBackend,
};

fn assert_public_type<T: ?Sized>() {}

#[test]
fn reusable_backend_modules_are_rooted_directly_under_backend() {
    assert_public_type::<Error>();
    assert_public_type::<MlxBackend<'static>>();
    assert_public_type::<MlxNeuralBackend>();
    assert_public_type::<MlxKeyValueState>();
    assert_public_type::<WeightStoreError>();
}
