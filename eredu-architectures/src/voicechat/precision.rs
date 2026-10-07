//! Pinned perception and speech generation arithmetic uses Float32.

use eredu_nn::{ParameterMetadata, ParameterVisitor, Parameterized, Tensor, TensorElementType};

/// Rejects unknown or mixed precision before the component submits operations.
/// Callers translate the reason into their component's typed policy error.
pub(crate) fn require_float32<T: Tensor>(
    module: &impl Parameterized<T>,
    input: &T,
) -> Result<(), &'static str> {
    if input.element_type() != Some(TensorElementType::F32) {
        return Err("VoiceChat neural input requires known Float32 precision");
    }
    struct Check(bool);
    impl<'a, T: Tensor> ParameterVisitor<'a, T> for Check {
        fn visit(&mut self, _: ParameterMetadata, value: &'a T) {
            self.0 &= value.element_type() == Some(TensorElementType::F32);
        }
    }
    let mut check = Check(true);
    module.visit_parameters(&mut check);
    if !check.0 {
        return Err("VoiceChat neural parameters require known Float32 precision");
    }
    Ok(())
}
