//! One borrowed semantic sequence description for all native realizations.
use super::*;
use crate::decoder::identity::Metadata;

/// One exact segment of an architecture-admitted prediction token sequence.
pub enum PredictionTokenPart<'a, T> {
    /// Actual token payload retained by the admitted prepared input.
    Tokens(&'a T),
    /// Architecture-declared token repeated for this admitted decoder extent.
    Repeated {
        /// Exact placeholder token, including the full unsigned token domain.
        token: u32,
        /// Exact number of decoder positions occupied by this part.
        positions: u64,
    },
}

pub(crate) fn visit<T: Tensor, P>(
    input: PreparedCompositeInput<'_, T, P>,
    placeholder: impl Fn(&P) -> Option<(u32, u64)>,
    visitor: &mut dyn FnMut(PredictionTokenPart<'_, T>) -> Result<(), eredu_nn::Error>,
) -> Result<(), eredu_nn::Error> {
    let metadata = Metadata::new(input.metadata());
    let admitted = input.admitted().ordinary().ok_or_else(|| metadata.error(format_args!(
        "generic placeholder conversion requires ordinary family admission")))?;
    for (part, plan) in input.prepared().parts().iter().zip(admitted.parts()) {
        let part = match placeholder(plan) {
            Some((token, positions)) => PredictionTokenPart::Repeated { token, positions },
            None => match (part.modality(), part.payload()) {
                (eredu_core::InputModality::Text, eredu_runtime::PreparedInputPayload::TokenIds(tokens)) =>
                    PredictionTokenPart::Tokens(tokens),
                _ => return Err(metadata.error(format_args!(
                    "composite prediction input has no declared token identity"))),
            },
        };
        visitor(part)?;
    }
    Ok(())
}

pub(crate) fn materialize<T: Tensor>(part: PredictionTokenPart<'_, T>, context: &T::Context,
    metadata: Metadata<'_>) -> Result<T, eredu_nn::Error> {
    metadata.controls::<(T, [i32; 2], PredictionTokenPart<'_, T>)>()?;
    match part {
        PredictionTokenPart::Tokens(value) => Ok(value.clone()),
        PredictionTokenPart::Repeated { token, positions } => T::full_u32(token,
            &[1, i32::try_from(positions).map_err(|cause| metadata.source(cause))?], context),
    }
}
