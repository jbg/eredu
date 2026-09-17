//! Gemma's shared prepared input policy with counted destinations.
pub(in crate::media_plan) mod raw;
pub(in crate::media_plan) mod view;
use super::*;
use super::inkling::Destination;
use crate::gemma4::FamilyConfig;

pub(crate) fn admit<T, I: PreparedInputInspector<T>>(
    args: &FamilyConfig,
    input: &PreparedModelInput<T>,
    inspector: &I,
    context: &WorkspaceContext,
) -> Result<AdmittedCompositeInput<Gemma4InputPartPlan>, Error> {
    let metadata = Metadata::new(Some(context));
    metadata.controls::<(
        &FamilyConfig, &PreparedModelInput<T>, &I, &WorkspaceContext,
        (&FamilyConfig, &I, &WorkspaceContext, Metadata<'_>),
        AdmittedCompositeInput<Gemma4InputPartPlan>,
        Result<AdmittedCompositeInput<Gemma4InputPartPlan>, Error>,
    )>()?;
    complete(input, inspector, context, |input| {
        metadata.controls::<(
            MediaAdmissionInput, Gemma4InputPartPlan,
            Result<Gemma4InputPartPlan, Error>, Option<Result<Gemma4InputPartPlan, Error>>,
            &FamilyConfig, &MediaAdmissionInput, &InputPartDescriptor,
            InputModality, InputPayloadKind, &[u64], u64, u32,
            std::fmt::Arguments<'_>, Result<u64, Error>, Result<u32, Error>,
        )>()?;
        // The inspector must supply the exact evaluated metadata source through
        // its counted companions; their default refusal remains authoritative.
        let inspected = inspect(input, inspector, context)?;
        raw::part(args, &inspected, metadata)
    })
}

pub(in crate::media_plan) fn placeholder<D: Destination>(
    args: &FamilyConfig,
    modality: InputModality,
    destination: D,
) -> Result<u32, D::Error> {
    let token = match modality {
        InputModality::Text => Some(args.text.pad_token_id),
        InputModality::Image => args.image_token_id,
        InputModality::Video => args.video_token_id,
        InputModality::Audio => args.audio_token_id,
        _ => None,
    };
    token.and_then(|token| u32::try_from(token).ok()).ok_or_else(|| {
        destination.unsupported(
            &args.model_type,
            format_args!("Gemma 4 has no valid {} placeholder", modality.as_str()),
        )
    })
}


#[cfg(test)]
mod tests;
