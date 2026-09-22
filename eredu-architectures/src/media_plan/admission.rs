//! One prepared-input validation path with ordinary or counted destinations.
use super::*;
use crate::decoder::identity::Metadata;
use eredu_nn::{workspace::WorkspaceContext, Error};
pub(crate) mod gemma;
pub(super) mod inkling;
pub(in crate::media_plan) mod muse;
pub(crate) use inkling::admit as inkling;

pub(super) fn inspect_shared<T, E>(
    input: &PreparedInputPart<T>,
    descriptor: impl FnOnce(&PreparedInputPart<T>) -> Result<InputPartDescriptor, E>,
    shape: impl Fn(&InputTensorIdentity) -> Result<Vec<u64>, E>,
    i32_values: impl Fn(&T) -> Result<Vec<i32>, E>,
    bool_values: impl Fn(&T) -> Result<Vec<bool>, E>,
) -> Result<MediaAdmissionInput, E> {
    let descriptor = descriptor(input)?;
    let inspect_metadata = descriptor.payload_kind() == InputPayloadKind::Tensor
        && descriptor.modality() != InputModality::Text;
    let metadata_shape = |key| descriptor.metadata_value(key).map(&shape).transpose();
    let i32_metadata = |key| -> Result<Option<MetadataValues<i32>>, E> {
        if !inspect_metadata {
            return Ok(None);
        }
        input
            .metadata_value(key)
            .map(|tensor| {
                Ok(MetadataValues {
                    shape: metadata_shape(key)?.expect("runtime metadata has a descriptor"),
                    values: i32_values(tensor)?,
                })
            })
            .transpose()
    };
    let bool_metadata = |key| -> Result<Option<MetadataValues<bool>>, E> {
        if !inspect_metadata {
            return Ok(None);
        }
        input
            .metadata_value(key)
            .map(|tensor| {
                Ok(MetadataValues {
                    shape: metadata_shape(key)?.expect("runtime metadata has a descriptor"),
                    values: bool_values(tensor)?,
                })
            })
            .transpose()
    };
    let payload_shape = shape(descriptor.payload())?;
    let patch_grid = i32_metadata(InputMetadataKey::PatchGrid)?;
    let patch_positions = i32_metadata(InputMetadataKey::PatchPositions)?;
    let audio_mask = bool_metadata(InputMetadataKey::AudioMask)?;
    Ok(MediaAdmissionInput {
        descriptor,
        payload_shape,
        patch_grid,
        patch_positions,
        audio_mask,
    })
}

pub(super) enum Rejection {
    Identity,
    Overflow,
    Empty,
}
impl Rejection {
    pub(super) fn ordinary(self) -> CapabilityError {
        match self {
            Self::Identity => CapabilityError::Observation(
                "prepared-input tensor identity changed before architecture admission".into(),
            ),
            Self::Overflow => CapabilityError::ArithmeticOverflow {
                operation: "composite decoder-position total",
            },
            Self::Empty => CapabilityError::UnsupportedInput {
                architecture: "replicated composite".into(),
                reason: "prepared input occupies no decoder positions".into(),
            },
        }
    }
    fn metadata(self, metadata: Metadata<'_>) -> Error {
        let cause = (|| {
            Ok::<_, Error>(match self {
                Self::Identity => CapabilityError::Observation(metadata.text(
                    "prepared-input tensor identity changed before architecture admission",
                )?),
                Self::Overflow => CapabilityError::ArithmeticOverflow {
                    operation: "composite decoder-position total",
                },
                Self::Empty => CapabilityError::UnsupportedInput {
                    architecture: metadata.text("replicated composite")?,
                    reason: metadata.text("prepared input occupies no decoder positions")?,
                },
            })
        })();
        match cause {
            Ok(cause) => metadata.source(cause),
            Err(cause) => cause,
        }
    }
}

pub(super) fn complete_shared<T, P: CompositePartPlan, E>(
    input: &PreparedModelInput<T>,
    identity: PreparedInputIdentity,
    mut admit_part: impl FnMut(&PreparedInputPart<T>) -> Result<P, E>,
    vector: impl FnOnce(usize) -> Result<Vec<P>, E>,
    error: impl Fn(Rejection) -> E,
) -> Result<AdmittedCompositeInput<P>, E> {
    if &identity != input.identity() {
        return Err(error(Rejection::Identity));
    }
    let mut active_modalities = InputModalities {
        text: false,
        image: false,
        audio: false,
        video: false,
    };
    let mut decoder_positions = 0u64;
    let mut parts = vector(input.len())?;
    for part in input.parts() {
        let plan = admit_part(part)?;
        decoder_positions = decoder_positions
            .checked_add(plan.decoder_positions())
            .ok_or_else(|| error(Rejection::Overflow))?;
        match part.modality() {
            InputModality::Text => active_modalities.text = true,
            InputModality::Image => active_modalities.image = true,
            InputModality::Audio => active_modalities.audio = true,
            InputModality::Video => active_modalities.video = true,
            _ => {}
        }
        parts.push(plan);
    }
    if decoder_positions == 0 {
        return Err(error(Rejection::Empty));
    }
    Ok(AdmittedCompositeInput {
        identity,
        parts,
        decoder_positions,
        active_modalities,
    })
}

pub(super) fn vl_plan<E>(
    view: qwen::QwenPartRef<'_>,
    grid: impl FnOnce(&[[i32; 3]]) -> Result<Vec<(i32, i32, i32)>, E>,
) -> Result<QwenVlInputPartPlan, E> {
    Ok(match view.role {
        qwen::QwenPartRole::Tokens => QwenVlInputPartPlan::TextTokens {
            positions: view.positions,
        },
        qwen::QwenPartRole::Projected => QwenVlInputPartPlan::ProjectedText {
            positions: view.positions,
        },
        qwen::QwenPartRole::Encoded => QwenVlInputPartPlan::Media {
            ingress: QwenVisionIngressPlan {
                placeholder_token_id: view.placeholder,
                placeholder_count: view.positions,
                patch_grid: grid(view.grid)?,
            },
            shape: MediaShapePlan {
                decoder_positions: view.positions,
                execution_workspace_scalars: view.workspace_scalars,
            },
        },
    })
}
pub(super) fn hybrid_plan<E>(
    view: qwen::QwenPartRef<'_>,
    grid: impl FnOnce(&[[i32; 3]]) -> Result<Vec<(i32, i32, i32)>, E>,
) -> Result<QwenHybridInputPartPlan, E> {
    Ok(match view.role {
        qwen::QwenPartRole::Tokens => QwenHybridInputPartPlan::TextTokens {
            positions: view.positions,
        },
        qwen::QwenPartRole::Projected => QwenHybridInputPartPlan::Projected {
            modality: view.modality,
            positions: view.positions,
        },
        qwen::QwenPartRole::Encoded => QwenHybridInputPartPlan::Media {
            ingress: QwenVisionIngressPlan {
                placeholder_token_id: view.placeholder,
                placeholder_count: view.positions,
                patch_grid: grid(view.grid)?,
            },
            shape: MediaShapePlan {
                decoder_positions: view.positions,
                execution_workspace_scalars: view.workspace_scalars,
            },
        },
    })
}

fn inspect<T>(
    part: &PreparedInputPart<T>,
    inspector: &impl PreparedInputInspector<T>,
    context: &WorkspaceContext,
) -> Result<MediaAdmissionInput, Error> {
    let metadata = Metadata::new(Some(context));
    metadata.controls::<(
        MediaAdmissionInput,
        InputPartDescriptor,
        Vec<u64>,
        MetadataValues<i32>,
        MetadataValues<bool>,
    )>()?;
    inspect_shared(
        part,
        |part| {
            part.descriptor_with_metadata(context, |tensor| {
                inspector.identity_with_metadata(tensor, context)
            })
        },
        |identity| {
            let mut shape = metadata.vector(identity.shape().len())?;
            for &dimension in identity.shape() {
                shape.push(u64::try_from(dimension).map_err(|_| {
                    metadata.source(CapabilityError::ArithmeticOverflow {
                        operation: "prepared-input tensor dimension",
                    })
                })?);
            }
            Ok(shape)
        },
        |tensor| inspector.i32_values_with_metadata(tensor, context),
        |tensor| inspector.bool_values_with_metadata(tensor, context),
    )
}
fn complete<T, P: CompositePartPlan>(
    input: &PreparedModelInput<T>,
    inspector: &impl PreparedInputInspector<T>,
    context: &WorkspaceContext,
    admit_part: impl FnMut(&PreparedInputPart<T>) -> Result<P, Error>,
) -> Result<AdmittedCompositeInput<P>, Error> {
    let metadata = Metadata::new(Some(context));
    metadata.controls::<(
        AdmittedCompositeInput<P>,
        PreparedInputIdentity,
        InputPartDescriptor,
        Vec<P>,
        Vec<InputPartDescriptor>,
        Rejection,
    )>()?;
    let mut descriptors = metadata.vector(input.len())?;
    for part in input.parts() {
        descriptors.push(part.descriptor_with_metadata(context, |tensor| {
            inspector.identity_with_metadata(tensor, context)
        })?);
    }
    let identity =
        PreparedInputIdentity::new(descriptors).map_err(|cause| metadata.source(cause))?;
    complete_shared(
        input,
        identity,
        admit_part,
        |count| metadata.vector(count),
        |cause| cause.metadata(metadata),
    )
}
fn grid(rows: &[[i32; 3]], metadata: Metadata<'_>) -> Result<Vec<(i32, i32, i32)>, Error> {
    let mut result = metadata.vector(rows.len())?;
    result.extend(rows.iter().map(|row| (row[0], row[1], row[2])));
    Ok(result)
}
fn semantic(cause: MediaSemanticError, architecture: &str, metadata: Metadata<'_>) -> Error {
    match cause.legacy_with(architecture, |text| metadata.format(text)) {
        Ok(cause) => metadata.source(cause),
        Err(cause) => cause,
    }
}
pub(super) fn vl_policy(args: &QwenVlModelArgs) -> qwen::QwenPolicy<'_> {
    qwen::QwenPolicy {
        hidden: args.text.hidden_size,
        vision: Some(&args.vision),
        image: Some(args.image_token_id),
        video: Some(args.video_token_id),
        projected_media: false,
    }
}
pub(super) fn hybrid_policy<'a>(
    text: &HybridConfig,
    vision: Option<&'a VisionConfig>,
    image: Option<i32>,
    video: Option<i32>,
) -> qwen::QwenPolicy<'a> {
    qwen::QwenPolicy {
        hidden: text.hidden_size,
        vision,
        image,
        video,
        projected_media: true,
    }
}
pub(crate) fn qwen_vl<T>(
    args: &QwenVlModelArgs,
    input: &PreparedModelInput<T>,
    inspector: &impl PreparedInputInspector<T>,
    context: &WorkspaceContext,
) -> Result<AdmittedCompositeInput<QwenVlInputPartPlan>, Error> {
    let metadata = Metadata::new(Some(context));
    complete(input, inspector, context, |part| {
        let inspected = inspect(part, inspector, context)?;
        let view = qwen::qwen_part(vl_policy(args), inspected.qwen_ref())
            .map_err(|cause| semantic(cause, &args.model_type, metadata))?;
        vl_plan(view, |rows| grid(rows, metadata))
    })
}
pub(crate) fn qwen_hybrid<T>(
    args: &ParsedHybridConfig,
    input: &PreparedModelInput<T>,
    inspector: &impl PreparedInputInspector<T>,
    context: &WorkspaceContext,
) -> Result<AdmittedCompositeInput<QwenHybridInputPartPlan>, Error> {
    let metadata = Metadata::new(Some(context));
    complete(input, inspector, context, |part| {
        let inspected = inspect(part, inspector, context)?;
        let view = qwen::qwen_part(
            hybrid_policy(
                &args.text,
                args.vision.as_ref(),
                args.image_token_id,
                args.video_token_id,
            ),
            inspected.qwen_ref(),
        )
        .map_err(|cause| semantic(cause, &args.text.model_type, metadata))?;
        hybrid_plan(view, |rows| grid(rows, metadata))
    })
}
