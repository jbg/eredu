//! Shared Inkling input semantics with caller-owned diagnostic destinations.
use super::*;
use std::fmt;

pub(in crate::media_plan) trait Destination: Copy {
    type Error;
    fn controls<T>(self) -> Result<(), Self::Error> { Ok(()) }
    fn unsupported(self, architecture: &str, reason: impl fmt::Display) -> Self::Error;
    fn configuration(self, field: &'static str, detail: fmt::Arguments<'_>) -> Self::Error;
    fn overflow(self, operation: &'static str) -> Self::Error;
    fn positive(self, value: i32, field: &'static str) -> Result<u64, Self::Error> {
        u64::try_from(value).map_err(|_| {
            self.configuration(
                field,
                format_args!("expected a non-negative value, got {value}"),
            )
        })
    }
    fn checked_add(
        self,
        left: u64,
        right: u64,
        operation: &'static str,
    ) -> Result<u64, Self::Error> {
        left.checked_add(right)
            .ok_or_else(|| self.overflow(operation))
    }
    fn checked_mul(
        self,
        left: u64,
        right: u64,
        operation: &'static str,
    ) -> Result<u64, Self::Error> {
        left.checked_mul(right)
            .ok_or_else(|| self.overflow(operation))
    }
    fn batch_one_sequence(
        self,
        shape: &[u64],
        rank: usize,
        name: impl fmt::Display,
        architecture: &str,
    ) -> Result<u64, Self::Error> {
        if shape.len() != rank
            || shape.first() != Some(&1)
            || shape.get(1).copied().unwrap_or(0) == 0
        {
            return Err(self.unsupported(
                architecture,
                format_args!("prepared {name} must be batch-one with rank {rank}, got {shape:?}"),
            ));
        }
        Ok(shape[1])
    }
}
#[derive(Clone, Copy)]
pub(in crate::media_plan) struct Ordinary;
impl Destination for Ordinary {
    type Error = CapabilityError;
    fn unsupported(self, architecture: &str, reason: impl fmt::Display) -> CapabilityError {
        CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: reason.to_string(),
        }
    }
    fn configuration(self, field: &'static str, detail: fmt::Arguments<'_>) -> CapabilityError {
        CapabilityError::InvalidConfiguration {
            field,
            detail: detail.to_string(),
        }
    }
    fn overflow(self, operation: &'static str) -> CapabilityError {
        CapabilityError::ArithmeticOverflow { operation }
    }
}
impl Destination for Metadata<'_> {
    type Error = Error;
    fn controls<T>(self) -> Result<(), Error> { Metadata::controls::<T>(self) }
    fn unsupported(self, architecture: &str, reason: impl fmt::Display) -> Error {
        let result = (|| {
            self.controls::<(
                CapabilityError,
                Result<CapabilityError, Error>,
                &str,
                fmt::Arguments<'_>,
            )>()?;
            Ok::<_, Error>(CapabilityError::UnsupportedInput {
                architecture: self.text(architecture)?,
                reason: self.format(format_args!("{reason}"))?,
            })
        })();
        match result {
            Ok(cause) => self.source(cause),
            Err(cause) => cause,
        }
    }
    fn configuration(self, field: &'static str, detail: fmt::Arguments<'_>) -> Error {
        let result = (|| {
            self.controls::<(
                CapabilityError,
                Result<CapabilityError, Error>,
                &str,
                fmt::Arguments<'_>,
            )>()?;
            Ok::<_, Error>(CapabilityError::InvalidConfiguration {
                field,
                detail: self.format(detail)?,
            })
        })();
        match result {
            Ok(cause) => self.source(cause),
            Err(cause) => cause,
        }
    }
    fn overflow(self, operation: &'static str) -> Error {
        self.source(CapabilityError::ArithmeticOverflow { operation })
    }
}

pub(crate) fn admit<T, I: PreparedInputInspector<T>>(
    args: &crate::inkling::ModelArgs,
    input: &PreparedModelInput<T>,
    inspector: &I,
    context: &WorkspaceContext,
) -> Result<AdmittedCompositeInput<InklingInputPartPlan>, Error> {
    let metadata = Metadata::new(Some(context));
    metadata.controls::<(
        &crate::inkling::ModelArgs,
        &PreparedModelInput<T>,
        &I,
        &WorkspaceContext,
        (&crate::inkling::ModelArgs, &I, &WorkspaceContext, Metadata<'_>),
        AdmittedCompositeInput<InklingInputPartPlan>,
        Result<AdmittedCompositeInput<InklingInputPartPlan>, Error>,
    )>()?;
    complete(input, inspector, context, |input| {
        metadata.controls::<(
            MediaAdmissionInput,
            InklingInputPartPlan,
            MediaShapePlan,
            InklingIngressPlan,
            Result<InklingInputPartPlan, Error>,
            Result<MediaShapePlan, Error>,
            &crate::inkling::ModelArgs,
            &MediaAdmissionInput,
            &InputPartDescriptor,
            InputModality,
            InputPayloadKind,
            &[u64],
            [u64; 4],
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            usize,
            fmt::Arguments<'_>,
            Result<u64, Error>,
        )>()?;
        let inspected = inspect(input, inspector, context)?;
        part(args, &inspected, metadata)
    })
}

pub(in crate::media_plan) fn media<D: Destination>(
    args: &crate::inkling::ModelArgs,
    input: &MediaAdmissionInput,
    destination: D,
) -> Result<MediaShapePlan, D::Error> {
    match input.modality() {
        InputModality::Image => {
            let config = args.vision_config.as_ref().ok_or_else(|| {
                destination.unsupported(
                    &args.model_type,
                    "loaded Inkling model has no vision configuration",
                )
            })?;
            if input.payload_shape.len() != 5 || input.payload_shape[1..] != [2, 40, 40, 3] {
                return Err(destination.unsupported(
                    &args.model_type,
                    format_args!(
                        "Inkling image patches must be [patches, 2, 40, 40, 3], got {:?}",
                        input.payload_shape
                    ),
                ));
            }
            let patches = input.payload_shape[0];
            let text_hidden =
                destination.positive(config.text_hidden_size, "Inkling vision output size")?;
            let layer_outputs = [
                destination.checked_mul(
                    destination.checked_mul(
                        destination.checked_mul(patches, 2, "Inkling vision time")?,
                        8 * 8,
                        "Inkling vision grid",
                    )?,
                    128,
                    "Inkling vision layer 1",
                )?,
                destination.checked_mul(
                    destination.checked_mul(
                        destination.checked_mul(patches, 2, "Inkling vision time")?,
                        4 * 4,
                        "Inkling vision grid",
                    )?,
                    512,
                    "Inkling vision layer 2",
                )?,
                destination.checked_mul(
                    destination.checked_mul(patches, 2, "Inkling vision time")?,
                    4_800,
                    "Inkling vision layer 3",
                )?,
                destination.checked_mul(patches, text_hidden, "Inkling vision layer 4")?,
            ];
            let graph_scalars = layer_outputs.iter().try_fold(0u64, |total, value| {
                destination.checked_add(
                    total,
                    destination.checked_mul(12, *value, "Inkling vision layer workspace")?,
                    "Inkling vision graph workspace",
                )
            })?;
            Ok(MediaShapePlan {
                decoder_positions: patches,
                execution_workspace_scalars: graph_scalars,
            })
        }
        InputModality::Audio => {
            let config = args.audio_config.as_ref().ok_or_else(|| {
                destination.unsupported(
                    &args.model_type,
                    "loaded Inkling model has no audio configuration",
                )
            })?;
            let [1, padded_frames, payload_codebooks] = input.payload_shape.as_slice() else {
                return Err(destination.unsupported(
                    &args.model_type,
                    format_args!(
                        "Inkling audio tokens must be [1, frames, codebooks], got {:?}",
                        input.payload_shape
                    ),
                ));
            };
            let (padded_frames, payload_codebooks) = (*padded_frames, *payload_codebooks);
            let codebooks =
                destination.positive(config.num_codebooks, "Inkling audio codebooks")?;
            if payload_codebooks != codebooks {
                return Err(destination.unsupported(
                    &args.model_type,
                    format_args!(
                        "Inkling audio payload has {payload_codebooks} codebooks, expected {codebooks}"
                    ),
                ));
            }
            let frames = if let Some(mask) = &input.audio_mask {
                if mask.shape != [1, padded_frames]
                    || u64::try_from(mask.values.len()).ok() != Some(padded_frames)
                {
                    return Err(destination.unsupported(
                        &args.model_type,
                        format_args!(
                            "Inkling audio mask must be [1, {padded_frames}], got {:?}",
                            mask.shape
                        ),
                    ));
                }
                let frames = mask.values.iter().take_while(|value| **value).count();
                if mask.values[frames..].iter().any(|value| *value) {
                    return Err(destination.unsupported(
                        &args.model_type,
                        "Inkling audio mask must describe one valid prefix",
                    ));
                }
                u64::try_from(frames)
                    .map_err(|_| destination.overflow("Inkling valid audio frame count"))?
            } else {
                padded_frames
            };
            let hidden =
                destination.positive(config.text_hidden_size, "Inkling audio hidden size")?;
            let embedded = destination.checked_mul(
                destination.checked_mul(
                    padded_frames,
                    codebooks,
                    "Inkling audio frame codebooks",
                )?,
                hidden,
                "Inkling audio embedding elements",
            )?;
            let reduced =
                destination.checked_mul(padded_frames, hidden, "Inkling audio reduced elements")?;
            let graph_scalars = destination.checked_add(
                destination.checked_mul(4, embedded, "Inkling audio embedding workspace")?,
                destination.checked_mul(12, reduced, "Inkling audio reduction/norm workspace")?,
                "Inkling audio graph workspace",
            )?;
            Ok(MediaShapePlan {
                decoder_positions: frames,
                execution_workspace_scalars: graph_scalars,
            })
        }
        InputModality::Video => Err(destination.unsupported(
            &args.model_type,
            "video is not a supported Inkling modality",
        )),
        InputModality::Text => {
            Err(destination.unsupported(&args.model_type, "text is not an Inkling media modality"))
        }
        _ => Err(destination.unsupported(
            &args.model_type,
            "unknown modality is not supported by Inkling",
        )),
    }
}

pub(in crate::media_plan) fn part<D: Destination>(
    args: &crate::inkling::ModelArgs,
    inspected: &MediaAdmissionInput,
    destination: D,
) -> Result<InklingInputPartPlan, D::Error> {
    let descriptor = &inspected.descriptor;
    let modality = descriptor.modality();
    let payload = descriptor.payload_kind();
    let shape = &inspected.payload_shape;
    match (modality, payload) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(InklingInputPartPlan::TextTokens {
            positions: destination.batch_one_sequence(
                &shape,
                2,
                "Inkling text token IDs",
                &args.model_type,
            )?,
        }),
        (
            modality @ (InputModality::Image | InputModality::Audio),
            InputPayloadKind::Embeddings,
        ) => {
            let positions = destination.batch_one_sequence(
                &shape,
                3,
                format_args!("Inkling {} embeddings", modality.as_str()),
                &args.model_type,
            )?;
            let hidden =
                destination.positive(args.text_config.hidden_size, "Inkling text hidden size")?;
            if shape[2] != hidden {
                return Err(destination.unsupported(
                    &args.model_type,
                    format_args!(
                        "prepared Inkling {} embeddings must have hidden width {hidden}, got {shape:?}",
                        modality.as_str()
                    ),
                ));
            }
            let placeholder_token_id = match modality {
                InputModality::Image => args.image_token_id,
                InputModality::Audio => args.audio_token_id,
                InputModality::Text | InputModality::Video => unreachable!(),
                _ => {
                    return Err(destination
                        .unsupported(&args.model_type, "unknown projected Inkling modality"));
                }
            };
            Ok(InklingInputPartPlan::Projected {
                modality,
                placeholder_token_id,
                positions,
            })
        }
        (modality @ (InputModality::Image | InputModality::Audio), InputPayloadKind::Tensor) => {
            let shape = media(args, inspected, destination)?;
            let ingress = InklingIngressPlan {
                placeholder_token_id: if modality == InputModality::Image {
                    args.image_token_id
                } else {
                    args.audio_token_id
                },
                placeholder_count: shape.decoder_positions,
            };
            Ok(InklingInputPartPlan::Media {
                modality,
                ingress,
                shape,
            })
        }
        (modality, payload) => Err(destination.unsupported(
            &args.model_type,
            format_args!(
                "Inkling does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}
