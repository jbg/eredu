//! Borrowed views of legacy descriptors and original host slots for one worker.
use super::*;
use crate::media_plan::qwen::{MediaSemanticError, ShapeRef};
use eredu_runtime::input::host::{HostInputPartView, HostTensorValues, PreparedHostPart};

#[derive(Clone, Copy)]
pub(super) struct Shape<'a>(ShapeRef<'a>);
impl Shape<'_> {
    pub(super) fn len(self) -> usize { self.0.len() }
    pub(super) fn at(self, axis: usize) -> u64 { self.0.get(axis).expect("validated shape axis") }
    pub(super) fn matches(self, expected: &[u64]) -> bool {
        self.len() == expected.len() && expected.iter().enumerate().all(|(axis, value)| self.0.get(axis) == Some(*value))
    }
    fn batch_one<D: Destination>(self, rank: usize, name: impl std::fmt::Display, architecture: &str, destination: D) -> Result<u64, D::Error> {
        if self.len() != rank || self.0.get(0) != Some(1) || self.0.get(1).unwrap_or(0) == 0 {
            return Err(destination.unsupported(architecture, format_args!("prepared {name} must be batch-one with rank {rank}, got {self:?}")));
        }
        Ok(self.at(1))
    }
}
impl std::fmt::Debug for Shape<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            ShapeRef::Host(values) => values.fmt(f),
            ShapeRef::Legacy(values) => values.fmt(f),
        }
    }
}
#[derive(Clone, Copy)]
pub(super) struct Values<'a, T> {
    pub(super) shape: Shape<'a>,
    pub(super) values: &'a [T],
}
impl<'a, T> Values<'a, T> {
    pub(super) fn legacy(input: &'a MetadataValues<T>) -> Self {
        Self { shape: Shape(ShapeRef::Legacy(&input.shape)), values: &input.values }
    }
}
#[derive(Clone, Copy)]
pub(in crate::media_plan) struct Input<'a> {
    pub(super) modality: InputModality,
    pub(super) kind: InputPayloadKind,
    pub(super) shape: Shape<'a>,
    pub(super) patch_extent: Option<[usize; 3]>,
    pub(super) audio_valid_frames: Option<usize>,
    pub(super) patch_grid: Option<Values<'a, i32>>,
    pub(super) patch_positions: Option<Values<'a, i32>>,
    pub(super) audio_mask: Option<Values<'a, bool>>,
}
impl<'a> Input<'a> {
    pub(super) fn legacy(input: &'a MediaAdmissionInput) -> Self {
        fn values<T>(input: &MetadataValues<T>) -> Values<'_, T> {
            Values { shape: Shape(ShapeRef::Legacy(&input.shape)), values: &input.values }
        }
        Self {
            modality: input.descriptor.modality(), kind: input.descriptor.payload_kind(),
            shape: Shape(ShapeRef::Legacy(&input.payload_shape)),
            patch_extent: input.patch_extent(), audio_valid_frames: input.audio_valid_frames(),
            patch_grid: input.patch_grid.as_ref().map(values),
            patch_positions: input.patch_positions.as_ref().map(values),
            audio_mask: input.audio_mask.as_ref().map(values),
        }
    }
    pub(in crate::media_plan) fn original(part: &PreparedHostPart<'a>) -> Result<Self, MediaSemanticError> {
        let integers = |key| -> Result<Option<Values<'a, i32>>, MediaSemanticError> {
            part.metadata_view(key).map(|(_, value)| {
                let HostTensorValues::I32(values) = value.values else {
                    return Err(MediaSemanticError::input("Gemma original integer metadata type"));
                };
                Ok(Values { shape: Shape(ShapeRef::Host(value.shape)), values })
            }).transpose()
        };
        let audio_mask = part.metadata_view(InputMetadataKey::AudioMask).map(|(_, value)| {
            let HostTensorValues::Bool(values) = value.values else {
                return Err(MediaSemanticError::input("Gemma original Boolean metadata type"));
            };
            Ok(Values { shape: Shape(ShapeRef::Host(value.shape)), values })
        }).transpose()?;
        Ok(Self {
            modality: part.modality(), kind: part.kind(), shape: Shape(ShapeRef::Host(part.payload_view().shape)),
            patch_extent: part.extents().iter().find_map(|extent| match extent {
                InputExtent::PatchGrid { time, height, width } => Some([*time, *height, *width]), _ => None,
            }),
            audio_valid_frames: part.extents().iter().find_map(|extent| match extent {
                InputExtent::AudioValidFrames(frames) => Some(*frames), _ => None,
            }),
            patch_grid: integers(InputMetadataKey::PatchGrid)?,
            patch_positions: integers(InputMetadataKey::PatchPositions)?, audio_mask,
        })
    }
    pub(super) fn projected<D: Destination>(self, args: &FamilyConfig, destination: D) -> Option<Result<Gemma4InputPartPlan, D::Error>> {
        let shape = self.shape;
        match (self.modality, self.kind) {
            (InputModality::Text, InputPayloadKind::TokenIds) => Some(
                shape.batch_one(2, "Gemma text token IDs", &args.model_type, destination)
                    .map(|positions| Gemma4InputPartPlan::TextTokens { positions }),
            ),
            (modality, InputPayloadKind::Embeddings) => Some((|| {
                let positions = shape.batch_one(3, format_args!("Gemma {} embeddings", modality.as_str()), &args.model_type, destination)?;
                let hidden = destination.positive(args.text.hidden_size, "Gemma text hidden size")?;
                if shape.at(2) != hidden {
                    return Err(destination.unsupported(&args.model_type, format_args!("prepared Gemma {} embeddings must have hidden width {hidden}, got {shape:?}", modality.as_str())));
                }
                Ok(Gemma4InputPartPlan::Projected { modality, placeholder_token_id: placeholder(args, modality, destination)?, positions })
            })()),
            _ => None,
        }
    }
}

/// Original preflight has no allocation authority. Its failure is a fixed
/// semantic cause; the same worker emits the full paid diagnostic in admission.
#[derive(Clone, Copy)]
pub(in crate::media_plan) struct SourceDestination;
impl Destination for SourceDestination {
    type Error = MediaSemanticError;
    fn unsupported(self, _: &str, _: impl std::fmt::Display) -> Self::Error {
        MediaSemanticError::input("Gemma original source violates shared input admission")
    }
    fn configuration(self, field: &'static str, _: std::fmt::Arguments<'_>) -> Self::Error {
        MediaSemanticError::input(field)
    }
    fn overflow(self, operation: &'static str) -> Self::Error { MediaSemanticError::overflow(operation) }
}
