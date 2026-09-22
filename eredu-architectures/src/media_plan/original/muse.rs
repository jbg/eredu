//! Borrowed Muse grid and scalar plans over the authenticated host source.
use super::*;
use crate::media_plan::admission::{
    gemma::view::{Input, SourceDestination},
    inkling::Destination,
};
use crate::media_plan::{MediaShapePlan, MuseGlimmerInputPartPlan as Part};
#[derive(Clone)]
pub(crate) enum MuseInputPartRef<'a> {
    TextTokens {
        positions: u64,
    },
    Vision {
        modality: InputModality,
        placeholder_token_id: u32,
        placeholder_count: u64,
        shape: MediaShapePlan,
        grid: GridRows<'a>,
    },
}
impl<'a> MuseInputPartRef<'a> {
    pub(crate) fn ordinary(part: &'a Part) -> Self {
        match part {
            Part::TextTokens { positions } => Self::TextTokens {
                positions: *positions,
            },
            Part::Vision {
                modality,
                ingress,
                shape,
            } => Self::Vision {
                modality: *modality,
                placeholder_token_id: ingress.placeholder_token_id,
                placeholder_count: ingress.placeholder_count,
                shape: shape.clone(),
                grid: GridRows::Tuples(&ingress.patch_grid),
            },
        }
    }
}
pub(super) fn semantic_part<'a>(
    args: &crate::muse_glimmer::DecoderConfig,
    part: eredu_runtime::input::host::PreparedHostPart<'a>,
) -> Result<SemanticPart<'a>, MediaSemanticError> {
    let input = Input::original(&part)?;
    let (role, positions, placeholder, workspace_scalars, grid) = match (input.modality, input.kind)
    {
        (InputModality::Text, eredu_core::InputPayloadKind::TokenIds) => (
            CompositeSemanticRole::Tokens,
            input.shape.batch_one(
                2,
                "Muse text token IDs",
                &args.model_type,
                SourceDestination,
            )?,
            0,
            0,
            &[][..],
        ),
        (
            modality @ (InputModality::Image | InputModality::Video),
            eredu_core::InputPayloadKind::Tensor,
        ) => {
            let shape = crate::media_plan::admission::muse::media(args, input, SourceDestination)?;
            let grid = input
                .patch_grid
                .expect("admitted Muse grid")
                .values
                .as_chunks::<3>()
                .0;
            (
                CompositeSemanticRole::Encoded,
                shape.decoder_positions,
                if modality == InputModality::Image {
                    args.image_token_id
                } else {
                    args.video_token_id
                },
                shape.execution_workspace_scalars,
                grid,
            )
        }
        _ => return Err(SourceDestination.unsupported(&args.model_type, "Muse payload kind")),
    };
    Ok(SemanticPart {
        role,
        modality: input.modality,
        positions,
        placeholder,
        grid,
        workspace_scalars,
    })
}
impl BoundPreparedMediaSemantics {
    pub(crate) fn is_muse(&self) -> bool {
        self.1 == SemanticKind::Muse
    }
    pub(crate) fn muse_part(&self, index: usize) -> MuseInputPartRef<'_> {
        assert!(self.is_muse(), "typed Muse admission source");
        let record = &self.records()[index];
        let positions = record.end - record.start;
        match record.role {
            CompositeSemanticRole::Tokens => MuseInputPartRef::TextTokens { positions },
            CompositeSemanticRole::Encoded => {
                let source = self
                    .source()
                    .part(record.source_part)
                    .expect("compiled Muse source");
                let (slot, grid) = source
                    .metadata_view(InputMetadataKey::PatchGrid)
                    .expect("compiled Muse grid");
                assert_eq!(Some(slot), record.grid_slot);
                let HostTensorValues::I32(values) = grid.values else {
                    unreachable!("compiled Muse I32 grid")
                };
                MuseInputPartRef::Vision {
                    modality: record.modality,
                    placeholder_token_id: record.placeholder,
                    placeholder_count: positions,
                    shape: MediaShapePlan {
                        decoder_positions: positions,
                        execution_workspace_scalars: record.workspace_scalars,
                    },
                    grid: GridRows::Arrays(values.as_chunks::<3>().0),
                }
            }
            CompositeSemanticRole::Projected => unreachable!("Muse admits raw media"),
        }
    }
}
