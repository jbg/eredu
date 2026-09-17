//! Fixed Gemma ingress geometry projected from already compiled source records.
use super::*;
use crate::media_plan::{Gemma4InputPartPlan as Part, MediaShapePlan};

impl BoundPreparedMediaSemantics {
    pub(crate) fn is_gemma(&self) -> bool { self.1 == SemanticKind::Gemma }

    /// This copies only fixed scalar fields. The compiler has already checked
    /// the exact payload, extent and metadata values through ordinary admission.
    pub(crate) fn gemma_part(&self, index: usize) -> Part {
        assert!(self.is_gemma(), "typed Gemma admission source");
        let record = &self.records()[index];
        let positions = record.end - record.start;
        match record.role {
            CompositeSemanticRole::Tokens => Part::TextTokens { positions },
            CompositeSemanticRole::Projected => Part::Projected {
                modality: record.modality, placeholder_token_id: record.placeholder, positions,
            },
            CompositeSemanticRole::Encoded => {
                let source = self.source().part(record.source_part).expect("compiled source part");
                let padded = i32::try_from(source.payload_view().shape[1]).expect("compiled padded extent");
                let shape = MediaShapePlan {
                    decoder_positions: positions, execution_workspace_scalars: record.workspace_scalars,
                };
                match record.modality {
                    InputModality::Image | InputModality::Video => {
                        let (slot, grid) = source.metadata_view(InputMetadataKey::PatchGrid).expect("compiled grid");
                        assert_eq!(Some(slot), record.grid_slot);
                        let HostTensorValues::I32(values) = grid.values else { unreachable!("compiled I32 grid") };
                        let &[1, height, width] = values else { unreachable!("compiled frame grid") };
                        Part::Vision {
                            placeholder_token_id: record.placeholder,
                            ingress: crate::gemma4::VisionIngressPartPlan {
                                padded_patches: padded,
                                valid_patches: height.checked_mul(width).expect("compiled patch extent"),
                                decoder_positions: i32::try_from(positions).expect("compiled decoder extent"),
                                grid_height: height, grid_width: width,
                            }, shape,
                        }
                    }
                    InputModality::Audio => {
                        let valid = source.extents().iter().find_map(|extent| match extent {
                            eredu_core::InputExtent::AudioValidFrames(value) => Some(*value), _ => None,
                        }).expect("compiled audio extent");
                        let ingress = crate::gemma4::AudioIngressPartPlan::with_diagnostic(
                            i32::try_from(valid).expect("compiled audio extent"), padded, |_| (),
                        ).expect("compiled audio subsampling");
                        assert_eq!(ingress.decoder_positions as u64, positions);
                        Part::Audio { placeholder_token_id: record.placeholder, ingress, shape }
                    }
                    _ => unreachable!("compiled Gemma encoder modality"),
                }
            }
        }
    }
}
