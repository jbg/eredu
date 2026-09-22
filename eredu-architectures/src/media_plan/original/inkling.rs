//! Fixed Inkling plans projected from authenticated source records.
use super::*;
use crate::media_plan::{InklingIngressPlan, InklingInputPartPlan as Part, MediaShapePlan};
impl BoundPreparedMediaSemantics {
    pub(crate) fn is_inkling(&self) -> bool {
        self.1 == SemanticKind::Inkling
    }
    pub(crate) fn inkling_part(&self, index: usize) -> Part {
        assert!(self.is_inkling(), "typed Inkling admission source");
        let record = &self.records()[index];
        let positions = record.end - record.start;
        match record.role {
            CompositeSemanticRole::Tokens => Part::TextTokens { positions },
            CompositeSemanticRole::Projected => Part::Projected {
                modality: record.modality,
                placeholder_token_id: record.placeholder,
                positions,
            },
            CompositeSemanticRole::Encoded => Part::Media {
                modality: record.modality,
                ingress: InklingIngressPlan {
                    placeholder_token_id: record.placeholder,
                    placeholder_count: positions,
                },
                shape: MediaShapePlan {
                    decoder_positions: positions,
                    execution_workspace_scalars: record.workspace_scalars,
                },
            },
        }
    }
}
