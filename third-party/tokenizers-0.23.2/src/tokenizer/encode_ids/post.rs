//! Checked single-input projection through the same ordered template visitor.
use super::*;
use crate::processors::{
    template::{compiled::TemplateProcessing, visit::PieceView},
    PostProcessorWrapper,
};

pub(super) struct PostPlan<'a> {
    template: Option<&'a TemplateProcessing>,
    add_special_tokens: bool,
    special_ids: usize,
}
impl<'a> PostPlan<'a> {
    pub(super) fn inspect(
        source: &'a Tokenizer,
        add_special_tokens: bool,
    ) -> Result<Self, EncodeIdsError> {
        let mut result = Self {
            template: None,
            add_special_tokens,
            special_ids: 0,
        };
        match &source.post_processor {
            None => {}
            Some(PostProcessorWrapper::Sequence(sequence)) => {
                for item in sequence.as_ref() {
                    result.accept(item)?;
                }
            }
            Some(item) => result.accept(item)?,
        }
        Ok(result)
    }
    fn accept(&mut self, item: &'a PostProcessorWrapper) -> Result<(), EncodeIdsError> {
        match item {
            PostProcessorWrapper::ByteLevel(byte) if !byte.trim_offsets => Ok(()),
            PostProcessorWrapper::Template(template) if self.template.is_none() => {
                self.special_ids = template
                    .single_ids(self.add_special_tokens)
                    .ok_or(EncodeIdsError::PipelineProfile)?;
                self.template = Some(template);
                Ok(())
            }
            _ => Err(EncodeIdsError::PipelineProfile),
        }
    }
    pub(super) fn special_ids(&self) -> usize {
        self.special_ids
    }
    pub(super) fn control_bytes() -> Option<usize> {
        [
            size_of::<Self>(),
            size_of::<Result<Self, EncodeIdsError>>(),
            size_of::<std::slice::Iter<'_, PostProcessorWrapper>>(),
            size_of::<Result<(), EncodeIdsError>>(),
            size_of::<Option<&[u32]>>(),
            TemplateProcessing::visit_control_bytes()?,
        ]
        .iter()
        .try_fold(0usize, |n, v| n.checked_add(*v))
    }
    pub(super) fn encode(
        self,
        encoder: &mut PostInputEncoder<'_, '_>,
    ) -> Result<(), EncodeIdsError> {
        if let Some(template) = self.template {
            template.visit_single(self.add_special_tokens, |piece| match piece {
                PieceView::Sequence { index, .. } => {
                    debug_assert_eq!(index, 0);
                    encoder.piece(None)
                }
                PieceView::Special { ids, .. } => encoder.piece(Some(ids)),
            })
        } else {
            encoder.piece(None)
        }
    }
}
