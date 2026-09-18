//! The selected model mechanism inside the single aggregate constructor.
use super::*;
use crate::models::{unigram::UnigramCompilePlan, wordlevel::WordLevelCompilePlan};
#[derive(Debug)]
pub(super) enum Plan<'a> {
    Bpe(BpeCompilePlan<'a>),
    WordLevel(WordLevelCompilePlan<'a>),
    Unigram(UnigramCompilePlan<'a>),
}
impl<'a> Plan<'a> {
    pub(super) fn prepare(bytes: &'a [u8]) -> Result<Self, Error> {
        let input = std::str::from_utf8(bytes).map_err(|e| err(K::InvalidUtf8, e.valid_up_to()))?;
        let mut fields = Reader::new(
            input,
            Span {
                start: 0,
                end: input.len(),
            },
        )
        .object()?;
        while let Some((key, value)) = fields.next()? {
            if key.is("type") && text_is(input, value, "Unigram")? {
                return Ok(Self::Unigram(UnigramCompilePlan::prepare_model_json(
                    bytes,
                )?));
            }
            if key.is("type") && text_is(input, value, "WordLevel")? {
                return Ok(Self::WordLevel(WordLevelCompilePlan::prepare_model_json(
                    bytes,
                )?));
            }
        }
        Ok(Self::Bpe(BpeCompilePlan::prepare_model_json(bytes)?))
    }
    pub(super) fn requirements(&self) -> Requirements {
        match self {
            Self::Bpe(p) => {
                let q = p.requirements();
                Requirements {
                    slots: q.vocabulary_slots(),
                    bytes: q.spelling_bytes(),
                    extent: q.id_extent(),
                    buffers: q.buffer_bytes(),
                    controls: q.control_bytes(),
                }
            }
            Self::WordLevel(p) => {
                let q = p.requirements();
                Requirements {
                    slots: q.vocabulary_slots(),
                    bytes: q.spelling_bytes(),
                    extent: q.id_extent(),
                    buffers: q.buffer_bytes(),
                    controls: q.control_bytes(),
                }
            }
            Self::Unigram(p) => {
                let q = p.requirements();
                Requirements {
                    slots: q.vocabulary_slots(),
                    bytes: q.spelling_bytes(),
                    extent: q.id_extent(),
                    buffers: q.buffer_bytes(),
                    controls: q.control_bytes(),
                }
            }
        }
    }
    pub(super) fn compile(self) -> Result<ModelWrapper, Cause> {
        match self {
            Self::Bpe(p) => p.compile().map(ModelWrapper::BPE).map_err(Cause::Model),
            Self::WordLevel(p) => p
                .compile()
                .map(ModelWrapper::WordLevel)
                .map_err(Cause::WordLevel),
            Self::Unigram(p) => p
                .compile()
                .map(ModelWrapper::Unigram)
                .map_err(Cause::Unigram),
        }
    }
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    pub(super) fn fail_reservation(self, stage: usize) -> Self {
        match self {
            Self::Bpe(p) => Self::Bpe(p.fail_reservation(stage)),
            Self::WordLevel(p) => Self::WordLevel(p.fail_reservation(stage)),
            Self::Unigram(p) => Self::Unigram(p.fail_reservation(stage)),
        }
    }
}
pub(super) struct Requirements {
    slots: usize,
    bytes: usize,
    extent: u64,
    buffers: usize,
    controls: usize,
}
impl Requirements {
    pub(super) fn vocabulary_slots(&self) -> usize {
        self.slots
    }
    pub(super) fn spelling_bytes(&self) -> usize {
        self.bytes
    }
    pub(super) fn id_extent(&self) -> u64 {
        self.extent
    }
    pub(super) fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    pub(super) fn control_bytes(&self) -> usize {
        self.controls + size_of::<Self>()
    }
}
