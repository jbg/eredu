//! Actual normalization/matching selection, with one source-bound NFC workspace.
use super::*;
use crate::{
    normalizers::NormalizerWrapper,
    tokenizer::added_vocabulary::IdentityMatching,
};
use unicode_normalization_alignments::workspace as nfc;

pub(super) enum Plan<'a> {
    Identity {
        source: IdentityMatching<'a>,
        bytes: usize,
    },
    Nfc {
        added: &'a super::super::AddedVocabulary,
        plan: nfc::Plan<'a>,
    },
}
impl<'a> Plan<'a> {
    pub(super) fn inspect(
        source: &'a Tokenizer,
        input: &'a str,
    ) -> Result<Self, EncodeIdsError> {
        match source.normalizer.as_ref() {
            None => Ok(Self::Identity {
                source: IdentityMatching::checked(
                    &source.added_vocabulary,
                    None,
                )
                .ok_or(EncodeIdsError::AddedProfile)?,
                bytes: input.len(),
            }),
            Some(NormalizerWrapper::NFC(_))
                if source.added_vocabulary.raw_packed_literals() =>
            {
                Ok(Self::Nfc {
                    added: &source.added_vocabulary,
                    plan: nfc::Plan::new(input)
                        .map_err(|_| EncodeIdsError::Overflow)?,
                })
            }
            _ => Err(EncodeIdsError::PipelineProfile),
        }
    }
    pub(super) fn text_bound(&self) -> usize {
        match self {
            Self::Identity { bytes, .. } => *bytes,
            Self::Nfc { plan, .. } => plan.requirements().text_capacity(),
        }
    }
    pub(super) fn capacities(&self) -> [usize; 3] {
        match self {
            Self::Identity { .. } => [0; 3],
            Self::Nfc { plan, .. } => {
                let r = plan.requirements();
                [r.scalar_capacity(), r.scalar_capacity(), r.text_capacity()]
            }
        }
    }
    pub(super) fn buffer_bytes(&self) -> usize {
        match self {
            Self::Identity { .. } => 0,
            Self::Nfc { plan, .. } => plan.requirements().buffer_bytes(),
        }
    }
    pub(super) fn control_bytes(&self) -> Result<usize, EncodeIdsError> {
        let selected = match self {
            Self::Identity { source, .. } => {
                source.control_bytes::<EncodeIdsError>()
            }
            Self::Nfc { added, plan } => {
                added.raw_literal_control_bytes().and_then(|n| {
                    n.checked_add(plan.requirements().control_bytes())
                })
            }
        }
        .ok_or(EncodeIdsError::Overflow)?;
        [
            selected,
            size_of::<Self>(),
            size_of::<Result<Self, EncodeIdsError>>(),
            size_of::<Runtime<'_>>(),
            size_of::<Result<Runtime<'_>, EncodeIdsError>>(),
            size_of::<Option<nfc::Retired>>(),
            size_of::<&mut nfc::Workspace<'_>>(),
            size_of::<&mut InputEncoder<'_, '_>>(),
            size_of::<Result<(), EncodeIdsError>>(),
        ]
        .iter()
        .try_fold(0usize, |n, &x| n.checked_add(x))
        .ok_or(EncodeIdsError::Overflow)
    }
    pub(super) fn prepare(self) -> Result<Runtime<'a>, EncodeIdsError> {
        match self {
            Self::Identity { source, .. } => Ok(Runtime::Identity(source)),
            Self::Nfc { added, plan } => Ok(Runtime::Nfc {
                added,
                workspace: plan
                    .prepare()
                    .map_err(EncodeIdsError::NormalizationPreparation)?,
            }),
        }
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    pub(super) fn fail(
        self,
        buffer: NormalizationBuffer,
    ) -> Result<Self, EncodeIdsError> {
        match self {
            Self::Nfc { added, plan } => Ok(Self::Nfc {
                added,
                plan: plan.fail_reservation(buffer),
            }),
            _ => Err(EncodeIdsError::PipelineProfile),
        }
    }
}
pub(super) enum Runtime<'a> {
    Identity(IdentityMatching<'a>),
    Nfc {
        added: &'a super::super::AddedVocabulary,
        workspace: nfc::Workspace<'a>,
    },
}
impl Runtime<'_> {
    pub(super) fn encode(
        &mut self,
        encoder: &mut InputEncoder<'_, '_>,
    ) -> Result<(), EncodeIdsError> {
        let text = encoder.input;
        match self {
            Self::Identity(source) => {
                source.visit(text, |id, offsets| encoder.span(id, offsets))
            }
            Self::Nfc { added, workspace } => {
                added.visit_matches(text, false, |id, (start, end)| {
                    if let Some(id) = id {
                        encoder.output.ids.push(id);
                        Ok(())
                    } else {
                        let normalized = workspace
                            .normalize(start..end)
                            .map_err(EncodeIdsError::NormalizationRange)?;
                        encoder.regex.encode_span(
                            encoder.output,
                            encoder.model,
                            normalized,
                        )
                    }
                })
            }
        }
    }
    pub(super) fn retire(self) -> Option<nfc::Retired> {
        match self {
            Self::Identity(_) => None,
            Self::Nfc { workspace, .. } => Some(workspace.retire()),
        }
    }
}
