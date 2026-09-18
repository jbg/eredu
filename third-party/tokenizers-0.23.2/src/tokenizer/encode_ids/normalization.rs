//! Actual normalization/matching selection over the canonical source components.
use super::*;
use crate::tokenizer::normalization_source::pipeline;
use crate::tokenizer::{added_vocabulary::IdentityMatching, normalization_source::Shape};
use unicode_normalization_alignments::workspace as nfc;

pub(super) enum Plan<'a> {
    Pipeline {
        added: &'a super::super::AddedVocabulary,
        shape: Shape<'a>,
        bounds: pipeline::Bounds,
        total: usize,
        failure: Option<usize>,
    },
    Identity {
        source: IdentityMatching<'a>,
        bytes: usize,
    },
    Transform {
        added: &'a super::super::AddedVocabulary,
        shape: Shape<'a>,
        nfc: Option<nfc::Plan<'a>>,
        text: usize,
        total: usize,
    },
}
impl<'a> Plan<'a> {
    pub(super) fn inspect(
        source: TokenizerInput<'a>,
        input: &'a str,
    ) -> Result<Self, EncodeIdsError> {
        let shape = Shape::inspect(source.root.normalizer.as_ref())
            .ok_or(EncodeIdsError::PipelineProfile)?;
        let shape = if source.remove_input_prefixes {
            shape.prefix_free()
        } else {
            shape
        };
        if shape.is_identity() {
            return Ok(Self::Identity {
                source: IdentityMatching::checked(source.added, None)
                    .ok_or(EncodeIdsError::AddedProfile)?,
                bytes: input.len(),
            });
        }
        if shape.pipeline.is_some() {
            let bounds = shape
                .pipeline_bounds(input.len())
                .ok_or(EncodeIdsError::Overflow)?;
            let mut total = 0usize;
            source
                .added
                .visit_matches(input, false, |id, (start, end)| {
                    let bytes = if id.is_some() {
                        1
                    } else {
                        shape
                            .pipeline_bounds(end - start)
                            .ok_or(EncodeIdsError::Overflow)?
                            .output
                    };
                    total = total.checked_add(bytes).ok_or(EncodeIdsError::Overflow)?;
                    Ok::<_, EncodeIdsError>(())
                })?;
            return Ok(Self::Pipeline {
                added: source.added,
                shape,
                bounds,
                total: total.max(bounds.output),
                failure: None,
            });
        }
        let nfc = if shape.nfc {
            Some(nfc::Plan::new(input, shape.before).map_err(|_| EncodeIdsError::Overflow)?)
        } else {
            None
        };
        let text = if shape.after.is_empty() && shape.literal.is_none() {
            0
        } else {
            shape.text_bound(input).ok_or(EncodeIdsError::Overflow)?
        };
        // Every raw-token boundary starts a new normalization span. Prefixes are
        // applied once to each nonempty unmatched span, not once to the input.
        let mut total = 0usize;
        source
            .added
            .visit_matches(input, false, |id, (start, end)| {
                let bytes = if id.is_some() {
                    1
                } else {
                    shape
                        .text_bound(&input[start..end])
                        .ok_or(EncodeIdsError::Overflow)?
                };
                total = total.checked_add(bytes).ok_or(EncodeIdsError::Overflow)?;
                Ok::<_, EncodeIdsError>(())
            })?;
        total = total.max(shape.text_bound(input).ok_or(EncodeIdsError::Overflow)?);
        nfc.as_ref()
            .map_or(0, |plan| plan.requirements().buffer_bytes())
            .checked_add(text)
            .ok_or(EncodeIdsError::Overflow)?;
        nfc.as_ref()
            .map_or(0, |plan| plan.requirements().text_capacity())
            .checked_add(text)
            .ok_or(EncodeIdsError::Overflow)?;
        Ok(Self::Transform {
            added: source.added,
            shape,
            nfc,
            text,
            total,
        })
    }
    pub(super) fn text_bound(&self) -> usize {
        match self {
            Self::Identity { bytes, .. } => *bytes,
            Self::Transform { total, .. } | Self::Pipeline { total, .. } => *total,
        }
    }
    pub(super) fn capacities(&self) -> [usize; 3] {
        match self {
            Self::Identity { .. } => [0; 3],
            Self::Pipeline { bounds, .. } => bounds.capacities(),
            Self::Transform { nfc, text, .. } => match nfc {
                Some(plan) => {
                    let r = plan.requirements();
                    [
                        r.scalar_capacity(),
                        r.scalar_capacity(),
                        r.text_capacity() + text,
                    ]
                }
                None => [0, 0, *text],
            },
        }
    }
    pub(super) fn buffer_bytes(&self) -> usize {
        match self {
            Self::Identity { .. } => 0,
            Self::Pipeline { bounds, .. } => {
                bounds.buffers().expect("checked normalization buffers")
            }
            Self::Transform { nfc, text, .. } => {
                nfc.as_ref()
                    .map_or(0, |plan| plan.requirements().buffer_bytes())
                    + text
            }
        }
    }
    pub(super) fn control_bytes(&self) -> Result<usize, EncodeIdsError> {
        let selected = match self {
            Self::Identity { source, .. } => source.control_bytes::<EncodeIdsError>(),
            Self::Pipeline { added, bounds, .. } => added
                .raw_literal_control_bytes()
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_add(bounds.controls()?)),
            Self::Transform { added, nfc, .. } => added
                .raw_literal_control_bytes()
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| {
                    n.checked_add(
                        nfc.as_ref()
                            .map_or(0, |plan| plan.requirements().control_bytes()),
                    )
                }),
        }
        .ok_or(EncodeIdsError::Overflow)?;
        [
            selected,
            size_of::<Self>(),
            size_of::<Result<Self, EncodeIdsError>>(),
            size_of::<Runtime<'_>>(),
            size_of::<Result<Runtime<'_>, EncodeIdsError>>(),
            size_of::<Retired>(),
            size_of::<Option<Retired>>(),
            size_of::<&mut nfc::Workspace<'_>>(),
            size_of::<&mut InputEncoder<'_, '_>>(),
            size_of::<Result<(), EncodeIdsError>>(),
            size_of::<Shape<'_>>(),
            crate::normalizers::Replace::literal_control_bytes(),
            size_of::<[usize; 3]>(),
            size_of::<&mut usize>(),
            size_of::<String>(),
            size_of::<(&str, usize)>(),
            size_of::<(usize, bool)>(),
        ]
        .iter()
        .try_fold(0usize, |n, &x| n.checked_add(x))
        .ok_or(EncodeIdsError::Overflow)
    }
    pub(super) fn prepare(self) -> Result<Runtime<'a>, EncodeIdsError> {
        match self {
            Self::Identity { source, .. } => Ok(Runtime::Identity(source)),
            Self::Pipeline {
                added,
                shape,
                bounds,
                failure,
                ..
            } => Ok(Runtime::Pipeline {
                added,
                shape,
                storage: pipeline::Storage::prepare(bounds, failure)
                    .map_err(EncodeIdsError::NormalizationPipelinePreparation)?,
            }),
            Self::Transform {
                added,
                shape,
                nfc,
                text,
                ..
            } => {
                let nfc = nfc
                    .map(|plan| {
                        plan.prepare()
                            .map_err(EncodeIdsError::NormalizationPreparation)
                    })
                    .transpose()?;
                let mut output = String::new();
                output
                    .try_reserve_exact(text)
                    .map_err(EncodeIdsError::Reserve)?;
                Ok(Runtime::Transform {
                    added,
                    shape,
                    nfc,
                    text: output,
                })
            }
        }
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    pub(super) fn fail_pipeline(&mut self, stage: usize) {
        match self {
            Self::Pipeline { failure, .. } => *failure = Some(stage),
            _ => panic!("ordered normalization failure requires its actual pipeline"),
        }
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    pub(super) fn fail(self, buffer: NormalizationBuffer) -> Result<Self, EncodeIdsError> {
        match self {
            Self::Pipeline {
                added,
                shape,
                bounds,
                total,
                ..
            } if matches!(buffer, NormalizationBuffer::Text) => Ok(Self::Pipeline {
                added,
                shape,
                bounds,
                total,
                failure: Some(1),
            }),
            Self::Transform {
                added,
                shape,
                nfc: Some(plan),
                text,
                total,
            } => Ok(Self::Transform {
                added,
                shape,
                nfc: Some(plan.fail_reservation(buffer)),
                text,
                total,
            }),
            Self::Transform {
                added,
                shape,
                nfc: None,
                total,
                ..
            } if matches!(buffer, NormalizationBuffer::Text) => Ok(Self::Transform {
                added,
                shape,
                nfc: None,
                text: usize::MAX,
                total,
            }),
            _ => Err(EncodeIdsError::PipelineProfile),
        }
    }
}
pub(super) enum Runtime<'a> {
    Pipeline {
        added: &'a super::super::AddedVocabulary,
        shape: Shape<'a>,
        storage: pipeline::Storage,
    },
    Identity(IdentityMatching<'a>),
    Transform {
        added: &'a super::super::AddedVocabulary,
        shape: Shape<'a>,
        nfc: Option<nfc::Workspace<'a>>,
        text: String,
    },
}
impl Runtime<'_> {
    pub(super) fn encode(
        &mut self,
        encoder: &mut InputEncoder<'_, '_>,
    ) -> Result<(), EncodeIdsError> {
        let input = encoder.input;
        match self {
            Self::Identity(source) => source.visit(input, |id, offsets| encoder.span(id, offsets)),
            Self::Pipeline {
                added,
                shape,
                storage,
            } => added.visit_matches(input, false, |id, (start, end)| {
                if let Some(id) = id {
                    encoder.output.ids.push(id);
                    return Ok(());
                }
                let initial_span = start == 0;
                let (normalized, initial_end) = storage
                    .apply(*shape, &input[start..end])
                    .map_err(EncodeIdsError::NormalizationPipeline)?;
                added.visit_matches(normalized, true, |id, (start, end)| {
                    if let Some(id) = id {
                        encoder.output.ids.push(id);
                        Ok(())
                    } else {
                        encoder.regex.encode_span(
                            encoder.output,
                            encoder.model,
                            &normalized[start..end],
                            if initial_span { initial_end.saturating_sub(start).min(end - start) } else { 0 },
                        )
                    }
                })
            }),
            Self::Transform {
                added,
                shape,
                nfc,
                text,
            } => added.visit_matches(input, false, |id, (start, end)| {
                if let Some(id) = id {
                    encoder.output.ids.push(id);
                    return Ok(());
                }
                let initial_span = start == 0;
                let (normalized, mut initial_end) = match nfc {
                    Some(workspace) => {
                        workspace
                            .normalize(start..end)
                            .map_err(EncodeIdsError::NormalizationRange)?;
                        workspace.normalized()
                    }
                    None => {
                        let text = &input[start..end];
                        (text, text.chars().next().map_or(0, char::len_utf8))
                    }
                };
                let normalized = if let Some(literal) = shape.literal {
                    initial_end = literal.write_literal(normalized, text);
                    text.as_str()
                } else if !normalized.is_empty() && !shape.after.is_empty() {
                    text.clear();
                    text.push_str(shape.after);
                    text.push_str(normalized);
                    initial_end += shape.after.len();
                    text.as_str()
                } else {
                    normalized
                };
                added.visit_matches(normalized, true, |id, (start, end)| {
                    if let Some(id) = id {
                        encoder.output.ids.push(id);
                        Ok(())
                    } else {
                        encoder.regex.encode_span(
                            encoder.output,
                            encoder.model,
                            &normalized[start..end],
                            if initial_span { initial_end.saturating_sub(start).min(end - start) } else { 0 },
                        )
                    }
                })
            }),
        }
    }
    pub(super) fn retire(self) -> Option<Retired> {
        match self {
            Self::Identity(_) => None,
            Self::Pipeline { storage, .. } => Some(Retired {
                nfc: None,
                text: String::new(),
                pipeline: Some(storage),
            }),
            Self::Transform { nfc, text, .. } => Some(Retired {
                nfc: nfc.map(nfc::Workspace::retire),
                text,
                pipeline: None,
            }),
        }
    }
}
#[derive(Debug)]
pub(super) struct Retired {
    pipeline: Option<pipeline::Storage>,
    nfc: Option<nfc::Retired>,
    text: String,
}
impl Retired {
    pub(super) fn capacities(&self) -> [usize; 3] {
        let mut capacities = self.nfc.as_ref().map_or([0; 3], nfc::Retired::capacities);
        capacities[2] +=
            self.text.capacity() + self.pipeline.as_ref().map_or(0, |s| s.capacities()[2]);
        capacities
    }
}
