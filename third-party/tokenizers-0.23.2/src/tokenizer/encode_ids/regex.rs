//! The actual immutable pipeline and one lexical search workspace per E.
use super::*;
#[cfg(feature = "fancy-regex")]
use crate::pre_tokenizers::compiled_split;
use crate::{models::bpe::BPE, pre_tokenizers::PreTokenizerWrapper};
use core::marker::PhantomData;

pub(super) struct RegexPlan<'a> {
    numeric: bool,
    byte_level: bool,
    #[cfg(feature = "fancy-regex")]
    plan: Option<fancy_regex::workspace::Plan<'a>>,
    _source: PhantomData<&'a Tokenizer>,
}
impl<'a> RegexPlan<'a> {
    pub(super) fn inspect(source: &'a Tokenizer) -> Result<Self, EncodeIdsError> {
        if source.padding.is_some() || source.truncation.is_some() {
            return Err(EncodeIdsError::PipelineProfile);
        }
        if source.pre_tokenizer.is_none() {
            return Ok(Self {
                numeric: false,
                byte_level: false,
                #[cfg(feature = "fancy-regex")]
                plan: None,
                _source: PhantomData,
            });
        }
        if matches!(&source.pre_tokenizer, Some(PreTokenizerWrapper::ByteLevel(byte))
            if !byte.add_prefix_space && !byte.use_regex)
        {
            return Ok(Self {
                numeric: false,
                byte_level: true,
                #[cfg(feature = "fancy-regex")]
                plan: None,
                _source: PhantomData,
            });
        }
        #[cfg(feature = "fancy-regex")]
        {
            let Some(PreTokenizerWrapper::Sequence(sequence)) = &source.pre_tokenizer else {
                return Err(EncodeIdsError::PipelineProfile);
            };
            let (numeric, plan) = match sequence.as_ref() {
                [
                    PreTokenizerWrapper::CompiledRegexSplit(split),
                    PreTokenizerWrapper::ByteLevel(byte),
                ] if !byte.add_prefix_space && !byte.use_regex => (false, split.workspace_plan()),
                [
                    PreTokenizerWrapper::Digits(digits),
                    PreTokenizerWrapper::CompiledByteLevel(byte),
                ] if digits.individual_digits
                    && !byte.settings().add_prefix_space
                    && byte.settings().use_regex =>
                {
                    (true, byte.workspace_plan())
                }
                _ => return Err(EncodeIdsError::PipelineProfile),
            };
            Ok(Self {
                numeric,
                byte_level: true,
                plan: Some(plan.map_err(EncodeIdsError::RegexPlan)?),
                _source: PhantomData,
            })
        }
        #[cfg(not(feature = "fancy-regex"))]
        {
            Err(EncodeIdsError::PipelineProfile)
        }
    }
    pub(super) fn delegate_count(&self) -> usize {
        #[cfg(feature = "fancy-regex")]
        if let Some(plan) = &self.plan {
            return plan
                .requirements()
                .capacity(fancy_regex::workspace::Buffer::Delegates);
        }
        0
    }
    pub(super) fn mapped_capacity(&self, bytes: usize) -> Result<usize, EncodeIdsError> {
        if self.byte_level {
            return bytes.checked_mul(2).ok_or(EncodeIdsError::Overflow);
        }
        let _ = bytes;
        Ok(0)
    }
    pub(super) fn required_bytes(&self) -> Result<usize, EncodeIdsError> {
        let mut bytes = size_of::<Self>()
            .checked_add(size_of::<Runtime<'a>>())
            .and_then(|n| n.checked_add(size_of::<Result<Runtime<'a>, EncodeIdsError>>()))
            .ok_or(EncodeIdsError::Overflow)?;
        if self.byte_level {
            bytes = bytes
                .checked_add(mapping_control_bytes().ok_or(EncodeIdsError::Overflow)?)
                .ok_or(EncodeIdsError::Overflow)?;
        }
        #[cfg(feature = "fancy-regex")]
        if let Some(plan) = &self.plan {
            return bytes
                .checked_add(plan.requirements().required_bytes())
                .and_then(|n| n.checked_add(compiled_split::visit_control_bytes()?))
                .and_then(|n| {
                    n.checked_add(if self.numeric {
                        numeric_control_bytes()?
                    } else {
                        0
                    })
                })
                .and_then(|n| n.checked_add(size_of::<Result<(), EncodeIdsError>>()))
                .ok_or(EncodeIdsError::Overflow);
        }
        Ok(bytes)
    }
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    pub(super) fn fail(
        &mut self,
        failure: fancy_regex::workspace::PrepareFailure,
    ) -> Result<(), EncodeIdsError> {
        let plan = self.plan.take().ok_or(EncodeIdsError::PipelineProfile)?;
        self.plan = Some(
            plan.fail_reservation(failure)
                .map_err(EncodeIdsError::RegexPlan)?,
        );
        Ok(())
    }
    pub(super) fn prepare(self) -> Result<Runtime<'a>, EncodeIdsError> {
        #[cfg(feature = "fancy-regex")]
        let workspace = match self.plan {
            Some(plan) => Some(
                plan.prepare()
                    .map_err(|error| EncodeIdsError::RegexPreparation(error.retire()))?,
            ),
            None => None,
        };
        Ok(Runtime {
            numeric: self.numeric,
            byte_level: self.byte_level,
            #[cfg(feature = "fancy-regex")]
            workspace,
            _source: PhantomData,
        })
    }
}
pub(super) struct Runtime<'a> {
    numeric: bool,
    byte_level: bool,
    #[cfg(feature = "fancy-regex")]
    workspace: Option<fancy_regex::workspace::Workspace<'a>>,
    _source: PhantomData<&'a Tokenizer>,
}
impl Runtime<'_> {
    pub(super) fn encode_span(
        &mut self,
        output: &mut EncodeIdsOutput,
        model: &BPE,
        text: &str,
    ) -> Result<(), EncodeIdsError> {
        if self.numeric {
            // The exact ordinary Digits predicate, never the regex engine's Unicode table.
            let numeric: fn(char) -> bool = char::is_numeric;
            for piece in crate::tokenizer::pattern::char_coverage(text, &numeric) {
                let ((start, end), _) = match piece {
                    Ok(piece) => piece,
                    Err(never) => match never {},
                };
                self.encode_piece(output, model, &text[start..end])?;
            }
            return Ok(());
        }
        self.encode_piece(output, model, text)
    }
    fn encode_piece(
        &mut self,
        output: &mut EncodeIdsOutput,
        model: &BPE,
        text: &str,
    ) -> Result<(), EncodeIdsError> {
        #[cfg(feature = "fancy-regex")]
        if let Some(workspace) = &mut self.workspace {
            let mut encoder = SpanEncoder {
                output,
                model,
                text,
            };
            return compiled_split::visit_spans(workspace, text, move |offsets| {
                encoder.encode(offsets)
            });
        }
        if self.byte_level {
            // The ordinary ByteLevel(false, _, false) maps the complete input
            // span once. Reuse the same mapping/BPE worker as regex pieces.
            return SpanEncoder {
                output,
                model,
                text,
            }
            .encode((0, text.len()));
        }
        output
            .scratch
            .encode(model, text, &mut output.ids)
            .map_err(|_| EncodeIdsError::MissingUnknown)
    }
}
struct SpanEncoder<'a> {
    output: &'a mut EncodeIdsOutput,
    model: &'a BPE,
    text: &'a str,
}
impl SpanEncoder<'_> {
    fn encode(&mut self, (start, end): crate::Offsets) -> Result<(), EncodeIdsError> {
        self.output.mapped.clear();
        self.output.mapped.extend(
            crate::pre_tokenizers::byte_level::transformations(&self.text[start..end])
                .map(|(c, _)| c),
        );
        // Keep the full mapped split for the model's existing ignore_merges fast path.
        self.output
            .scratch
            .encode(self.model, &self.output.mapped, &mut self.output.ids)
            .map_err(|_| EncodeIdsError::MissingUnknown)
    }
}

#[cfg(feature = "fancy-regex")]
fn numeric_control_bytes() -> Option<usize> {
    use crate::tokenizer::pattern::{CharMatches, Coverage};
    [
        size_of::<fn(char) -> bool>(),
        size_of::<Coverage<CharMatches<'static, 'static, fn(char) -> bool>>>(),
        size_of::<Option<Result<(crate::Offsets, bool), std::convert::Infallible>>>(),
        size_of::<Result<(), EncodeIdsError>>(),
    ]
    .iter()
    .try_fold(0usize, |sum, &n| sum.checked_add(n))
}

fn mapping_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<SpanEncoder<'_>>(),
        core::mem::size_of_val(
            &crate::pre_tokenizers::byte_level::transformations("").map(|(c, _)| c),
        ),
        size_of::<crate::Offsets>(),
        size_of::<Result<(), EncodeIdsError>>(),
    ];
    parts
        .iter()
        .copied()
        .try_fold(core::mem::size_of_val(&parts), usize::checked_add)
}
