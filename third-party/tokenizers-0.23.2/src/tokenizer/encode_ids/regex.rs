//! The actual immutable pipeline and one lexical search workspace per E.
use super::*;
use crate::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use crate::pre_tokenizers::PreTokenizerWrapper;
#[cfg(feature = "fancy-regex")]
use crate::utils::regex as regex_spans;
use core::marker::PhantomData;
mod pipeline;
#[derive(Clone, Copy)]
struct Meta {
    replacement: char,
    scheme: PrependScheme,
    split: bool,
}
impl Meta {
    fn from_source(source: &Metaspace, remove_prefix: bool) -> Self {
        Self {
            replacement: source.get_replacement(),
            scheme: if remove_prefix {
                PrependScheme::Never
            } else {
                source.get_prepend_scheme()
            },
            split: source.get_split(),
        }
    }
}

pub(super) struct RegexPlan<'a> {
    pipeline: Option<pipeline::Plan<'a>>,
    numeric: bool,
    byte_level: bool,
    matches_only: bool,
    metaspace: Option<Meta>,
    #[cfg(feature = "fancy-regex")]
    plan: Option<fancy_regex::workspace::Plan<'a>>,
    _source: PhantomData<&'a Tokenizer>,
}
impl<'a> RegexPlan<'a> {
    pub(super) fn inspect(
        source: TokenizerInput<'a>,
        bytes: usize,
    ) -> Result<Self, EncodeIdsError> {
        let root = source.root;
        if root.padding.is_some() || root.truncation.is_some() {
            return Err(EncodeIdsError::PipelineProfile);
        }
        let mut out = Self {
            pipeline: None,
            numeric: false,
            byte_level: false,
            matches_only: false,
            metaspace: None,
            #[cfg(feature = "fancy-regex")]
            plan: None,
            _source: PhantomData,
        };
        let values = match root.pre_tokenizer.as_ref() {
            None => &[][..],
            Some(PreTokenizerWrapper::Sequence(sequence)) => sequence.as_ref(),
            Some(value) => std::slice::from_ref(value),
        };
        match values {
            [] => return Ok(out),
            [PreTokenizerWrapper::Metaspace(value)] => {
                out.metaspace = Some(Meta::from_source(value, source.remove_input_prefixes));
                return Ok(out);
            }
            [PreTokenizerWrapper::ByteLevel(value)]
                if !value.add_prefix_space && !value.use_regex =>
            {
                out.byte_level = true;
                return Ok(out);
            }
            _ => {}
        }
        #[cfg(feature = "fancy-regex")]
        {
            let (numeric, byte_level, matches_only, plan) = match values {
                [PreTokenizerWrapper::Whitespace(whitespace)] => {
                    (false, false, true, whitespace.workspace_plan())
                }
                [PreTokenizerWrapper::Split(split), PreTokenizerWrapper::ByteLevel(byte)]
                    if !byte.add_prefix_space && !byte.use_regex =>
                {
                    (false, true, false, split.workspace_plan())
                }
                [PreTokenizerWrapper::Digits(digits), PreTokenizerWrapper::ByteLevel(byte)]
                    if digits.individual_digits && !byte.add_prefix_space && byte.use_regex =>
                {
                    (true, true, false, byte.workspace_plan())
                }
                _ => {
                    out.pipeline = Some(pipeline::Plan::inspect(
                        values,
                        source.remove_input_prefixes,
                        bytes,
                    )?);
                    return Ok(out);
                }
            };
            out.numeric = numeric;
            out.byte_level = byte_level;
            out.matches_only = matches_only;
            out.plan = Some(
                plan.ok_or(EncodeIdsError::PipelineProfile)?
                    .map_err(EncodeIdsError::RegexPlan)?,
            );
            Ok(out)
        }
        #[cfg(not(feature = "fancy-regex"))]
        {
            out.pipeline = Some(pipeline::Plan::inspect(
                values,
                source.remove_input_prefixes,
                bytes,
            )?);
            Ok(out)
        }
    }
    pub(super) fn symbol_capacity(&self, bytes: usize) -> Result<usize, EncodeIdsError> {
        if let Some(plan) = &self.pipeline {
            return Ok(plan.text());
        }
        if let Some(meta) = self.metaspace {
            // Each nonempty input span can add at most one replacement scalar.
            // There are at most B such spans for B input bytes; replacing each
            // space contributes at most replacement.len_utf8() bytes per input.
            return bytes
                .checked_mul(2 * meta.replacement.len_utf8())
                .ok_or(EncodeIdsError::Overflow);
        }
        Ok(bytes)
    }
    pub(super) fn delegate_count(&self) -> usize {
        if let Some(plan) = &self.pipeline {
            return plan.delegate_count();
        }
        #[cfg(feature = "fancy-regex")]
        if let Some(plan) = &self.plan {
            return plan
                .requirements()
                .capacity(fancy_regex::workspace::Buffer::Delegates);
        }
        0
    }
    pub(super) fn mapped_capacity(&self, bytes: usize) -> Result<usize, EncodeIdsError> {
        if self.pipeline.is_some() {
            return Ok(0);
        }
        if self.metaspace.is_some() {
            return Ok(bytes);
        }
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
        if let Some(plan) = &self.pipeline {
            return bytes
                .checked_add(plan.required_bytes())
                .ok_or(EncodeIdsError::Overflow);
        }
        if self.metaspace.is_some() {
            bytes = bytes
                .checked_add(size_of::<Meta>())
                .and_then(|n| n.checked_add(size_of::<std::str::CharIndices<'_>>()))
                .and_then(|n| n.checked_add(size_of::<std::str::Chars<'_>>()))
                .and_then(|n| n.checked_add(size_of::<(usize, char, bool)>()))
                .ok_or(EncodeIdsError::Overflow)?;
        }
        if self.byte_level {
            bytes = bytes
                .checked_add(mapping_control_bytes().ok_or(EncodeIdsError::Overflow)?)
                .ok_or(EncodeIdsError::Overflow)?;
        }
        #[cfg(feature = "fancy-regex")]
        if let Some(plan) = &self.plan {
            return bytes
                .checked_add(plan.requirements().required_bytes())
                .and_then(|n| n.checked_add(regex_spans::visit_control_bytes()?))
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
        if let Some(plan) = &mut self.pipeline {
            plan.fail(0, failure);
            return Ok(());
        }
        let plan = self.plan.take().ok_or(EncodeIdsError::PipelineProfile)?;
        self.plan = Some(
            plan.fail_reservation(failure)
                .map_err(EncodeIdsError::RegexPlan)?,
        );
        Ok(())
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    pub(super) fn fail_pipeline_buffer(&mut self, stage: usize) {
        self.pipeline
            .as_mut()
            .expect("ordered pipeline destination")
            .fail_buffer(stage);
    }
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    pub(super) fn fail_at(
        &mut self,
        ordinal: usize,
        failure: fancy_regex::workspace::PrepareFailure,
    ) -> Result<(), EncodeIdsError> {
        if let Some(plan) = &mut self.pipeline {
            plan.fail(ordinal, failure);
            Ok(())
        } else if ordinal == 0 {
            self.fail(failure)
        } else {
            Err(EncodeIdsError::PipelineProfile)
        }
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
            pipeline: self.pipeline.map(pipeline::Plan::prepare).transpose()?,
            numeric: self.numeric,
            byte_level: self.byte_level,
            matches_only: self.matches_only,
            metaspace: self.metaspace,
            #[cfg(feature = "fancy-regex")]
            workspace,
            _source: PhantomData,
        })
    }
}
pub(super) struct Runtime<'a> {
    pipeline: Option<pipeline::Runtime<'a>>,
    numeric: bool,
    byte_level: bool,
    matches_only: bool,
    metaspace: Option<Meta>,
    #[cfg(feature = "fancy-regex")]
    workspace: Option<fancy_regex::workspace::Workspace<'a>>,
    _source: PhantomData<&'a Tokenizer>,
}
impl Runtime<'_> {
    pub(super) fn encode_span(
        &mut self,
        output: &mut EncodeIdsOutput,
        model: &ModelWrapper,
        text: &str,
        initial_origin: usize,
    ) -> Result<(), EncodeIdsError> {
        if let Some(pipeline) = &mut self.pipeline {
            return pipeline.encode(output, model, text, initial_origin);
        }
        if let Some(meta) = self.metaspace {
            return meta.encode(output, model, text, initial_origin != 0);
        }
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
        model: &ModelWrapper,
        text: &str,
    ) -> Result<(), EncodeIdsError> {
        #[cfg(feature = "fancy-regex")]
        if let Some(workspace) = &mut self.workspace {
            let mut encoder = SpanEncoder {
                output,
                model,
                text,
                byte_level: self.byte_level,
            };
            return regex_spans::visit_spans(workspace, text, self.matches_only, move |offsets| {
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
                byte_level: true,
            }
            .encode((0, text.len()));
        }
        encode_model(
            model,
            &mut output.scratch,
            &mut output.unigram,
            text,
            &mut output.ids,
        )
    }
}
struct SpanEncoder<'a> {
    output: &'a mut EncodeIdsOutput,
    model: &'a ModelWrapper,
    text: &'a str,
    byte_level: bool,
}
impl SpanEncoder<'_> {
    fn encode(&mut self, (start, end): crate::Offsets) -> Result<(), EncodeIdsError> {
        if !self.byte_level {
            return encode_model(
                self.model,
                &mut self.output.scratch,
                &mut self.output.unigram,
                &self.text[start..end],
                &mut self.output.ids,
            );
        }
        self.output.mapped.clear();
        self.output.mapped.extend(
            crate::pre_tokenizers::byte_level::transformations(&self.text[start..end])
                .map(|(c, _)| c),
        );
        // Keep the full mapped split for the model's existing ignore_merges fast path.
        encode_model(
            self.model,
            &mut self.output.scratch,
            &mut self.output.unigram,
            &self.output.mapped,
            &mut self.output.ids,
        )
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

impl Meta {
    // Shared bounded Metaspace writer and split visitation, used by standalone
    // and ordered execution. The first-origin frontier comes from the same
    // normalization changes rather than the stage's ordinal.
    fn write(self, text: &str, initial: usize, out: &mut String) -> usize {
        let begin = out.len();
        let first = text
            .chars()
            .next()
            .map(|c| if c == ' ' { self.replacement } else { c });
        let prepend = self.scheme == PrependScheme::Always
            || self.scheme == PrependScheme::First && initial != 0;
        let mut frontier = 0;
        if !text.is_empty() && prepend && first != Some(self.replacement) {
            out.push(self.replacement);
            if initial != 0 {
                frontier = out.len() - begin;
            }
        }
        for (at, c) in text.char_indices() {
            out.push(if c == ' ' { self.replacement } else { c });
            if at < initial {
                frontier = out.len() - begin;
            }
        }
        frontier
    }
    fn visit<E>(
        self,
        text: &str,
        mut visit: impl FnMut(crate::Offsets) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut start = 0;
        if self.split {
            for (at, c) in text.char_indices() {
                if c == self.replacement && at > start {
                    visit((start, at))?;
                    start = at;
                }
            }
        }
        visit((start, text.len()))
    }
    fn encode(
        self,
        output: &mut EncodeIdsOutput,
        model: &ModelWrapper,
        text: &str,
        initial_origin: bool,
    ) -> Result<(), EncodeIdsError> {
        if text.is_empty() {
            return Ok(());
        }
        output.mapped.clear();
        self.write(
            text,
            if initial_origin {
                text.chars().next().map_or(0, char::len_utf8)
            } else {
                0
            },
            &mut output.mapped,
        );
        let (text, scratch, unigram, ids) = (
            &output.mapped,
            &mut output.scratch,
            &mut output.unigram,
            &mut output.ids,
        );
        self.visit(text, |(start, end)| {
            encode_model(model, scratch, unigram, &text[start..end], ids)
        })
    }
}
