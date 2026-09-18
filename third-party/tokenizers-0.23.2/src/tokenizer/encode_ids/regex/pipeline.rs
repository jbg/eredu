//! Ordered pre-tokenizer stages over two paid split/text destinations. Every
//! regex stage owns the workspace planned from its exact retained component.
use super::*;
use crate::pre_tokenizers::PreTokenizerWrapper as P;

#[derive(Clone, Copy)]
enum Kind {
    Meta(Meta),
    Digits(bool),
    Split { matches_only: bool },
    Byte { prefix: bool },
}
struct Leaf<'a> {
    kind: Kind,
    #[cfg(feature = "fancy-regex")]
    regex: Option<fancy_regex::workspace::Plan<'a>>,
    _source: PhantomData<&'a P>,
}
impl<'a> Leaf<'a> {
    fn inspect(source: &'a P, remove_prefix: bool) -> Result<Self, EncodeIdsError> {
        let kind = match source {
            P::Metaspace(meta) => Kind::Meta(Meta::from_source(meta, remove_prefix)),
            P::Digits(digits) => Kind::Digits(digits.individual_digits),
            P::ByteLevel(byte) => Kind::Byte {
                prefix: byte.add_prefix_space,
            },
            P::Whitespace(_) => Kind::Split { matches_only: true },
            P::Split(_) => Kind::Split {
                matches_only: false,
            },
            _ => return Err(EncodeIdsError::PipelineProfile),
        };
        #[cfg(feature = "fancy-regex")]
        let regex = match source {
            P::Whitespace(value) => Some(
                value
                    .workspace_plan()
                    .ok_or(EncodeIdsError::PipelineProfile)?
                    .map_err(EncodeIdsError::RegexPlan)?,
            ),
            P::Split(value) => Some(
                value
                    .workspace_plan()
                    .ok_or(EncodeIdsError::PipelineProfile)?
                    .map_err(EncodeIdsError::RegexPlan)?,
            ),
            P::ByteLevel(value) if value.use_regex => Some(
                value
                    .workspace_plan()
                    .ok_or(EncodeIdsError::PipelineProfile)?
                    .map_err(EncodeIdsError::RegexPlan)?,
            ),
            _ => None,
        };
        #[cfg(not(feature = "fancy-regex"))]
        if matches!(source, P::Whitespace(_) | P::Split(_))
            || matches!(source, P::ByteLevel(value) if value.use_regex)
        {
            return Err(EncodeIdsError::PipelineProfile);
        }
        Ok(Self {
            kind,
            #[cfg(feature = "fancy-regex")]
            regex,
            _source: PhantomData,
        })
    }
}
pub(super) struct Plan<'a> {
    source: &'a [P],
    remove_prefix: bool,
    text: usize,
    prefix_text: usize,
    required: usize,
    delegates: usize,
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    failure: Option<(usize, fancy_regex::workspace::PrepareFailure)>,
    #[cfg(feature = "tokenizer-compiler-test-support")]
    buffer_failure: Option<usize>,
}
impl<'a> Plan<'a> {
    pub(super) fn inspect(
        source: &'a [P],
        remove_prefix: bool,
        bytes: usize,
    ) -> Result<Self, EncodeIdsError> {
        let overflow = || EncodeIdsError::Overflow;
        let mut text = bytes;
        let mut prefix = false;
        let mut required = 0usize;
        let mut delegates = 0usize;
        for source in source {
            let leaf = Leaf::inspect(source, remove_prefix)?;
            let factor = match leaf.kind {
                Kind::Meta(meta) => 2 * meta.replacement.len_utf8(),
                Kind::Byte { prefix: true } => {
                    prefix = true;
                    4
                }
                Kind::Byte { prefix: false } => 2,
                _ => 1,
            };
            text = text.checked_mul(factor).ok_or_else(overflow)?;
            #[cfg(feature = "fancy-regex")]
            if let Some(plan) = leaf.regex {
                required = required
                    .checked_add(plan.requirements().required_bytes())
                    .ok_or_else(overflow)?;
                delegates = delegates
                    .checked_add(
                        plan.requirements()
                            .capacity(fancy_regex::workspace::Buffer::Delegates),
                    )
                    .ok_or_else(overflow)?;
            }
        }
        let prefix_text = if prefix { text } else { 0 };
        for bytes in [
            Layout::array::<Stage<'a>>(source.len())
                .map_err(|_| overflow())?
                .size(),
            Layout::array::<Span>(text)
                .map_err(|_| overflow())?
                .size()
                .checked_mul(2)
                .ok_or_else(overflow)?,
            text.checked_mul(2).ok_or_else(overflow)?,
            prefix_text,
            size_of::<Self>(),
            size_of::<Runtime<'a>>(),
            size_of::<Leaf<'a>>(),
            size_of::<Kind>(),
            size_of::<Result<Self, EncodeIdsError>>(),
            size_of::<Result<Runtime<'a>, EncodeIdsError>>(),
            size_of::<Result<Leaf<'a>, EncodeIdsError>>(),
            size_of::<Span>(),
            size_of::<[Buffer; 2]>(),
            size_of::<(&mut Buffer, &str, usize, Kind)>(),
            size_of::<(usize, usize, usize, bool)>(),
            size_of::<Stage<'a>>(),
            size_of::<Vec<Stage<'a>>>(),
            size_of::<Result<(), EncodeIdsError>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<std::slice::Iter<'_, P>>(),
            size_of::<std::slice::IterMut<'_, Stage<'a>>>(),
            size_of::<std::slice::Iter<'_, Span>>(),
            size_of::<Option<(usize, usize)>>(),
            size_of::<(&str, usize)>(),
            size_of::<(&Buffer, &mut Buffer)>(),
            size_of::<(&mut Vec<Span>, usize, usize)>(),
            size_of::<(&mut Buffer, usize)>(),
            size_of::<std::str::CharIndices<'_>>(),
            size_of::<std::str::Chars<'_>>(),
            mapping_control_bytes().ok_or_else(overflow)?,
        ] {
            required = required.checked_add(bytes).ok_or_else(overflow)?;
        }
        #[cfg(feature = "fancy-regex")]
        {
            required = required
                .checked_add(regex_spans::visit_control_bytes().ok_or_else(overflow)?)
                .and_then(|n| n.checked_add(numeric_control_bytes()?))
                .ok_or_else(overflow)?;
        }
        Ok(Self {
            source,
            remove_prefix,
            text,
            prefix_text,
            required,
            delegates,
            #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
            failure: None,
            #[cfg(feature = "tokenizer-compiler-test-support")]
            buffer_failure: None,
        })
    }
    pub(super) fn text(&self) -> usize {
        self.text
    }
    pub(super) fn required_bytes(&self) -> usize {
        self.required
    }
    pub(super) fn delegate_count(&self) -> usize {
        self.delegates
    }
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    pub(super) fn fail(&mut self, ordinal: usize, target: fancy_regex::workspace::PrepareFailure) {
        self.failure = Some((ordinal, target));
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    pub(super) fn fail_buffer(&mut self, stage: usize) {
        assert!(stage < 6);
        self.buffer_failure = Some(stage);
    }
    fn capacity(&self, stage: usize, actual: usize) -> usize {
        #[cfg(feature = "tokenizer-compiler-test-support")]
        if self.buffer_failure == Some(stage) {
            return usize::MAX;
        }
        let _ = stage;
        actual
    }
    pub(super) fn prepare(self) -> Result<Runtime<'a>, EncodeIdsError> {
        let mut out = Runtime {
            stages: Vec::new(),
            buffers: [Buffer::default(), Buffer::default()],
            prefix: String::new(),
            limit: self.text,
        };
        out.stages
            .try_reserve_exact(self.capacity(0, self.source.len()))
            .map_err(EncodeIdsError::Reserve)?;
        for (index, buffer) in out.buffers.iter_mut().enumerate() {
            buffer
                .text
                .try_reserve_exact(self.capacity(1 + 2 * index, self.text))
                .map_err(EncodeIdsError::Reserve)?;
            buffer
                .spans
                .try_reserve_exact(self.capacity(2 + 2 * index, self.text))
                .map_err(EncodeIdsError::Reserve)?;
        }
        out.prefix
            .try_reserve_exact(self.capacity(5, self.prefix_text))
            .map_err(EncodeIdsError::Reserve)?;
        #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
        let failure = self.failure;
        #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
        let mut ordinal = 0;
        for source in self.source {
            let leaf = Leaf::inspect(source, self.remove_prefix)?;
            #[cfg(feature = "fancy-regex")]
            let regex = leaf
                .regex
                .map(|plan| {
                    #[cfg(feature = "tokenizer-compiler-test-support")]
                    let plan = {
                        let selected = ordinal;
                        ordinal += 1;
                        match failure {
                            Some((at, failure)) if at == selected => plan
                                .fail_reservation(failure)
                                .map_err(EncodeIdsError::RegexPlan)?,
                            _ => plan,
                        }
                    };
                    plan.prepare()
                        .map_err(|error| EncodeIdsError::RegexPreparation(error.retire()))
                })
                .transpose()?;
            out.stages.push(Stage {
                kind: leaf.kind,
                #[cfg(feature = "fancy-regex")]
                regex,
                _source: PhantomData,
            });
        }
        Ok(out)
    }
}
#[derive(Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
    initial: usize,
}
#[derive(Default)]
struct Buffer {
    text: String,
    spans: Vec<Span>,
}
impl Buffer {
    fn span(&mut self, start: usize, end: usize, initial: usize) {
        if start == end {
            return;
        }
        assert!(
            self.spans.len() < self.spans.capacity(),
            "checked split population"
        );
        self.spans.push(Span {
            start,
            end,
            initial,
        });
    }
    fn copy(&mut self, input: &str, initial: usize, (start, end): crate::Offsets) {
        if start == end {
            return;
        }
        let begin = self.text.len();
        self.text.push_str(&input[start..end]);
        self.span(
            begin,
            self.text.len(),
            initial.saturating_sub(start).min(end - start),
        );
    }
    fn bytes(&mut self, input: &str, initial: usize, (start, end): crate::Offsets) {
        if start == end {
            return;
        }
        let begin = self.text.len();
        let text = &input[start..end];
        let mut chars = text.chars();
        let mut at = start;
        let mut previous = false;
        let mut frontier = 0;
        for (c, change) in crate::pre_tokenizers::byte_level::transformations(text) {
            let is_initial = if change > 0 { previous } else { at < initial };
            if change <= 0 {
                at += chars
                    .next()
                    .expect("ordinary byte mapping source")
                    .len_utf8();
            }
            previous = is_initial;
            self.text.push(c);
            if is_initial {
                frontier = self.text.len() - begin;
            }
        }
        self.span(begin, self.text.len(), frontier);
    }
}
struct Stage<'a> {
    kind: Kind,
    #[cfg(feature = "fancy-regex")]
    regex: Option<fancy_regex::workspace::Workspace<'a>>,
    _source: PhantomData<&'a P>,
}
pub(super) struct Runtime<'a> {
    stages: Vec<Stage<'a>>,
    buffers: [Buffer; 2],
    prefix: String,
    limit: usize,
}
impl Runtime<'_> {
    pub(super) fn encode(
        &mut self,
        output: &mut EncodeIdsOutput,
        model: &ModelWrapper,
        text: &str,
        initial: usize,
    ) -> Result<(), EncodeIdsError> {
        if text.is_empty() {
            return Ok(());
        }
        assert!(
            text.len() <= self.limit,
            "checked pre-tokenizer source bytes"
        );
        self.buffers[0].text.clear();
        self.buffers[0].spans.clear();
        self.buffers[0].copy(text, initial, (0, text.len()));
        let mut current = 0;
        for stage in &mut self.stages {
            let [a, b] = &mut self.buffers;
            let (source, dest) = if current == 0 { (&*a, b) } else { (&*b, a) };
            dest.text.clear();
            dest.spans.clear();
            for span in &source.spans {
                let input = &source.text[span.start..span.end];
                match stage.kind {
                    Kind::Meta(meta) => {
                        let begin = dest.text.len();
                        let frontier = meta.write(input, span.initial, &mut dest.text);
                        let normalized = &dest.text[begin..];
                        let spans = &mut dest.spans;
                        meta.visit(normalized, |(start, end)| {
                            if start != end {
                                assert!(
                                    spans.len() < spans.capacity(),
                                    "checked Metaspace split population"
                                );
                                spans.push(Span {
                                    start: begin + start,
                                    end: begin + end,
                                    initial: frontier.saturating_sub(start).min(end - start),
                                });
                            }
                            Ok::<_, EncodeIdsError>(())
                        })?;
                    }
                    Kind::Digits(individual) => {
                        let numeric: fn(char) -> bool = char::is_numeric;
                        let mut pending = None;
                        for item in crate::tokenizer::pattern::char_coverage(input, &numeric) {
                            let ((start, end), matched) = match item {
                                Ok(item) => item,
                                Err(never) => match never {},
                            };
                            if matched && !individual {
                                pending = Some((pending.map_or(start, |(start, _)| start), end));
                            } else {
                                if let Some(offsets) = pending.take() {
                                    dest.copy(input, span.initial, offsets);
                                }
                                dest.copy(input, span.initial, (start, end));
                            }
                        }
                        if let Some(offsets) = pending {
                            dest.copy(input, span.initial, offsets);
                        }
                    }
                    Kind::Split { matches_only } => {
                        #[cfg(feature = "fancy-regex")]
                        regex_spans::visit_spans(
                            stage.regex.as_mut().expect("selected source regex"),
                            input,
                            matches_only,
                            |offsets| {
                                dest.copy(input, span.initial, offsets);
                                Ok::<_, EncodeIdsError>(())
                            },
                        )?;
                        #[cfg(not(feature = "fancy-regex"))]
                        {
                            let _ = matches_only;
                            return Err(EncodeIdsError::PipelineProfile);
                        }
                    }
                    Kind::Byte { prefix } => {
                        let (input, initial) = if prefix && !input.starts_with(' ') {
                            self.prefix.clear();
                            self.prefix.push(' ');
                            self.prefix.push_str(input);
                            (
                                self.prefix.as_str(),
                                if span.initial == 0 {
                                    0
                                } else {
                                    span.initial + 1
                                },
                            )
                        } else {
                            (input, span.initial)
                        };
                        #[cfg(feature = "fancy-regex")]
                        if let Some(regex) = &mut stage.regex {
                            regex_spans::visit_spans(regex, input, false, |offsets| {
                                dest.bytes(input, initial, offsets);
                                Ok::<_, EncodeIdsError>(())
                            })?;
                            continue;
                        }
                        dest.bytes(input, initial, (0, input.len()));
                    }
                }
                assert!(
                    dest.text.len() <= self.limit && dest.text.capacity() <= self.limit,
                    "checked pre-tokenizer text population"
                );
            }
            current = 1 - current;
        }
        let buffer = &self.buffers[current];
        for span in &buffer.spans {
            encode_model(
                model,
                &mut output.scratch,
                &mut output.unigram,
                &buffer.text[span.start..span.end],
                &mut output.ids,
            )?;
        }
        Ok(())
    }
}
