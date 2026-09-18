//! Exact borrowed root planning and fresh aggregate HF construction. No account.
mod decode;
mod model;
mod normalizer;
mod pre;
mod regex;
use self::regex::{RegexSelection, RegexState};
use super::{
    added_vocabulary::AddedVocabularyRecipe, AddedVocabulary, AddedVocabularyCompileError,
    AddedVocabularyCompileFailure, Tokenizer, TokenizerImpl,
};
use crate::processors::template::compiled::{compiler as template, TemplateCompileFailure};
use crate::{
    decoders::{sequence::Sequence as DecodeSequence, DecoderWrapper},
    models::{
        bpe::{BpeCompileError, BpeCompileFailure, BpeCompilePlan},
        unigram::{UnigramCompileError, UnigramCompileFailure},
        wordlevel::{WordLevelCompileError, WordLevelCompileFailure},
        ModelWrapper,
    },
    normalizers::NormalizerWrapper,
    pre_tokenizers::{
        byte_level::{ByteLevel, ByteLevelSettings},
        digits::Digits,
        sequence::Sequence as PreSequence,
        PreTokenizerWrapper,
    },
    processors::{sequence::Sequence as PostSequence, PostProcessorWrapper},
    utils::borrowed_json::{self as json, Reader, Span},
};
use std::{alloc::Layout, collections::TryReserveError, convert::TryFrom, fmt, mem::size_of};

/// Fixed source validation categories for the current aggregate constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerCompileErrorKind {
    /// Input is not UTF-8.
    InvalidUtf8,
    /// Input is not valid JSON for the selected field.
    InvalidJson,
    /// JSON nesting exceeds the fixed parser stack.
    DepthLimit,
    /// A required field is absent.
    MissingField,
    /// A recognized field occurs more than once.
    DuplicateField,
    /// A field has no constructor lowering yet.
    UnsupportedField,
    /// The tokenizer format version is not 1.0.
    Version,
    /// The selected regex source or component settings lack this constructor profile.
    RegexProfile,
    /// The selected component or settings lack complete construction support.
    ComponentProfile,
    /// An actual reserved capacity exceeds its accepted source extent.
    CapacityExceeded,
    /// A checked host extent overflowed.
    Overflow,
}
/// Fixed root error or exact borrowed component diagnostic; no allocated message.
#[derive(Debug, Clone, Copy)]
pub enum TokenizerCompileError {
    /// Root or selected pipeline component validation failed.
    Root {
        /// Fixed validation category.
        kind: TokenizerCompileErrorKind,
        /// Byte offset in the complete root input.
        offset: usize,
    },
    /// Model planning diagnostic; its offset is relative to the model object.
    Model(BpeCompileError),
    /// WordLevel model planning diagnostic.
    WordLevel(WordLevelCompileError),
    /// Scored Unigram model planning diagnostic.
    Unigram(UnigramCompileError),
    /// Added-array planning diagnostic; its offset is relative to that array.
    Added(AddedVocabularyCompileError),
}
impl TokenizerCompileError {
    /// Root category, or ComponentProfile for a component-specific source.
    pub fn kind(&self) -> TokenizerCompileErrorKind {
        match self {
            Self::Root { kind, .. } => *kind,
            _ => TokenizerCompileErrorKind::ComponentProfile,
        }
    }
}
impl fmt::Display for TokenizerCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tokenizer construction {self:?}")
    }
}
impl std::error::Error for TokenizerCompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(e) => Some(e),
            Self::WordLevel(e) => Some(e),
            Self::Unigram(e) => Some(e),
            Self::Added(e) => Some(e),
            _ => None,
        }
    }
}
impl From<json::Error> for TokenizerCompileError {
    fn from(e: json::Error) -> Self {
        err(
            match e.kind {
                json::Kind::InvalidJson => K::InvalidJson,
                json::Kind::DepthLimit => K::DepthLimit,
            },
            e.offset,
        )
    }
}
impl From<BpeCompileError> for TokenizerCompileError {
    fn from(e: BpeCompileError) -> Self {
        Self::Model(e)
    }
}
impl From<WordLevelCompileError> for TokenizerCompileError {
    fn from(e: WordLevelCompileError) -> Self {
        Self::WordLevel(e)
    }
}
impl From<AddedVocabularyCompileError> for TokenizerCompileError {
    fn from(e: AddedVocabularyCompileError) -> Self {
        Self::Added(e)
    }
}
pub(crate) type Error = TokenizerCompileError;
pub(crate) type K = TokenizerCompileErrorKind;
pub(crate) fn err(kind: K, offset: usize) -> Error {
    Error::Root { kind, offset }
}
pub(crate) fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(err(K::Overflow, 0))
}
pub(crate) fn fields<const N: usize>(
    input: &str,
    span: Span,
    names: [&str; N],
) -> Result<[Option<Span>; N], Error> {
    let mut out = [None; N];
    let mut object = Reader::new(input, span).object()?;
    while let Some((key, value)) = object.next()? {
        let i = names
            .iter()
            .position(|n| key.is(n))
            .ok_or(err(K::UnsupportedField, key.offset))?;
        if out[i].replace(value).is_some() {
            return Err(err(K::DuplicateField, key.offset));
        }
    }
    Ok(out)
}
pub(crate) fn required(value: Option<Span>, at: usize) -> Result<Span, Error> {
    value.ok_or(err(K::MissingField, at))
}
pub(crate) fn text_is(input: &str, span: Span, expected: &str) -> Result<bool, Error> {
    let mut r = Reader::new(input, span);
    let t = r.string()?;
    r.finish()?;
    Ok(t.is(expected))
}
fn boolean(input: &str, value: Option<Span>, default: bool) -> Result<bool, Error> {
    match value {
        None => Ok(default),
        Some(s) => match &input[s.start..s.end] {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(err(K::InvalidJson, s.start)),
        },
    }
}
fn absent(input: &str, value: Option<Span>) -> Option<Span> {
    value.filter(|s| &input[s.start..s.end] != "null")
}
#[derive(Debug, Clone, Copy)]
enum Role {
    Pre,
    Post,
    Decode,
}
#[derive(Debug, Clone, Copy)]
enum Inline {
    Byte(ByteLevelSettings),
    Digits(bool),
    Metaspace(pre::Meta),
    Decode(decode::Inline),
    Regex(RegexSelection),
    Template {
        span: Span,
        requirements: template::Requirements,
    },
}
fn inline(input: &str, span: Span, role: Role) -> Result<Inline, Error> {
    // First select the actual type without building serde strings/components.
    let mut r = Reader::new(input, span).object()?;
    let mut typ = None;
    while let Some((key, value)) = r.next()? {
        if key.is("type") {
            if typ.replace(value).is_some() {
                return Err(err(K::DuplicateField, key.offset));
            }
        }
    }
    let typ = required(typ, span.start)?;
    if matches!(role, Role::Pre) && text_is(input, typ, "Metaspace")? {
        return pre::select(input, span).map(Inline::Metaspace);
    }
    if matches!(role, Role::Pre) && text_is(input, typ, "Whitespace")? {
        fields(input, span, ["type"])?;
        return regex::whitespace(span).map(Inline::Regex);
    }
    if text_is(input, typ, "Split")? {
        return regex::selection(input, span, role).map(Inline::Regex);
    }
    if text_is(input, typ, "ByteLevel")? {
        let f = fields(
            input,
            span,
            ["type", "add_prefix_space", "trim_offsets", "use_regex"],
        )?;
        let value = ByteLevelSettings {
            add_prefix_space: boolean(input, f[1], true)?,
            trim_offsets: boolean(input, f[2], true)?,
            use_regex: boolean(input, f[3], true)?,
        };
        if matches!(role, Role::Pre) && value.use_regex {
            return regex::implicit_byte_level(span, value).map(Inline::Regex);
        }
        Ok(Inline::Byte(value))
    } else if matches!(role, Role::Decode) {
        decode::select(input, span, typ).map(Inline::Decode)
    } else if matches!(role, Role::Pre) && text_is(input, typ, "Digits")? {
        let f = fields(input, span, ["type", "individual_digits"])?;
        Ok(Inline::Digits(boolean(input, f[1], false)?))
    } else if matches!(role, Role::Post) && text_is(input, typ, "TemplateProcessing")? {
        let plan = template::Plan::prepare(input, span)?;
        Ok(Inline::Template {
            span,
            requirements: plan.requirements(),
        })
    } else {
        Err(err(K::ComponentProfile, span.start))
    }
}
#[derive(Debug, Clone, Copy)]
struct Component {
    value: Option<Span>,
    array: Option<Span>,
    count: usize,
    role: Role,
    regex: Option<RegexSelection>,
    template_buffers: usize,
    template_controls: usize,
}
impl Component {
    fn plan(input: &str, value: Option<Span>, role: Role) -> Result<Self, Error> {
        let value = absent(input, value);
        let mut result = Self {
            value,
            array: None,
            count: 0,
            role,
            regex: None,
            template_buffers: 0,
            template_controls: 0,
        };
        let Some(span) = value else {
            // Absent decode constructs no component. The shared fixed-profile
            // source selects ordinary space joining and prices its separators.
            return Ok(result);
        };
        let mut object = Reader::new(input, span).object()?;
        let mut typ = None;
        while let Some((k, v)) = object.next()? {
            if k.is("type") {
                if typ.replace(v).is_some() {
                    return Err(err(K::DuplicateField, k.offset));
                }
            }
        }
        if text_is(input, required(typ, span.start)?, "Sequence")? {
            let member = match role {
                Role::Pre => "pretokenizers",
                Role::Post => "processors",
                Role::Decode => "decoders",
            };
            let f = fields(input, span, ["type", member])?;
            let array = required(f[1], span.start)?;
            let mut a = pre::Items::new(input, array, role)?;
            let mut decoder_components = [crate::decoders::fixed_profile::Component::Other; 5];
            while let Some(item) = a.next()? {
                let value = inline(input, item, role)?;
                result.add_template(value)?;
                if matches!(role, Role::Decode) {
                    let component = decoder_components
                        .get_mut(result.count)
                        .ok_or(err(K::ComponentProfile, item.start))?;
                    *component = match value {
                        Inline::Byte(_) => crate::decoders::fixed_profile::Component::ByteLevel,
                        Inline::Decode(value) => value.component(),
                        _ => return Err(err(K::ComponentProfile, item.start)),
                    };
                }
                result.count = add(result.count, 1)?;
            }
            if matches!(role, Role::Decode)
                && crate::decoders::fixed_profile::Profile::sequence(
                    &decoder_components[..result.count],
                )
                .is_none()
            {
                return Err(err(K::ComponentProfile, array.start));
            }
            result.array = Some(array);
        } else {
            let value = inline(input, span, role)?;
            result.add_template(value)?;
            if let Inline::Decode(value) = value {
                if crate::decoders::fixed_profile::Profile::sequence(&[value.component()]).is_none()
                {
                    return Err(err(K::ComponentProfile, span.start));
                }
            }
            if let Inline::Regex(selected) = value {
                result.regex = Some(selected);
            }
        }
        Ok(result)
    }
    fn add_template(&mut self, value: Inline) -> Result<(), Error> {
        if let Inline::Metaspace(value) = value {
            self.template_buffers = add(self.template_buffers, value.bytes())?;
        }
        if let Inline::Decode(value) = value {
            self.template_buffers = add(self.template_buffers, value.bytes())?;
        }
        if let Inline::Template { requirements, .. } = value {
            self.template_buffers = add(self.template_buffers, requirements.buffers)?;
            self.template_controls = add(self.template_controls, requirements.controls)?;
        }
        Ok(())
    }
    fn bytes(&self) -> Result<usize, Error> {
        let layout = match self.role {
            Role::Pre => Layout::array::<PreTokenizerWrapper>(self.count),
            Role::Post => Layout::array::<PostProcessorWrapper>(self.count),
            Role::Decode => Layout::array::<DecoderWrapper>(self.count),
        };
        add(
            layout.map(|l| l.size()).map_err(|_| err(K::Overflow, 0))?,
            self.template_buffers,
        )
    }
}
/// Checked compiler buffers, named controls and the borrowed decoder envelope.
#[derive(Debug, Clone, Copy)]
pub struct TokenizerCompileRequirements {
    buffers: usize,
    controls: usize,
    total: usize,
    decoder_ids: usize,
    decoder_bytes: usize,
    domain_extent: u64,
}
impl TokenizerCompileRequirements {
    /// Dense ID-domain envelope derived from the actual fresh constructor.
    /// Existing model IDs survive; new added IDs start at model population and
    /// advance at most once per raw added entry. Declared added JSON IDs do not
    /// control that assignment. Duplicate/empty/model-overlap entries may leave
    /// an unused tail; this diagnostic allocates no dense storage.
    pub fn domain_extent(&self) -> u64 {
        self.domain_extent
    }
    /// Simultaneous model, added and pipeline destination capacities.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Named compiler, partial, error and return representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked sum of buffers and controls; not an allocation grant.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
    /// Maximum actual model-plus-added ID visits, independent of largest ID.
    pub fn decoder_id_slots(&self) -> usize {
        self.decoder_ids
    }
    /// ByteLevel raw spelling envelope including added-ID shadowing.
    pub fn decoder_piece_bytes(&self) -> usize {
        self.decoder_bytes
    }
}
/// One source borrow, one fresh complete inline-profile tokenizer construction.
///
/// ```compile_fail
/// use tokenizers::TokenizerCompilePlan;
/// fn repeat(plan:TokenizerCompilePlan<'_>){let _=plan.compile();let _=plan.compile();}
/// ```
#[derive(Debug)]
pub struct TokenizerCompilePlan<'a> {
    input: &'a str,
    model: model::Plan<'a>,
    added: AddedVocabularyRecipe<'a>,
    normalizer: normalizer::Plan<'a>,
    pre: Component,
    post: Component,
    decoder: Component,
    regex: RegexState<'a>,
    requirements: TokenizerCompileRequirements,
    encode_special_tokens: bool,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    fail_at: Option<usize>,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    added_failure: Option<usize>,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    template_failure: Option<usize>,
}
impl<'a> TokenizerCompilePlan<'a> {
    /// Borrows and validates one complete root without allocating destinations.
    pub fn prepare_json(bytes: &'a [u8]) -> Result<Self, Error> {
        let input = std::str::from_utf8(bytes).map_err(|e| err(K::InvalidUtf8, e.valid_up_to()))?;
        let mut r = Reader::new(
            input,
            Span {
                start: 0,
                end: input.len(),
            },
        );
        let root = r.value()?;
        r.finish()?;
        let f = fields(
            input,
            root,
            [
                "version",
                "truncation",
                "padding",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
                "model",
            ],
        )?;
        if let Some(version) = f[0] {
            if !text_is(input, version, "1.0")? {
                return Err(err(K::Version, version.start));
            }
        }
        for span in [f[1], f[2]].iter().flatten() {
            if &input[span.start..span.end] != "null" {
                return Err(err(K::ComponentProfile, span.start));
            }
        }
        // Selected execution components are rejected before model planning/admission.
        let pre = Component::plan(input, f[5], Role::Pre)?;
        let post = Component::plan(input, f[6], Role::Post)?;
        let decoder = Component::plan(input, f[7], Role::Decode)?;
        let regex = RegexState::plan(input, pre)?;
        let normalizer = normalizer::Plan::inspect(input, f[4])?;
        let span = required(f[8], root.start)?;
        let model = model::Plan::prepare(&bytes[span.start..span.end])?;
        let added_bytes = f[3].map_or(b"[]".as_slice(), |s| &bytes[s.start..s.end]);
        let added = AddedVocabularyRecipe::prepare_json(added_bytes, |text| {
            normalizer.pattern_bounds(text).map_err(|_| {
                super::added_vocabulary::AddedVocabularyCompileError {
                    kind: super::added_vocabulary::AddedVocabularyCompileErrorKind::Overflow,
                    offset: text.offset,
                }
            })
        })?;
        let m = model.requirements();
        let a = added.requirements();
        let buffers = add(
            add(
                add(m.buffer_bytes(), a.buffer_bytes())?,
                normalizer.buffer_bytes()?,
            )?,
            add(
                add(add(pre.bytes()?, post.bytes()?)?, decoder.bytes()?)?,
                regex.buffer_bytes(),
            )?,
        )?;
        let control_parts = [
            decode::control_bytes()?,
            normalizer.control_bytes()?,
            pre::controls()?,
            pre.template_controls,
            post.template_controls,
            decoder.template_controls,
            regex
                .required_bytes()?
                .checked_sub(regex.buffer_bytes())
                .ok_or_else(|| err(K::Overflow, 0))?,
            m.control_bytes(),
            a.control_bytes(),
            size_of::<Self>(),
            size_of::<(Self, bool)>(),
            size_of::<Result<Self, Error>>(),
            size_of::<TokenizerCompileRequirements>(),
            size_of::<Component>(),
            size_of::<[Component; 3]>(),
            size_of::<Result<Inline, Error>>(),
            size_of::<Result<[Option<Span>; 9], Error>>(),
            size_of::<Result<[Option<Span>; 4], Error>>(),
            size_of::<Result<[Option<Span>; 2], Error>>(),
            size_of::<Inline>(),
            size_of::<Partial>(),
            size_of::<Tokenizer>(),
            size_of::<
                TokenizerImpl<
                    ModelWrapper,
                    NormalizerWrapper,
                    PreTokenizerWrapper,
                    PostProcessorWrapper,
                    DecoderWrapper,
                >,
            >(),
            size_of::<TokenizerCompileFailure>(),
            size_of::<Cause>(),
            size_of::<Error>(),
            size_of::<Result<Tokenizer, TokenizerCompileFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<PostProcessorWrapper, Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<AddedVocabulary, AddedVocabularyCompileFailure>>(),
            size_of::<Option<Span>>() * 9,
            size_of::<json::Object<'_>>(),
            size_of::<json::Array<'_>>(),
            size_of::<Reader<'_>>(),
            json::stack_bytes(),
        ];
        let controls = control_parts
            .iter()
            .try_fold(std::mem::size_of_val(&control_parts), |sum, &n| {
                sum.checked_add(n)
            })
            .ok_or(err(K::Overflow, 0))?;
        let requirements = TokenizerCompileRequirements {
            buffers,
            controls,
            total: add(buffers, controls)?,
            decoder_ids: add(m.vocabulary_slots(), a.raw_slots())?,
            domain_extent: m.id_extent().max(
                u64::try_from(add(m.vocabulary_slots(), a.raw_slots())?)
                    .map_err(|_| err(K::Overflow, 0))?,
            ),
            decoder_bytes: add(
                m.spelling_bytes(),
                a.content_bytes()
                    .checked_mul(2)
                    .ok_or(err(K::Overflow, 0))?,
            )?,
        };
        Ok(Self {
            input,
            model,
            added,
            normalizer,
            pre,
            post,
            decoder,
            regex,
            requirements,
            encode_special_tokens: false,
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            fail_at: None,
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            added_failure: None,
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            template_failure: None,
        })
    }
    /// Selects the actual HF special-token splitting policy before construction.
    /// This fixed option is omitted by HF tokenizer JSON; it allocates no storage
    /// and cannot change or adopt an already constructed source.
    pub fn with_encode_special_tokens(mut self, enabled: bool) -> Self {
        self.encode_special_tokens = enabled;
        self
    }
    /// Returns the sealed source-derived requirements without construction.
    pub fn requirements(&self) -> TokenizerCompileRequirements {
        self.requirements
    }
    /// Development-only capacity overflow on actual destination reserves.
    /// Slots 0/1/2 are pre/post/decoder sequences; 3/4 are literal pattern/content;
    /// slot 5 is the Metaspace replacement spelling.
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    #[doc(hidden)]
    pub fn fail_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 6);
        self.fail_at = Some(stage);
        self
    }
    /// Development-only capacity overflow on the actual selected model reserve.
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    #[doc(hidden)]
    pub fn fail_model_reservation(mut self, stage: usize) -> Self {
        self.model = self.model.fail_reservation(stage);
        self
    }
    /// Development-only overflow of one original normalizer destination reserve.
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    #[doc(hidden)]
    pub fn fail_normalizer_reservation(mut self, stage: usize) -> Self {
        self.normalizer = self.normalizer.fail_reservation(stage);
        self
    }
    /// Development-only capacity overflow on the actual selected added reserve.
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    #[doc(hidden)]
    pub fn fail_added_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 5);
        self.added_failure = Some(stage);
        self
    }
    /// Development-only overflow of one actual template buffer reserve.
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    #[doc(hidden)]
    pub fn fail_template_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 5);
        self.template_failure = Some(stage);
        self
    }
    /// Development-only fixed regex construction target; it cannot replace a source.
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    #[doc(hidden)]
    pub fn fail_regex_construction(
        mut self,
        target: fancy_regex::workspace::construction::ConstructionFailure,
    ) -> Self {
        self.regex.fail(target);
        self
    }
    /// Development-only fixed constructor target in the actual ordered regex inventory.
    #[cfg(all(feature = "fancy-regex", feature = "tokenizer-compiler-test-support"))]
    #[doc(hidden)]
    pub fn fail_regex_construction_at(
        mut self,
        ordinal: usize,
        target: fancy_regex::workspace::construction::ConstructionFailure,
    ) -> Self {
        self.regex.fail_at(ordinal, target);
        self
    }
    /// Consumes the root once and constructs every component fresh.
    pub fn compile(self) -> Result<Tokenizer, TokenizerCompileFailure> {
        let mut partial = Partial::default();
        let fail_at = {
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            {
                self.fail_at
            }
            #[cfg(not(any(test, feature = "tokenizer-compiler-test-support")))]
            {
                None::<usize>
            }
        };
        let template_failure = {
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            {
                self.template_failure
            }
            #[cfg(not(any(test, feature = "tokenizer-compiler-test-support")))]
            {
                None::<usize>
            }
        };
        let result = (|| -> Result<(), Cause> {
            let mut regex = self.regex;
            partial.model = Some(self.model.compile()?);
            partial.normalizer = self.normalizer.fill(&mut partial.normalizer_state)?;
            fill_components(
                self.input,
                self.pre,
                self.post,
                self.decoder,
                &mut regex,
                fail_at,
                template_failure,
                &mut partial,
            )?;
            let plan = self.added.bind(
                partial.model.as_ref().expect("constructed model"),
                partial.normalizer.as_ref(),
            )?;
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            let plan = if let Some(stage) = self.added_failure {
                plan.fail_reservation(stage)
            } else {
                plan
            };
            partial.added = Some(plan.compile().map_err(Cause::Added)?);
            partial
                .added
                .as_mut()
                .expect("constructed added vocabulary")
                .set_encode_special_tokens(self.encode_special_tokens);
            Ok(())
        })();
        if let Err(cause) = result {
            return Err(TokenizerCompileFailure { cause, partial });
        }
        Ok(Tokenizer(TokenizerImpl {
            model: partial.model.take().expect("model"),
            normalizer: partial.normalizer.take(),
            pre_tokenizer: partial.pre.take(),
            post_processor: partial.post.take(),
            decoder: partial.decoder.take(),
            added_vocabulary: partial.added.take().expect("added"),
            truncation: None,
            padding: None,
        }))
    }
}
#[derive(Debug, Default)]
struct Partial {
    model: Option<ModelWrapper>,
    added: Option<AddedVocabulary>,
    normalizer: Option<NormalizerWrapper>,
    normalizer_state: normalizer::State,
    pre: Option<PreTokenizerWrapper>,
    post: Option<PostProcessorWrapper>,
    decoder: Option<DecoderWrapper>,
    pre_items: Vec<PreTokenizerWrapper>,
    post_items: Vec<PostProcessorWrapper>,
    decode_items: Vec<DecoderWrapper>,
    decode_strings: [Vec<u8>; 2],
    pre_literal: Vec<u8>,
}
fn requested(stage: usize, n: usize, fail: Option<usize>) -> usize {
    if fail == Some(stage) {
        usize::MAX
    } else {
        n
    }
}
fn fill_components(
    input: &str,
    pre: Component,
    post: Component,
    decoder: Component,
    regex: &mut RegexState<'_>,
    fail: Option<usize>,
    template_failure: Option<usize>,
    p: &mut Partial,
) -> Result<(), Cause> {
    for component in [pre, post, decoder] {
        let Some(span) = component.value else {
            continue;
        };
        if let Some(array) = component.array {
            match component.role {
                Role::Pre => p
                    .pre_items
                    .try_reserve_exact(requested(0, component.count, fail))?,
                Role::Post => {
                    p.post_items
                        .try_reserve_exact(requested(1, component.count, fail))?
                }
                Role::Decode => {
                    p.decode_items
                        .try_reserve_exact(requested(2, component.count, fail))?
                }
            };
            let actual = match component.role {
                Role::Pre => p.pre_items.capacity(),
                Role::Post => p.post_items.capacity(),
                Role::Decode => p.decode_items.capacity(),
            };
            if actual > component.count {
                return Err(err(K::CapacityExceeded, array.start).into());
            }
            let mut a = pre::Items::new(input, array, component.role)?;
            while let Some(item) = a.next().map_err(Error::from)? {
                let item = inline(input, item, component.role)?;
                match (component.role, item) {
                    (Role::Pre, Inline::Byte(b)) => p.pre_items.push(b.build().into()),
                    (Role::Pre, Inline::Digits(d)) => p.pre_items.push(Digits::new(d).into()),
                    (Role::Pre, Inline::Metaspace(value)) => {
                        let value = value.compile(p, fail)?;
                        p.pre_items.push(value);
                    }
                    (Role::Pre, Inline::Regex(selected)) => {
                        p.pre_items.push(regex.compile(selected)?)
                    }
                    (Role::Post, Inline::Byte(b)) => p.post_items.push(b.build().into()),
                    (Role::Post, Inline::Template { span, .. }) => p
                        .post_items
                        .push(compile_template(input, span, template_failure)?),
                    (Role::Decode, Inline::Byte(b)) => p.decode_items.push(b.build().into()),
                    (Role::Decode, Inline::Decode(value)) => {
                        let value = value.compile(input, fail, p)?;
                        p.decode_items.push(value);
                    }
                    _ => return Err(err(K::ComponentProfile, span.start).into()),
                }
            }
            match component.role {
                Role::Pre => {
                    p.pre = Some(PreSequence::new(std::mem::take(&mut p.pre_items)).into())
                }
                Role::Post => {
                    p.post = Some(PostSequence::new(std::mem::take(&mut p.post_items)).into())
                }
                Role::Decode => {
                    p.decoder =
                        Some(DecodeSequence::new(std::mem::take(&mut p.decode_items)).into())
                }
            }
        } else {
            match (component.role, inline(input, span, component.role)?) {
                (Role::Pre, Inline::Byte(b)) => p.pre = Some(b.build().into()),
                (Role::Pre, Inline::Digits(d)) => p.pre = Some(Digits::new(d).into()),
                (Role::Pre, Inline::Regex(selected)) => p.pre = Some(regex.compile(selected)?),
                (Role::Pre, Inline::Metaspace(value)) => p.pre = Some(value.compile(p, fail)?),
                (Role::Post, Inline::Byte(b)) => p.post = Some(b.build().into()),
                (Role::Post, Inline::Template { span, .. }) => {
                    p.post = Some(compile_template(input, span, template_failure)?)
                }
                (Role::Decode, Inline::Byte(b)) => p.decoder = Some(b.build().into()),
                (Role::Decode, Inline::Decode(value)) => {
                    p.decoder = Some(value.compile(input, fail, p)?)
                }
                _ => return Err(err(K::ComponentProfile, span.start).into()),
            }
        }
    }
    Ok(())
}
fn compile_template(
    input: &str,
    span: Span,
    failure: Option<usize>,
) -> Result<PostProcessorWrapper, Cause> {
    let plan = template::Plan::prepare(input, span)?;
    let value = plan.compile(failure).map_err(Cause::Template)?;
    Ok(PostProcessorWrapper::Template(value))
}
#[derive(Debug)]
enum Cause {
    Source(Error),
    Model(BpeCompileFailure),
    WordLevel(WordLevelCompileFailure),
    Unigram(UnigramCompileFailure),
    Added(AddedVocabularyCompileFailure),
    Template(TemplateCompileFailure),
    Reserve(TryReserveError),
    #[cfg(feature = "fancy-regex")]
    Regex(fancy_regex::workspace::construction::Failure),
}
impl From<Error> for Cause {
    fn from(e: Error) -> Self {
        Self::Source(e)
    }
}
impl From<AddedVocabularyCompileError> for Cause {
    fn from(e: AddedVocabularyCompileError) -> Self {
        Self::Source(e.into())
    }
}
impl From<TryReserveError> for Cause {
    fn from(e: TryReserveError) -> Self {
        Self::Reserve(e)
    }
}
/// Owns every completed component and the actual failing constructor prefix.
#[derive(Debug)]
pub struct TokenizerCompileFailure {
    cause: Cause,
    partial: Partial,
}
impl TokenizerCompileFailure {
    /// Whether a successfully constructed model is retained by this failure.
    pub fn completed_model(&self) -> bool {
        self.partial.model.is_some()
    }
    /// Literal normalizer destinations retained after a partial construction.
    pub fn literal_normalizer_capacities(&self) -> [usize; 2] {
        self.partial.normalizer_state.literal_capacities()
    }
    /// Actual template constructor failure and all partial destination capacities.
    pub fn template_failure(&self) -> Option<&TemplateCompileFailure> {
        match &self.cause {
            Cause::Template(error) => Some(error),
            _ => None,
        }
    }
    /// Borrows a fixed root construction diagnostic, when present.
    pub fn source_error(&self) -> Option<&Error> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            _ => None,
        }
    }
    /// Borrows the actual pipeline destination reserve failure, when present.
    pub fn allocation_error(&self) -> Option<&TryReserveError> {
        match &self.cause {
            Cause::Reserve(e) => Some(e),
            _ => None,
        }
    }
}
impl fmt::Display for TokenizerCompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Model(e) => fmt::Display::fmt(e, f),
            Cause::WordLevel(e) => fmt::Display::fmt(e, f),
            Cause::Unigram(e) => fmt::Display::fmt(e, f),
            Cause::Added(e) => fmt::Display::fmt(e, f),
            Cause::Template(e) => fmt::Display::fmt(e, f),
            Cause::Reserve(e) => fmt::Display::fmt(e, f),
            #[cfg(feature = "fancy-regex")]
            Cause::Regex(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for TokenizerCompileFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Source(e) => Some(e),
            Cause::Model(e) => Some(e),
            Cause::WordLevel(e) => Some(e),
            Cause::Unigram(e) => Some(e),
            Cause::Added(e) => Some(e),
            Cause::Template(e) => Some(e),
            Cause::Reserve(e) => Some(e),
            #[cfg(feature = "fancy-regex")]
            Cause::Regex(e) => Some(e),
        }
    }
}
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod template_tests;

impl From<UnigramCompileError> for TokenizerCompileError {
    fn from(e: UnigramCompileError) -> Self {
        Self::Unigram(e)
    }
}

#[cfg(test)]
mod literal_tests;
