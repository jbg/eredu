//! One-attempt ID destinations for checked identity and compiled regex pipelines.
mod normalization;
mod post;
mod regex;
use self::post::PostPlan;
use self::regex::RegexPlan;
use super::{ModelWrapper, Tokenizer};
use crate::models::bpe::BpeScratch;
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of};
pub use unicode_normalization_alignments::workspace::Buffer as NormalizationBuffer;

/// Fixed profile/geometry diagnostics and the actual failed target reserve.
#[derive(Debug)]
pub enum EncodeIdsError {
    /// Normalization, pre/postprocessing, padding or truncation needs a later operation plan.
    PipelineProfile,
    /// Added normalization/word/strip modes or legacy matching need a later plan.
    AddedProfile,
    /// A cache, non-packed model, dropout, nonempty prefix/suffix or fallback needs a later plan.
    ModelProfile,
    /// Checked destination geometry overflowed.
    Overflow,
    /// The selected model's configured unknown spelling has no vocabulary ID.
    MissingUnknown,
    /// The original attempt on an actual destination failed.
    Reserve(TryReserveError),
    /// Actual partial normalization destinations and their real reserve error.
    NormalizationPreparation(
        unicode_normalization_alignments::workspace::PrepareFailure,
    ),
    /// Invalid source subrange; the original ID path derives ranges from its matcher.
    NormalizationRange(
        unicode_normalization_alignments::workspace::InvalidRange,
    ),
    /// Exact immutable source workspace geometry/profile rejection.
    #[cfg(feature = "fancy-regex")]
    RegexPlan(fancy_regex::workspace::PlanError),
    /// The real reserve cause after all borrowed regex partial storage retires.
    #[cfg(feature = "fancy-regex")]
    RegexPreparation(fancy_regex::workspace::RetiredPrepareError),
    /// The unchanged terminal search error; no partial successful IDs are published.
    #[cfg(feature = "fancy-regex")]
    RegexRuntime(fancy_regex::Error),
}
impl fmt::Display for EncodeIdsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PipelineProfile => f.write_str("ID encoding requires a supported immutable pipeline and single-input template"),
            Self::AddedProfile => f.write_str("identity ID encoding requires packed literal added tokens without word or strip modes"),
            Self::ModelProfile => f.write_str("identity ID encoding requires packed uncached BPE without dropout, nonempty affixes or byte fallback"),
            Self::Overflow => f.write_str("identity ID encoding layout overflow"),
            Self::MissingUnknown => f.write_str("configured unknown token is absent from the model vocabulary"),
            Self::Reserve(error) => fmt::Display::fmt(error, f),
            Self::NormalizationPreparation(error) => fmt::Display::fmt(error, f),
            Self::NormalizationRange(error) => fmt::Display::fmt(error, f),
            #[cfg(feature = "fancy-regex")]
            Self::RegexPlan(error) => fmt::Display::fmt(error, f),
            #[cfg(feature = "fancy-regex")]
            Self::RegexPreparation(error) => fmt::Display::fmt(error, f),
            #[cfg(feature = "fancy-regex")]
            Self::RegexRuntime(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for EncodeIdsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Reserve(error) => Some(error),
            Self::NormalizationPreparation(error) => Some(error),
            Self::NormalizationRange(error) => Some(error),
            #[cfg(feature = "fancy-regex")]
            Self::RegexPlan(error) => Some(error),
            #[cfg(feature = "fancy-regex")]
            Self::RegexPreparation(error) => Some(error),
            #[cfg(feature = "fancy-regex")]
            Self::RegexRuntime(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(feature = "fancy-regex")]
impl From<fancy_regex::Error> for EncodeIdsError {
    fn from(error: fancy_regex::Error) -> Self {
        Self::RegexRuntime(error)
    }
}

/// Actual destination layouts plus named plan/return/error representations.
#[derive(Debug, Clone, Copy)]
pub struct EncodeIdsRequirements {
    symbols: usize,
    merges: usize,
    ids: usize,
    mapped: usize,
    regex: usize,
    regex_delegates: usize,
    normalization: [usize; 3],
    buffers: usize,
    controls: usize,
    total: usize,
}
impl EncodeIdsRequirements {
    /// Maximum initial symbols in one reused word destination.
    pub fn symbol_capacity(&self) -> usize {
        self.symbols
    }
    /// Bound on all insertions into the existing no-dropout merge heap.
    pub fn merge_capacity(&self) -> usize {
        self.merges
    }
    /// Maximum committed output IDs; no token-spelling or Encoding vectors.
    pub fn id_capacity(&self) -> usize {
        self.ids
    }
    /// Capacity of the single reused mapped UTF-8 split buffer; zero for identity.
    pub fn mapped_capacity(&self) -> usize {
        self.mapped
    }
    /// Combined actual regex workspace and its named visitation/retirement controls.
    pub fn regex_workspace_bytes(&self) -> usize {
        self.regex
    }
    /// Number of actual source-bound delegates prepared once for this operation.
    pub fn regex_delegate_count(&self) -> usize {
        self.regex_delegates
    }
    /// Source-derived decomposition/recomposition/text capacities, zero without NFC.
    pub fn normalization_capacities(&self) -> [usize; 3] {
        self.normalization
    }
    /// Sum of actual Symbol, Merge, u32 and mapped-u8 array layouts.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Named concrete controls, including retained partial and result overlaps.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked complete operation requirements, without account authority.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}

/// One immutable source/input borrow. Profile checks and geometry allocate nothing.
pub struct EncodeIdsPlan<'a> {
    source: &'a Tokenizer,
    input: &'a str,
    regex: RegexPlan<'a>,
    post: PostPlan<'a>,
    matching: normalization::Plan<'a>,
    requirements: EncodeIdsRequirements,
    #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
    failure: Option<usize>,
}
impl<'a> EncodeIdsPlan<'a> {
    /// Inspects the actual source and selected single-template special policy.
    /// Embedded special matching remains a separate immutable source setting.
    pub fn prepare(
        source: &'a Tokenizer,
        input: &'a str,
        add_special_tokens: bool,
    ) -> Result<Self, EncodeIdsError> {
        let post = PostPlan::inspect(source, add_special_tokens)?;
        let regex = RegexPlan::inspect(source)?;
        let matching = normalization::Plan::inspect(source, input)?;
        match &source.model {
            ModelWrapper::BPE(model)
                if model.has_identity_packed_profile() => {}
            _ => return Err(EncodeIdsError::ModelProfile),
        }
        let symbols = matching.text_bound();
        let merges = symbols
            .saturating_sub(1)
            .checked_mul(3)
            .ok_or(EncodeIdsError::Overflow)?;
        let ids = symbols
            .checked_add(post.special_ids())
            .ok_or(EncodeIdsError::Overflow)?;
        let mapped = regex.mapped_capacity(symbols)?;
        let regex_bytes = regex.required_bytes()?;
        let regex_delegates = regex.delegate_count();
        let normalization = matching.capacities();
        let buffers = BpeScratch::buffer_bytes(symbols, merges)
            .and_then(|bytes| {
                bytes.checked_add(Layout::array::<u32>(ids).ok()?.size())
            })
            .and_then(|bytes| {
                bytes.checked_add(Layout::array::<u8>(mapped).ok()?.size())
            })
            .and_then(|bytes| bytes.checked_add(matching.buffer_bytes()))
            .ok_or(EncodeIdsError::Overflow)?;
        let controls = [
            PostPlan::control_bytes().ok_or(EncodeIdsError::Overflow)?,
            size_of::<InputEncoder<'_, '_>>(),
            size_of::<PostInputEncoder<'_, '_>>(),
            size_of::<&mut InputEncoder<'_, '_>>(),
            size_of::<&mut PostInputEncoder<'_, '_>>(),
            size_of::<Self>(),
            size_of::<Result<Self, EncodeIdsError>>(),
            size_of::<EncodeIdsRequirements>(),
            size_of::<[usize; 3]>(),
            size_of::<String>(),
            size_of::<BpeScratch>(),
            size_of::<EncodeIdsOutput>(),
            size_of::<EncodeIdsFailure>(),
            size_of::<EncodeIdsError>(),
            size_of::<Result<(), EncodeIdsError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<EncodeIdsOutput, EncodeIdsFailure>>(),
            matching.control_bytes()?,
        ]
        .iter()
        .copied()
        .try_fold(0usize, usize::checked_add)
        .ok_or(EncodeIdsError::Overflow)?;
        let total = buffers
            .checked_add(controls)
            .and_then(|bytes| bytes.checked_add(regex_bytes))
            .ok_or(EncodeIdsError::Overflow)?;
        Ok(Self {
            source,
            input,
            regex,
            post,
            matching,
            requirements: EncodeIdsRequirements {
                symbols,
                merges,
                ids,
                mapped,
                regex: regex_bytes,
                regex_delegates,
                normalization,
                buffers,
                controls,
                total,
            },
            #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
            failure: None,
        })
    }
    /// Exact checked requirements. No grant or reserve occurs here.
    pub fn requirements(&self) -> EncodeIdsRequirements {
        self.requirements
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only capacity overflow of the selected actual target reserve.
    pub fn fail_reservation(mut self, stage: usize) -> Self {
        assert!(stage < 4);
        self.failure = Some(stage);
        self
    }
    /// Development-only actual outer/nested regex reserve selection.
    #[cfg(all(
        feature = "fancy-regex",
        feature = "tokenizer-compiler-test-support"
    ))]
    #[doc(hidden)]
    pub fn fail_regex_reservation(
        mut self,
        failure: fancy_regex::workspace::PrepareFailure,
    ) -> Result<Self, EncodeIdsError> {
        self.regex.fail(failure)?;
        Ok(self)
    }
    /// Development-only overflow of one actual NFC target reserve.
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    pub fn fail_normalization_reservation(
        mut self,
        buffer: NormalizationBuffer,
    ) -> Result<Self, EncodeIdsError> {
        self.matching = self.matching.fail(buffer)?;
        Ok(self)
    }
    fn capacity(&self, stage: usize, actual: usize) -> usize {
        #[cfg(any(test, feature = "tokenizer-compiler-test-support"))]
        if self.failure == Some(stage) {
            return usize::MAX;
        }
        let _ = stage;
        actual
    }
    /// Attempts each real destination once, then runs the shared match/merge workers.
    /// Failure owns every actual partial destination; this plan cannot be replayed.
    pub fn encode(self) -> Result<EncodeIdsOutput, EncodeIdsFailure> {
        let mut output = EncodeIdsOutput {
            scratch: BpeScratch::new(),
            ids: Vec::new(),
            mapped: String::new(),
            normalization: None,
        };
        let result = (|| {
            output
                .scratch
                .reserve_symbols(self.capacity(0, self.requirements.symbols))
                .map_err(EncodeIdsError::Reserve)?;
            output
                .scratch
                .reserve_merges(self.capacity(1, self.requirements.merges))
                .map_err(EncodeIdsError::Reserve)?;
            output
                .ids
                .try_reserve_exact(self.capacity(2, self.requirements.ids))
                .map_err(EncodeIdsError::Reserve)?;
            let mapped_requested = self.capacity(3, self.requirements.mapped);
            if mapped_requested != 0 {
                output
                    .mapped
                    .try_reserve_exact(mapped_requested)
                    .map_err(EncodeIdsError::Reserve)?;
            }
            let input = self.input;
            let source = self.source;
            let post = self.post;
            let regex_plan = self.regex;
            let mut matching = self.matching.prepare()?;
            let result = (|| {
                let mut regex = regex_plan.prepare()?;
                let mapped_reserved = output.mapped.capacity();
                let reserved = output.capacities();
                let ModelWrapper::BPE(model) = &source.model else {
                    unreachable!("checked immutable model")
                };
                let result = post.encode(&mut PostInputEncoder {
                    input: InputEncoder {
                        output: &mut output,
                        regex: &mut regex,
                        model,
                        input,
                    },
                    matching: &mut matching,
                });
                // The same E guard surrounds this lexical retirement. No borrowed
                // workspace escapes into the owned successful result or failure.
                drop(regex);
                debug_assert_eq!(
                    output.mapped.capacity(),
                    mapped_reserved,
                    "encoding grew mapped storage"
                );
                debug_assert_eq!(
                    output.capacities(),
                    reserved,
                    "encoding grew a destination"
                );
                result
            })();
            // All later errors first retire the exact NFC destinations into the
            // same owned output. No input/cursor borrow escapes E.
            output.normalization = matching.retire();
            result
        })();
        match result {
            Ok(()) => Ok(output),
            Err(cause) => Err(EncodeIdsFailure {
                cause,
                partial: output,
            }),
        }
    }
}
// Named synchronous callback state; requirements include both states and the
// sole mutable reference captured by each shared visitor callback.
struct InputEncoder<'a, 's> {
    output: &'a mut EncodeIdsOutput,
    regex: &'a mut regex::Runtime<'s>,
    model: &'a crate::models::bpe::BPE,
    input: &'a str,
}
impl InputEncoder<'_, '_> {
    fn span(
        &mut self,
        id: Option<u32>,
        (start, end): crate::Offsets,
    ) -> Result<(), EncodeIdsError> {
        if let Some(id) = id {
            self.output.ids.push(id);
            Ok(())
        } else {
            self.regex.encode_span(
                self.output,
                self.model,
                &self.input[start..end],
            )
        }
    }
}
struct PostInputEncoder<'a, 's> {
    input: InputEncoder<'a, 's>,
    matching: &'a mut normalization::Runtime<'s>,
}
impl PostInputEncoder<'_, '_> {
    fn piece(
        &mut self,
        special: Option<&[u32]>,
    ) -> Result<(), EncodeIdsError> {
        if let Some(ids) = special {
            self.input.output.ids.extend_from_slice(ids);
            Ok(())
        } else {
            self.matching.encode(&mut self.input)
        }
    }
}
impl fmt::Debug for EncodeIdsPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodeIdsPlan")
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}
/// Move-only operation destinations, exposing only the completed IDs and capacities.
#[derive(Debug)]
pub struct EncodeIdsOutput {
    scratch: BpeScratch,
    ids: Vec<u32>,
    mapped: String,
    normalization:
        Option<unicode_normalization_alignments::workspace::Retired>,
}
impl EncodeIdsOutput {
    /// Borrows the completed IDs; no destination ownership escapes.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    /// Actual retired normalization capacities; source/cursor borrows have ended.
    pub fn normalization_capacities(&self) -> [usize; 3] {
        self.normalization
            .as_ref()
            .map_or([0; 3], |n| n.capacities())
    }
    /// Actual retained mapped split capacity, including a partial reserve prefix.
    pub fn mapped_capacity(&self) -> usize {
        self.mapped.capacity()
    }
    /// Actual symbol, heap and ID capacities, including retained partial reserves.
    pub fn capacities(&self) -> [usize; 3] {
        let (symbols, merges) = self.scratch.capacities();
        [symbols, merges, self.ids.capacity()]
    }
}
/// Actual cause and every partial destination from the single attempt.
#[derive(Debug)]
pub struct EncodeIdsFailure {
    cause: EncodeIdsError,
    partial: EncodeIdsOutput,
}
impl EncodeIdsFailure {
    /// All actual normalization destinations, including a failed preparation prefix.
    pub fn normalization_capacities(&self) -> [usize; 3] {
        match &self.cause {
            EncodeIdsError::NormalizationPreparation(error) => {
                error.capacities()
            }
            _ => self.partial.normalization_capacities(),
        }
    }

    /// Borrows the fixed or actual reserve failure.
    pub fn cause(&self) -> &EncodeIdsError {
        &self.cause
    }
    /// Actual capacities retained in this failure.
    pub fn capacities(&self) -> [usize; 3] {
        self.partial.capacities()
    }
    /// Actual mapped destination capacity retained alongside the real failure.
    pub fn mapped_capacity(&self) -> usize {
        self.partial.mapped_capacity()
    }
    /// Number of IDs produced before a terminal model error; not a successful result.
    pub fn partial_id_count(&self) -> usize {
        self.partial.ids.len()
    }
}
impl fmt::Display for EncodeIdsFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for EncodeIdsFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
#[cfg(test)]
mod tests;

#[cfg(all(
    test,
    feature = "fancy-regex",
    feature = "tokenizer-compiler-test-support"
))]
mod regex_tests;

#[cfg(all(
    test,
    feature = "fancy-regex",
    feature = "tokenizer-compiler-test-support"
))]
mod nfc_tests;
