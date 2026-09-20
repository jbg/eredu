//! One original aggregate tokenizer residence; public managed operations remain closed.
mod encode;
pub use encode::{OriginalEncodedTokenIds, OriginalTokenizerEncodeError};
mod input;
pub use input::{OriginalTokenizerInput, OriginalTokenizerInputError};
mod text_source_error;
pub use text_source_error::{OriginalTextSourceError, OriginalTokenizerSourceError};
mod source_budget;
pub use source_budget::{OriginalTextSourceBudget, OriginalTextSourceBudgetError};

use super::loaded_decode_source::Allowance;
use super::{WorkingMemoryError, WorkingMemoryPool};
use eredu_core::{BackendFailure, TokenFilter};
use eredu_text::tokenizer_storage::{
    PreparedTokenizer, TokenizerConstructionFailure, TokenizerPlan, InputPrefixPlan, InputPrefixFailure, TokenizerSourceError,
};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::size_of,
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
struct Payload {
    model: PreparedTokenizer,
    domain: Option<TokenFilter>,
    input_prefix_root: Option<OriginalTokenizer>,
    allowance: Allowance,
}

/// Closed shared source retaining its complete original compiler allowance.
/// No Arc, Weak, source extraction/refill or independent-byte constructor escapes.
/// This retains construction headroom for stock HF and its fresh decode program.
/// Dependency internals use estimates; caller JSON, chat rendering and encoding
/// operation allocations have separate reservations.
///
/// ```compile_fail
/// # use eredu_runtime::working_memory::OriginalTokenizer;
/// fn escape(model:OriginalTokenizer) -> &'static str { model.spelling(90).unwrap() }
/// ```
/// ```compile_fail
/// # use eredu_runtime::working_memory::OriginalTokenizer;
/// fn raw(model:OriginalTokenizer) { let _ = model.source(); }
/// ```
/// ```compile_fail
/// # use eredu_runtime::working_memory::{OriginalTokenizer,WorkingMemoryPool};
/// fn adopt(pool:&WorkingMemoryPool,model:OriginalTokenizer) { let _ = pool.compile_tokenizer(model); }
/// ```
pub struct OriginalTokenizer(Option<Arc<Payload>>);
impl OriginalTokenizer {
    /// Remove input prefixes using a separately admitted tokenizer copy.
    /// Identity removal aliases this exact source. A changed source reserves a
    /// full construction estimate and retains the original identity and payer.
    pub fn input_prefix_normalized_source(&self) -> Result<Self, OriginalTokenizerPrefixError> {
        let plan = self.payload().model.input_prefix_plan().map_err(|error| OriginalTokenizerError {
            cause: Cause::Source(error), settlement: None, _completed: None, domain: None,
            input_prefix_root: Some(self.clone()), allowance: None,
        })?;
        let Some(plan) = plan else { return Ok(self.clone()); };
        let extent = match self.payload().domain.as_ref() { Some(TokenFilter::Allowed(mask)) => Some(mask.len()), None => None, _ => unreachable!("original canonical domain is a dense mask") };
        self.pool().compile_tokenizer_inner(PrefixSourcePlan { plan, extent }, || {}, false, Some(self))
    }

    pub(super) fn tokenization_is_canonical(&self) -> bool {
        self.payload().model.tokenization_is_canonical()
    }

    /// Compares this originally constructed source with the selected loaded
    /// tokenizer without exposing or cloning its HF aggregate. No funding is granted.
    pub fn matches_configuration(&self, selected: &eredu_text::tokenizer::Tokenizer) -> bool {
        self.payload().model.matches_configuration(selected)
    }

    pub(super) fn matches_semantic_root(&self, original: &Self) -> bool {
        self.same_source(original)
            || self
                .payload()
                .input_prefix_root
                .as_ref()
                .is_some_and(|root| root.same_source(original))
    }

    /// Borrows the canonical token domain selected before this source's original
    /// C admission. No shared-filter registry owner or mutable mask escapes.
    /// Decoder-only sources return None and cannot be upgraded after construction.
    pub fn generation_domain(&self) -> Option<&TokenFilter> {
        self.payload().domain.as_ref()
    }
    fn payload(&self) -> &Payload {
        self.0.as_deref().expect("live original tokenizer")
    }
    /// Model-plus-added ID population length, retaining duplicate visits; no map or cache access.
    pub fn token_count(&self) -> usize {
        self.payload().model.ids().count()
    }
    /// Looks up a spelling without allocating or mutating the model.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.payload().model.token_id(token)
    }
    /// Exact added-token identity from the same immutable compiled source.
    /// This fact grants no encoding, allocation or structural-profile authority.
    pub fn added_token_id(&self, token: &str) -> Option<u32> {
        self.payload().model.added_token_id(token)
    }
    /// Borrows one canonical spelling for no longer than this owner borrow.
    pub fn spelling(&self, id: u32) -> Option<&str> {
        self.payload().model.spelling(id)
    }
    /// Borrows canonical forward IDs without traversing their sparse extent.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.payload().model.ids()
    }
    pub(super) fn pool(&self) -> &WorkingMemoryPool {
        self.payload().allowance.pool()
    }
    /// Borrows the same immutable decode program; no owner extraction or replacement.
    pub fn decode_source(&self) -> &eredu_text::decoder_storage::PreparedDecodeSource {
        self.payload().model.decode_source()
    }
    /// Borrows exact lexical token bytes from this immutable source. The plan
    /// creates no trie or tokenizer and grants no independent source authority.
    pub fn token_byte_vocabulary(
        &self,
    ) -> Result<
        eredu_text::token_bytes::PackedTokenBytePlan<'_>,
        eredu_text::token_bytes::TokenByteError,
    > {
        self.payload().model.token_byte_vocabulary()
    }
    /// Borrows exact trie-input byte geometry from this original source, with
    /// special markers retained. It grants no tokenizer-environment authority.
    pub fn token_trie_vocabulary(
        &self,
    ) -> Result<
        eredu_text::token_bytes::PackedTokenBytePlan<'_>,
        eredu_text::token_bytes::TokenByteError,
    > {
        self.payload().model.token_trie_vocabulary()
    }
    pub(super) fn token_trie_plan<'a>(
        &'a self,
        info: &eredu_text::token_trie_storage::TokRxInfo,
        eos: &'a [u32],
    ) -> Result<
        eredu_text::token_trie_storage::TokenTriePlan<'a>,
        eredu_text::token_trie_storage::TokenTrieSourceError,
    > {
        eredu_text::token_trie_storage::TokenTriePlan::prepare(&self.payload().model, info, eos)
    }
    /// Tests source special membership without allocating.
    pub fn is_special(&self, token: &str) -> bool {
        self.payload().model.is_special(token)
    }
    /// Full original allowance held through final payload/control retirement.
    pub fn original_bytes(&self) -> u64 {
        self.payload().allowance.bytes()
    }
    /// Exact source identity, with no allocation or accounting change.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live source"),
            other.0.as_ref().expect("live source"),
        )
    }
    /// Checks the original domain without minting a request or another hold.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.payload().allowance.pool().same_domain(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl fmt::Debug for OriginalTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalTokenizer")
            .field("token_count", &self.token_count())
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}
impl Clone for OriginalTokenizer {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live source"))))
    }
}
impl Drop for OriginalTokenizer {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // Every strong owner participates; no external Weak exists. Payload
            // returns after the Arc allocation is gone, with allowance last.
            drop(Arc::into_inner(owner));
        }
    }
}

#[derive(Debug)]
enum Cause<C = TokenizerConstructionFailure> {
    Admission(WorkingMemoryError),
    Compilation(C),
    Source(TokenizerSourceError),
    Domain(TryReserveError),
}
/// Closed terminal failure retaining the compiler prefix and original allowance.
/// Admission rejection has no allowance. No partial/guard extraction or retry exists.
pub struct OriginalTokenizerError<C = TokenizerConstructionFailure> {
    cause: Cause<C>,
    settlement: Option<WorkingMemoryError>,
    _completed: Option<PreparedTokenizer>,
    domain: Option<TokenFilter>,
    input_prefix_root: Option<OriginalTokenizer>,
    allowance: Option<Allowance>,
}
/// Prefix normalization uses the same terminal custody with only its reachable
/// projection failures; full model compiler storage cannot enter this path.
pub type OriginalTokenizerPrefixError = OriginalTokenizerError<InputPrefixFailure>;
impl<C: std::error::Error + Send + Sync + 'static> OriginalTokenizerError<C> {
    /// Retains the complete compiler failure directly in the neutral envelope.
    pub fn into_backend_failure(self) -> BackendFailure {
        let kind = match self.accounting_failure() {
            Some(WorkingMemoryError::Poisoned | WorkingMemoryError::IdentityMismatch) => {
                eredu_core::BackendFailureKind::InvalidSession
            }
            Some(WorkingMemoryError::UnknownBound) => eredu_core::BackendFailureKind::Unsupported,
            _ => eredu_core::BackendFailureKind::ResourceExhausted,
        };
        BackendFailure::new(kind, self)
    }
    fn rejected_with_root(cause: WorkingMemoryError, root: Option<&OriginalTokenizer>) -> Self {
        Self {
            cause: Cause::Admission(cause),
            settlement: None,
            _completed: None,
            domain: None,
            input_prefix_root: root.cloned(),
            allowance: None,
        }
    }
    /// Retained original bytes; zero means rejection preceded compilation.
    pub fn retained_bytes(&self) -> u64 {
        self.allowance.as_ref().map_or(0, |a| a.bytes())
    }
    /// Actual compiler failure from an admitted reserve attempt.
    pub fn compiler_failure(&self) -> Option<&C> {
        match &self.cause {
            Cause::Compilation(error) => Some(error),
            _ => None,
        }
    }
    /// Actual failed reserve of the C-owned validity destination, if selected.
    pub fn domain_failure(&self) -> Option<&TryReserveError> {
        match &self.cause {
            Cause::Domain(error) => Some(error),
            _ => None,
        }
    }
    /// Actual retained mask capacity, including any failed completed prefix.
    pub fn domain_capacity(&self) -> usize {
        match &self.domain {
            Some(TokenFilter::Allowed(mask)) => mask.capacity(),
            _ => 0,
        }
    }
    /// Original admission or terminal accounting failure, if present.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.settlement.as_ref().or_else(|| match &self.cause {
            Cause::Admission(error) => Some(error),
            _ => None,
        })
    }
}
impl<C: std::error::Error + Send + Sync + 'static> fmt::Debug for OriginalTokenizerError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalTokenizerError")
            .field("cause", &self.cause)
            .field("settlement", &self.settlement)
            .field("completed", &self._completed.is_some())
            .field("retains_input_prefix_root", &self.input_prefix_root.is_some())
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}
impl<C: fmt::Display> fmt::Display for OriginalTokenizerError<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Admission(error) => fmt::Display::fmt(error, f),
            Cause::Compilation(error) => fmt::Display::fmt(error, f),
            Cause::Source(error) => fmt::Display::fmt(error, f),
            Cause::Domain(error) => fmt::Display::fmt(error, f),
        }
    }
}
impl<C: std::error::Error + Send + Sync + 'static> std::error::Error for OriginalTokenizerError<C> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Admission(error) => Some(error),
            Cause::Compilation(error) => Some(error),
            Cause::Source(error) => Some(error),
            Cause::Domain(error) => Some(error),
        }
    }
}

// Actual producer types select their own failure transport while sharing all
// admission, validity-mask publication and retirement mechanics.
trait SourcePlan: Sized {
    type Failure: std::error::Error + Send + Sync + 'static;
    fn required_bytes(&self) -> usize;
    fn generation_controls() -> Option<usize>;
    fn domain_extent(&self) -> Option<usize>;
    fn compile(self) -> Result<PreparedTokenizer, Self::Failure>;
}
impl SourcePlan for TokenizerPlan<'_> {
    type Failure = TokenizerConstructionFailure;
    fn generation_controls() -> Option<usize> { OriginalTokenizerSourceError::tokenizer_controls() }
    fn required_bytes(&self) -> usize { self.requirements().required_bytes() }
    fn domain_extent(&self) -> Option<usize> { self.generation_domain_extent() }
    fn compile(self) -> Result<PreparedTokenizer, Self::Failure> { TokenizerPlan::compile(self) }
}
struct PrefixSourcePlan<'a> { plan: InputPrefixPlan<'a>, extent: Option<usize> }
impl SourcePlan for PrefixSourcePlan<'_> {
    type Failure = InputPrefixFailure;
    fn generation_controls() -> Option<usize> { Some(0) }
    fn required_bytes(&self) -> usize { self.plan.requirements().required_bytes() }
    fn domain_extent(&self) -> Option<usize> { self.extent }
    fn compile(self) -> Result<PreparedTokenizer, Self::Failure> { self.plan.compile() }
}

impl WorkingMemoryPool {
    /// Source-derived construction estimate plus closed owner/error controls.
    /// This query grants no budget and takes no ownership of the borrowed plan.
    pub fn tokenizer_required_bytes(plan: &TokenizerPlan<'_>) -> Result<u64, WorkingMemoryError> {
        Self::tokenizer_construction_required_bytes::<TokenizerPlan<'_>>(plan.requirements().required_bytes(), plan.generation_domain_extent())
    }
    fn tokenizer_construction_required_bytes<P: SourcePlan>(required: usize, domain_extent: Option<usize>) -> Result<u64, WorkingMemoryError> {
        let arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .map_err(|_| WorkingMemoryError::Overflow)?
            .0
            .pad_to_align()
            .size();
        let controls = [
            arc,
            size_of::<Payload>(),
            size_of::<Option<Payload>>(),
            size_of::<Arc<Payload>>(),
            size_of::<Option<Arc<Payload>>>(),
            size_of::<OriginalTokenizer>(),
            size_of::<Option<OriginalTokenizer>>(),
            size_of::<Option<&OriginalTokenizer>>(),
            size_of::<(&OriginalTokenizer, P)>(),
            size_of::<P>(),
            size_of::<Result<Option<InputPrefixPlan<'_>>, TokenizerSourceError>>(),
            size_of::<Result<(), BackendFailure>>(),
            size_of::<Allowance>(),
            size_of::<Result<Allowance, WorkingMemoryError>>(),
            size_of::<Cause<P::Failure>>(),
            size_of::<Option<WorkingMemoryError>>(),
            size_of::<Option<PreparedTokenizer>>(),
            size_of::<Option<TokenFilter>>(), // temporary mask before final owner
            size_of::<Vec<bool>>(),           // actual reserve target before publication
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<(), Cause<P::Failure>>>(),
            size_of::<Option<usize>>(), // selected domain capacity
            size_of::<usize>(),         // actual target reserve count
            size_of::<usize>(),         // actual consistent generation-domain extent
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<OriginalTokenizerError<P::Failure>>(),
            size_of::<Result<OriginalTokenizer, OriginalTokenizerError<P::Failure>>>(),
            size_of::<Result<OriginalTokenizer, BackendFailure>>(),
            BackendFailure::source_retention_peak_bytes::<OriginalTokenizerError<P::Failure>>()
                .ok_or(WorkingMemoryError::Overflow)?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        let controls = if domain_extent.is_some() {
            controls
                .checked_add(
                    P::generation_controls()
                        .ok_or(WorkingMemoryError::Overflow)?,
                )
                .ok_or(WorkingMemoryError::Overflow)?
        } else {
            controls
        };
        required
            .checked_add(
                Layout::array::<bool>(domain_extent.unwrap_or(0))
                    .map_err(|_| WorkingMemoryError::Overflow)?
                    .size(),
            )
            .and_then(|bytes| bytes.checked_add(controls))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Admits before the first compiler reserve and consumes the actual plan once.
    /// Terminal compilation ends only its active count; every byte remains held
    /// through source/error retirement. No second quote, registration or HF claim.
    pub fn compile_tokenizer(
        &self,
        plan: TokenizerPlan<'_>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerError> {
        self.compile_tokenizer_with(plan, || {})
    }
    fn compile_tokenizer_with(
        &self,
        plan: TokenizerPlan<'_>,
        after_admission: impl FnOnce(),
    ) -> Result<OriginalTokenizer, OriginalTokenizerError> {
        self.compile_tokenizer_inner(plan, after_admission, false, None)
    }
    fn compile_tokenizer_inner<P: SourcePlan>(
        &self,
        plan: P,
        after_admission: impl FnOnce(),
        fail_domain_reserve: bool,
        root: Option<&OriginalTokenizer>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerError<P::Failure>> {
        let rejected = |cause| OriginalTokenizerError::rejected_with_root(cause, root);
        let bytes = Self::tokenizer_construction_required_bytes::<P>(plan.required_bytes(), plan.domain_extent()).map_err(rejected)?;
        let mut allowance = self.admit_source_compiler(bytes).map_err(rejected)?;
        // Private test hook runs with the actual original guard installed; the
        // public entry supplies only a zero-sized no-op.
        after_admission();
        let domain_extent = plan.domain_extent();
        // Inner compiler locals/prefix retire before allowance on unwind.
        let compiled = plan.compile();
        match compiled {
            Err(error) => {
                let settlement = allowance.end_compilation().err();
                Err(OriginalTokenizerError {
                    cause: Cause::Compilation(error),
                    settlement,
                    _completed: None,
                    domain: None,
                    input_prefix_root: root.cloned(),
                    allowance: Some(allowance),
                })
            }
            Ok(source) => {
                // These real completed parts remain below the original guard on
                // error/unwind. One actual reserve, never a substitute buffer.
                let mut domain = domain_extent.map(|_| TokenFilter::Allowed(Vec::new()));
                let filled = if root.is_some_and(|original| {
                    !source.is_input_prefix_derivative_of(&original.payload().model)
                }) {
                    Err(Cause::Admission(WorkingMemoryError::IdentityMismatch))
                } else {
                    fill_domain(&source, domain.as_mut(), domain_extent, fail_domain_reserve)
                };
                if let Err(cause) = filled {
                    let settlement = allowance.end_compilation().err();
                    return Err(OriginalTokenizerError {
                        cause,
                        settlement,
                        _completed: Some(source),
                        domain,
                        input_prefix_root: root.cloned(),
                        allowance: Some(allowance),
                    });
                }
                // Allocate the final source control while compilation is still
                // active. Only then may another host preparation begin.
                let mut owner = Arc::new(Payload {
                    model: source,
                    domain,
                    input_prefix_root: root.cloned(),
                    allowance,
                });
                let settlement = Arc::get_mut(&mut owner)
                    .expect("unexposed source")
                    .allowance
                    .end_compilation();
                match settlement {
                    Ok(()) => Ok(OriginalTokenizer(Some(owner))),
                    Err(error) => {
                        let Payload {
                            model: source,
                            domain,
                            input_prefix_root,
                            allowance,
                        } = Arc::into_inner(owner).expect("unexposed source");
                        Err(OriginalTokenizerError {
                            cause: Cause::Admission(error),
                            settlement: None,
                            _completed: Some(source),
                            domain,
                            input_prefix_root,
                            allowance: Some(allowance),
                        })
                    }
                }
            }
        }
    }
}
fn fill_domain<C>(
    source: &PreparedTokenizer,
    domain: Option<&mut TokenFilter>,
    extent: Option<usize>,
    fail_reserve: bool,
) -> Result<(), Cause<C>> {
    let Some(TokenFilter::Allowed(mask)) = domain else {
        return Ok(());
    };
    let extent = extent.expect("selected domain has source-derived extent");
    let requested = if fail_reserve { usize::MAX } else { extent };
    mask.try_reserve_exact(requested).map_err(Cause::Domain)?;
    if mask.capacity() > extent {
        return Err(Cause::Admission(WorkingMemoryError::Overflow));
    }
    mask.resize(extent, false);
    let mut actual_extent = 0usize;
    for id in source.ids() {
        if source.spelling(id).and_then(|text| source.token_id(text)) == Some(id) {
            let slot = mask
                .get_mut(id as usize)
                .ok_or(Cause::Admission(WorkingMemoryError::Overflow))?;
            *slot = true;
            actual_extent = actual_extent.max(id as usize + 1);
        }
    }
    // The source plan admits the maximum fresh added-token population before
    // construction. Existing-token overlaps may leave unused trailing capacity.
    // Logical validity ends at the last actual consistent ID, just as the
    // ordinary vocabulary filter does; retain the original paid allocation.
    mask.truncate(actual_extent);
    Ok(())
}
#[cfg(test)]
mod tests;

/// Concrete original aggregate construction; tokenizer policy stays in text/facade.
/// This prerequisite does not activate managed public encoding or chat.
pub trait OriginalTokenizerBackend: eredu_core::TextGenerationBackend {
    /// Checks canonical text IDs in an already completed original prepared input.
    /// Implementations borrow authenticated host source slots, never evaluate native
    /// values or promote ordinary arrays. The selected request still performs its
    /// own complete source/session comparison before execution. Default is refusal.
    fn validate_original_prepared_input_domain(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _prompt: &Self::Prompt,
        _source: &OriginalTokenizer,
    ) -> Result<(), eredu_core::TokenInputRejection> {
        Err(eredu_core::TokenInputRejection::Unsupported)
    }

    /// Starts source-bound semantic funding before facade preparation.
    /// Default refusal performs no source work or detached custody conversion.
    fn prepare_semantic_source(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _source: &OriginalTokenizer,
        _capacity: u64,
    ) -> Result<super::PreparedSemanticSource, eredu_core::SpeculativeOutputError> {
        Err(eredu_core::SpeculativeOutputError::Storage(
            "prepared semantic source is unavailable",
        ))
    }

    /// Checks the exact selected execution and original tokenizer pool before
    /// semantic preparation is associated with a model input. No work is granted.
    fn validate_semantic_source(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _preparation: &super::PreparedSemanticSource,
    ) -> Result<(), eredu_core::TokenInputRejection> {
        Err(eredu_core::TokenInputRejection::Unsupported)
    }

    /// Compiles an exact borrowed ID sequence through original I and completed B
    /// under the selected execution's closed preparation. Implementations must
    /// authenticate its pool/execution and tokenizer domain before materialization;
    /// encoded callers separately authenticate their actual E source before
    /// lending these IDs. All wrapper/error producers use the same host account.
    /// The default performs no input work and returns a fixed neutral refusal.
    fn prepare_semantic_prompt(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _preparation: &super::PreparedSemanticSource,
        _input: &eredu_core::TokenIdsInputPlan<'_>,
        _chunk: Option<std::num::NonZeroU64>,
    ) -> Result<Self::Prompt, BackendFailure> {
        Err(eredu_core::TokenInputRejection::Unsupported.into_backend_failure())
    }

    /// Retains the request ceiling before stop/tokenizer preparation. The default
    /// refuses without source work; no caller may replace this with a late check.
    fn prepare_original_text_source_budget(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _source: &OriginalTokenizer,
        _capacity: u64,
    ) -> Result<OriginalTextSourceBudget, OriginalTextSourceError> {
        Err(eredu_core::TokenInputRejection::Unsupported.into())
    }
    /// Read-only same-pool validation before the facade constructs request sources
    /// or encodes input. The default rejects without allocation or callbacks.
    fn validate_original_tokenizer_source(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _source: &OriginalTokenizer,
    ) -> Result<(), BackendFailure> {
        Err(eredu_core::TokenInputRejection::Unsupported.into_backend_failure())
    }
    /// Original-text stop operation with a by-value failure, including rejection
    /// before S admission. The default invokes no source/compiler callback.
    fn compile_original_text_stop_source(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _plan: eredu_text::stop_storage::StopCompilePlan<'_>,
    ) -> Result<super::OriginalStopSource, OriginalTextSourceError> {
        Err(eredu_core::TokenInputRejection::Unsupported.into())
    }
    /// Original-text encoding with the actual by-value E/source failure. The
    /// default allocates no wrapper and performs no input encoding.
    fn encode_original_text_ids(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _source: &OriginalTokenizer,
        _input: &str,
        _add_special_tokens: bool,
    ) -> Result<OriginalEncodedTokenIds, OriginalTextSourceError> {
        Err(eredu_core::TokenInputRejection::Unsupported.into())
    }
    /// Consumes the actual file or retained configuration source to construct a
    /// fresh generation-enabled C. The default rejects without source work; a
    /// decoder-only C is never promoted.
    fn compile_original_tokenizer_source_for_generation(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _input: OriginalTokenizerInput<'_>,
    ) -> Result<OriginalTokenizer, OriginalTokenizerSourceError> {
        Err(eredu_core::TokenInputRejection::Unsupported.into())
    }
    /// Encodes the checked identity profile under the runtime's original E allowance.
    /// Default rejection performs no tokenizer operation or destination allocation.
    fn encode_original_tokenizer_ids(
        _runtime: &eredu_core::ModelRuntime<Self>,
        _source: &OriginalTokenizer,
        _input: &str,
        _add_special_tokens: bool,
    ) -> Result<OriginalEncodedTokenIds, BackendFailure> {
        Err(BackendFailure::new(
            eredu_core::BackendFailureKind::Unsupported,
            WorkingMemoryError::UnknownBound,
        ))
    }
    /// Reads one exact prepared file and constructs a fresh aggregate in the
    /// same account. The compatibility default rejects without reading.
    fn compile_original_tokenizer_file(
        runtime: &eredu_core::ModelRuntime<Self>,
        read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
    ) -> Result<OriginalTokenizer, BackendFailure> {
        let _ = (runtime, read);
        Err(BackendFailure::new(
            eredu_core::BackendFailureKind::Unsupported,
            WorkingMemoryError::UnknownBound,
        ))
    }
    /// Compiles the borrowed root in the actual runtime account before publication.
    fn compile_original_tokenizer(
        runtime: &eredu_core::ModelRuntime<Self>,
        plan: TokenizerPlan<'_>,
    ) -> Result<OriginalTokenizer, BackendFailure>;
}
