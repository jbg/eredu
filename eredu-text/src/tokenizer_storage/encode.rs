//! ID-only operation adapters over the same privately retained HF aggregate.
use super::PreparedTokenizer;
use std::{fmt, mem::size_of};
pub use tokenizers::{EncodeIdsError, NormalizationBuffer};

/// Non-Clone source/input plan; no HF source or configuration is exposed.
/// ```compile_fail
/// use eredu_text::tokenizer_storage::EncodeIdsPlan;
/// fn repeat(plan: EncodeIdsPlan<'_>) { let _ = plan.encode(); let _ = plan.encode(); }
/// ```
#[derive(Debug)]
pub struct EncodeIdsPlan<'a> {
    inner: tokenizers::EncodeIdsPlan<'a>,
    required: usize,
}
impl<'a> EncodeIdsPlan<'a> {
    /// Checks the actual immutable profile before any operation reserve.
    pub fn prepare(
        source: &'a PreparedTokenizer,
        input: &'a str,
        add_special_tokens: bool,
    ) -> Result<Self, EncodeIdsError> {
        let inner =
            tokenizers::EncodeIdsPlan::prepare(&source.tokenizer, input, add_special_tokens)?;
        let required = [
            inner.requirements().required_bytes(),
            size_of::<Self>(),
            size_of::<Result<Self, EncodeIdsError>>(),
            size_of::<EncodedTokenIds>(),
            size_of::<EncodeIdsFailure>(),
            size_of::<Result<EncodedTokenIds, EncodeIdsFailure>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(EncodeIdsError::Overflow)?;
        Ok(Self { inner, required })
    }
    /// Complete checked HF and adapter destination/control requirement.
    pub fn required_bytes(&self) -> usize {
        self.required
    }
    /// Actual symbol, merge-heap and ID capacity bounds.
    pub fn capacities(&self) -> [usize; 3] {
        let r = self.inner.requirements();
        [r.symbol_capacity(), r.merge_capacity(), r.id_capacity()]
    }
    /// Source-derived NFC decomposition/recomposition/text capacities, zero without NFC.
    pub fn normalization_capacities(&self) -> [usize; 3] {
        self.inner.requirements().normalization_capacities()
    }
    /// Actual capacity of the single mapped split destination.
    pub fn mapped_capacity(&self) -> usize {
        self.inner.requirements().mapped_capacity()
    }
    /// Actual immutable regex source delegate population; zero for identity encoding.
    pub fn regex_delegate_count(&self) -> usize {
        self.inner.requirements().regex_delegate_count()
    }
    /// Executes the same HF matcher and merge worker once.
    pub fn encode(self) -> Result<EncodedTokenIds, EncodeIdsFailure> {
        self.inner
            .encode()
            .map(EncodedTokenIds)
            .map_err(EncodeIdsFailure)
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Fail one actual NFC target reserve, preserving its source-bound plan.
    pub fn fail_normalization_reservation(
        mut self,
        target: NormalizationBuffer,
    ) -> Result<Self, EncodeIdsError> {
        self.inner = self.inner.fail_normalization_reservation(target)?;
        Ok(self)
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Select one actual bound regex reserve failure before original E admission.
    pub fn fail_regex_reservation(
        mut self,
        target: super::RegexWorkspaceFailure,
    ) -> Result<Self, EncodeIdsError> {
        self.inner = self.inner.fail_regex_reservation(target)?;
        Ok(self)
    }
    #[cfg(feature = "tokenizer-compiler-test-support")]
    #[doc(hidden)]
    /// Development-only overflow of one actual target reserve, never a surrogate allocation.
    pub fn fail_reservation(mut self, stage: usize) -> Self {
        self.inner = self.inner.fail_reservation(stage);
        self
    }
}
/// Completed move-only destinations. Only ID borrows and capacity diagnostics escape.
/// ```compile_fail
/// use eredu_text::tokenizer_storage::EncodedTokenIds;
/// fn extract(ids: EncodedTokenIds) { let _ = ids.into_vec(); }
/// ```
#[derive(Debug)]
pub struct EncodedTokenIds(tokenizers::EncodeIdsOutput);
impl EncodedTokenIds {
    /// Actual NFC capacities retained through success or the precise failed prefix.
    pub fn normalization_capacities(&self) -> [usize; 3] {
        self.0.normalization_capacities()
    }

    /// Borrows the completed IDs for the lifetime of this destination owner.
    pub fn ids(&self) -> &[u32] {
        self.0.ids()
    }
    /// Actual retained mapped split capacity; regex workspaces have already retired.
    pub fn mapped_capacity(&self) -> usize {
        self.0.mapped_capacity()
    }
    /// Actual retained symbol, heap and ID capacities.
    pub fn capacities(&self) -> [usize; 3] {
        self.0.capacities()
    }
}
/// Closed real failure retaining token/mapped destinations and the real cause.
/// Borrowed regex workspace prefixes retire under E before this owned error returns.
#[derive(Debug)]
pub struct EncodeIdsFailure(tokenizers::EncodeIdsFailure);
impl EncodeIdsFailure {
    /// Actual NFC capacities retained through success or the precise failed prefix.
    pub fn normalization_capacities(&self) -> [usize; 3] {
        self.0.normalization_capacities()
    }

    /// Fixed profile/model cause or actual TryReserveError.
    pub fn cause(&self) -> &EncodeIdsError {
        self.0.cause()
    }
    /// Actual mapped destination retained after other regex storage has retired.
    pub fn mapped_capacity(&self) -> usize {
        self.0.mapped_capacity()
    }
    /// Actual retained destination capacities.
    pub fn capacities(&self) -> [usize; 3] {
        self.0.capacities()
    }
    /// Produced prefix length; no successful partial output or retry is exposed.
    pub fn partial_id_count(&self) -> usize {
        self.0.partial_id_count()
    }
}
impl fmt::Display for EncodeIdsFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for EncodeIdsFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}
