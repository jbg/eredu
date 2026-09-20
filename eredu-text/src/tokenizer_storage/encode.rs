//! Admitted encoding through stock HF, retaining its complete Encoding output.
use super::PreparedTokenizer;
use std::fmt;

/// Input planning or upstream encoding error.
#[derive(Debug, thiserror::Error)]
pub enum EncodeIdsError {
    /// Admission estimate exceeds the host address space.
    #[error("tokenizer encoding estimate overflow")]
    Overflow,
    /// Original stock tokenizer error.
    #[error("tokenizer encoding failed: {0}")]
    Upstream(#[source] tokenizers::Error),
}

/// One borrowed source and input, consumed by one admitted operation.
/// ```compile_fail
/// use eredu_text::tokenizer_storage::EncodeIdsPlan;
/// fn repeat(plan: EncodeIdsPlan<'_>) { let _ = plan.encode(); let _ = plan.encode(); }
/// ```
#[derive(Debug)]
pub struct EncodeIdsPlan<'a> {
    source: &'a PreparedTokenizer,
    input: &'a str,
    add_special_tokens: bool,
    required: usize,
}
impl<'a> EncodeIdsPlan<'a> {
    /// Estimates upstream scratch and output from the actual UTF-8 input length.
    /// This does not allocate and is not a dependency memory ceiling.
    pub fn prepare(
        source: &'a PreparedTokenizer,
        input: &'a str,
        add_special_tokens: bool,
    ) -> Result<Self, EncodeIdsError> {
        let required = source
            .root()
            .estimate
            .encoding(input.len())
            .ok_or(EncodeIdsError::Overflow)?;
        Ok(Self {
            source,
            input,
            add_special_tokens,
            required,
        })
    }
    /// Estimated operation footprint, retained with the output or failure.
    pub fn required_bytes(&self) -> usize {
        self.required
    }
    /// Calls the upstream encoding path once. Offsets, masks and other upstream
    /// Encoding fields remain owned with the IDs; no partial IDs are published.
    pub fn encode(self) -> Result<EncodedTokenIds, EncodeIdsFailure> {
        self.source
            .input_view()
            .encode(self.input, self.add_special_tokens)
            .map(EncodedTokenIds)
            .map_err(|cause| EncodeIdsFailure(EncodeIdsError::Upstream(cause)))
    }
}
/// Move-only upstream Encoding owner with read-only token IDs.
#[derive(Debug)]
pub struct EncodedTokenIds(tokenizers::Encoding);
impl EncodedTokenIds {
    /// Borrows completed IDs without transferring their allocation.
    pub fn ids(&self) -> &[u32] {
        self.0.get_ids()
    }
}
/// Original upstream failure. Upstream retires its own temporary workspaces
/// before returning; the runtime retains the operation reservation with this error.
#[derive(Debug)]
pub struct EncodeIdsFailure(EncodeIdsError);
impl EncodeIdsFailure {
    /// Original error, including the upstream cause chain.
    pub fn cause(&self) -> &EncodeIdsError {
        &self.0
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
