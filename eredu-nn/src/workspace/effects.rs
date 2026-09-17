//! Borrowed storage declarations and the shared ordinary/fixed validation rules.
use super::{Error, WorkspaceLayoutError, WorkspaceLayoutView, WorkspaceOutputStorage};

/// One result's storage rule, borrowing ordered alias candidates when present.
/// Duplicated candidates remain part of the declaration and are not reordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceOutputStorageView<'a> {
    /// An independent allocation, including mechanism-specific padding.
    Allocate(u64),
    /// The complete backing of one invocation input.
    AliasInput(usize),
    /// The backing of a strictly earlier result of the same invocation.
    AliasOutput(usize),
    /// A possible independent allocation and every possible input alias.
    AllocateOrAliasInputs {
        /// Capacity of the possible independent allocation.
        bytes: u64,
        /// Input ordinals in the mechanism's original order.
        inputs: &'a [usize],
    },
}

impl WorkspaceOutputStorage {
    /// Borrows a declaration without copying its possible-alias list.
    pub fn as_view(&self) -> WorkspaceOutputStorageView<'_> {
        match self {
            Self::Allocate(bytes) => WorkspaceOutputStorageView::Allocate(*bytes),
            Self::AliasInput(input) => WorkspaceOutputStorageView::AliasInput(*input),
            Self::AliasOutput(output) => WorkspaceOutputStorageView::AliasOutput(*output),
            Self::AllocateOrAliasInputs { bytes, inputs } => {
                WorkspaceOutputStorageView::AllocateOrAliasInputs {
                    bytes: *bytes,
                    inputs,
                }
            }
        }
    }
}

/// A storage declaration is invalid. This value owns no diagnostic allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WorkspaceEffectError {
    /// Output cardinality or the tensor assumption is missing.
    #[error("workspace mechanism returned an incomplete storage declaration")]
    Incomplete,
    /// The ordinary represented-layout calculation failed.
    #[error(transparent)]
    Layout(#[from] WorkspaceLayoutError),
    /// The possible allocation cannot contain the represented result.
    #[error("workspace output bound is smaller than represented storage")]
    UndersizedAllocation,
    /// A definite input alias is outside the invocation's inputs.
    #[error("workspace output aliases a nonexistent input")]
    InputAlias,
    /// An output alias refers to this output or a later one.
    #[error("workspace output must alias an earlier output")]
    OutputAlias,
    /// A possible input alias is outside the invocation's inputs.
    #[error("workspace output may alias a nonexistent input")]
    PossibleInputAlias,
    /// The host assumption contains no non-whitespace characters.
    #[error("workspace host bound requires implementation assumptions")]
    HostAssumptions,
}

impl WorkspaceEffectError {
    pub(super) fn into_ordinary(self) -> Error {
        // Existing errors have no added source wrapper. Preserve the original
        // layout overflow path and the existing backend diagnostic messages.
        match self {
            Self::Layout(cause) => cause.into_ordinary(),
            other => Error::backend(other),
        }
    }
}

/// Checks the declaration header before inspecting individual effects.
/// Tensor assumptions preserve the ordinary nonempty rule (without trimming).
pub fn validate_workspace_tensor_declaration(
    actual_outputs: usize,
    expected_outputs: usize,
    assumptions: &str,
) -> Result<(), WorkspaceEffectError> {
    if actual_outputs != expected_outputs || assumptions.is_empty() {
        return Err(WorkspaceEffectError::Incomplete);
    }
    Ok(())
}

/// Validates one result in invocation order without allocating.
/// For a possible allocation, byte geometry precedes alias-range validation.
pub fn validate_workspace_output_storage(
    effect: WorkspaceOutputStorageView<'_>,
    output: WorkspaceLayoutView<'_>,
    output_index: usize,
    input_count: usize,
) -> Result<(), WorkspaceEffectError> {
    use WorkspaceOutputStorageView as Storage;
    match effect {
        Storage::Allocate(bytes) | Storage::AllocateOrAliasInputs { bytes, .. }
            if bytes < output.bytes()? =>
        {
            Err(WorkspaceEffectError::UndersizedAllocation)
        }
        Storage::AliasInput(input) if input >= input_count => Err(WorkspaceEffectError::InputAlias),
        Storage::AliasOutput(prior) if prior >= output_index => {
            Err(WorkspaceEffectError::OutputAlias)
        }
        Storage::AllocateOrAliasInputs { inputs, .. }
            if inputs.iter().any(|input| *input >= input_count) =>
        {
            Err(WorkspaceEffectError::PossibleInputAlias)
        }
        _ => Ok(()),
    }
}

/// Checks the separate host assumption using the ordinary Unicode trim rule.
pub fn validate_workspace_host_assumptions(assumptions: &str) -> Result<(), WorkspaceEffectError> {
    if assumptions.trim().is_empty() {
        Err(WorkspaceEffectError::HostAssumptions)
    } else {
        Ok(())
    }
}
