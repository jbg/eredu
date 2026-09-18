//! Borrowed prospective source storage. The caller retains the actual account.
pub use regex_syntax::allocation::{Allocation, AllocationError, Allocator, Unenforced};
use std::{borrow::Cow, fmt};

/// A translation failure without an allocated diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranslationError {
    /// Invalid or unsupported ECMA spelling.
    Syntax,
    /// Refusal before source storage growth.
    Allocation(AllocationError),
}
impl From<AllocationError> for TranslationError {
    fn from(error: AllocationError) -> Self {
        Self::Allocation(error)
    }
}
impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax => f.write_str("invalid or unsupported ECMA regular expression"),
            Self::Allocation(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for TranslationError {}

pub(crate) fn ast_error(error: regex_syntax::ast::Error) -> TranslationError {
    match error.kind() {
        regex_syntax::ast::ErrorKind::Allocation(error) => TranslationError::Allocation(*error),
        _ => TranslationError::Syntax,
    }
}

pub(crate) fn replace(
    pattern: &mut Cow<'_, str>,
    start: usize,
    end: usize,
    replacement: &str,
    allocation: Allocator<'_>,
) -> Result<(), AllocationError> {
    let length = pattern
        .len()
        .checked_sub(end - start)
        .and_then(|len| len.checked_add(replacement.len()))
        .ok_or(AllocationError::SizeOverflow)?;
    match pattern {
        Cow::Borrowed(source) => {
            allocation.reserve(length)?;
            let mut output = String::new();
            output
                .try_reserve_exact(length)
                .map_err(|_| AllocationError::HostAllocation)?;
            output.push_str(&source[..start]);
            output.push_str(replacement);
            output.push_str(&source[end..]);
            *pattern = Cow::Owned(output);
        }
        Cow::Owned(buffer) => {
            if length > buffer.capacity() {
                let capacity = buffer
                    .capacity()
                    .checked_mul(2)
                    .ok_or(AllocationError::SizeOverflow)?
                    .max(length);
                allocation.reserve(capacity)?;
                buffer
                    .try_reserve_exact(capacity - buffer.len())
                    .map_err(|_| AllocationError::HostAllocation)?;
            }
            buffer.replace_range(start..end, replacement);
        }
    }
    Ok(())
}
