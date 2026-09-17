//! Existing substring search and short-circuit sequence membership.
#![forbid(unsafe_code)]

pub(crate) fn text_contains(container: &str, needle: &str) -> bool {
    container.contains(needle)
}

/// Storage supplies equality and its own error domain. Each element and the
/// closure result retire before the next element, including early success.
pub(crate) fn sequence_contains<I: Iterator, E>(
    values: I,
    mut equal: impl FnMut(I::Item) -> Result<bool, E>,
) -> Result<bool, E> {
    for value in values {
        if equal(value)? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn text_control_bytes() -> usize {
    use std::mem::size_of;
    // The standard string-pattern searcher is contained in this named stable
    // iterator type; include it alongside the actual borrowed operands/result.
    // This is a concrete control census, not a native whole-thread stack claim.
    size_of::<std::str::MatchIndices<'_, &str>>()
        + size_of::<(&str, &str)>()
        + size_of::<Option<(usize, &str)>>()
        + size_of::<bool>()
}
