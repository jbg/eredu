//! Selection only: evaluation, fallback ownership and undefined policy stay
//! with the caller. Query truth lazily in the same ordinary branch.
#![forbid(unsafe_code)]
pub(crate) fn use_fallback<E>(
    undefined: bool,
    lax: bool,
    truth: impl FnOnce() -> Result<bool, E>,
) -> Result<bool, E> {
    if undefined {
        Ok(true)
    } else if lax {
        truth().map(|value| !value)
    } else {
        Ok(false)
    }
}
