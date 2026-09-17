//! Shared borrowed prefix/suffix traversal for ordinary and paid controllers.
/// The proposed canonical history does not extend the durable committed prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("constrained sampler history diverges from its committed logical prefix")]
pub struct ControllerHistoryError;
/// Validated borrowed suffix. No clone, parser, allocation or token validation occurs.
#[derive(Debug, Clone, Copy)]
pub struct ControllerHistorySuffix<'a>(&'a [u32]);
impl<'a> ControllerHistorySuffix<'a> {
    /// Validates the same committed-prefix relation before any provisional copy.
    pub fn new(committed: &[u32], proposed: &'a [u32]) -> Result<Self, ControllerHistoryError> {
        if !proposed.starts_with(committed) {
            return Err(ControllerHistoryError);
        }
        Ok(Self(&proposed[committed.len()..]))
    }
    /// Visits each canonical suffix token in order, stopping at the first failure.
    pub fn visit<E>(self, mut commit: impl FnMut(u32) -> Result<(), E>) -> Result<(), E> {
        for &token in self.0 {
            commit(token)?;
        }
        Ok(())
    }
}
