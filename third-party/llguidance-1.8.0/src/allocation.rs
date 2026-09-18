//! Borrowed dependency policies backed by the same retained compiler account.
pub(crate) struct CompilerAllocation<'a>(pub(crate) &'a derivre::ParserAllocationFunding);
impl serde_json::allocation::Allocation for CompilerAllocation<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), serde_json::allocation::AllocationError> {
        self.0.reserve(bytes).map_err(|_| serde_json::allocation::AllocationError::Refused)
    }
}
impl referencing::allocation::Allocation for CompilerAllocation<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), referencing::allocation::AllocationError> {
        self.0.reserve(bytes).map_err(|_| referencing::allocation::AllocationError::Refused)
    }
    fn is_enforced(&self) -> bool { self.0.is_enforced() }
}

pub(crate) trait CollectFunded<T, E>: Iterator<Item = Result<T, E>> + Sized {
    fn collect_with_funding(self, funding: &derivre::ParserAllocationFunding) -> Result<Vec<T>, E>
    where E: From<derivre::ParserStorageError> {
        let mut values = Vec::new();
        for value in self { funding.try_push(&mut values, value?)?; }
        Ok(values)
    }
}
impl<I, T, E> CollectFunded<T, E> for I where I: Iterator<Item = Result<T, E>> {}
