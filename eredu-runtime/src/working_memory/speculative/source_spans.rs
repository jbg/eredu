//! A descriptive span table keeps the payer that constructed its backing.
use super::{HostSourceConstructionFacts, WorkingMemoryError};
use eredu_nn::workspace::{
    HostMetadataFunding, WorkspaceMetadataAllocation, WorkspaceMetadataError,
};
use std::mem::{size_of, size_of_val};

/// Fixed-capacity source facts in the actual equation order. Construction pays
/// the table before allocation; moving it into an invocation preserves that
/// same charge. This description grants no source or native execution authority.
#[derive(Debug)]
pub struct SpeculativeHostSourceSpans {
    values: Vec<Option<HostSourceConstructionFacts>>,
    capacity: usize,
    // The backing and its values retire before their original metadata payer.
    _funding: HostMetadataFunding,
}
impl SpeculativeHostSourceSpans {
    /// Construct through the shared metadata allocator. As with its NN worker,
    /// the caller must retain funding through any escaping construction error.
    pub fn new(capacity: usize, funding: &HostMetadataFunding) -> Result<Self, eredu_nn::Error> {
        let append = [
            size_of::<(&mut Self, Option<HostSourceConstructionFacts>)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ];
        let per_row = append
            .into_iter()
            .try_fold(size_of_val(&append), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let fixed = [
            size_of::<Self>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            size_of::<(usize, &HostMetadataFunding)>(),
            per_row
                .checked_mul(capacity)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        ];
        funding
            .reserve_metadata(
                fixed
                    .into_iter()
                    .try_fold(size_of_val(&fixed), usize::checked_add)
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )
            .map_err(|cause| eredu_nn::Error::from(WorkspaceMetadataError::from(cause)))?;
        Ok(Self {
            values: funding.metadata_vec(capacity)?,
            capacity,
            _funding: funding.clone(),
        })
    }
    /// Append one actual row without allocating or replenishing capacity.
    pub fn push(
        &mut self,
        facts: Option<HostSourceConstructionFacts>,
    ) -> Result<(), WorkingMemoryError> {
        if self.values.len() >= self.capacity {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.values.push(facts);
        Ok(())
    }
    /// Borrow the description without detaching its allocation custody.
    pub fn as_slice(&self) -> &[Option<HostSourceConstructionFacts>] {
        &self.values
    }
}
