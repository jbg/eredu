use super::Value;
use crate::allocation::{Allocation, AllocationError, Allocator, Unenforced};

impl Clone for Value {
    fn clone(&self) -> Self {
        self.try_clone_with_allocations(&Unenforced).expect("ordinary JSON value clone")
    }
}

impl Value {
    /// Deep-copy the actual selected representation through the same ordinary
    /// producer. A refusal stops before the destination allocation is reached.
    pub fn try_clone_with_allocations(&self, funding: &dyn Allocation) -> Result<Self, AllocationError> {
        let allocator = Allocator::new(funding);
        Ok(match self {
            Self::Null => Self::Null,
            Self::Bool(value) => Self::Bool(*value),
            Self::Number(value) => Self::Number(value.try_clone_with_allocations(funding)?),
            Self::String(value) => Self::String(allocator.copy_string(value)?),
            Self::Array(values) => {
                let mut result = alloc::vec::Vec::new();
                allocator.grow(&mut result, values.len())?;
                for value in values { result.push(value.try_clone_with_allocations(funding)?); }
                Self::Array(result)
            }
            Self::Object(values) => Self::Object(values.try_clone_with_allocations(funding)?),
        })
    }
}
