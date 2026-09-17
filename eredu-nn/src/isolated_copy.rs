//! Fixed storage-copy sequencing shared by native and metadata realizations.

/// The two mechanisms of an isolated logical array copy.
///
/// Implementations supply row-major materialization followed by an independent
/// data copy. This realization contract grants no admission or storage custody.
pub trait IsolatedCopyMechanism {
    /// Native or metadata value produced by each operation.
    type Value;
    /// The realization's original error type.
    type Error;
    /// Materializes row-major storage, possibly sharing an already suitable input.
    fn contiguous(&self) -> Result<Self::Value, Self::Error>;
    /// Consumes that materialization and independently copies its logical data.
    fn deep_copy(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
}

/// Executes the fixed row-major-materialization then independent-copy program.
///
/// The second operation runs only after the first succeeds. Metadata admission
/// uses its own closed realization; accepting this trait is not an account API.
pub fn isolated_copy<M: IsolatedCopyMechanism>(mechanism: M) -> Result<M::Value, M::Error> {
    let contiguous = mechanism.contiguous()?;
    mechanism.deep_copy(contiguous)
}

/// The existing workspace realization shared by native source tracing and
/// portable paged-state copy projection. Construction adds no source authority.
pub struct WorkspaceIsolatedCopy<'a> {
    source: crate::workspace::WorkspaceTensor,
    context: &'a crate::workspace::WorkspaceContext,
}
impl<'a> WorkspaceIsolatedCopy<'a> {
    /// Uses the actual previously projected value in the same trace context.
    pub fn new(
        source: crate::workspace::WorkspaceTensor,
        context: &'a crate::workspace::WorkspaceContext,
    ) -> Self {
        Self { source, context }
    }
}
impl IsolatedCopyMechanism for WorkspaceIsolatedCopy<'_> {
    type Value = crate::workspace::WorkspaceTensor;
    type Error = crate::Error;
    fn contiguous(&self) -> Result<Self::Value, Self::Error> {
        self.source.contiguous(self.context)
    }
    fn deep_copy(&self, contiguous: Self::Value) -> Result<Self::Value, Self::Error> {
        contiguous.deep_copy(self.context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Mechanism<'a> {
        steps: &'a RefCell<Vec<&'static str>>,
        failure: Option<&'static str>,
    }
    impl IsolatedCopyMechanism for Mechanism<'_> {
        type Value = u32;
        type Error = &'static str;
        fn contiguous(&self) -> Result<u32, Self::Error> {
            self.steps.borrow_mut().push("contiguous");
            if self.failure == Some("contiguous") {
                Err("contiguous")
            } else {
                Ok(17)
            }
        }
        fn deep_copy(&self, value: u32) -> Result<u32, Self::Error> {
            self.steps.borrow_mut().push("deep_copy");
            assert_eq!(value, 17);
            if self.failure == Some("deep_copy") {
                Err("deep_copy")
            } else {
                Ok(23)
            }
        }
    }

    #[test]
    fn fixed_program_stops_at_each_original_failure_and_forwards_the_intermediate() {
        for failure in [Some("contiguous"), Some("deep_copy"), None] {
            let steps = RefCell::new(Vec::new());
            let result = isolated_copy(Mechanism {
                steps: &steps,
                failure,
            });
            assert_eq!(result, failure.map_or(Ok(23), Err));
            assert_eq!(
                *steps.borrow(),
                if failure == Some("contiguous") {
                    vec!["contiguous"]
                } else {
                    vec!["contiguous", "deep_copy"]
                }
            );
        }
    }
}
