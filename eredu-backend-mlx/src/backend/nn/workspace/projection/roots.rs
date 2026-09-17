//! Shared counted source-root construction for native copy metadata workers.
use super::*;
use eredu_nn::workspace::WorkspaceMetadataError;

impl<'a> ExistingArrayProjection<'a> {
    /// Counts the actual borrowed source traversal before allocating its root
    /// table. The same immutable visitor then supplies its exact initial values.
    /// A copy worker reserves later destination roots independently.
    pub(crate) fn project_roots(
        &mut self,
        mut visit: impl FnMut(&mut dyn FnMut(&'a Array)),
    ) -> Result<Vec<WorkspaceTensor>, Error> {
        let mut count = Some(0usize);
        visit(&mut |_| count = count.and_then(|n| n.checked_add(1)));
        let count = count.ok_or(WorkspaceMetadataError::Overflow)?;
        self.reserve_sources(count)
            .map_err(|cause| cause.in_context(self.context))?;
        let mut roots = self.context.metadata_vec(count)?;
        let mut error = None;
        visit(&mut |array| {
            if error.is_some() {
                return;
            }
            if roots.len() == count {
                error =
                    Some(self.context.metadata_error(format_args!(
                        "source root traversal changed after counting"
                    )));
                return;
            }
            match self.project(array) {
                Ok(value) => roots.push(value),
                Err(cause) => error = Some(cause),
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        if roots.len() != count {
            return Err(self
                .context
                .metadata_error(format_args!("source root traversal changed after counting")));
        }
        Ok(roots)
    }
}
