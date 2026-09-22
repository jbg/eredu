//! Explicit metadata destination retained by an actual layered operation.
use eredu_nn::workspace::WorkspaceContext;

/// Loans an existing metadata destination and preserves its typed errors.
/// This supplies no tensor, completion, or native submission authority.
#[derive(Debug)]
pub struct LayeredMetadata<E> {
    error: fn(eredu_nn::Error) -> E,
    // Retire the destination after every caller control in this wrapper.
    context: WorkspaceContext,
}
impl<E> LayeredMetadata<E> {
    /// Shares the exact existing Context; creates no allowance or new account.
    pub fn new(context: &WorkspaceContext, error: fn(eredu_nn::Error) -> E) -> Self {
        Self {
            error,
            context: context.clone(),
        }
    }

    pub(crate) fn require_funding<T>(&self) -> Result<(), E> {
        if self.context.metadata_funding().is_none() {
            return Err((self.error)(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        self.context
            .charge_metadata(std::mem::size_of::<(Self, T, Result<(), E>)>())
            .map_err(|cause| (self.error)(cause.into()))
    }
}

pub(super) struct Destination<E>(pub Option<LayeredMetadata<E>>);
impl<E> Destination<E> {
    pub fn context(&self) -> Option<&WorkspaceContext> {
        self.0.as_ref().map(|source| &source.context)
    }
    pub fn map(&self, error: eredu_nn::Error) -> E {
        (self
            .0
            .as_ref()
            .expect("only checked destinations fail")
            .error)(error)
    }
    pub fn controls<T>(&self) -> Result<(), E> {
        match self.context() {
            Some(context) => context
                .charge_metadata(std::mem::size_of::<T>())
                .map_err(|cause| self.map(cause.into())),
            None => Ok(()),
        }
    }
    pub fn text(&self, args: std::fmt::Arguments<'_>) -> Result<String, E> {
        match self.context() {
            Some(context) => context
                .metadata_string(args)
                .map_err(|cause| self.map(cause)),
            None => Ok(args.to_string()),
        }
    }
    pub fn vector<T>(&self, count: usize) -> Result<Vec<T>, E> {
        match self.context() {
            Some(context) => context.metadata_vec(count).map_err(|cause| self.map(cause)),
            None => Ok(Vec::with_capacity(count)),
        }
    }
    pub fn reserve<T>(&self, values: &mut Vec<T>, additional: usize) -> Result<(), E> {
        match self.context() {
            Some(context) => context
                .reserve_metadata_vec(values, additional)
                .map_err(|cause| self.map(cause)),
            None => {
                values.reserve(additional);
                Ok(())
            }
        }
    }
    pub fn push<T>(&self, values: &mut Vec<T>, value: T) -> Result<(), E> {
        self.reserve(values, 1)?;
        values.push(value);
        Ok(())
    }
    pub fn completions<T>(&self, _: &Option<T>, count: usize) -> Result<Vec<Option<T>>, E> {
        self.controls::<(Vec<Option<T>>, Result<Vec<Option<T>>, E>)>()?;
        let mut values = self.vector(count)?;
        values.resize_with(count, || None);
        Ok(values)
    }
    pub fn schedule<'a>(
        &self,
        graph: &'a crate::ExecutionGraph,
    ) -> Result<crate::ExecutionGroupSchedule<'a>, E> {
        match self.context() {
            Some(context) => crate::ExecutionGroupSchedule::new_with_metadata(graph, context)
                .map_err(|cause| self.map(cause)),
            None => Ok(crate::ExecutionGroupSchedule::new(graph)),
        }
    }
    // The same borrowed visitor supplies the count and values; no owning
    // iterator or tensor clone is constructed in the collection pass.
    pub fn retained<'a, T: 'a>(
        &self,
        mut visit: impl FnMut(&mut dyn FnMut(&'a T)),
    ) -> Result<Vec<&'a T>, E> {
        self.controls::<(Vec<&'a T>, Option<usize>, bool)>()?;
        if self.context().is_none() {
            let mut values = Vec::new();
            visit(&mut |value| values.push(value));
            return Ok(values);
        }
        let mut count = 0usize;
        let mut overflow = false;
        visit(&mut |_| match count.checked_add(1) {
            Some(next) => count = next,
            None => overflow = true,
        });
        if overflow {
            return Err(self.map(eredu_nn::workspace::WorkspaceMetadataError::Overflow.into()));
        }
        let mut values = self.vector(count)?;
        let mut failure = None;
        visit(&mut |value| {
            if failure.is_none() {
                failure = self.push(&mut values, value).err();
            }
        });
        match failure {
            Some(error) => Err(error),
            None => Ok(values),
        }
    }
}
