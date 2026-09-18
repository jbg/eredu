//! Owned declarations from the same actual composite graph worker.
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec};

#[derive(Clone, Copy)]
pub(crate) struct Destination<'a>(pub(crate) Option<&'a WorkspaceContext>);
impl Destination<'_> {
    pub(crate) fn error(self, args: std::fmt::Arguments<'_>) -> Error {
        match self.0 {
            Some(context) => context.metadata_error(args),
            None => Error::backend(args),
        }
    }
    pub(crate) fn controls<T>(self) -> Result<(), Error> {
        if let Some(context) = self.0 {
            let parts = [
                size_of::<T>(),
                size_of::<Self>(),
                size_of::<Result<T, Error>>(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?;
            context.charge_metadata(bytes)?;
        }
        Ok(())
    }
    pub(crate) fn vector<T>(self, count: usize) -> Result<Vec<T>, Error> {
        match self.0 {
            Some(context) => context.metadata_vec(count),
            None => Ok(Vec::with_capacity(count)),
        }
    }
    pub(crate) fn collect<T, I>(self, values: I) -> Result<Vec<T>, Error>
    where I: IntoIterator<Item = T> {
        if self.0.is_none() { return Ok(values.into_iter().collect()); }
        let mut values = values.into_iter();
        self.controls::<(I::IntoIter, Vec<T>, Option<T>, usize)>()?;
        let limit = values.size_hint().1.ok_or_else(|| self.error(format_args!(
            "composite destination has no finite source population")))?;
        let mut output = self.vector(limit)?;
        for value in values.by_ref() {
            if output.len() == limit { return Err(self.error(format_args!(
                "composite destination exceeded its source population"))); }
            output.push(value);
        }
        Ok(output)
    }
    pub(crate) fn try_collect<T, I>(self, values: I) -> Result<Vec<T>, Error>
    where I: IntoIterator<Item = Result<T, Error>> {
        if self.0.is_none() { return values.into_iter().collect(); }
        let mut values = values.into_iter();
        self.controls::<(I::IntoIter, Vec<T>, Result<T, Error>, usize)>()?;
        let limit = values.size_hint().1.ok_or_else(|| self.error(format_args!(
            "composite destination has no finite source population")))?;
        let mut output = self.vector(limit)?;
        for value in values.by_ref() {
            let value = value?;
            if output.len() == limit { return Err(self.error(format_args!(
                "composite destination exceeded its source population"))); }
            output.push(value);
        }
        Ok(output)
    }
    pub(crate) fn text(self, args: std::fmt::Arguments<'_>) -> Result<String, Error> {
        match self.0 {
            Some(context) => context.metadata_string(args),
            None => Ok(args.to_string()),
        }
    }
    pub(crate) fn boundary_spec(
        self, role: &str, shape: &[eredu_runtime::BoundaryTensorDimension],
        dtype: eredu_runtime::BoundaryTensorDtype,
    ) -> Result<eredu_runtime::BoundaryTensorSpec, Error> {
        self.controls::<(&str, &[eredu_runtime::BoundaryTensorDimension],
            eredu_runtime::BoundaryTensorDtype, eredu_runtime::BoundaryTensorSpec)>()?;
        match self.0 {
            Some(context) => eredu_runtime::BoundaryTensorSpec::new_with_metadata(role, shape, dtype, context),
            None => Ok(eredu_runtime::BoundaryTensorSpec::new(role, shape.iter().copied(), dtype)),
        }
    }
    pub(crate) fn boundary_schema(
        self, identity: &'static str, primary: eredu_runtime::BoundaryTensorSpec,
        auxiliary: Vec<eredu_runtime::BoundaryTensorSpec>,
    ) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.controls::<(&'static str, eredu_runtime::BoundaryTensorSpec,
            Vec<eredu_runtime::BoundaryTensorSpec>, eredu_runtime::BoundaryWireSchema)>()?;
        match self.0 {
            Some(context) => eredu_runtime::BoundaryWireSchema::from_owned_with_metadata(identity, primary, auxiliary, context),
            None => eredu_runtime::BoundaryWireSchema::new(identity, primary, auxiliary).map_err(Error::backend_retained_source),
        }
    }
    pub(crate) fn resolve_boundary(
        self, schema: &eredu_runtime::BoundaryWireSchema, batch: i32, sequences: &[i32],
    ) -> Result<eredu_runtime::ResolvedBoundaryWireSchema, Error> {
        self.controls::<(&eredu_runtime::BoundaryWireSchema, i32, &[i32],
            eredu_runtime::ResolvedBoundaryWireSchema)>()?;
        match self.0 {
            Some(context) => schema.resolve_each_with_metadata(batch, sequences, context),
            None => schema.resolve_each(batch, sequences.iter().copied()).map_err(Error::backend_retained_source),
        }
    }
    pub(crate) fn boundary_value<T>(
        self, role: &str, tensor: T,
    ) -> Result<eredu_runtime::ArchitectureBoundaryValue<T>, Error> {
        self.controls::<(&str, T, eredu_runtime::ArchitectureBoundaryValue<T>)>()?;
        match self.0 {
            Some(context) => eredu_runtime::ArchitectureBoundaryValue::new_with_metadata(role, tensor, context),
            None => eredu_runtime::ArchitectureBoundaryValue::new(role, tensor).map_err(Error::backend_retained_source),
        }
    }
    pub(crate) fn group(
        self,
        id: &str,
        dependencies: &[&str],
    ) -> Result<ExecutionGroupSpec, Error> {
        self.controls::<ExecutionGroupSpec>()?;
        let id = self.text(format_args!("{id}"))?;
        let mut names = self.vector(dependencies.len())?;
        for name in dependencies {
            names.push(self.text(format_args!("{name}"))?);
        }
        Ok(ExecutionGroupSpec::from_parts(id, names))
    }
    pub(crate) fn layout(
        self,
        graph: &ExecutionGraph,
        counts: &[usize],
    ) -> Result<eredu_runtime::ExecutionUnitLayout, Error> {
        match self.0 {
            Some(context) => {
                eredu_runtime::ExecutionUnitLayout::new_with_metadata(graph, counts, context)
            }
            None => eredu_runtime::ExecutionUnitLayout::new(graph, counts.iter().copied())
                .map_err(Error::backend),
        }
    }

    pub(crate) fn finish(
        self,
        groups: Vec<ExecutionGroupSpec>,
        output: &str,
    ) -> Result<ExecutionGraph, Error> {
        match self.0 {
            Some(context) => ExecutionGraph::new_with_metadata(groups, output, context),
            None => ExecutionGraph::new(groups, output).map_err(Error::backend),
        }
    }
}
