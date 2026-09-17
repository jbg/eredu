//! Paid destinations for finite, architecture-owned pipeline auxiliary values.
//! Families declare roles and dimensions; this worker never chooses a family,
//! reconstructs a model, or grants native transport permission.
use crate::composite_execution::graph::Destination;
use eredu_nn::{
    workspace::{WorkspaceContext, WorkspaceMetadataError},
    Error,
};
use eredu_runtime::{
    ArchitectureBoundaryError, ArchitectureBoundaryValue, BoundaryTensorDimension as Dim,
    BoundaryTensorDtype as Dtype, BoundaryTensorSpec, BoundaryWireSchema,
};

pub(crate) type Field<'a> = (&'a str, &'a [Dim], Dtype);
pub(crate) type Repeated<'a> = (&'a str, usize, &'a [Dim], Dtype);

pub(crate) fn schema(
    context: &WorkspaceContext,
    identity: &'static str,
    hidden: i32,
    fixed: &[Field<'_>],
    repeated: Option<Repeated<'_>>,
) -> Result<BoundaryWireSchema, Error> {
    let destination = Destination(Some(context));
    destination.controls::<(
        &WorkspaceContext,
        &str,
        i32,
        &[Field<'_>],
        Option<Repeated<'_>>,
        BoundaryWireSchema,
        BoundaryTensorSpec,
        Vec<BoundaryTensorSpec>,
        String,
        [Dim; 3],
        std::slice::Iter<'_, Field<'_>>,
        std::ops::Range<usize>,
    )>()?;
    let count = fixed
        .len()
        .checked_add(repeated.map_or(0, |(_, count, _, _)| count))
        .ok_or(WorkspaceMetadataError::Overflow)?;
    let primary = BoundaryTensorSpec::new_with_metadata(
        "hidden",
        &[Dim::Batch, Dim::Sequence, Dim::Fixed(hidden)],
        Dtype::Activation,
        context,
    )?;
    let mut auxiliary = destination.vector(count)?;
    for &(role, shape, dtype) in fixed {
        auxiliary.push(BoundaryTensorSpec::new_with_metadata(
            role, shape, dtype, context,
        )?);
    }
    if let Some((prefix, count, shape, dtype)) = repeated {
        for index in 0..count {
            let role = destination.text(format_args!("{prefix}.{index}"))?;
            auxiliary.push(BoundaryTensorSpec::new_with_metadata(
                &role, shape, dtype, context,
            )?);
        }
    }
    BoundaryWireSchema::from_owned_with_metadata(identity, primary, auxiliary, context)
}

fn count(
    context: &WorkspaceContext,
    identity: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Error> {
    Destination(Some(context)).controls::<(
        &WorkspaceContext,
        &str,
        usize,
        usize,
        ArchitectureBoundaryError,
    )>()?;
    if actual != expected {
        return Err(
            context.metadata_source(ArchitectureBoundaryError::TensorCount {
                boundary: identity,
                expected,
                actual,
            }),
        );
    }
    Ok(())
}

/// Move values through one paid canonical role vector. The caller owns the
/// enclosing metadata account until the boundary and dependent work retire.
pub(crate) fn encode<T, const N: usize>(
    context: &WorkspaceContext,
    identity: &'static str,
    fixed: [(&'static str, T); N],
    tail: Vec<T>,
    expected_tail: usize,
    prefix: &str,
) -> Result<Vec<ArchitectureBoundaryValue<T>>, Error> {
    let destination = Destination(Some(context));
    destination.controls::<(
        &WorkspaceContext,
        &str,
        [(&str, T); N],
        Vec<T>,
        usize,
        Vec<ArchitectureBoundaryValue<T>>,
        std::array::IntoIter<(&str, T), N>,
        std::iter::Enumerate<std::vec::IntoIter<T>>,
        String,
    )>()?;
    count(context, identity, expected_tail, tail.len())?;
    let capacity = N
        .checked_add(expected_tail)
        .ok_or(WorkspaceMetadataError::Overflow)?;
    let mut values = destination.vector(capacity)?;
    for (role, value) in fixed {
        values.push(ArchitectureBoundaryValue::new_with_metadata(
            role, value, context,
        )?);
    }
    for (index, value) in tail.into_iter().enumerate() {
        let role = destination.text(format_args!("{prefix}.{index}"))?;
        values.push(
            ArchitectureBoundaryValue::new(role, value)
                .map_err(|cause| context.metadata_source(cause))?,
        );
    }
    Ok(values)
}

/// Validate cardinality before moving the fixed fields and variable tail.
/// The original tensor values are moved, never cloned or synthesized.
pub(crate) fn decode<T, const N: usize>(
    context: &WorkspaceContext,
    identity: &'static str,
    tensors: Vec<T>,
    expected_tail: usize,
) -> Result<([T; N], Vec<T>), Error> {
    let destination = Destination(Some(context));
    destination.controls::<(
        &WorkspaceContext,
        &str,
        Vec<T>,
        usize,
        [T; N],
        std::vec::IntoIter<T>,
    )>()?;
    let expected = N
        .checked_add(expected_tail)
        .ok_or(WorkspaceMetadataError::Overflow)?;
    count(context, identity, expected, tensors.len())?;
    let mut tail = destination.vector(expected_tail)?;
    let mut tensors = tensors.into_iter();
    let fixed = std::array::from_fn(|_| tensors.next().expect("validated fixed boundary fields"));
    tail.extend(tensors);
    Ok((fixed, tail))
}

#[cfg(test)]
pub(crate) mod tests;
