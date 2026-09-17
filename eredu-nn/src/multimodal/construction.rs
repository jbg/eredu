//! Shared ordinary/counted host destinations for ordered media and rotary inputs.
use super::*;
use crate::workspace::{WorkspaceContext, WorkspaceMetadataError};

#[derive(Clone, Copy)]
struct Destination<'a>(Option<&'a WorkspaceContext>);
impl Destination<'_> {
    fn vector<T>(self, count: usize) -> Result<Vec<T>, Error> {
        match self.0 {
            Some(context) => context.metadata_vec(count),
            None => Ok(Vec::with_capacity(count)),
        }
    }
    fn error(self, message: std::fmt::Arguments<'_>) -> Error {
        match self.0 {
            Some(context) => context.metadata_error(message),
            None => Error::backend(message),
        }
    }
    fn controls<T>(self) -> Result<(), Error> {
        if let Some(context) = self.0 {
            let sizes = [
                std::mem::size_of::<T>(),
                std::mem::size_of::<Result<T, Error>>(),
                std::mem::size_of::<Self>(),
            ];
            context.charge_metadata(
                sizes
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&sizes), usize::checked_add)
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )?;
        }
        Ok(())
    }
}

/// Uses the ordinary ordered-input validator and tensor equations, admitting
/// its host clone vectors before construction. The enclosing owner retains the
/// metadata funding through returned tensors and errors.
pub fn assemble_ordered_inputs_with_metadata<T: Tensor>(
    parts: &[OrderedInputPart<'_, T>],
    hidden_size: i32,
    context: &T::Context,
    metadata: &WorkspaceContext,
) -> Result<OrderedModelInput<T>, Error> {
    assemble(
        parts,
        hidden_size,
        context,
        Some(metadata).filter(|context| context.uses_checked_metadata()),
    )
}

pub(super) fn assemble<T: Tensor>(
    parts: &[OrderedInputPart<'_, T>],
    hidden_size: i32,
    context: &T::Context,
    metadata: Option<&WorkspaceContext>,
) -> Result<OrderedModelInput<T>, Error> {
    let destination = Destination(metadata);
    destination.controls::<(OrderedModelInput<T>, Vec<T>, OrderedInputPart<'_, T>)>()?;

    if parts.is_empty() || hidden_size <= 0 {
        return Err(destination.error(format_args!(
            "ordered input assembly requires parts and a positive hidden size",
        )));
    }
    let batch = parts[0].token_ids.shape().first().copied().unwrap_or(0);
    for (index, part) in parts.iter().enumerate() {
        let tokens = part.token_ids.shape();
        let embeddings = part.embeddings.shape();
        if tokens.len() != 2
            || embeddings.len() != 3
            || tokens[0] != batch
            || embeddings[0] != batch
            || tokens[1] != embeddings[1]
            || embeddings[2] != hidden_size
        {
            return Err(destination.error(format_args!(
                "ordered input part {index} has incompatible token/embedding shapes {tokens:?} and {embeddings:?}; expected batch {batch} and hidden {hidden_size}"
            )));
        }
    }
    let mut values = destination.vector(parts.len())?;
    values.extend(parts.iter().map(|part| part.token_ids.clone()));
    let token_ids = T::concatenate(&values, 1, context)?;
    drop(values);
    let mut values = destination.vector(parts.len())?;
    values.extend(parts.iter().map(|part| part.embeddings.clone()));
    let embeddings = T::concatenate(&values, 1, context)?;
    drop(values);
    Ok(OrderedModelInput {
        token_ids,
        embeddings,
    })
}

/// Uses the ordinary multi-axis rotary validator and backend operation, with
/// diagnostics constructed through the active metadata destination.
pub fn multi_axis_rotary_embeddings_with_metadata<T: Tensor>(
    position_ids: &T,
    spec: &MultiAxisRotarySpec,
    context: &T::Context,
    metadata: &WorkspaceContext,
) -> Result<(T, T), Error> {
    rotary(
        position_ids,
        spec,
        context,
        Some(metadata).filter(|context| context.uses_checked_metadata()),
    )
}

pub(super) fn rotary<T: Tensor>(
    position_ids: &T,
    spec: &MultiAxisRotarySpec,
    context: &T::Context,
    metadata: Option<&WorkspaceContext>,
) -> Result<(T, T), Error> {
    let destination = Destination(metadata);
    destination.controls::<(T, T, &MultiAxisRotarySpec)>()?;

    let _ = spec.dimensions_with_diagnostic(
        |message| destination.error(message),
        |cause| match destination.0 {
            Some(metadata) => metadata.metadata_source(cause),
            None => Error::backend_source(cause),
        },
    )?;
    let shape = position_ids.shape();
    if shape.len() < 2 || shape.last().copied() != Some(spec.axes.len() as i32) {
        return Err(destination.error(format_args!(
            "multi-axis position IDs must end in {} axes, got {shape:?}",
            spec.axes.len()
        )));
    }
    if let Some(metadata) = destination.0 {
        // The neutral producer owns the exact frequency destination. The
        // backend consumes borrowed prepared values through its existing
        // equation; it cannot allocate an unaccounted host frequency vector.
        destination.controls::<(
            Vec<f32>, usize, PreparedMultiAxisRotary<'_>,
            Result<PreparedMultiAxisRotary<'_>, RotaryTableError>,
            Result<usize, RotaryTableError>,
        )>()?;
        let spec = spec.as_ref();
        let count = spec.frequency_count().map_err(|cause| metadata.metadata_source(cause))?;
        let mut frequencies = destination.vector(count)?;
        frequencies.resize(count, 0.0);
        spec.fill_frequencies(&mut frequencies).map_err(|cause| metadata.metadata_source(cause))?;
        let prepared = PreparedMultiAxisRotary::new(spec, &frequencies)
            .map_err(|cause| metadata.metadata_source(cause))?;
        T::multi_axis_rotary_embeddings_prepared(position_ids, prepared, context)
    } else {
        T::multi_axis_rotary_embeddings(position_ids, spec, context)
    }
}
