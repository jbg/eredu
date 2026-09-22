//! Shared outer normalization, row slices and output joins for an addressable
//! invocation. Compact construction and each completed numerical child remain
//! separate sources; this projection alone never establishes complete admission.
use super::*;
use eredu_nn::Tensor;
use std::mem::{size_of, size_of_val};
mod child;
mod facts;
mod local;
mod observation;
mod ordinary;
mod parameters;
mod row_candidates;
mod sources;
pub(crate) use facts::MlxAddressableWorkspaceMechanisms;
pub(crate) use ordinary::{OrdinaryAddressableProgram, OrdinaryAddressableSources};
pub(crate) use sources::{AddressableInvocation, AddressableQuoteRef, AddressableSources};
mod quote;
pub(crate) use child::AddressableChildSource;
pub(crate) use quote::{AddressableEquationSource, AddressableQuote};

pub(crate) struct AddressableParentSource {
    pub(crate) report: WorkspaceTraceReport,
    pub(crate) output_layouts: Vec<WorkspaceLayout>,
    pub(crate) child_invocations: usize,
    pub(crate) retained_child_outputs: usize,
}
impl AddressableParentSource {
    /// The child supplies the exact output representation for each result role.
    /// It is already independently qualified; no F32/contiguity fact is inferred
    /// from the logical declaration. The maximum full chunk determines shape.
    pub(crate) fn prepare(
        source: WorkspaceAddressableRegionView<'_>,
        inputs: &[WorkspaceLayout],
        child_outputs: &[WorkspaceLayout],
        mechanism: ResidentExecutionMechanisms,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Error> {
        let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone())?;
        context.charge_metadata(size_of::<(
            Self,
            WorkspaceAddressableRegionView<'_>,
            WorkspaceContext,
            Result<Self, Error>,
            Vec<WorkspaceTensor>,
            Vec<Vec<WorkspaceTensor>>,
            [i32; 2],
            [usize; 4],
            Vec<WorkspaceLayout>,
        )>())?;
        source.validate()?;
        let invalid =
            || {
                context.metadata_error(format_args!(
                "addressable parent differs from its original inputs or completed child outputs: \
                 group={} bank={} unit={} prefill={} chunks={:?} dimensions={:?} partitions={:?}; \
                 inputs={:?}; completed_children={:?}",
                source.owner_group, source.bank, source.unit, source.prefill,
                source.chunks, source.kernel.dimensions(), source.tensor_partitions,
                inputs, child_outputs,
            ))
            };
        let chunks = source.chunks.len().ok_or_else(invalid)?;
        let roles = 1 + usize::from(source.kernel.separate_bias(source.tensor_partitions));
        if inputs.len() != 4 || child_outputs.len() != roles || chunks == 0 {
            return Err(invalid());
        }
        let shape = ExpertRegionInputShape::inspect(inputs[0].shape(), inputs[1].shape())?;
        let (width, output_width) = source.kernel.dimensions();
        if usize::try_from(shape.rows).ok() != Some(source.chunks.rows)
            || usize::try_from(shape.routes).ok() != Some(source.chunks.routes)
            || shape.width != width
            || inputs[1].shape() != inputs[2].shape()
            || inputs[1].shape() != inputs[3].shape()
            || !matches!(
                inputs[1].dtype(),
                WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
            )
        {
            return Err(invalid());
        }
        let largest = i32::try_from(source.chunks.rows.min(source.chunks.chunk_rows))
            .map_err(|_| invalid())?;
        for layout in child_outputs {
            if layout.shape() != [largest, output_width]
                || layout.dtype() != WorkspaceDtype::Float32
                || layout.representation().is_none()
            {
                return Err(invalid());
            }
        }
        let mut original = context.metadata_vec(4)?;
        for layout in inputs {
            original.push(WorkspaceTensor::existing(
                context
                    .layout(layout.shape(), layout.dtype())?
                    .with_representation(layout.representation()),
                &context,
            )?);
        }
        // Match the actual provider's four unconditional reshapes, even when
        // a caller already supplied rank-two input.
        context.begin_span();
        let mut normalized = context.metadata_vec(4)?;
        for (index, value) in original.iter().enumerate() {
            normalized.push(value.reshape(
                &[shape.rows, if index == 0 { width } else { shape.routes }],
                &context,
            )?);
        }
        let mut completed = context.metadata_vec(roles)?;
        for _ in 0..roles {
            completed.push(context.metadata_vec(chunks)?);
        }
        for ordinal in 0..chunks {
            let range = source.chunks.range(ordinal).ok_or_else(invalid)?;
            let start = i32::try_from(range.start).map_err(|_| invalid())?;
            let end = i32::try_from(range.end).map_err(|_| invalid())?;
            for value in &normalized {
                // The native indexed worker uses a rank-preserving axis-zero
                // range, including for a full-width or one-row chunk.
                let _ = value.narrow_axis(0, start, end, &context)?;
            }
            for (role, layout) in child_outputs.iter().enumerate() {
                // Existing is a completed child frontier, not a new parameter,
                // zero tensor or second numerical equation.
                completed[role].push(WorkspaceTensor::existing(
                    context
                        .layout(&[end - start, output_width], layout.dtype())?
                        .with_representation(layout.representation()),
                    &context,
                )?);
            }
        }
        let mut shape = context.metadata_vec(inputs[0].shape().len())?;
        shape.extend_from_slice(inputs[0].shape());
        *shape.last_mut().ok_or_else(invalid)? = output_width;
        let mut outputs = context.metadata_vec(roles)?;
        let mut layouts = context.metadata_vec(roles)?;
        for mut values in completed {
            let joined = if values.len() == 1 {
                values.pop().ok_or_else(invalid)?
            } else {
                WorkspaceTensor::concatenate(&values, 0, &context)?
            };
            let output = joined.reshape(&shape, &context)?;
            layouts.push(
                context
                    .layout(output.shape(), output.layout().dtype())?
                    .with_representation(output.layout().representation()),
            );
            outputs.push(output);
        }
        let report = context.finish_report(&outputs)?;
        Ok(Self {
            report,
            output_layouts: layouts,
            child_invocations: chunks,
            retained_child_outputs: chunks.checked_mul(roles).ok_or_else(invalid)?,
        })
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
