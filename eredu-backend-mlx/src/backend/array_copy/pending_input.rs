//! Closed numerical preparation of authenticated saved text input.

use super::{ExistingArrayProjection, IsolatedArrayCopy};
use eredu_nn::{
    workspace::{WorkspaceDtype, WorkspaceOperationKind, WorkspaceTensor},
    Tensor,
};
use safemlx::{error::Exception, Array, ArrayDescriptorError, ArrayDescriptorFacts, Dtype, Stream};
use std::{cell::RefCell, collections::TryReserveError};

mod metadata;
mod original;
pub(crate) use metadata::PendingTokenMetadataError;
pub(crate) use original::{RegisteredArrayCopy, RegisteredArrayCopyCustody};
pub(crate) type OriginalPendingTokenCopy = RegisteredArrayCopy;

#[derive(Debug, thiserror::Error)]
pub(crate) enum PendingTokenInputError {
    #[error("pending input does not match its saved scalar or complete text-matrix geometry")]
    InvalidShape,
    #[error("pending token input must use Int32 or Uint32, got {actual:?}")]
    InvalidDtype { actual: Dtype },
    #[error("pending token input has no completed certified backing")]
    UnsettledSource,
    #[error("pending token source descriptor or backing changed after preparation")]
    SourceChanged,
    #[error("pending token source metadata: {0}")]
    Metadata(#[from] ArrayDescriptorError),
    #[error("pending token workspace: {0}")]
    Workspace(#[from] eredu_nn::Error),
    #[error("pending token native preparation: {0}")]
    Native(#[from] Exception),
    #[error("pending token recovery collector is already borrowed")]
    CollectorBusy,
    #[error("pending token recovery needs {needed} slots, but only {available} remain")]
    CollectorCapacity { needed: usize, available: usize },
    #[error("pending token recovery collector capacity: {0}")]
    Allocation(#[from] TryReserveError),
}

/// Fixed source-only refusal available before host admission. No workspace
/// trace, owned shape, native operation or formatted exception is constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PendingTokenSourceCause {
    #[error("pending input does not match its saved scalar or complete text-matrix geometry")]
    InvalidShape,
    #[error("pending token input must use Int32 or Uint32, got {actual:?}")]
    InvalidDtype { actual: Dtype },
    #[error("pending token input has no completed certified backing")]
    UnsettledSource,
    #[error("pending token source descriptor or backing changed after preparation")]
    SourceChanged,
    #[error(transparent)]
    Descriptor(#[from] ArrayDescriptorError),
}
impl From<PendingTokenSourceCause> for PendingTokenInputError {
    fn from(cause: PendingTokenSourceCause) -> Self {
        match cause {
            PendingTokenSourceCause::InvalidShape => Self::InvalidShape,
            PendingTokenSourceCause::InvalidDtype { actual } => Self::InvalidDtype { actual },
            PendingTokenSourceCause::UnsettledSource => Self::UnsettledSource,
            PendingTokenSourceCause::SourceChanged => Self::SourceChanged,
            PendingTokenSourceCause::Descriptor(cause) => Self::Metadata(cause),
        }
    }
}

/// Complete numerical population of this one pending-input invocation. Fresh
/// request admission must include it; this value provides no scope or quota.
pub(crate) struct PendingTokenNativePopulation {
    pub(crate) construction: crate::backend::nn::workspace::IsolatedCopyNativeLayout,
    pub(crate) logical_backing_bytes: usize,
    pub(crate) backing_births: usize,
}

/// Binds the actual scalar and its observed descriptor without numerical work.
///
/// The enclosing saved-token owner must establish the token's numeric validity
/// and retain the source, exclusive settled access, allocation authority and
/// recovery custody. Shape/dtype/backing checks cannot establish vocabulary
/// validity or prevent another native alias from changing values. The snapshot
/// is private inspection metadata, never a detached allocation certificate.
///
/// This prices and executes only numerical input preparation. Host input-part
/// containers, semantic identities, fresh request admission and completion are
/// separate obligations; no token value is downloaded or added to history.
pub(crate) struct PreparedPendingTokenInput<'a> {
    source: &'a Array,
    observed: ArrayDescriptorFacts,
    shape: PendingInputShape,
}

#[derive(Clone, Copy)]
enum PendingInputShape {
    Scalar,
    Matrix(i32),
}

impl PendingInputShape {
    fn target(self) -> [i32; 2] {
        [
            1,
            match self {
                Self::Scalar => 1,
                Self::Matrix(positions) => positions,
            },
        ]
    }
}

impl<'a> PreparedPendingTokenInput<'a> {
    pub(crate) fn new(source: &'a Array) -> Result<Self, PendingTokenInputError> {
        Self::new_fixed(source).map_err(Into::into)
    }

    pub(crate) fn new_fixed(source: &'a Array) -> Result<Self, PendingTokenSourceCause> {
        Self::with_shape(source, PendingInputShape::Scalar)
    }

    /// The immutable saved prompt supplies the position count. Requiring the
    /// complete [1, N] source prevents a scalar or sliced prefix from being
    /// silently reclassified as pending prefill.
    pub(crate) fn new_prefill_fixed(
        source: &'a Array,
        positions: std::num::NonZeroU64,
    ) -> Result<Self, PendingTokenSourceCause> {
        let positions =
            i32::try_from(positions.get()).map_err(|_| PendingTokenSourceCause::InvalidShape)?;
        Self::with_shape(source, PendingInputShape::Matrix(positions))
    }

    fn with_shape(
        source: &'a Array,
        shape: PendingInputShape,
    ) -> Result<Self, PendingTokenSourceCause> {
        let observed = observe(source, shape)?;
        Ok(Self {
            source,
            observed,
            shape,
        })
    }

    pub(crate) fn source(&self) -> &'a Array {
        self.source
    }

    pub(crate) fn positions(&self) -> u64 {
        self.shape.target()[1] as u64
    }

    /// Logical byte count and rank of the shared program's final packed
    /// uint32 matrix. Both admitted source dtypes have the same width, and the
    /// isolate/reshape/cast program preserves every element. No physical fit
    /// or native construction is inferred from these immutable source facts.
    pub(crate) fn logical_output(&self) -> (usize, usize) {
        (self.observed.logical_bytes(), self.shape.target().len())
    }

    /// Number of produced descriptors, excluding the caller-owned source.
    /// The caller may reserve a complete enclosing collector before admission.
    pub(crate) fn retained_descriptor_count(&self) -> usize {
        3 + usize::from(self.observed.dtype() == Dtype::Int32)
    }

    fn validate(&self) -> Result<(), PendingTokenInputError> {
        self.validate_fixed().map_err(Into::into)
    }

    pub(crate) fn validate_fixed(&self) -> Result<(), PendingTokenSourceCause> {
        let current = observe(self.source, self.shape)?;
        if current != self.observed {
            return Err(PendingTokenSourceCause::SourceChanged);
        }
        Ok(())
    }

    /// Runs the same ordered program with workspace values. The caller supplies
    /// the shared source projection and span, preserving cross-role aliases.
    /// Missing operation or host facts remain incomplete in that context.
    pub(crate) fn trace(
        &self,
        projection: &mut ExistingArrayProjection<'a>,
    ) -> Result<WorkspaceTensor, PendingTokenInputError> {
        self.validate()?;
        prepare(
            &mut WorkspacePendingInput {
                source: self.source,
                projection,
                shape: self.shape.target(),
            },
            self.observed.dtype() == Dtype::Int32,
        )
    }

    /// Copies only under the caller's existing authority. Every produced array
    /// is retained before another fallible native operation; the caller keeps
    /// the collector and actual saved source until exact recovery settles.
    /// Reserving descriptor capacity is bookkeeping, not a numerical grant.
    pub(crate) fn copy_retained(
        self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<PreparedPendingTokenArray, PendingTokenInputError> {
        self.validate()?;
        let original = safemlx::OriginalScopeObserver::try_current()?.is_some();
        {
            let mut roots = roots
                .try_borrow_mut()
                .map_err(|_| PendingTokenInputError::CollectorBusy)?;
            let needed = self.retained_descriptor_count();
            let available = roots.capacity() - roots.len();
            if original && available < needed {
                return Err(PendingTokenInputError::CollectorCapacity { needed, available });
            }
            if !original {
                roots.try_reserve_exact(needed)?;
            }
        }
        let array = prepare(
            &mut NativePendingInput {
                source: self.source,
                stream,
                roots,
                shape: self.shape.target(),
            },
            self.observed.dtype() == Dtype::Int32,
        )?;
        if original {
            // The isolate leaf already completed Contiguous before its eager
            // copy. This distinct frontier completes reshape/optional cast in
            // the same admitted bank, before any prompt publication or read.
            safemlx::OperationEvent::complete_nested([&array], stream)?;
        }
        Ok(PreparedPendingTokenArray { array })
    }
}

fn observe(
    source: &Array,
    shape: PendingInputShape,
) -> Result<ArrayDescriptorFacts, PendingTokenSourceCause> {
    let descriptor = source.try_descriptor()?;
    let facts = descriptor.facts();
    // Every valid dimension is one, so rank completely describes the shape.
    // Repeating this check plus fixed-facts equality preserves the old exact
    // shape/dtype/bytes/allocation comparison without owning a shape Vec.
    let valid = match shape {
        PendingInputShape::Scalar => descriptor.shape().iter().all(|&size| size == 1),
        PendingInputShape::Matrix(positions) => descriptor.shape() == [1, positions],
    };
    if !valid {
        return Err(PendingTokenSourceCause::InvalidShape);
    }
    if !matches!(facts.dtype(), Dtype::Int32 | Dtype::Uint32) {
        return Err(PendingTokenSourceCause::InvalidDtype {
            actual: facts.dtype(),
        });
    }
    let bytes = usize::try_from(shape.target()[1])
        .ok()
        .and_then(|n| n.checked_mul(4));
    if bytes != Some(facts.logical_bytes()) {
        return Err(PendingTokenSourceCause::InvalidShape);
    }
    if facts.allocation().is_none() {
        return Err(PendingTokenSourceCause::UnsettledSource);
    }
    Ok(facts)
}

impl PreparedPendingTokenInput<'_> {
    /// Pure source-control contribution, excluding the separately planned host
    /// part, workspace trace and complete request Graph/Record/buffer owners.
    pub(crate) fn source_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            Array::descriptor_control_bytes()?,
            safemlx::OperationEvent::nested_completion_control_bytes::<1>()?,
            size_of::<Self>(),
            size_of::<PendingTokenSourceCause>(),
            size_of::<Result<Self, PendingTokenSourceCause>>(),
            size_of::<Result<ArrayDescriptorFacts, PendingTokenSourceCause>>(),
            size_of::<PendingTokenNativePopulation>(),
            size_of::<Option<PendingTokenNativePopulation>>(),
            size_of::<NativePendingInput<'_>>(),
            size_of::<PreparedPendingTokenArray>(),
            size_of::<PendingTokenInputError>(),
            size_of::<Result<PreparedPendingTokenArray, PendingTokenInputError>>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<std::cell::RefMut<'_, Vec<Array>>>(),
            size_of::<Option<safemlx::OriginalScopeObserver>>(),
            size_of::<[i32; 2]>(),
            size_of::<usize>() * 2,
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }

    /// Same source-derived constructor/dispatch populations consumed by native
    /// isolated copies, plus this program's actual reshape/cast frontier. The
    /// fresh resume request must join these extents before constructing a bank.
    pub(crate) fn original_population(&self) -> Option<PendingTokenNativePopulation> {
        let signed = self.observed.dtype() == Dtype::Int32;
        Some(PendingTokenNativePopulation {
            construction: crate::backend::nn::workspace::IsolatedCopyNativeLayout::pending_input(
                self.observed.rank().max(2),
                signed,
            )?,
            logical_backing_bytes: self
                .observed
                .logical_bytes()
                .checked_mul(2 + usize::from(signed))?,
            backing_births: 2 + usize::from(signed),
        })
    }
}

/// Isolated numerical [1, N] U32 input, with caller-owned recovery custody.
/// It contains no input identity, controller, request receipt or host container.
pub(crate) struct PreparedPendingTokenArray {
    array: Array,
}

impl PreparedPendingTokenArray {
    pub(crate) fn array(&self) -> &Array {
        &self.array
    }

    /// Moves the numerical result into an enclosing prepared owner. Extraction
    /// provides no allocation permission, completion or full prompt proof.
    pub(crate) fn into_array(self) -> Array {
        self.array
    }
}

trait PendingInputMechanism {
    type Value;
    type Error;
    fn isolate(&mut self) -> Result<Self::Value, Self::Error>;
    fn reshape(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    fn cast(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
}

fn prepare<M: PendingInputMechanism>(mechanism: &mut M, cast: bool) -> Result<M::Value, M::Error> {
    let isolated = mechanism.isolate()?;
    let reshaped = mechanism.reshape(isolated)?;
    if cast {
        mechanism.cast(reshaped)
    } else {
        Ok(reshaped)
    }
}

struct WorkspacePendingInput<'a, 'p> {
    source: &'a Array,
    projection: &'p mut ExistingArrayProjection<'a>,
    shape: [i32; 2],
}

impl PendingInputMechanism for WorkspacePendingInput<'_, '_> {
    type Value = WorkspaceTensor;
    type Error = PendingTokenInputError;
    fn isolate(&mut self) -> Result<Self::Value, PendingTokenInputError> {
        Ok(IsolatedArrayCopy::new(self.source).trace(self.projection)?)
    }
    fn reshape(&mut self, value: Self::Value) -> Result<Self::Value, PendingTokenInputError> {
        Ok(value.reshape(&self.shape, self.projection.context())?)
    }
    fn cast(&mut self, value: Self::Value) -> Result<Self::Value, PendingTokenInputError> {
        let context = self.projection.context();
        let mut outputs = context.metadata_vec(1)?;
        outputs.push(context.layout(&self.shape, WorkspaceDtype::Uint32)?);
        Ok(context
            .execute(
                WorkspaceOperationKind::Elementwise("cast_u32"),
                &[&value],
                outputs,
            )?
            .remove(0))
    }
}

struct NativePendingInput<'a> {
    source: &'a Array,
    stream: &'a Stream,
    roots: &'a RefCell<Vec<Array>>,
    shape: [i32; 2],
}

impl NativePendingInput<'_> {
    fn retain(&self, value: &Array) -> Result<(), PendingTokenInputError> {
        let retained = value.try_clone_handle()?;
        self.roots.borrow_mut().push(retained);
        Ok(())
    }
}

impl PendingInputMechanism for NativePendingInput<'_> {
    type Value = Array;
    type Error = PendingTokenInputError;
    fn isolate(&mut self) -> Result<Self::Value, PendingTokenInputError> {
        Ok(IsolatedArrayCopy::new(self.source).copy_retained(self.stream, self.roots)?)
    }
    fn reshape(&mut self, value: Self::Value) -> Result<Self::Value, PendingTokenInputError> {
        #[cfg(test)]
        tests::after_isolation()?;
        let reshaped = value.reshape(&self.shape, self.stream)?;
        self.retain(&reshaped)?;
        Ok(reshaped)
    }
    fn cast(&mut self, value: Self::Value) -> Result<Self::Value, PendingTokenInputError> {
        let cast = value.as_type::<u32>(self.stream)?;
        self.retain(&cast)?;
        Ok(cast)
    }
}

#[cfg(test)]
mod tests;
