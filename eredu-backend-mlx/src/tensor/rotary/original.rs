//! Concrete owners of the existing prepared spatial equation.
use super::*;
use safemlx::{OriginalScopeObserver, error::Exception};
use std::{fmt, mem::size_of};

#[derive(Debug, Clone, Copy, thiserror::Error)]
enum Invalid {
    #[error("multi-axis position geometry differs from its axes")]
    Geometry,
    #[error("multi-axis position dimensions overflowed i32")]
    Overflow,
    #[error("prepared multi-axis rotary requires a qualified prepared construction profile")]
    Profile,
}

#[derive(Debug)]
struct Failure<E> {
    cause: E,
    // The closed neutral error releases its own shells before this source;
    // the role's accepted failure-account custody stays last through its cause.
    // It adds no Scope/Record/root back-reference and does not mark failure.
    _custody: Exception,
}
impl<E: fmt::Display> fmt::Display for Failure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<E: std::error::Error + 'static> std::error::Error for Failure<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

pub(super) struct Execution {
    observer: Option<OriginalScopeObserver>,
}
impl Execution {
    pub(super) fn current() -> Result<Self, Error> {
        Ok(Self {
            observer: OriginalScopeObserver::try_current()
                .map_err(Error::backend_retained_source)?,
        })
    }
    pub(super) fn source<E: std::error::Error + Send + Sync + 'static>(&self, cause: E) -> Error {
        match &self.observer {
            Some(observer) => Error::backend_retained_source(Failure {
                cause,
                _custody: observer.invalid_input_error(),
            }),
            None => Error::backend_retained_source(cause),
        }
    }
    pub(super) fn observer(&self) -> Option<&OriginalScopeObserver> {
        self.observer.as_ref()
    }
    pub(super) fn profile_error(&self) -> Error {
        self.source(Invalid::Profile)
    }
    pub(super) fn native(&self, result: Result<Array, Exception>) -> Result<Array, Error> {
        result.map_err(|cause| self.source(cause))
    }
    pub(super) fn geometry_error(&self) -> Error {
        if self.observer.is_some() {
            self.source(Invalid::Geometry)
        } else {
            Error::backend("multi-axis position geometry differs from its axes")
        }
    }
    pub(super) fn overflow_error(&self) -> Error {
        if self.observer.is_some() {
            self.source(Invalid::Overflow)
        } else {
            Error::backend("multi-axis position dimensions overflowed i32")
        }
    }
    pub(super) fn require_profile(
        &self,
        spec: MultiAxisRotarySpecRef<'_>,
        input_rank: usize,
        prepared: bool,
    ) -> Result<(), Error> {
        if self.observer.is_some()
            && (!prepared || PreparedRotaryProfile::inspect(spec, input_rank).is_none())
        {
            return Err(self.source(Invalid::Profile));
        }
        Ok(())
    }
}

/// The actual prepared constructor population. It grants no authority or
/// source identity; the recipe and execution entry consume this same profile.
#[derive(Clone, Copy)]
pub(crate) struct PreparedRotaryProfile {
    pub(crate) primitives: usize,
    pub(crate) edges: usize,
    pub(crate) seeds: usize,
    pub(crate) cloned_handles: usize,
    pub(crate) maximum_operands: usize,
    row_bank: safemlx::ops::OriginalArrayRowsLayout,
    row_capacity: usize,
    columns: usize,
    concatenations: usize,
}
impl PreparedRotaryProfile {
    pub(crate) fn inspect(spec: MultiAxisRotarySpecRef<'_>, input_rank: usize) -> Option<Self> {
        let dimensions = spec.dimensions().ok()?;
        let axes = spec.axes.len();
        let half = usize::try_from(dimensions / 2).ok()?;
        if input_rank < 2 || i32::try_from(axes).is_err() {
            return None;
        }
        let linear = |factor: usize, count: usize, fixed: usize| {
            factor.checked_mul(count)?.checked_add(fixed)
        };
        let (primitives, edges, seeds, columns, concatenations, cloned_handles, capacity) =
            match spec.layout {
                // Same signed envelope: column Slice/reshape and saturation, then
                // copied frequency, cast/expand and product. Independent expansion
                // moves before row append, preserving all numerical constructors.
                MultiAxisRotaryLayout::IndependentAxes => (
                    linear(34, axes, 9)?,
                    linear(41, axes, 8)?,
                    axes.checked_mul(5)?,
                    axes,
                    axes.checked_add(1)?,
                    axes,
                    axes,
                ),
                MultiAxisRotaryLayout::SplitHalves => (
                    linear(31, axes, 12)?,
                    linear(37, axes, 12)?,
                    axes.checked_mul(5)?,
                    axes,
                    2,
                    1,
                    axes,
                ),
                MultiAxisRotaryLayout::RoundRobinSections => (
                    linear(24, half, 19)?,
                    linear(29, half, 20)?,
                    linear(4, half, 1)?,
                    half,
                    2,
                    1,
                    half,
                ),
            };
        let row_bank = safemlx::ops::OriginalArrayRowsLayout::inspect(capacity, input_rank)?;
        // This extra control envelope supplies the actual row header/backing
        // and borrowed output Shape copies; it is not another equation node.
        let primitives = primitives.checked_add(row_bank.resident_controls())?;
        primitives.checked_add(seeds)?; // Lowering::plain's closed sum
        Some(Self {
            primitives,
            edges,
            seeds,
            cloned_handles,
            maximum_operands: row_bank.maximum_operands(),
            row_bank,
            row_capacity: capacity,
            columns,
            concatenations,
        })
    }

    pub(crate) fn ordinary_row_bytes(self) -> Option<usize> {
        if self.row_capacity <= INLINE_VALUES {
            Some(0)
        } else {
            self.row_capacity.checked_mul(size_of::<Array>())
        }
    }

    pub(crate) fn control_bytes(self) -> Option<usize> {
        use eredu_nn::multimodal::{PreparedMultiAxisRotary, RotaryAxisSpec, RotaryTableError};
        // Name each owning local population. Scalar Arrays and views created
        // by the nested column worker are separate from the outer owners.
        // Outer Array locals: reshape, widening, frequency copy, column,
        // cast, expand, joined angles/half, selected rows, round-robin copy,
        // selected cast/product, cos/cos-reshape and sin/sin-reshape (16).
        // Column locals: selected column, offset, addition, both signed
        // endpoints, saturation and minimum (7).
        let failure = Error::retained_source_construction_bytes::<Failure<Exception>>()?
            .max(Error::retained_source_construction_bytes::<
                Failure<RotaryTableError>,
            >()?)
            .max(Error::retained_source_construction_bytes::<Failure<Invalid>>()?)
            .max(Error::retained_source_construction_bytes::<Exception>()?);
        let fixed = [
            size_of::<Self>(),
            size_of::<Execution>(),
            failure,
            size_of::<rows::Rows<'static>>(),
            size_of::<Result<rows::Rows<'static>, Error>>(),
            size_of::<Result<(), Error>>(),
            self.row_bank.control_bytes()?,
            size_of::<std::borrow::Cow<'_, [f32]>>(),
            size_of::<PreparedMultiAxisRotary<'_>>(),
            size_of::<MultiAxisRotarySpecRef<'_>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, RotaryAxisSpec>>>(),
            size_of::<(Array, Array)>(),
            size_of::<[Array; 2]>(),
            size_of::<Result<(MlxTensor, MlxTensor), Error>>(),
            size_of::<Result<Array, Error>>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<(&Array, &Stream, &MultiAxisRotarySpecRef<'_>, &bool, &bool)>(),
            size_of::<&Execution>(),
            size_of::<&[i32]>(),
            size_of::<&[f32]>(),
            size_of::<&[Array]>(),
            size_of::<&MlxTensor>(),
            size_of::<&Stream>(),
            size_of::<Option<&[f32]>>(),
            3 * size_of::<i32>(),
            6 * size_of::<usize>(),
            2 * size_of::<bool>(),
            4 * size_of::<i64>(),
            size_of::<[i32; 2]>(),
            size_of::<[Array; 16]>(),
            size_of::<[Array; 7]>(),
            // clip owns its bounds, then two successive optional pairs.
            size_of::<(Array, Array)>(),
            2 * size_of::<(Option<Array>, Option<Array>)>(),
            OriginalScopeObserver::control_bytes()?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        fixed
            .checked_add(
                self.columns
                    .checked_mul(safemlx::ops::indexing::inline_basic_index_control_bytes()?)?,
            )?
            .checked_add(
                self.concatenations
                    .checked_mul(safemlx::ops::concatenate_axis_control_bytes()?)?,
            )
    }
}
