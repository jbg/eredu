//! Requested typed storage, separate from allocator charge and native fit.
use eredu_gguf::{ConversionPlan, LogicalDtype};
use std::{alloc::Layout, collections::TryReserveError};
pub(crate) mod supplied;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransformKind {
    NativeBytes { width: usize },
    HalfBits,
    Move,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypedOutputRequest {
    pub(crate) dtype: LogicalDtype,
    pub(crate) kind: TransformKind,
    pub(crate) input_elements: usize,
    pub(crate) output_elements: usize,
    pub(crate) declared_shape_elements: Option<u64>,
}
impl TypedOutputRequest {
    /// Map the neutral converter's ordered storage requests to MLX host scalars.
    /// This does not repeat GGML dispatch or use declared shape as payload length.
    pub(crate) fn for_output(plan: &ConversionPlan, ordinal: usize) -> Option<Self> {
        let output = plan.outputs().get(ordinal)?;
        let n = plan.requested_elements();
        let dtype = output.dtype();
        let (kind, input_elements, output_elements) = match (plan.outputs().len(), ordinal, dtype) {
            (1, 0, LogicalDtype::U8) => (TransformKind::Move, n[2], n[2]),
            (1, 0, dtype) => {
                let width = match dtype {
                    LogicalDtype::F32 | LogicalDtype::I32 => 4,
                    LogicalDtype::F16 | LogicalDtype::Bf16 | LogicalDtype::I16 => 2,
                    LogicalDtype::I8 => 1,
                    LogicalDtype::I64 | LogicalDtype::F64 => 8,
                    LogicalDtype::U8 | LogicalDtype::U32 => return None,
                };
                // Divisibility is checked at the existing decode boundary, after
                // names/shapes. Even malformed byte extents never become shape lengths.
                (TransformKind::NativeBytes { width }, n[2], n[2] / width)
            }
            (2 | 3, 0, LogicalDtype::U32) => (TransformKind::Move, n[3], n[3]),
            (2, 1, LogicalDtype::U8) => (TransformKind::Move, n[6], n[6]),
            (3, 1, LogicalDtype::F16) => (TransformKind::HalfBits, n[4], n[4]),
            (3, 2, LogicalDtype::F16) => (TransformKind::HalfBits, n[5], n[5]),
            _ => return None,
        };
        Some(Self {
            dtype,
            kind,
            input_elements,
            output_elements,
            declared_shape_elements: output.shape_elements(),
        })
    }
    pub(crate) fn accepts_input(self, count: usize) -> bool {
        match self.kind {
            TransformKind::NativeBytes { .. } => count == self.input_elements,
            // Affine G2 requests may cover malformed emission beyond shape. The
            // actual G2 converter validates its final shape before this consumer.
            TransformKind::HalfBits | TransformKind::Move => count <= self.input_elements,
        }
    }
    pub(crate) fn requested_layout(self) -> Option<Layout> {
        let n = self.output_elements;
        match self.dtype {
            LogicalDtype::F32 => Layout::array::<f32>(n),
            LogicalDtype::F16 => Layout::array::<half::f16>(n),
            LogicalDtype::Bf16 => Layout::array::<half::bf16>(n),
            LogicalDtype::I8 => Layout::array::<i8>(n),
            LogicalDtype::I16 => Layout::array::<i16>(n),
            LogicalDtype::I32 => Layout::array::<i32>(n),
            LogicalDtype::I64 => Layout::array::<i64>(n),
            LogicalDtype::F64 => Layout::array::<f64>(n),
            LogicalDtype::U8 => Layout::array::<u8>(n),
            LogicalDtype::U32 => Layout::array::<u32>(n),
        }
        .ok()
    }
}

use eredu_runtime::working_memory::{
    HostDestinationCause, OriginalHostDestinationBank, OriginalHostVec,
};

#[derive(Debug)]
pub(crate) enum DestinationCause {
    Layout,
    Reserve(TryReserveError),
    Original(HostDestinationCause),
    Extent { requested: usize, actual: usize },
    Capacity { requested: usize, actual: usize },
    AlreadyFilled,
}

/// Same typed payload with either explicit unqualified Vec storage or its
/// separately admitted owning destination. Neither variant separates custody.
#[derive(Debug)]
pub(crate) enum TypedValues<T> {
    /// A nonowning admission refusal; no Vec or receipt was constructed.
    Absent,
    Unqualified(Vec<T>),
    Admitted(OriginalHostVec<T>),
    Supplied {
        values: OriginalHostVec<T>,
        emitted: usize,
        limit: usize,
    },
}
impl<T> From<Vec<T>> for TypedValues<T> {
    fn from(v: Vec<T>) -> Self {
        Self::Unqualified(v)
    }
}
impl<T> TypedValues<T> {
    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Self::Absent => &[],
            Self::Unqualified(v) => v,
            Self::Admitted(v) => v.as_slice(),
            Self::Supplied {
                values, emitted, ..
            } => &values.as_slice()[..*emitted],
        }
    }
    pub(crate) fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }
    pub(crate) fn len(&self) -> usize {
        self.as_slice().len()
    }
    pub(crate) fn capacity(&self) -> usize {
        match self {
            Self::Absent => 0,
            Self::Unqualified(v) => v.capacity(),
            Self::Admitted(v) => v.capacity(),
            Self::Supplied { values, .. } => values.capacity(),
        }
    }
    pub(crate) fn admitted_capacity(&self) -> Option<usize> {
        match self {
            Self::Admitted(v) => Some(v.requested_elements()),
            Self::Supplied { limit, .. } => Some(*limit),
            _ => None,
        }
    }
}
impl<T: safemlx::ArrayElement> safemlx::OwnedHostCopyBuffer<T> for TypedValues<T> {
    fn as_slice(&self) -> &[T] {
        Self::as_slice(self)
    }
    fn capacity(&self) -> usize {
        Self::capacity(self)
    }
}

#[derive(Debug)]
pub(crate) struct TypedDestination<T> {
    values: TypedValues<T>,
    requested: usize,
    started: bool,
}
impl<T> TypedDestination<T> {
    pub(crate) fn try_new(requested: usize) -> Result<Self, (Self, DestinationCause)> {
        let mut values = Vec::new();
        let cause = if Layout::array::<T>(requested).is_err() {
            Some(DestinationCause::Layout)
        } else {
            values
                .try_reserve_exact(requested)
                .err()
                .map(DestinationCause::Reserve)
        };
        let out = Self {
            values: values.into(),
            requested,
            started: false,
        };
        match cause {
            Some(cause) => Err((out, cause)),
            None => Ok(out),
        }
    }
    pub(crate) fn try_new_admitted(
        requested: usize,
        bank: &mut OriginalHostDestinationBank,
    ) -> Result<Self, (Self, DestinationCause)>
    where
        T: Copy,
    {
        match bank.try_vec(requested) {
            Ok(values) => Ok(Self {
                values: TypedValues::Admitted(values),
                requested,
                started: false,
            }),
            Err(error) => {
                let (cause, values) = error.into_parts();
                Err((
                    Self {
                        values: values.map_or(TypedValues::Absent, TypedValues::Admitted),
                        requested,
                        started: false,
                    },
                    DestinationCause::Original(cause),
                ))
            }
        }
    }
    pub(crate) fn fill<I: IntoIterator<Item = T>>(
        &mut self,
        expected: usize,
        input: I,
    ) -> Result<(), DestinationCause> {
        if self.started {
            return Err(DestinationCause::AlreadyFilled);
        }
        self.started = true;
        if expected > self.requested {
            return Err(DestinationCause::Extent {
                requested: self.requested,
                actual: expected,
            });
        }
        if self.values.capacity() < self.requested {
            return Err(DestinationCause::Capacity {
                requested: self.requested,
                actual: self.values.capacity(),
            });
        }
        match &mut self.values {
            TypedValues::Absent | TypedValues::Supplied { .. } => {
                Err(DestinationCause::AlreadyFilled)
            }
            TypedValues::Admitted(v) => v
                .try_fill(expected, input)
                .map_err(DestinationCause::Original),
            TypedValues::Unqualified(v) => {
                for value in input {
                    if v.len() == expected || v.len() == v.capacity() {
                        return Err(DestinationCause::Extent {
                            requested: expected,
                            actual: v.len().saturating_add(1),
                        });
                    }
                    v.push(value);
                }
                if v.len() != expected {
                    return Err(DestinationCause::Extent {
                        requested: expected,
                        actual: v.len(),
                    });
                }
                Ok(())
            }
        }
    }
    pub(crate) fn values(&self) -> &TypedValues<T> {
        &self.values
    }
    pub(crate) fn into_values(self) -> TypedValues<T> {
        self.values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypedTransformFacts {
    pub(crate) input_elements: usize,
    pub(crate) input_capacity: usize,
    pub(crate) requested_output_elements: usize,
    pub(crate) output_elements: usize,
    pub(crate) output_capacity: usize,
    pub(crate) input_element_bytes: usize,
    pub(crate) output_element_bytes: usize,
    pub(crate) separate_destination: bool,
}

#[cfg(test)]
mod tests;
