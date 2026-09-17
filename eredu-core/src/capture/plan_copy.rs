//! Exact destination copy of an already admitted declaration, without readmission.
use super::*;
use crate::{HostPreparationAuthority, ObservationValueType, TensorAxis};
use std::{alloc::Layout, mem::size_of};

/// Fixed copy failure. Any partial destination retires before caller custody.
#[derive(Debug, thiserror::Error)]
pub enum CapturePlanCopyError {
    /// A checked destination or control size overflowed.
    #[error("capture source copy size overflow")]
    Overflow,
    /// A real destination allocation was refused.
    #[error("capture source allocation failed: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
    /// The allocator did not supply the exact selected destination layout.
    #[error("capture source destination capacity differs from its plan")]
    Capacity,
}

/// Borrowed source and complete destination/control layout. This is descriptive;
/// the caller must reserve it before invoking the consuming copy constructor.
#[derive(Debug)]
pub struct PreparedCapturePlanCopy<'a> {
    source: &'a AdmittedCapturePlan,
    bytes: usize,
}
impl<'a> PreparedCapturePlanCopy<'a> {
    /// Count the same closed DTO worker without allocating a destination.
    pub fn inspect(source: &'a AdmittedCapturePlan) -> Result<Self, CapturePlanCopyError> {
        let mut worker = Worker::new(false)?;
        drop(worker.plan(source)?);
        Ok(Self {
            source,
            bytes: worker.bytes,
        })
    }
    /// Fixed nonallocating census stack, paid by the actual preparation caller.
    pub fn inspection_control_bytes() -> Option<usize> {
        [
            size_of::<Worker>(),
            size_of::<Self>(),
            size_of::<AdmittedCapturePlan>(),
            size_of::<CaptureSelection>(),
            size_of::<CaptureSlice>(),
            size_of::<ObservationPoint>(),
            size_of::<TensorAxis>(),
            size_of::<String>(),
            size_of::<CapturePlanCopyError>(),
            size_of::<Result<Self, CapturePlanCopyError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|n| n.checked_mul(2))
    }
    /// Complete copy, shared-source shell, and fixed return/error controls.
    /// Constructing the supplied host token is the caller's separate obligation.
    pub const fn required_bytes(&self) -> usize {
        self.bytes
    }
    /// Fresh destination only. Existing plan buffers are never moved into custody.
    /// Source semantics/digest are copied exactly; this issues no new admission.
    pub fn copy(
        self,
        host: HostPreparationAuthority,
    ) -> Result<SharedCapturePlan, CapturePlanCopyError> {
        let mut worker = Worker::new(true)?;
        let plan = worker.plan(self.source)?;
        if worker.bytes != self.bytes {
            return Err(CapturePlanCopyError::Capacity);
        }
        Ok(SharedCapturePlan::from_prepared_copy(plan, host))
    }
}
pub(crate) struct Worker {
    emit: bool,
    bytes: usize,
}
impl Worker {
    /// Shared closed DTO destination primitives; the semantic caller supplies
    /// its concrete owner/query controls before entering the same copy worker.
    pub(crate) fn with_controls(emit: bool, bytes: usize) -> Self {
        Self { emit, bytes }
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    fn new(emit: bool) -> Result<Self, CapturePlanCopyError> {
        let mut value = Self { emit, bytes: 0 };
        value.add(
            size_of::<Self>()
                + size_of::<PreparedCapturePlanCopy<'_>>()
                + size_of::<AdmittedCapturePlan>() * 2
                + size_of::<HostPreparationAuthority>()
                + size_of::<CapturePlanCopyError>()
                + size_of::<Result<SharedCapturePlan, CapturePlanCopyError>>(),
        )?;
        value.add(
            usize::try_from(
                SharedCapturePlan::new_owner_control_bytes()
                    .ok_or(CapturePlanCopyError::Overflow)?,
            )
            .map_err(|_| CapturePlanCopyError::Overflow)?,
        )?;
        value.add(
            usize::try_from(
                SharedCapturePlan::ordinary_host_control_bytes()
                    .ok_or(CapturePlanCopyError::Overflow)?,
            )
            .map_err(|_| CapturePlanCopyError::Overflow)?,
        )?;
        Ok(value)
    }
    pub(crate) fn add(&mut self, bytes: usize) -> Result<(), CapturePlanCopyError> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or(CapturePlanCopyError::Overflow)?;
        Ok(())
    }
    pub(crate) fn text(&mut self, source: &str) -> Result<String, CapturePlanCopyError> {
        self.add(source.len())?;
        self.add(size_of::<&str>() + size_of::<(&mut Self, String)>())?;
        self.add(
            size_of::<String>() * 2
                + size_of::<Result<String, CapturePlanCopyError>>()
                + size_of::<std::collections::TryReserveError>(),
        )?;
        let mut output = String::new();
        if self.emit {
            output.try_reserve_exact(source.len())?;
            if output.capacity() != source.len() {
                return Err(CapturePlanCopyError::Capacity);
            }
            output.push_str(source);
        }
        Ok(output)
    }
    /// Exact concatenation used by immutable companion identities; no temporary
    /// formatted String is allocated during counting or construction.
    pub(crate) fn text_parts(&mut self, parts: &[&str]) -> Result<String, CapturePlanCopyError> {
        let length = parts
            .iter()
            .try_fold(0usize, |n, part| n.checked_add(part.len()))
            .ok_or(CapturePlanCopyError::Overflow)?;
        self.add(length)?;
        self.add(
            size_of::<(&mut Self, &[&str], usize)>()
                + size_of::<String>() * 2
                + size_of::<Result<String, CapturePlanCopyError>>()
                + size_of::<std::collections::TryReserveError>(),
        )?;
        let mut output = String::new();
        if self.emit {
            output.try_reserve_exact(length)?;
            if output.capacity() != length {
                return Err(CapturePlanCopyError::Capacity);
            }
            for part in parts {
                output.push_str(part);
            }
        }
        Ok(output)
    }
    pub(crate) fn vector<S, T>(
        &mut self,
        source: &[S],
        mut copy: impl FnMut(&mut Self, &S) -> Result<T, CapturePlanCopyError>,
    ) -> Result<Vec<T>, CapturePlanCopyError> {
        self.add(
            Layout::array::<T>(source.len())
                .map_err(|_| CapturePlanCopyError::Overflow)?
                .size(),
        )?;
        self.add(
            size_of::<Vec<T>>() * 2
                + size_of::<T>() * 2
                + size_of::<Result<Vec<T>, CapturePlanCopyError>>()
                + size_of::<std::collections::TryReserveError>(),
        )?;
        self.add(std::mem::size_of_val(&copy))?;
        self.add(size_of::<(&mut Self, &[S], usize)>())?;
        let mut output = Vec::new();
        if self.emit {
            output.try_reserve_exact(source.len())?;
            if size_of::<T>() != 0 && output.capacity() != source.len() {
                return Err(CapturePlanCopyError::Capacity);
            }
        }
        for source in source {
            let value = copy(self, source)?;
            if self.emit {
                output.push(value);
            }
        }
        Ok(output)
    }
    fn plan(
        &mut self,
        source: &AdmittedCapturePlan,
    ) -> Result<AdmittedCapturePlan, CapturePlanCopyError> {
        let AdmittedCapturePlan {
            plan,
            points,
            request,
            invocation_bounds,
            text_origin,
            identity,
        } = source;
        let CapturePlan {
            schema_version,
            selections,
            limits,
        } = plan;
        Ok(AdmittedCapturePlan {
            plan: CapturePlan {
                schema_version: *schema_version,
                limits: {
                    let CaptureLimits {
                        per_step,
                        cumulative,
                        physical_native_bytes,
                        on_limit,
                    } = limits;
                    CaptureLimits {
                        per_step: *per_step,
                        cumulative: *cumulative,
                        physical_native_bytes: *physical_native_bytes,
                        on_limit: *on_limit,
                    }
                },
                selections: self.vector(selections, Self::selection)?,
            },
            points: self.vector(points, Self::point)?,
            request: *request,
            invocation_bounds: *invocation_bounds,
            text_origin: *text_origin,
            identity: self.text(identity)?,
        })
    }
    fn selection(
        &mut self,
        value: &CaptureSelection,
    ) -> Result<CaptureSelection, CapturePlanCopyError> {
        let CaptureSelection {
            id,
            path,
            schedule,
            slices,
            transform,
        } = value;
        let transform = match transform {
            CaptureTransform::Histogram { edges } => CaptureTransform::Histogram {
                edges: self.vector(edges, |_, v| Ok(*v))?,
            },
            CaptureTransform::TokenScores { token_ids } => CaptureTransform::TokenScores {
                token_ids: self.vector(token_ids, |_, v| Ok(*v))?,
            },
            CaptureTransform::RoutedUnits => CaptureTransform::RoutedUnits,
            CaptureTransform::Preview { max_elements } => CaptureTransform::Preview {
                max_elements: *max_elements,
            },
            CaptureTransform::Slice => CaptureTransform::Slice,
            CaptureTransform::FullTensor => CaptureTransform::FullTensor,
            CaptureTransform::Summary => CaptureTransform::Summary,
            CaptureTransform::TopCandidates { count } => {
                CaptureTransform::TopCandidates { count: *count }
            }
        };
        Ok(CaptureSelection {
            id: self.text(id)?,
            path: self.text(path)?,
            schedule: {
                let CaptureSchedule {
                    prefill,
                    decode,
                    first_prediction,
                    end_prediction,
                    every,
                } = schedule;
                CaptureSchedule {
                    prefill: *prefill,
                    decode: *decode,
                    first_prediction: *first_prediction,
                    end_prediction: *end_prediction,
                    every: *every,
                }
            },
            transform,
            slices: self.vector(slices, |w, v| {
                let CaptureSlice {
                    axis,
                    start,
                    end,
                    stride,
                } = v;
                Ok(CaptureSlice {
                    axis: w.text(axis)?,
                    start: *start,
                    end: *end,
                    stride: *stride,
                })
            })?,
        })
    }
    fn point(
        &mut self,
        value: &ObservationPoint,
    ) -> Result<ObservationPoint, CapturePlanCopyError> {
        let ObservationPoint {
            path,
            node_id,
            meaning,
            value_type,
            dtype,
            axes,
            prefill,
            decode,
            requirements,
            position,
            retained_bytes,
            host_bytes,
        } = value;
        let value_type = match value_type {
            ObservationValueType::Tensor => ObservationValueType::Tensor,
            ObservationValueType::RoutedUnits { routing, geometry } => {
                ObservationValueType::RoutedUnits {
                    routing: self.text(routing)?,
                    geometry: *geometry,
                }
            }
        };
        Ok(ObservationPoint {
            path: self.text(path)?,
            node_id: self.text(node_id)?,
            meaning: self.text(meaning)?,
            value_type,
            dtype: *dtype,
            axes: axes
                .as_ref()
                .map(|axes| {
                    self.vector(axes, |w, v| {
                        let TensorAxis { name, dimension } = v;
                        let dimension = match dimension {
                            SymbolicDimension::Known(n) => SymbolicDimension::Known(*n),
                            SymbolicDimension::Batch => SymbolicDimension::Batch,
                            SymbolicDimension::Sequence => SymbolicDimension::Sequence,
                            SymbolicDimension::TokenRows => SymbolicDimension::TokenRows,
                            SymbolicDimension::Context => SymbolicDimension::Context,
                            SymbolicDimension::MediaPositions => SymbolicDimension::MediaPositions,
                            SymbolicDimension::Unknown => SymbolicDimension::Unknown,
                        };
                        Ok(TensorAxis {
                            name: w.text(name)?,
                            dimension,
                        })
                    })
                })
                .transpose()?,
            prefill: *prefill,
            decode: *decode,
            requirements: self.vector(requirements, |_, v| Ok(*v))?,
            position: *position,
            retained_bytes: *retained_bytes,
            host_bytes: *host_bytes,
        })
    }
}
