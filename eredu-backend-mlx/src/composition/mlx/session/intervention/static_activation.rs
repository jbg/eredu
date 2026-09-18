//! Shared static activation primitives for ordinary, cold, and original execution.
use super::*;
use crate::backend::array_copy::{CaptureCompletion, CaptureTensorNativeError as Failure};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype as D, WorkspaceFloatingType as F, WorkspaceOperationKind as K,
    WorkspaceTensor,
};
use eredu_runtime::capture::CaptureExecutionError;
use half::slice::HalfBitsSliceExt;
use safemlx::{OriginalScopeObserver, PreparedArrayClone};
use std::{
    cell::RefCell,
    mem::{size_of, size_of_val},
};

mod native;
mod trace;
mod sparse;
pub(crate) use sparse::{PreparedSparseActivation, SparseActivationFailure, SparseScalarEnvelope};
use native::Native;

#[derive(Clone, Copy)]
struct Shape {
    axes: [i32; 32],
    rank: usize,
}
impl Shape {
    fn new(shape: &[u64]) -> Result<Self, Failure> {
        if shape.len() > 32 {
            return Err(Failure::ShapeMismatch);
        }
        let mut result = Self {
            axes: [0; 32],
            rank: shape.len(),
        };
        for (out, input) in result.axes.iter_mut().zip(shape) {
            *out = i32::try_from(*input).map_err(|_| Failure::GeometryOverflow)?;
        }
        Ok(result)
    }
    fn slice(&self) -> &[i32] {
        &self.axes[..self.rank]
    }
}
#[derive(Clone, Copy)]
pub(super) enum Binary {
    Add,
    Multiply,
}
#[derive(Clone, Copy)]
pub(super) enum IndexSource<'a> {
    Actual(&'a [i32]),
    // A source-derived capacity used only by count/cold equation workers.
    // The native worker cannot interpret it as concrete route indices.
    Bound(usize),
}
impl IndexSource<'_> {
    fn len(self) -> usize { match self { Self::Actual(values) => values.len(), Self::Bound(n) => n } }
}
pub(super) enum Op<'a, V> {
    Indices(IndexSource<'a>),
    Reshape(&'a V, &'a [i32]),
    IndexedSelect(&'a V, &'a V),
    IndexedUpdate(&'a V, &'a V, &'a V),
    Select(&'a V, &'a ResolvedCaptureSlice),
    Update(&'a V, &'a ResolvedCaptureSlice, &'a V),
    Zero(&'a [u64], InterventionDtype),
    Scalar(f32),
    Cast(&'a V, InterventionDtype),
    Broadcast(&'a V, &'a V),
    Mask(&'a [bool], &'a V),
    Columns(&'a [u32], bool, i32),
    Tensor(&'a InterventionTensor),
    Binary(Binary, &'a V, &'a V),
    Where(&'a V, &'a V, &'a V),
}
pub(super) trait Kernel {
    type Value;
    fn shape<'a>(&self, value: &'a Self::Value) -> &'a [i32];
    fn dtype(&self, value: &Self::Value) -> Result<InterventionDtype, Failure>;
    fn emit(&mut self, op: Op<'_, Self::Value>) -> Result<Self::Value, Failure>;
}
pub(super) struct Adapter<T>(T);
impl<T: Kernel> CaptureBackend for Adapter<T> {
    type Tensor = T::Value;
    type Error = Failure;
    fn shape(&self, value: &Self::Tensor) -> Result<Vec<u64>, Failure> {
        Ok(self.0.shape(value).iter().map(|x| *x as u64).collect())
    }
    fn estimate(
        &self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "activation primitive has no capture transform".into(),
        ))
    }
    fn transform(
        &mut self,
        _: &Self::Tensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Failure> {
        Err(Failure::ClaimMismatch)
    }
}
impl<T: Kernel> InterventionBackend for Adapter<T> {
    fn matches_intervention_shape(
        &self,
        value: &Self::Tensor,
        shape: &[u64],
    ) -> Result<bool, Failure> {
        let actual = self.0.shape(value);
        Ok(actual.len() == shape.len()
            && actual
                .iter()
                .zip(shape)
                .all(|(a, b)| u64::try_from(*a).ok() == Some(*b)))
    }
    fn intervention_dtype(&self, value: &Self::Tensor) -> Result<InterventionDtype, Failure> {
        self.0.dtype(value)
    }
    fn validate_intervention_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        NativeInterventionEstimator.validate_geometry(source, slice)
    }
    fn select_region(
        &mut self,
        value: &Self::Tensor,
        slice: &ResolvedCaptureSlice,
    ) -> Result<Self::Tensor, Failure> {
        self.0.emit(Op::Select(value, slice))
    }
    fn update_region(
        &mut self,
        value: &Self::Tensor,
        slice: &ResolvedCaptureSlice,
        replacement: &Self::Tensor,
    ) -> Result<Self::Tensor, Failure> {
        self.0.emit(Op::Update(value, slice, replacement))
    }
    fn zeros(&mut self, shape: &[u64], dtype: InterventionDtype) -> Result<Self::Tensor, Failure> {
        self.0.emit(Op::Zero(shape, dtype))
    }
    fn scale(&mut self, value: &Self::Tensor, factor: f32) -> Result<Self::Tensor, Failure> {
        let scalar = self.0.emit(Op::Scalar(factor))?;
        let scalar = self.0.emit(Op::Cast(&scalar, self.0.dtype(value)?))?;
        self.0.emit(Op::Binary(Binary::Multiply, value, &scalar))
    }
    fn fill_masked(
        &mut self,
        value: &Self::Tensor,
        keep: &[bool],
        fill: f32,
    ) -> Result<Self::Tensor, Failure> {
        let mask = self.0.emit(Op::Mask(keep, value))?;
        self.fill(&mask, value, fill)
    }
    fn mask_components(
        &mut self,
        value: &Self::Tensor,
        ids: &[u32],
        keep: bool,
    ) -> Result<Self::Tensor, Failure> {
        self.columns(value, ids, keep, 0.0)
    }
    fn realize_tensor(&mut self, tensor: &InterventionTensor) -> Result<Self::Tensor, Failure> {
        self.0.emit(Op::Tensor(tensor))
    }
    fn add(&mut self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Failure> {
        self.0.emit(Op::Binary(Binary::Add, a, b))
    }
    fn fill_columns(
        &mut self,
        value: &Self::Tensor,
        ids: &[u32],
        fill: f32,
    ) -> Result<Self::Tensor, Failure> {
        self.columns(value, ids, false, fill)
    }
}
impl<T: Kernel> Adapter<T> {
    pub(super) fn indexed_select(&mut self, source: &T::Value, indices: &T::Value) -> Result<T::Value, Failure> {
        self.0.emit(Op::IndexedSelect(source, indices))
    }
    pub(super) fn indexed_update(&mut self, source: &T::Value, indices: &T::Value, update: &T::Value) -> Result<T::Value, Failure> {
        self.0.emit(Op::IndexedUpdate(source, indices, update))
    }
    fn fill(&mut self, mask: &T::Value, value: &T::Value, fill: f32) -> Result<T::Value, Failure> {
        let scalar = self.0.emit(Op::Scalar(fill))?;
        let scalar = self.0.emit(Op::Cast(&scalar, self.0.dtype(value)?))?;
        self.0.emit(Op::Where(mask, value, &scalar))
    }
    fn columns(
        &mut self,
        value: &T::Value,
        ids: &[u32],
        keep: bool,
        fill: f32,
    ) -> Result<T::Value, Failure> {
        let width = *self.0.shape(value).last().ok_or(Failure::ShapeMismatch)?;
        let mask = self.0.emit(Op::Columns(ids, keep, width))?;
        let mask = self.0.emit(Op::Broadcast(&mask, value))?;
        self.fill(&mask, value, fill)
    }
}

/// Borrowed immutable source geometry; no admission, native work or owned plan.
pub(crate) struct PreparedStaticActivation<'a> {
    action: &'a InterventionAction,
    slice: &'a ResolvedCaptureSlice,
    source: &'a [u64],
    dtype: InterventionDtype,
}
/// Actual safe output handles and host staging for the selected shared worker.
pub(crate) struct StaticActivationPopulation {
    pub retained_roots: usize,
    pub completions: usize,
    pub host_bytes: usize,
    pub controls: usize,
}
impl<'a> PreparedStaticActivation<'a> {
    /// Fixed constructor/count/validation frames, reserved against source H
    /// before inspecting an action. No native roots or payload are constructed.
    pub(crate) fn inspection_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>() * 2,
            size_of::<Adapter<trace::Count>>(),
            size_of::<trace::Count>(),
            size_of::<[trace::Value; 4]>(),
            size_of::<[Shape; 4]>(),
            size_of::<Op<'static, trace::Value>>(),
            size_of::<Result<trace::Value, CaptureExecutionError<Failure>>>(),
            size_of::<Result<Self, Failure>>(),
            size_of::<Result<trace::Count, Failure>>(),
            size_of::<Result<StaticActivationPopulation, Failure>>(),
            size_of::<(
                &InterventionAction,
                &ResolvedCaptureSlice,
                &[u64],
                InterventionDtype,
            )>(),
            size_of::<[&[u64]; 5]>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'static, u32>>>(),
            size_of::<std::slice::Iter<'static, u64>>(),
            size_of::<[usize; 12]>(),
            size_of::<[Option<u32>; 2]>(),
            size_of::<[Option<InterventionDtype>; 2]>(),
            size_of::<Result<InterventionDtype, Failure>>(),
            size_of::<CaptureError>(),
            size_of::<Failure>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    pub(crate) fn new(
        action: &'a InterventionAction,
        slice: &'a ResolvedCaptureSlice,
        source: &'a [u64],
        dtype: InterventionDtype,
    ) -> Result<Self, Failure> {
        Shape::new(source)?;
        Shape::new(&slice.shape)?;
        let starts = Shape::new(&slice.starts)?;
        let ends = Shape::new(&slice.ends)?;
        let strides = Shape::new(&slice.strides)?;
        for ((a, b), step) in starts.slice().iter().zip(ends.slice()).zip(strides.slice()) {
            if *step <= 0
                || b.checked_sub(*a)
                    .and_then(|width| width.checked_add(*step - 1))
                    .is_none()
            {
                return Err(Failure::GeometryOverflow);
            }
        }

        let value = Self {
            action,
            slice,
            source,
            dtype,
        };
        // Execute the real portable validation and action worker on fixed
        // metadata. The counter has no device, stream, tensor or allocation.
        value.count()?;
        Ok(value)
    }
    fn count(&self) -> Result<trace::Count, Failure> {
        let input = trace::Value {
            shape: Shape::new(self.source)?,
            dtype: Some(self.dtype),
        };
        let mut worker = Adapter(trace::Count::default());
        eredu_runtime::intervention::apply_activation_with_source_shape(
            &mut worker,
            &input,
            self.action,
            self.slice,
            self.source,
        )
        .map_err(policy_failure)?;
        Ok(worker.0)
    }
    pub(crate) fn population(&self) -> Result<StaticActivationPopulation, Failure> {
        let count = self.count()?;
        let controls = control_bytes(count.roots).ok_or(Failure::GeometryOverflow)?;
        Ok(StaticActivationPopulation {
            retained_roots: count.roots,
            completions: count.roots,
            host_bytes: count.host_bytes,
            controls,
        })
    }
    pub(crate) fn validate_workspace_source(input:&WorkspaceTensor,dtype:InterventionDtype)->Result<(),Failure> {
        trace::validate_precision(input,dtype)
    }
    pub(crate) fn trace(
        &self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<WorkspaceTensor, Failure> {
        Self::validate_workspace_source(input, self.dtype)?;
        context
            .charge_metadata(trace::control_bytes().ok_or(Failure::GeometryOverflow)?)
            .map_err(eredu_nn::Error::from)?;
        let input = trace::TracedValue {
            tensor: input.clone(),
            dtype: Some(self.dtype),
        };
        let mut worker = Adapter(trace::Trace { context, retained });
        eredu_runtime::intervention::apply_activation_with_source_shape(
            &mut worker,
            &input,
            self.action,
            self.slice,
            self.source,
        )
        .map(|value| value.tensor)
        .map_err(policy_failure)
    }
    pub(crate) fn execute_array(
        &self,
        input: &Array,
        stream: &Stream,
        observer: &OriginalScopeObserver,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<Array, Failure> {
        let completion = CaptureCompletion::Original(observer);
        completion.validate_identity()?;
        let population = self.population()?;
        {
            let roots = roots.try_borrow().map_err(|_| Failure::CollectorBusy)?;
            if roots.capacity() - roots.len() < population.retained_roots {
                return Err(Failure::GeometryOverflow);
            }
        }
        let mut worker = Adapter(Native {
            stream,
            completion,
            roots: Some(roots),
        });
        if worker.intervention_dtype(input)? != self.dtype {
            return Err(Failure::ShapeMismatch);
        }
        eredu_runtime::intervention::apply_activation_with_source_shape(
            &mut worker,
            input,
            self.action,
            self.slice,
            self.source,
        )
        .map_err(policy_failure)
    }
}
// Validation failures are impossible after exact source/admitted-plan matching,
// but retain their original typed policy cause rather than format a diagnostic.
fn policy_failure(cause: CaptureExecutionError<Failure>) -> Failure {
    match cause {
        CaptureExecutionError::Backend(cause) => cause,
        CaptureExecutionError::Admission(cause) => Failure::ActivationPolicy(cause),
    }
}
fn control_bytes(roots: usize) -> Option<usize> {
    let frames = [
        size_of::<PreparedStaticActivation<'static>>() * 2,
        size_of::<StaticActivationPopulation>(),
        size_of::<Adapter<Native<'static, 'static>>>(),
        size_of::<Adapter<trace::Count>>(),
        size_of::<Op<'static, Array>>(),
        size_of::<[Shape; 4]>(),
        size_of::<[Array; 3]>(),
        size_of::<[Result<Array, Failure>; 3]>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<Array, CaptureExecutionError<Failure>>>(),
        size_of::<Result<MlxTensor, Failure>>(),
        size_of::<[&[u64]; 5]>(),
        size_of::<[&Array; 4]>(),
        size_of::<Vec<bool>>(),
        size_of::<CaptureError>(),
        size_of::<CaptureExecutionError<Failure>>(),
        // Fixed runtime diagnostics from source/region/rank/output comparison;
        // exactly one error String can be created by this checked worker.
        [
            "activation source shape mismatch",
            "activation slice rank mismatch",
            "invalid activation region",
            "column masks require the complete final axis",
            "native activation primitive changed shape or dtype",
        ]
        .iter()
        .map(|s| s.len())
        .max()?,
        size_of::<Result<(), std::collections::TryReserveError>>(),
        size_of::<std::cell::RefMut<'static, Vec<Array>>>(),
        size_of::<std::cell::Ref<'static, Vec<Array>>>(),
        OriginalScopeObserver::control_bytes()?,
        Array::static_slice_update_control_bytes()?,
        // The native worker settles every emitted root, including values
        // disconnected by a later full replacement. Match its exact fixed
        // one-root completion worker and borrowed return/control carriers.
        safemlx::OperationEvent::nested_completion_control_bytes::<1>()?
            .checked_mul(roots)?,
        size_of::<(&Array, &Stream, CaptureCompletion<'static>)>(),
        size_of::<safemlx::EvaluatedArray<'static>>(),
        size_of::<Result<safemlx::EvaluatedArray<'static>, Failure>>(),
        PreparedArrayClone::control_bytes()?
            .checked_add(Array::inspection_clone_handle_bytes())?
            .checked_mul(roots)?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

/// Ordinary execution delegates to the same native primitive worker. Its
/// caller retains the ordinary capture owner and establishes completion.
pub(super) fn ordinary(
    stream: &Stream,
    action: impl FnOnce(&mut Adapter<Native<'_, '_>>) -> Result<Array, Failure>,
) -> Result<MlxTensor, Error> {
    action(&mut Adapter(Native {
        stream,
        completion: CaptureCompletion::Ordinary,
        roots: None,
    }))
    .map(MlxTensor::from_array)
    .map_err(|cause| Error::Neural(eredu_nn::Error::backend_retained_source(cause)))
}
