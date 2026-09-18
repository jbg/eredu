//! Paid compact binding rows and the shared direct physical-field binder.
use super::*;
use eredu_nn::Parameter;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::{alloc::Layout, mem::{size_of, size_of_val}};

#[derive(Debug, thiserror::Error)]
pub(in crate::backend::nn::shared) enum CompactBindingCause {
    #[error("compact parameter bindings differ from the retained physical fields")]
    Identity,
    #[error("compact parameter binding shape or encoding differs")]
    Geometry,
    #[error("compact parameter binding attempt is spent")]
    Spent,
    #[error("compact parameter binding metadata overflow")]
    Overflow,
    #[error("compact parameter binding source: {0}")]
    Source(#[source] ParameterSourceError),
    #[error("compact parameter binding funding: {0}")]
    Funding(#[source] HostMetadataFundingError),
    #[error("compact parameter binding destination: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct FundedFailure {
    #[source] cause: CompactBindingCause,
    _funding: HostMetadataFunding,
}
pub(in crate::backend::nn::shared) fn binding_error(cause: CompactBindingCause) -> ComputeError {
    ComputeError::backend_retained_source(cause)
}

struct Binding<'a> { name: &'a str, value: Option<Array> }
/// A finite destination whose names borrow the retained compact source table.
/// Arrays move into the existing module; no alias or map is reconstructed.
pub(crate) struct PreparedCompactBindings<'a> {
    rows: Vec<Binding<'a>>,
    limit: usize,
    attempted: bool,
    funding: HostMetadataFunding,
}
impl<'a> PreparedCompactBindings<'a> {
    /// Largest exact fixed binder frame among the three selected grouped leaves.
    /// Row allocation is separate and paid by `layout_bytes`.
    pub(crate) fn binding_control_bytes(rows: usize) -> Option<usize> {
        Some(named_control_bytes::<Self>(rows)?.max(linear_control_bytes()?))
    }

    pub(crate) fn layout_bytes(rows: usize) -> Option<usize> {
        let frames = [size_of::<Self>(), size_of::<Binding<'_>>(),
            size_of::<Result<Self, ComputeError>>(), size_of::<(usize, bool, &'a str, Array)>(),
            Layout::array::<Binding<'a>>(rows).ok()?.size(),
            ComputeError::retained_source_construction_bytes::<FundedFailure>()?,
            ComputeError::retained_source_construction_bytes::<CompactBindingCause>()?];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(crate) fn new(rows: usize, funding: &HostMetadataFunding) -> Result<Self, ComputeError> {
        let failure = |cause| ComputeError::backend_retained_source(FundedFailure {
            cause, _funding: funding.clone(),
        });
        funding.reserve_metadata(Self::layout_bytes(rows).ok_or_else(||failure(CompactBindingCause::Overflow))?)
            .map_err(|cause|failure(CompactBindingCause::Funding(cause)))?;
        let mut values = Vec::new();
        values.try_reserve_exact(rows).map_err(|cause|failure(CompactBindingCause::Allocation(cause)))?;
        Ok(Self { rows: values, limit: rows, attempted: false, funding: funding.clone() })
    }
    /// The actual source supplies its exact lexical order once, without growth.
    pub(crate) fn push(&mut self, name: &'a str, value: Array) -> Result<(), ComputeError> {
        if self.attempted || self.rows.len() == self.limit || name.is_empty()
            || self.rows.last().is_some_and(|previous| previous.name >= name) {
            return Err(self.failure(CompactBindingCause::Identity));
        }
        self.rows.push(Binding { name, value: Some(value) });
        Ok(())
    }
    pub(in crate::backend::nn::shared) fn funding(&self) -> &HostMetadataFunding { &self.funding }
    pub(in crate::backend::nn::shared) fn consumed(&self) -> bool {
        self.attempted && self.rows.len() == self.limit && self.rows.iter().all(|row|row.value.is_none())
    }
    fn index(&self, name: &str) -> Option<usize> {
        self.rows.binary_search_by(|row|row.name.cmp(name)).ok()
    }
}

// Only these two native-owned destinations implement the shared worker input.
pub(in crate::backend::nn::shared) trait CompactBindingValues {
    fn len(&self) -> usize;
    fn ready(&self) -> bool;
    fn value(&self, name: &str) -> Option<&Array>;
    fn take(&mut self, name: &str) -> Option<Array>;
    fn begin(&mut self, controls: Option<usize>) -> Result<(), ComputeError>;
    fn failure(&self, cause: CompactBindingCause) -> ComputeError;
}
impl CompactBindingValues for BTreeMap<String, Array> {
    fn len(&self) -> usize { BTreeMap::len(self) }
    fn ready(&self) -> bool { true }
    fn value(&self, name: &str) -> Option<&Array> { self.get(name) }
    fn take(&mut self, name: &str) -> Option<Array> { self.remove(name) }
    fn begin(&mut self, controls: Option<usize>) -> Result<(), ComputeError> {
        controls.map(|_|()).ok_or_else(||binding_error(CompactBindingCause::Overflow))
    }
    fn failure(&self, cause: CompactBindingCause) -> ComputeError { binding_error(cause) }
}
impl CompactBindingValues for PreparedCompactBindings<'_> {
    fn len(&self) -> usize { self.rows.len() }
    fn ready(&self) -> bool { self.rows.len() == self.limit && self.rows.iter().all(|row|row.value.is_some()) }
    fn value(&self, name: &str) -> Option<&Array> { self.rows.get(self.index(name)?)?.value.as_ref() }
    fn take(&mut self, name: &str) -> Option<Array> {
        let index = self.index(name)?;
        self.rows[index].value.take()
    }
    fn begin(&mut self, controls: Option<usize>) -> Result<(), ComputeError> {
        if self.attempted { return Err(self.failure(CompactBindingCause::Spent)); }
        self.attempted = true;
        let controls = controls.ok_or_else(||self.failure(CompactBindingCause::Overflow))?;
        self.funding.reserve_metadata(controls).map_err(|cause|self.failure(CompactBindingCause::Funding(cause)))
    }
    fn failure(&self, cause: CompactBindingCause) -> ComputeError {
        ComputeError::backend_retained_source(FundedFailure { cause, _funding: self.funding.clone() })
    }
}

pub(in crate::backend::nn::shared) fn validate(placeholder: &Array, value: &Array,
    floating_shape: Option<&[i32]>) -> Result<(), CompactBindingCause> {
    validate_shape(placeholder.shape(), placeholder.dtype(), value, floating_shape)
}
pub(in crate::backend::nn::shared) fn validate_shape(shape: &[i32], dtype: Dtype, value: &Array,
    floating_shape: Option<&[i32]>) -> Result<(), CompactBindingCause> {
    let floating = matches!(value.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16);
    let expected = if floating { floating_shape.unwrap_or(shape) } else { shape };
    if value.shape() != expected || (floating_shape.is_some() && !floating && value.dtype() != dtype) {
        return Err(CompactBindingCause::Geometry);
    }
    Ok(())
}

struct SourceCheck;
impl<'a> ParameterSourceVisitor<'a, MlxTensor> for SourceCheck {
    fn parameter(&mut self, _: ParameterMetadataView<'a>, _: &'a MlxTensor) {}
    fn retained(&mut self, _: &'a MlxTensor) {}
}
struct Validate<'v, V: ?Sized> {
    values: &'v V,
    shapes: &'v [(&'v str, [i32; 3])],
    count: usize,
    failure: Option<CompactBindingCause>,
}
impl<'a, V: CompactBindingValues + ?Sized> NativeParameterSourceVisitor<'a> for Validate<'_, V> {
    fn parameter(&mut self, name: &'static str, placeholder: &'a Array, _: bool) {
        if self.failure.is_some() { return; }
        let Some(value) = self.values.value(name) else { self.failure = Some(CompactBindingCause::Identity);return; };
        self.failure = validate(placeholder, value,
            self.shapes.iter().find(|(key,_)|*key==name).map(|(_,shape)|shape.as_slice())).err();
        match self.count.checked_add(1) {
            Some(count) => self.count = count,
            None => self.failure = Some(CompactBindingCause::Overflow),
        }
    }
    fn retained(&mut self, _: &'a Array) {}
}
struct Move<'v, V: ?Sized> { values: &'v mut V }
impl<'a, V: CompactBindingValues + ?Sized> NativeParameterSourceVisitorMut<'a> for Move<'_, V> {
    fn parameter(&mut self, name: &'static str, value: &'a mut Array, _: bool) {
        *value = self.values.take(name).expect("validated unique physical binding");
    }
}

pub(in crate::backend::nn::shared) fn linear_control_bytes() -> Option<usize> {
    let frames = [size_of::<[Option<&str>;4]>(), size_of::<[i32;3]>(),
        size_of::<[Option<&Parameter<MlxTensor>>;4]>(),size_of::<[Option<&mut Parameter<MlxTensor>>;4]>(),
        size_of::<(usize,&str,&Array,Option<Array>,Option<&[i32]>)>(),
        size_of::<Result<(),ComputeError>>(),size_of::<CompactBindingCause>(),
        Array::descriptor_comparison_control_bytes()?.checked_mul(8)?,
        ComputeError::retained_source_construction_bytes::<FundedFailure>()?];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
pub(super) fn named_control_bytes<V: CompactBindingValues + ?Sized>(rows: usize) -> Option<usize> {
    let frames = [size_of::<Validate<'_,V>>(),size_of::<Move<'_,V>>(),size_of::<SourceCheck>(),
        size_of::<(&V,&mut V,&Array,Option<Array>,usize)>(),size_of::<Result<(),ComputeError>>(),
        size_of::<CompactBindingCause>(),size_of::<[(&str,[i32;3]);2]>(),
        super::sources::binding_visit_control_bytes()?,
        Array::descriptor_comparison_control_bytes()?.checked_mul(rows)?.checked_mul(2)?,
        ComputeError::retained_source_construction_bytes::<FundedFailure>()?];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
pub(super) fn bind_named<M: NativeRetainedValues,V: CompactBindingValues + ?Sized>(
    module: &mut M, topology: &dyn NativeParameterTopology, values: &mut V,
    floating_shapes: &[(&str,[i32;3])]) -> Result<(),ComputeError> {
    values.begin(named_control_bytes::<V>(topology.len()))?;
    if !values.ready() || values.len()!=topology.len() || (0..topology.len()).any(|index|values.value(topology.key(index).expect("immutable topology index")).is_none()) {
        return Err(values.failure(CompactBindingCause::Identity));
    }
    // Reuse the immutable source verifier, including exact field uniqueness.
    // The subsequent mutable walk can only consume those already checked rows.
    super::visit_module_parameter_sources(module,topology,&mut SourceCheck)
        .map_err(|cause|values.failure(CompactBindingCause::Source(cause)))?;
    let mut check=Validate { values:&*values,shapes:floating_shapes,count:0,failure:None };
    module.visit_native_parameter_sources(&mut check)
        .map_err(|cause|values.failure(CompactBindingCause::Source(cause)))?;
    if let Some(cause)=check.failure { return Err(values.failure(cause)); }
    if check.count!=values.len() { return Err(values.failure(CompactBindingCause::Identity)); }
    module.visit_native_parameter_sources_mut(&mut Move { values })
        .map_err(|cause|values.failure(CompactBindingCause::Source(cause)))?;
    Ok(())
}
