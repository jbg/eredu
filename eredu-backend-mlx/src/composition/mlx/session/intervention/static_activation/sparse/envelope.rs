//! Explicit prospective index extent over an actual fixed Units source layout.
use super::*;
use eredu_runtime::intervention::RoutedInterventionNumericalAction as Numerical;

/// This envelope contains no route rows or selected indices and cannot execute.
/// The source owner must additionally validate original declaration identity,
/// actual callback coordinates, and its exact selected index count. A zero
/// count keeps the same five-source validation but constructs no numerical edit.
pub(crate) struct SparseScalarEnvelope {
    shape: [i32; 2],
    dtype: InterventionDtype,
    index_count: usize,
    action: Option<InterventionAction>,
    slice: ResolvedCaptureSlice,
    _funding: HostMetadataFunding,
}
impl SparseScalarEnvelope {
    pub(crate) fn prepare(action: &InterventionAction, shape: [i32; 2], index_count: usize, context: &WorkspaceContext)
        -> Result<Self, Failure> {
        context.charge_metadata(Self::control_bytes().ok_or(Failure::GeometryOverflow)?)
            .map_err(eredu_nn::Error::from)?;
        let funding = context.metadata_funding().ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)
            .map_err(eredu_nn::Error::from)?;
        let (dtype, action) = match Numerical::from_action(action).map_err(|_| Failure::ClaimMismatch)? {
            Numerical::Zero(dtype) => (dtype, InterventionAction::Zero { dtype }),
            Numerical::Scale(dtype, factor) => (dtype, InterventionAction::Scale { dtype, factor }),
            // These need the actual borrowed gathered-payload source adapter.
            // A scalar envelope cannot stand in for a replacement/additive source.
            Numerical::Replace(_) | Numerical::Add(_) => return Err(Failure::ClaimMismatch),
        };
        let count = shape[0].checked_mul(shape[1]).filter(|_| shape.iter().all(|n| *n > 0))
            .ok_or(Failure::GeometryOverflow)?;
        if index_count > usize::try_from(count).map_err(|_| Failure::GeometryOverflow)? {
            return Err(Failure::ShapeMismatch);
        }
        let count = u64::try_from(index_count).map_err(|_| Failure::GeometryOverflow)?;
        let action = (index_count != 0).then_some(action);
        let mut axis = |value| -> Result<Vec<u64>, Failure> {
            let mut v = context.metadata_vec(1)?; v.push(value); Ok(v)
        };
        let slice = ResolvedCaptureSlice { starts: axis(0)?, ends: axis(count as u64)?,
            strides: axis(1)?, shape: axis(count as u64)? };
        Ok(Self { shape, dtype, index_count, action, slice, _funding: funding })
    }
    pub(crate) fn index_count(&self) -> usize { self.index_count }
    pub(crate) fn population(&self) -> Result<StaticActivationPopulation, Failure> {
        let input = trace::Value { shape: Shape::new(&self.shape.map(|n| n as u64))?, dtype: Some(self.dtype) };
        let mut worker = Adapter(trace::Count::default());
        run_selected(&mut worker, &input, self.shape, self.dtype, IndexSource::Bound(self.index_count),
            self.action.as_ref(), &self.slice)?;
        Ok(StaticActivationPopulation { retained_roots: worker.0.roots, completions: worker.0.roots,
            host_bytes: worker.0.host_bytes,
            controls: control_bytes(worker.0.roots).and_then(|n| n.checked_add(PreparedSparseActivation::inspection_control_bytes()?))
                .and_then(|n| n.checked_add(Self::control_bytes()?))
                .and_then(|n| n.checked_add(Array::flat_index_update_control_bytes()?)).ok_or(Failure::GeometryOverflow)?,
        })
    }
    pub(crate) fn trace_bound(&self, input: &WorkspaceTensor, context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>) -> Result<Option<WorkspaceTensor>, Failure> {
        PreparedStaticActivation::validate_workspace_source(input, self.dtype)?;
        context.charge_metadata(Self::control_bytes().and_then(|n| n.checked_add(trace::control_bytes()?))
            .ok_or(Failure::GeometryOverflow)?).map_err(eredu_nn::Error::from)?;
        let input = trace::TracedValue { tensor: input.clone(), dtype: Some(self.dtype) };
        let mut worker = Adapter(trace::Trace { context, retained });
        run_selected(&mut worker, &input, self.shape, self.dtype, IndexSource::Bound(self.index_count),
            self.action.as_ref(), &self.slice).map(|value| value.map(|v| v.tensor))
    }
    /// This checks the structural refinement only. It grants no source identity,
    /// accounting, native scope, completed rows or execution permission.
    pub(crate) fn validate_refinement(&self, actual: &PreparedSparseActivation) -> Result<(), Failure> {
        if actual.shape != self.shape || actual.dtype != self.dtype || actual.indices.len() != self.index_count {
            return Err(Failure::ShapeMismatch);
        }
        match actual.lowered.action() {
            None if actual.indices.is_empty() => Ok(()),
            Some(action) if Some(action) == self.action.as_ref() => Ok(()),
            _ => Err(Failure::ClaimMismatch),
        }
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [size_of::<Self>(), size_of::<Result<Self, Failure>>(),
            size_of::<IndexSource<'_>>(), size_of::<Numerical<'_>>(),
            size_of::<(&Self, &PreparedSparseActivation)>(),
            size_of::<(&InterventionAction, [i32; 2], usize, &WorkspaceContext)>(),
            size_of::<Result<(), Failure>>(), size_of::<[Option<WorkspaceTensor>; 2]>(),
            size_of::<Result<Option<WorkspaceTensor>, Failure>>(),
            size_of::<(&Self, &WorkspaceTensor, &WorkspaceContext, &mut Vec<WorkspaceTensor>)>(),
        ];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
