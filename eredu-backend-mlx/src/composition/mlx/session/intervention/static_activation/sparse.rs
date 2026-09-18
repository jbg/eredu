//! Exact lowered sparse recipe retained through index ingress and execution.
use super::*;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::intervention::PreparedRoutedIntervention;
mod envelope;
pub(crate) use envelope::SparseScalarEnvelope;

/// Actual completed rows and paid signed-index destination. This owns no
/// native grant; its caller must retain the accepted numerical role and roots.
#[derive(Debug)]
pub(crate) struct PreparedSparseActivation {
    lowered: PreparedRoutedIntervention,
    indices: Vec<i32>,
    shape: [i32; 2],
    dtype: InterventionDtype,
    // All aliases and C above retire before their concrete allocation account.
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
#[error("original sparse native preparation failed: {cause}")]
pub(crate) struct SparseActivationFailure {
    #[source]
    cause: Failure,
    _owner: PreparedSparseActivation,
}
impl PreparedSparseActivation {
    /// Convert only the actual native-row-order indices, under an existing
    /// metadata account. Partial construction retains the original row owner.
    pub(crate) fn prepare(lowered: PreparedRoutedIntervention, shape: [i32; 2],
        dtype: InterventionDtype, funding: HostMetadataFunding,
    ) -> Result<Self, SparseActivationFailure> {
        let mut owner = Self { lowered, indices: Vec::new(), shape, dtype, funding };
        let result = (|| -> Result<(), Failure> {
            owner.funding.reserve_metadata(Self::inspection_control_bytes().ok_or(Failure::GeometryOverflow)?)
                .map_err(eredu_nn::workspace::WorkspaceMetadataError::from).map_err(eredu_nn::Error::from)?;
            let elements = shape[0].checked_mul(shape[1]).filter(|_| shape.iter().all(|n| *n > 0))
                .ok_or(Failure::GeometryOverflow)?;
            let operation = owner.lowered.coordinates().0;
            let declaration = owner.lowered.source().plan().admission();
            let bank = declaration.points().get(operation).and_then(|point| point.routed_units.as_ref())
                .ok_or(Failure::ClaimMismatch)?.geometry;
            let (_, range) = owner.lowered.source_chunk();
            let rows = range[1].checked_sub(range[0]).and_then(|n| n.checked_mul(bank.routes_per_token))
                .ok_or(Failure::GeometryOverflow)?;
            if u64::try_from(shape[0]).ok() != Some(rows)
                || u64::try_from(shape[1]).ok() != Some(bank.units_per_expert)
                || declaration.plan().operations.get(operation).and_then(|operation| operation.action.dtype()) != Some(dtype) {
                return Err(Failure::ClaimMismatch);
            }
            let count = owner.lowered.indices().len();
            i32::try_from(count).map_err(|_| Failure::GeometryOverflow)?;
            if owner.lowered.action().is_some() != (count != 0) {
                return Err(Failure::ClaimMismatch);
            }
            owner.indices = owner.funding.metadata_vec(count)?;
            for &index in owner.lowered.indices() {
                let index = i32::try_from(index).map_err(|_| Failure::GeometryOverflow)?;
                if index < 0 || index >= elements { return Err(Failure::ShapeMismatch); }
                owner.indices.push(index);
            }
            owner.population()?;
            Ok(())
        })();
        match result { Ok(()) => Ok(owner), Err(cause) => Err(SparseActivationFailure { cause, _owner: owner }) }
    }
    pub(crate) fn lowered(&self) -> &PreparedRoutedIntervention { &self.lowered }
    pub(crate) fn inspection_control_bytes() -> Option<usize> {
        let frames = [size_of::<Self>() * 2, size_of::<SparseActivationFailure>(),
            size_of::<Result<Self, SparseActivationFailure>>(),
            size_of::<[Shape; 2]>(), size_of::<[trace::Value; 2]>(),
            size_of::<Adapter<trace::Count>>(), size_of::<Result<Option<trace::Value>, Failure>>(),
            size_of::<Result<(), Failure>>(), size_of::<std::slice::Iter<'static, u64>>(),
            size_of::<[i32; 5]>(), size_of::<[usize; 4]>(), size_of::<[u64; 5]>(),
            size_of::<RoutedUnitGeometry>(), size_of::<[&InterventionAction; 2]>(),
            PreparedStaticActivation::inspection_control_bytes()?];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    fn run<T: Kernel>(&self, worker: &mut Adapter<T>, input: &T::Value) -> Result<Option<T::Value>, Failure> {
        run_selected(worker, input, self.shape, self.dtype, IndexSource::Actual(&self.indices),
            self.lowered.action(), self.lowered.local_slice())
    }
    pub(crate) fn population(&self) -> Result<StaticActivationPopulation, Failure> {
        let input = trace::Value { shape: Shape::new(&[self.shape[0] as u64, self.shape[1] as u64])?, dtype: Some(self.dtype) };
        let mut worker = Adapter(trace::Count::default());
        self.run(&mut worker, &input)?;
        Ok(StaticActivationPopulation { retained_roots: worker.0.roots, completions: worker.0.roots,
            host_bytes: worker.0.host_bytes,
            controls: control_bytes(worker.0.roots).and_then(|n| n.checked_add(Self::inspection_control_bytes()?))
                .and_then(|n| n.checked_add(Array::flat_index_update_control_bytes()?)).ok_or(Failure::GeometryOverflow)?,
        })
    }
    /// Trace these actual selected rows. A prospective enclosing envelope must
    /// be established separately; this method never labels a maximum as actual.
    pub(crate) fn trace(&self, input: &WorkspaceTensor, context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>) -> Result<Option<WorkspaceTensor>, Failure> {
        PreparedStaticActivation::validate_workspace_source(input, self.dtype)?;
        context.charge_metadata(trace::control_bytes().ok_or(Failure::GeometryOverflow)?)
            .map_err(eredu_nn::Error::from)?;
        let input = trace::TracedValue { tensor: input.clone(), dtype: Some(self.dtype) };
        let mut worker = Adapter(trace::Trace { context, retained });
        self.run(&mut worker, &input).map(|value| value.map(|value| value.tensor))
    }
    /// Execute under the caller's original role. Success returns the same owner
    /// for receipt completion; failure keeps it alongside all earlier roots in Q.
    pub(crate) fn execute(self, input: &Array, stream: &Stream,
        observer: &OriginalScopeObserver, roots: &RefCell<Vec<Array>>,
    ) -> Result<(Option<Array>, Self), SparseActivationFailure> {
        let result = (|| -> Result<Option<Array>, Failure> {
            let completion = CaptureCompletion::Original(observer);
            completion.validate_identity()?;
            let population = self.population()?;
            {
                let borrowed = roots.try_borrow().map_err(|_| Failure::CollectorBusy)?;
                if borrowed.capacity() - borrowed.len() < population.retained_roots { return Err(Failure::GeometryOverflow); }
            }
            let mut worker = Adapter(Native { stream, completion, roots: Some(roots) });
            self.run(&mut worker, input)
        })();
        match result { Ok(value) => Ok((value, self)), Err(cause) => Err(SparseActivationFailure { cause, _owner: self }) }
    }
}

// The same graph construction order consumes actual native indices or an
// explicitly descriptive extent. Only Native rejects the latter; no dummy
// indices, fake row values or source registration are manufactured for quoting.
fn run_selected<T: Kernel>(worker: &mut Adapter<T>, input: &T::Value, shape: [i32; 2],
    dtype: InterventionDtype, indices: IndexSource<'_>, action: Option<&InterventionAction>,
    slice: &ResolvedCaptureSlice,
) -> Result<Option<T::Value>, Failure> {
    if worker.0.shape(input) != shape || worker.0.dtype(input)? != dtype {
        return Err(Failure::ShapeMismatch);
    }
    let Some(action) = action else { return Ok(None); };
    let count = [u64::try_from(indices.len()).map_err(|_| Failure::GeometryOverflow)?];
    let indices = worker.0.emit(Op::Indices(indices))?;
    let flat = worker.0.emit(Op::Reshape(input, &[shape[0].checked_mul(shape[1]).ok_or(Failure::GeometryOverflow)?]))?;
    let selected = worker.indexed_select(&flat, &indices)?;
    let changed = eredu_runtime::intervention::apply_activation_with_source_shape(
        worker, &selected, action, slice, &count).map_err(policy_failure)?;
    let output = worker.indexed_update(&flat, &indices, &changed)?;
    worker.0.emit(Op::Reshape(&output, &shape)).map(Some)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
