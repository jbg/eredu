//! Original declarations, paid slice destinations, and the shared static worker.
use super::*;
use crate::backend::nn::workspace::CpuCaptureLoan;
use crate::composition::mlx::session::intervention::{
    NativeInterventionEstimator, PreparedStaticActivation,
};
use eredu_core::intervention::{InterventionDtype, InterventionEstimator};
use eredu_runtime::capture::FundedCaptureError;
use eredu_runtime::working_memory::{
    CaptureInterventionClaim, ClaimedIntervention, OriginalInterventionSource,
};

mod evidence;

struct Edit {
    slice: ResolvedCaptureSlice,
    usage: CaptureUsage,
}
pub(super) struct Edits {
    rows: Vec<Option<Edit>>,
    source: OriginalInterventionSource,
    shape: [u64; 3],
    prediction: u64,
    pub roots: usize,
    pub completions: usize,
    pub controls: usize,
}
impl Edits {
    pub(super) fn prepare(
        source: &OriginalInterventionSource,
        prediction: u64,
        vocabulary: i32,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let cold = |cause| super::cold(context, cause);
        let phase = if prediction == 0 {
            CapturePhase::Prefill
        } else {
            CapturePhase::Decode
        };
        let shape = [
            1,
            1,
            u64::try_from(vocabulary).map_err(|_| cold(CaptureTensorNativeError::ShapeMismatch))?,
        ];
        let plan = source.plan().admission();
        let mut rows = context.metadata_vec(plan.plan().operations.len())?;
        let mut roots = 0usize;
        let mut completions = 0usize;
        let mut controls = 0usize;
        for (index, operation) in plan.plan().operations.iter().enumerate() {
            if !operation.schedule.includes(phase, prediction) {
                rows.push(None);
                continue;
            }
            let vector = || -> Result<Vec<u64>, Error> {
                let mut value = context.metadata_vec(shape.len())?;
                value.resize(shape.len(), 0);
                Ok(value)
            };
            let mut slice = ResolvedCaptureSlice {
                starts: vector()?,
                ends: vector()?,
                strides: vector()?,
                shape: vector()?,
            };
            plan.resolve_prepared_at(
                index,
                phase,
                prediction,
                &shape,
                InterventionDtype::Float32,
                &mut slice,
            )
            .map_err(|cause| super::cold(context, cause))?;
            let program = PreparedStaticActivation::new(
                &operation.action,
                &slice,
                &shape,
                InterventionDtype::Float32,
            )
            .map_err(cold)?;
            let population = program.population().map_err(cold)?;
            let usage = NativeInterventionEstimator
                .activation_usage(&shape, &slice, &operation.action)
                .map_err(|cause| super::cold(context, cause))?;
            roots = roots
                .checked_add(population.retained_roots)
                .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?;
            completions = completions
                .checked_add(population.completions)
                .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?;
            controls = controls
                .checked_add(population.controls)
                .and_then(|n| n.checked_add(population.host_bytes))
                .and_then(|n| n.checked_add(runtime_controls()?))
                .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?;
            let (evidence_roots, evidence_completions, evidence_controls) =
                evidence::population(source, index, prediction, context)?;
            roots = roots
                .checked_add(evidence_roots)
                .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?;
            completions = completions
                .checked_add(evidence_completions)
                .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?;
            controls = controls
                .checked_add(evidence_controls)
                .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?;
            rows.push(Some(Edit { slice, usage }));
        }
        if roots != 0 {
            // Restore the original sampler shape once, then retain that actual
            // reshape root through policy construction and the final root union.
            roots = roots
                .checked_add(1)
                .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?;
            controls = controls
                .checked_add(
                    safemlx::PreparedArrayClone::control_bytes()
                        .and_then(|n| n.checked_add(Array::inspection_clone_handle_bytes()))
                        .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?,
                )
                .ok_or_else(|| cold(CaptureTensorNativeError::GeometryOverflow))?;
        }
        Ok(Self {
            rows,
            source: source.clone(),
            shape,
            prediction,
            roots,
            completions,
            controls,
        })
    }
    pub(super) fn trace(
        &self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
        retained: &mut Vec<WorkspaceTensor>,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        let mut effective = None;
        for (index, row) in self.rows.iter().enumerate() {
            let Some(edit) = row else { continue };
            let action = &self.source.plan().admission().plan().operations[index].action;
            let program = PreparedStaticActivation::new(
                action,
                &edit.slice,
                &self.shape,
                InterventionDtype::Float32,
            )
            .map_err(|cause| cold(context, cause))?;
            evidence::trace(
                &self.source,
                index,
                self.prediction,
                InterventionEvidenceSide::Before,
                effective.as_ref().unwrap_or(input),
                context,
                retained,
            )?;
            effective = Some(
                program
                    .trace(effective.as_ref().unwrap_or(input), context, retained)
                    .map_err(|cause| cold(context, cause))?,
            );
            evidence::trace(
                &self.source,
                index,
                self.prediction,
                InterventionEvidenceSide::After,
                effective.as_ref().expect("actual edit output"),
                context,
                retained,
            )?;
        }
        Ok(effective)
    }
    fn checked<'a>(
        &'a self,
        source: &Array,
        claim: &CaptureInterventionClaim<'_>,
        custody: &OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<(&'a Edit, PreparedStaticActivation<'a>), CaptureTensorNativeError> {
        claim.validate_numerical_custody(custody)?;
        claim.validate_source(&self.source)?;
        if source.dtype() != safemlx::Dtype::Float32
            || source.shape().len() != self.shape.len()
            || source
                .shape()
                .iter()
                .zip(self.shape)
                .any(|(a, b)| u64::try_from(*a).ok() != Some(b))
        {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        let edit = self
            .rows
            .get(claim.index())
            .and_then(Option::as_ref)
            .ok_or(CaptureTensorNativeError::ClaimMismatch)?;
        let action = &self.source.plan().admission().plan().operations[claim.index()].action;
        Ok((
            edit,
            PreparedStaticActivation::new(
                action,
                &edit.slice,
                &self.shape,
                InterventionDtype::Float32,
            )?,
        ))
    }
    pub(super) fn usage(
        &self,
        source: &Array,
        claim: &CaptureInterventionClaim<'_>,
        custody: &OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<CaptureUsage, CaptureTensorNativeError> {
        self.checked(source, claim, custody)
            .map(|(edit, _)| edit.usage)
    }
    pub(super) fn execute(
        &self,
        source: &Array,
        claim: CaptureInterventionClaim<'_>,
        charged: CaptureUsage,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        custody: &OriginalSpeculativeNumericalBudgetCustody,
        observer: &OriginalScopeObserver,
        cpu: Option<&CpuCaptureLoan<'_>>,
    ) -> Result<(Array, ClaimedIntervention), FundedCaptureError<CaptureTensorNativeError>> {
        let (edit, program) = self
            .checked(source, &claim, custody)
            .map_err(FundedCaptureError::Backend)?;
        if edit.usage != charged {
            return Err(FundedCaptureError::Backend(
                CaptureTensorNativeError::ClaimMismatch,
            ));
        }
        let (stream, observer) = if let Some(loan) = cpu {
            loan.validate().map_err(FundedCaptureError::Backend)?;
            (loan.stream(), loan.observer())
        } else {
            PreparedCaptureTensor::validate_stream(stream).map_err(FundedCaptureError::Backend)?;
            (stream, observer)
        };
        let value = program
            .execute_array(source, stream, observer, roots)
            .map_err(FundedCaptureError::Backend)?;
        let receipt = claim.finish(charged)?;
        Ok((value, receipt))
    }
    pub(super) fn validate(
        &self,
        pool: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<(), eredu_runtime::working_memory::WorkingMemoryError> {
        self.source.validate_pool(pool)
    }
}
fn runtime_controls() -> Option<usize> {
    let frames = [
        size_of::<Edit>(),
        size_of::<Option<&CpuCaptureLoan<'static>>>(),
        size_of::<(&Stream, &OriginalScopeObserver)>(),
        CpuCaptureLoan::control_bytes()?,
        size_of::<ResolvedCaptureSlice>(),
        size_of::<[u64; 3]>(),
        size_of::<CaptureInterventionClaim<'static>>(),
        size_of::<ClaimedIntervention>(),
        size_of::<CaptureTensorNativeError>(),
        size_of::<FundedCaptureError<CaptureTensorNativeError>>(),
        size_of::<Result<(Array, ClaimedIntervention), FundedCaptureError<CaptureTensorNativeError>>>(
        ),
        size_of::<Result<CaptureUsage, FundedCaptureError<CaptureTensorNativeError>>>(),
        size_of::<Result<Option<Array>, FundedCaptureError<CaptureTensorNativeError>>>(),
        size_of::<[Option<Array>; 2]>(),
        size_of::<NativeInterventionEstimator>(),
        Stream::device_type_control_bytes()?,
        OriginalScopeObserver::control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
pub(super) fn planning_control_bytes() -> Option<usize> {
    let frames = [
        PreparedStaticActivation::inspection_control_bytes()?,
        size_of::<Edits>(),
        size_of::<Option<Edits>>(),
        size_of::<Vec<Option<Edit>>>(),
        size_of::<ResolvedCaptureSlice>(),
        size_of::<[Vec<u64>; 4]>(),
        size_of::<PreparedStaticActivation<'static>>(),
        size_of::<Result<Edits, Error>>(),
        size_of::<Result<Option<WorkspaceTensor>, Error>>(),
        size_of::<eredu_core::intervention::InterventionGeometryError>(),
        size_of::<Option<WorkspaceTensor>>(),
        evidence::planning_controls()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
