//! Same selected-output/scalar readouts on the actual pre/post-edit graph value.
use super::super::Selection;
use super::*;
use eredu_runtime::working_memory::{
    CaptureInterventionEvidenceClaim, CaptureInterventionEvidenceKind, ClaimedInterventionEvidence,
};

fn selection<'a>(
    source: &'a OriginalInterventionSource,
    operation: usize,
    prediction: u64,
    side: InterventionEvidenceSide,
    context: &WorkspaceContext,
) -> Result<Option<Selection<'a>>, Error> {
    let Some(companion) = source.plan().evidence(operation) else {
        return Ok(None);
    };
    if companion.operation() != operation
        || companion.geometry_source().plan().selections.len() != 2
    {
        return Err(cold(context, CaptureTensorNativeError::ClaimMismatch));
    }
    Selection::prepare(
        companion.geometry_source(),
        side.index(),
        prediction,
        context,
    )
    .map(Some)
}
pub(super) fn population(
    source: &OriginalInterventionSource,
    operation: usize,
    prediction: u64,
    context: &WorkspaceContext,
) -> Result<(usize, usize, usize), Error> {
    let mut result = (0usize, 0usize, 0usize);
    for side in [
        InterventionEvidenceSide::Before,
        InterventionEvidenceSide::After,
    ] {
        let Some(program) = selection(source, operation, prediction, side, context)? else {
            continue;
        };
        let (roots, completions, controls) = program
            .population()
            .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
        result.0 = result
            .0
            .checked_add(roots)
            .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
        result.1 = result
            .1
            .checked_add(completions)
            .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
        result.2 = result
            .2
            .checked_add(controls)
            .and_then(|n| n.checked_add(runtime_controls()?))
            .ok_or_else(|| cold(context, CaptureTensorNativeError::GeometryOverflow))?;
    }
    Ok(result)
}
pub(super) fn trace(
    source: &OriginalInterventionSource,
    operation: usize,
    prediction: u64,
    side: InterventionEvidenceSide,
    input: &WorkspaceTensor,
    context: &WorkspaceContext,
    retained: &mut Vec<WorkspaceTensor>,
) -> Result<(), Error> {
    let Some(program) = selection(source, operation, prediction, side, context)? else {
        return Ok(());
    };
    // Both readout adapters retain the actual input alias. After an edit this is
    // lazy: the selected-output/flatten frontier evaluates its complete ancestry.
    retained.push(input.clone());
    program.trace(input, context, retained)
}
impl Edits {
    fn check_evidence(
        &self,
        claim: &CaptureInterventionEvidenceClaim<'_, '_>,
        custody: &OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<(), CaptureTensorNativeError> {
        claim.validate_source(&self.source)?;
        claim.validate_numerical_custody(custody)?;
        let (index, _, phase, prediction) = claim.coordinate();
        if prediction != self.prediction
            || phase
                != if prediction == 0 {
                    CapturePhase::Prefill
                } else {
                    CapturePhase::Decode
                }
            || self.rows.get(index).and_then(Option::as_ref).is_none()
        {
            return Err(CaptureTensorNativeError::ClaimMismatch);
        }
        Ok(())
    }
    pub(in crate::composition::mlx::speculative::sampling::numerical::capture) fn evidence_usage(
        &self,
        source: &Array,
        claim: &CaptureInterventionEvidenceClaim<'_, '_>,
        custody: &OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<(TensorDtype, CaptureUsage), CaptureTensorNativeError> {
        self.check_evidence(claim, custody)?;
        let (dtype, usage) = match claim.kind() {
            CaptureInterventionEvidenceKind::Preview(claim) => (
                PreparedCaptureTensor::validate_borrowed_source(source, claim.geometry())?,
                crate::composition::mlx::session::bounded_capture::estimate_tensor_geometry(
                    claim.geometry(),
                )
                .map_err(CaptureTensorNativeError::ActivationPolicy)?,
            ),
            CaptureInterventionEvidenceKind::Summary(claim) => (
                PreparedCaptureSummary::from_geometry(claim.geometry())?.validate_source(source)?,
                crate::composition::mlx::session::bounded_capture::estimate_summary(
                    claim.geometry(),
                )
                .map_err(CaptureTensorNativeError::ActivationPolicy)?,
            ),
        };
        let dtype = match dtype {
            safemlx::Dtype::Float32 => TensorDtype::F32,
            safemlx::Dtype::Float16 => TensorDtype::F16,
            safemlx::Dtype::Bfloat16 => TensorDtype::Bf16,
            _ => unreachable!("checked floating evidence"),
        };
        Ok((dtype, usage))
    }
    pub(in crate::composition::mlx::speculative::sampling::numerical::capture) fn capture_evidence<
        'a,
    >(
        &self,
        source: &Array,
        claim: CaptureInterventionEvidenceClaim<'a, '_>,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        custody: &OriginalSpeculativeNumericalBudgetCustody,
        observer: &OriginalScopeObserver,
        cpu: Option<&CpuCaptureLoan<'_>>,
    ) -> Result<ClaimedInterventionEvidence<'a>, FundedCaptureError<CaptureTensorNativeError>> {
        let result: Result<ClaimedInterventionEvidence<'a>, CaptureTensorNativeError> = (|| {
            self.check_evidence(&claim, custody)?;
            let (receipt, kind) = claim.into_parts();
            // No assumption of completed post-edit input. Preview settles its
            // selected output; Summary settles flat and then each scalar. Their
            // dependency traversal includes all lazy action ancestors. We retain
            // every root in the existing final numerical union and Recovery.
            match kind {
                CaptureInterventionEvidenceKind::Preview(claim) => {
                    let value = if let Some(loan) = cpu {
                        crate::backend::array_copy::execute_speculative_capture_cpu(
                            source, claim, roots, custody, loan,
                        )?
                    } else {
                        crate::backend::array_copy::execute_speculative_capture(
                            source, claim, stream, roots, custody, observer,
                        )?
                    };
                    Ok(receipt.finish_preview(value)?)
                }
                CaptureInterventionEvidenceKind::Summary(claim) => {
                    let value = if let Some(loan) = cpu {
                        crate::backend::array_copy::execute_speculative_summary_cpu(
                            source, claim, roots, custody, loan,
                        )?
                    } else {
                        crate::backend::array_copy::execute_speculative_summary(
                            source, claim, stream, roots, custody, observer,
                        )?
                    };
                    Ok(receipt.finish_summary(value)?)
                }
            }
        })(
        );
        result.map_err(FundedCaptureError::Backend)
    }
}
fn runtime_controls() -> Option<usize> {
    use eredu_runtime::working_memory::InterventionEvidenceReceipt;
    let frames = [
        size_of::<Option<&CpuCaptureLoan<'static>>>(),
        CpuCaptureLoan::control_bytes()?,
        size_of::<CaptureInterventionEvidenceClaim<'static, 'static>>(),
        size_of::<CaptureInterventionEvidenceKind<'static, 'static>>(),
        size_of::<InterventionEvidenceReceipt<'static>>(),
        size_of::<ClaimedInterventionEvidence<'static>>(),
        size_of::<Option<ClaimedInterventionEvidence<'static>>>(),
        size_of::<
            Result<
                Option<ClaimedInterventionEvidence<'static>>,
                FundedCaptureError<CaptureTensorNativeError>,
            >,
        >(),
        size_of::<Result<ClaimedInterventionEvidence<'static>, CaptureTensorNativeError>>(),
        size_of::<
            Result<
                ClaimedInterventionEvidence<'static>,
                FundedCaptureError<CaptureTensorNativeError>,
            >,
        >(),
        size_of::<Result<(TensorDtype, CaptureUsage), CaptureTensorNativeError>>(),
        size_of::<Result<(TensorDtype, CaptureUsage), FundedCaptureError<CaptureTensorNativeError>>>(
        ),
        size_of::<(
            &Edits,
            &Array,
            &Stream,
            &RefCell<Vec<Array>>,
            &OriginalSpeculativeNumericalBudgetCustody,
            &OriginalScopeObserver,
        )>(),
        size_of::<(usize, InterventionEvidenceSide, CapturePhase, u64)>(),
        size_of::<Option<TensorDtype>>(),
        size_of::<CaptureUsage>(),
        size_of::<Option<CaptureSkipReason>>(),
        size_of::<Result<(), FundedCaptureError<CaptureTensorNativeError>>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
pub(super) fn planning_controls() -> Option<usize> {
    let frames = [
        size_of::<Selection<'static>>(),
        size_of::<Result<Option<Selection<'static>>, Error>>(),
        size_of::<(usize, usize, usize)>(),
        size_of::<Result<(usize, usize, usize), Error>>(),
        size_of::<[InterventionEvidenceSide; 2]>(),
        size_of::<(&OriginalInterventionSource, usize, u64, &WorkspaceContext)>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
