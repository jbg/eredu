//! Same native selection and host writer under an exact model or numerical claim.
use super::*;
use eredu_runtime::working_memory::{
    OriginalSpeculativeBudgetCustody, OriginalSpeculativeNumericalBudgetCustody,
    ScheduledCaptureTensor,
};
use safemlx::{EvaluatedArray, OperationEvent, OriginalScopeObserver, PreparedArrayClone};
use std::mem::{size_of, size_of_val};

/// A borrowed exact account, selected by the enclosing admitted invocation.
/// Converting the owner kind supplies no source, scope, or completion authority.
#[derive(Clone, Copy)]
pub(crate) enum CaptureCustody<'a> {
    Model(&'a OriginalSpeculativeBudgetCustody),
    Numerical(&'a OriginalSpeculativeNumericalBudgetCustody),
}
impl<'a> From<&'a OriginalSpeculativeBudgetCustody> for CaptureCustody<'a> {
    fn from(value: &'a OriginalSpeculativeBudgetCustody) -> Self {
        Self::Model(value)
    }
}
impl<'a> From<&'a OriginalSpeculativeNumericalBudgetCustody> for CaptureCustody<'a> {
    fn from(value: &'a OriginalSpeculativeNumericalBudgetCustody) -> Self {
        Self::Numerical(value)
    }
}

impl TransferDestination for ScheduledCaptureTensor<'_, '_> {
    type Error = WorkingMemoryError;
    fn validate(&self) -> Result<(), Self::Error> {
        ScheduledCaptureTensor::validate(self)
    }
    fn push_f32(&mut self, value: f32) -> Result<(), Self::Error> {
        ScheduledCaptureTensor::push_f32(self, value)
    }
    fn push_u64(&mut self, value: u64) -> Result<(), Self::Error> {
        ScheduledCaptureTensor::push_u64(self, value)
    }
}
/// Source belongs to the caller's authenticated model or completed numerical value.
/// Existing Q is retained by the same recovery owner; no C publication or source
/// adoption is performed. This exact phase's Scope settles the selected result.
pub(crate) fn execute<'custody>(
    source: &Array,
    claim: CaptureTensorClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    observer: &OriginalScopeObserver,
) -> Result<ClaimedCaptureTensor, CaptureTensorNativeError> {
    execute_shared(source, claim, stream, roots, custody.into(), observer, None)
}

/// The numerical caller lends its actual CPU recipe and currently entered
/// observer. Shared claim/source validation and the ordinary transform remain
/// identical; this path creates no additional completion allowance.
pub(crate) fn execute_cpu<'custody>(
    source: &Array,
    claim: CaptureTensorClaim<'_, '_>,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    loan: &crate::backend::nn::workspace::CpuCaptureLoan<'_>,
) -> Result<ClaimedCaptureTensor, CaptureTensorNativeError> {
    execute_shared(
        source,
        claim,
        loan.stream(),
        roots,
        custody.into(),
        loan.observer(),
        Some(loan),
    )
}
fn execute_shared(
    source: &Array,
    claim: CaptureTensorClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: CaptureCustody<'_>,
    observer: &OriginalScopeObserver,
    cpu: Option<&crate::backend::nn::workspace::CpuCaptureLoan<'_>>,
) -> Result<ClaimedCaptureTensor, CaptureTensorNativeError> {
    match custody {
        CaptureCustody::Model(owner) => claim.validate_model_custody(owner)?,
        CaptureCustody::Numerical(owner) => claim.validate_numerical_custody(owner)?,
    }
    let dtype = PreparedCaptureTensor::validate_borrowed_source(source, claim.geometry())?;
    match cpu {
        Some(loan) => loan.validate()?,
        None => PreparedCaptureTensor::validate_stream(stream)?,
    };
    let program = Selection::with_conversion(claim.geometry(), ConversionMode::actual(dtype))?;
    let completion = CaptureCompletion::Original(observer);
    completion.validate()?;
    // One retained input and at most Slice/reshape/Preview-Slice/AsType outputs.
    completion.reserve_roots(roots, 5)?;
    let mut destination = claim.prepare()?;
    execute_selected(
        source,
        &program,
        &mut destination,
        stream,
        roots,
        completion,
    )?;
    destination
        .finish()
        .map_err(|failure| failure.into_owned_error().into())
}
pub(crate) fn control_bytes() -> Option<usize> {
    let clone =
        PreparedArrayClone::control_bytes()?.checked_add(Array::inspection_clone_handle_bytes())?;
    let parts = [
        OperationEvent::nested_completion_control_bytes::<1>()?,
        clone.checked_mul(6)?,
        Stream::device_type_control_bytes()?,
        OriginalScopeObserver::control_bytes()?,
        EvaluatedArray::iteration_control_bytes::<f32>()?,
        size_of::<CaptureCustody<'static>>(),
        crate::backend::nn::workspace::CpuCaptureLoan::control_bytes()?,
        size_of::<Option<&crate::backend::nn::workspace::CpuCaptureLoan<'static>>>(),
        size_of::<(
            &Array,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &crate::backend::nn::workspace::CpuCaptureLoan<'static>,
        )>(),
        size_of::<Result<ClaimedCaptureTensor, CaptureTensorNativeError>>(),
        size_of::<(
            &Array,
            &Stream,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &OriginalScopeObserver,
        )>(),
        size_of::<Selection>(),
        size_of::<Native<'static>>(),
        size_of::<CaptureCompletion<'static>>(),
        size_of::<CaptureTensorNativeError>(),
        size_of::<ScheduledCaptureTensor<'static, 'static>>(),
        size_of::<Result<ClaimedCaptureTensor, CaptureTensorNativeError>>(),
        size_of::<eredu_runtime::working_memory::ScheduledCaptureTensorFinishError<'static, 'static>>(
        ),
        size_of::<eredu_runtime::working_memory::ScheduledCaptureTensorFailure>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

use super::super::{CandidateExtraction, TokenScoreProgram};
use eredu_core::capture::{CandidateDomain, CaptureTokenDomain};
use eredu_runtime::working_memory::{
    CaptureCandidateClaim, CaptureTokenScoreClaim, ClaimedCaptureCandidates,
    ClaimedCaptureTokenScores, ScheduledCaptureCandidates, ScheduledCaptureTokenScores,
};

fn retain_readout(
    value: &Array,
    roots: &RefCell<Vec<Array>>,
    completion: CaptureCompletion<'_>,
) -> Result<(), CaptureTensorNativeError> {
    let alias = completion.clone_array(value)?;
    let mut values = roots
        .try_borrow_mut()
        .map_err(|_| CaptureTensorNativeError::CollectorBusy)?;
    if values.len() == values.capacity() {
        return Err(WorkingMemoryError::UnknownBound.into());
    }
    values.push(alias);
    Ok(())
}

/// Authenticated source, same sorted candidate program and typed host
/// destination. Only this phase's internal three read frontiers are submitted.
pub(crate) fn execute_candidates<'custody>(
    source: &Array,
    claim: CaptureCandidateClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    observer: &OriginalScopeObserver,
    domain: Option<CaptureTokenDomain<'_>>,
) -> Result<ClaimedCaptureCandidates, CaptureTensorNativeError> {
    execute_candidates_shared(
        source,
        claim,
        stream,
        roots,
        custody.into(),
        observer,
        domain,
        None,
    )
}

/// Borrow the selected complete CPU recipe; keep the ordinary sort/take worker.
pub(crate) fn execute_candidates_cpu<'custody>(
    source: &Array,
    claim: CaptureCandidateClaim<'_, '_>,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    loan: &crate::backend::nn::workspace::CpuCaptureLoan<'_>,
    domain: Option<CaptureTokenDomain<'_>>,
) -> Result<ClaimedCaptureCandidates, CaptureTensorNativeError> {
    execute_candidates_shared(
        source,
        claim,
        loan.stream(),
        roots,
        custody.into(),
        loan.observer(),
        domain,
        Some(loan),
    )
}
fn execute_candidates_shared(
    source: &Array,
    claim: CaptureCandidateClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: CaptureCustody<'_>,
    observer: &OriginalScopeObserver,
    domain: Option<CaptureTokenDomain<'_>>,
    cpu: Option<&crate::backend::nn::workspace::CpuCaptureLoan<'_>>,
) -> Result<ClaimedCaptureCandidates, CaptureTensorNativeError> {
    match custody {
        CaptureCustody::Model(owner) => claim.validate_model_custody(owner)?,
        CaptureCustody::Numerical(owner) => claim.validate_numerical_custody(owner)?,
    }
    let program = CandidateExtraction::from_geometry(claim.geometry())?;
    program.validate_source(source)?;
    match cpu {
        Some(loan) => loan.validate()?,
        None => PreparedCaptureTensor::validate_stream(stream)?,
    };
    let completion = CaptureCompletion::Original(observer);
    completion.validate()?;
    completion.reserve_roots(roots, 1 + CandidateExtraction::ROOTS)?;
    let summary = domain.map(|domain| domain.summary(claim.geometry().vocabulary() as u32));
    let mut destination = claim.prepare(summary)?;
    let mut retain = |value: &Array| retain_readout(value, roots, completion);
    retain(source)?;
    let (ids, scores) = program.execute_with_completion(source, stream, completion, &mut retain)?;
    program.read_with_completion(&ids, &scores, stream, completion, |id, score| {
        destination.push(
            id,
            score,
            domain.is_none_or(|domain| domain.filter.allows(id)),
        )?;
        Ok(())
    })?;
    Ok(destination.finish()?)
}

/// The same finite/log-partition/rank/alternative worker consumes the supplied
/// IDs and the actual pre-force domain. Its source is kept by the same Recovery.
pub(crate) fn execute_token_scores<'custody>(
    source: &Array,
    claim: CaptureTokenScoreClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    observer: &OriginalScopeObserver,
    domain: Option<CaptureTokenDomain<'_>>,
) -> Result<ClaimedCaptureTokenScores, CaptureTensorNativeError> {
    execute_token_scores_shared(
        source,
        claim,
        stream,
        roots,
        custody.into(),
        observer,
        domain,
        None,
    )
}

/// Same selected-ID program and claim, borrowing the already entered CPU role.
pub(crate) fn execute_token_scores_cpu<'custody>(
    source: &Array,
    claim: CaptureTokenScoreClaim<'_, '_>,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    loan: &crate::backend::nn::workspace::CpuCaptureLoan<'_>,
    domain: Option<CaptureTokenDomain<'_>>,
) -> Result<ClaimedCaptureTokenScores, CaptureTensorNativeError> {
    execute_token_scores_shared(
        source,
        claim,
        loan.stream(),
        roots,
        custody.into(),
        loan.observer(),
        domain,
        Some(loan),
    )
}
fn execute_token_scores_shared(
    source: &Array,
    claim: CaptureTokenScoreClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: CaptureCustody<'_>,
    observer: &OriginalScopeObserver,
    domain: Option<CaptureTokenDomain<'_>>,
    cpu: Option<&crate::backend::nn::workspace::CpuCaptureLoan<'_>>,
) -> Result<ClaimedCaptureTokenScores, CaptureTensorNativeError> {
    match custody {
        CaptureCustody::Model(owner) => claim.validate_model_custody(owner)?,
        CaptureCustody::Numerical(owner) => claim.validate_numerical_custody(owner)?,
    }
    let program = TokenScoreProgram::from_geometry(claim.geometry())?;
    program.validate_source(source)?;
    match cpu {
        Some(loan) => loan.validate()?,
        None => PreparedCaptureTensor::validate_stream(stream)?,
    };
    let completion = CaptureCompletion::Original(observer);
    completion.validate()?;
    let count = program
        .population()?
        .retained_outputs
        .checked_add(1)
        .ok_or(CaptureTensorNativeError::GeometryOverflow)?;
    completion.reserve_roots(roots, count)?;
    let summary = domain.map(|domain| domain.summary(program.vocabulary() as u32));
    // Program borrows IDs in the source declaration, not in the moved claim.
    let mut destination = claim.prepare(summary)?;
    let mut retain = |value: &Array| retain_readout(value, roots, completion);
    retain(source)?;
    let partition = program.execute(
        source,
        stream,
        completion,
        &mut retain,
        &mut |_| {},
        |id| domain.is_none_or(|domain| domain.filter.allows(id)),
        |score| {
            destination.push(score)?;
            Ok(())
        },
    )?;
    Ok(destination.finish(partition)?)
}

fn readout_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<(
            &Array,
            &Stream,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &OriginalScopeObserver,
        )>(),
        size_of::<CaptureCustody<'static>>(),
        size_of::<Option<CaptureTokenDomain<'static>>>(),
        size_of::<Option<CandidateDomain>>(),
        size_of::<CaptureCompletion<'static>>(),
        size_of::<std::cell::RefMut<'static, Vec<Array>>>(),
        size_of::<Result<(), CaptureTensorNativeError>>(),
        size_of::<(&RefCell<Vec<Array>>, CaptureCompletion<'static>)>(),
        size_of::<&mut dyn FnMut(&Array) -> Result<(), CaptureTensorNativeError>>(),
        Stream::device_type_control_bytes()?,
        OriginalScopeObserver::control_bytes()?,
        // Separate source alias; worker queries cover each actual intermediate.
        PreparedArrayClone::control_bytes()?.checked_add(Array::inspection_clone_handle_bytes())?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
pub(crate) fn candidate_control_bytes() -> Option<usize> {
    let frames = [
        crate::backend::nn::workspace::CpuCaptureLoan::control_bytes()?,
        size_of::<Option<&crate::backend::nn::workspace::CpuCaptureLoan<'static>>>(),
        size_of::<(
            &Array,
            CaptureCandidateClaim<'static, 'static>,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &crate::backend::nn::workspace::CpuCaptureLoan<'static>,
            Option<CaptureTokenDomain<'static>>,
        )>(),
        size_of::<(
            &Array,
            CaptureCandidateClaim<'static, 'static>,
            &Stream,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &OriginalScopeObserver,
            Option<CaptureTokenDomain<'static>>,
            Option<&crate::backend::nn::workspace::CpuCaptureLoan<'static>>,
        )>(),
        size_of::<CaptureCandidateClaim<'static, 'static>>(),
        size_of::<ScheduledCaptureCandidates<'static, 'static>>(),
        size_of::<Result<ScheduledCaptureCandidates<'static, 'static>, CaptureRunHostError>>(),
        size_of::<
            Result<
                ClaimedCaptureCandidates,
                eredu_runtime::working_memory::CaptureCandidateFailure,
            >,
        >(),
        size_of::<Result<ClaimedCaptureCandidates, CaptureTensorNativeError>>(),
        size_of::<(
            &mut ScheduledCaptureCandidates<'static, 'static>,
            Option<CaptureTokenDomain<'static>>,
        )>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_add(CandidateExtraction::control_bytes()?)?
        .checked_add(readout_control_bytes()?)
}
pub(crate) fn token_score_control_bytes(program: TokenScoreProgram<'_>) -> Option<usize> {
    let frames = [
        crate::backend::nn::workspace::CpuCaptureLoan::control_bytes()?,
        size_of::<Option<&crate::backend::nn::workspace::CpuCaptureLoan<'static>>>(),
        size_of::<(
            &Array,
            CaptureTokenScoreClaim<'static, 'static>,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &crate::backend::nn::workspace::CpuCaptureLoan<'static>,
            Option<CaptureTokenDomain<'static>>,
        )>(),
        size_of::<(
            &Array,
            CaptureTokenScoreClaim<'static, 'static>,
            &Stream,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &OriginalScopeObserver,
            Option<CaptureTokenDomain<'static>>,
            Option<&crate::backend::nn::workspace::CpuCaptureLoan<'static>>,
        )>(),
        size_of::<CaptureTokenScoreClaim<'static, 'static>>(),
        size_of::<ScheduledCaptureTokenScores<'static, 'static>>(),
        size_of::<Result<ScheduledCaptureTokenScores<'static, 'static>, CaptureRunHostError>>(),
        size_of::<
            Result<
                ClaimedCaptureTokenScores,
                eredu_runtime::working_memory::CaptureTokenScoreFailure,
            >,
        >(),
        size_of::<Result<ClaimedCaptureTokenScores, CaptureTensorNativeError>>(),
        size_of::<&mut ScheduledCaptureTokenScores<'static, 'static>>(),
        size_of::<Option<CaptureTokenDomain<'static>>>(),
        size_of::<Result<f64, CaptureTensorNativeError>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_add(program.control_bytes()?)?
        .checked_add(readout_control_bytes()?)
}

/// The shared selected/flattened Summary worker under the same original phase.
/// The completed numerical input already retains source custody; only the
/// selected flatten and scalar reads require new completion frontiers.
pub(crate) fn execute_summary<'custody>(
    source: &Array,
    claim: eredu_runtime::working_memory::CaptureSummaryClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    observer: &OriginalScopeObserver,
) -> Result<eredu_runtime::working_memory::ClaimedCaptureSummary, CaptureTensorNativeError> {
    execute_summary_shared(source, claim, stream, roots, custody.into(), observer, None)
}

pub(crate) fn execute_summary_cpu<'custody>(
    source: &Array,
    claim: eredu_runtime::working_memory::CaptureSummaryClaim<'_, '_>,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    loan: &crate::backend::nn::workspace::CpuCaptureLoan<'_>,
) -> Result<eredu_runtime::working_memory::ClaimedCaptureSummary, CaptureTensorNativeError> {
    execute_summary_shared(
        source,
        claim,
        loan.stream(),
        roots,
        custody.into(),
        loan.observer(),
        Some(loan),
    )
}
fn execute_summary_shared(
    source: &Array,
    claim: eredu_runtime::working_memory::CaptureSummaryClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: CaptureCustody<'_>,
    observer: &OriginalScopeObserver,
    cpu: Option<&crate::backend::nn::workspace::CpuCaptureLoan<'_>>,
) -> Result<eredu_runtime::working_memory::ClaimedCaptureSummary, CaptureTensorNativeError> {
    match custody {
        CaptureCustody::Model(owner) => claim.validate_model_custody(owner)?,
        CaptureCustody::Numerical(owner) => claim.validate_numerical_custody(owner)?,
    }
    let program = PreparedCaptureSummary::from_geometry(claim.geometry())?;
    program.validate_source(source)?;
    match cpu {
        Some(loan) => loan.validate()?,
        None => PreparedCaptureTensor::validate_stream(stream)?,
    };
    let population = program
        .population()
        .ok_or(CaptureTensorNativeError::GeometryOverflow)?;
    let completion = CaptureCompletion::Original(observer);
    completion.validate()?;
    completion.reserve_roots(roots, population.retained_roots)?;
    let mut retain = |value: &Array| retain_readout(value, roots, completion);
    retain(source)?;
    let summary = match cpu {
        Some(loan) => program.execute_cpu(source, loan, roots, &mut retain)?,
        None => program.execute(source, stream, completion, roots, &mut retain)?,
    };
    Ok(match custody {
        CaptureCustody::Model(owner) => claim.finish_model(owner, summary)?,
        CaptureCustody::Numerical(owner) => claim.finish_numerical(owner, summary)?,
    })
}

pub(crate) fn summary_control_bytes(program: &PreparedCaptureSummary) -> Option<usize> {
    use eredu_runtime::working_memory::{
        CaptureSummaryClaim, CaptureSummaryFailure, ClaimedCaptureSummary,
    };
    let frames = [
        size_of::<CaptureSummaryClaim<'static, 'static>>() * 3,
        size_of::<Option<&crate::backend::nn::workspace::CpuCaptureLoan<'static>>>(),
        crate::backend::nn::workspace::CpuCaptureLoan::control_bytes()?,
        size_of::<(
            &Array,
            &Stream,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &OriginalScopeObserver,
        )>() * 2,
        size_of::<Result<ClaimedCaptureSummary, CaptureTensorNativeError>>(),
        size_of::<eredu_core::capture::CaptureSummary>(),
        size_of::<Result<eredu_core::capture::CaptureSummary, CaptureTensorNativeError>>(),
        size_of::<Result<ClaimedCaptureSummary, CaptureSummaryFailure>>(),
        size_of::<Result<ClaimedCaptureSummary, CaptureTensorNativeError>>(),
        size_of::<super::super::CaptureNativePopulation>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_add(program.control_bytes()?)?
        .checked_add(readout_control_bytes()?)
}

/// The same fixed-edge selected/flattened Histogram worker under its exact
/// original claim. Source custody already belongs to this phase's Recovery;
/// only flatten and scalar reads create completion frontiers here.
pub(crate) fn execute_histogram<'custody>(
    source: &Array,
    claim: eredu_runtime::working_memory::CaptureHistogramClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    observer: &OriginalScopeObserver,
) -> Result<eredu_runtime::working_memory::ClaimedCaptureHistogram, CaptureTensorNativeError> {
    execute_histogram_shared(source, claim, stream, roots, custody.into(), observer, None)
}

pub(crate) fn execute_histogram_cpu<'custody>(
    source: &Array,
    claim: eredu_runtime::working_memory::CaptureHistogramClaim<'_, '_>,
    roots: &RefCell<Vec<Array>>,
    custody: impl Into<CaptureCustody<'custody>>,
    loan: &crate::backend::nn::workspace::CpuCaptureLoan<'_>,
) -> Result<eredu_runtime::working_memory::ClaimedCaptureHistogram, CaptureTensorNativeError> {
    execute_histogram_shared(
        source,
        claim,
        loan.stream(),
        roots,
        custody.into(),
        loan.observer(),
        Some(loan),
    )
}
fn execute_histogram_shared(
    source: &Array,
    claim: eredu_runtime::working_memory::CaptureHistogramClaim<'_, '_>,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
    custody: CaptureCustody<'_>,
    observer: &OriginalScopeObserver,
    cpu: Option<&crate::backend::nn::workspace::CpuCaptureLoan<'_>>,
) -> Result<eredu_runtime::working_memory::ClaimedCaptureHistogram, CaptureTensorNativeError> {
    match custody {
        CaptureCustody::Model(owner) => claim.validate_model_custody(owner)?,
        CaptureCustody::Numerical(owner) => claim.validate_numerical_custody(owner)?,
    }
    // Edges borrow the immutable admission, not this moved claim or a new copy.
    let program = PreparedCaptureHistogram::from_geometry(claim.geometry())?;
    program.validate_source(source)?;
    match cpu {
        Some(loan) => loan.validate()?,
        None => PreparedCaptureTensor::validate_stream(stream)?,
    };
    let population = program
        .population()
        .ok_or(CaptureTensorNativeError::GeometryOverflow)?;
    let completion = CaptureCompletion::Original(observer);
    completion.validate()?;
    completion.reserve_roots(roots, population.retained_roots)?;
    let mut destination = claim.prepare()?;
    let mut retain = |value: &Array| retain_readout(value, roots, completion);
    retain(source)?;
    let mut record = |index, count| {
        destination.add_bin(index, count)?;
        Ok(())
    };
    let totals = match cpu {
        Some(loan) => program.execute_cpu(source, loan, roots, &mut retain, &mut record)?,
        None => program.execute(source, stream, completion, roots, &mut retain, &mut record)?,
    };
    Ok(match custody {
        CaptureCustody::Model(owner) => {
            destination.finish_model(owner, totals.below, totals.above, totals.non_finite)?
        }
        CaptureCustody::Numerical(owner) => {
            destination.finish_numerical(owner, totals.below, totals.above, totals.non_finite)?
        }
    })
}

pub(crate) fn histogram_control_bytes(program: &PreparedCaptureHistogram<'_>) -> Option<usize> {
    use super::super::HistogramTotals;
    use eredu_runtime::working_memory::{
        CaptureHistogramClaim, CaptureHistogramFailure, ClaimedCaptureHistogram,
        ScheduledCaptureHistogram,
    };
    let frames = [
        size_of::<CaptureHistogramClaim<'static, 'static>>() * 3,
        size_of::<Option<&crate::backend::nn::workspace::CpuCaptureLoan<'static>>>(),
        crate::backend::nn::workspace::CpuCaptureLoan::control_bytes()?,
        size_of::<(
            &Array,
            &Stream,
            &RefCell<Vec<Array>>,
            CaptureCustody<'static>,
            &OriginalScopeObserver,
        )>() * 2,
        size_of::<Result<ClaimedCaptureHistogram, CaptureTensorNativeError>>(),
        size_of::<Result<PreparedCaptureHistogram<'static>, CaptureTensorNativeError>>(),
        // The selected exact-owner finish owns the populated builder while forwarding to the
        // existing exact partition validator; the native caller retains custody.
        size_of::<(
            ScheduledCaptureHistogram<'static, 'static>,
            CaptureCustody<'static>,
            u64,
            u64,
            u64,
        )>(),
        size_of::<ScheduledCaptureHistogram<'static, 'static>>(),
        size_of::<Result<ScheduledCaptureHistogram<'static, 'static>, CaptureRunHostError>>(),
        size_of::<HistogramTotals>(),
        size_of::<Result<HistogramTotals, CaptureTensorNativeError>>(),
        size_of::<Result<ClaimedCaptureHistogram, CaptureHistogramFailure>>(),
        size_of::<Result<ClaimedCaptureHistogram, CaptureTensorNativeError>>(),
        size_of::<&mut ScheduledCaptureHistogram<'static, 'static>>(),
        size_of::<(usize, u64)>(),
        size_of::<super::super::CaptureNativePopulation>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)?
        .checked_add(program.control_bytes()?)?
        .checked_add(readout_control_bytes()?)
}
