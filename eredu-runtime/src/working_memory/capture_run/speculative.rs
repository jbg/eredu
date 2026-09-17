//! One capture claim row belonging to an existing numerical occurrence.
use super::*;
use crate::speculative::numerical::{SpeculativeNumericalKind, SpeculativeNumericalProgram};
use crate::working_memory::OriginalSpeculativeNumericalBudgetCustody;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::working_memory) struct SpeculativeCaptureBinding {
    source: eredu_core::SharedStorageIdentity,
    prediction: u64,
    program: SpeculativeNumericalProgram,
    bytes: u64,
    intervention: Option<InterventionBinding>,
}

#[derive(Debug, Clone)]
struct InterventionBinding(super::super::OriginalInterventionSource);
impl PartialEq for InterventionBinding {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_source(&other.0)
    }
}
impl Eq for InterventionBinding {}

impl SpeculativeCaptureBinding {
    pub(in crate::working_memory) fn validate_source_pool(
        &self,
        pool: &super::super::WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        if let Some(binding) = &self.intervention {
            binding.0.validate_pool(pool)?;
        }
        Ok(())
    }
    pub(in crate::working_memory) fn source(&self) -> &eredu_core::SharedStorageIdentity {
        &self.source
    }
}

/// Fixed host destinations for one sampler-input observation. Repeated
/// target/draft/replay coordinates each require a separately consumed numerical
/// phase. This contains no second occurrence counter, native grant or ledger.
#[derive(Debug)]
pub struct SpeculativeCaptureHostPlan<'a> {
    plan: CaptureRunHostPlan<'a>,
    program: SpeculativeNumericalProgram,
    peak: u64,
    intervention: Option<interventions::StepPlan<'a>>,
}
impl<'a> SpeculativeCaptureHostPlan<'a> {
    /// Bind the actual immutable one-row capture source and known processing
    /// call. Native/source C admission and cumulative logical quota remain the
    /// enclosing producer's obligations, before any source is read.
    pub fn prepare(
        source: &'a SharedCapturePlan,
        prediction: u64,
        program: SpeculativeNumericalProgram,
    ) -> Result<Self, CaptureRunHostError> {
        let SpeculativeNumericalKind::ProcessLogits(policy) = program.kind() else {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        };
        let admission = source.admission();
        if admission.is_empty()
            || admission.request().batch != 1
            || admission.request().prompt_tokens != 1
            || policy.history_len() as u64 != prediction
            || program.shape()[..program.shape().len() - 1]
                .iter()
                .any(|&dimension| dimension != 1)
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let phase = super::plan::phase(
            usize::try_from(prediction).map_err(|_| WorkingMemoryError::Overflow)?,
        );
        let geometry = admission
            .geometry_at(phase, prediction, None)
            .map_err(CaptureStepError::from)?;
        let shape = [
            1,
            1,
            *program.shape().last().expect("validated rank") as u64,
        ];
        for (selection, point) in admission.plan().selections.iter().zip(admission.points()) {
            if selection.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH
                || !matches!(
                    selection.transform,
                    CaptureTransform::FullTensor
                        | CaptureTransform::Slice
                        | CaptureTransform::Preview { .. }
                        | CaptureTransform::TokenScores { .. }
                        | CaptureTransform::TopCandidates { .. }
                        | CaptureTransform::Summary
                        | CaptureTransform::Histogram { .. }
                )
            {
                return Err(WorkingMemoryError::UnknownBound.into());
            }
            geometry
                .validate_actual(point, &shape)
                .map_err(CaptureStepError::from)?;
        }
        let end = prediction
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let plan = CaptureRunHostPlan::prepare_range(source, prediction, end, true)?;
        let controls = [
            size_of::<Self>(),
            size_of::<SpeculativeCaptureBinding>(),
            size_of::<SpeculativeCapturePreparationError>(),
            size_of::<(
                OriginalSpeculativeNumericalBudgetCustody,
                crate::capture::FundedSpeculativeCaptureInvocation,
            )>(),
            size_of::<
                Result<
                    (
                        OriginalSpeculativeNumericalBudgetCustody,
                        crate::capture::FundedSpeculativeCaptureInvocation,
                    ),
                    SpeculativeCapturePreparationError,
                >,
            >(),
            size_of::<Option<SpeculativeCaptureBinding>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
        let controls = controls
            .checked_add(crate::capture::funded::speculative::control_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)?;
        let peak = plan
            .initialization_peak_bytes()
            .checked_add(controls)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            plan,
            program,
            peak,
            intervention: None,
        })
    }
    /// Add the actual copied edit source and fixed attributed outcomes to this
    /// same numerical occurrence. Native edits/evidence remain a separately
    /// qualified producer; this supplies no native authority or completion.
    pub fn with_interventions(
        mut self,
        source: &'a super::super::OriginalInterventionSource,
    ) -> Result<Self, CaptureRunHostError> {
        if self.intervention.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        let phase = super::plan::phase(self.plan.first_prediction);
        let plan =
            interventions::StepPlan::prepare(self.source(), source, phase, self.prediction())?;
        self.peak = self
            .peak
            .checked_add(plan.peak())
            .ok_or(WorkingMemoryError::Overflow)?;
        self.intervention = Some(plan);
        Ok(self)
    }
    /// Exact original declaration source; no ownership or admission is created.
    pub fn source(&self) -> &'a SharedCapturePlan {
        self.plan.source()
    }
    /// Logical prediction coordinate, independent of physical phase ordinal.
    pub fn prediction(&self) -> u64 {
        self.plan.first_prediction as u64
    }
    /// Existing frame, tensor, fixed claims and concrete handoff controls.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
    pub(in crate::working_memory) fn validate_program(
        &self,
        program: SpeculativeNumericalProgram,
    ) -> Result<(), WorkingMemoryError> {
        if self.program == program {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    pub(in crate::working_memory) fn binding(&self) -> SpeculativeCaptureBinding {
        SpeculativeCaptureBinding {
            source: self.source().storage_identity().clone(),
            prediction: self.prediction(),
            program: self.program,
            bytes: self.peak,
            intervention: self
                .intervention
                .as_ref()
                .map(|plan| InterventionBinding(plan.source.clone())),
        }
    }
    pub(in crate::working_memory) fn matches(&self, binding: &SpeculativeCaptureBinding) -> bool {
        binding.source == *self.source().storage_identity()
            && binding.prediction == self.prediction()
            && binding.program == self.program
            && binding.bytes == self.peak
            && match (&binding.intervention, &self.intervention) {
                (None, None) => true,
                (Some(binding), Some(plan)) => binding.0.same_source(plan.source),
                _ => false,
            }
    }
    pub(in crate::working_memory) fn construct(
        self,
        custody: OriginalSpeculativeNumericalBudgetCustody,
    ) -> Result<PreparedCaptureRun, CaptureRunHostError> {
        let source = self.intervention.as_ref().map(|plan| plan.source);
        let mut bank = super::construct(
            self.plan,
            source,
            CaptureTensorCustody::Speculative(custody),
        )?;
        bank.protected = self.peak;
        Ok(bank)
    }
}

/// A consumed numerical occurrence keeps its accepted account through every
/// failed source/geometry check or partial host construction. There is no retry
/// constructor or transfer of this custody into a text request.
#[derive(Debug)]
pub struct SpeculativeCapturePreparationError {
    cause: CaptureRunHostError,
    _custody: OriginalSpeculativeNumericalBudgetCustody,
}
impl SpeculativeCapturePreparationError {
    pub(in crate::working_memory) fn new(
        cause: CaptureRunHostError,
        custody: OriginalSpeculativeNumericalBudgetCustody,
    ) -> Self {
        Self {
            cause,
            _custody: custody,
        }
    }
    /// Exact typed rejection; the accepted account remains owned by this error.
    pub fn cause(&self) -> &CaptureRunHostError {
        &self.cause
    }
}
impl fmt::Display for SpeculativeCapturePreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for SpeculativeCapturePreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
