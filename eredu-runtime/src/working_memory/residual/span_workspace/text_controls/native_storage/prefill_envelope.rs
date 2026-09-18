//! Complete construction plus carried roots across the shared successful schedule.
use super::*;

/// Physical populations supplied by the selected native lowering for one exact
/// equation record. These are source facts, not an allocation or release grant.
#[derive(Clone, Copy, Debug)]
pub struct NativeEquationStorage {
    /// Complete current construction, including checkpoint and rollback copies.
    pub construction_bytes: u64,
    /// Actual opening state, including the initial resumed/session source.
    pub opening_state_bytes: Option<u64>,
    /// Actual closing state and every possible aliased backing.
    pub closing_state_bytes: Option<u64>,
    /// Current returned readout's complete backing; outputs may all escape.
    pub output_bytes: Option<u64>,
    /// Complete native producers of this span's retained validation results.
    pub validation_producer_bytes: Option<u64>,
}

/// Allocation-free reduction over one borrowed, actual equation schedule. A
/// mechanism may use this only when successful prefill advancement retires the
/// complete construction/checkpoint/recovery prefix before the next operation.
/// Failure must stop advancement and retain that prefix. Other persistent roots
/// must be included in the supplied populations or keep the candidate unknown.
/// No success is inferred from allocator observations or an advertised ceiling.
pub struct NativePrefillEnvelopeBuilder<'a> {
    plan: &'a InferenceSpanWorkspacePlan,
    next: usize,
    decode_started: bool,
    completed_decodes: bool,
    prefill_sum: u64,
    decode_sum: u64,
    prefill_peak: Option<u64>,
    equation_peak: Option<u64>,
    outputs: Option<u64>,
    validations: Option<u64>,
}

/// Closed reduction tied to the original schedule. Prepared input, sampling and
/// host controls remain outside this equation contribution. Prefill-only mode
/// retains its final envelope through summed decode. Complete-equation mode
/// retains all escaped score/validation backing across successful decode cuts.
/// No cumulative population or budget counter is refunded.
#[derive(Clone)]
pub struct NativePrefillEnvelope {
    plan: InferenceSpanWorkspacePlan,
    equation_bytes: u64,
    original_equation_bytes: u64,
    successful_prefill_candidate: Option<u64>,
    successful_equation_candidate: Option<u64>,
}

fn joined(values: &[Option<u64>]) -> Option<u64> {
    values
        .iter()
        .try_fold(0u64, |sum, next| sum.checked_add((*next)?))
}
fn maximum(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    a.zip(b).map(|(a, b)| a.max(b))
}
impl<'a> NativePrefillEnvelopeBuilder<'a> {
    /// Borrows the already retained shared-driver schedule; creates no owner.
    pub fn new(plan: &'a InferenceSpanWorkspacePlan) -> Self {
        Self {
            plan,
            next: 0,
            decode_started: false,
            completed_decodes: false,
            prefill_sum: 0,
            decode_sum: 0,
            prefill_peak: Some(0),
            equation_peak: Some(0),
            outputs: Some(0),
            validations: Some(0),
        }
    }
    /// Includes decode only when the actual shared driver requires prior model
    /// and sampling completion, successful publication and exact native Record
    /// retirement before another decode. Every surviving score/token alias must
    /// be covered by the cumulative output and separate full sampler populations.
    /// This creates no completion evidence or authority on its own.
    pub fn new_completed_equations(plan: &'a InferenceSpanWorkspacePlan) -> Self {
        Self {
            completed_decodes: true,
            ..Self::new(plan)
        }
    }

    /// Accepts exactly the next plan record. The full generation remains valid
    /// independently when optional carryover information is missing/overflows.
    pub fn push(
        &mut self,
        span: &InferenceWorkspaceSpan,
        row: NativeEquationStorage,
    ) -> Result<(), WorkingMemoryError> {
        let record = self
            .plan
            .records()
            .get(self.next)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if record.span() != span
            || row.construction_bytes
                < record
                    .new_tensor_allocation_bytes()
                    .ok_or(WorkingMemoryError::UnknownBound)?
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let prefill = matches!(span, InferenceWorkspaceSpan::Prefill(_));
        match span {
            InferenceWorkspaceSpan::Sampling(_) => return Err(WorkingMemoryError::IdentityMismatch),
            InferenceWorkspaceSpan::Prefill(_) => {
                if self.decode_started {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                self.prefill_sum = self
                    .prefill_sum
                    .checked_add(row.construction_bytes)
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
            InferenceWorkspaceSpan::Decode { .. } => {
                self.decode_started = true;
                self.decode_sum = self
                    .decode_sum
                    .checked_add(row.construction_bytes)
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        if prefill || self.completed_decodes {
            // Opening state includes all shallow checkpoint aliases; the whole
            // current generation retains deep checkpoint/rollback construction.
            // C0 is the actual seeded source, including a decode-only resume.
            let during = joined(&[
                row.opening_state_bytes,
                self.outputs,
                self.validations,
                Some(row.construction_bytes),
            ]);
            // Keep every score backing because an emitted token may alias it.
            // Full separate sampler generations cover RNG/token-owned births.
            self.outputs = joined(&[self.outputs, row.output_bytes]);
            self.validations = joined(&[self.validations, row.validation_producer_bytes]);
            let after = joined(&[row.closing_state_bytes, self.outputs, self.validations]);
            self.equation_peak = maximum(self.equation_peak, maximum(during, after));
            if prefill {
                self.prefill_peak = self.equation_peak;
            }
        }
        self.next += 1;
        Ok(())
    }
    /// Finishes only the complete schedule. Unknown carryover preserves the
    /// existing all-generation budget. A complete smaller candidate additionally
    /// covers full opening state, without issuing any new source/residency credit.
    pub fn finish(self) -> Result<NativePrefillEnvelope, WorkingMemoryError> {
        if self.next != self.plan.records().len() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let original = self
            .prefill_sum
            .checked_add(self.decode_sum)
            .ok_or(WorkingMemoryError::Overflow)?;
        let (fallback, tail, candidate) = if self.completed_decodes {
            (original, 0, self.equation_peak)
        } else {
            (self.prefill_sum, self.decode_sum, self.prefill_peak)
        };
        let selected = candidate.map_or(fallback, |candidate| candidate.min(fallback));
        Ok(NativePrefillEnvelope {
            plan: self.plan.clone(),
            equation_bytes: selected
                .checked_add(tail)
                .ok_or(WorkingMemoryError::Overflow)?,
            original_equation_bytes: original,
            successful_prefill_candidate: self.prefill_peak,
            successful_equation_candidate: self
                .completed_decodes
                .then_some(self.equation_peak)
                .flatten(),
        })
    }
    /// Exact scalar/builder/closed-plan and error transports; no Vec or new Arc.
    pub fn control_bytes() -> Option<usize> {
        use std::mem::size_of;
        let parts = [
            size_of::<Self>(),
            size_of::<NativeEquationStorage>(),
            size_of::<NativePrefillEnvelope>(),
            size_of::<Option<NativePrefillEnvelope>>(),
            size_of::<Result<NativePrefillEnvelope, WorkingMemoryError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<[Option<u64>; 4]>(),
            size_of::<&InferenceWorkspaceSpan>(),
            size_of::<Option<u64>>() * 3,
            size_of::<Self>(), // completed-equation constructor's base value
            size_of::<(u64, u64, Option<u64>)>(), // selected fallback/tail/candidate
            size_of::<u64>() * 3, // original sum, selected value and final addition
        ];
        parts
            .into_iter()
            .try_fold(size_of::<[usize; 12]>(), usize::checked_add)
    }
}
impl NativePrefillEnvelope {
    /// Selected equation contribution; the candidate conservatively includes
    /// full initial state while preserving all existing source credits.
    pub fn equation_bytes(&self) -> u64 {
        self.equation_bytes
    }
    /// Complete original generation sum, retained for diagnostic comparison.
    pub fn original_equation_bytes(&self) -> u64 {
        self.original_equation_bytes
    }
    /// Successful-prefill candidate including full opening/closing backing.
    pub fn successful_prefill_candidate(&self) -> Option<u64> {
        self.successful_prefill_candidate
    }
    /// Full successful-equation candidate, only when completed decode retirement
    /// was explicitly selected and every carried population remained known.
    pub fn successful_equation_candidate(&self) -> Option<u64> {
        self.successful_equation_candidate
    }
}
impl<M: OriginalNativeStorageMechanism> PreparedNativeStoragePlan<M> {
    /// Consumes the same source-plan reduction into the selected native budget.
    /// The backend must already enforce the successful sealed completion and
    /// exact retirement boundaries selected by the builder. This method neither
    /// performs nor manufactures completion.
    pub fn with_completed_prefill_envelope(
        mut self,
        envelope: NativePrefillEnvelope,
    ) -> Result<Self, WorkingMemoryError> {
        if !self.plan.same_plan(&envelope.plan) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let layout = self.layout.get_mut().ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !layout.exact_storage
            || layout.equation_generations.is_some()
            || layout
                .capacity
                .is_none_or(|cap| envelope.equation_bytes > cap)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        layout.equation_generations = Some(envelope.equation_bytes);
        Ok(self)
    }
}
