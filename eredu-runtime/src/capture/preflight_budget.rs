//! Shared per-step/cumulative arithmetic for ordinary and prepared preflight.
use super::*;
pub(crate) struct PreflightBudget<'a> {
    plan: &'a AdmittedCapturePlan,
    next: u64,
    remaining: u64,
    base: CaptureUsage,
    step: CaptureUsage,
    total: CaptureUsage,
}
impl<'a> PreflightBudget<'a> {
    pub(crate) fn new(
        plan: &'a AdmittedCapturePlan,
        base: CaptureUsage,
        next: u64,
        inherited: CaptureUsage,
    ) -> Result<Self, CaptureError> {
        let remaining = plan
            .request()
            .max_predictions
            .checked_sub(next)
            .ok_or_else(|| {
                CaptureError::Invalid("continuation exceeds admitted prediction range".into())
            })?;
        if let Some(budget) = base.exceeded(plan.plan().limits.per_step) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: false,
            });
        }
        Ok(Self {
            plan,
            next,
            remaining,
            base,
            step: base,
            total: inherited.checked_add(base.checked_mul(remaining)?)?,
        })
    }
    pub(crate) fn begin_phase(&mut self, phase: CapturePhase) -> bool {
        self.step = self.base;
        self.remaining != 0
            && !(phase == CapturePhase::Prefill && self.next > 0)
            && !(phase == CapturePhase::Decode && self.plan.request().max_predictions <= 1)
    }
    pub(crate) fn add(&mut self, cost: CaptureUsage, count: u64) -> Result<(), CaptureError> {
        self.step = self.step.checked_add(cost)?;
        self.total = self.total.checked_add(cost.checked_mul(count)?)?;
        Ok(())
    }
    pub(crate) fn add_selection(
        &mut self,
        cost: CaptureUsage,
        count: u64,
    ) -> Result<(), CaptureError> {
        if self.plan.plan().limits.on_limit == CaptureLimitPolicy::Fail {
            self.add(cost, count)?;
        }
        Ok(())
    }
    pub(crate) fn finish_phase(&self) -> Result<(), CaptureError> {
        if let Some(budget) = self.step.exceeded(self.plan.plan().limits.per_step) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: false,
            });
        }
        Ok(())
    }
    pub(crate) fn finish(self) -> Result<(), CaptureError> {
        if let Some(budget) = self.total.exceeded(self.plan.plan().limits.cumulative) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: true,
            });
        }
        Ok(())
    }
}
