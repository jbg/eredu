//! Per-Work projections of the retained ordinary request census.
use super::super::*;
use eredu_runtime::working_memory::{InferenceTextStep, SamplingWorkspacePhase};

impl ResidentNativeRecipe {
    pub(crate) fn ordinary_transfer_programs(
        &self,
    ) -> impl Iterator<Item = &OrdinaryAddressableProgram> {
        self.records
            .iter()
            .filter_map(|row| row.ordinary_addressable.as_deref())
    }

    /// One ordinary model Work also owns its unchanged sampling step. Prefill
    /// chunks share that Work; decode rows use their actual attempt coordinate.
    pub(crate) fn ordinary_step_call_metadata(
        &self,
        step: &InferenceTextStep,
        prefill: bool,
    ) -> Option<u64> {
        if step.request().geometry() != self.plan.geometry() || prefill != (step.attempt() == 0) {
            return None;
        }
        let mut model = 0u64;
        let mut found = false;
        for row in &self.records {
            let selected = match row.span {
                InferenceWorkspaceSpan::Prefill(_) => prefill,
                InferenceWorkspaceSpan::Decode { index, .. } => {
                    !prefill && index.checked_add(1)? == step.attempt()
                }
                InferenceWorkspaceSpan::Sampling(_) => return None,
            };
            if selected {
                found = true;
                model = model.checked_add(self.ordinary_row_call_controls(row)?.metadata_bytes)?;
            }
        }
        if !found {
            return None;
        }
        let sampling = self
            .sampling
            .rows()
            .iter()
            .find(|row| {
                row.phase
                    == SamplingWorkspacePhase::Step {
                        index: step.attempt(),
                    }
            })?
            .ordinary_calls?
            .metadata_bytes;
        model.checked_add(sampling)
    }

    pub(crate) fn ordinary_preparation_call_metadata(&self) -> Option<u64> {
        self.sampling
            .rows()
            .iter()
            .find(|row| row.phase == SamplingWorkspacePhase::Preparation)?
            .ordinary_calls
            .map(|calls| calls.metadata_bytes)
    }
}
