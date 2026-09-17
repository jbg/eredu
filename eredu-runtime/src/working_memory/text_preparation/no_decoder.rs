//! Original prompt continuation without a decoder table or host initializer.

use super::*;

impl InferencePreparationStage {
    /// Consumes the original prompt claim when its selected native program
    /// constructs no decoder table. The exact reservation/run and mandatory
    /// complete registered source inventory are checked atomically before one
    /// native scope opens. No table identity, host hold, account, byte allowance
    /// or second prompt claim is created.
    ///
    /// The backend must bind actual absence, retain the actual source payload,
    /// and perform only its independently quoted native/pending-input program.
    /// This inventory is custody and source-health evidence, not proof of an
    /// arbitrary program's completeness. The returned finish-only remainder
    /// must stay unfinished until all original prompt preparation succeeds.
    pub fn construct_without_decoder<K: Clone + Ord + Send + Sync + 'static>(
        self,
        funding: &WorkingMemoryFundingRun,
        complete_source: WorkingMemoryStorage<K>,
    ) -> Result<(InferencePromptCompletion, WorkingMemoryFundingScope), WorkingMemoryError> {
        let reservation = self.validate_dense_prompt_construction()?;
        let native = funding.open_no_decoder_prompt_scope(&reservation, complete_source)?;
        Ok((InferencePromptCompletion::new(self), native))
    }
}

#[cfg(test)]
mod tests;
