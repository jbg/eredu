//! Cumulative canonical source publications for sampling and expert metadata.
use super::*;
use eredu_runtime::working_memory::HostSourceConstructionFacts;
impl ResidentNativeRecipe {
    pub(crate) fn initialized_input_source_facts(
        &self,
        outputs: u64,
    ) -> Result<Option<HostSourceConstructionFacts>, crate::backend::error::Error> {
        let Some(source) = self.parallel_control_source()? else {
            return Ok(None);
        };
        let overflow = || {
            crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            )
        };
        source
            .funding()
            .reserve_metadata(
                std::mem::size_of::<(&Self, u64, usize)>()
                    .checked_add(std::mem::size_of::<
                        Result<Option<HostSourceConstructionFacts>, crate::backend::error::Error>,
                    >())
                    .ok_or_else(overflow)?,
            )
            .map_err(crate::backend::error::Error::WorkspacePlanning)?;
        let sampling = source.sampling_source_facts(outputs)?;
        let mut bytes = sampling.capacity_bytes();
        let mut attempts = sampling.maximum_attempts();
        for row in &self.records {
            if let Some(facts) = row.initialized_input_source_facts()? {
                bytes = bytes
                    .checked_add(facts.capacity_bytes())
                    .ok_or_else(overflow)?;
                attempts = attempts
                    .checked_add(facts.maximum_attempts())
                    .ok_or_else(overflow)?;
            }
        }
        Ok(Some(
            HostSourceConstructionFacts::new(bytes, attempts, 0)
                .map_err(crate::backend::error::Error::PrefillControl)?,
        ))
    }
}

impl ResidentSpanRecipe {
    /// Integer source publications from this exact retained model itinerary.
    /// Sampling outside the model span retains its own separate source bank.
    pub(crate) fn initialized_input_source_facts(
        &self,
    ) -> Result<Option<HostSourceConstructionFacts>, crate::backend::error::Error> {
        let Some(invocation) = self.parallel() else {
            return Ok(None);
        };
        invocation
            .source()
            .funding()
            .reserve_metadata(std::mem::size_of::<(
                &Self,
                (u64, usize),
                Result<Option<HostSourceConstructionFacts>, crate::backend::error::Error>,
            )>())
            .map_err(crate::backend::error::Error::WorkspacePlanning)?;
        let (bytes, attempts) = invocation.expert_input_source_facts()?;
        if attempts == 0 {
            if bytes != 0 {
                return Err(crate::backend::error::Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ));
            }
            return Ok(None);
        }
        HostSourceConstructionFacts::new(bytes, attempts, 0)
            .map(Some)
            .map_err(crate::backend::error::Error::PrefillControl)
    }
}
