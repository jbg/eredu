//! Cumulative canonical source publications for sampling and expert metadata.
use super::*;
use eredu_runtime::working_memory::HostSourceConstructionFacts;
impl ResidentNativeRecipe {
    pub(crate) fn initialized_input_source_facts(&self,outputs:u64)
        ->Result<Option<HostSourceConstructionFacts>,crate::backend::error::Error> {
        let Some(source)=self.parallel_control_source()? else{return Ok(None)};
        let overflow=||crate::backend::error::Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow);
        source.funding().reserve_metadata(std::mem::size_of::<(&Self,u64,usize)>()
            .checked_add(std::mem::size_of::<Result<Option<HostSourceConstructionFacts>,crate::backend::error::Error>>())
            .ok_or_else(overflow)?).map_err(crate::backend::error::Error::WorkspacePlanning)?;
        let sampling=source.sampling_source_facts(outputs)?;
        let mut bytes=sampling.capacity_bytes();let mut attempts=sampling.maximum_attempts();
        for invocation in self.records.iter().filter_map(|row|row.parallel()) {
            let (additional,count)=invocation.expert_input_source_facts()?;
            bytes=bytes.checked_add(additional).ok_or_else(overflow)?;
            attempts=attempts.checked_add(count).ok_or_else(overflow)?;
        }
        Ok(Some(HostSourceConstructionFacts::new(bytes,attempts,0)
            .map_err(crate::backend::error::Error::PrefillControl)?))
    }
}
