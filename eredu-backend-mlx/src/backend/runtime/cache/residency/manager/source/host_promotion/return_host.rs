//! A completed reload reuses its immutable Host backing without another copy.
use super::*;
use crate::backend::nn::workspace::OriginalPagedHostReturn;
impl PreparedCacheHostPromotion {
    pub(crate) fn host_capacity(&self) -> Option<u64> {
        self.descriptors.iter().try_fold(0u64, |n, d| {
            n.checked_add(u64::try_from(d.allocation().bytes()).ok()?)
        })
    }

    pub(crate) fn was_demoted(&self) -> bool {
        self.demoted
    }
    pub(crate) fn owns_pin(&self) -> bool {
        true
    }
    pub(crate) fn return_to_host(
        &mut self,
        proof: &OriginalPagedHostReturn<'_, '_>,
    ) -> Result<(), Exception> {
        let source = proof.source();
        if !self.published
            || !self.completed
            || self.demoted
            || self.replaced.is_some()
            || self.disk_replaced.is_some()
            || self.backing.is_some()
            || proof.id() != &self.id
        {
            return Err(source.error(CacheSourceError::Identity));
        }
        source.validate_manager(&self.manager, self.generation)?;
        self.validate_host_source(source)?;
        let arrays = [
            self.outputs[0].as_ref().expect("completed first"),
            self.outputs[1].as_ref().expect("completed second"),
        ];
        let eviction = proof.bind(arrays);
        let capacity = self
            .descriptors
            .iter()
            .try_fold(0u64, |n, d| {
                n.checked_add(u64::try_from(d.allocation().bytes()).ok()?)
            })
            .ok_or_else(|| source.error(CacheSourceError::Overflow))?;
        self.replaced = Some(super::super::host_demotion::commit_host(
            &self.manager,
            &eviction,
            self.host.clone(),
            capacity,
            &mut self.reservation,
            false,
        )?);
        self.demoted = true;
        Ok(())
    }
}
