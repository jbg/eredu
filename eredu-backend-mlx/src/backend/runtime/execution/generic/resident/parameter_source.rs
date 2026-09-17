//! Borrow actual idle units and their exact loaded transfer; no manager entry.
use super::*;
pub(crate) use crate::backend::nn::shared::ParameterSourceCounts as ResidentParameterCounts;
use crate::backend::nn::shared::{ParameterCountError, ParameterSourceCounter as Counter};
use eredu_nn::ParameterSourceError;
use safemlx::{ArrayDescriptorError, RuntimeCallGuard};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ResidentSourceError {
    #[error("resident source selection differs at unit {unit}")]
    Selection { unit: usize },
    #[error("resident source unit {unit} is currently acquired")]
    NotIdle { unit: usize },
    #[error("resident parameter source failed at unit {unit}")]
    Traversal {
        unit: usize,
        #[source]
        source: ParameterSourceError,
    },
    #[error("resident descriptor failed at unit {unit}, slot {slot}")]
    Descriptor {
        unit: usize,
        slot: usize,
        #[source]
        source: ArrayDescriptorError,
    },
    #[error("resident source population count overflow")]
    CountOverflow,
}

/// This borrow retains the policy's layout, source and actual residency transfer.
/// It grants no native readiness, deduplicated storage or request authority.
pub(crate) struct ResidentParameterSource<'s, U: 'static> {
    policy: &'s MlxResidentPolicy<U>,
}

pub(crate) struct CountedResidentParameterSource<'s, U: 'static> {
    source: ResidentParameterSource<'s, U>,
    counts: ResidentParameterCounts,
}
impl<U: 'static> MlxResidentPolicy<U> {
    pub(crate) fn parameter_sources(
        &self,
    ) -> Result<ResidentParameterSource<'_, U>, ResidentSourceError> {
        if self.layout.len() != self.units.len()
            || self.unit_ids.len() != self.units.len()
            || self._transfer.leases().len() != self.units.len()
        {
            return Err(ResidentSourceError::Selection {
                unit: self.units.len(),
            });
        }
        // Validate the whole retained selection before any caller can observe a
        // row. No lookup acquires residency or initializes a replacement module.
        for (unit, id) in self.unit_ids.iter().enumerate() {
            let address = self
                .layout
                .address(unit)
                .ok_or(ResidentSourceError::Selection { unit })?;
            if self.layout.ordinal(address.group(), address.index()) != Some(unit)
                || self._transfer.leases()[unit].id() != id
            {
                return Err(ResidentSourceError::Selection { unit });
            }
            if self.units[unit].is_none() {
                return Err(ResidentSourceError::NotIdle { unit });
            }
        }
        Ok(ResidentParameterSource { policy: self })
    }
}
impl<'s, U: 'static> ResidentParameterSource<'s, U> {
    pub(crate) fn unit(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
    ) -> Result<&'s U, ResidentSourceError> {
        if self.policy.layout.address(ordinal) != Some(address) {
            return Err(ResidentSourceError::Selection { unit: ordinal });
        }
        self.policy
            .units
            .get(ordinal)
            .and_then(Option::as_ref)
            .map(|unit| &unit.inner)
            .ok_or(ResidentSourceError::NotIdle { unit: ordinal })
    }
    pub(crate) fn layout(&self) -> &'s ExecutionUnitLayout {
        &self.policy.layout
    }
}
impl<'s, U: Parameterized<MlxTensor> + 'static> ResidentParameterSource<'s, U> {
    pub(crate) fn count(
        self,
        guard: &mut RuntimeCallGuard,
    ) -> Result<CountedResidentParameterSource<'s, U>, ResidentSourceError> {
        let mut counter = Counter::new(guard);
        for unit in 0..self.policy.units.len() {
            counter.slot = 0;
            let source = self.unit(
                unit,
                self.policy
                    .layout
                    .address(unit)
                    .expect("validated immutable layout"),
            )?;
            counter
                .observe_source(source)
                .map_err(|error| resident_error(unit, error))?;
        }
        let counts = counter.counts;
        Ok(CountedResidentParameterSource {
            source: self,
            counts,
        })
    }
}
impl<'s, U: 'static> CountedResidentParameterSource<'s, U> {
    pub(crate) const fn counts(&self) -> ResidentParameterCounts {
        self.counts
    }
    pub(crate) fn source(&self) -> &ResidentParameterSource<'s, U> {
        &self.source
    }
}

fn resident_error(unit: usize, error: ParameterCountError) -> ResidentSourceError {
    match error {
        ParameterCountError::Traversal(source) => ResidentSourceError::Traversal { unit, source },
        ParameterCountError::Descriptor { slot, source } => {
            ResidentSourceError::Descriptor { unit, slot, source }
        }
        ParameterCountError::CountOverflow => ResidentSourceError::CountOverflow,
    }
}

#[cfg(test)]
mod tests;
