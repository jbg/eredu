//! Checked symbolic state geometry shared by execution, persistence and inspection.

use super::{CachePolicyError, StateComponentPolicy, StateTensorDimension, StateTensorPresence};

/// Element-count bounds for a component or its maximum over a prefix interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateElementBounds {
    /// Required elements; optional presence can make this zero.
    pub minimum: u64,
    /// Maximum elements permitted by the symbolic shape and presence contract.
    pub maximum: u64,
}

impl StateComponentPolicy {
    /// Evaluates the ordinary shape and presence contract at an exact prefix.
    /// Does not imply allocation capacity or that a live state is installed.
    pub fn element_bounds(
        &self,
        batch: u64,
        prefix: u64,
    ) -> Result<StateElementBounds, CachePolicyError> {
        self.element_peak_bounds(batch, prefix, prefix)
    }

    /// Bounds this component's largest payload over an inclusive prefix interval.
    ///
    /// Quotient/remainder axes may be nonmonotone. Independent dimension maxima
    /// provide a conservative upper end even when their maxima occur at different
    /// prefixes. Endpoint payloads provide a lower end; no token-wise iteration
    /// or assumption of linear growth is used. Optional components retain zero
    /// as their lower end. The shape, not a family or forecast, owns this rule.
    pub fn element_peak_bounds(
        &self,
        batch: u64,
        first: u64,
        last: u64,
    ) -> Result<StateElementBounds, CachePolicyError> {
        if batch == 0 || first > last {
            return Err(invalid(
                "state geometry requires positive batch and ordered prefixes",
            ));
        }
        let first = match self.presence {
            StateTensorPresence::PrefixAtLeast(n) => first.max(u64::from(n.get())),
            _ => first,
        };
        if first > last {
            return Ok(StateElementBounds {
                minimum: 0,
                maximum: 0,
            });
        }
        let present = |prefix| match self.presence {
            StateTensorPresence::PrefixRemainderNonZero(n) => prefix % u64::from(n.get()) != 0,
            _ => true,
        };
        if first == last && !present(first) {
            return Ok(StateElementBounds {
                minimum: 0,
                maximum: 0,
            });
        }
        // A divisor of one never admits a remainder-conditioned component.
        if matches!(self.presence, StateTensorPresence::PrefixRemainderNonZero(n) if n.get() == 1) {
            return Ok(StateElementBounds {
                minimum: 0,
                maximum: 0,
            });
        }
        let elements = |prefix| -> Result<u64, CachePolicyError> {
            if !present(prefix) {
                return Ok(0);
            }
            self.shape.iter().try_fold(1u64, |count, axis| {
                count
                    .checked_mul(value(*axis, batch, prefix))
                    .ok_or_else(|| invalid("state element count overflowed"))
            })
        };
        let minimum = if self.presence == StateTensorPresence::Optional {
            0
        } else {
            elements(first)?.max(elements(last)?)
        };
        let maximum = self.shape.iter().try_fold(1u64, |count, axis| {
            let largest = match axis {
                StateTensorDimension::PrefixTokensRem(n) => {
                    let divisor = u64::from(n.get());
                    if first / divisor == last / divisor {
                        last % divisor
                    } else {
                        divisor - 1
                    }
                }
                _ => value(*axis, batch, last),
            };
            count
                .checked_mul(largest)
                .ok_or_else(|| invalid("state horizon element count overflowed"))
        })?;
        Ok(StateElementBounds { minimum, maximum })
    }
}

fn value(axis: StateTensorDimension, batch: u64, prefix: u64) -> u64 {
    match axis {
        StateTensorDimension::Batch => batch,
        StateTensorDimension::PrefixTokens => prefix,
        StateTensorDimension::PrefixTokensDiv(n) => prefix / u64::from(n.get()),
        StateTensorDimension::PrefixTokensRem(n) => prefix % u64::from(n.get()),
        StateTensorDimension::Fixed(n) => u64::from(n.get()),
        StateTensorDimension::Scalar => 1,
    }
}

fn invalid(message: &str) -> CachePolicyError {
    CachePolicyError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        LayerCachePolicy, MutableStateResidency, StateTensorDtype, StateTensorPolicy,
        StateTensorRole,
    };
    use std::num::NonZeroU32;

    fn component(
        shape: Vec<StateTensorDimension>,
        presence: StateTensorPresence,
    ) -> StateComponentPolicy {
        let mut tensor = StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            shape,
            StateTensorDtype::Float32,
            MutableStateResidency::LayerScopedOffloadable,
        )
        .unwrap();
        tensor.presence = presence;
        LayerCachePolicy::fixed_only(vec![tensor])
            .unwrap()
            .components()
            .remove(0)
    }

    #[test]
    fn nonlinear_horizon_covers_interior_and_optional_presence() {
        let n = NonZeroU32::new(4).unwrap();
        let c = component(
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensDiv(n),
                StateTensorDimension::PrefixTokensRem(n),
            ],
            StateTensorPresence::Required,
        );
        let bound = c.element_peak_bounds(2, 3, 8).unwrap();
        assert_eq!(bound.minimum, 0);
        assert_eq!(bound.maximum, 12);
        for prefix in 3..=8 {
            assert!(c.element_bounds(2, prefix).unwrap().maximum <= bound.maximum);
        }
        let c = component(
            vec![StateTensorDimension::Fixed(n)],
            StateTensorPresence::Optional,
        );
        assert_eq!(
            c.element_bounds(1, 5).unwrap(),
            StateElementBounds {
                minimum: 0,
                maximum: 4
            }
        );
    }

    #[test]
    fn conditional_zero_scalar_and_overflow_are_explicit() {
        let n = NonZeroU32::new(4).unwrap();
        let c = component(
            vec![StateTensorDimension::Scalar],
            StateTensorPresence::PrefixAtLeast(n),
        );
        assert_eq!(c.element_bounds(1, 3).unwrap().maximum, 0);
        assert_eq!(
            c.element_peak_bounds(1, 0, 4).unwrap(),
            StateElementBounds {
                minimum: 1,
                maximum: 1
            }
        );
        let c = component(
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokens,
            ],
            StateTensorPresence::Required,
        );
        assert!(c.element_bounds(u64::MAX, 2).is_err());
        assert!(c.element_peak_bounds(1, 5, 4).is_err());
        assert!(c.element_bounds(0, 0).is_err());
    }
}
