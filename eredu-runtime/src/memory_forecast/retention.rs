//! Read-only retention subledgers. Pending conversions keep their ordinary
//! workspace envelope: a retention ceiling never bounds temporary execution.
use super::*;
use eredu_core::residency::{
    ParameterConversionRetentionPolicy as Policy, ParameterConversionRetentionReport,
};

/// One budget's possible future persistent payload, not an extra memory charge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversionRetentionGroupMemoryPlan {
    /// Effective selection and current claims/reservations, preserving unknown facts.
    pub report: ParameterConversionRetentionReport,
    /// Potential new admissions after published claims and outstanding reservations.
    /// First-admitted order is not predicted. Oversized attributed tensors are excluded.
    pub additional_admission_payload: MemoryBytes,
    /// Reserved payload may publish without a second allocation. Unattributed
    /// publication/backing transitions keep the phase upper unknown; this is not
    /// an additional allocation to sum into the pending conversion envelope.
    pub pending_publication_payload: MemoryBytes,
}

/// Scoped descriptive subset of parameter and workspace accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversionRetentionMemoryPlan {
    /// One entry per execution budget, regardless of its number of units or ranks.
    pub groups: Vec<ConversionRetentionGroupMemoryPlan>,
    /// Current exact backing identities/bindings, already in resident parameters.
    /// Absent means unavailable, and an empty vector means observed empty.
    pub retained_conversions: Option<Vec<crate::ResidentParameterConversion>>,
}

fn invalid(detail: impl Into<String>) -> CapabilityError {
    CapabilityError::InvalidConfiguration {
        field: "parameter conversion forecast",
        detail: detail.into(),
    }
}

impl ConversionRetentionMemoryPlan {
    /// Describes future admissions without reserving capacity, creating resources,
    /// choosing winners, or modifying the ordinary conversion workspace allowance.
    pub fn observe(
        reports: &[ParameterConversionRetentionReport],
        conversions: Option<Vec<crate::ResidentParameterConversion>>,
        topologies: &[&crate::execution_topology::TextExecutionTopology],
    ) -> Result<Self, CapabilityError> {
        if let Some(conversions) = &conversions {
            crate::residency::conversion_payload_bytes(conversions)
                .map_err(|e| invalid(e.to_string()))?;
        }
        let mut identities = std::collections::BTreeSet::new();
        let mut groups = Vec::new();
        for report in reports {
            if !identities.insert(&report.group) {
                return Err(invalid("duplicate conversion retention budget scope"));
            }
            let policy = report.policy.value().map(|p| p.effective.normalized());
            let usage = report.usage.value();
            let pending_publication_payload = usage.map_or_else(
                || MemoryBytes::unknown("outstanding conversion reservations unavailable"),
                |usage| {
                    MemoryBytes::estimated(
                        0,
                        usage.reserved_payload_bytes,
                        "reserved conversions may publish; reservation is not a second allocation",
                    )
                },
            );
            let capacity = match (policy, usage) {
                (Some(Policy::Disabled), Some(u))
                    if u.retained_payload_bytes != 0 || u.reserved_payload_bytes != 0 =>
                {
                    return Err(invalid(
                        "disabled retention has live claims or reservations",
                    ));
                }
                (Some(Policy::Disabled), _) => Some(0),
                (Some(Policy::Bounded { max_bytes }), Some(u)) => Some(
                    max_bytes
                        .checked_sub(u.retained_payload_bytes)
                        .and_then(|n| n.checked_sub(u.reserved_payload_bytes))
                        .ok_or_else(|| invalid("retained plus reserved payload exceeds policy"))?,
                ),
                _ => None,
            };
            let mut eligible = Some(0u64);
            if topologies.is_empty() {
                eligible = None;
            }
            for topology in topologies {
                let Some(total) = topology.selected_parameter_promotion_bytes else {
                    eligible = None;
                    break;
                };
                let attributed = topology
                    .selected_parameter_promotion_payloads
                    .values()
                    .try_fold(0u64, |n, &b| {
                        n.checked_add(b)
                            .ok_or_else(|| invalid("promotion payload overflow"))
                    })?;
                if attributed > total {
                    return Err(invalid("promotion attribution exceeds total"));
                }
                let accepted = topology
                    .selected_parameter_promotion_payloads
                    .values()
                    .filter(|&&bytes| capacity.is_none_or(|cap| bytes <= cap))
                    .try_fold(total - attributed, |n, &b| {
                        n.checked_add(b)
                            .ok_or_else(|| invalid("eligible payload overflow"))
                    })?;
                eligible = eligible.and_then(|n| n.checked_add(accepted));
            }
            // Unknown policy does not prove eligibility, even with known geometry.
            let upper = match (policy, capacity, eligible) {
                (Some(Policy::Disabled), _, _) => Some(0),
                (Some(Policy::Bounded { .. }), Some(cap), Some(bytes)) => Some(cap.min(bytes)),
                (Some(Policy::Bounded { .. }), Some(cap), None) => Some(cap),
                (Some(Policy::Unlimited), _, bytes) => bytes,
                _ => None,
            };
            groups.push(ConversionRetentionGroupMemoryPlan {
                report: report.clone(),
                additional_admission_payload: MemoryBytes {
                    lower_bytes: 0,
                    upper_bytes: upper,
                    kind: eredu_core::ObservationKind::Estimated,
                    detail: "potential persistent subset of pending conversion workspace; bounded by eligible payload and unreserved group capacity, with no predicted admission order".into(),
                },
                pending_publication_payload,
            });
        }
        Ok(Self {
            groups,
            retained_conversions: conversions,
        })
    }

    /// Refreshes the pending subledger after exact credits or embedded composition.
    pub fn refresh(
        &mut self,
        topologies: &[&crate::execution_topology::TextExecutionTopology],
    ) -> Result<(), CapabilityError> {
        *self = Self::observe(
            &self
                .groups
                .iter()
                .map(|g| g.report.clone())
                .collect::<Vec<_>>(),
            self.retained_conversions.clone(),
            topologies,
        )?;
        Ok(())
    }
}

/// Exact scoped binding credit. Logical names alone never cross budget scopes.
/// Callers provide current observations from their selected retaining owners.
pub(crate) fn credit_topology(
    topology: &mut crate::execution_topology::TextExecutionTopology,
    conversions: &[crate::ResidentParameterConversion],
    reports: Option<&[ParameterConversionRetentionReport]>,
) -> Result<u64, CapabilityError> {
    let Some(total) = topology.selected_parameter_promotion_bytes.as_mut() else {
        return Ok(0);
    };
    let attributed = topology
        .selected_parameter_promotion_payloads
        .values()
        .try_fold(0u64, |n, &b| {
            n.checked_add(b)
                .ok_or_else(|| invalid("promotion overflow"))
        })?;
    if attributed > *total {
        return Err(invalid("promotion attribution exceeds aggregate"));
    }
    let mut credited = 0u64;
    for conversion in conversions {
        let eredu_core::resources::ResourceSize::Fixed { extent } = &conversion.allocation.size
        else {
            return Err(invalid("conversion payload is not fixed"));
        };
        for binding in &conversion.bindings {
            if let Some(reports) = reports {
                if !reports.iter().any(|r| {
                    binding.retention_group.as_ref() == Some(&r.group)
                        && r.usage.value().is_some_and(|u| {
                            u.retained_payload_bytes >= extent.payload.lower_bytes
                                && u.retained_claims > 0
                        })
                }) {
                    continue;
                }
            }
            let Some(name) = &binding.logical_target else {
                continue;
            };
            if let Some(bytes) = topology.selected_parameter_promotion_payloads.get_mut(name) {
                if *bytes != 0 && *bytes == extent.payload.lower_bytes {
                    credited = credited
                        .checked_add(*bytes)
                        .ok_or_else(|| invalid("conversion credit overflow"))?;
                    *bytes = 0;
                }
            }
        }
    }
    *total = total
        .checked_sub(credited)
        .ok_or_else(|| invalid("credit exceeds promotion total"))?;
    Ok(credited)
}

/// Deduplicates only actual shared backing in a single matching execution pool.
/// Independent groups keep their claims; neither policy ceilings nor names imply sharing.
pub(crate) fn shared_conversion_payload(
    target: &GenerationMemoryRequest,
    draft: &GenerationMemoryRequest,
    domain: &MemoryDomain,
) -> Result<u64, CapabilityError> {
    let sole_pool = |r: &GenerationMemoryRequest| {
        let mut pools = r.domains.iter().filter(|d| !d.executions.is_empty());
        pools.next().is_some_and(|d| &d.domain == domain) && pools.next().is_none()
    };
    if !sole_pool(target) || !sole_pool(draft) {
        return Ok(0);
    }
    fn observations(
        r: &GenerationMemoryRequest,
    ) -> Option<&Vec<crate::ResidentParameterConversion>> {
        r.parameter_conversion_retention
            .as_ref()
            .and_then(|p| p.retained_conversions.as_ref())
    }
    let (Some(a), Some(b)) = (observations(target), observations(draft)) else {
        return Ok(0);
    };
    for conversions in [a, b] {
        crate::residency::conversion_payload_bytes(conversions)
            .map_err(|e| invalid(e.to_string()))?;
    }
    let mut shared = 0u64;
    for left in a {
        if let Some(right) = b
            .iter()
            .find(|right| right.allocation.identity == left.allocation.identity)
        {
            if left.allocation.size != right.allocation.size {
                return Err(invalid("shared conversion has conflicting extents"));
            }
            let eredu_core::resources::ResourceSize::Fixed { extent } = &left.allocation.size
            else {
                return Err(invalid("shared conversion is not fixed"));
            };
            shared = shared
                .checked_add(extent.payload.lower_bytes)
                .ok_or_else(|| invalid("shared conversion overflow"))?;
        }
    }
    Ok(shared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_topology::{ProjectionTopology, TextExecutionTopology};
    use eredu_core::{residency::*, resources::*, Observed};

    fn identity(key: &str) -> ResourceIdentity {
        ResourceIdentity {
            scope: "test".into(),
            key: key.into(),
        }
    }
    fn report(policy: Policy, retained: u64, reserved: u64) -> ParameterConversionRetentionReport {
        ParameterConversionRetentionReport {
            group: ParameterConversionRetentionGroup(identity("group")),
            policy: Observed::exact(
                ParameterConversionRetentionPolicyReport::resolve(
                    Some(policy),
                    ParameterConversionRetentionEligibility::Eligible,
                ),
                "fixture",
            ),
            usage: Observed::exact(
                ParameterConversionRetentionUsage {
                    retained_claims: u64::from(retained != 0),
                    retained_payload_bytes: retained,
                    reserved_payload_bytes: reserved,
                    retained_backing_capacity_bytes: Observed::unavailable("padding unknown"),
                },
                "fixture",
            ),
        }
    }
    fn topology() -> TextExecutionTopology {
        TextExecutionTopology {
            hidden_size: 4,
            vocabulary_size: 4,
            layers: vec![],
            output: ProjectionTopology {
                input: 4,
                output: 4,
                format: eredu_checkpoint::LinearFormat::Dense,
                bias: false,
                parameter: "a".into(),
            },
            output_invocations: 1,
            output_softcap: false,
            selected_parameter_promotion_bytes: Some(300),
            selected_parameter_promotion_payloads: [("a".into(), 100), ("b".into(), 200)].into(),
            missing: vec![],
        }
    }
    fn conversion(
        group: ParameterConversionRetentionGroup,
        bytes: u64,
    ) -> crate::ResidentParameterConversion {
        crate::ResidentParameterConversion {
            allocation: ResourceAllocation {
                identity: identity("allocation"),
                uses: vec![ResourceUse {
                    owner: identity("owner"),
                    role: ResourceRole::Parameters,
                }],
                placement: Observed::unavailable("native pool unknown"),
                size: ResourceSize::Fixed {
                    extent: ResourceExtent {
                        payload: ResourceByteBounds::exact(bytes),
                        capacity: ResourceByteBounds::unknown(bytes, "padding unknown"),
                    },
                },
            },
            bindings: vec![crate::ResidentParameterConversionBinding {
                retention_group: Some(group),
                owner: identity("owner"),
                unit: OffloadUnitId::new("unit").unwrap(),
                name: "a".into(),
                logical_target: Some("a".into()),
            }],
        }
    }
    #[test]
    fn bounds_cover_empty_partial_full_oversized_disabled_unlimited_and_reservations() {
        let t = topology();
        for (policy, retained, reserved, expected) in [
            (Policy::Disabled, 0, 0, 0),
            (Policy::Bounded { max_bytes: 0 }, 0, 0, 0),
            (Policy::Bounded { max_bytes: 99 }, 0, 0, 0),
            (Policy::Bounded { max_bytes: 100 }, 0, 0, 100),
            (Policy::Bounded { max_bytes: 250 }, 0, 0, 250),
            (Policy::Bounded { max_bytes: 300 }, 100, 0, 200),
            (Policy::Bounded { max_bytes: 300 }, 300, 0, 0),
            (Policy::Bounded { max_bytes: 300 }, 100, 100, 100),
            (Policy::Unlimited, 0, 0, 300),
        ] {
            let r = report(policy, retained, reserved);
            let before = r.clone();
            let plan = ConversionRetentionMemoryPlan::observe(&[r.clone()], None, &[&t]).unwrap();
            assert_eq!(
                plan.groups[0].additional_admission_payload.upper_bytes,
                Some(expected)
            );
            assert_eq!(
                plan.groups[0].pending_publication_payload.upper_bytes,
                Some(reserved)
            );
            assert_eq!(r, before, "forecast does not mutate a budget observation");
            assert_eq!(
                t.selected_parameter_promotion_bytes,
                Some(300),
                "temporary cast envelope is independent of retention"
            );
        }
        assert!(ConversionRetentionMemoryPlan::observe(
            &[report(Policy::Bounded { max_bytes: 100 }, 100, 1)],
            None,
            &[&t]
        )
        .is_err());
    }
    #[test]
    fn exact_scoped_binding_credit_refreshes_after_trim_and_invalidation() {
        let r = report(Policy::Bounded { max_bytes: 300 }, 100, 0);
        let mut t = topology();
        let c = conversion(r.group.clone(), 100);
        assert_eq!(
            credit_topology(&mut t, &[c.clone()], Some(&[r.clone()])).unwrap(),
            100
        );
        assert_eq!(
            credit_topology(&mut t, &[c.clone()], Some(&[r.clone()])).unwrap(),
            0
        );
        assert_eq!(t.selected_parameter_promotion_bytes, Some(200));
        // Reconstruct from immutable selected geometry at every observation.
        for conversions in [
            vec![],
            vec![conversion(
                ParameterConversionRetentionGroup(identity("other-group")),
                100,
            )],
            vec![conversion(r.group.clone(), 50)],
        ] {
            let mut refreshed = topology();
            assert_eq!(
                credit_topology(&mut refreshed, &conversions, Some(&[r.clone()])).unwrap(),
                0
            );
            assert_eq!(refreshed.selected_parameter_promotion_bytes, Some(300));
        }
        let mut shared = c.clone();
        shared.bindings.push(shared.bindings[0].clone());
        assert_eq!(
            credit_topology(&mut topology(), &[shared], Some(&[r.clone()])).unwrap(),
            100
        );
        let trimmed = report(Policy::Bounded { max_bytes: 300 }, 0, 0);
        assert_eq!(
            credit_topology(&mut topology(), &[c], Some(&[trimmed.clone()])).unwrap(),
            0,
            "another owner holding backing cannot keep a released claim's credit"
        );
        assert_eq!(
            ConversionRetentionMemoryPlan::observe(&[trimmed], Some(vec![]), &[&topology()])
                .unwrap()
                .groups[0]
                .additional_admission_payload
                .upper_bytes,
            Some(300)
        );
    }
    #[test]
    fn missing_policy_and_mechanisms_stay_unknown_and_round_trip() {
        let mut r = report(Policy::Bounded { max_bytes: 100 }, 0, 0);
        let plan = ConversionRetentionMemoryPlan::observe(&[r.clone()], None, &[]).unwrap();
        assert_eq!(
            plan.groups[0].additional_admission_payload.upper_bytes,
            Some(100)
        );
        // This bounded subset conveys no upper bound for temporary casts or other mechanisms.
        assert_eq!(
            serde_json::from_str::<ConversionRetentionMemoryPlan>(
                &serde_json::to_string(&plan).unwrap()
            )
            .unwrap(),
            plan
        );
        r.policy = Observed::unavailable("historical observation");
        assert!(
            ConversionRetentionMemoryPlan::observe(&[r.clone()], None, &[&topology()])
                .unwrap()
                .groups[0]
                .additional_admission_payload
                .upper_bytes
                .is_none()
        );
        r.policy = Observed::exact(
            ParameterConversionRetentionPolicyReport::resolve(
                Some(Policy::Unlimited),
                ParameterConversionRetentionEligibility::Eligible,
            ),
            "fixture",
        );
        assert!(ConversionRetentionMemoryPlan::observe(&[r], None, &[])
            .unwrap()
            .groups[0]
            .additional_admission_payload
            .upper_bytes
            .is_none());
    }
}
