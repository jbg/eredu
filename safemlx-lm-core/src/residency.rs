//! Backend-neutral residency policy, planning, accounting, and reports.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Logical storage tier. A backend maps these roles to concrete memory.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// Execution-visible memory.
    Execution,
    /// Host-accessible memory.
    Host,
    /// Persistent storage.
    Disk,
}

/// Intended lifetime of a resident unit.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyPolicy {
    /// Retain for model lifetime.
    Pinned,
    /// Retain for a bounded execution window.
    Windowed,
    /// Evict according to cache policy.
    Cacheable,
}

/// One independently managed weight/cache resource.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceSpec {
    /// Stable logical identifier.
    pub id: String,
    /// Accounted bytes.
    pub bytes: u64,
    /// Target tier.
    pub tier: MemoryTier,
    /// Lifetime policy.
    pub policy: ResidencyPolicy,
}

/// Portable validated residency plan.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResidencyPlan {
    /// Schema version, currently one.
    pub schema_version: u32,
    /// Optional budgets by tier.
    pub budgets: BTreeMap<MemoryTier, u64>,
    /// Ordered logical resource assignments.
    pub resources: Vec<ResourceSpec>,
}

impl ResidencyPlan {
    /// Validates identifiers, sizes, and finite tier budgets.
    pub fn validate(&self) -> Result<(), ResidencyError> {
        if self.schema_version != 1 {
            return Err(ResidencyError::Schema(self.schema_version));
        }
        let mut totals = BTreeMap::<MemoryTier, u64>::new();
        let mut ids = std::collections::BTreeSet::new();
        for resource in &self.resources {
            if resource.id.trim().is_empty() {
                return Err(ResidencyError::EmptyId);
            }
            if !ids.insert(&resource.id) {
                return Err(ResidencyError::Duplicate(resource.id.clone()));
            }
            if resource.bytes == 0 {
                return Err(ResidencyError::ZeroBytes(resource.id.clone()));
            }
            let total = totals.entry(resource.tier).or_default();
            *total = total
                .checked_add(resource.bytes)
                .ok_or(ResidencyError::Overflow)?;
        }
        for (tier, total) in totals {
            if let Some(limit) = self.budgets.get(&tier) {
                if total > *limit {
                    return Err(ResidencyError::Budget {
                        tier,
                        total,
                        limit: *limit,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Portable resource-accounting observation.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResidencyReport {
    /// Resident bytes by logical tier.
    pub resident_bytes: BTreeMap<MemoryTier, u64>,
    /// Peak bytes by logical tier.
    pub peak_bytes: BTreeMap<MemoryTier, u64>,
    /// Bytes transferred between tiers.
    pub transferred_bytes: u64,
    /// Cache eviction count.
    pub evictions: u64,
}

/// Residency-plan validation error.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ResidencyError {
    /// Unsupported schema version.
    #[error("unsupported residency schema version {0}")]
    Schema(u32),
    /// Empty resource identifier.
    #[error("residency resource id must not be empty")]
    EmptyId,
    /// Duplicate resource.
    #[error("duplicate residency resource {0}")]
    Duplicate(String),
    /// Zero-sized resource.
    #[error("residency resource {0} has zero bytes")]
    ZeroBytes(String),
    /// Accounting overflow.
    #[error("residency byte accounting overflow")]
    Overflow,
    /// Tier budget exceeded.
    #[error("residency tier {tier:?} uses {total} bytes, exceeding {limit}")]
    Budget {
        /// Tier.
        tier: MemoryTier,
        /// Planned bytes.
        total: u64,
        /// Budget bytes.
        limit: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plan_and_report_round_trip() {
        let plan = ResidencyPlan {
            schema_version: 1,
            budgets: [(MemoryTier::Execution, 10)].into(),
            resources: vec![ResourceSpec {
                id: "layer.0".into(),
                bytes: 8,
                tier: MemoryTier::Execution,
                policy: ResidencyPolicy::Windowed,
            }],
        };
        plan.validate().unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        assert_eq!(serde_json::from_str::<ResidencyPlan>(&json).unwrap(), plan);
        let report = ResidencyReport {
            resident_bytes: [(MemoryTier::Execution, 8)].into(),
            ..Default::default()
        };
        assert_eq!(
            serde_json::from_str::<ResidencyReport>(&serde_json::to_string(&report).unwrap())
                .unwrap(),
            report
        );
    }
}
