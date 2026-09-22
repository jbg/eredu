//! Accounted allocations, finite overhead estimates, and unknown overhead.
//!
//! These contracts compare a declared reservation scope with a memory policy.
//! They do not reserve capacity, authenticate storage, authorize execution, or
//! establish a process-memory ceiling. Inference admission and execution-context
//! policy propagation are separate from this pure evaluator.
//!
//! ```
//! use eredu_core::{
//!     evaluate_memory_requirements, FiniteMemoryEstimate, MemoryContribution,
//!     MemoryOverheadPolicy, MemoryPolicyDecision,
//! };
//!
//! let contributions = [
//!     MemoryContribution::Accounted { source: "input buffer", bytes: 1024 },
//!     MemoryContribution::Estimated {
//!         source: "parser",
//!         range: FiniteMemoryEstimate::new(128, 512)?,
//!         basis: "input-derived parser estimate",
//!     },
//!     MemoryContribution::Unknown {
//!         source: "driver",
//!         reason: "no finite estimate for internal bookkeeping",
//!     },
//! ];
//! let allowed = evaluate_memory_requirements(
//!     &contributions, MemoryOverheadPolicy::default(), Some(1792), 256,
//! )?;
//! assert_eq!(allowed.decision(), MemoryPolicyDecision::Permitted);
//! assert_eq!(allowed.report().budget_charge_bytes(), 1792);
//! assert_eq!(allowed.warnings().next().unwrap().source, "driver");
//!
//! let finite = evaluate_memory_requirements(
//!     &contributions, MemoryOverheadPolicy::RequireFiniteEstimates, Some(1792), 256,
//! )?;
//! assert_eq!(finite.decision(), MemoryPolicyDecision::FiniteEstimateRequired);
//! assert_eq!(finite.report(), allowed.report());
//! // Neither evaluation reserves capacity or establishes native completion.
//! # Ok::<(), eredu_core::MemoryContractError>(())
//! ```

mod domain;
pub use domain::{
    DomainMemoryCharge, DomainMemoryRequirements, DomainOverheadEstimate, MemoryDeviceId,
    MemoryDomainDescription, MemoryDomainError, MemoryDomainId, MemoryHeadroomDeclarations,
    MemoryLimit, MemoryLimitDeclarations, MemoryLimits, MemoryLocation, MemoryPlacement,
    MemoryPlacementKind, MemoryTopology, PlacementAllowance,
};

/// Treatment of required overhead without a finite estimate.
///
/// The runtime contract requires consumers to inherit this policy from their
/// execution context. Policy selection does not change the execution mechanism.
/// Even finite estimates remain estimates rather than guaranteed allocation bounds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MemoryOverheadPolicy {
    /// Permit fitting charges while retaining unknown overhead and emitting
    /// the evaluation's source-labelled warnings.
    #[default]
    AllowUnknownOverhead,
    /// Reject any required contribution without a finite estimate.
    RequireFiniteEstimates,
}

/// Inclusive finite estimated range in bytes, with ordered endpoints.
///
/// Neither endpoint is a guaranteed bound on actual allocations. The upper
/// estimate determines the allowance charged by the memory policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteMemoryEstimate {
    lower_bytes: u64,
    upper_bytes: u64,
}

impl FiniteMemoryEstimate {
    /// Validates the estimated range. Equal endpoints and zero are valid.
    pub const fn new(lower_bytes: u64, upper_bytes: u64) -> Result<Self, MemoryContractError> {
        if lower_bytes > upper_bytes {
            return Err(MemoryContractError::InvalidEstimateRange {
                lower_bytes,
                upper_bytes,
            });
        }
        Ok(Self {
            lower_bytes,
            upper_bytes,
        })
    }

    /// Lower estimated endpoint; not a guaranteed minimum allocation.
    pub const fn lower_bytes(self) -> u64 {
        self.lower_bytes
    }

    /// Upper estimated endpoint; not a guaranteed maximum allocation.
    pub const fn upper_bytes(self) -> u64 {
        self.upper_bytes
    }
}

/// One required, disjoint contribution to a declared reservation scope.
///
/// Sources are nonempty diagnostic labels, not storage identities. Repeated
/// labels do not deduplicate contributions. The caller must establish actual
/// allocation identity and lifetime overlap before supplying the scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryContribution<'a> {
    /// Capacity whose allocation size and lifetime Eredu controls. This
    /// declaration alone provides no evidence of reservation or ownership.
    Accounted {
        /// Allocation or mechanism being accounted for.
        source: &'a str,
        /// Accounted allocation capacity, including any controlled spare capacity.
        bytes: u64,
    },
    /// Finite estimated overhead, kept separate from controlled allocations.
    Estimated {
        /// Mechanism responsible for this overhead.
        source: &'a str,
        /// Finite estimated range; its upper endpoint is the budget allowance.
        range: FiniteMemoryEstimate,
        /// Nonempty derivation or assumptions supporting the estimate.
        basis: &'a str,
    },
    /// Overhead with range `[0, ∞)`. No finite allowance or amount of additional
    /// headroom changes this classification or establishes an upper bound.
    Unknown {
        /// Mechanism responsible for this overhead.
        source: &'a str,
        /// Nonempty explanation of the missing finite estimate.
        reason: &'a str,
    },
}

impl<'a> MemoryContribution<'a> {
    /// Diagnostic source label, without granting source identity or authority.
    pub const fn source(self) -> &'a str {
        match self {
            Self::Accounted { source, .. }
            | Self::Estimated { source, .. }
            | Self::Unknown { source, .. } => source,
        }
    }
}

/// Source-labelled unknown overhead with range `[0, ∞)`.
///
/// A permitted evaluation exposes these records as warnings to emit through the
/// consumer's diagnostic transport. Reports retain them even when policy rejects
/// the operation. This record contains no finite byte value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownMemoryOverhead<'a> {
    /// Index in the report's original contribution slice.
    pub contribution_index: usize,
    /// Mechanism responsible for this overhead.
    pub source: &'a str,
    /// Explanation of the missing finite estimate.
    pub reason: &'a str,
}

/// Validated contributions and finite charges for one reservation scope.
///
/// The scope must already account for shared identities and simultaneous
/// lifetimes. No credit is inferred for existing storage or disjoint lifetimes.
/// The report borrows descriptions; a retaining runtime must supply their funded
/// owner. Copying this view grants no storage, reservation, or execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRequirementReport<'a> {
    contributions: &'a [MemoryContribution<'a>],
    accounted_bytes: u64,
    estimated_range: FiniteMemoryEstimate,
    headroom_bytes: u64,
    budget_charge_bytes: u64,
    unknown_count: usize,
}

impl<'a> MemoryRequirementReport<'a> {
    /// All original contributions, preserving classification, sources, and order.
    pub const fn contributions(&self) -> &'a [MemoryContribution<'a>] {
        self.contributions
    }

    /// Sum of declared controlled allocation capacities.
    pub const fn accounted_bytes(&self) -> u64 {
        self.accounted_bytes
    }

    /// Sum of finite estimated ranges. Unknown overhead is reported separately;
    /// this range alone does not describe all overhead when unknown entries exist.
    pub const fn estimated_range(&self) -> FiniteMemoryEstimate {
        self.estimated_range
    }

    /// Allowance for finite estimates, equal to their summed upper endpoints.
    pub const fn estimated_allowance_bytes(&self) -> u64 {
        self.estimated_range.upper_bytes()
    }

    /// Additional configured allowance, separate from the finite estimates.
    pub const fn headroom_bytes(&self) -> u64 {
        self.headroom_bytes
    }

    /// Accounted bytes plus finite estimate allowances and additional headroom.
    /// This is a policy charge, not a total-memory estimate or process ceiling.
    pub const fn budget_charge_bytes(&self) -> u64 {
        self.budget_charge_bytes
    }

    /// Number of required contributions whose overhead remains `[0, ∞)`.
    pub const fn unknown_count(&self) -> usize {
        self.unknown_count
    }

    /// Every unknown contribution in input order, including on rejected reports.
    pub fn unknown_contributions(&self) -> impl Iterator<Item = UnknownMemoryOverhead<'a>> + '_ {
        self.contributions
            .iter()
            .enumerate()
            .filter_map(|(contribution_index, contribution)| match *contribution {
                MemoryContribution::Unknown { source, reason } => Some(UnknownMemoryOverhead {
                    contribution_index,
                    source,
                    reason,
                }),
                _ => None,
            })
    }
}

/// Outcome of the pure memory-policy comparison, without execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPolicyDecision {
    /// The finite charge fits any supplied budget and the selected overhead
    /// policy is satisfied. Independent admission and safety checks still apply.
    Permitted,
    /// The finite charge exceeds the configured budget, under either policy.
    BudgetExceeded {
        /// Accounted bytes, finite estimate allowances, and additional headroom.
        required_bytes: u64,
        /// Configured budget against which the charge was compared.
        budget_bytes: u64,
    },
    /// Strict policy requires finite estimates for the report's unknown entries.
    FiniteEstimateRequired,
}

/// Memory-policy decision with its complete borrowed report.
///
/// A decision does not reserve capacity or override invalid identities,
/// unsupported mechanisms, resource exhaustion, or unsafe completion states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "inspect the decision and deliver any warnings before proceeding"]
pub struct MemoryPolicyEvaluation<'a> {
    report: MemoryRequirementReport<'a>,
    policy: MemoryOverheadPolicy,
    budget_bytes: Option<u64>,
    decision: MemoryPolicyDecision,
}

impl<'a> MemoryPolicyEvaluation<'a> {
    /// Validated report retained on both permitted and rejected evaluations.
    pub const fn report(&self) -> &MemoryRequirementReport<'a> {
        &self.report
    }

    /// Overhead policy used by this evaluation.
    pub const fn policy(&self) -> MemoryOverheadPolicy {
        self.policy
    }

    /// Optional budget used for the finite charge comparison.
    pub const fn budget_bytes(&self) -> Option<u64> {
        self.budget_bytes
    }

    /// Policy comparison result; permission here is not execution authority.
    pub const fn decision(&self) -> MemoryPolicyDecision {
        self.decision
    }

    /// Warnings the consumer must emit if this permitted operation proceeds.
    ///
    /// Each unknown contribution produces one warning in input order. Rejected
    /// evaluations have no proceeding-with-unknown warnings; their reports still
    /// expose every unknown contribution. No diagnostic transport is selected here.
    pub fn warnings(&self) -> impl Iterator<Item = UnknownMemoryOverhead<'a>> + '_ {
        self.report
            .unknown_contributions()
            .filter(|_| self.decision == MemoryPolicyDecision::Permitted)
    }
}

/// Invalid memory declarations or arithmetic, independent of overhead policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MemoryContractError {
    /// The estimated lower endpoint exceeds the estimated upper endpoint.
    #[error("memory estimate lower endpoint {lower_bytes} exceeds upper endpoint {upper_bytes}")]
    InvalidEstimateRange {
        /// Supplied lower estimate.
        lower_bytes: u64,
        /// Supplied upper estimate.
        upper_bytes: u64,
    },
    /// A required description is empty or contains only whitespace.
    #[error("memory contribution {contribution_index} has an empty {field}")]
    MissingDescription {
        /// Index in the input contribution slice.
        contribution_index: usize,
        /// Required description: `source`, `basis`, or `reason`.
        field: &'static str,
    },
    /// A finite byte sum does not fit in `u64`.
    #[error("memory arithmetic overflow while computing {operation}")]
    ArithmeticOverflow {
        /// Stable description of the overflowing sum.
        operation: &'static str,
    },
}

/// Evaluates one declared reservation scope without allocation or side effects.
///
/// Accounted capacities and upper finite estimates are summed with additional
/// headroom. Unknown entries remain `[0, ∞)` and never contribute a fabricated
/// finite value. All declarations and finite sums are checked before applying
/// policy, so an unknown entry cannot hide a later invalid declaration or overflow.
/// Budget exhaustion takes precedence over the strict unknown-overhead refusal.
/// Equality with a supplied budget is permitted; no budget skips only comparison.
///
/// Callers establish disjoint contributions and actual lifetime overlap. This
/// function does not authenticate sources, reserve capacity, emit diagnostics,
/// select mechanisms, or establish completion. A permitted operation with unknown
/// overhead requires the consumer to emit the returned warnings and preserve the
/// unknown entries in its report.
pub fn evaluate_memory_requirements<'a>(
    contributions: &'a [MemoryContribution<'a>],
    policy: MemoryOverheadPolicy,
    budget_bytes: Option<u64>,
    headroom_bytes: u64,
) -> Result<MemoryPolicyEvaluation<'a>, MemoryContractError> {
    let mut report = MemoryRequirementReport {
        contributions,
        accounted_bytes: 0,
        estimated_range: FiniteMemoryEstimate {
            lower_bytes: 0,
            upper_bytes: 0,
        },
        headroom_bytes,
        budget_charge_bytes: 0,
        unknown_count: 0,
    };
    for (contribution_index, contribution) in contributions.iter().enumerate() {
        let description = |field, value: &str| {
            if value.trim().is_empty() {
                Err(MemoryContractError::MissingDescription {
                    contribution_index,
                    field,
                })
            } else {
                Ok(())
            }
        };
        description("source", contribution.source())?;
        match *contribution {
            MemoryContribution::Accounted { bytes, .. } => {
                report.accounted_bytes = add(report.accounted_bytes, bytes, "accounted bytes")?;
            }
            MemoryContribution::Estimated { range, basis, .. } => {
                description("basis", basis)?;
                report.estimated_range.lower_bytes = add(
                    report.estimated_range.lower_bytes,
                    range.lower_bytes(),
                    "lower estimated bytes",
                )?;
                report.estimated_range.upper_bytes = add(
                    report.estimated_range.upper_bytes,
                    range.upper_bytes(),
                    "upper estimated bytes",
                )?;
            }
            MemoryContribution::Unknown { reason, .. } => {
                description("reason", reason)?;
                // At most one count per element of the supplied slice.
                report.unknown_count += 1;
            }
        }
    }
    report.budget_charge_bytes = add(
        add(
            report.accounted_bytes,
            report.estimated_allowance_bytes(),
            "accounted bytes plus estimated allowances",
        )?,
        headroom_bytes,
        "budget charge including headroom",
    )?;
    let decision = match budget_bytes {
        Some(budget_bytes) if report.budget_charge_bytes > budget_bytes => {
            MemoryPolicyDecision::BudgetExceeded {
                required_bytes: report.budget_charge_bytes,
                budget_bytes,
            }
        }
        _ if policy == MemoryOverheadPolicy::RequireFiniteEstimates
            && report.unknown_count != 0 =>
        {
            MemoryPolicyDecision::FiniteEstimateRequired
        }
        _ => MemoryPolicyDecision::Permitted,
    };
    Ok(MemoryPolicyEvaluation {
        report,
        policy,
        budget_bytes,
        decision,
    })
}

fn add(left: u64, right: u64, operation: &'static str) -> Result<u64, MemoryContractError> {
    left.checked_add(right)
        .ok_or(MemoryContractError::ArithmeticOverflow { operation })
}

#[cfg(test)]
mod tests;
