//! Finite dependency allowances, separately classified from controlled metadata.
use super::funding::{AccountNode, AccountTicket, PendingAccount, PendingOriginal};
use super::transaction_buffers::RequirementProjectionSource;
use super::*;
use eredu_core::{DomainOverheadEstimate, FiniteMemoryEstimate};
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};
use std::sync::atomic::{AtomicU64, Ordering};
mod cumulative;

/// Source-labelled finite estimate and its shared prepaid Host allowance.
/// Clones retain the same charge. This grants no native or inference authority.
#[derive(Clone, Debug)]
pub struct DependencyEstimateFunding {
    // Description and its shared header retire before the final funding owner.
    estimate: EstimateOwner,
    funding: HostMetadataFunding,
}
impl DependencyEstimateFunding {
    /// The admitted range and its derivation; the upper endpoint is an estimate.
    pub fn estimate(&self) -> &DomainOverheadEstimate {
        &self.estimate.value().estimate
    }
    /// Debits the existing finite allowance without relabelling it as accounted.
    pub fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }
}

/// Typed refusal retaining an accepted account if publication is poisoned.
#[derive(Debug)]
pub struct DependencyEstimateAdmissionError {
    cause: WorkingMemoryError,
    _ticket: Option<AccountTicket>,
}
impl DependencyEstimateAdmissionError {
    /// Original capacity, arithmetic, identity or funding failure.
    pub fn cause(&self) -> &WorkingMemoryError {
        &self.cause
    }
}
impl From<WorkingMemoryError> for DependencyEstimateAdmissionError {
    fn from(cause: WorkingMemoryError) -> Self {
        Self {
            cause,
            _ticket: None,
        }
    }
}
impl std::fmt::Display for DependencyEstimateAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for DependencyEstimateAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

#[derive(Debug)]
struct EstimateDescription {
    estimate: DomainOverheadEstimate,
    ticket: AccountTicket,
}
#[derive(Debug)]
struct EstimateOwner(Option<Arc<EstimateDescription>>);
impl EstimateOwner {
    fn value(&self) -> &EstimateDescription {
        self.0.as_deref().expect("live dependency estimate")
    }
}
impl Clone for EstimateOwner {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for EstimateOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // The shared header is freed before the description and ticket.
            drop(Arc::into_inner(owner));
        }
    }
}
#[derive(Debug)]
struct EstimateAccount {
    spent: AtomicU64,
    limit: u64,
    estimate: EstimateOwner,
}
impl Drop for EstimateAccount {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.estimate.value().ticket.quarantine();
        }
    }
}
impl HostMetadataAccount for EstimateAccount {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.estimate
            .value()
            .ticket
            .status()
            .map_err(|cause| match cause {
                WorkingMemoryError::Poisoned => HostMetadataFundingError::Poisoned,
                _ => HostMetadataFundingError::Unavailable,
            })?;
        let bytes = u64::try_from(bytes).map_err(|_| HostMetadataFundingError::Overflow)?;
        self.spent
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |spent| {
                spent.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map(|_| ())
            .map_err(|spent| match spent.checked_add(bytes) {
                None => HostMetadataFundingError::Overflow,
                Some(required) => HostMetadataFundingError::Capacity {
                    required,
                    available: self.limit,
                },
            })
    }
}

struct EstimateProjection {
    controls: u64,
    upper: u64,
}
impl RequirementProjectionSource for EstimateProjection {
    fn fill(
        &self,
        topology: &MemoryTopology,
        charges: &mut [DomainMemoryCharge],
    ) -> Result<(), WorkingMemoryError> {
        for (slot, (domain, _)) in topology.domains().enumerate() {
            charges[slot] = if domain == topology.host_domain() {
                let charge = DomainMemoryCharge {
                    accounted_bytes: self.controls,
                    estimated_overhead_bytes: self.upper,
                    ..Default::default()
                };
                charge.total()?;
                charge
            } else {
                DomainMemoryCharge::default()
            };
        }
        Ok(())
    }
}

impl MemoryLedger {
    /// Atomically admits a finite Host estimate and exact accounting controls.
    /// Descriptions are copied only after acceptance; labels never deduplicate
    /// charges. Unlimited limits retain arithmetic, exclusion and custody checks.
    /// This does not apply `MemoryOverheadPolicy` or certify inference completeness.
    pub fn reserve_host_dependency_estimate(
        &self,
        execution: &InferenceExecutionIdentity,
        limits: &MemoryLimits,
        source: &str,
        range: FiniteMemoryEstimate,
        basis: &str,
    ) -> Result<DependencyEstimateFunding, DependencyEstimateAdmissionError> {
        self.reserve_dependency_estimate(execution, limits, source, range, basis, 0)
    }

    fn reserve_dependency_estimate(
        &self,
        execution: &InferenceExecutionIdentity,
        limits: &MemoryLimits,
        source: &str,
        range: FiniteMemoryEstimate,
        basis: &str,
        retention_controls: u64,
    ) -> Result<DependencyEstimateFunding, DependencyEstimateAdmissionError> {
        if source.trim().is_empty() || basis.trim().is_empty() {
            return Err(
                WorkingMemoryError::Domain(MemoryDomainError::InvalidEstimateDescription).into(),
            );
        }
        let overflow = || WorkingMemoryError::Overflow;
        let constructor = HostMetadataFunding::constructor_bytes::<EstimateAccount>()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(overflow)?;
        let limit = range
            .upper_bytes()
            .checked_add(constructor)
            .ok_or_else(overflow)?;
        let controls = controls(self.topology(), source.len(), basis.len(), constructor)?
            .checked_add(retention_controls)
            .ok_or_else(overflow)?;
        let pending = self.accept_projected(
            execution,
            EstimateProjection {
                controls,
                upper: range.upper_bytes(),
            },
            &eredu_core::MemoryLimitDeclarations::default(),
            Some(limits),
            &[],
            controls,
            |_| Ok(()),
        )?;
        let ticket = pending.publish();
        if let Err(cause) = ticket.status() {
            return Err(DependencyEstimateAdmissionError {
                cause,
                _ticket: Some(ticket),
            });
        }
        let reported = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)
            .and_then(|mut usage| {
                usage
                    .funding
                    .report_constructor_controls(ticket.id(), controls)
            });
        if let Err(cause) = reported {
            return Err(DependencyEstimateAdmissionError {
                cause,
                _ticket: Some(ticket),
            });
        }
        let estimate = EstimateOwner(Some(Arc::new(EstimateDescription {
            estimate: DomainOverheadEstimate {
                domain: self.topology().host_domain(),
                source: source.to_owned(),
                range,
                basis: basis.to_owned(),
            },
            ticket,
        })));
        let funding = HostMetadataFunding::new(EstimateAccount {
            estimate: estimate.clone(),
            spent: AtomicU64::new(0),
            limit,
        })
        .map_err(|cause| WorkingMemoryError::MetadataConstruction(cause.into()))?;
        Ok(DependencyEstimateFunding { estimate, funding })
    }
}

fn controls(
    topology: &MemoryTopology,
    source: usize,
    basis: usize,
    constructor: u64,
) -> Result<u64, WorkingMemoryError> {
    let frames = [
        size_of::<EstimateProjection>(),
        size_of::<AccountNode>(),
        size_of::<AccountTicket>(),
        size_of::<PendingAccount>(),
        size_of::<PendingOriginal>(),
        size_of::<PreparedAccountCommit<'_>>(),
        size_of::<DependencyEstimateFunding>(),
        size_of::<DependencyEstimateAdmissionError>(),
        size_of::<Result<DependencyEstimateFunding, DependencyEstimateAdmissionError>>(),
        size_of::<Result<PendingAccount, WorkingMemoryError>>(),
        size_of::<Result<HostMetadataFunding, HostMetadataFundingError>>(),
        size_of::<(
            &MemoryLedger,
            &InferenceExecutionIdentity,
            &MemoryLimits,
            &str,
            FiniteMemoryEstimate,
            &str,
        )>(),
        size_of::<(&EstimateAccount, usize, u64, u64)>(),
        size_of::<Result<u64, u64>>(),
    ];
    let fixed = frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
        .and_then(|bytes| bytes.checked_add(source))
        .and_then(|bytes| bytes.checked_add(basis))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    // The accepted account owns the old transaction vectors; replacements
    // occupy the baseline's reusable scratch slots. Its balance vector is new.
    let dense = topology
        .len()
        .checked_mul(size_of::<DomainMemoryCharge>() + size_of::<eredu_core::MemoryLimit>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    [
        constructor,
        dense,
        funding::domain_balance_bytes(topology)?,
        qualified_storage::shared_bytes::<EstimateDescription>()?,
    ]
    .into_iter()
    .try_fold(fixed, |sum, bytes| sum.checked_add(bytes))
    .ok_or(WorkingMemoryError::Overflow)
}

#[cfg(test)]
mod tests;
