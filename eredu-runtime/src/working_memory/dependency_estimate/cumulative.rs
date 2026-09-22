//! Each bounded dependency input enters the same atomic domain transaction.
use super::*;

#[derive(Debug)]
struct Node {
    next: Option<Box<Node>>,
    estimate: DependencyEstimateFunding,
}

#[derive(Debug)]
struct State {
    constructor: Option<usize>,
    head: Option<Box<Node>>,
}
impl Drop for State {
    fn drop(&mut self) {
        while let Some(node) = self.head.take() {
            // Free the actual link before its estimate's final funding alias.
            let Node { next, estimate } = *node;
            self.head = next;
            drop(estimate);
        }
    }
}

#[derive(Debug)]
struct CumulativeAccount {
    state: std::sync::Mutex<State>,
    ledger: MemoryLedger,
    execution: InferenceExecutionIdentity,
    limits: MemoryLimits,
    source: &'static str,
    basis: &'static str,
    // Last: the account's descriptors and shared shells retire first.
    construction: HostMetadataFunding,
}

fn failure(cause: &WorkingMemoryError) -> HostMetadataFundingError {
    match cause {
        WorkingMemoryError::Domain(cause) => HostMetadataFundingError::Domain(*cause),
        WorkingMemoryError::Poisoned => HostMetadataFundingError::Poisoned,
        WorkingMemoryError::Overflow => HostMetadataFundingError::Overflow,
        _ => HostMetadataFundingError::Unavailable,
    }
}

impl HostMetadataAccount for CumulativeAccount {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| HostMetadataFundingError::Poisoned)?;
        if let Some(expected) = state.constructor {
            // The closed HostMetadataFunding constructor calls this once before
            // publishing the account. Its exact shell is not a dependency estimate.
            if bytes != expected {
                return Err(HostMetadataFundingError::Unavailable);
            }
            self.construction.reserve_metadata(bytes)?;
            state.constructor = None;
            return Ok(());
        }
        let upper = u64::try_from(bytes).map_err(|_| HostMetadataFundingError::Overflow)?;
        let range =
            FiniteMemoryEstimate::new(0, upper).map_err(|_| HostMetadataFundingError::Overflow)?;
        let controls = size_of::<Node>()
            .checked_add(size_of::<(&Self, usize, State, FiniteMemoryEstimate)>())
            .and_then(|bytes| {
                bytes.checked_add(size_of::<
                    std::sync::LockResult<std::sync::MutexGuard<'_, State>>,
                >())
            })
            .and_then(|bytes| {
                bytes.checked_add(size_of::<
                    Result<DependencyEstimateFunding, DependencyEstimateAdmissionError>,
                >())
            })
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(HostMetadataFundingError::Overflow)?;
        let estimate = self
            .ledger
            .reserve_dependency_estimate(
                &self.execution,
                &self.limits,
                self.source,
                range,
                self.basis,
                controls,
            )
            .map_err(|error| failure(error.cause()))?;
        // Admission includes this exact link. No fallible operation follows;
        // refusal above preserves both the chain and every domain/account.
        let mut node = Box::new(Node {
            next: None,
            estimate,
        });
        node.next = state.head.take();
        state.head = Some(node);
        Ok(())
    }
}

impl MemoryLedger {
    /// Builds cumulative source-labelled dependency admission for bounded inputs.
    /// Each debit is an already calculated finite upper estimate, admitted by
    /// the ordinary ledger transaction and retained until the last funding alias.
    /// Controlled bookkeeping uses `construction`; it is reported separately.
    /// The caller remains responsible for bounds, basis and payload custody.
    pub fn prepare_dependency_estimates(
        &self,
        execution: &InferenceExecutionIdentity,
        limits: &MemoryLimits,
        source: &'static str,
        basis: &'static str,
        construction: HostMetadataFunding,
    ) -> Result<HostMetadataFunding, HostMetadataFundingError> {
        limits
            .validate(self.topology())
            .map_err(HostMetadataFundingError::Domain)?;
        if source.trim().is_empty() || basis.trim().is_empty() {
            return Err(HostMetadataFundingError::Domain(
                MemoryDomainError::InvalidEstimateDescription,
            ));
        }
        let bytes = limits
            .backing_bytes()
            .map_err(HostMetadataFundingError::Domain)?;
        let bytes = bytes
            .checked_add(
                crate::working_memory::fixed_baseline::pal_mutex_bytes()
                    .ok_or(HostMetadataFundingError::Unavailable)?,
            )
            .ok_or(HostMetadataFundingError::Overflow)?;
        let frames = size_of::<(
            CumulativeAccount,
            State,
            std::sync::LockResult<std::sync::MutexGuard<'_, State>>,
            Result<HostMetadataFunding, HostMetadataFundingError>,
        )>();
        let bytes = usize::try_from(bytes)
            .ok()
            .and_then(|bytes| bytes.checked_add(frames))
            .ok_or(HostMetadataFundingError::Overflow)?;
        construction.reserve_metadata(bytes)?;
        let constructor = HostMetadataFunding::constructor_bytes::<CumulativeAccount>()
            .ok_or(HostMetadataFundingError::Overflow)?;
        HostMetadataFunding::new(CumulativeAccount {
            state: std::sync::Mutex::new(State {
                constructor: Some(constructor),
                head: None,
            }),
            ledger: self.clone(),
            execution: execution.clone(),
            limits: limits.clone(),
            source,
            basis,
            construction,
        })
    }
}
