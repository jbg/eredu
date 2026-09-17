//! Closed host container for one already-prepared incremental token tensor.

use super::{
    InferencePromptCompletion, InferenceRequest, WorkingMemoryError, WorkingMemoryFundingRun,
    WorkingMemoryReservation, funding::WorkingMemoryDecoderHostScope,
};
use crate::{PreparedInputPart, PreparedInputPayload};
use eredu_core::{HostPreparationAuthority, InputModality, PreparedInputError};
use std::{fmt, marker::PhantomData, mem::size_of, sync::Arc};

/// Cold geometry for the closed single-part incremental text input constructor.
///
/// The retained payload is one boxed `PreparedInputPart<T>`. Its metadata map and
/// extents are empty and never allocate. The conservative initialization peak
/// also includes two explicit moved part values and the incoming T handle.
/// Nested tensor backing needs independent funding and completion. This is not
/// an allocator, reference-count bookkeeping or compiler-frame bound, and does
/// not establish token contents, shape, dtype, cache identity or a runnable step.
#[must_use = "the diagnostic alone grants no construction authority"]
pub struct PendingTokenInputHostPlan<T> {
    retained: u64,
    peak: u64,
    tensor: PhantomData<fn() -> T>,
}

impl<T> PendingTokenInputHostPlan<T> {
    /// Computes exact inline payload geometry without allocating or inspecting T.
    pub fn prepare() -> Result<Self, WorkingMemoryError> {
        let slot_size = size_of::<PreparedInputPart<T>>();
        if slot_size > isize::MAX as usize {
            return Err(WorkingMemoryError::Overflow);
        }
        let retained = u64::try_from(slot_size).map_err(|_| WorkingMemoryError::Overflow)?;
        let tensor = u64::try_from(size_of::<T>()).map_err(|_| WorkingMemoryError::Overflow)?;
        let peak = retained
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(tensor))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            retained,
            peak,
            tensor: PhantomData,
        })
    }

    /// One actual retained part; empty nested containers contribute zero.
    pub fn retained_bytes(&self) -> u64 {
        self.retained
    }

    /// Protected payload and explicit move envelope, held through owner retirement.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }

    /// Additional original host preparation for this worker's actual shared
    /// shell and fixed transports. The boxed part and its moves are already in
    /// the independently protected payload peak and are not counted again.
    pub fn original_control_bytes() -> Option<usize> {
        let shared = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<PendingTokenParts<T>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            shared,
            size_of::<Self>(),
            size_of::<PreparedPendingTokenInputHost<T>>(),
            size_of::<FundedPendingTokenInput<T>>(),
            size_of::<PendingTokenParts<T>>(),
            size_of::<Arc<PendingTokenParts<T>>>(),
            size_of::<Option<Arc<PendingTokenParts<T>>>>(),
            size_of::<Option<PendingTokenParts<T>>>(),
            size_of::<InferencePendingPromptCompletion>(),
            size_of::<Result<PreparedPendingTokenInputHost<T>, WorkingMemoryError>>(),
            size_of::<
                Result<
                    (FundedPendingTokenInput<T>, InferencePendingPromptCompletion),
                    PendingTokenInputConstructionError,
                >,
            >(),
            size_of::<PendingTokenInputConstructionError>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}

impl<T> fmt::Debug for PendingTokenInputHostPlan<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingTokenInputHostPlan")
            .field("retained_bytes", &self.retained)
            .field("initialization_peak_bytes", &self.peak)
            .finish()
    }
}

pub(super) fn prepare<T>(
    completion: InferencePromptCompletion,
    reservation: &WorkingMemoryReservation,
    funding: &WorkingMemoryFundingRun,
    request: InferenceRequest,
    host: Option<HostPreparationAuthority>,
) -> Result<PreparedPendingTokenInputHost<T>, WorkingMemoryError> {
    let plan = PendingTokenInputHostPlan::prepare()?;
    let custody =
        funding.open_pending_input_scope(reservation, plan.initialization_peak_bytes())?;
    Ok(PreparedPendingTokenInputHost {
        plan,
        completion,
        request,
        custody,
        host,
    })
}

/// Move-only protected construction, minted only from the original prompt's
/// completion and exact fresh funding account. No public bytes/scope constructor
/// or callback is exposed. Dropping it before fill releases only its host hold.
#[must_use = "construct once or retire the unused host hold"]
pub struct PreparedPendingTokenInputHost<T> {
    plan: PendingTokenInputHostPlan<T>,
    completion: InferencePromptCompletion,
    request: InferenceRequest,
    custody: WorkingMemoryDecoderHostScope,
    // Retires after both the native payload and the actual shared allocation.
    host: Option<HostPreparationAuthority>,
}

impl<T> PreparedPendingTokenInputHost<T> {
    /// The same cold plan whose peak was protected before construction.
    pub fn plan(&self) -> &PendingTokenInputHostPlan<T> {
        &self.plan
    }

    /// Rechecks the original stage/account before independently funded native
    /// preparation. This borrows existing authority and cannot renew a claim.
    /// Construction performs the same check again immediately before host fill.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        let reservation = self.completion.validate_pending_token_input()?;
        self.custody
            .validate_pending_input_allocation(&reservation, self.plan.peak)
    }

    /// Moves an independently funded numerical token into one
    /// Text/TokenIds part. No tensor clone, value read, metadata entry, extent,
    /// semantic fingerprint, callback or second preparation claim is created.
    ///
    /// Native implementations must prove the supplied tensor is their exact
    /// prepared one-position token and retain its independent native source,
    /// collector and recovery custody until exact completion. Pending native
    /// work is permitted; this host construction certifies no native settlement.
    /// A closed/quarantined run or consumed prompt stage rejects before fill.
    pub fn construct(
        self,
        payload: T,
    ) -> Result<
        (FundedPendingTokenInput<T>, InferencePendingPromptCompletion),
        PendingTokenInputConstructionError,
    > {
        if let Err(error) = self.validate() {
            drop(payload);
            return Err(error.into());
        }
        #[cfg(test)]
        tests::before_construct();
        // The incoming payload is consumed/dropped by this constructor before
        // self's final host custody can retire on structural failure.
        let part = PreparedInputPart::new(
            InputModality::Text,
            PreparedInputPayload::TokenIds(payload),
            [],
        )?;
        let Self {
            plan,
            completion,
            request,
            custody,
            host,
        } = self;
        let input = FundedPendingTokenInput {
            owner: Some(Arc::new(PendingTokenParts {
                parts: Box::new([part]),
                request,
                retained: plan.retained,
                protected: plan.peak,
                custody,
                _host: host,
            })),
        };
        Ok((input, InferencePendingPromptCompletion { completion }))
    }
}

impl<T> fmt::Debug for PreparedPendingTokenInputHost<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedPendingTokenInputHost")
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

/// Closed part construction preserves its original structural error.
#[derive(Debug, thiserror::Error)]
pub enum PendingTokenInputConstructionError {
    /// The original prompt/account stopped permitting new construction.
    #[error("pending input account: {0}")]
    Memory(#[from] WorkingMemoryError),
    /// Structural validation of the fixed text part failed.
    #[error("pending input part: {0}")]
    Part(#[from] PreparedInputError),
}

struct PendingTokenParts<T> {
    // The actual boxed allocation retires first. A tensor's destructor may
    // reenter accounting while the independent host hold is still active.
    parts: Box<[PreparedInputPart<T>; 1]>,
    request: InferenceRequest,
    retained: u64,
    protected: u64,
    #[allow(dead_code)]
    custody: WorkingMemoryDecoderHostScope,
    _host: Option<HostPreparationAuthority>,
}

/// Shared read-only ownership of the actual one-part payload and its protected
/// host hold. Clones alias the allocation; no mutable or owning part export is
/// available. The owner intentionally supplies no replacement cache identity:
/// callers keep the original committed prefix on their bound decoder state.
pub struct FundedPendingTokenInput<T> {
    owner: Option<Arc<PendingTokenParts<T>>>,
}

impl<T> FundedPendingTokenInput<T> {
    fn owner(&self) -> &PendingTokenParts<T> {
        self.owner.as_deref().expect("live pending input")
    }
    /// Actual part storage, borrowed without any container or tensor copy.
    pub fn parts(&self) -> &[PreparedInputPart<T>] {
        &self.owner().parts[..]
    }
    /// Original exact request; borrowing it does not mint another start claim.
    pub fn request(&self) -> &InferenceRequest {
        &self.owner().request
    }
    /// Actual retained inline part payload, excluding the protected temporaries.
    pub fn retained_bytes(&self) -> u64 {
        self.owner().retained
    }
    /// Conservative original construction hold; aliases retain it together.
    pub fn protected_bytes(&self) -> u64 {
        self.owner().protected
    }
}

impl<T> Clone for FundedPendingTokenInput<T> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
        }
    }
}

impl<T> Drop for FundedPendingTokenInput<T> {
    fn drop(&mut self) {
        // With no exposed weak handles, exactly one alias obtains the payload.
        // Arc::into_inner retires the shared allocation before returning it;
        // payload fields then retire before their host preparation and charge.
        if let Some(owner) = self.owner.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

impl<T> fmt::Debug for FundedPendingTokenInput<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FundedPendingTokenInput")
            .field("retained_bytes", &self.retained_bytes())
            .field("protected_bytes", &self.protected_bytes())
            .finish_non_exhaustive()
    }
}

/// Finish-only remainder after the single token container was constructed.
/// There is no conversion back to a constructor or to a native work scope.
#[derive(Debug)]
#[must_use = "finish only after the original prompt is fully prepared"]
pub struct InferencePendingPromptCompletion {
    completion: InferencePromptCompletion,
}

impl InferencePendingPromptCompletion {
    /// Completes the original prompt claim; owned binding still precedes prefill.
    pub fn finish(self) -> Result<(), WorkingMemoryError> {
        self.completion.finish()
    }
}

#[cfg(test)]
mod tests;
