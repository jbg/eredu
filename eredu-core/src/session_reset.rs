//! Original admission for an explicit selected-state reset, separate from text.

/// Limits for one new reset operation; prior accounts remain charged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionResetLimits {
    /// Ceiling on the complete resource domain.
    pub capacity_bytes: u64,
    /// Optional limit on this operation including its safety reserve.
    pub application_memory_budget_bytes: Option<u64>,
    /// Additional bytes retained with this operation's final owners.
    pub safety_reserve_bytes: u64,
}
impl SessionResetLimits {
    /// Selects a domain ceiling without an additional application limit.
    pub const fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            application_memory_budget_bytes: None,
            safety_reserve_bytes: 0,
        }
    }
}

/// Fixed rejection before an original reset owner exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SessionResetRejection {
    /// No complete provider is installed for this selected execution.
    #[error("an originally funded reset provider is unavailable")]
    Unsupported,
    /// A producer or completion still excludes a nonblocking reset entry.
    #[error("the model session is not ready for an admitted reset")]
    Busy,
    /// The claim names a different actual session.
    #[error("reset claim belongs to a different session")]
    ForeignSession,
    /// A checked requirement or safety sum cannot be represented.
    #[error("reset requirement overflow")]
    Overflow,
    /// Complete incremental demand exceeds the application limit.
    #[error("reset requires {required_bytes} bytes; application limit is {budget_bytes}")]
    ApplicationBudgetExceeded {
        /// Original demand plus safety.
        required_bytes: u64,
        /// Requested incremental limit.
        budget_bytes: u64,
    },
}

/// A move-only claim issued by the actual ModelRuntime reset entry.
/// It grants no byte allocation; the concrete provider must bind the current
/// selected source and obtain original domain acceptance before construction.
/// It does not itself attest synchronization or native readiness.
#[derive(Debug)]
pub struct SessionResetClaim<'a> {
    target: usize,
    limits: SessionResetLimits,
    _admission: &'a crate::SessionAdmission,
}
impl<'a> SessionResetClaim<'a> {
    pub(crate) fn new<T>(
        session: &T,
        admission: &'a crate::SessionAdmission,
        limits: SessionResetLimits,
    ) -> Self {
        Self::from_prepared_address(
            std::ptr::from_ref(session).cast::<()>() as usize,
            admission,
            limits,
        )
    }
    // The core preparation owner records this address while borrowing the actual
    // session. Its associated readiness keeps that borrow until consumption.
    pub(crate) fn from_prepared_address(
        target: usize,
        admission: &'a crate::SessionAdmission,
        limits: SessionResetLimits,
    ) -> Self {
        Self {
            target,
            limits,
            _admission: admission,
        }
    }
    /// Checks the actual borrowed backend session before any bank is consumed.
    pub fn validate_session<T>(&self, session: &T) -> Result<(), SessionResetRejection> {
        if self.target == std::ptr::from_ref(session).cast::<()>() as usize {
            Ok(())
        } else {
            Err(SessionResetRejection::ForeignSession)
        }
    }
    /// Validates the actual current capability report against original admission.
    /// This borrowed query grants no source or allocation authority.
    pub fn validate_capabilities(
        &self,
        actual: crate::SessionCapabilities,
    ) -> Result<(), crate::SessionAdmissionError> {
        self._admission.validate(actual)
    }
    /// Original caller policy, with no mutable or refill surface.
    pub const fn limits(&self) -> SessionResetLimits {
        self.limits
    }
    /// Performs the shared checked incremental comparison exactly once. The
    /// provider supplies requirements from its concrete source-bound plan;
    /// this value alone cannot construct runtime funding or a destination.
    pub fn compare(
        self,
        required_bytes: u64,
    ) -> Result<SessionResetAcceptance, SessionResetRejection> {
        let bytes = required_bytes
            .checked_add(self.limits.safety_reserve_bytes)
            .ok_or(SessionResetRejection::Overflow)?;
        if let Some(budget_bytes) = self.limits.application_memory_budget_bytes {
            if bytes > budget_bytes {
                return Err(SessionResetRejection::ApplicationBudgetExceeded {
                    required_bytes: bytes,
                    budget_bytes,
                });
            }
        }
        Ok(SessionResetAcceptance {
            bytes,
            limits: self.limits,
        })
    }
}

/// Consumed comparison result, not an execution, account or source capability.
#[derive(Debug)]
pub struct SessionResetAcceptance {
    bytes: u64,
    limits: SessionResetLimits,
}
impl SessionResetAcceptance {
    /// Complete original demand plus safety reserve.
    pub const fn required_bytes(&self) -> u64 {
        self.bytes
    }
    /// Original caller policy.
    pub const fn limits(&self) -> SessionResetLimits {
        self.limits
    }
}
