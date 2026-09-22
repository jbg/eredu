//! Original admission for an explicit selected-state reset, separate from text.

/// Limits for one new reset operation; prior accounts remain charged.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct SessionResetLimits {
    /// Total live-charge limits in every physical domain.
    pub memory_limits: crate::MemoryLimitDeclarations,
    /// Additional domain-attributed bytes retained with this operation's owners.
    pub additional_headroom: crate::MemoryHeadroomDeclarations,
}
impl SessionResetLimits {
    /// Selects domain limits without additional headroom.
    pub const fn new(memory_limits: crate::MemoryLimitDeclarations) -> Self {
        Self {
            memory_limits,
            additional_headroom: crate::MemoryHeadroomDeclarations::none(),
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
    /// A domain declaration, identity, or checked requirement is invalid.
    #[error(transparent)]
    MemoryDomain(#[from] crate::MemoryDomainError),
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
    pub const fn limits(&self) -> &SessionResetLimits {
        &self.limits
    }
    /// Resolves the complete domain demand and policy exactly once. The provider
    /// supplies requirements from its source-bound plan. The ledger atomically
    /// compares these against existing live charges before issuing funding.
    pub fn compare(
        self,
        topology: &crate::MemoryTopology,
        requirements: crate::DomainMemoryRequirements,
    ) -> Result<SessionResetAcceptance, SessionResetRejection> {
        requirements.validate(topology)?;
        let requirements =
            requirements.checked_add(&self.limits.additional_headroom.resolve(topology)?)?;
        let limits = self.limits.memory_limits.resolve(topology)?;
        for (domain, charge) in requirements.iter() {
            limits.get(domain)?.check(domain, 0, charge.total()?)?;
        }
        Ok(SessionResetAcceptance {
            requirements,
            limits,
        })
    }
}

/// Consumed comparison result, not an execution, account or source capability.
#[derive(Debug)]
pub struct SessionResetAcceptance {
    requirements: crate::DomainMemoryRequirements,
    limits: crate::MemoryLimits,
}
impl SessionResetAcceptance {
    /// Moves the pure comparison into the ledger without duplicating vectors.
    /// This result still supplies neither source ownership nor execution authority.
    pub fn into_parts(self) -> (crate::DomainMemoryRequirements, crate::MemoryLimits) {
        (self.requirements, self.limits)
    }

    /// Complete original demand plus domain-attributed headroom.
    pub const fn requirements(&self) -> &crate::DomainMemoryRequirements {
        &self.requirements
    }
    /// Original caller policy.
    pub const fn limits(&self) -> &crate::MemoryLimits {
        &self.limits
    }
}
