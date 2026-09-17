//! Lexical access to the current selected execution owner before chunk preparation.
trait OpeningExecutionValues<T> {
    fn visit(&self, visitor: &mut dyn FnMut(&T)) -> bool;
}

/// Read-only access to the actual execution paired with this SessionPrefill.
///
/// The runtime alone constructs this view while retaining its exclusive session
/// loan. This is source access, not proof of native settlement, registrations,
/// complete residency-manager inventory, available memory or execution authority.
/// The concrete traversal and any inventory it builds retain their own physical
/// cost obligations; this adapter itself allocates nothing and clones no values.
pub struct PrefillOpeningExecution<'a, T> {
    source: &'a dyn OpeningExecutionValues<T>,
    prepared: Option<&'a crate::PreparedLayeredObservationPaths>,
}
impl<T> PrefillOpeningExecution<'_, T> {
    /// Actual stored token validated by the selected strategy for this loan.
    /// None means no prepared-token validation was requested. This borrow is
    /// neither persistent binding nor freshness after the callback returns.
    pub fn prepared_paths(&self) -> Option<&crate::PreparedLayeredObservationPaths> {
        self.prepared
    }

    /// Visits the current execution's retained parameters and numerical helpers.
    /// False means incomplete coverage, possibly after visiting known values.
    /// Consumers must preserve that distinction rather than infer an empty source.
    pub fn visit(&self, visitor: &mut dyn FnMut(&T)) -> bool {
        self.source.visit(visitor)
    }

    /// Fixed borrowed view/source controls and their construction moves only.
    /// Native inventory containers and concrete traversal controls are separate.
    pub fn control_peak_bytes() -> Option<u64> {
        std::mem::size_of::<Self>()
            .checked_add(std::mem::size_of::<RuntimeOpeningExecution<'_, (), ()>>())?
            .checked_mul(3)?
            .try_into()
            .ok()
    }
}

// Stores only the actual runtime loan and the selected strategy's fixed visitor.
// No user callback can manufacture a PrefillOpeningExecution.
pub(crate) struct RuntimeOpeningExecution<'a, R, T> {
    runtime: &'a R,
    visit: fn(&R, &mut dyn FnMut(&T)) -> bool,
}
impl<'a, R, T> RuntimeOpeningExecution<'a, R, T> {
    pub(crate) fn new(runtime: &'a R, visit: fn(&R, &mut dyn FnMut(&T)) -> bool) -> Self {
        Self { runtime, visit }
    }
    pub(crate) fn borrow_prepared<'s>(
        &'s self,
        prepared: Option<&'s crate::PreparedLayeredObservationPaths>,
    ) -> PrefillOpeningExecution<'s, T> {
        PrefillOpeningExecution {
            source: self,
            prepared,
        }
    }
    pub(crate) fn borrow(&self) -> PrefillOpeningExecution<'_, T> {
        PrefillOpeningExecution {
            source: self,
            prepared: None,
        }
    }
}
impl<R, T> OpeningExecutionValues<T> for RuntimeOpeningExecution<'_, R, T> {
    fn visit(&self, visitor: &mut dyn FnMut(&T)) -> bool {
        (self.visit)(self.runtime, visitor)
    }
}
