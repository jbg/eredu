//! The final recovery Box exists before the native Scope constructor.
use super::*;
use safemlx::{PreparedSubmissionScopeOwner, SubmissionScopeOwnerCause};

/// Failure before both preparations exist, preserving the exact supplied owners.
pub(crate) struct RecoveryPreparationError<T, C> {
    pub(crate) cause: SubmissionScopeOwnerCause,
    pub(crate) retention: T,
    pub(crate) custody: C,
}

/// The same never-started recovery and Scope owner, returned intact on Busy.
pub(crate) struct PreparedRecoveryError<T: Retention, C: Send + 'static> {
    pub(crate) cause: SubmissionScopeOwnerCause,
    pub(crate) pending: PreparedRecovery<T, C>,
}

pub(crate) struct PreparedRecovery<T: Retention, C: Send + 'static> {
    // Prepared node Drop uses the existing guarded quarantine path, but never
    // reports Status or invokes retention callbacks before a native begin.
    node: Option<NodeOwner<T, SubmissionScope>>,
    scope: Option<PreparedSubmissionScopeOwner<C>>,
}
impl<T: Retention, C: Send + 'static> std::fmt::Debug for PreparedRecovery<T, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedRecovery").finish_non_exhaustive()
    }
}
impl<T: Retention, C: Send + 'static> PreparedRecovery<T, C> {
    pub(crate) fn new(retention: T, custody: C) -> Result<Self, RecoveryPreparationError<T, C>> {
        let scope = match PreparedSubmissionScopeOwner::try_new(custody) {
            Ok(scope) => scope,
            Err(error) => {
                let (cause, custody) = error.into_parts();
                return Err(RecoveryPreparationError {
                    cause,
                    retention,
                    custody,
                });
            }
        };
        // Box::new has the ordinary process-abort OOM contract; all recoverable
        // Scope-owner allocation failures above preserve their supplied owners.
        // No native constructor or submission has happened at this boundary.
        let node = NodeOwner(Some(Box::new(Node {
            probe: None,
            next: None,
            seal_attempted: false,
            seal_finished: false,
            callback_failed: Cell::new(false),
            last_status: Cell::new(None),
            retention,
            registration: None,
        })));
        Ok(Self {
            node: Some(node),
            scope: Some(scope),
        })
    }
    pub(crate) fn with_record_quota(
        mut self,
        quota: Option<safemlx::SubmissionRecordQuota>,
    ) -> Self {
        if let Some(quota) = quota {
            self.scope = Some(
                self.scope
                    .take()
                    .expect("unconsumed prepared scope")
                    .with_record_quota(quota),
            );
        }
        self
    }
    pub(crate) fn with_graph_quota(mut self, quota: Option<safemlx::SubmissionGraphQuota>) -> Self {
        if let Some(quota) = quota {
            self.scope = Some(
                self.scope
                    .take()
                    .expect("unconsumed prepared scope")
                    .with_graph_quota(quota),
            );
        }
        self
    }
    pub(crate) fn try_begin(self) -> Result<Recovery<T>, PreparedRecoveryError<T, C>> {
        self.try_begin_with_parent(None)
    }
    /// A separately admitted control event retains this exact active original
    /// parent while using its own prepared Graph/Record arenas.
    pub(crate) fn try_begin_with_parent(
        mut self,
        parent: Option<&safemlx::OriginalScopeObserver>,
    ) -> Result<Recovery<T>, PreparedRecoveryError<T, C>> {
        let prepared = self.scope.take().expect("unconsumed prepared scope");
        let begun = match parent {
            Some(parent) => SubmissionScope::try_begin_original_child(prepared, parent),
            None => SubmissionScope::try_begin_retaining(prepared),
        };
        match begun {
            Ok(scope) => {
                // Only moves follow native acceptance. No allocation, callback
                // or replacement node can intervene before closed ownership.
                self.node
                    .as_mut()
                    .expect("prepared recovery node")
                    .node_mut()
                    .probe = Some(scope);
                Ok(Recovery {
                    node: self.node.take(),
                })
            }
            Err(error) => {
                let (cause, prepared) = error.into_parts();
                self.scope = Some(prepared);
                Err(PreparedRecoveryError {
                    cause,
                    pending: self,
                })
            }
        }
    }
    /// Cold requested representations and named overlap for exactly one role.
    /// T/C dynamic allocations and total producer population are separate.
    pub(crate) fn control_bytes() -> Option<u64> {
        let layout = PreparedSubmissionScopeOwner::<C>::layout()?;
        let native = [
            layout.rust_node_bytes,
            layout.native_scope_bytes,
            layout.native_retirement_control_bytes,
            layout.prepared_bytes,
            layout.preparation_control_bytes,
            layout.begin_control_bytes,
            layout.retirement_control_bytes,
            layout.preparation_failure_bytes,
            layout.begin_failure_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        let extra = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Self, RecoveryPreparationError<T, C>>>(),
            size_of::<RecoveryPreparationError<T, C>>(),
            size_of::<PreparedRecoveryError<T, C>>(),
            size_of::<Result<Recovery<T>, PreparedRecoveryError<T, C>>>(),
            size_of::<Option<SubmissionScope>>(),
            size_of::<Option<&safemlx::OriginalScopeObserver>>(),
            size_of::<Result<SubmissionScope, safemlx::SubmissionScopeOwnerError<PreparedSubmissionScopeOwner<C>>>>(),
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)?;
        Recovery::<T>::node_control_bytes()?.checked_add(u64::try_from(extra).ok()?)
    }
    #[cfg(test)]
    pub(crate) fn allocation_identity(&self) -> usize {
        std::ptr::from_ref(self.node.as_ref().expect("prepared node").node()) as usize
    }
}

#[cfg(test)]
mod tests;
