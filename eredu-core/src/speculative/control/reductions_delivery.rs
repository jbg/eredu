//! Borrowed delivery of one logical prefill aggregate and its retained custody.
use super::SpeculativePrefillReductions;
use crate::capture::retained_payload::RetainedCaptureOwner;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, ops::Deref};

type ReductionsOwner = RetainedCaptureOwner<Option<SpeculativePrefillReductions>>;

/// Unique empty destination for an already accounted logical prefill report.
///
/// This runtime bridge allocates only its final shared owner and concrete custody
/// Box. It grants no quota, source identity, admission or native completion. The
/// constructing runtime accounts for all report fields and provisional overlap.
#[doc(hidden)]
pub struct PreparedSpeculativePrefillReductions(ReductionsOwner);
impl PreparedSpeculativePrefillReductions {
    /// Exact retained owner and concrete custody layouts, excluding payloads.
    pub fn retained_control_bytes<C: Send + Sync + 'static>() -> Option<u64> {
        ReductionsOwner::retained_control_bytes::<C>()
    }
    /// Allocates the empty destination after its actual controls are reserved.
    /// Custody must not retain this owner, directly or indirectly.
    pub fn retain(custody: impl Send + Sync + 'static) -> Self {
        Self(ReductionsOwner::retain(None, custody))
    }
    /// Moves an already constructed report into the unique prepared destination.
    /// No allocation, validation, callback, refund or completion occurs here.
    pub fn finish(
        mut self,
        report: SpeculativePrefillReductions,
    ) -> SharedSpeculativePrefillReductions {
        *self.0.get_mut() = Some(report);
        SharedSpeculativePrefillReductions(self.0)
    }
}
impl fmt::Debug for PreparedSpeculativePrefillReductions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedSpeculativePrefillReductions")
            .finish_non_exhaustive()
    }
}

/// Immutable logical report shared with all of its retained host custody.
///
/// Cloning shares records and payloads. The last owner retires their storage
/// before custody, using the same closed owner as shared physical capture frames.
/// There is no mutable, owning raw or weak export. Serialization is the ordinary
/// report; deserialization cannot manufacture protected ownership.
#[derive(Clone)]
pub struct SharedSpeculativePrefillReductions(ReductionsOwner);
impl SharedSpeculativePrefillReductions {
    /// Packages already accounted payload and custody. This storage bridge alone
    /// establishes no quota, source authority or successful completion.
    #[doc(hidden)]
    pub fn retain(
        report: SpeculativePrefillReductions,
        custody: impl Send + Sync + 'static,
    ) -> Self {
        PreparedSpeculativePrefillReductions::retain(custody).finish(report)
    }
    /// Borrows the report without transferring its payload or custody.
    pub fn as_reductions(&self) -> &SpeculativePrefillReductions {
        self.0.get().as_ref().expect("published prefill reductions")
    }
    /// Whether both aliases retain the same actual report allocation.
    pub fn same_storage(&self, other: &Self) -> bool {
        self.0.same_storage(&other.0)
    }
}
impl AsRef<SpeculativePrefillReductions> for SharedSpeculativePrefillReductions {
    fn as_ref(&self) -> &SpeculativePrefillReductions {
        self.as_reductions()
    }
}
impl PartialEq for SharedSpeculativePrefillReductions {
    fn eq(&self, other: &Self) -> bool {
        self.as_reductions() == other.as_reductions()
    }
}
impl fmt::Debug for SharedSpeculativePrefillReductions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedSpeculativePrefillReductions")
            .field(
                "logical_invocation",
                &self.as_reductions().logical_invocation,
            )
            .field("records", &self.as_reductions().records.len())
            .field("interventions", &self.as_reductions().interventions.len())
            .finish_non_exhaustive()
    }
}
impl Serialize for SharedSpeculativePrefillReductions {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_reductions().serialize(serializer)
    }
}

impl Deref for SharedSpeculativePrefillReductions {
    type Target = SpeculativePrefillReductions;
    fn deref(&self) -> &Self::Target { self.as_reductions() }
}
impl<'de> Deserialize<'de> for SharedSpeculativePrefillReductions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Independent diagnostic ownership, never source or funding authority.
        SpeculativePrefillReductions::deserialize(deserializer).map(|report| Self::retain(report, ()))
    }
}

#[cfg(test)]
mod tests;
