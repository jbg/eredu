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

/// Ordinary or shared delivery of one logical prefill report.
///
/// Both variants have the report's existing wire format. Decoding produces only
/// ordinary caller-owned data. Shared clones retain the actual paid owner; an
/// explicit clone of the borrowed raw report is separate caller-owned storage.
#[derive(Debug, Clone)]
pub enum SpeculativePrefillReductionsDelivery {
    /// Ordinary caller-owned report, including deserialized reports.
    Legacy(SpeculativePrefillReductions),
    /// Immutable report retaining its constructing runtime's custody.
    Shared(SharedSpeculativePrefillReductions),
}
impl SpeculativePrefillReductionsDelivery {
    /// Borrows the report in either storage representation.
    pub fn as_reductions(&self) -> &SpeculativePrefillReductions {
        match self {
            Self::Legacy(value) => value,
            Self::Shared(value) => value.as_reductions(),
        }
    }
    /// Borrows the shared owner when the runtime supplied one.
    pub fn shared(&self) -> Option<&SharedSpeculativePrefillReductions> {
        match self {
            Self::Legacy(_) => None,
            Self::Shared(value) => Some(value),
        }
    }
}
impl From<SpeculativePrefillReductions> for SpeculativePrefillReductionsDelivery {
    fn from(value: SpeculativePrefillReductions) -> Self {
        Self::Legacy(value)
    }
}
impl From<SharedSpeculativePrefillReductions> for SpeculativePrefillReductionsDelivery {
    fn from(value: SharedSpeculativePrefillReductions) -> Self {
        Self::Shared(value)
    }
}
impl AsRef<SpeculativePrefillReductions> for SpeculativePrefillReductionsDelivery {
    fn as_ref(&self) -> &SpeculativePrefillReductions {
        self.as_reductions()
    }
}
impl Deref for SpeculativePrefillReductionsDelivery {
    type Target = SpeculativePrefillReductions;
    fn deref(&self) -> &Self::Target {
        self.as_reductions()
    }
}
impl PartialEq for SpeculativePrefillReductionsDelivery {
    fn eq(&self, other: &Self) -> bool {
        self.as_reductions() == other.as_reductions()
    }
}
impl Serialize for SpeculativePrefillReductionsDelivery {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_reductions().serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for SpeculativePrefillReductionsDelivery {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        SpeculativePrefillReductions::deserialize(deserializer).map(Self::Legacy)
    }
}

#[cfg(test)]
mod tests;
