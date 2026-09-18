//! Borrowed parameter declarations and explicit retained-value coverage.
use crate::{LinearCompanionRole, LinearRowLayout, ParameterId, ParameterMetadata, ParameterSpec};

/// Parameter metadata borrowing the actual retained construction declaration.
/// Current trainability is separate from the construction default.
///
/// The view cannot outlive its retained specification:
/// ```compile_fail,E0597
/// use eredu_nn::{ParameterMetadataView, ParameterSpec};
/// let view;
/// { let spec = ParameterSpec::trainable("weight").unwrap();
///   view = ParameterMetadataView::from_spec(&spec, false); }
/// assert!(!view.trainable());
/// ```
/// Text and current state are borrowed independently of ordinary conversion:
/// ```
/// use eredu_nn::{ParameterMetadataView, ParameterSpec};
/// let spec = ParameterSpec::trainable("weight").unwrap();
/// let view = ParameterMetadataView::from_spec(&spec, false);
/// assert!(std::ptr::eq(view.id(), &spec.id));
/// assert_eq!(view.to_owned().id, spec.id);
/// assert!(!view.trainable());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ParameterMetadataView<'a> {
    id: &'a ParameterId,
    trainable: bool,
    alias_of: Option<&'a ParameterId>,
    group: Option<&'a str>,
    linear_companion: Option<LinearCompanionRole>,
    linear_companion_of: Option<&'a ParameterId>,
    linear_row_layout: LinearRowLayout,
}
impl<'a> ParameterMetadataView<'a> {
    /// Borrows the actual specification; creates no names or owning metadata.
    pub fn from_spec(spec: &'a ParameterSpec, trainable: bool) -> Self {
        Self {
            id: &spec.id,
            trainable,
            alias_of: spec.alias_of.as_ref(),
            group: spec.group.as_deref(),
            linear_companion: spec.linear_companion,
            linear_companion_of: spec.linear_companion_of.as_ref(),
            linear_row_layout: spec.linear_row_layout,
        }
    }
    /// Exact retained authoritative parameter identity.
    pub const fn id(self) -> &'a ParameterId {
        self.id
    }
    /// Current trainability, which can differ from the construction default.
    pub const fn trainable(self) -> bool {
        self.trainable
    }
    /// Exact authoritative target for a logical alias, if declared.
    pub const fn alias_of(self) -> Option<&'a ParameterId> {
        self.alias_of
    }
    /// Borrowed atomic encoding/sharding group.
    pub const fn group(self) -> Option<&'a str> {
        self.group
    }
    /// Physical linear-companion role.
    pub const fn linear_companion(self) -> Option<LinearCompanionRole> {
        self.linear_companion
    }
    /// Primary weight to which this physical companion belongs.
    pub const fn linear_companion_of(self) -> Option<&'a ParameterId> {
        self.linear_companion_of
    }
    /// Exact independent row-block origins from the same declaration.
    pub const fn linear_row_layout(self) -> LinearRowLayout {
        self.linear_row_layout
    }
    /// Explicit owned output conversion; clones retained names at the caller boundary.
    pub fn to_owned(self) -> ParameterMetadata {
        ParameterMetadata {
            id: self.id.clone(),
            trainable: self.trainable,
            alias_of: self.alias_of.cloned(),
            group: self.group.map(str::to_owned),
            linear_companion: self.linear_companion,
            linear_companion_of: self.linear_companion_of.cloned(),
            linear_row_layout: self.linear_row_layout,
        }
    }
}
impl ParameterMetadata {
    /// Borrows an explicitly retained metadata output without cloning its names.
    pub fn as_view(&self) -> ParameterMetadataView<'_> {
        ParameterMetadataView {
            id: &self.id,
            trainable: self.trainable,
            alias_of: self.alias_of.as_ref(),
            group: self.group.as_deref(),
            linear_companion: self.linear_companion,
            linear_companion_of: self.linear_companion_of.as_ref(),
            linear_row_layout: self.linear_row_layout,
        }
    }
}

/// Fixed failure of a strict borrowed source traversal, never an owned diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ParameterSourceError {
    /// A skipped field has no complete retained-payload classification.
    #[error("parameter source contains an unclassified retained field")]
    UnclassifiedRetainedField,
    /// Actual physical membership differs from its retained named topology.
    #[error("parameter source topology differs at slot {slot}")]
    TopologyMismatch {
        /// Source-local physical slot or topology-entry ordinal.
        slot: usize,
    },
    /// An exact source count cannot be represented.
    #[error("parameter source count overflow")]
    CountOverflow,
}

/// Caller-owned observer of a module's actual borrowed parameter/value sources.
///
/// Named values already count as retained numerical fields; `retained` exposes
/// only additional fields. Aliases may repeat actual storage. A source traversal
/// error always prevents complete coverage, regardless of partial callbacks.
/// Arbitrary observer implementations are not thereby certified allocation-free.
/// Retaining a source row prevents mutation while the row is used:
/// ```compile_fail,E0502
/// use eredu_nn::{Parameter, ParameterSpec, Parameterized, ParameterSourceVisitor, ParameterMetadataView};
/// struct Row<'a>(Option<&'a i32>);
/// impl<'a> ParameterSourceVisitor<'a, i32> for Row<'a> {
///     fn parameter(&mut self, _: ParameterMetadataView<'a>, v: &'a i32) { self.0 = Some(v); }
///     fn retained(&mut self, _: &'a i32) {}
/// }
/// let mut parameter = Parameter::new(ParameterSpec::trainable("w").unwrap(), 3_i32);
/// let mut row = Row(None);
/// parameter.visit_parameter_sources(&mut row).unwrap();
/// parameter.set_trainable(false);
/// assert_eq!(row.0, Some(&3));
/// ```
pub trait ParameterSourceVisitor<'a, T: 'a> {
    /// One actual named parameter and its retained declaration.
    fn parameter(&mut self, metadata: ParameterMetadataView<'a>, value: &'a T);
    /// One auxiliary numerical owner without inventing a parameter identity.
    fn retained(&mut self, value: &'a T);
}

#[cfg(test)]
mod tests;
