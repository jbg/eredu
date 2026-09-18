//! Flat caller-owned destinations for cold operation and host facts.
use super::{WorkspaceOperationView, WorkspaceOutputStorageView};

pub(super) mod owned;

/// One output's storage declaration, with possible aliases in a separate buffer.
/// This contains facts only; it does not own storage or admission authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceOutputEffect {
    /// An independent allocation, including selected allocator padding.
    Allocate(u64),
    /// The complete backing of an invocation input.
    AliasInput(usize),
    /// The complete backing of a strictly earlier invocation output.
    AliasOutput(usize),
    /// A possible allocation plus an ordered range of possible input aliases.
    AllocateOrAliasInputs {
        /// Capacity of the possible independent allocation.
        bytes: u64,
        /// First entry in the separate alias buffer.
        alias_start: usize,
        /// Number of entries; duplicates retain their original meaning.
        alias_count: usize,
    },
}

impl WorkspaceOutputEffect {
    /// Resolves this declaration against its actual alias storage.
    /// A fabricated or mismatched range returns `None`, without indexing panic.
    pub fn as_view(self, aliases: &[usize]) -> Option<WorkspaceOutputStorageView<'_>> {
        Some(match self {
            Self::Allocate(bytes) => WorkspaceOutputStorageView::Allocate(bytes),
            Self::AliasInput(input) => WorkspaceOutputStorageView::AliasInput(input),
            Self::AliasOutput(output) => WorkspaceOutputStorageView::AliasOutput(output),
            Self::AllocateOrAliasInputs {
                bytes,
                alias_start,
                alias_count,
            } => WorkspaceOutputStorageView::AllocateOrAliasInputs {
                bytes,
                inputs: aliases.get(alias_start..alias_start.checked_add(alias_count)?)?,
            },
        })
    }
}

/// Exact populations in one known operation's emitted fact storage.
/// These counts are not a source certificate or a promise of funded capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceEffectLayout {
    /// One flat storage declaration per output.
    pub outputs: usize,
    /// Ordered possible-alias entries across all outputs.
    pub aliases: usize,
    /// Exact UTF-8 bytes in the tensor assumption, without a terminator.
    pub assumption_bytes: usize,
}

/// A known operation's fixed scalar facts and destination populations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceOperationFacts {
    /// Exact required destination lengths.
    pub layout: WorkspaceEffectLayout,
    /// Additional tensor backing retained through the enclosing completion.
    pub scratch_bytes: u64,
}

/// A known host-workspace fact and its separate assumption population.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceHostFacts {
    /// Host numerical payload; this excludes separately declared controls.
    pub bytes: u64,
    /// Exact UTF-8 bytes in the host assumption, without a terminator.
    pub assumption_bytes: usize,
}

/// Which exact fact destination has an incompatible length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceFactDestinationKind {
    /// Flat output declarations.
    Outputs,
    /// Ordered possible input aliases.
    Aliases,
    /// Tensor implementation assumptions.
    TensorAssumptions,
    /// Host implementation assumptions.
    HostAssumptions,
}

/// A fact destination's actual length differs from its exact required length.
/// Both short and long destinations reject; this owns no diagnostic allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("workspace {kind:?} destination length {actual} differs from {expected}")]
pub struct WorkspaceFactDestinationError {
    /// The first mismatched destination in output/alias/text order.
    pub kind: WorkspaceFactDestinationKind,
    /// Required exact number of cells or bytes.
    pub expected: usize,
    /// Supplied number of cells or bytes.
    pub actual: usize,
}

fn exact(
    kind: WorkspaceFactDestinationKind,
    expected: usize,
    actual: usize,
) -> Result<(), WorkspaceFactDestinationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(WorkspaceFactDestinationError {
            kind,
            expected,
            actual,
        })
    }
}

/// Caller-owned operation fact buffers. No method reserves or replaces them.
pub struct WorkspaceEffectDestination<'a> {
    /// Initialized cells overwritten only after full operation validation.
    pub outputs: &'a mut [WorkspaceOutputEffect],
    /// Initialized ordered-alias cells, including any duplicate candidates.
    pub aliases: &'a mut [usize],
    /// Exact assumption bytes; the producer writes valid UTF-8.
    pub assumptions: &'a mut [u8],
}

impl WorkspaceEffectDestination<'_> {
    /// Checks all destination lengths without changing any cell.
    /// The selected producer must first validate the entire operation, then
    /// call this method, then perform its deterministic emission pass.
    pub fn validate(
        &self,
        layout: WorkspaceEffectLayout,
    ) -> Result<(), WorkspaceFactDestinationError> {
        use WorkspaceFactDestinationKind::*;
        exact(Outputs, layout.outputs, self.outputs.len())?;
        exact(Aliases, layout.aliases, self.aliases.len())?;
        exact(
            TensorAssumptions,
            layout.assumption_bytes,
            self.assumptions.len(),
        )
    }
}

/// Caller-owned host assumption storage, independent of tensor effects.
pub struct WorkspaceHostDestination<'a> {
    /// Exact assumption bytes; the producer writes valid UTF-8.
    pub assumptions: &'a mut [u8],
}

impl WorkspaceHostDestination<'_> {
    /// Checks the complete host destination without changing its bytes.
    pub fn validate(&self, facts: WorkspaceHostFacts) -> Result<(), WorkspaceFactDestinationError> {
        exact(
            WorkspaceFactDestinationKind::HostAssumptions,
            facts.assumption_bytes,
            self.assumptions.len(),
        )
    }
}

/// Cold borrowed facts for a selected implementation with a fixed error type.
///
/// This separate capability preserves the existing ordinary mechanism trait.
/// Fixed count/fill implementations must perform no allocating fallback,
/// native query, or user callback. A source companion is prepared only through
/// the explicit per-operation hook before those calls. Every fill first
/// runs the same complete checked worker as count and validates all destination
/// lengths before the first write, including late geometry errors. Unknown
/// facts return `None` and leave every destination unchanged.
///
/// The concrete provider, immutable inputs and funding remain obligations of the
/// caller's original construction path. This public contract itself grants no
/// source authenticity, pool residence, memory credit or execution authority.
pub trait WorkspaceFactMechanisms {
    /// Fixed geometry/arithmetic/specification and exact-destination failures.
    type Error;

    /// Prepares an exact source companion before the pure count/fill worker.
    /// The default lends these same immutable facts. A source-aware provider
    /// may inspect its retained implementation and reserve its own concrete
    /// source preparation on `funding`, then lend a stack-local fixed provider.
    /// It must not execute tensor work, select another implementation, or infer
    /// source authority from geometry. The loan is local to this operation;
    /// no mutable current-source slot may escape into a later invocation.
    ///
    /// The caller prepays its shared error/transport shell before this hook.
    /// Implementations must pay any additional preparation and retained errors
    /// before producing them; a funding refusal must itself allocate nothing.
    fn with_prepared_facts<T>(
        &self,
        _operation: WorkspaceOperationView<'_>,
        _funding: Option<&super::HostMetadataFunding>,
        visit: impl FnOnce(&dyn WorkspaceFactMechanisms<Error = Self::Error>) -> T,
    ) -> Result<T, Self::Error>
    where
        Self: Sized,
    {
        Ok(visit(self))
    }

    /// Validates the operation and counts its exact tensor fact populations.
    fn operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error>;

    /// Validates the whole operation and destinations before emitting effects.
    fn write_operation_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error>;

    /// Validates and counts the separate host fact using the same selected rules.
    fn host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error>;

    /// Validates the complete host fact and destination before writing text.
    fn write_host_facts(
        &self,
        operation: WorkspaceOperationView<'_>,
        destination: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error>;
}

#[cfg(test)]
mod tests;
