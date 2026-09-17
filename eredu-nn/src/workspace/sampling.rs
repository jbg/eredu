//! Exact sampling primitive descriptors; ordering and adaptive policy live in runtime.

/// A model-independent native sampling operation. Shapes and dtypes accompany
/// each descriptor in `WorkspaceOperation`; facts must cover every numerical
/// value consistent with them, including arbitrary token distributions.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum WorkspaceSamplingOperation {
    /// Initial two-word explicit random key derived from a host seed.
    CreateRandomKey,
    /// Range validation retaining a dependency on the checked integer token.
    ValidateToken {
        /// Exclusive upper bound of valid canonical token IDs.
        cardinality: u32,
    },
    /// Repetition and additive penalties over the selected history window.
    Penalties {
        /// Number of accepted tokens in the selected penalty window.
        history_positions: usize,
        /// Whether sign-dependent repetition scaling is enabled.
        repetition: bool,
        /// Whether frequency or presence subtraction is enabled.
        additive: bool,
    },
    /// Exact top-k threshold and masking, with `0 < keep < vocabulary`.
    TopK {
        /// Number of highest logits retained.
        keep: u32,
    },
    /// Nucleus sorting, prefix probability scan and scatter to original order.
    TopP,
    /// Relative probability cutoff and masking.
    MinP,
    /// A nontrivial closed vocabulary mask, including host expansion.
    TokenFilter,
    /// Either a closed vocabulary mask or an unchanged input at this step.
    /// Facts cover the complete masked branch, including host expansion, and
    /// preserve the unfiltered branch's backing with `AllocateOrAliasInputs`
    /// containing input zero. Subsequent outputs may retain either backing;
    /// quoting two homogeneous schedules does not bound every mixed schedule.
    OptionalTokenFilter,
    /// Adaptive cutoff and best-token fallback, after penalties/temperature.
    MirostatCutoff,
    /// Final-axis argmax returning one unsigned token per row.
    Greedy,
    /// Split one two-word random key into two retained two-word keys.
    SplitRandomKey,
    /// Static row of an actual split-key table, preserving the selected axis.
    SelectRandomKey {
        /// First-axis row; the complete trailing two-word key is retained.
        index: u32,
    },
    /// Explicit-key F32 uniform draw in [0, 1), with output shape [1].
    /// Sequential state splitting is recorded separately.
    UniformUnitInterval,
    /// Explicit-key Gumbel sampling returning one token per row.
    Categorical,
    /// Softmax and extraction of one committed scalar probability.
    TokenProbability,
    /// Completion and host access to an already computed scalar token.
    ReadToken,
}
