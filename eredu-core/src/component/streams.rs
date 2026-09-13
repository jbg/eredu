//! Measured multi-stream residual propagation, separate from causal derivatives.
use super::{ComponentNormalization, ComponentScalar};
use serde::{Deserialize, Serialize};

/// A residual with several streams, followed by a learned final collapse.
/// At fixed measured coefficients, each stream update is linear in its incoming
/// residual and sublayer write. Coefficients must be remeasured for every trial.
/// Differences between a hook and its effective companion are separate terms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentStreamResidual {
    /// Number of parallel residual streams.
    pub streams: usize,
    /// Initial residual before these ordered sublayer cycles.
    pub base: ComponentStreamBase,
    /// Execution order, including multiple cycles in the same decoder block.
    pub cycles: Vec<ComponentStreamCycle>,
    /// Final collapse immediately before the enclosing readout normalization.
    pub head: ComponentStreamHead,
}

/// Geometry of the initial residual, independent of checkpoint naming.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentStreamBase {
    /// Repeat `[batch, sequence, hidden]` into each stream.
    Broadcast {
        /// Effective source observation before stream broadcast.
        input: String,
    },
    /// Already shaped `[batch, sequence, stream, hidden]`.
    Streams {
        /// Effective source observation already carrying the stream axis.
        input: String,
    },
}

/// Effective parameters used to predict input-dependent mixing coefficients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentStreamCoefficients {
    /// Owning architecture parameter group.
    pub parameter_group: String,
    /// Matrix reading the normalized, flattened incoming streams.
    pub function: String,
    /// Additive coefficient-logit base.
    pub base: String,
    /// Learned logit scales (three for a cycle, one for a final head).
    pub scale: String,
    /// Normalization over the flattened stream/hidden axes.
    pub normalization: ComponentNormalization,
    /// Positive offset in coefficient construction and Sinkhorn normalization.
    pub epsilon: ComponentScalar,
}

/// `output[..., i, h] = post[..., i] * write[..., h] +
/// sum_j combination[..., j, i] * input[..., j, h]`, before output intervention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentStreamCycle {
    /// Sublayer owning this cycle.
    pub node_id: String,
    /// Logical decoder invocation ordinal.
    pub layer_index: usize,
    /// Effective incoming residual streams.
    pub input: String,
    /// Pre-sublayer collapse; has a separate effective intervention companion.
    /// Its original value is `sum_j pre[..., j] * input[..., j, h]`.
    pub collapsed: String,
    /// Effective complete sublayer write, after any tensor-parallel reduction.
    pub write: String,
    /// Original expanded streams, with an effective intervention companion.
    pub output: String,
    /// Actual read-only collapse coefficients `[batch, sequence, stream]`.
    pub pre: String,
    /// Actual read-only injection coefficients with the same geometry.
    pub post: String,
    /// Actual matrix `[batch, sequence, input_stream, output_stream]`.
    pub combination: String,
    /// Parameters and RMS preparation used to predict the coefficients.
    pub coefficients: ComponentStreamCoefficients,
    /// Number of alternating row/column normalization iterations.
    pub sinkhorn_iterations: usize,
}

/// Final `sum_j coefficients[..., j] * input[..., j, h]`. Its output is the
/// enclosing readout's pre-normalization residual, before intervention there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentStreamHead {
    /// Effective incoming streams before the learned final collapse.
    pub input: String,
    /// Actual read-only collapse coefficients `[batch, sequence, stream]`.
    pub coefficients: String,
    /// Parameters and RMS preparation used to predict these coefficients.
    pub parameters: ComponentStreamCoefficients,
}
