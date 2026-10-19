//! Ordered channelwise residual updates, independent of additive component sums.
use super::{ComponentScalar, ComponentTransformParameter};
use serde::{Deserialize, Serialize};

/// A sublayer's complete write combined with its actual incoming residual.
/// Input and effective-output observations preserve intervention differences
/// between cycles. A readout containing cycles follows this ordered recurrence
/// instead of adding all component writes with one constant coefficient.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentResidualCycle {
    /// Stable invocation identity, distinct from shared parameter storage.
    pub id: String,
    /// Architecture node owning the combination and coefficient parameters.
    pub node_id: String,
    /// Decoder invocation containing this cycle.
    pub layer_index: usize,
    /// Effective incoming residual, after preceding interventions.
    pub input: String,
    /// Effective complete sublayer write, after its normalization and reduction.
    pub write: String,
    /// Combined residual before intervention.
    pub output: String,
    /// Actual residual consumed after this combination's intervention.
    pub effective_output: String,
    /// Exact coefficient construction and arithmetic.
    pub combination: ComponentResidualCombination,
}

/// Coefficients act independently on each residual channel. Both variants cast
/// inputs and coefficients to FP32, compute `alpha * residual + beta * write`
/// in that order, and restore the incoming residual dtype at the output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentResidualCombination {
    /// With `z = float32(gate) / temperature`, `beta = sigmoid(z)` and
    /// `alpha = sqrt(max(sigmoid(-z) * (1 + beta), minimum_variance))` in FP32.
    LearnedVariancePreserving {
        /// Learned channelwise gate, not an activation-dependent router.
        gate: ComponentTransformParameter,
        /// Positive reference temperature.
        temperature: ComponentScalar,
        /// Positive floor inside the square root.
        minimum_variance: ComponentScalar,
    },
    /// Independent stored coefficients already contain the checkpoint's gate
    /// conversion; no second sigmoid, temperature or square root is applied.
    StoredCoefficients {
        /// Incoming residual coefficient.
        alpha: ComponentTransformParameter,
        /// Sublayer-write coefficient.
        beta: ComponentTransformParameter,
    },
}
