//! Architecture equations connecting observed component inputs and writes.

use super::{ComponentNonlinearity, ComponentNormalization, ComponentScalar};
use serde::{Deserialize, Serialize};

/// A tensor transformation between component boundaries. These declarations
/// describe execution, not derivatives or an assumption of additive attribution.
/// All paths join the ordinary observation catalog; the owning node also fixes
/// the prediction scope and logical invocation when parameters are shared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentTensorTransform {
    /// Stable identity of this transformation within the architecture descriptor.
    pub id: String,
    /// Architecture node executing this transformation.
    pub node_id: String,
    /// Actual input consumed by the equation, after any upstream intervention.
    pub input: String,
    /// Equation result before any intervention at its output.
    pub output: String,
    /// Effective result when the output has an intervention boundary. Absent
    /// means `output` is read-only evidence of the value consumed downstream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_output: Option<String>,
    /// Exact transformation and its effective parameter identities.
    pub equation: ComponentTensorTransformEquation,
}

/// Effective parameter used by a declared tensor transformation. The ordinary
/// loaded parameter catalog supplies operation support, geometry and provenance;
/// this declaration alone grants no query, copy or editing authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentTransformParameter {
    /// Canonical effective parameter identity.
    pub parameter: String,
    /// Architecture parameter group containing this parameter.
    pub parameter_group: String,
    /// Shared physical identity across logical invocations.
    pub shared_parameter: String,
}

/// Exact tensor equations used between component observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentTensorTransformEquation {
    /// Apply one shared right-hand matrix to every consecutive input group.
    /// Input `[batch, sequence, groups * input_width]` is reshaped into groups;
    /// each group multiplies the same `[input_width, output_width]` parameter.
    /// Output is `[batch, sequence, groups, output_width]`, before any axis
    /// permutation by its consumer. There is no bias or activation in this stage.
    SharedGroupedProjection {
        /// Shared effective right-hand matrix.
        weight: ComponentTransformParameter,
        /// Number of consecutive input groups.
        groups: usize,
        /// Width of each input group.
        input_width: usize,
        /// Width produced for each group.
        output_width: usize,
    },
    /// Multiply every input element by this architecture constant.
    ConstantScale {
        /// Binary32 constant used by execution.
        scale: ComponentScalar,
    },
    /// Multiply every input element by one learned scalar. The effective loaded
    /// parameter has shape `[1]`; it can be queried or edited independently.
    LearnedScale {
        /// Exact scalar parameter, including its shared identity.
        scale: ComponentTransformParameter,
    },
    /// Normalize the final axis using the declared groups, gain, bias and epsilon.
    /// Repeated normalization applications have separate transform identities.
    Normalization {
        /// Equation applied to this invocation's input.
        normalization: ComponentNormalization,
    },
    /// Causal depthwise convolution on `[batch, sequence, channel]` input.
    /// The effective kernel is `[channels, 1, taps]`, ordered oldest to current:
    /// `y[t,c] = sum_j kernel[c,0,j] * x[t + j + 1 - taps,c]`.
    /// Positions before the beginning of the logical stream contribute zero.
    /// Earlier positions may come from retained history or this invocation;
    /// the output observation covers only the invocation's current positions.
    CausalDepthwiseConvolution {
        /// Effective kernel parameter.
        kernel: ComponentTransformParameter,
        /// Global semantic channel count before partitioning.
        channels: usize,
        /// Positive number of consecutive taps; `taps - 1` is the current tap.
        taps: usize,
        /// Add the unmodified current input to the convolution result.
        residual: bool,
        /// Optional elementwise activation after the convolution and residual.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        activation: Option<ComponentNonlinearity>,
    },
}
