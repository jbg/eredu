//! Ordinary reusable text-module topology retained by cold preparation.
//!
//! Architecture records describe construction and invocation geometry. Optional
//! backend storage refinements attach only to loaded observations; allocator
//! policy and native hardware selection remain outside this topology.

use eredu_checkpoint::LinearFormat;
use eredu_nn::{Error, LinearSpec};
use serde::{Deserialize, Serialize};

/// Selected projection geometry and authoritative parameter owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionTopology {
    /// Complete input width.
    pub input: u64,
    /// Output width.
    pub output: u64,
    /// Physical matrix encoding.
    #[serde(with = "linear_format")]
    pub format: LinearFormat,
    /// Whether an output bias is applied.
    pub bias: bool,
    /// Canonical parameter identity, preserving explicit aliases.
    pub parameter: String,
}

impl ProjectionTopology {
    /// Reuses the exact specification consumed by `NeuralBackend::linear`.
    pub fn from_spec(spec: &LinearSpec) -> Result<Self, Error> {
        if spec.input <= 0 || spec.output <= 0 {
            return Err(Error::backend(
                "projection topology dimensions must be positive",
            ));
        }
        Ok(Self {
            input: spec.input as u64,
            output: spec.output as u64,
            format: spec.format.encoding(),
            bias: spec.bias.is_some(),
            parameter: spec
                .weight
                .alias_of
                .as_ref()
                .unwrap_or(&spec.weight.id)
                .as_str()
                .into(),
        })
    }
}

/// Reusable token-mixer equations selected for a layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TokenMixerTopology {
    /// An ordinary mixer whose invocation geometry is not yet available.
    Unknown {
        /// Concrete missing mechanism, never interpreted as zero work.
        reason: String,
    },
    /// Ordinary grouped-query attention with its selected arithmetic contract.
    Attention {
        /// Query heads.
        query_heads: u64,
        /// Key/value heads.
        kv_heads: u64,
        /// Query/key width per head.
        key_width: u64,
        /// Value width per head.
        value_width: u64,
        /// Preserve the input-dtype score/probability rounding boundaries.
        input_scores: bool,
        /// Scores require a tanh cap.
        softcap: bool,
        /// Learned sink logits are present.
        sinks: bool,
        /// Exact split/fused input and output projections in invocation order.
        projections: Vec<ProjectionTopology>,
        /// A sigmoid gate multiplies the attended query features.
        #[serde(default)]
        output_gate: bool,
        /// Q/K normalization is applied before attention.
        query_key_normalization: bool,
        /// Rotary transforms are applied to query and key values.
        rotary: bool,
    },
    /// Three-way gated causal convolution and output projection.
    GatedConvolution {
        /// Independent convolution channels.
        channels: u64,
        /// Causal kernel width.
        kernel: u64,
        /// Input and output projections in invocation order.
        projections: Vec<ProjectionTopology>,
    },
}

/// Feed-forward equations composed with a token mixer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeedForwardTopology {
    /// Split or fused gated product followed by an output projection.
    Gated {
        /// Width of the gated intermediate.
        intermediate_size: u64,
        /// Exact input and output projections in invocation order.
        projections: Vec<ProjectionTopology>,
    },
    /// Top-k routing into a packed gated-product expert bank.
    Routed {
        /// Available expert count.
        experts: u64,
        /// Selected experts per input row.
        selected: u64,
        /// Intermediate width within each expert.
        intermediate_size: u64,
        /// Router projection before scoring and selection.
        router: ProjectionTopology,
        /// Packed bank projections, with dimensions per expert.
        projections: Vec<ProjectionTopology>,
    },
    /// A mechanism not yet described by ordinary preparation.
    Unknown {
        /// Concrete missing invocation fact; never interpreted as zero work.
        reason: String,
    },
}

impl FeedForwardTopology {
    /// Derives invocation geometry from the exact selector and expert construction specs.
    pub fn from_grouped_specs(
        selector: &eredu_nn::TopKGroupSelectorSpec,
        bank: &eredu_nn::GroupedGatedProductSpec,
    ) -> Result<Self, Error> {
        bank.validate()?;
        selector.validate()?;
        if selector.selection().group_count() != bank.group_count() {
            return Err(Error::backend("selector and bank group counts disagree"));
        }
        if selector.input_transform().is_some() || selector.coefficient_scale().is_some() {
            return Ok(Self::Unknown {
                reason: "selector input transformation or learned coefficient module has no invocation topology".into(),
            });
        }
        let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = bank.layout() else {
            return Ok(Self::Unknown { reason: "independently materialized grouped parameters need invocation and reduction topology".into() });
        };
        let projection =
            |spec: &eredu_nn::GroupedProjectionSpec, input: i32, output: i32| ProjectionTopology {
                input: input as u64,
                output: output as u64,
                format: spec.format().encoding(),
                bias: spec.bias().is_some(),
                parameter: spec
                    .weight()
                    .alias_of
                    .as_ref()
                    .unwrap_or(&spec.weight().id)
                    .as_str()
                    .into(),
            };
        let doubled = bank
            .intermediate_dimensions()
            .checked_mul(2)
            .ok_or_else(|| Error::backend("grouped gate/up width overflow"))?;
        Ok(Self::Routed {
            experts: bank.group_count() as u64,
            selected: selector.selection().top_k() as u64,
            intermediate_size: bank.intermediate_dimensions() as u64,
            router: ProjectionTopology {
                input: selector.input_dimensions() as u64,
                output: selector.selection().group_count() as u64,
                format: selector.format().encoding(),
                bias: selector.bias().is_some(),
                parameter: selector
                    .weight()
                    .alias_of
                    .as_ref()
                    .unwrap_or(&selector.weight().id)
                    .as_str()
                    .into(),
            },
            projections: vec![
                projection(gate_up, bank.input_dimensions(), doubled),
                projection(
                    down,
                    bank.intermediate_dimensions(),
                    bank.output_dimensions(),
                ),
            ],
        })
    }
}

/// One sequential residual block using the reusable mechanisms above.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextLayerTopology {
    /// Ordered fusion projections applied before the residual block.
    #[serde(default)]
    pub input_projections: Vec<ProjectionTopology>,
    /// Stateful token mixing.
    pub mixer: TokenMixerTopology,
    /// Dense or routed feed-forward execution.
    pub feed_forward: FeedForwardTopology,
    /// Hidden-width normalization invocations (Q/K norms are described above).
    pub normalization_count: u64,
}

/// Ordinary construction topology for one target or prediction execution stack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextExecutionTopology {
    /// Residual stream width.
    pub hidden_size: u64,
    /// Output vocabulary width.
    pub vocabulary_size: u64,
    /// Ordered layer invocations; shared parameters keep their original identities.
    pub layers: Vec<TextLayerTopology>,
    /// Output projection, including the tied embedding owner when applicable.
    pub output: ProjectionTopology,
    /// Number of output projection invocations in this execution stack.
    #[serde(default = "one_output_invocation")]
    pub output_invocations: u64,
    /// Final vocabulary logits require a tanh cap.
    #[serde(default)]
    pub output_softcap: bool,
    /// Potential F32 parameter conversion payload from selected physical tasks.
    #[serde(default)]
    pub selected_parameter_promotion_bytes: Option<u64>,
    /// Canonical selected task identities for the promotion payload above.
    /// Empty means attribution is unavailable, so observed conversions cannot credit it.
    #[serde(default)]
    pub selected_parameter_promotion_payloads: std::collections::BTreeMap<String, u64>,
    /// Architecture dtype-flow declarations. For each ordinary projection owner,
    /// all listed ungrouped learned RMS gains must produce F32 for its input to
    /// be F32. The equations between normalization and projection must preserve
    /// that dtype. List every source when an owner has multiple invocations.
    /// Absence means unknown; state storage width is not an activation contract.
    #[serde(default)]
    pub projection_input_normalizations: std::collections::BTreeMap<String, Vec<String>>,
    /// Backend observations: these bound gains make ungrouped learned RMS output
    /// F32 for every supported floating activation dtype, without evaluation.
    #[serde(default)]
    pub f32_rms_normalization_gains: std::collections::BTreeSet<String>,
    /// Current backend binding facts. Cold/older reports default to no coverage.
    #[serde(default)]
    pub projection_storage:
        std::collections::BTreeMap<String, crate::projection_memory::ProjectionStorageFacts>,
    /// Additional unsupported module equations or invocation facts.
    pub missing: Vec<String>,
}

fn one_output_invocation() -> u64 {
    1
}

// Checkpoint formats deliberately have no frontend serde dependency. Preserve the
// full encoding here using the GGML wire code rather than a lossy debug string.
mod linear_format {
    use super::*;
    use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, WeightQuantization};
    #[derive(Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Wire {
        Dense,
        Quantized {
            quantization: WeightQuantization,
        },
        E4M3BlockFp8 {
            rows: i32,
            columns: i32,
            exponent_scales: bool,
        },
    }
    pub fn serialize<S: serde::Serializer>(
        format: &LinearFormat,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let wire = match *format {
            LinearFormat::Dense => Wire::Dense,
            format @ (LinearFormat::Affine(_)
            | LinearFormat::MxFp4
            | LinearFormat::GgufIQuant { .. }) => Wire::Quantized {
                quantization: format
                    .weight_quantization()
                    .expect("matched quantized format"),
            },
            LinearFormat::E4M3BlockFp8(v) => Wire::E4M3BlockFp8 {
                rows: v.block_rows,
                columns: v.block_columns,
                exponent_scales: v.scale_encoding == BlockFp8ScaleEncoding::Ue8m0,
            },
        };
        wire.serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<LinearFormat, D::Error> {
        let format = match Wire::deserialize(deserializer)? {
            Wire::Dense => LinearFormat::Dense,
            Wire::Quantized { quantization } => quantization.into(),
            Wire::E4M3BlockFp8 {
                rows,
                columns,
                exponent_scales,
            } => LinearFormat::E4M3BlockFp8(BlockFp8Format {
                block_rows: rows,
                block_columns: columns,
                scale_encoding: if exponent_scales {
                    BlockFp8ScaleEncoding::Ue8m0
                } else {
                    BlockFp8ScaleEncoding::FloatingPoint
                },
            }),
        };
        format.validate().map_err(serde::de::Error::custom)?;
        Ok(format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::*;

    #[test]
    fn grouped_provider_keeps_custom_selector_modules_explicitly_unknown() {
        let parameter = |name| ParameterSpec::trainable(name).unwrap();
        let format = || LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap();
        let selector = TopKGroupSelectorSpec::new(
            8,
            parameter("router"),
            format(),
            TopKGroupSelectionSpec::new(4, 2, GroupScoring::Softmax, true).unwrap(),
        )
        .unwrap();
        let bank = GroupedGatedProductSpec::new(
            4,
            8,
            16,
            8,
            GatedProductPolicy::ordinary_silu(),
            GatedProductGroupLayout::Packed {
                gate_up: GroupedProjectionSpec::new(parameter("gate_up"), None, format()).unwrap(),
                down: GroupedProjectionSpec::new(parameter("down"), None, format()).unwrap(),
            },
        )
        .unwrap();
        assert!(matches!(
            FeedForwardTopology::from_grouped_specs(&selector, &bank).unwrap(),
            FeedForwardTopology::Routed { .. }
        ));
        let transformed = selector.clone().with_input_transform(
            SelectorInputTransformSpec::new(1e-5, parameter("scale"), true).unwrap(),
        );
        let scaled = selector.with_coefficient_scale(parameter("coefficient"));
        for selector in [transformed, scaled] {
            assert!(
                matches!(FeedForwardTopology::from_grouped_specs(&selector, &bank).unwrap(), FeedForwardTopology::Unknown { reason } if !reason.is_empty())
            );
        }
    }
}
