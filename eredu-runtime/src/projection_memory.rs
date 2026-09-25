//! Backend-supplied projection storage facts. These records carry no native
//! handles or device-selection policy; absent coverage never implies zero cost.
use eredu_core::checkpoint::TensorDtype;
use serde::{Deserialize, Serialize};

/// A row interval over which one selected mechanism avoids full-weight promotion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionRowStorage {
    /// First covered logical row count, inclusive.
    pub first: u64,
    /// Last covered logical row count, inclusive.
    pub last: u64,
    /// Backend mechanism identifier for diagnostic attribution.
    pub mechanism: String,
    /// Exact logical native partial-result payload per row (zero for unsplit GEMM).
    pub partial_bytes_per_row: u64,
}

/// Facts for the current immutable resident parameter binding. The backend must
/// establish dtype, final weight layout and device eligibility without execution.
/// They apply only to ordinary dense projection, never routed/packed operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionStorageFacts {
    /// Exact bound matrix input width.
    pub input: u64,
    /// Exact bound matrix output width.
    pub output: u64,
    /// Activation representation covered by these facts.
    pub activation: TensorDtype,
    /// Current narrow storage representation.
    pub weight: TensorDtype,
    /// Output element bytes; outputs are accounted separately by topology.
    pub output_scalar_bytes: u64,
    /// Layout-dependent activation copy upper payload per row; lower is zero.
    /// Forecasting does not know the eventual activation strides.
    pub activation_copy_bytes_per_row: u64,
    /// Disjoint coverage intervals. All other rows retain the previous fallback.
    pub rows: Vec<ProjectionRowStorage>,
}
impl ProjectionStorageFacts {
    /// Reject malformed coverage rather than silently treating it as a bound.
    pub fn validate(&self) -> Result<(), eredu_core::CapabilityError> {
        let mut previous = 0;
        for range in &self.rows {
            if self.input == 0
                || self.output == 0
                || self.output_scalar_bytes == 0
                || range.first <= previous
                || range.last < range.first
                || range.mechanism.is_empty()
            {
                return Err(eredu_core::CapabilityError::InvalidConfiguration {
                    field: "projection storage facts",
                    detail: "invalid or overlapping projection coverage".into(),
                });
            }
            previous = range.last;
        }
        Ok(())
    }

    // Verification and final chunks can use fewer rows than the phase maximum.
    // Native split-K storage is not monotone in rows: retain interior maxima.
    fn prefix(&self, rows: u64) -> Result<(bool, u64), eredu_core::CapabilityError> {
        let mut next = 1;
        let mut partials = 0;
        for range in &self.rows {
            if range.first > rows {
                break;
            }
            if range.first != next {
                return Ok((false, partials));
            }
            let end = range.last.min(rows);
            partials = partials.max(end.checked_mul(range.partial_bytes_per_row).ok_or(
                eredu_core::CapabilityError::ArithmeticOverflow {
                    operation: "projection partial-result envelope",
                },
            )?);
            if end == rows {
                return Ok((true, partials));
            }
            next = end + 1;
        }
        Ok((false, partials))
    }

    /// Exact matching is required; a fact for another shape or dtype gives no credit.
    pub fn for_invocation(
        &self,
        input: u64,
        output: u64,
        activation: &TensorDtype,
        rows: u64,
    ) -> Option<&ProjectionRowStorage> {
        if self.input != input || self.output != output || &self.activation != activation {
            return None;
        }
        self.rows
            .iter()
            .find(|range| range.first <= rows && rows <= range.last)
    }
}

/// Per-invocation refinement: only fully covered canonical owners get conversion
/// credit. Shared owners with even one uncovered invocation retain their allowance.
/// Returned scratch is a conservative sum of native partials and possible input
/// copies; output tensors remain with the existing topology producer.
pub(crate) fn refine(
    topology: &crate::execution_topology::TextExecutionTopology,
    rows: u64,
    output_rows: u64,
) -> Result<(u64, u64, Vec<String>), eredu_core::CapabilityError> {
    use crate::execution_topology::{FeedForwardTopology as F, TokenMixerTopology as T};
    use std::collections::BTreeMap;
    let invalid = || eredu_core::CapabilityError::ArithmeticOverflow {
        operation: "projection mechanism workspace",
    };
    for fact in topology.projection_storage.values() {
        fact.validate()?;
    }
    let mut covered = BTreeMap::new();
    let mut scratch = 0u64;
    let mut details = Vec::new();
    let activation = TensorDtype::F32;
    let mut visit = |p: &crate::execution_topology::ProjectionTopology,
                     m: u64,
                     ordinary: bool|
     -> Result<(), eredu_core::CapabilityError> {
        let fact = topology
            .projection_storage
            .get(&p.parameter)
            .filter(|_| ordinary && p.format == eredu_checkpoint::LinearFormat::Dense)
            .filter(|_| {
                topology
                    .projection_input_normalizations
                    .get(&p.parameter)
                    .is_some_and(|gains| {
                        !gains.is_empty()
                            && gains
                                .iter()
                                .all(|gain| topology.f32_rms_normalization_gains.contains(gain))
                    })
            });
        let range = fact.and_then(|f| f.for_invocation(p.input, p.output, &activation, m));
        let eligible = match fact {
            Some(f) if range.is_some() => f.prefix(m)?.0,
            _ => false,
        };
        covered
            .entry(p.parameter.clone())
            .and_modify(|prior| *prior &= eligible)
            .or_insert(eligible);
        if let (Some(f), Some(range)) = (fact, range) {
            let partials = m
                .checked_mul(range.partial_bytes_per_row)
                .ok_or_else(invalid)?;
            let copy = m
                .checked_mul(f.activation_copy_bytes_per_row)
                .ok_or_else(invalid)?;
            let partial_upper = f.prefix(m)?.1;
            scratch = scratch
                .checked_add(partial_upper)
                .and_then(|n| n.checked_add(copy))
                .ok_or_else(invalid)?;
            details.push(format!("{} rows={m}: {}; no full-weight promotion; output={} bytes separately counted, split-K partials={partials} bytes (shorter-row partial envelope {partial_upper}), activation-layout copy=0..{copy} bytes; payload excludes allocator rounding", p.parameter, range.mechanism, m.checked_mul(p.output).and_then(|n| n.checked_mul(f.output_scalar_bytes)).ok_or_else(invalid)?));
        }
        Ok(())
    };
    for layer in &topology.layers {
        for p in &layer.input_projections {
            visit(p, rows, true)?;
        }
        match &layer.mixer {
            T::Attention { projections, .. } | T::GatedConvolution { projections, .. } => {
                for p in projections {
                    visit(p, rows, true)?;
                }
            }
            T::Unknown { .. } => {}
        }
        match &layer.feed_forward {
            F::Gated { projections, .. } => {
                for p in projections {
                    visit(p, rows, true)?;
                }
            }
            F::Routed {
                router,
                projections,
                ..
            } => {
                visit(router, rows, true)?;
                for p in projections {
                    visit(p, rows, false)?;
                }
            }
            F::Unknown { .. } => {}
        }
    }
    visit(
        &topology.output,
        output_rows,
        topology.output_invocations == 1,
    )?;
    let credit = covered
        .into_iter()
        .filter(|(_, yes)| *yes)
        .try_fold(0u64, |sum, (id, _)| {
            // Exact attribution is mandatory. Aggregate-only or retained/credited
            // entries cannot be subtracted again. Full F32 matrix payload must match.
            let f = &topology.projection_storage[&id];
            let expected = f
                .input
                .checked_mul(f.output)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(invalid)?;
            let bytes = topology
                .selected_parameter_promotion_payloads
                .get(&id)
                .copied()
                .filter(|&n| n == expected)
                .unwrap_or(0);
            sum.checked_add(bytes).ok_or_else(invalid)
        })?;
    Ok((credit, scratch, details))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_topology::*;
    fn topology() -> TextExecutionTopology {
        let p = ProjectionTopology {
            input: 128,
            output: 128,
            format: eredu_checkpoint::LinearFormat::Dense,
            bias: true,
            parameter: "shared".into(),
        };
        TextExecutionTopology {
            hidden_size: 128,
            vocabulary_size: 128,
            layers: vec![TextLayerTopology {
                input_projections: vec![p.clone()],
                mixer: TokenMixerTopology::GatedConvolution {
                    channels: 128,
                    kernel: 3,
                    projections: vec![],
                },
                feed_forward: FeedForwardTopology::Gated {
                    intermediate_size: 128,
                    projections: vec![],
                },
                normalization_count: 2,
            }],
            output: p,
            output_invocations: 1,
            output_softcap: false,
            selected_parameter_promotion_bytes: Some(128 * 128 * 4),
            selected_parameter_promotion_payloads: [("shared".into(), 128 * 128 * 4)].into(),
            missing: vec![],
            projection_input_normalizations: [("shared".into(), vec!["norm".into()])].into(),
            f32_rms_normalization_gains: ["norm".into()].into(),
            projection_storage: [(
                "shared".into(),
                ProjectionStorageFacts {
                    input: 128,
                    output: 128,
                    activation: TensorDtype::F32,
                    weight: TensorDtype::Bf16,
                    output_scalar_bytes: 4,
                    activation_copy_bytes_per_row: 128 * 4,
                    rows: vec![
                        ProjectionRowStorage {
                            first: 1,
                            last: 1,
                            mechanism: "gemv".into(),
                            partial_bytes_per_row: 0,
                        },
                        ProjectionRowStorage {
                            first: 2,
                            last: 2000,
                            mechanism: "gemm".into(),
                            partial_bytes_per_row: 128 * 4 * 2,
                        },
                    ],
                },
            )]
            .into(),
        }
    }
    #[test]
    fn covered_invocations_deduplicate_promotion_but_sum_native_workspace() {
        let t = topology();
        let (credit, scratch, details) = refine(&t, 128, 128).unwrap();
        assert_eq!(credit, 128 * 128 * 4);
        assert_eq!(scratch, 2 * 128 * (128 * 4 * 2 + 128 * 4));
        assert_eq!(details.len(), 2);
        let (credit, scratch, _) = refine(&t, 1, 1).unwrap();
        assert_eq!(credit, 128 * 128 * 4);
        assert_eq!(scratch, 2 * 128 * 4);
    }
    #[test]
    fn partial_shared_coverage_wrong_dtype_shape_and_unattributed_cost_stay_conservative() {
        let mut t = topology();
        assert_eq!(refine(&t, 2001, 1).unwrap().0, 0);
        t.f32_rms_normalization_gains.clear();
        assert_eq!(refine(&t, 128, 128).unwrap().0, 0);
        t.f32_rms_normalization_gains.insert("norm".into());
        t.projection_input_normalizations
            .get_mut("shared")
            .unwrap()
            .push("unknown".into());
        assert_eq!(refine(&t, 128, 128).unwrap().0, 0);
        t.projection_input_normalizations
            .get_mut("shared")
            .unwrap()
            .pop();
        t.output.input += 1;
        assert_eq!(refine(&t, 128, 128).unwrap().0, 0);
        t.output.input -= 1;
        t.selected_parameter_promotion_payloads.clear();
        assert_eq!(refine(&t, 128, 128).unwrap().0, 0);
    }
    #[test]
    fn split_k_interior_peaks_and_coverage_gaps_are_conservative() {
        let mut t = topology();
        let fact = t.projection_storage.get_mut("shared").unwrap();
        fact.rows[1].last = 8;
        fact.rows[1].partial_bytes_per_row = 4096;
        fact.rows.push(ProjectionRowStorage {
            first: 9,
            last: 2000,
            mechanism: "gemm".into(),
            partial_bytes_per_row: 0,
        });
        let (credit, scratch, _) = refine(&t, 128, 128).unwrap();
        assert_eq!(credit, 128 * 128 * 4);
        assert_eq!(scratch, 2 * (8 * 4096 + 128 * 128 * 4));
        t.projection_storage.get_mut("shared").unwrap().rows[2].first = 10;
        assert_eq!(refine(&t, 128, 128).unwrap().0, 0);
        t.projection_storage.get_mut("shared").unwrap().rows[2].first = 8;
        assert!(refine(&t, 128, 128).is_err());
    }

    #[test]
    fn retained_conversion_credit_and_old_cold_documents_never_get_double_credit() {
        let mut t = topology();
        t.selected_parameter_promotion_payloads
            .insert("shared".into(), 0);
        t.selected_parameter_promotion_bytes = Some(0);
        assert_eq!(refine(&t, 128, 128).unwrap().0, 0);
        let mut value = serde_json::to_value(topology()).unwrap();
        value.as_object_mut().unwrap().remove("projection_storage");
        let cold: TextExecutionTopology = serde_json::from_value(value).unwrap();
        assert_eq!(refine(&cold, 128, 128).unwrap(), (0, 0, vec![]));
    }
}
