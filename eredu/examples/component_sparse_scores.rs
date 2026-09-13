//! Measured additive score accounting for the released BF16 LFM2 consumer.
use anyhow::{ensure, Context};
use eredu::api::LoadedModel;
use eredu_core::{capture::CaptureUsage, component::*, parameters::*, ArchitectureDescriptor};
use eredu_evaluation::component_attribution::{signed_sum, MeasuredReadout, ReadoutNormalization};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

use super::component_sparse_analysis::{bf16, values};

fn dot(left: &[f64], right: &[f64]) -> f64 {
    signed_sum(left.iter().zip(right).map(|(a, b)| a * b))
}

fn output_step(value: f32) -> f64 {
    let bits = value.abs().to_bits() & 0xffff0000;
    f32::from_bits(bits + 0x10000) as f64 - f32::from_bits(bits) as f64
}

pub struct ScoreReadout {
    pub directions: Vec<Vec<f32>>,
    targets: [u32; 2],
    rows: Vec<Vec<f64>>,
    measured: Vec<MeasuredReadout>,
    residual: Vec<f32>,
    embedding: Vec<f32>,
    head: Vec<f32>,
}

impl ScoreReadout {
    pub fn new<B: ParameterBackend>(
        model: &mut LoadedModel<B>,
        graph: &ArchitectureDescriptor,
        facts: &ParameterDiscovery,
        step: &Value,
        targets: [u32; 2],
        budget: CaptureUsage,
    ) -> anyhow::Result<Self> {
        let readout = graph
            .component_readout
            .as_ref()
            .context("readout declaration")?;
        let norm = &readout.normalization;
        let head_parameter = facts
            .parameters
            .iter()
            .find(|parameter| parameter.id == readout.weight)
            .context("loaded readout parameter")?;
        ensure!(
            targets[0] != targets[1]
                && head_parameter.input_transform == ProjectionInputTransform::Identity
                && head_parameter.dtype
                    == Some(eredu_core::intervention::InterventionDtype::Bfloat16),
            "released score fixture requires distinct targets and a BF16 identity head"
        );
        ensure!(
            readout.block_normalizations.is_empty()
                && readout.stream_residual.is_none()
                && matches!(readout.output_transform, ComponentOutputTransform::Identity)
                && readout.bias.is_none()
                && norm.bias.is_none()
                && norm.kind == ComponentNormalizationKind::Rms
                && norm.groups == 1
                && norm.gain_offset.value() == 0.
                && readout.embedding_scale.value() == 1.,
            "released score fixture requires additive, bias-free RMS readout"
        );
        let mut query = |name: &str, row: Option<u32>| -> anyhow::Result<Vec<f32>> {
            let parameter = facts
                .parameters
                .iter()
                .find(|p| p.id == name)
                .context("effective score parameter")?;
            let mut region = ParameterRegion {
                starts: vec![0; parameter.shape.len()],
                shape: parameter.shape.clone(),
            };
            if let Some(row) = row {
                ensure!(
                    region.shape.len() == 2 && u64::from(row) < region.shape[0],
                    "readout row geometry"
                );
                region.starts[0] = row.into();
                region.shape[0] = 1;
            }
            Ok(model
                .query_parameter(&facts.identity, name, region, budget)?
                .values)
        };
        let target = query(&readout.weight, Some(targets[0]))?;
        let alternative = query(&readout.weight, Some(targets[1]))?;
        let gain = query(norm.gain.as_deref().context("declared readout gain")?, None)?;
        let residual = values(&step["readout"]["residual"])?;
        let embedding = values(&step["readout"]["embedding"])?;
        let head = values(&step["head_input"])?;
        ensure!(
            residual.len() == target.len()
                && alternative.len() == target.len()
                && gain.len() == target.len()
                && embedding.len() == target.len()
                && head.len() == target.len(),
            "readout evidence geometry"
        );
        let rows = vec![
            target.iter().map(|v| *v as f64).collect::<Vec<_>>(),
            target
                .iter()
                .zip(&alternative)
                .map(|(a, b)| *a as f64 - *b as f64)
                .collect(),
        ];
        let residual64: Vec<_> = residual.iter().map(|v| *v as f64).collect();
        let gain64: Vec<_> = gain.iter().map(|v| *v as f64).collect();
        let measured = rows
            .iter()
            .map(|row| {
                MeasuredReadout::new(
                    &residual64,
                    row,
                    0.,
                    ReadoutNormalization {
                        kind: norm.kind,
                        epsilon: norm.epsilon.value() as f64,
                        gain: &gain64,
                        bias: None,
                        groups: 1,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let directions = measured
            .iter()
            .map(|value| value.direction.iter().map(|v| *v as f32).collect())
            .collect();
        Ok(Self {
            directions,
            targets,
            rows,
            measured,
            residual,
            embedding,
            head,
        })
    }

    pub fn finish(
        &self,
        graph: &ArchitectureDescriptor,
        step: &Value,
        groups: &serde_json::Map<String, Value>,
    ) -> anyhow::Result<Value> {
        let mut order = BTreeMap::new();
        let mut other = BTreeSet::new();
        for group in &graph.components {
            ensure!(
                group.residual_scale.value() == 1.,
                "fixture scalar residual scale"
            );
            let attention = matches!(
                group.activation_equation,
                ComponentActivation::Attention { .. }
            );
            let key = format!(
                "{}:{}",
                if attention { "attention" } else { "dense" },
                group.layer_index
            );
            ensure!(
                order
                    .insert((group.layer_index, if attention { 0 } else { 1 }), key)
                    .is_none(),
                "duplicate scalar invocation"
            );
        }
        for group in &graph.routed_components {
            ensure!(
                group
                    .residual_scale
                    .is_some_and(|scale| scale.value() == 1.),
                "fixture routed residual scale"
            );
            ensure!(
                order
                    .insert(
                        (group.layer_index, 1),
                        format!("routed:{}", group.layer_index)
                    )
                    .is_none(),
                "duplicate routed invocation"
            );
        }
        for write in &graph
            .component_readout
            .as_ref()
            .context("readout")?
            .other_writes
        {
            if graph
                .routed_components
                .iter()
                .any(|g| super::contains_node(graph, &write.node_id, &g.node_id))
            {
                continue;
            }
            ensure!(
                write.residual_scale.value() == 1.,
                "fixture mixer residual scale"
            );
            let key = format!("convolution:{}", write.layer_index);
            other.insert(key.clone());
            ensure!(
                order.insert((write.layer_index, 0), key).is_none(),
                "duplicate mixer invocation"
            );
        }
        let width = self.residual.len();
        let mut state = self.embedding.clone();
        let mut residual_rounding = [vec![], vec![]];
        let mut other_terms = [vec![], vec![]];
        for key in order.values() {
            let write = values(&step["writes"][key])?;
            ensure!(
                write.len() >= width && write.len() % width == 0,
                "complete write geometry"
            );
            let write = &write[write.len() - width..];
            for (index, (current, value)) in state.iter_mut().zip(write).enumerate() {
                let next = bf16(*current + *value);
                let rounding = next as f64 - *current as f64 - *value as f64;
                for mode in 0..2 {
                    residual_rounding[mode].push(rounding * self.directions[mode][index] as f64);
                    if other.contains(key) {
                        other_terms[mode].push(*value as f64 * self.directions[mode][index] as f64);
                    }
                }
                *current = next;
            }
        }
        ensure!(
            state == self.residual,
            "captured writes do not reproduce the BF16 residual"
        );
        let logits = values(&step["logits"])?;
        ensure!(
            self.targets
                .iter()
                .all(|token| (*token as usize) < logits.len()),
            "score target outside captured vocabulary"
        );
        let mut scores = vec![];
        for mode in 0..2 {
            let direction: Vec<f64> = self.directions[mode].iter().map(|v| *v as f64).collect();
            let embedding = signed_sum(
                self.embedding
                    .iter()
                    .zip(&direction)
                    .map(|(v, d)| *v as f64 * d),
            );
            let mut terms = vec![];
            let mut operator_rounding = vec![];
            for group in groups.values() {
                let evidence = &group["score_directions"][mode];
                let component = evidence["signed_component_sum"]
                    .as_f64()
                    .context("score component sum")?;
                terms.push(component);
                operator_rounding.push(
                    evidence["signed_observed_write"]
                        .as_f64()
                        .context("score observed write")?
                        - component,
                );
            }
            let components = signed_sum(terms);
            let operator_rounding = signed_sum(operator_rounding);
            let other_sum = signed_sum(other_terms[mode].iter().copied());
            let residual_rounding = signed_sum(residual_rounding[mode].iter().copied());
            let residual: Vec<_> = self.residual.iter().map(|v| *v as f64).collect();
            let direction_rounding = signed_sum(
                residual
                    .iter()
                    .zip(self.measured[mode].direction.iter().zip(&direction))
                    .map(|(v, (exact, rounded))| v * (exact - rounded)),
            );
            let head: Vec<_> = self.head.iter().map(|v| *v as f64).collect();
            let affine_score = dot(&head, &self.rows[mode]);
            let normalization_rounding = affine_score
                - dot(&residual, &self.measured[mode].direction)
                - self.measured[mode].offset;
            let reconstructed = signed_sum([
                components,
                operator_rounding,
                embedding,
                other_sum,
                residual_rounding,
                direction_rounding,
                normalization_rounding,
                self.measured[mode].offset,
            ]);
            let target = logits[self.targets[0] as usize];
            let alternative = logits[self.targets[1] as usize];
            ensure!(
                bf16(target) == target && bf16(alternative) == alternative,
                "score output is not BF16 represented"
            );
            let score = target as f64 - if mode == 1 { alternative as f64 } else { 0. };
            let bound = output_step(target)
                + if mode == 1 {
                    output_step(alternative)
                } else {
                    0.
                };
            ensure!(
                (reconstructed - affine_score).abs() <= 1e-9 * (1. + affine_score.abs()),
                "additive score accounting differs from independent head dot"
            );
            ensure!(
                (score - reconstructed).abs() <= bound + 1e-9,
                "score exceeds BF16 output rounding bound"
            );
            scores.push(json!({"mode": if mode == 0 { "target" } else { "margin" },
                "target": self.targets[0], "alternative": self.targets[1], "components": components,
                "operator_rounding": operator_rounding, "embedding": embedding, "other_writes": other_sum,
                "residual_rounding": residual_rounding, "direction_rounding": direction_rounding,
                "normalization_rounding": normalization_rounding, "normalization_offset": self.measured[mode].offset,
                "reconstructed": reconstructed, "affine_score": affine_score, "score": score,
                "absolute_error": (score - reconstructed).abs(), "head_rounding_bound": bound}));
        }
        Ok(
            json!({"prediction": step["prediction"], "residual_replay_exact": true,
            "residual_writes": order.len(), "other_write_count": other.len(), "scores": scores}),
        )
    }
}
