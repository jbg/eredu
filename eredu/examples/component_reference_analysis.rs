//! Consumer-side signed attribution. No native tensors or checkpoint-name parsing.
use anyhow::{ensure, Context};
use eredu::api::LoadedModel;
use eredu_core::{capture::CaptureUsage, component::*, parameters::*, ArchitectureDescriptor};
use eredu_evaluation::component_attribution::{signed_sum, MeasuredReadout, ReadoutNormalization};
use std::collections::BTreeMap;

pub fn parameter_limits() -> CaptureUsage {
    CaptureUsage {
        captures: 2048,
        retained_bytes: 32 << 30,
        host_bytes: 128 << 20,
        encoded_bytes: 128 << 20,
    }
}

fn values<'a>(evidence: &'a BTreeMap<String, Vec<f32>>, path: &str) -> anyhow::Result<&'a [f32]> {
    evidence
        .get(path)
        .map(Vec::as_slice)
        .with_context(|| format!("missing effective evidence: {path}"))
}

fn contained_by_whole_write(
    architecture: &ArchitectureDescriptor,
    readout: &ComponentReadout,
    node: &str,
) -> bool {
    let mut current = Some(node);
    for _ in 0..=architecture.nodes.len() {
        let Some(node) = current else {
            return false;
        };
        if readout
            .other_writes
            .iter()
            .any(|write| write.node_id == node)
        {
            return true;
        }
        current = architecture
            .node(node)
            .and_then(|node| node.parent.as_deref());
    }
    false
}

fn query<B: ParameterBackend>(
    model: &mut LoadedModel<B>,
    facts: &ParameterDiscovery,
    parameter: &str,
    row: Option<u64>,
    limits: CaptureUsage,
) -> anyhow::Result<Vec<f32>> {
    let descriptor = facts
        .parameters
        .iter()
        .find(|p| p.id == parameter)
        .context("loaded parameter")?;
    let mut region = ParameterRegion {
        starts: vec![0; descriptor.shape.len()],
        shape: descriptor.shape.clone(),
    };
    if let Some(row) = row {
        region.starts[0] = row;
        region.shape[0] = 1;
    }
    Ok(model
        .query_parameter(&facts.identity, parameter, region, limits)?
        .values)
}

fn measured<B: ParameterBackend>(
    model: &mut LoadedModel<B>,
    facts: &ParameterDiscovery,
    residual: &[f32],
    direction: &[f64],
    bias: f64,
    norm: &ComponentNormalization,
    limits: CaptureUsage,
) -> anyhow::Result<MeasuredReadout> {
    let residual: Vec<_> = residual.iter().map(|x| *x as f64).collect();
    let gain = match &norm.gain {
        Some(id) => query(model, facts, id, None, limits)?
            .into_iter()
            .map(|x| x as f64 + norm.gain_offset.value() as f64)
            .collect(),
        None => vec![1.0 + norm.gain_offset.value() as f64; residual.len()],
    };
    let offset: Option<Vec<f64>> = norm
        .bias
        .as_ref()
        .map(|id| {
            query(model, facts, id, None, limits).map(|v| v.into_iter().map(f64::from).collect())
        })
        .transpose()?;
    Ok(MeasuredReadout::new(
        &residual,
        direction,
        bias,
        ReadoutNormalization {
            kind: norm.kind,
            epsilon: norm.epsilon.value() as f64,
            gain: &gain,
            bias: offset.as_deref(),
            groups: norm.groups,
        },
    )?)
}

pub fn reconstruct<B: ParameterBackend>(
    model: &mut LoadedModel<B>,
    architecture: &ArchitectureDescriptor,
    evidence: &BTreeMap<String, Vec<f32>>,
    target: u32,
    alternative: u32,
) -> anyhow::Result<serde_json::Value> {
    let readout = architecture
        .component_readout
        .as_ref()
        .context("declared readout")?;
    ensure!(
        readout.block_normalizations.is_empty() && readout.stream_residual.is_none(),
        "this released-fixture consumer expects additive block residuals"
    );
    let limits = parameter_limits();
    let facts = model.parameter_discovery()?;
    if readout.projection_input.is_none() {
        ensure!(facts.parameters.iter().any(|p| p.id == readout.weight
            && p.input_transform == ProjectionInputTransform::Identity),
            "readout input arithmetic requires actual-input evidence or a verified identity transform");
    }
    let target_row = query(model, &facts, &readout.weight, Some(target as u64), limits)?;
    let alternative_row = query(
        model,
        &facts,
        &readout.weight,
        Some(alternative as u64),
        limits,
    )?;
    let output_bias = readout
        .bias
        .as_ref()
        .map(|id| query(model, &facts, id, None, limits))
        .transpose()?;
    let residual = values(evidence, &format!("{}.effective", readout.residual))?;
    let embedding = values(evidence, &format!("{}.effective", readout.embedding))?;
    let scores = values(evidence, &format!("{}.effective", readout.linear_scores))?;
    let mut results = vec![];
    for margin in [false, true] {
        let row: Vec<_> = target_row
            .iter()
            .zip(&alternative_row)
            .map(|(a, b)| *a as f64 - if margin { *b as f64 } else { 0.0 })
            .collect();
        let bias = output_bias.as_ref().map_or(0.0, |v| {
            v[target as usize] as f64
                - if margin {
                    v[alternative as usize] as f64
                } else {
                    0.0
                }
        });
        let final_readout = measured(
            model,
            &facts,
            residual,
            &row,
            bias,
            &readout.normalization,
            limits,
        )?;
        // The final projection may transform its normalized input (for example,
        // dynamic block FP8). Keep this measured whole-head correction separate
        // from the residual components and normalization offset.
        let projection_input_correction = match &readout.projection_input {
            Some(path) => {
                let actual_input = values(evidence, path)?;
                let normalized = values(evidence, &format!("{}.effective", readout.normalized))?;
                ensure!(
                    actual_input.len() == row.len() && normalized.len() == row.len(),
                    "readout projection-input geometry"
                );
                signed_sum(actual_input.iter().zip(normalized).zip(&row).map(
                    |((actual, supplied), direction)| {
                        (*actual as f64 - *supplied as f64) * direction
                    },
                ))
            }
            None => 0.0,
        };
        let mut terms = vec![
            final_readout.offset,
            projection_input_correction,
            signed_sum(
                embedding
                    .iter()
                    .zip(&final_readout.direction)
                    .map(|(x, d)| *x as f64 * d),
            ),
        ];
        for write in &readout.other_writes {
            let output = values(evidence, &write.effective_output)?;
            ensure!(
                output.len() == final_readout.direction.len(),
                "whole residual-write geometry"
            );
            terms.push(
                write.residual_scale.value() as f64
                    * signed_sum(
                        output
                            .iter()
                            .zip(&final_readout.direction)
                            .map(|(value, direction)| *value as f64 * direction),
                    ),
            );
        }
        let mut components = 0;
        let mut nested_terms = Vec::new();
        let mut largest = (String::new(), 0usize, 0.0f64);
        for group in &architecture.components {
            let nested = contained_by_whole_write(architecture, readout, &group.node_id);
            let direction = match &group.output_normalization {
                Some(norm) => measured(
                    model,
                    &facts,
                    values(
                        evidence,
                        &format!("{}.write.effective", group.input.trim_end_matches(".input")),
                    )?,
                    &final_readout.direction,
                    0.0,
                    norm,
                    limits,
                )?,
                None => MeasuredReadout {
                    direction: final_readout.direction.clone(),
                    offset: 0.0,
                },
            };
            let weight = facts
                .parameters
                .iter()
                .find(|p| p.id == group.write_weight)
                .context("write parameter")?;
            ensure!(group.write_input.is_some()
                || weight.input_transform == ProjectionInputTransform::Identity,
                "component input arithmetic requires actual-input evidence or a verified identity transform");
            let projected = model.project_parameter(
                &facts.identity,
                &weight.id,
                ParameterProjection {
                    region: ParameterRegion {
                        starts: vec![0, 0],
                        shape: weight.shape.clone(),
                    },
                    axis: 0,
                    directions: 1,
                    coefficients: direction.direction.iter().map(|x| *x as f32).collect(),
                },
                limits,
            )?;
            let activations = values(
                evidence,
                group
                    .write_input
                    .as_deref()
                    .unwrap_or(&group.effective_activation),
            )?;
            ensure!(
                projected.values.len() == activations.len(),
                "component projection geometry"
            );
            let scale = group.residual_scale.value() as f64;
            for (index, (activation, projection)) in
                activations.iter().zip(projected.values).enumerate()
            {
                let contribution = *activation as f64 * projection as f64 * scale;
                if nested {
                    nested_terms.push(contribution);
                } else {
                    terms.push(contribution);
                }
                components += 1;
                if contribution.abs() > largest.2.abs() {
                    largest = (group.id.clone(), index, contribution);
                }
            }
            if !nested {
                terms.push(direction.offset * scale);
                if let Some(bias) = &group.write_bias {
                    let bias = query(model, &facts, bias, None, limits)?;
                    terms.push(
                        scale
                            * signed_sum(
                                direction
                                    .direction
                                    .iter()
                                    .zip(bias)
                                    .map(|(d, b)| d * b as f64),
                            ),
                    );
                }
            }
        }
        let actual = scores[target as usize] as f64
            - if margin {
                scores[alternative as usize] as f64
            } else {
                0.0
            };
        let reconstructed = signed_sum(terms);
        let error = (actual - reconstructed).abs();
        ensure!(
            error <= 3e-4 + 3e-4 * actual.abs(),
            "signed reconstruction: {reconstructed} versus {actual} (error {error})"
        );
        results.push(
            serde_json::json!({"margin":margin,"score":actual,"reconstructed":reconstructed,
            "absolute_error":error,"components":components,"projection_input_correction":projection_input_correction,
            "undecomposed_writes":readout.other_writes.len(),"largest_absolute_component":largest,
            "nested_components":nested_terms.len(),"nested_component_sum":signed_sum(nested_terms)}),
        );
    }
    // A backward-analysis join: selected downstream read and an upstream write column,
    // then a bounded writer-to-reader projection. Calibration/graph selection remains here.
    let downstream = architecture
        .components
        .last()
        .context("downstream component group")?;
    let read = downstream.reads.first().context("read relation")?;
    let row = read.rows.row(1).context("component read mapping")?;
    let read_values = query(model, &facts, &read.weight, Some(row as u64), limits)?;
    let read_input_transform = &facts
        .parameters
        .iter()
        .find(|parameter| parameter.id == read.weight)
        .context("downstream read parameter")?
        .input_transform;
    let upstream = architecture
        .components
        .first()
        .context("upstream component group")?;
    let write = facts
        .parameters
        .iter()
        .find(|p| p.id == upstream.write_weight)
        .context("upstream write")?;
    let column = model.query_parameter(
        &facts.identity,
        &write.id,
        ParameterRegion {
            starts: vec![0, 1],
            shape: vec![write.shape[0], 1],
        },
        limits,
    )?;
    let backward = model.project_parameter(
        &facts.identity,
        &write.id,
        ParameterProjection {
            region: ParameterRegion {
                starts: vec![0, 1],
                shape: vec![write.shape[0], 1],
            },
            axis: 0,
            directions: 1,
            coefficients: read_values.clone(),
        },
        limits,
    )?;
    let expected = signed_sum(
        read_values
            .iter()
            .zip(&column.values)
            .map(|(r, w)| *r as f64 * *w as f64),
    );
    ensure!(
        (expected - backward.values[0] as f64).abs() <= 2e-6 + 2e-6 * expected.abs(),
        "backward projection"
    );
    Ok(
        serde_json::json!({"signed_reconstruction":results,"backward_raw_read_write_dot":backward.values[0],
        "backward_input_transform":read_input_transform,
        "backward_note":"Raw geometric affinity; selected input quantization, layer-input normalization and calibration remain consumer calculations"}),
    )
}
