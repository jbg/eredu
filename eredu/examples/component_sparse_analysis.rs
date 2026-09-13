//! Bounded public-API reconstruction for the released BF16 sparse fixture.
use super::component_sparse_scores::ScoreReadout;
use anyhow::{ensure, Context};
use eredu::api::LoadedModel;
use eredu_core::{capture::CaptureUsage, component::*, parameters::*, ArchitectureDescriptor};
use eredu_evaluation::component_attribution::signed_sum;
use serde_json::{json, Value};

pub(super) fn values(value: &Value) -> anyhow::Result<Vec<f32>> {
    let values: Vec<f32> = serde_json::from_value(value.clone())?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "non-finite component evidence"
    );
    Ok(values)
}

pub(super) fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}

/// Conservative cumulative work allowance for this finite reference experiment.
/// A projection charges retained packed sources again even when host output is
/// small. This sum describes repeated work, not simultaneously allocated memory.
pub fn retention_allowance(
    graph: &ArchitectureDescriptor,
    facts: &ParameterDiscovery,
    reference: &Value,
    token: u64,
    all_predictions: bool,
) -> anyhow::Result<u64> {
    let mut total = 0u64;
    for trial in reference["trials"]
        .as_object()
        .context("planned trials")?
        .values()
    {
        for step in trial
            .as_array()
            .context("planned predictions")?
            .iter()
            .take(if all_predictions { usize::MAX } else { 1 })
        {
            let token = if step["prediction"].as_u64() == Some(0) {
                token
            } else {
                0
            };
            for selected in reference["captured"].as_array().context("planned groups")? {
                let kind = selected["kind"].as_str().context("group kind")?;
                let layer = selected["layer"].as_u64().context("group layer")? as usize;
                let key = format!("{kind}:{layer}");
                let (parameter, selected_elements, routes) = if kind == "routed" {
                    let group = graph
                        .routed_components
                        .iter()
                        .find(|g| g.layer_index == layer)
                        .context("routed group")?;
                    let column = group.write_column(
                        &RoutedComponentId {
                            group: group.id.clone(),
                            expert: 0,
                            index: 0,
                        },
                        facts,
                    )?;
                    let routes = step["captures"][&key]
                        .as_array()
                        .context("reference selected routes")?
                        .iter()
                        .filter(|row| row["token"].as_u64() == Some(token))
                        .count() as u64;
                    (
                        column.parameter,
                        (group.output_width as u64)
                            .checked_mul(group.units_per_expert as u64)
                            .context("expert extent overflow")?,
                        routes,
                    )
                } else {
                    let group = super::dense_group(graph, kind, layer)?;
                    let parameter = facts
                        .parameters
                        .iter()
                        .find(|p| p.id == group.write_weight)
                        .context("write parameter")?;
                    (
                        parameter,
                        parameter
                            .shape
                            .iter()
                            .try_fold(1u64, |a, b| a.checked_mul(*b))
                            .context("matrix extent overflow")?,
                        1,
                    )
                };
                let source = parameter
                    .shape
                    .iter()
                    .try_fold(1u64, |a, b| a.checked_mul(*b))
                    .context("source extent overflow")?;
                // Two projections per route. Allow two source-sized workspaces and
                // eight selected F32 buffers, plus bounded output/metadata margin.
                let cost = source
                    .checked_mul(16)
                    .and_then(|v| {
                        selected_elements
                            .checked_mul(32)
                            .and_then(|s| v.checked_add(s))
                    })
                    .and_then(|v| v.checked_add(1 << 20))
                    .and_then(|v| v.checked_mul(2))
                    .and_then(|v| v.checked_mul(routes))
                    .context("projection work allowance overflow")?;
                total = total
                    .checked_add(cost)
                    .context("cumulative projection allowance overflow")?;
            }
        }
    }
    Ok(total)
}

/// Reconstructs the final prefill position. LFM2's released equation rounds each
/// expert projection and weighted result to BF16, then adds by expert identity.
/// The signed real-valued sum remains separate from these physical roundings.
pub fn reconstruct<B: ParameterBackend>(
    model: &mut LoadedModel<B>,
    graph: &ArchitectureDescriptor,
    selected: &[Value],
    step: &Value,
    token: u64,
    budget: CaptureUsage,
    targets: Option<[u32; 2]>,
) -> anyhow::Result<Value> {
    let facts = model.parameter_discovery()?;
    let score_readout = targets
        .map(|targets| ScoreReadout::new(model, graph, &facts, step, targets, budget))
        .transpose()?;
    let mut usage = facts.usage;
    let mut reports = serde_json::Map::new();
    for selection in selected {
        let kind = selection["kind"].as_str().context("component kind")?;
        let layer = selection["layer"].as_u64().context("component layer")? as usize;
        let key = format!("{kind}:{layer}");
        let mut routes = vec![];
        if kind == "routed" {
            let group = graph
                .routed_components
                .iter()
                .find(|g| g.layer_index == layer)
                .context("routed component declaration")?;
            ensure!(
                group.write_bias.is_none(),
                "fixture expects bias-free routed writes"
            );
            for row in step["captures"][&key]["effective"]
                .as_array()
                .context("effective routed units")?
            {
                if row["token"].as_u64() != Some(token) {
                    continue;
                }
                let expert = row["expert"].as_u64().context("expert identity")? as usize;
                let column = group.write_column(
                    &RoutedComponentId {
                        group: group.id.clone(),
                        expert,
                        index: 0,
                    },
                    &facts,
                )?;
                let mut region = column.region;
                *region.shape.last_mut().context("write unit axis")? =
                    group.units_per_expert as u64;
                let coefficient =
                    row["coefficient"].as_f64().context("routing coefficient")? as f32;
                routes.push((
                    expert,
                    column.parameter,
                    region,
                    coefficient,
                    values(&row["values"])?,
                ));
            }
            ensure!(!routes.is_empty(), "missing selected routes");
            routes.sort_by_key(|route| route.0);
            ensure!(
                routes.windows(2).all(|pair| pair[0].0 != pair[1].0),
                "duplicate selected expert"
            );
        } else {
            let group = super::dense_group(graph, kind, layer)?;
            ensure!(
                group.write_bias.is_none() && group.output_normalization.is_none(),
                "fixture expects bias-free unnormalized scalar writes"
            );
            let parameter = facts
                .parameters
                .iter()
                .find(|p| p.id == group.write_weight)
                .context("effective write parameter")?;
            let units = values(&step["captures"][&key]["effective"])?;
            ensure!(
                units.len() >= group.count && units.len() % group.count == 0,
                "unit geometry"
            );
            routes.push((
                0,
                parameter,
                ParameterRegion {
                    starts: vec![0; parameter.shape.len()],
                    shape: parameter.shape.clone(),
                },
                1.,
                units[units.len() - group.count..].to_vec(),
            ));
        }
        let width = routes[0].2.shape[routes[0].2.shape.len() - 2] as usize;
        let mut directions = vec![(0..width)
            .map(|i| (i as i32 % 7 - 3) as f32 / 8.)
            .collect::<Vec<_>>()];
        if let Some(readout) = &score_readout {
            directions.extend(readout.directions.iter().cloned());
        }
        ensure!(
            directions.iter().all(|direction| direction.len() == width),
            "score direction width"
        );
        let mut physical = vec![0f32; width];
        let mut projected_terms = vec![vec![]; directions.len()];
        let mut component_terms = vec![vec![]; directions.len()];
        for (expert, parameter, region, coefficient, units) in routes {
            ensure!(
                parameter.dtype == Some(eredu_core::intervention::InterventionDtype::Bfloat16)
                    && parameter.input_transform == ProjectionInputTransform::Identity,
                "reconstruction requires measured BF16 identity projection inputs"
            );
            let unit_axis = region.shape.len() - 1;
            ensure!(
                units.len() as u64 == region.shape[unit_axis],
                "selected expert unit geometry"
            );
            let write = model
                .project_parameter(
                    &facts.identity,
                    &parameter.id,
                    ParameterProjection {
                        region: region.clone(),
                        axis: unit_axis,
                        directions: 1,
                        coefficients: units.clone(),
                    },
                    budget,
                )
                .with_context(|| format!("{key} expert {expert} write projection"))?;
            let write = write.values;
            ensure!(
                write.len() == width && write.iter().all(|v| v.is_finite()),
                "projected write geometry"
            );
            let columns = model
                .project_parameter(
                    &facts.identity,
                    &parameter.id,
                    ParameterProjection {
                        region,
                        axis: unit_axis - 1,
                        directions: directions.len() as u64,
                        coefficients: directions.iter().flatten().copied().collect(),
                    },
                    budget,
                )
                .with_context(|| format!("{key} expert {expert} signed column projection"))?;
            usage = columns.usage;
            let columns = columns.values;
            ensure!(
                columns.len() == units.len() * directions.len()
                    && columns.iter().all(|v| v.is_finite()),
                "signed write-column geometry"
            );
            for (index, direction) in directions.iter().enumerate() {
                component_terms[index].extend(units.iter().enumerate().map(
                    |(unit_index, unit)| {
                        *unit as f64
                            * columns[unit_index * directions.len() + index] as f64
                            * coefficient as f64
                    },
                ));
                projected_terms[index].extend(write.iter().zip(direction).map(
                    |(value, direction)| *value as f64 * *direction as f64 * coefficient as f64,
                ));
            }
            for (total, value) in physical.iter_mut().zip(write) {
                *total = bf16(*total + bf16(bf16(value) * coefficient));
            }
        }
        let observed = values(&step["writes"][&key])?;
        ensure!(
            observed.len() >= width && observed.len() % width == 0,
            "observed write geometry"
        );
        let observed = &observed[observed.len() - width..];
        let errors: Vec<_> = physical
            .iter()
            .zip(observed)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .collect();
        let signed_components = signed_sum(component_terms[0].iter().copied());
        let signed_projected = signed_sum(projected_terms[0].iter().copied());
        let signed_observed = signed_sum(
            observed
                .iter()
                .zip(&directions[0])
                .map(|(v, d)| *v as f64 * *d as f64),
        );
        let score_directions = directions.iter().enumerate().skip(1).map(|(index, direction)| {
            let components = signed_sum(component_terms[index].iter().copied());
            let projected = signed_sum(projected_terms[index].iter().copied());
            json!({"signed_component_sum": components, "signed_projected_write": projected,
                "signed_observed_write": signed_sum(observed.iter().zip(direction).map(|(v,d)| *v as f64 * *d as f64)),
                "projection_associativity_error": (components - projected).abs()})
        }).collect::<Vec<_>>();
        reports.insert(key, json!({
            "prediction": step["prediction"], "token": token, "components": component_terms[0].len(),
            "positive_components": component_terms[0].iter().filter(|v| **v > 0.).count(),
            "negative_components": component_terms[0].iter().filter(|v| **v < 0.).count(),
            "physical_write_exact": physical == observed,
            "different_write_coordinates": errors.iter().filter(|v| **v != 0.).count(),
            "write_max_abs": errors.iter().copied().fold(0f64, f64::max),
            "write_rms": (signed_sum(errors.iter().map(|v| v*v)) / width as f64).sqrt(),
            "signed_component_sum": signed_components, "signed_projected_write": signed_projected,
            "projection_associativity_error": (signed_components - signed_projected).abs(),
            "signed_observed_write": signed_observed,
            "physical_rounding_shift": signed_observed - signed_components,
            "score_directions": score_directions,
        }));
    }
    let score = score_readout
        .map(|readout| readout.finish(graph, step, &reports))
        .transpose()?;
    let mut output = json!({"groups": reports, "parameter_usage": usage});
    if let Some(score) = score {
        output["score_reconstruction"] = score;
    }
    Ok(output)
}
