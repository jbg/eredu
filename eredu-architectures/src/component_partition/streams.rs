//! Complete residual values are replicated, even when their sublayer writes
//! were assembled from tensor-parallel partials. Ownership comes from the
//! declared coefficient parameters and execution boundaries.
use super::*;
use eredu_core::{
    ObservationPosition, SymbolicDimension, TensorAxis,
    capture::CaptureError,
    component::{ComponentStreamBase, ComponentStreamCoefficients, ComponentStreamResidual},
};

pub(super) fn register(
    observations: &mut SourceMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    streams: &ComponentStreamResidual,
    owns: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
    base: (bool, ObservationHookSite),
    head: (bool, ObservationHookSite),
) -> Result<(), ComponentPartitionError> {
    worker(
        observations,
        descriptor,
        streams,
        owns,
        base,
        head,
        Destination(None),
    )
}
fn same_axes(actual: Option<&[TensorAxis]>, expected: &[(&str, SymbolicDimension)]) -> bool {
    actual.is_some_and(|axes| {
        axes.len() == expected.len()
            && axes
                .iter()
                .zip(expected)
                .all(|(actual, (name, dimension))| {
                    actual.name == *name && actual.dimension == *dimension
                })
    })
}
pub(super) fn worker(
    observations: &mut SourceMap<String, PartitionedObservation>,
    descriptor: &ArchitectureDescriptor,
    streams: &ComponentStreamResidual,
    owns: &impl Fn(&str) -> Result<bool, ComponentPartitionError>,
    base: (bool, ObservationHookSite),
    head: (bool, ObservationHookSite),
    allocation: Destination<'_>,
) -> Result<(), ComponentPartitionError> {
    allocation.controls::<(
        &mut SourceMap<String, PartitionedObservation>,
        &ArchitectureDescriptor,
        &ComponentStreamResidual,
        (bool, ObservationHookSite),
        (bool, ObservationHookSite),
        [(&str, SymbolicDimension); 14],
        &str,
        bool,
        usize,
    )>()?;
    allocation.controls_of(&owns)?;
    let invalid =
        || allocation.capture_invalid(format_args!("invalid stream-residual declaration"));
    if streams.streams == 0 {
        return Err(invalid().into());
    }
    let (base_path, broadcast) = match &streams.base {
        ComponentStreamBase::Broadcast { input } => (input, true),
        ComponentStreamBase::Streams { input } => (input, false),
    };
    let base_point = descriptor
        .observations
        .points
        .iter()
        .find(|p| p.path == *base_path)
        .ok_or_else(|| allocation.capture_missing(base_path))?;
    let hidden = match base_point.axes.as_deref().and_then(|axes| axes.last()) {
        Some(TensorAxis {
            name,
            dimension: SymbolicDimension::Known(width),
        }) if name == "hidden" && *width > 0 => *width,
        _ => return Err(invalid().into()),
    };
    hidden.checked_mul(streams.streams).ok_or_else(invalid)?;
    streams
        .streams
        .checked_mul(streams.streams)
        .ok_or_else(invalid)?;
    // These expected axes are finite borrowed declarations. Their former
    // owning strings/vectors served only equality checks, in every profile.
    let hidden_axes = [
        ("batch", SymbolicDimension::Batch),
        ("sequence", SymbolicDimension::Sequence),
        ("hidden", SymbolicDimension::Known(hidden)),
    ];
    let vector = [
        ("batch", SymbolicDimension::Batch),
        ("sequence", SymbolicDimension::Sequence),
        ("stream", SymbolicDimension::Known(streams.streams)),
    ];
    let residual = [
        ("batch", SymbolicDimension::Batch),
        ("sequence", SymbolicDimension::Sequence),
        ("stream", SymbolicDimension::Known(streams.streams)),
        ("hidden", SymbolicDimension::Known(hidden)),
    ];
    let matrix = [
        ("batch", SymbolicDimension::Batch),
        ("sequence", SymbolicDimension::Sequence),
        ("input_stream", SymbolicDimension::Known(streams.streams)),
        ("output_stream", SymbolicDimension::Known(streams.streams)),
    ];
    let coefficient_owner = |parameters: &ComponentStreamCoefficients| {
        let local = owns(&parameters.function)?;
        if owns(&parameters.base)? != local || owns(&parameters.scale)? != local {
            return Err(invalid().into());
        }
        Ok::<_, ComponentPartitionError>(local)
    };
    allocation.controls_of(&coefficient_owner)?;
    let mut add = |path: &str, axes: &[(&str, SymbolicDimension)], axis: &str, local, site| {
        // Effective paths are an explicit public observation convention. Check
        // their catalog timing and geometry before pairing them with originals.
        let point = descriptor
            .observations
            .points
            .iter()
            .find(|p| p.path == path)
            .ok_or_else(|| allocation.capture_missing(path))?;
        if path.ends_with(".effective") && point.position != ObservationPosition::AfterIntervention
        {
            return Err(invalid().into());
        }
        let original = if point.position == ObservationPosition::AfterIntervention {
            path.strip_suffix(".effective").ok_or_else(invalid)?
        } else {
            path
        };
        if !same_axes(point.axes.as_deref(), axes) {
            return Err(invalid().into());
        }
        let original_point = descriptor
            .observations
            .points
            .iter()
            .find(|p| p.path == original)
            .ok_or_else(|| allocation.capture_missing(original))?;
        if !same_axes(original_point.axes.as_deref(), axes)
            || (original != path
                && original_point.position != ObservationPosition::BeforeIntervention)
        {
            return Err(invalid().into());
        }
        if let Some(effective) = descriptor
            .observations
            .points
            .iter()
            .find(|p| p.path.strip_suffix(".effective") == Some(original))
        {
            if !same_axes(effective.axes.as_deref(), axes)
                || effective.position != ObservationPosition::AfterIntervention
            {
                return Err(invalid().into());
            }
        }
        observations::replicated(
            observations,
            descriptor,
            original,
            axis,
            local,
            site,
            allocation,
        )
    };
    allocation.controls_of(&add)?;
    add(
        base_path,
        if broadcast { &hidden_axes } else { &residual },
        "hidden",
        base.0,
        base.1,
    )?;
    for cycle in &streams.cycles {
        let local = coefficient_owner(&cycle.coefficients)?;
        for path in [&cycle.input, &cycle.output] {
            add(path, &residual, "hidden", local, ObservationHookSite::Unit)?;
        }
        for path in [&cycle.collapsed, &cycle.write] {
            add(
                path,
                &hidden_axes,
                "hidden",
                local,
                ObservationHookSite::Unit,
            )?;
        }
        for path in [&cycle.pre, &cycle.post] {
            add(path, &vector, "stream", local, ObservationHookSite::Unit)?;
        }
        add(
            &cycle.combination,
            &matrix,
            "input_stream",
            local,
            ObservationHookSite::Unit,
        )?;
    }
    if coefficient_owner(&streams.head.parameters)? != head.0 {
        return Err(invalid().into());
    }
    add(&streams.head.input, &residual, "hidden", head.0, head.1)?;
    add(
        &streams.head.coefficients,
        &vector,
        "stream",
        head.0,
        head.1,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ModelConfigurationResolver;

    fn descriptor() -> ArchitectureDescriptor {
        crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&serde_json::json!({
                "model_type": "deepseek_v4", "hidden_size": 8, "moe_intermediate_size": 8,
                "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 4, "qk_rope_head_dim": 2, "q_lora_rank": 4,
                "o_lora_rank": 2, "o_groups": 2, "vocab_size": 16,
                "max_position_embeddings": 64, "sliding_window": 4, "compress_ratios": [0,4],
                "index_n_heads": 2, "index_head_dim": 4, "index_topk": 1,
                "hc_mult": 2, "hc_sinkhorn_iters": 2, "n_routed_experts": 2,
                "n_shared_experts": 1, "num_experts_per_tok": 1, "num_hash_layers": 1
            }))
            .unwrap()
            .architecture_plan()
            .architecture_descriptor()
    }

    #[test]
    fn stream_source_refuses_each_destination_on_the_original_worker() {
        let descriptor = descriptor();
        let streams = descriptor
            .component_readout
            .as_ref()
            .unwrap()
            .stream_residual
            .as_ref()
            .unwrap();
        construction::tests::verify(|allocation| {
            let mut result = SourceMap::new();
            worker(
                &mut result,
                &descriptor,
                streams,
                &|_| Ok(true),
                (true, ObservationHookSite::Input),
                (true, ObservationHookSite::Readout),
                allocation,
            )?;
            Ok(result)
        });
    }

    #[test]
    fn stream_capture_follows_invocations_and_keeps_complete_writes_disjoint() {
        let descriptor = descriptor();
        let streams = descriptor
            .component_readout
            .as_ref()
            .unwrap()
            .stream_residual
            .as_ref()
            .unwrap();
        for selected_layer in [Some(0), Some(1), None] {
            let owned = streams
                .cycles
                .iter()
                .filter(|cycle| Some(cycle.layer_index) == selected_layer)
                .flat_map(|cycle| {
                    [
                        &cycle.coefficients.function,
                        &cycle.coefficients.base,
                        &cycle.coefficients.scale,
                    ]
                })
                .chain(
                    (selected_layer == Some(1))
                        .then_some([
                            &streams.head.parameters.function,
                            &streams.head.parameters.base,
                            &streams.head.parameters.scale,
                        ])
                        .into_iter()
                        .flatten(),
                )
                .collect::<Vec<_>>();
            let mut observations = SourceMap::new();
            register(
                &mut observations,
                &descriptor,
                streams,
                &|parameter| Ok(owned.iter().any(|p| p.as_str() == parameter)),
                (selected_layer == Some(0), ObservationHookSite::Input),
                (selected_layer == Some(1), ObservationHookSite::Readout),
            )
            .unwrap();
            for cycle in &streams.cycles {
                for path in [
                    &cycle.input,
                    &cycle.collapsed,
                    &cycle.write,
                    &cycle.output,
                    &cycle.pre,
                    &cycle.post,
                    &cycle.combination,
                ] {
                    let point = &observations[path];
                    assert_eq!(
                        point.coordinates().is_some(),
                        Some(cycle.layer_index) == selected_layer
                    );
                    assert_eq!(point.combination(), PartitionCaptureCombination::Disjoint);
                    assert_eq!(point.site(), ObservationHookSite::Unit);
                }
                let original = cycle.write.strip_suffix(".effective").unwrap();
                assert_eq!(observations[original], observations[&cycle.write]);
            }
            let base = match &streams.base {
                ComponentStreamBase::Broadcast { input } => input,
                _ => unreachable!(),
            };
            assert_eq!(observations[base].site(), ObservationHookSite::Input);
            assert_eq!(
                observations[base].coordinates().is_some(),
                selected_layer == Some(0)
            );
            assert_eq!(
                observations[&streams.head.input].site(),
                ObservationHookSite::Readout
            );
            assert_eq!(
                observations[&streams.head.coefficients]
                    .coordinates()
                    .is_some(),
                selected_layer == Some(1)
            );
        }
    }

    #[test]
    fn stream_capture_rejects_invalid_matrix_orientation_effective_geometry_and_ownership() {
        let original = descriptor();
        for failure in 0..4 {
            let mut descriptor = original.clone();
            let streams = original
                .component_readout
                .as_ref()
                .unwrap()
                .stream_residual
                .as_ref()
                .unwrap();
            if failure < 3 {
                let path = if failure == 0 {
                    &streams.cycles[0].combination
                } else {
                    &streams.head.input
                };
                let point = descriptor
                    .observations
                    .points
                    .iter_mut()
                    .find(|p| &p.path == path)
                    .unwrap();
                match failure {
                    0 => point.axes.as_mut().unwrap().swap(2, 3),
                    1 => point.axes.as_mut().unwrap()[2].dimension = SymbolicDimension::Known(3),
                    _ => point.position = ObservationPosition::BeforeIntervention,
                }
            }
            let result = register(
                &mut SourceMap::new(),
                &descriptor,
                streams,
                &|name| Ok(failure != 3 || name != streams.cycles[0].coefficients.base),
                (true, ObservationHookSite::Input),
                (true, ObservationHookSite::Readout),
            );
            assert!(result.is_err(), "malformed declaration {failure}");
        }
    }
}
