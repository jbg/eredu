//! Actual Flash-Next prediction seams, independent of component score equations.
use super::*;
use crate::qwen4_exp::{mtp::PredictionSpec, target::MixerSpec};
use eredu_core::{speculative::SpeculativeCaptureScope, SymbolicDimension};

pub(super) fn register(
    placement: &mut ComponentPartitionLayout,
    descriptor: &ArchitectureDescriptor,
    spec: &PredictionSpec,
    local: &LocalModelLayout,
    execution: &crate::speculative_execution::SpeculativeActivationExecution,
) -> Result<(), ComponentPartitionError> {
    if spec.units.len() != execution.depth {
        return Err(CaptureError::Invalid(
            "prediction capture depth differs from prepared modules".into(),
        )
        .into());
    }
    let hidden = spec.fusion.geometry.hidden_size() as usize;
    for unit in &spec.units {
        let scope = SpeculativeCaptureScope::Prediction { depth: unit.depth };
        execution.validate_scope(scope)?;
        let root = format!("mtp.layers.{}.prediction", unit.depth);
        let mut registrar = Registrar {
            placement,
            descriptor,
            local,
            scope,
        };
        // Fusion, gated residual transport and collapse are replicated over TP;
        // the complete prediction invocation is also replicated over PP and EP.
        for suffix in [
            "input",
            "embedding",
            "fusion",
            "capture",
            "residual",
            "normalized",
            "readout.projection_input",
        ] {
            registrar.replicated(&format!("{root}.{suffix}"), "hidden", hidden)?;
        }
        // The shared target vocabulary head is replicated by prediction
        // construction; it is not a member of the prediction-only weight set.
        // Its global vocabulary dimension is authored in the joined catalog.
        for suffix in ["readout.linear", "logits"] {
            let path = format!("{root}.{suffix}");
            if let Some(point) = descriptor.observations.get(&path) {
                let width = point
                    .axes
                    .iter()
                    .flatten()
                    .find_map(|axis| match (&*axis.name, &axis.dimension) {
                        ("vocabulary", SymbolicDimension::Known(width)) => Some(*width),
                        _ => None,
                    })
                    .ok_or_else(|| ComponentPartitionError::InvalidPlacement(path.clone()))?;
                registrar.replicated(&path, "vocabulary", width)?;
            }
        }
        let mixer = match &unit.mixer {
            MixerSpec::Indexed(indexed) => {
                for suffix in ["channels", "write_input"] {
                    registrar.matrix_axis(
                        &format!("{root}.attention.{suffix}"),
                        "attention_channel",
                        &indexed.projections[3],
                        1,
                    )?;
                }
                let selection = indexed.indexer.selection;
                registrar.replicated(
                    &format!("{root}.attention.selected_positions"),
                    "selected_position",
                    (selection.token_budget + selection.ratio - 1) as usize,
                )?;
                "attention"
            }
            MixerSpec::Recurrent(recurrent) => {
                let recurrent = &recurrent.mixer;
                for (suffix, axis, linear, dimension) in [
                    ("qkv.projected", "qkv_channel", &recurrent.input_qkv, 0),
                    ("qkv.convolved", "qkv_channel", &recurrent.input_qkv, 0),
                    ("update.projected", "value_head", &recurrent.input_beta, 0),
                    ("decay.projected", "value_head", &recurrent.input_decay, 0),
                    ("gate.projected", "value_channel", &recurrent.input_gate, 0),
                    ("channels", "value_channel", &recurrent.output, 1),
                    ("write_input", "value_channel", &recurrent.output, 1),
                ] {
                    registrar.matrix_axis(
                        &format!("{root}.mixer.{suffix}"),
                        axis,
                        linear,
                        dimension,
                    )?;
                }
                "mixer"
            }
        };
        for part in [mixer, "feed_forward"] {
            for suffix in ["input", "write", "output", "residual"] {
                registrar.replicated(&format!("{root}.{part}.{suffix}"), "hidden", hidden)?;
            }
        }
    }
    Ok(())
}

struct Registrar<'a> {
    placement: &'a mut ComponentPartitionLayout,
    descriptor: &'a ArchitectureDescriptor,
    local: &'a LocalModelLayout,
    scope: SpeculativeCaptureScope,
}
impl Registrar<'_> {
    fn replicated(
        &mut self,
        path: &str,
        axis: &str,
        width: usize,
    ) -> Result<(), ComponentPartitionError> {
        self.insert(path, axis, ComponentCoordinateMap::range(width, 0..width)?)
    }

    fn matrix_axis(
        &mut self,
        path: &str,
        axis: &str,
        linear: &eredu_nn::LinearSpec,
        dimension: usize,
    ) -> Result<(), ComponentPartitionError> {
        if self.descriptor.observations.get(path).is_none() {
            return Ok(());
        }
        let name = linear.weight.id.as_str();
        let tensor = self
            .local
            .tensor(name)
            .ok_or_else(|| ComponentPartitionError::MissingWeight(name.into()))?;
        let count = if dimension == 0 {
            linear.output
        } else {
            linear.input
        } as usize;
        let coordinates = derive_matrix_axis_coordinates(name, count, tensor, dimension)?;
        self.insert(path, axis, coordinates)
    }

    fn insert(
        &mut self,
        path: &str,
        axis: &str,
        coordinates: ComponentCoordinateMap,
    ) -> Result<(), ComponentPartitionError> {
        // An omitted catalog point grants no producer authority. This keeps a
        // partial family description useful without inventing internal hooks.
        let Some(point) = self.descriptor.observations.get(path) else {
            return Ok(());
        };
        let mut matching = point
            .axes
            .iter()
            .flatten()
            .filter(|value| value.name == axis);
        if matching.next().map(|value| &value.dimension)
            != Some(&SymbolicDimension::Known(coordinates.global_count()))
            || matching.next().is_some()
            || point.value_type != eredu_core::ObservationValueType::Tensor
            || crate::speculative_execution::speculative_capture_scope(
                self.descriptor,
                &point.node_id,
            )? != self.scope
        {
            return Err(ComponentPartitionError::InvalidPlacement(path.into()));
        }
        let placement = PartitionedObservation {
            axis: axis.into(),
            coordinates: Some(coordinates.clone()),
            exports: true,
            site: ObservationHookSite::Unit,
            combination: PartitionCaptureCombination::Disjoint,
        };
        insert_observation(&mut self.placement.observations, path, placement)?;
        // Effective seams share physical coordinates, but remain independently
        // validated catalog points with the same prediction invocation owner.
        if !path.ends_with(".effective") {
            let effective = format!("{path}.effective");
            if self.descriptor.observations.get(&effective).is_some() {
                self.insert(&effective, axis, coordinates)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{ModelConfigurationResolver, ParallelRankTopology, ParallelTopology};

    #[test]
    fn prediction_projection_preserves_fused_order_and_rejects_stale_catalogs() {
        let config = serde_json::json!({
            "model_type":"qwen3_5_text", "vocab_size":16, "hidden_size":8,
            "num_hidden_layers":2, "intermediate_size":12, "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":2, "max_position_embeddings":64,
            "linear_conv_kernel_dim":3, "linear_key_head_dim":2, "linear_value_head_dim":2,
            "linear_num_key_heads":2, "linear_num_value_heads":2,
            "layer_types":["linear_attention","full_attention"], "mtp_num_hidden_layers":2
        });
        let mut descriptor = crate::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor();
        let mut point = descriptor
            .observations
            .points
            .iter()
            .find(|point| {
                crate::speculative_execution::speculative_capture_scope(&descriptor, &point.node_id)
                    .unwrap()
                    == SpeculativeCaptureScope::Prediction { depth: 0 }
            })
            .unwrap()
            .clone();
        let path = "prediction.fused";
        point.path = path.into();
        point.value_type = eredu_core::ObservationValueType::Tensor;
        point.axes = Some(vec![eredu_core::TensorAxis {
            name: "qkv_channel".into(),
            dimension: SymbolicDimension::Known(12),
        }]);
        descriptor.observations.points.push(point.clone());
        point.path = format!("{path}.effective");
        descriptor.observations.points.push(point);
        let linear = eredu_nn::LinearSpec {
            input: 8,
            output: 12,
            weight: eredu_nn::ParameterSpec::trainable("fused.weight").unwrap(),
            bias: None,
            format: eredu_nn::LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)
                .unwrap(),
        };
        let mut local = LocalModelLayout::default();
        local.insert(
            "fused.weight".into(),
            LocalTensorLayout::new(
                "fused.weight",
                eredu_runtime::ParameterRole::ColumnProjection,
                vec![12, 8],
                vec![6, 8],
                TensorPlacement::Indices {
                    axis: 0,
                    indices: vec![1, 2, 5, 6, 9, 10],
                },
                None,
                None,
                false,
            ),
        );
        let empty = || ComponentPartitionLayout {
            topology: ParallelRankTopology::new(ParallelTopology::new(2, 2, 2, 1).unwrap(), 7)
                .unwrap(),
            groups: BTreeMap::new(),
            paths: BTreeMap::new(),
            observations: BTreeMap::new(),
            routed: BTreeMap::new(),
        };
        let register = |descriptor: &ArchitectureDescriptor, scope, local: &LocalModelLayout| {
            let mut placement = empty();
            Registrar {
                placement: &mut placement,
                descriptor,
                local,
                scope,
            }
            .matrix_axis(path, "qkv_channel", &linear, 0)?;
            Ok::<_, ComponentPartitionError>(placement)
        };
        let scope = SpeculativeCaptureScope::Prediction { depth: 0 };
        let placement = register(&descriptor, scope, &local).unwrap();
        for path in [path.to_owned(), format!("{path}.effective")] {
            let point = placement.observation(&path).unwrap();
            assert_eq!(point.site(), ObservationHookSite::Unit);
            let coordinates = point.coordinates().unwrap();
            assert_eq!(
                (0..coordinates.local_count())
                    .map(|i| coordinates.local_to_global(i).unwrap())
                    .collect::<Vec<_>>(),
                vec![1, 2, 5, 6, 9, 10]
            );
        }
        assert!(register(
            &descriptor,
            SpeculativeCaptureScope::Prediction { depth: 1 },
            &local
        )
        .is_err());
        assert!(register(&descriptor, scope, &LocalModelLayout::default()).is_err());
        descriptor
            .observations
            .points
            .iter_mut()
            .find(|point| point.path == path)
            .unwrap()
            .axes
            .as_mut()
            .unwrap()[0]
            .dimension = SymbolicDimension::Known(11);
        assert!(register(&descriptor, scope, &local).is_err());
    }
}
