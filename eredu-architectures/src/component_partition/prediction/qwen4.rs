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
