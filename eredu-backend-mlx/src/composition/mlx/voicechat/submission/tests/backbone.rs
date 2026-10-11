//! Compact nonzero Nemotron units for native coupled-execution coverage.
use crate::{backend::nn::shared::MlxNeuralBackend, MlxTensor};
use eredu_architectures::nemotron_h;
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use safemlx::{Device, DeviceType, Stream};
use serde_json::Value;

pub(super) fn args() -> nemotron_h::ModelArgs {
    nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":24, "hidden_size":4,
        "intermediate_size":8, "num_hidden_layers":3,
        "hybrid_override_pattern":"M*-", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":2, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":2, "ssm_state_size":3,
        "conv_kernel":3, "chunk_size":2, "num_nextn_predict_layers":0,
        "tie_word_embeddings":false, "residual_in_fp32":true,
        "layer_norm_epsilon":1e-5
    }))
    .unwrap()
}

struct BindStatic<'a> {
    parameters: Value,
    stream: &'a Stream,
}

impl<'a> BindStatic<'a> {
    fn new(stream: &'a Stream) -> Self {
        let fixture: Value =
            serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
        Self {
            parameters: fixture["parameters"].clone(),
            stream,
        }
    }
}

impl<'a> ParameterVisitorMut<'a, MlxTensor> for BindStatic<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut MlxTensor) {
        let name = match metadata.id.as_str() {
            "model.embeddings.weight" => "stt_model.embed_tokens.weight",
            "model.norm_f.weight" => "stt_model.llm.norm_f.weight",
            "lm_head.weight" => "stt_model.lm_head.weight",
            "stt_model.function_head.weight" => "stt_model.function_head.weight",
            name => panic!("unexpected static parameter {name}"),
        };
        let value = &self.parameters[name];
        let shape: Vec<_> = value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|axis| axis.as_i64().unwrap() as i32)
            .collect();
        let values: Vec<_> = value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(slot.shape(), shape, "{name}");
        *slot = MlxTensor::from_f32_slice(&values, &shape, self.stream).unwrap();
    }
}

pub(super) fn model(stream: &Stream) -> nemotron_h::LayeredModel<MlxNeuralBackend> {
    let mut model = nemotron_h::LayeredModel::new(args(), stream).unwrap();
    model
        .static_modules_mut()
        .visit_parameters_mut(&mut BindStatic::new(stream));
    model
}

/// Select the native parameter provider over the independent pinned function head.
pub(super) fn function(
    bounded: bool,
    stream: &Stream,
) -> crate::composition::mlx::voicechat::MlxFunctionHead {
    use eredu_architectures::{
        artifact_preparation::PhysicalArtifactId,
        voicechat::{
            artifact::ParameterAdmission, checkpoint::Component,
            function_head_layered::FunctionHeadDimensions,
            function_head_preparation::PreparedFunctionHeadBindings,
        },
    };
    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe,
        store::{MemoryWeightStore, SharedCheckpointSource, TensorSelection},
    };
    use eredu_core::residency::OffloadConfig;
    use eredu_runtime::{LayerWeightResidency, LayerwiseLoadOptions};
    use std::{collections::BTreeMap, sync::Arc};

    let name = "stt_model.function_head.weight";
    let binding = BindStatic::new(stream);
    let bytes: Vec<_> = binding.parameters[name]["values"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|value| (value.as_f64().unwrap() as f32).to_le_bytes())
        .collect();
    let source: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([(
            name.into(),
            safetensors::Dtype::F32,
            vec![24, 4],
            bytes,
        )])
        .unwrap(),
    );
    let recipe = DerivedWeightRecipe::Source {
        key: name.into(),
        selection: TensorSelection::Full,
    };
    let output = recipe.infer(source.as_ref()).unwrap();
    let stored_bytes = output.byte_len();
    let parameters = BTreeMap::from([(
        name.into(),
        ParameterAdmission {
            linear_format: eredu_checkpoint::LinearFormat::Dense,
            name: name.into(),
            component: Component::FunctionHead,
            artifact: PhysicalArtifactId::Primary,
            output,
            stored_bytes,
            conversion_workspace_bytes: 0,
            materialization_peak_bytes: recipe.peak_materialization_bytes(source.as_ref()).unwrap()
                + stored_bytes,
            recipe,
        },
    )]);
    let prepared = PreparedFunctionHeadBindings::from_parameters(
        FunctionHeadDimensions::new(4, 24).unwrap(),
        source,
        &parameters,
    )
    .unwrap();
    let residency = if bounded {
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(stored_bytes), Some(1 << 20), 1).unwrap(),
        ))
    } else {
        LayerWeightResidency::FullyResident
    };
    let source_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    crate::composition::mlx::voicechat::materialize_voicechat_function_head(
        prepared,
        residency,
        &source_stream,
        stream,
    )
    .unwrap()
    .0
}

// Match the scalar reference harness's deterministic projection and learned
// normalization initialization. Direct Mamba parameters are explicitly nonzero
// here: its unloaded zero defaults would otherwise suppress recurrent coverage.
fn values(name: &str, length: usize, norm: bool) -> Vec<f32> {
    let mut hash = 2_166_136_261_u32;
    for byte in name.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    (0..length)
        .map(|index| {
            let mixed = hash
                .wrapping_add((index as u32).wrapping_mul(747_796_405))
                .rotate_left((index % 23) as u32);
            let centered = (mixed % 2001) as f32 - 1000.0;
            if norm {
                1.0 + centered / 50_000.0
            } else {
                centered / 25_000.0
            }
        })
        .collect()
}

/// Bind immediately after `ResidentRuntime::new`, which constructs the units.
pub(super) fn bind_units(units: &mut [Vec<nemotron_h::Unit<MlxNeuralBackend>>], stream: &Stream) {
    struct Bind<'s>(&'s Stream);
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut MlxTensor) {
            let name = metadata.id.as_str();
            let shape = slot.shape().to_vec();
            let count = shape.iter().map(|&axis| axis as usize).product();
            let values = if name.ends_with(".mamba.dt_bias") {
                vec![-1.; count]
            } else if name.ends_with(".mamba.A_log") {
                vec![-0.3; count]
            } else if name.ends_with(".mamba.D") {
                vec![1.; count]
            } else {
                values(name, count, name.ends_with(".norm.weight"))
            };
            *slot = MlxTensor::from_f32_slice(&values, &shape, self.0).unwrap();
        }
    }
    assert_eq!(units.iter().map(Vec::len).sum::<usize>(), 3);
    for unit in units.iter_mut().flatten() {
        unit.visit_parameters_mut(&mut Bind(stream));
    }
}
