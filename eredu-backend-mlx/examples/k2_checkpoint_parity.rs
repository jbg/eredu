//! Explicit released-checkpoint validation of the shared K2 equations.
//! Usage: k2_checkpoint_parity ARTIFACT_DIRECTORY REFERENCE.safetensors

use eredu_architectures::{decoder, k2_horizon};
use eredu_backend_mlx::backend::runtime::checkpoint::{
    recipe::MlxWeightRecipeExt,
    store::{MlxParameterMaterializationContext, WeightMaterialization},
};
use eredu_backend_mlx::{
    backend::{nn::shared::MlxNeuralBackend, runtime::cache::state::MlxKeyValueLayerState},
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointSource, SafetensorsWeightStore, TensorSelection},
};
use eredu_nn::{
    EmbeddingOperator, GroupedGatedProductOperator, GroupedLinearOperator, GroupedNeuralBackend,
    GroupedRelu2Operator, LinearOperator, NeuralBackend, NormalizationOperator, ParameterMetadata,
    ParameterVisitorMut, Parameterized,
};
use eredu_runtime::{ExpertPass, RoutedExpertProvider, RoutedExpertRequest};
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Default)]
struct Capture {
    step: usize,
    output: Option<PathBuf>,
    values: BTreeMap<String, Array>,
}
impl Capture {
    fn routes(
        &mut self,
        request: &RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<(), eredu_nn::Error> {
        if self.output.is_none() {
            return Ok(());
        }
        let prefix = format!(
            "routes.{}.{}.{}",
            self.step,
            request.layer,
            request.bank.value()
        );
        for (field, value) in [
            ("input", request.input),
            ("ids", request.routes.group_indices()),
            ("scores", request.routes.selected_scores()),
            ("coefficients", request.routes.coefficients()),
        ] {
            let dtype = if field == "ids" {
                Dtype::Int32
            } else {
                Dtype::Float32
            };
            self.values.insert(
                format!("{prefix}.{field}"),
                value
                    .as_array()
                    .as_dtype(dtype, stream)
                    .map_err(eredu_nn::Error::backend)?,
            );
        }
        Ok(())
    }
}
impl RoutedExpertProvider<MlxNeuralBackend> for Capture {
    type Error = eredu_nn::Error;
    fn forward_grouped(
        &mut self,
        bank: &mut <MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        self.routes(&request, stream)?;
        let output = bank.forward_grouped(request.input, request.routes, stream)?;
        if self.output.is_some() {
            self.values.insert(
                format!(
                    "routes.{}.{}.{}.output",
                    self.step,
                    request.layer,
                    request.bank.value()
                ),
                output
                    .as_array()
                    .as_dtype(Dtype::Float32, stream)
                    .map_err(eredu_nn::Error::backend)?,
            );
        }
        Ok(output)
    }
    fn forward_linear_routed(
        &mut self,
        bank: &mut <MlxNeuralBackend as GroupedNeuralBackend>::LinearGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        self.routes(&request, stream)?;
        bank.forward_grouped(request.input, request.routes, stream)
    }
    fn forward_relu2_routed(
        &mut self,
        bank: &mut <MlxNeuralBackend as GroupedNeuralBackend>::Relu2Groups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        self.routes(&request, stream)?;
        bank.forward_grouped(request.input, request.routes, stream)
    }
}

struct Bind<'s> {
    source: &'s dyn CheckpointSource,
    recipes: &'s BTreeMap<String, DerivedWeightRecipe>,
    context: &'s MlxParameterMaterializationContext,
    stream: &'s Stream,
    fp32: bool,
    failure: Option<String>,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, tensor: &'a mut MlxTensor) {
        if self.failure.is_some() {
            return;
        }
        let name = metadata.id.as_str();
        let source_recipe = DerivedWeightRecipe::source(name, TensorSelection::Full);
        let recipe = self.recipes.get(name).unwrap_or(&source_recipe);
        let result = (|| -> anyhow::Result<Array> {
            let (output, sources) = recipe
                .prepare_materialization(self.source, self.context)?
                .into_parts();
            let value = WeightMaterialization::submit_retained(output, sources)?.synchronize()?;
            anyhow::ensure!(
                tensor.as_array().shape() == value.shape(),
                "{name}: shape mismatch {:?} != {:?}",
                tensor.as_array().shape(),
                value.shape()
            );
            Ok(if self.fp32 {
                value.as_dtype(Dtype::Float32, self.stream)?
            } else {
                value
            })
        })();
        match result {
            Ok(value) => *tensor = MlxTensor::from_array(value),
            Err(error) => self.failure = Some(format!("{name}: {error}")),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    anyhow::ensure!(
        arguments.len() >= 2,
        "usage: k2_checkpoint_parity ARTIFACT REFERENCE.safetensors"
    );
    let root = PathBuf::from(&arguments[0]);
    let inspection = eredu_architectures::configuration::inspect_artifact(&root)?;
    let (args, source): (_, Box<dyn CheckpointSource>) =
        if let Some(plan) = inspection.architecture_plan().gguf_plan() {
            let eredu_architectures::configuration::GgufModelConfig::K2Horizon(args) = plan.model()
            else {
                anyhow::bail!("validation artifact is not K2 Horizon");
            };
            (
                args.clone(),
                Box::new(eredu_checkpoint::gguf_store::open_prepared_gguf_source(
                    inspection.gguf_checkpoint().expect("admitted GGUF").clone(),
                    plan.checkpoint(),
                    plan.tensor_mapping(),
                    2,
                )?),
            )
        } else {
            let value = serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
            (
                k2_horizon::model_args_from_config_value(&value)?,
                Box::new(SafetensorsWeightStore::open_with_max_cached_shards(
                    &root, 2,
                )?),
            )
        };
    let device_type = match std::env::var("EREDU_VALIDATION_DEVICE").as_deref() {
        Ok("gpu") => DeviceType::Gpu,
        Ok("cpu") | Err(_) => DeviceType::Cpu,
        Ok(other) => anyhow::bail!("unknown validation device {other:?}; expected cpu or gpu"),
    };
    let stream = Stream::new_with_device(&Device::new(device_type, 0));
    let mut recipes =
        k2_horizon::linear_companion_recipes(source.as_ref(), &args).map_err(anyhow::Error::msg)?;
    for layer in 0..args.num_hidden_layers as usize {
        for (bank, count, enabled) in [
            (
                k2_horizon::ExpertBank::FeedForward,
                args.num_experts,
                args.is_sparse_layer(layer),
            ),
            (
                k2_horizon::ExpertBank::AttentionValue,
                args.mova_num_experts,
                args.is_mova_layer(layer),
            ),
        ] {
            if enabled {
                recipes.extend(
                    k2_horizon::expert_recipes(
                        source.as_ref(),
                        &args,
                        layer,
                        bank,
                        &(0..count as usize).collect::<Vec<_>>(),
                    )
                    .map_err(anyhow::Error::msg)?,
                );
            }
        }
    }
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let materialization = MlxParameterMaterializationContext::new(&weights_stream, &stream);
    let mut binder = Bind {
        source: source.as_ref(),
        recipes: &recipes,
        context: &materialization,
        stream: &stream,
        fp32: arguments.get(2).is_some_and(|s| s == "float32"),
        failure: None,
    };
    let reference = Array::load_safetensors(&arguments[1], &weights_stream)?;
    let mut static_modules = decoder::StaticModules::<MlxNeuralBackend>::new(&args, &stream)?;
    static_modules.visit_parameters_mut(&mut binder);
    anyhow::ensure!(binder.failure.is_none(), "{:?}", binder.failure);
    let layer_limit = std::env::var("EREDU_VALIDATION_LAYER_LIMIT")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(args.num_hidden_layers as usize);
    anyhow::ensure!(
        layer_limit > 0 && layer_limit <= args.num_hidden_layers as usize,
        "invalid diagnostic layer limit"
    );
    let teacher_layers = std::env::var_os("EREDU_VALIDATION_TEACHER_LAYERS").is_some();
    let layer_start = std::env::var("EREDU_VALIDATION_LAYER_START")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    anyhow::ensure!(
        layer_start < layer_limit && (layer_start == 0 || teacher_layers),
        "diagnostic layer start requires teacher inputs and a nonempty range"
    );
    let mut blocks = (layer_start..layer_limit)
        .map(|layer| {
            let mut block = k2_horizon::new_block::<MlxNeuralBackend>(&args, layer, &stream)?;
            block.visit_parameters_mut(&mut binder);
            if let Some(error) = &binder.failure {
                return Err(eredu_nn::Error::backend(error));
            }
            eprintln!("loaded layer {layer}");
            Ok::<_, eredu_nn::Error>(block)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut caches = (0..blocks.len())
        .map(|_| MlxKeyValueLayerState::Device(Default::default()))
        .collect::<Vec<_>>();
    let inputs = reference["input_ids"]
        .evaluated()?
        .as_slice::<i32>()
        .to_vec();
    let mut offset = 0;
    let mut results = Vec::new();
    let mut failures = Vec::new();
    let mut capture = Capture {
        output: std::env::var_os("EREDU_VALIDATION_CAPTURE").map(PathBuf::from),
        ..Capture::default()
    };
    let prefill_tokens = std::env::var("EREDU_VALIDATION_PREFILL_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    anyhow::ensure!(
        prefill_tokens > 0 && inputs.len() >= prefill_tokens,
        "reference must contain a nonempty prompt followed by cached inputs"
    );
    for (step, ids) in std::iter::once(&inputs[..prefill_tokens])
        .chain(inputs[prefill_tokens..].chunks(1))
        .enumerate()
    {
        capture.step = step;
        let ids = MlxTensor::from_array(Array::from_slice(ids, &[1, ids.len() as i32]));
        let sequence = ids.as_array().dim(1);
        let mut hidden = static_modules.embeddings.forward(&ids, &stream)?;
        let mask = MlxNeuralBackend::causal_mask(sequence, offset, None, &stream)?;
        for (layer, (block, cache)) in blocks.iter_mut().zip(&mut caches).enumerate() {
            let layer = layer_start + layer;
            if teacher_layers && layer > 0 {
                hidden = MlxTensor::from_array(
                    reference[&format!("layers.{step}.{}.output", layer - 1)]
                        .as_dtype(hidden.as_array().dtype(), &stream)?,
                );
            }
            if capture.output.is_some() {
                let normalized =
                    NormalizationOperator::forward(&mut block.input_norm, &hidden, &stream)?;
                capture.values.insert(
                    format!("layers.{step}.{layer}.input_norm"),
                    normalized.as_array().as_dtype(Dtype::Float32, &stream)?,
                );
            }
            hidden = block.forward_routed(
                layer,
                decoder::AttentionInput {
                    hidden: &hidden,
                    mask: Some(&mask),
                    cache: Some(cache),
                    allow_sliding_prefill: false,
                    rotary_position: None,
                },
                if step == 0 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                },
                &mut capture,
                &stream,
            )?;
            if capture.output.is_some() {
                capture.values.insert(
                    format!("layers.{step}.{layer}.output"),
                    hidden.as_array().as_dtype(Dtype::Float32, &stream)?,
                );
            }
            if step == 0 {
                eprintln!("layer {layer} output dtype {:?}", hidden.as_array().dtype());
            }
            if let Some(expected) = reference.get(&format!("layers.{step}.{layer}.output")) {
                let actual = hidden
                    .as_array()
                    .as_dtype(Dtype::Float32, &stream)?
                    .evaluated()?
                    .as_slice::<f32>()
                    .to_vec();
                let expected = expected.evaluated()?.as_slice::<f32>().to_vec();
                let aa = expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
                let diff = actual
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| f64::from(a - b).powi(2))
                    .sum::<f64>();
                eprintln!(
                    "step {step}, layer {layer}: relative L2 {}",
                    (diff / aa).sqrt()
                );
            }
        }
        if layer_limit != args.num_hidden_layers as usize || teacher_layers {
            if let Some(path) = &capture.output {
                Array::save_safetensors(&capture.values, None, path)?;
            }
            anyhow::bail!("diagnostic layer capture completed; full-logit validation was not run");
        }
        hidden = NormalizationOperator::forward(&mut static_modules.norm, &hidden, &stream)?;
        let logits = match &mut static_modules.lm_head {
            Some(head) => head.forward(&hidden, &stream)?,
            None => static_modules.embeddings.as_linear(&hidden, &stream)?,
        };
        let actual = logits
            .as_array()
            .as_dtype(Dtype::Float32, &stream)?
            .evaluated()?
            .as_slice::<f32>()
            .to_vec();
        let expected = reference[&format!("logits.{step}")]
            .evaluated()?
            .as_slice::<f32>()
            .to_vec();
        anyhow::ensure!(actual.len() == expected.len(), "logit shape mismatch");
        let dot = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>();
        let aa = actual.iter().map(|&a| f64::from(a).powi(2)).sum::<f64>();
        let bb = expected.iter().map(|&b| f64::from(b).powi(2)).sum::<f64>();
        let delta = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>();
        let relative_l2 = (delta / bb).sqrt();
        let cosine = dot / (aa * bb).sqrt();
        if let Some(path) = &capture.output {
            Array::save_safetensors(&capture.values, None, path)?;
        }
        eprintln!("step {step}: relative L2 {relative_l2}, cosine {cosine}");
        if !(relative_l2 <= 0.02 && cosine >= 0.999) {
            failures.push(format!(
                "step {step}: relative L2 {relative_l2}, cosine {cosine}"
            ));
        }
        let top = |row: &[f32]| {
            let mut ids = (0..row.len()).collect::<Vec<_>>();
            ids.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            ids.truncate(10);
            ids
        };
        let mut minimum_overlap = 10;
        let mut confident_argmax_mismatches = 0;
        for (a, b) in actual
            .chunks(args.vocab_size as usize)
            .zip(expected.chunks(args.vocab_size as usize))
        {
            let a_top = top(a);
            let b_top = top(b);
            let overlap = a_top.iter().filter(|id| b_top.contains(id)).count();
            minimum_overlap = minimum_overlap.min(overlap);
            if overlap < 8 {
                failures.push(format!("step {step}: top-10 overlap {overlap}"));
            }
            if b[b_top[0]] - b[b_top[1]] > 0.5 && a_top[0] != b_top[0] {
                confident_argmax_mismatches += 1;
                failures.push(format!("step {step}: confident argmax mismatch"));
            }
        }
        let width = args.vocab_size as usize;
        let actual_argmax = top(&actual[actual.len() - width..])[0];
        let reference_argmax = top(&expected[expected.len() - width..])[0];
        if std::env::var_os("EREDU_VALIDATION_GREEDY").is_some()
            && actual_argmax != reference_argmax
        {
            failures.push(format!(
                "step {step}: greedy argmax {actual_argmax} differs from {reference_argmax}"
            ));
        }
        results.push(serde_json::json!({"step":step,"relative_l2":relative_l2,"cosine":cosine,"minimum_top10_overlap":minimum_overlap,"confident_argmax_mismatches":confident_argmax_mismatches,"actual_argmax":actual_argmax,"reference_argmax":reference_argmax}));
        offset += sequence;
    }
    println!("{}", serde_json::to_string_pretty(&results)?);
    anyhow::ensure!(
        failures.is_empty(),
        "checkpoint parity failed: {}",
        failures.join("; ")
    );
    Ok(())
}
