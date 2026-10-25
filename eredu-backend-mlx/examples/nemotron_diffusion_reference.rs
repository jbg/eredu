//! Released shared-backbone comparison against the pinned NVIDIA implementation.
use eredu_architectures::{
    decoder::{self, Config},
    nemotron_labs_diffusion::{
        self,
        model::{TextInput, TextKernel, TextModel},
        FamilyConfig,
    },
};
use eredu_backend_mlx::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::state::MlxKeyValueState,
            checkpoint::binding::{
                build_neutral_module_bindings_with_recipes, materialize_module_bindings,
                populate_module_from_arrays_excluding,
            },
            execution::generic::prepare_layerwise_policy_with_bindings,
        },
        ExecutionContext,
    },
    MlxTensor,
};
use eredu_checkpoint::{
    safetensors::SafetensorsMetadataCatalog,
    store::{
        open_prepared_safetensors_source, CheckpointSource, CompositeCheckpointSource,
        SharedCheckpointSource,
    },
    validation::resolve_safetensors_plan,
};
use eredu_core::decoding::AttentionInvocation;
use eredu_nn::{AttentionCache, Parameterized, Tensor};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, LayerRuntimeState, LayerWeightResidency, LayerwiseLoadOptions,
    LayerwiseRuntime,
};
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use std::{error::Error, path::PathBuf};

type Backend = MlxNeuralBackend;
type Model = TextModel<Backend>;
type State = MlxKeyValueState;

fn bindings<M: Parameterized<MlxTensor>>(
    module: &M,
    source: &dyn CheckpointSource,
) -> Result<Vec<eredu_runtime::WeightBinding>, eredu_backend_mlx::backend::error::Error> {
    let mut recipes = nemotron_labs_diffusion::source_recipes(source)
        .map_err(eredu_backend_mlx::backend::error::Error::Parallel)?;
    for key in source.source_keys() {
        let canonical = nemotron_labs_diffusion::canonical_parameter_name(&key);
        if canonical == key
            && matches!(
                source.source_metadata(&key)?.stored_dtype,
                eredu_checkpoint::StoredDtype::F32
                    | eredu_checkpoint::StoredDtype::F16
                    | eredu_checkpoint::StoredDtype::BF16
            )
        {
            recipes.entry(canonical).or_insert_with(|| {
                eredu_checkpoint::recipe::DerivedWeightRecipe::Source {
                    key,
                    selection: eredu_checkpoint::store::TensorSelection::Full,
                }
            });
        }
    }
    for recipe in recipes.values_mut() {
        if let eredu_checkpoint::recipe::DerivedWeightRecipe::Source { key, .. } = recipe {
            if matches!(
                source.source_metadata(key)?.stored_dtype,
                eredu_checkpoint::StoredDtype::F32
                    | eredu_checkpoint::StoredDtype::F16
                    | eredu_checkpoint::StoredDtype::BF16
            ) {
                *recipe = eredu_checkpoint::recipe::DerivedWeightRecipe::Cast {
                    input: Box::new(recipe.clone()),
                    dtype: eredu_checkpoint::recipe::RecipeDtype::F32,
                };
            }
        }
    }
    build_neutral_module_bindings_with_recipes(module, source, &mut recipes).map_err(Into::into)
}
fn load_vision<M: Parameterized<MlxTensor>>(
    module: &mut M,
    source: &dyn CheckpointSource,
    stream: &Stream,
) -> Result<(), Box<dyn Error>> {
    let bindings = bindings(module, source)?;
    let arrays = materialize_module_bindings(source, &bindings, stream, stream)?;
    populate_module_from_arrays_excluding(module, &arrays, |_| false)?;
    Ok(())
}

fn compare(
    label: &str,
    actual: &MlxTensor,
    expected: &Array,
    stream: &Stream,
) -> Result<(), Box<dyn Error>> {
    if actual.shape() != expected.shape() {
        return Err(format!("{label}: incompatible shapes").into());
    }
    let a = actual
        .as_array()
        .as_dtype(Dtype::Float32, stream)?
        .into_evaluated()?
        .try_to_vec::<f32>()?;
    let b = expected
        .as_dtype(Dtype::Float32, stream)?
        .into_evaluated()?
        .try_to_vec::<f32>()?;
    let maximum = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rms = (a
        .iter()
        .zip(&b)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / a.len() as f64)
        .sqrt();
    println!("{label}: max_abs={maximum:.8} rms={rms:.8}");
    let tolerance = if label == "vision.patches" {
        0.00002
    } else {
        0.002
    };
    let relative = if label.contains(".layer.") {
        0.00002
    } else {
        0.0
    };
    if a.is_empty()
        || a.iter().chain(&b).any(|x| !x.is_finite())
        || a.iter()
            .zip(&b)
            .any(|(a, b)| (a - b).abs() > tolerance + relative * b.abs())
    {
        return Err(
            format!("{label}: comparison exceeds FP32 atol={tolerance} rtol={relative}").into(),
        );
    }
    Ok(())
}

struct Observer<'a> {
    label: &'a str,
    reference: &'a std::collections::HashMap<String, Array>,
    stream: &'a Stream,
}
impl eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error> for Observer<'_> {
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), eredu_nn::Error> {
        if let Some(index) = path
            .strip_prefix("model.vision_tower.transformer.layers.")
            .and_then(|p| p.strip_suffix(".output"))
            .and_then(|p| p.parse::<usize>().ok())
        {
            let label = format!("vision.layer.{index}");
            return compare(&label, value, &self.reference[&label], self.stream)
                .map_err(eredu_nn::Error::backend);
        }
        let Some(index) = path
            .strip_prefix("model.layers.")
            .and_then(|p| p.strip_suffix(".output"))
            .and_then(|p| p.parse::<usize>().ok())
        else {
            return Ok(());
        };
        let label = format!("{}.layer.{index}", self.label);
        compare(&label, value, &self.reference[&label], self.stream)
            .map_err(eredu_nn::Error::backend)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(
        args.next()
            .ok_or("usage: nemotron_diffusion_reference CHECKPOINT REFERENCE")?,
    );
    let reference_path = PathBuf::from(args.next().ok_or("missing reference")?);
    let mut gguf_path = None;
    let mut vision_gguf = None;
    let mut draft_adapter = false;
    let mut vision_only = false;
    let mut vision_context = false;
    let mut device_type = DeviceType::Cpu;
    let mut residency = LayerWeightResidency::FullyResident;
    while let Some(option) = args.next() {
        match option.to_str() {
            Some("--gguf") => {
                gguf_path = Some(PathBuf::from(args.next().ok_or("missing GGUF artifact")?))
            }
            Some("--vision-gguf") => {
                vision_gguf = Some(PathBuf::from(
                    args.next().ok_or("missing vision GGUF companion")?,
                ))
            }
            Some("--draft-adapter") => draft_adapter = true,
            Some("--vision-only") => vision_only = true,
            Some("--vision-context") => vision_context = true,
            Some("--device") => {
                device_type = match args.next().and_then(|s| s.into_string().ok()).as_deref() {
                    Some("cpu") => DeviceType::Cpu,
                    Some("gpu") => DeviceType::Gpu,
                    _ => return Err("device requires cpu or gpu".into()),
                }
            }
            Some("--residency") => {
                residency = match args.next().and_then(|s| s.into_string().ok()).as_deref() {
                    Some("resident") => LayerWeightResidency::FullyResident,
                    Some("host") => {
                        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::default())
                    }
                    Some("disk") => LayerWeightResidency::DenseDiskStream(
                        DenseDiskStreamLoadOptions::new(12 << 30, 2 << 30, 2, 2)?,
                    ),
                    _ => return Err("residency requires resident, host or disk".into()),
                }
            }
            _ => return Err("unexpected validation argument".into()),
        }
    }
    let (config, mut source): (_, SharedCheckpointSource) = if let Some(gguf) = gguf_path {
        if matches!(residency, LayerWeightResidency::DenseDiskStream(_)) {
            // Encoded input and F32 layer conversion need more than 2 GiB
            // together for 14B. This budget includes the conversion workspace.
            residency = LayerWeightResidency::DenseDiskStream(DenseDiskStreamLoadOptions::new(
                12 << 30,
                4 << 30,
                2,
                2,
            )?);
        }
        let (checkpoint, plan) = nemotron_labs_diffusion::gguf::TextGgufPlan::open(gguf)?;
        println!(
            "verified GGUF {}@{}; execution precision=F32 decoded matrices",
            plan.provenance().repository(),
            plan.provenance().revision()
        );
        let config = plan.config().clone();
        let source = plan.into_source(checkpoint, 2)?;
        (config, std::sync::Arc::new(source))
    } else {
        let config = FamilyConfig::from_hf_json(&std::fs::read(path.join("config.json"))?)?;
        let plan =
            nemotron_labs_diffusion::safetensors_plan(&config).map_err(std::io::Error::other)?;
        let catalog = SafetensorsMetadataCatalog::discover(&path)?;
        let resolution = resolve_safetensors_plan(&catalog, &plan)
            .map_err(|issues| std::io::Error::other(format!("admission: {issues:?}")))?;
        let source: SharedCheckpointSource = open_prepared_safetensors_source(
            catalog.admitted_shards(),
            catalog.tensors().clone(),
            resolution,
            2,
        )?;
        (config, source)
    };
    if let Some(vision_path) = vision_gguf {
        let (checkpoint, plan) =
            nemotron_labs_diffusion::vision_gguf::VisionGgufPlan::open(vision_path, &config)?;
        let denied = source
            .source_keys()
            .into_iter()
            .filter(|key| {
                let canonical = nemotron_labs_diffusion::canonical_parameter_name(key);
                canonical.starts_with("model.vision_tower.")
                    || canonical.starts_with("model.multi_modal_projector.")
            })
            .collect();
        let text: SharedCheckpointSource = std::sync::Arc::new(
            eredu_checkpoint::store::RestrictedCheckpointSource::excluding(
                source,
                "Nemotron text source with explicit vision GGUF companion",
                denied,
            )?,
        );
        let vision: SharedCheckpointSource = std::sync::Arc::new(plan.into_source(checkpoint, 2)?);
        source = std::sync::Arc::new(CompositeCheckpointSource::new([text, vision])?);
        println!(
            "verified pinned Pixtral GGUF companion; execution precision=F32 normalized matrices"
        );
    }
    let adapter = if draft_adapter {
        let directory = path.join("linear_spec_lora");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
        let adapter = nemotron_labs_diffusion::adapter::DraftAdapterConfig::from_json(
            &std::fs::read(directory.join("adapter_config.json"))?,
            manifest["id"]
                .as_str()
                .ok_or("missing pinned base identity")?,
        )
        .map_err(std::io::Error::other)?;
        let catalog =
            SafetensorsMetadataCatalog::discover(directory.join("adapter_model.safetensors"))?;
        let plan = nemotron_labs_diffusion::adapter::safetensors_plan(&config, &adapter)
            .map_err(std::io::Error::other)?;
        let resolution = resolve_safetensors_plan(&catalog, &plan)
            .map_err(|issues| std::io::Error::other(format!("adapter admission: {issues:?}")))?;
        let adapter_source = open_prepared_safetensors_source(
            catalog.admitted_shards(),
            catalog.tensors().clone(),
            resolution,
            2,
        )?;
        source = std::sync::Arc::new(CompositeCheckpointSource::new([source, adapter_source])?);
        Some(adapter)
    } else {
        None
    };
    let context = ExecutionContext::new(Device::new(device_type, 0));
    let stream = context.stream();
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let reference = Array::load_safetensors(reference_path, &weights_stream)?;
    if vision_context {
        type Composite = nemotron_labs_diffusion::composite::Model<Backend>;
        let mut model =
            nemotron_labs_diffusion::composite::CompositeKernel::new(config.clone(), stream)?;
        let (policy, metadata) = prepare_layerwise_policy_with_bindings(
            source,
            &mut model,
            (),
            std::marker::PhantomData::<State>,
            residency,
            stream,
            &weights_stream,
            |key| key.ends_with(".rotary_emb.inv_freq"),
            |modules, source| bindings(modules, source),
            |_, _, _, unit, source, _| bindings(&unit, source),
        )?;
        println!("residency: {metadata:?}");
        if residency == LayerWeightResidency::FullyResident {
            let policy = policy.into_resident(&model, stream, std::marker::PhantomData::<State>)?;
            let mut runtime = LayerwiseRuntime::<Composite, Backend, State, _>::new(model, policy);
            validate_vlm_context(&mut runtime, &config, &reference, stream)?;
            report(&runtime.policy().residency_report()?);
        } else {
            let mut runtime = LayerwiseRuntime::<Composite, Backend, State, _>::new(model, policy);
            validate_vlm_context(&mut runtime, &config, &reference, stream)?;
            report(&runtime.policy().residency_report()?);
        }
        return Ok(());
    }
    if vision_only {
        use nemotron_labs_diffusion::{
            processing::ImageProcessor,
            vision::{ImageEmbedding, ImageInput, ImageLayer, ImageProjector},
        };
        let processor = ImageProcessor::new(&config, 1400)?;
        let mut grids = Vec::new();
        let mut pixels = Vec::new();
        for index in 0..2 {
            let size = reference[&format!("vision.rgb_size.{index}")]
                .clone()
                .into_evaluated()?
                .try_to_vec::<i32>()?;
            let rgb = reference[&format!("vision.rgb.{index}")]
                .clone()
                .into_evaluated()?
                .try_to_vec::<u8>()?;
            let prepared = processor.prepare(eredu_media::image::RgbImageView::packed(
                &rgb,
                size[1] as u32,
                size[0] as u32,
            )?)?;
            grids.push((prepared.grid.0 as i32, prepared.grid.1 as i32));
            pixels.extend(prepared.patches);
        }
        let count = grids.iter().map(|(h, w)| h * w).sum::<i32>();
        let patches = MlxTensor::from_f32_slice(&pixels, &[count, 3 * 14 * 14], stream)?;
        compare(
            "vision.patches",
            &patches,
            &reference["vision.patches"],
            stream,
        )?;
        let mut image = ImageEmbedding::<Backend>::new(&config, stream)?;
        load_vision(&mut image, source.as_ref(), stream)?;
        let (mut hidden, state) = image.begin(
            ImageInput {
                patches: &patches,
                grids: &grids,
            },
            stream,
        )?;
        compare(
            "vision.embedding",
            &hidden,
            &reference["vision.embedding"],
            stream,
        )?;
        for index in 0..config
            .vision
            .as_ref()
            .ok_or("vision reference requires a tower")?
            .num_hidden_layers as usize
        {
            let mut layer =
                ImageLayer::<Backend>::new(config.vision.as_ref().unwrap(), index, stream)?;
            load_vision(&mut layer, source.as_ref(), stream)?;
            hidden = layer.forward(&hidden, &state, stream)?;
            compare(
                &format!("vision.layer.{index}"),
                &hidden,
                &reference[&format!("vision.layer.{index}")],
                stream,
            )?;
        }
        let mut projector = ImageProjector::<Backend>::new(&config, stream)?;
        load_vision(&mut projector, source.as_ref(), stream)?;
        let projected = projector.finish(&hidden, &state, stream)?;
        compare(
            "vision.projected",
            &projected,
            &reference["vision.projected"],
            stream,
        )?;
        return Ok(());
    }
    let text_config = match adapter {
        Some(adapter) => config.clone().with_draft_adapter(adapter),
        None => config.clone(),
    };
    let text_definition = std::sync::Arc::new(nemotron_labs_diffusion::definition::text(
        &text_config, None,
    )?);
    let mut model = TextKernel::from_definition(text_config, text_definition, stream)?;
    let (policy, metadata) = prepare_layerwise_policy_with_bindings(
        source,
        &mut model,
        (),
        std::marker::PhantomData::<State>,
        residency,
        stream,
        &weights_stream,
        |key| key.ends_with(".rotary_emb.inv_freq"),
        |modules, source| bindings(modules, source),
        |_, _, _, unit, source, _| bindings(&unit, source),
    )?;
    println!("residency: {metadata:?}");
    if residency == LayerWeightResidency::FullyResident {
        let policy = policy.into_resident(&model, stream, std::marker::PhantomData::<State>)?;
        let mut runtime = LayerwiseRuntime::<Model, Backend, State, _>::new(model, policy);
        validate_text(&mut runtime, &config, draft_adapter, &reference, stream)?;
        report(&runtime.policy().residency_report()?);
    } else {
        let mut runtime = LayerwiseRuntime::<Model, Backend, State, _>::new(model, policy);
        validate_text(&mut runtime, &config, draft_adapter, &reference, stream)?;
        report(&runtime.policy().residency_report()?);
    }
    Ok(())
}

fn report(report: &eredu_runtime::ResidencyReport) {
    println!(
        "residency telemetry: planned={:?} current={:?} peak={:?} evictions={:?}",
        report.offload().planned_bytes(),
        report.offload().resident_bytes(),
        report.offload().peak_resident_bytes(),
        report.offload().evictions()
    );
}

fn validate_text<P>(
    runtime: &mut LayerwiseRuntime<Model, Backend, State, P>,
    config: &FamilyConfig,
    draft_adapter: bool,
    reference: &std::collections::HashMap<String, Array>,
    stream: &Stream,
) -> Result<(), Box<dyn Error>>
where
    P: eredu_runtime::LayerwisePolicy<
        Backend,
        nemotron_labs_diffusion::model::TextLayer<Backend>,
        Error = eredu_backend_mlx::backend::error::Error,
    >,
{
    let mut state = State::device(decoder::state_layout(config)?)?;
    let cases: [&[i32]; 4] = [&[1, 100, 200], &[17], &[31], &[421]];
    for (step, tokens) in cases.into_iter().enumerate() {
        let tokens = MlxTensor::from_array(Array::from_slice(tokens, &[1, tokens.len() as i32]));
        let mut invocations = vec![
            (
                format!("causal.{step}"),
                tokens,
                AttentionInvocation::CAUSAL,
            ),
            (
                format!("block.{step}"),
                MlxTensor::from_array(reference[&format!("block.{step}.tokens")].clone()),
                AttentionInvocation::DIFFUSION,
            ),
        ];
        if draft_adapter {
            invocations.push((
                format!("draft.{step}"),
                MlxTensor::from_array(reference[&format!("block.{step}.tokens")].clone()),
                AttentionInvocation::DRAFT,
            ));
        }
        for (label, tokens, invocation) in invocations {
            let before: Vec<_> = (0..config.num_hidden_layers() as usize)
                .map(|i| state.layer(i).map(|cache| cache.offset()))
                .collect::<Result<_, _>>()?;
            let result = runtime.forward_with_observer(
                TextInput {
                    tokens: &tokens,
                    mask: None,
                    invocation,
                    feedback: None,
                },
                &mut state,
                stream,
                &mut Observer {
                    label: &label,
                    reference,
                    stream,
                },
            )?;
            compare(
                &format!("{label}.logits"),
                &result,
                &reference[&format!("{label}.logits")],
                stream,
            )?;
            if invocation != AttentionInvocation::CAUSAL {
                let after: Vec<_> = (0..config.num_hidden_layers() as usize)
                    .map(|i| state.layer(i).map(|cache| cache.offset()))
                    .collect::<Result<_, _>>()?;
                assert_eq!(
                    before, after,
                    "transient refinement advances canonical cache"
                );
            }
        }
    }
    Ok(())
}

fn vlm_text_input(
    tokens: MlxTensor,
) -> Result<eredu_runtime::PreparedModelInput<MlxTensor>, Box<dyn Error>> {
    use eredu_runtime::{
        PreparedInputInspector, PreparedInputPart, PreparedInputPayload, PreparedModelInput,
    };
    let inspector = eredu_backend_mlx::backend::runtime::media::input::MlxTensorInputInspector;
    Ok(PreparedModelInput::new(
        vec![PreparedInputPart::new(
            eredu_core::InputModality::Text,
            PreparedInputPayload::TokenIds(tokens),
            [],
        )?],
        |tensor| inspector.identity(tensor),
    )?)
}
fn vlm_prompt(
    config: &FamilyConfig,
    reference: &std::collections::HashMap<String, Array>,
    stream: &Stream,
) -> Result<eredu_runtime::PreparedModelInput<MlxTensor>, Box<dyn Error>> {
    use eredu_core::{InputExtent, InputMetadataKey, InputModality};
    use eredu_runtime::{
        PreparedInputInspector, PreparedInputPart, PreparedInputPayload, PreparedModelInput,
    };
    let text = |ids: &[i32]| {
        PreparedInputPart::new(
            InputModality::Text,
            PreparedInputPayload::TokenIds(MlxTensor::from_array(Array::from_slice(
                ids,
                &[1, ids.len() as i32],
            ))),
            [],
        )
    };
    let processor = nemotron_labs_diffusion::processing::ImageProcessor::new(config, 1400)?;
    let mut parts = vec![text(&[1])?];
    for index in 0..2 {
        let size = reference[&format!("vision.rgb_size.{index}")]
            .clone()
            .into_evaluated()?
            .try_to_vec::<i32>()?;
        let rgb = reference[&format!("vision.rgb.{index}")]
            .clone()
            .into_evaluated()?
            .try_to_vec::<u8>()?;
        let prepared = processor.prepare(eredu_media::image::RgbImageView::packed(
            &rgb,
            size[1] as u32,
            size[0] as u32,
        )?)?;
        let (height, width) = prepared.grid;
        let patches = MlxTensor::from_f32_slice(
            &prepared.patches,
            &[
                (height * width) as i32,
                3 * config.vision.as_ref().unwrap().patch_size.pow(2),
            ],
            stream,
        )?;
        parts.push(PreparedInputPart::new_with_extents(
            InputModality::Image,
            PreparedInputPayload::Tensor(patches),
            [(
                InputMetadataKey::PatchGrid,
                MlxTensor::from_i32_slice(&[1, height as i32, width as i32], &[1, 3], stream)?,
            )],
            [InputExtent::PatchGrid {
                time: 1,
                height: height as usize,
                width: width as usize,
            }],
        )?);
        parts.push(if index == 0 {
            text(&[4, 5])?
        } else {
            text(&[7])?
        });
    }
    let inspector = eredu_backend_mlx::backend::runtime::media::input::MlxTensorInputInspector;
    Ok(PreparedModelInput::new(parts, |tensor| {
        inspector.identity(tensor)
    })?)
}
fn validate_vlm_context<P>(
    runtime: &mut LayerwiseRuntime<
        nemotron_labs_diffusion::composite::Model<Backend>,
        Backend,
        State,
        P,
    >,
    config: &FamilyConfig,
    reference: &std::collections::HashMap<String, Array>,
    stream: &Stream,
) -> Result<(), Box<dyn Error>>
where
    P: eredu_runtime::LayerwisePolicy<
        Backend,
        nemotron_labs_diffusion::composite::Unit<Backend>,
        Error = eredu_backend_mlx::backend::error::Error,
    >,
{
    use eredu_architectures::composite_execution::{CompositeArchitecture, PreparedCompositeInput};
    type Composite = nemotron_labs_diffusion::composite::Model<Backend>;
    let inspector = eredu_backend_mlx::backend::runtime::media::input::MlxTensorInputInspector;
    let mut state = State::device(decoder::state_layout(config)?)?;
    let cases: [&[i32]; 4] = [&[], &[17], &[31], &[421]];
    for (step, ids) in cases.into_iter().enumerate() {
        let causal = if step == 0 {
            vlm_prompt(config, reference, stream)?
        } else {
            vlm_text_input(MlxTensor::from_array(Array::from_slice(
                ids,
                &[1, ids.len() as i32],
            )))?
        };
        let admitted = nemotron_labs_diffusion::input::admit_input(config, &causal, &inspector)?;
        if step == 0 {
            let tokens=<Composite as CompositeArchitecture<Backend,State>>::prepared_prediction_token_ids(PreparedCompositeInput::new(&causal,&admitted).map_err(std::io::Error::other)?,stream)?;
            compare(
                "context.tokens",
                &tokens,
                &reference["context.tokens"],
                stream,
            )?;
        }
        let block = vlm_text_input(MlxTensor::from_array(
            reference[&format!("block.{step}.tokens")].clone(),
        ))?;
        let block_admitted =
            nemotron_labs_diffusion::input::admit_input(config, &block, &inspector)?;
        for (label, input, admission, invocation) in [
            (
                format!("causal.{step}"),
                &causal,
                &admitted,
                AttentionInvocation::CAUSAL,
            ),
            (
                format!("block.{step}"),
                &block,
                &block_admitted,
                AttentionInvocation::DIFFUSION,
            ),
        ] {
            let before: Vec<_> = (0..config.num_hidden_layers() as usize)
                .map(|i| state.layer(i).map(|cache| cache.offset()))
                .collect::<Result<_, _>>()?;
            let output = runtime.forward_with_observer(
                PreparedCompositeInput::new(input, admission)
                    .map_err(std::io::Error::other)?
                    .with_attention_invocation(invocation),
                &mut state,
                stream,
                &mut Observer {
                    label: &label,
                    reference,
                    stream,
                },
            )?;
            compare(
                &format!("{label}.logits"),
                &output,
                &reference[&format!("{label}.logits")],
                stream,
            )?;
            let after: Vec<_> = (0..config.num_hidden_layers() as usize)
                .map(|i| state.layer(i).map(|cache| cache.offset()))
                .collect::<Result<_, _>>()?;
            if invocation == AttentionInvocation::DIFFUSION {
                assert_eq!(
                    before, after,
                    "VLM transient canvas advances its image/text prefix"
                );
            }
            println!("VLM {label}: canonical frontier={}", after[0]);
        }
    }
    Ok(())
}
