//! Validates released shared-stack execution against a pinned independent reference.
use eredu_architectures::diffusion_gemma::{
    self,
    model::{TextInput, TextKernel, TextModel},
    FamilyConfig,
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
        },
        ExecutionContext,
    },
    MlxTensor,
};
use eredu_checkpoint::{
    safetensors::SafetensorsMetadataCatalog,
    store::{open_prepared_safetensors_source, CheckpointSource},
    validation::resolve_safetensors_plan,
};
use eredu_nn::{AttentionCache, Parameterized, Tensor, TensorElementType};
use eredu_runtime::{
    LayerRuntimeState, LayeredArchitecture, LayerwiseRuntime, ResidentRuntime, ResidentUnitWindow,
};
use safemlx::{Array, Device, DeviceType, Stream};
use std::{collections::BTreeMap, error::Error, path::PathBuf};

type Backend = MlxNeuralBackend;
type Model = TextModel<Backend>;
type State = MlxKeyValueState;

struct LayerObserver<'a> {
    step: usize,
    refinement: bool,
    reference: &'a std::collections::HashMap<String, Array>,
    stream: &'a Stream,
}
impl eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error> for LayerObserver<'_> {
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), eredu_nn::Error> {
        let Some(index) = path
            .strip_prefix("model.decoder.layers.")
            .and_then(|path| path.strip_suffix(".output"))
            .and_then(|index| index.parse::<usize>().ok())
        else {
            return Ok(());
        };
        let label = if self.refinement {
            format!("decoder.{}.layer.{index}", self.step)
        } else {
            format!("layer.{index}")
        };
        if self.refinement || self.step == 0 {
            compare(&label, value, &self.reference[&label], self.stream)
                .map_err(eredu_nn::Error::backend)?;
        }
        Ok(())
    }
}
fn load<M: Parameterized<MlxTensor>>(
    module: &mut M,
    source: &dyn CheckpointSource,
    recipes: &mut BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    stream: &Stream,
) -> Result<(), Box<dyn Error>> {
    let bindings = build_neutral_module_bindings_with_recipes(module, source, recipes)?;
    let arrays = materialize_module_bindings(source, &bindings, stream, stream)?;
    populate_module_from_arrays_excluding(module, &arrays, |_| false)?;
    Ok(())
}
fn compare(
    label: &str,
    actual: &MlxTensor,
    reference: &Array,
    stream: &Stream,
) -> Result<(), Box<dyn Error>> {
    if actual.shape() != reference.shape() {
        return Err(format!("{label}: shape differs").into());
    }
    let a = actual
        .as_array()
        .as_dtype(safemlx::Dtype::Float32, stream)?
        .into_evaluated()?
        .try_to_vec::<f32>()?;
    let b = reference
        .as_dtype(safemlx::Dtype::Float32, stream)?
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
    let tolerance = 2e-5;
    if a.is_empty() || a.iter().chain(&b).any(|v| !v.is_finite()) || maximum > tolerance {
        return Err(format!(
            "{label}: released BF16 comparison exceeds absolute tolerance {tolerance}"
        )
        .into());
    }

    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(
        args.next()
            .ok_or("usage: diffusion_gemma_reference CHECKPOINT REFERENCE_SAFETENSORS")?,
    );
    let reference = PathBuf::from(args.next().ok_or("missing reference fixture")?);
    let vision_only = match args.next() {
        None => false,
        Some(option) if option == "--vision-only" => true,
        Some(_) => return Err("unexpected validation argument".into()),
    };
    let config = FamilyConfig::from_hf_json(&std::fs::read(path.join("config.json"))?)?;
    let plan = diffusion_gemma::safetensors_plan(&config).map_err(std::io::Error::other)?;
    let catalog = SafetensorsMetadataCatalog::discover(&path)?;
    let resolution = resolve_safetensors_plan(&catalog, &plan)
        .map_err(|issues| std::io::Error::other(format!("checkpoint admission: {issues:?}")))?;
    let source = open_prepared_safetensors_source(
        catalog.admitted_shards(),
        catalog.tensors().clone(),
        resolution,
        2,
    )?;
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let reference = Array::load_safetensors(reference, stream)?;
    if vision_only {
        use eredu_architectures::{diffusion_gemma::vision::ImageStatic, gemma4::VisionInput};
        let mut image = ImageStatic::<Backend>::new(&config, stream)?;
        load(&mut image, source.as_ref(), &mut BTreeMap::new(), stream)?;
        let patches = MlxTensor::from_array(
            reference["vision.patches"].as_dtype(safemlx::Dtype::Bfloat16, stream)?,
        );
        let positions = MlxTensor::from_array(reference["vision.positions"].clone());
        let grid = reference["vision.grid"]
            .clone()
            .into_evaluated()?
            .try_to_vec::<i32>()?;
        if grid.len() != 2 {
            return Err("vision reference grid requires height and width".into());
        }
        let count = patches.dim(1);
        let mask = MlxTensor::from_array(Array::from_slice(
            &vec![0.0f32; count as usize],
            &[1, 1, 1, count],
        ));
        let valid = MlxTensor::from_array(Array::from_slice(
            &vec![1.0f32; count as usize],
            &[1, count, 1],
        ));
        let (mut hidden, state) = image.begin(
            VisionInput {
                position_valid: &valid,
                patches: &patches,
                position_ids: &positions,
                key_mask: &mask,
                grid_extents: &[(grid[0], grid[1])],
            },
            stream,
        )?;
        compare(
            "vision.embedding",
            &hidden,
            &reference["vision.embedding"],
            stream,
        )?;
        for index in 0..config.vision.as_ref().unwrap().num_hidden_layers as usize {
            let mut layer = ImageStatic::<Backend>::build_layer(&config, index, stream)?;
            load(&mut layer, source.as_ref(), &mut BTreeMap::new(), stream)?;
            hidden = image.forward_layer(&mut layer, &hidden, &state, stream)?;
            compare(
                &format!("vision.layer.{index}"),
                &hidden,
                &reference[&format!("vision.layer.{index}")],
                stream,
            )?;
        }
        let pooled = image
            .tower
            .as_ref()
            .ok_or("image tower is not retained")?
            .finish(&hidden, &state, stream)?;
        compare(
            "vision.pooled",
            &pooled.reshape(&[pooled.dim(0) * pooled.dim(1), pooled.dim(2)], stream)?,
            &reference["vision.pooled"],
            stream,
        )?;
        let projected = image
            .projector
            .as_mut()
            .ok_or("image projector is not retained")?
            .forward(&pooled, stream)?;
        compare(
            "vision.projected",
            &projected.reshape(
                &[projected.dim(0) * projected.dim(1), projected.dim(2)],
                stream,
            )?,
            &reference["vision.projected"],
            stream,
        )?;
        return Ok(());
    }
    let mut resident = ResidentRuntime::<Model, Backend, State>::new(
        TextKernel::<Backend>::new(config.clone(), stream)?,
        stream,
    )?;
    load(
        <Model as LayeredArchitecture<Backend, State>>::static_modules_mut(
            resident.architecture_mut(),
        ),
        source.as_ref(),
        &mut BTreeMap::new(),
        stream,
    )?;
    for (index, unit) in resident.units_mut()[0].iter_mut().enumerate() {
        let expert = diffusion_gemma::expert_recipes(source.as_ref(), &config, index)
            .map_err(std::io::Error::other)?;
        let mut recipes = BTreeMap::from([
            (expert.target_gate_up, expert.gate_up),
            (expert.target_down, expert.down),
        ]);
        load(unit, source.as_ref(), &mut recipes, stream)?;
        if !recipes.is_empty() {
            return Err("unconsumed shared expert recipe".into());
        }
        println!("Loaded shared text layer {index}");
    }
    let (model, units) = resident.into_parts();
    let mut runtime =
        LayerwiseRuntime::<Model, Backend, State, _>::new(model, ResidentUnitWindow::new(units));
    let mut state = State::device(diffusion_gemma::state_layout(&config)?)?;
    let cases: [(&[i32], &[i32]); 3] = [
        (&[2, 100, 200], &[101, 211]),
        (&[17], &[421, 653]),
        (&[31, 29], &[1019, 2047]),
    ];
    let mut feedback = None;
    for (step, (tokens, canvas)) in cases.into_iter().enumerate() {
        let tokens = MlxTensor::from_array(Array::from_slice(tokens, &[1, tokens.len() as i32]));
        let mut observer = LayerObserver {
            step,
            refinement: false,
            reference: &reference,
            stream,
        };
        let encoded = runtime.forward_with_observer(
            TextInput::Context {
                tokens: &tokens,
                mask: None,
            },
            &mut state,
            stream,
            &mut observer,
        )?;
        compare(
            &format!("encode.{step}"),
            &encoded,
            &reference[&format!("encode.{step}")],
            stream,
        )?;
        let offsets: Vec<_> = (0..config.text.num_hidden_layers())
            .map(|i| state.layer(i).map(|cache| cache.offset()))
            .collect::<Result<_, _>>()?;
        let canvas = MlxTensor::from_array(Array::from_slice(canvas, &[1, canvas.len() as i32]));
        observer.refinement = true;
        let logits = runtime.forward_with_observer(
            TextInput::Refinement {
                tokens: &canvas,
                feedback: feedback.as_ref(),
                mask: None,
            },
            &mut state,
            stream,
            &mut observer,
        )?;
        compare(
            &format!("logits.{step}"),
            &logits,
            &reference[&format!("logits.{step}")],
            stream,
        )?;
        assert_eq!(
            (0..config.text.num_hidden_layers())
                .map(|i| state.layer(i).map(|cache| cache.offset()))
                .collect::<Result<Vec<_>, _>>()?,
            offsets
        );
        feedback = Some(
            logits
                .divide(&MlxTensor::from_f32_slice(&[0.8], &[1], stream)?, stream)?
                .cast_float(TensorElementType::Bf16, stream)?,
        );
    }
    Ok(())
}
