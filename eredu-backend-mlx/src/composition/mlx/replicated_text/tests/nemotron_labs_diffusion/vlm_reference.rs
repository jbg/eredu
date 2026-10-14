//! Released image processing and cached context through normal model loading.
use super::*;

#[test]
#[ignore = "requires the pinned released VLM checkpoint and independent image/context reference"]
fn normal_loader_released_pixtral_prepared_context_matches_independent_rows() {
    use eredu_core::decoding::AttentionInvocation;
    use safemlx::ops::indexing::TryIndexOp;
    let root = std::path::PathBuf::from(
        std::env::var_os("EREDU_NEMOTRON_RELEASE_ROOT").expect("pinned VLM checkpoint"),
    );
    let config = eredu_architectures::nemotron_labs_diffusion::FamilyConfig::from_hf_json(
        &std::fs::read(root.join("config.json")).unwrap(),
    )
    .unwrap();
    let processor = eredu_architectures::nemotron_labs_diffusion::processing::ImageProcessor::new(
        &config, 1400,
    )
    .unwrap();
    let (stream, weights) = execution_streams();
    let source = std::env::var_os("EREDU_NEMOTRON_MODEL_SOURCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.clone());
    let decoded_gguf = source
        .extension()
        .is_some_and(|extension| extension == "gguf");
    let backend = crate::native::backend(&stream, &weights);
    let model = eredu_core::load_model(
        &backend,
        &source,
        crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::layerwise_host(Default::default()),
            ),
        ),
    )
    .unwrap();
    let reference_path = std::env::var_os("EREDU_NEMOTRON_IMAGE_REFERENCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            root.join(if config.tensor_fp8() {
                "reference-mixed-context.safetensors"
            } else {
                "reference-bf16-context.safetensors"
            })
        });
    let reference = Array::load_safetensors(reference_path, &weights).unwrap();
    let compare_scores =
        |label: &str, actual: &Array, expected: &Array, invocation| -> Result<bool, Error> {
            if !config.tensor_fp8() && !decoded_gguf {
                return super::adapter::compare_bfloat_scores(
                    label, actual, expected, invocation, &stream,
                );
            }
            assert_eq!(
                actual.dtype(),
                safemlx::Dtype::Float32,
                "bounded decoded target precision"
            );
            let a = actual.clone().into_evaluated()?;
            let b = expected
                .as_dtype(safemlx::Dtype::Float32, &stream)?
                .into_evaluated()?;
            let mut maximum = 0.0f32;
            let mut valid = true;
            assert_eq!(actual.shape(), expected.shape());
            for (a, b) in a.as_slice::<f32>().iter().zip(b.as_slice::<f32>()) {
                maximum = maximum.max((a - b).abs());
                valid &=
                    a.is_finite() && b.is_finite() && (a - b).abs() <= 0.002 + 0.00002 * b.abs();
            }
            println!("{label}: output=Float32 max_abs={maximum:.8} within_tolerance={valid}");
            Ok(valid)
        };
    let mut patches = Vec::new();
    let mut grids = Vec::new();
    for image in 0..2 {
        let size = reference[&format!("vision.rgb_size.{image}")]
            .clone()
            .into_evaluated()
            .unwrap()
            .as_slice::<i32>()
            .to_vec();
        let rgb = reference[&format!("vision.rgb.{image}")]
            .clone()
            .into_evaluated()
            .unwrap()
            .as_slice::<u8>()
            .to_vec();
        let prepared = processor
            .prepare(
                eredu_media::image::RgbImageView::packed(&rgb, size[1] as u32, size[0] as u32)
                    .unwrap(),
            )
            .unwrap();
        patches.push(Array::from_slice(
            &prepared.patches,
            &[
                (prepared.grid.0 * prepared.grid.1) as i32,
                3 * config.vision.as_ref().unwrap().patch_size.pow(2),
            ],
        ));
        grids.push(prepared.grid);
    }
    let packed =
        safemlx::ops::concatenate_axis(&patches.iter().collect::<Vec<_>>(), 0, &stream).unwrap();
    let pixel_values = packed.clone().into_evaluated().unwrap();
    let reference_pixels = reference["vision.patches"]
        .clone()
        .into_evaluated()
        .unwrap();
    let pixel_error = pixel_values
        .as_slice::<f32>()
        .iter()
        .zip(reference_pixels.as_slice::<f32>())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // OpenCV's SIMD cubic reductions and scalar host reductions have distinct
    // FP32 accumulation orders. Check preprocessing before neural rounding.
    assert!(pixel_error <= 0.00001, "prepared pixel error {pixel_error}");
    let native = packed
        .as_dtype(safemlx::Dtype::Bfloat16, &stream)
        .unwrap()
        .as_dtype(safemlx::Dtype::Float32, &stream)
        .unwrap()
        .into_evaluated()
        .unwrap();
    let expected = reference["vision.patches"]
        .as_dtype(safemlx::Dtype::Bfloat16, &stream)
        .unwrap()
        .as_dtype(safemlx::Dtype::Float32, &stream)
        .unwrap()
        .into_evaluated()
        .unwrap();
    assert_eq!(
        native.as_slice::<f32>().len(),
        expected.as_slice::<f32>().len()
    );
    let mismatches = native
        .as_slice::<f32>()
        .iter()
        .zip(expected.as_slice::<f32>())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .collect::<Vec<_>>();
    println!(
        "prepared pixels: max_abs={pixel_error:.9} BF16_rounding_differences={}",
        mismatches.len()
    );
    // Neural comparisons consume identical prepared inputs. The processor is
    // independently checked above, before its FP32 differences can cross BF16
    // rounding boundaries and propagate through the complete vision tower.
    let mut offset = 0i32;
    for (patches, &(height, width)) in patches.iter_mut().zip(&grids) {
        let end = offset + (height * width) as i32;
        *patches = reference["vision.patches"]
            .try_index_device((offset..end, ..), &stream)
            .unwrap();
        offset = end;
    }
    let text = [
        Array::from_slice(&[1u32], &[1, 1]),
        Array::from_slice(&[4u32, 5], &[1, 2]),
        Array::from_slice(&[7u32], &[1, 1]),
    ];
    let image = (0..2)
        .map(|i| {
            input::input_part(
                eredu_core::InputModality::Image,
                input::InputPayload::Tensor(patches[i].clone()),
                [(
                    eredu_core::InputMetadataKey::PatchGrid,
                    Array::from_slice(&[1i32, grids[i].0 as i32, grids[i].1 as i32], &[1, 3]),
                )],
                [eredu_core::InputExtent::PatchGrid {
                    time: 1,
                    height: grids[i].0 as usize,
                    width: grids[i].1 as usize,
                }],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let parts = [
        input::token_ids_part(&text[0]).unwrap(),
        image[0].clone(),
        input::token_ids_part(&text[1]).unwrap(),
        image[1].clone(),
        input::token_ids_part(&text[2]).unwrap(),
    ];
    let input = input::ModelInput::new(&parts);
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    struct Layers<'a> {
        reference: &'a std::collections::HashMap<String, Array>,
        stream: &'a Stream,
        values: std::collections::HashMap<String, Array>,
    }
    impl eredu_runtime::ActivationObserver<Array, Error> for Layers<'_> {
        fn observe(&mut self, path: &str, value: &Array) -> Result<(), Error> {
            if path.starts_with("model.layers.0.") {
                self.values.insert(format!("raw.{path}"), value.clone());
            }
            let key = if let Some(index) = path
                .strip_prefix("model.vision_tower.transformer.layers.")
                .and_then(|s| s.strip_suffix(".output"))
            {
                Some(format!("vision.layer.{index}"))
            } else if path == "model.vision_tower.ln_pre.output" {
                Some("vision.embedding".into())
            } else if path == "model.multi_modal_projector.output" {
                Some("vision.projected".into())
            } else if let Some(index) = path
                .strip_prefix("model.layers.")
                .and_then(|s| s.strip_suffix(".output"))
                .filter(|s| s.parse::<usize>().is_ok())
            {
                Some(format!("causal.0.layer.{index}"))
            } else {
                None
            };
            if let Some(key) = key.filter(|key| self.reference.contains_key(key)) {
                self.values.insert(key.clone(), value.clone());
                assert_eq!(value.shape(), self.reference[&key].shape());
                let a = value
                    .as_dtype(safemlx::Dtype::Float32, self.stream)?
                    .into_evaluated()?;
                let b = self.reference[&key]
                    .as_dtype(safemlx::Dtype::Float32, self.stream)?
                    .into_evaluated()?;
                let max = a
                    .as_slice::<f32>()
                    .iter()
                    .zip(b.as_slice::<f32>())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!("{key}: dtype={:?} max_abs={max:.8}", value.dtype());
            }
            Ok(())
        }
    }
    runtime
        .session_mut()
        .with_model_operation(|model| {
            let model = model.erased_mut();
            let mut cache = model.prepare_autoregressive_cache()?;
            let mut observer = Layers {
                reference: &reference,
                stream: &stream,
                values: Default::default(),
            };
            observer
                .values
                .insert("processor.patches".into(), packed.clone());
            observer
                .values
                .insert("vision.patches".into(), reference["vision.patches"].clone());
            let (first, positions) =
                model.prepared_context_forward(input, &mut cache, &stream, &mut observer)?;
            observer
                .values
                .insert("causal.0.logits".into(), first.clone());
            let components = if decoded_gguf {
                source
                    .parent()
                    .unwrap()
                    .join("normal-loader-gguf-context-components.safetensors")
            } else {
                root.join(if config.tensor_fp8() {
                    "normal-loader-mixed-context-components.safetensors"
                } else {
                    "normal-loader-bf16-context-components.safetensors"
                })
            };
            Array::save_safetensors(&observer.values, None, components).unwrap();
            assert_eq!(positions, reference["context.tokens"].dim(1) as usize);
            let mut frontier = positions as u64;
            let mut failures = Vec::new();
            if !compare_scores(
                "causal.0",
                &first,
                &reference["causal.0.logits"],
                AttentionInvocation::CAUSAL,
            )? {
                failures.push("causal.0".to_owned());
            }
            for step in 0..4 {
                if step > 0 {
                    let token = [17u32, 31, 421][step - 1];
                    let scores = model.attention_forward_with_observer(
                        &Array::from_slice(&[token], &[1, 1]),
                        None,
                        AttentionInvocation::CAUSAL,
                        Some(&mut cache),
                        false,
                        &stream,
                        &mut eredu_runtime::NoopObserver,
                    )?;
                    frontier += 1;
                    let label = format!("causal.{step}");
                    if !compare_scores(
                        &label,
                        &scores,
                        &reference[&format!("{label}.logits")],
                        AttentionInvocation::CAUSAL,
                    )? {
                        failures.push(label);
                    }
                }
                let label = format!("block.{step}");
                let scores = model.attention_forward_with_observer(
                    &reference[&format!("{label}.tokens")],
                    None,
                    AttentionInvocation::DIFFUSION,
                    Some(&mut cache),
                    false,
                    &stream,
                    &mut eredu_runtime::NoopObserver,
                )?;
                if !compare_scores(
                    &label,
                    &scores,
                    &reference[&format!("{label}.logits")],
                    AttentionInvocation::DIFFUSION,
                )? {
                    failures.push(label);
                }
                assert_eq!(
                    cache.generation()?,
                    frontier,
                    "image context survives transient refinement"
                );
            }
            assert!(
                failures.is_empty(),
                "released image-context comparison differs: {failures:?}"
            );
            Ok(())
        })
        .unwrap();
}
