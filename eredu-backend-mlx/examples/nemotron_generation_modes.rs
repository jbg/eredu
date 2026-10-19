//! Normal loader and generation-driver comparison using pinned released references.
use eredu_backend_mlx::{backend::MlxBackend, MlxLoadRequest};
use eredu_core::{
    decoding::{DecodingMode, SpeculationOptionsRequest},
    execution_control::GenerationControlHandle,
    ControlledTextGeneration, ModelRuntime, TextAdvancement, TextGenerationBackend,
    TokenFilterController,
};
use eredu_runtime::{DenseDiskStreamLoadOptions, LayerwiseLoadOptions, WeightResidency};
use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType, Dtype, Stream};
use std::{error::Error, path::PathBuf};

struct ExcludeMask {
    vocabulary: usize,
    mask: u32,
}
impl TokenFilterController for ExcludeMask {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
        Ok(eredu_core::TokenFilter::allowed(
            (0..self.vocabulary)
                .map(|token| token != self.mask as usize)
                .collect(),
        )
        .unwrap())
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        assert_ne!(token, self.mask);
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn last_scores(scores: &Array, stream: &Stream) -> Result<Array, Box<dyn Error>> {
    Ok(if scores.shape().len() == 3 {
        scores.try_index_device((.., -1, ..), stream)?
    } else {
        scores.clone()
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let source = PathBuf::from(
        args.next()
            .ok_or("usage: nemotron_generation_modes CHECKPOINT REFERENCE [resident|host|disk]")?,
    );
    let reference = PathBuf::from(args.next().ok_or("missing independent reference")?);
    let residency = match args
        .next()
        .and_then(|arg| arg.into_string().ok())
        .as_deref()
    {
        None | Some("resident") => WeightResidency::fully_resident(),
        Some("host") => WeightResidency::layerwise_host(LayerwiseLoadOptions::default()),
        Some("disk") => WeightResidency::dense_disk_stream(DenseDiskStreamLoadOptions::new(
            12 << 30,
            4 << 30,
            2,
            2,
        )?),
        _ => return Err("residency requires resident, host or disk".into()),
    };
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights = Stream::new_with_device(&device);
    let reference = Array::load_safetensors(reference, &weights)?;
    let backend = eredu_backend_mlx::native::backend(&stream, &weights);
    let model = eredu_core::load_model(
        &backend,
        &source,
        MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
        ),
    )?;
    let mut runtime = ModelRuntime::from_prepared(backend, model)?;
    println!("normal model loader completed");
    let capabilities = MlxBackend::decoding_capabilities(&runtime);
    assert!(
        capabilities.autoregressive && capabilities.masked_unmasking && capabilities.mode_changes
    );
    let mask = capabilities
        .mask_token_id
        .ok_or("missing admitted mask identity")?;
    let mut vocabulary = 0;
    for (step, tokens) in [&[1u32, 100, 200][..], &[17], &[31], &[421]]
        .into_iter()
        .enumerate()
    {
        let submission = if step == 0 {
            let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), tokens.to_vec())?;
            runtime.prefill(prompt)?
        } else {
            runtime.decode(Array::from_slice(tokens, &[1, 1]))?
        };
        let scores = submission
            .wait()?
            .into_logits()
            .ok_or("missing causal logits")?;
        let actual = last_scores(scores.as_array(), &stream)?
            .as_dtype(Dtype::Float32, &stream)?
            .into_evaluated()?
            .try_to_vec::<f32>()?;
        let expected = last_scores(&reference[&format!("causal.{step}.logits")], &stream)?
            .as_dtype(Dtype::Float32, &stream)?
            .into_evaluated()?
            .try_to_vec::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        vocabulary = actual.len();
        let maximum = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let admitted = actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() <= 0.002 + 0.00002 * b.abs());
        println!("causal.{step}: max_abs={maximum:.8}; admitted={admitted}");
        if !admitted {
            return Err("normal loader differs from independently decoded artifact".into());
        }
    }
    runtime.reset()?;
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(70),
            temperature: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        },
    )?;
    let handle = GenerationControlHandle::default();
    let mut generation = ControlledTextGeneration::new(
        &mut runtime,
        vec![1, 100, 200],
        eredu_core::TextGenerationConfig::new(sampling),
        ExcludeMask { vocabulary, mask },
    )?;
    generation.bind_control_handle(&handle)?;
    let mut prefix = vec![1, 100, 200];
    for _ in 0..2 {
        prefix.push(generation.next().ok_or("early AR termination")??.token_id());
    }
    handle.request_decoding_mode(DecodingMode::Diffusion { policy: None })?;
    let TextAdvancement::ModeChanged(ack) =
        generation.advance_operation()?.ok_or("missing handoff")?
    else {
        return Err("missing AR handoff acknowledgement".into());
    };
    assert_eq!(ack.committed_position, 2);
    prefix.push(
        generation
            .next()
            .ok_or("early diffusion termination")??
            .token_id(),
    );
    handle.request_decoding_mode(DecodingMode::Autoregressive)?;
    loop {
        match generation
            .advance_operation()?
            .ok_or("early transition termination")?
        {
            TextAdvancement::Token(token) => prefix.push(token.token_id()),
            TextAdvancement::Progress(_) => {}
            TextAdvancement::ModeChanged(ack) => {
                assert_eq!(ack.committed_position as usize, prefix.len() - 3);
                println!(
                    "diffusion -> AR: committed_position={}",
                    ack.committed_position
                );
                break;
            }
        }
    }
    let resumed = generation
        .next()
        .ok_or("early resumed termination")??
        .token_id();
    drop(generation);
    runtime.reset()?;
    let expected = ControlledTextGeneration::new(
        &mut runtime,
        prefix,
        eredu_core::TextGenerationConfig::new(sampling),
        ExcludeMask { vocabulary, mask },
    )?
    .next()
    .ok_or("missing independent causal commitment")??
    .token_id();
    assert_eq!(resumed, expected);
    println!("resumed AR token={resumed}; fresh causal prefix agrees");
    runtime.reset()?;
    let causal = ControlledTextGeneration::new(
        &mut runtime,
        vec![1, 100, 200],
        eredu_core::TextGenerationConfig::new(sampling),
        ExcludeMask { vocabulary, mask },
    )?
    .take(4)
    .map(|token| token.map(|token| token.token_id()))
    .collect::<Result<Vec<_>, _>>()?;
    runtime.reset()?;
    let mode = DecodingMode::SelfSpeculation(
        SpeculationOptionsRequest {
            max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        }
        .try_into()?,
    );
    let speculative = ControlledTextGeneration::new(
        &mut runtime,
        vec![1, 100, 200],
        eredu_core::TextGenerationConfig::new(sampling).with_decoding_mode(Some(mode)),
        ExcludeMask { vocabulary, mask },
    )?
    .take(4)
    .map(|token| token.map(|token| token.token_id()))
    .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        speculative, causal,
        "released shared speculation preserves the greedy target"
    );
    println!("released self-speculation agrees with AR: {causal:?}");
    Ok(())
}
