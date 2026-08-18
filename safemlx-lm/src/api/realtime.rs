//! Codec-free realtime speech-to-speech token APIs.
//!
//! Realtime speech models in this crate operate on discrete codec tokens rather
//! than PCM. Callers are expected to encode live audio into model-native
//! codebook frames before calling these APIs, and decode emitted codebook frames
//! with a codec implementation outside `safemlx-lm`.

use safemlx::{
    ops::{indexing::TryIndexOp, stack_axis},
    Array, Stream,
};
pub use safemlx_lm_core::realtime::{
    RealtimeCompletedStep, RealtimeConfigError, RealtimeError, RealtimeSampling, RealtimeScheduler,
    RealtimeSession, RealtimeSpeechConfig,
};
pub use safemlx_lm_core::scheduler::{
    SchedulerCapabilities as RealtimeSchedulerCapabilities,
    SchedulerReport as RealtimeSchedulerReport,
};
use safemlx_lm_core::{
    realtime::RealtimeModel,
    scheduler::{RequestId, SchedulerLimits},
};
use serde::Deserialize;
use std::path::Path;

use crate::{
    api::{moshi, personaplex},
    backend::mlx::realtime::{
        MlxEncodedAudioOutput, MlxRealtimeBackend, MlxRealtimeInput, MlxRealtimeModel,
        RealtimeModelKind,
    },
    backend::mlx::{ensure_replicated_load_options, ModelLoadOptions},
    error::Error,
    runtime::checkpoint::artifact::{fingerprint_artifact, ArtifactFile, LoadedArtifactIdentity},
};

#[derive(Debug, Clone, Deserialize)]
struct RealtimeModelMetadata {
    #[serde(default)]
    model_type: Option<String>,
}

fn realtime_model_kind(model_dir: impl AsRef<Path>) -> Result<RealtimeModelKind, Error> {
    let config_path = model_dir.as_ref().join("config.json");
    if !config_path.exists() {
        return Ok(RealtimeModelKind::Moshi);
    }

    let metadata: RealtimeModelMetadata =
        serde_json::from_reader(std::fs::File::open(config_path)?)?;
    match metadata.model_type.as_deref() {
        None | Some("moshi") => Ok(RealtimeModelKind::Moshi),
        Some("personaplex") => Ok(RealtimeModelKind::PersonaPlex),
        Some(other) => Err(Error::UnsupportedArchitecture(format!(
            "{other} is not a realtime speech-to-speech token model"
        ))),
    }
}

fn realtime_artifact_identity(
    model_dir: &Path,
    kind: RealtimeModelKind,
) -> Result<LoadedArtifactIdentity, Error> {
    let index = model_dir.join("model.safetensors.index.json");
    let weight_files = if index.exists() {
        crate::runtime::checkpoint::load::safetensors_files(model_dir)?
    } else {
        match kind {
            RealtimeModelKind::Moshi => {
                let args = moshi::get_model_args(model_dir)?;
                vec![model_dir.join(args.moshi_name.as_deref().unwrap_or("model.safetensors"))]
            }
            RealtimeModelKind::PersonaPlex => {
                vec![model_dir.join(personaplex::MODEL_SAFETENSORS)]
            }
        }
    };
    let files = weight_files
        .into_iter()
        .map(|path| {
            let logical_name = path
                .strip_prefix(model_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            ArtifactFile::new(logical_name, path)
        })
        .collect::<Vec<_>>();
    fingerprint_artifact(kind.model_type(), files)
}

/// Loads a supported realtime speech-to-speech token model from a model directory.
///
/// This is the high-level realtime counterpart to [`crate::api::LoadedModel`].
/// It does not load a text tokenizer or audio codec: callers bring tokenization,
/// codec encode/decode, transport, and device I/O.
pub fn load_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<RealtimeModel<MlxRealtimeBackend>, Error> {
    load_model_with_options(
        model_dir,
        ModelLoadOptions::default(),
        stream,
        weights_stream,
    )
}

/// Loads a realtime model using the shared architecture-independent options.
///
/// Successful loads bind an immutable SHA-256 identity of the selected weight
/// files to the model. Realtime sessions use that identity, together with the
/// normalized execution configuration, when validating state handoff.
pub fn load_model_with_options(
    model_dir: impl AsRef<Path>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<RealtimeModel<MlxRealtimeBackend>, Error> {
    ensure_replicated_load_options(options)?;
    let model_dir = model_dir.as_ref();
    let kind = realtime_model_kind(model_dir)?;
    if options.weight_residency.expert_cache().is_some() {
        return Err(Error::UnsupportedArchitecture(format!(
            "{} does not contain routed experts",
            kind.model_type()
        )));
    }
    let execution = options.weight_residency.layers();
    let model = if let Some(quantization) = options.quantization {
        let transformed = match kind {
            RealtimeModelKind::Moshi => {
                moshi::load_model_quantized(model_dir, quantization, stream, weights_stream)?
            }
            RealtimeModelKind::PersonaPlex => {
                personaplex::load_model_quantized(model_dir, quantization, stream, weights_stream)?
            }
        };
        crate::architectures::moshi::layerwise::execute_transformed_model(
            transformed,
            stream,
            weights_stream,
        )?
    } else {
        match kind {
            RealtimeModelKind::Moshi => {
                crate::architectures::moshi::layerwise::load_moshi_layerwise_model(
                    model_dir,
                    execution,
                    stream,
                    weights_stream,
                )?
            }
            RealtimeModelKind::PersonaPlex => {
                crate::architectures::moshi::layerwise::load_personaplex_layerwise_model(
                    model_dir,
                    execution,
                    stream,
                    weights_stream,
                )?
            }
        }
    };
    let model = model.with_artifact_identity(realtime_artifact_identity(model_dir, kind)?);
    let model = match kind {
        RealtimeModelKind::Moshi => MlxRealtimeModel::Moshi(model),
        RealtimeModelKind::PersonaPlex => MlxRealtimeModel::PersonaPlex(model),
    };
    Ok(RealtimeModel::new(MlxRealtimeBackend::new(stream), model))
}

/// Greedily generates delay-aligned codec tokens through the canonical scheduler.
///
/// Input and output use `[batch, codebooks, frames]` layout. This helper does
/// not append encoded silence, so delayed tail frames are not flushed after the
/// supplied input ends.
pub fn generate_encoded_greedy(
    model: &mut RealtimeModel<MlxRealtimeBackend>,
    input_audio_tokens: &Array,
) -> Result<MlxEncodedAudioOutput, Error> {
    let stream = model.backend().stream().clone();
    let config = model.speech_config();
    let input_audio_codebooks = config.input_audio_codebooks() as i32;
    let generated_audio_codebooks = config.generated_audio_codebooks() as i32;
    if input_audio_tokens.shape().len() != 3 || input_audio_tokens.dim(1) != input_audio_codebooks {
        return Err(Error::Parallel(format!(
            "encoded input sequence must have shape [batch, {}, frames], got {:?}",
            input_audio_codebooks,
            input_audio_tokens.shape()
        )));
    }

    let batch = input_audio_tokens.dim(0);
    let request = RequestId::new(0);
    let mut scheduler =
        RealtimeScheduler::new(model, SchedulerLimits::new(1, 1)?).map_err(realtime_error)?;
    scheduler
        .register_request(model, request, RealtimeSampling::greedy())
        .map_err(realtime_error)?;
    let mut text = Vec::with_capacity(input_audio_tokens.dim(2) as usize);
    let mut audio = Vec::new();
    for frame in 0..input_audio_tokens.dim(2) {
        let input = input_audio_tokens.try_index_device((.., .., frame), &stream)?;
        scheduler
            .enqueue(model, request, MlxRealtimeInput::encoded_audio(&input))
            .map_err(realtime_error)?;
        let output = loop {
            if let Some(completed) = scheduler.run_queued(model).map_err(realtime_error)?.pop() {
                break completed.into_parts().1;
            }
            std::thread::yield_now();
        };
        text.push(output.text_token.squeeze_axes(&[-1], &stream)?);
        if let Some(tokens) = output.output_audio_tokens {
            audio.push(tokens);
        }
    }
    scheduler.finish_request(request).map_err(realtime_error)?;
    let text_tokens = if text.is_empty() {
        Array::zeros::<i32>(&[batch, 0], &stream)?
    } else {
        stack_axis(&text, 1, &stream)?
    };
    let audio_tokens = if audio.is_empty() {
        Array::zeros::<i32>(&[batch, generated_audio_codebooks, 0], &stream)?
    } else {
        stack_axis(&audio, 2, &stream)?
    };
    Ok(MlxEncodedAudioOutput {
        text_tokens,
        audio_tokens,
    })
}

fn realtime_error(error: RealtimeError<Error>) -> Error {
    Error::Parallel(error.to_string())
}

#[cfg(test)]
mod tests {
    use safemlx::{module::ModuleParameters, Array, Device, DeviceType, ExecutionContext, Stream};
    use safemlx_lm_core::realtime::RealtimeBackend;
    use safemlx_lm_core::scheduler::RequestStatus;

    use super::*;
    use crate::{
        backend::mlx::realtime::MlxRealtimeExecutionIdentity,
        runtime::generation::sampler::DefaultSampler,
    };

    fn tiny_args() -> moshi::ModelArgs {
        serde_json::from_value(serde_json::json!({
            "model_type": "moshi",
            "dim": 16,
            "text_card": 32,
            "n_q": 4,
            "dep_q": 2,
            "card": 8,
            "num_heads": 4,
            "num_layers": 2,
            "dim_feedforward": 32,
            "causal": true,
            "context": 16,
            "max_period": 10000.0,
            "positional_embedding": "rope",
            "depformer_dim": 8,
            "depformer_dim_feedforward": 16,
            "depformer_num_heads": 2,
            "depformer_num_layers": 2,
            "depformer_context": 2,
            "depformer_pos_emb": "none",
            "delays": [0, 0, 1, 0, 1]
        }))
        .unwrap()
    }

    fn tiny_model(stream: &Stream) -> RealtimeModel<MlxRealtimeBackend> {
        let mut resident = moshi::Model::new(tiny_args(), stream).unwrap();
        for (name, parameter) in resident.parameters_mut().flatten() {
            let shape = parameter.shape().to_vec();
            *parameter = if name.ends_with("norm.weight") {
                Array::ones::<f32>(&shape, stream).unwrap()
            } else {
                Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
            };
        }
        let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let model = MlxRealtimeModel::Moshi(
            crate::architectures::moshi::layerwise::execute_transformed_model(
                resident,
                stream,
                &weights_stream,
            )
            .unwrap(),
        );
        RealtimeModel::new(MlxRealtimeBackend::new(stream), model)
    }

    fn default_audio_samplers() -> Vec<DefaultSampler> {
        vec![DefaultSampler, DefaultSampler]
    }

    #[test]
    fn execution_identity_normalizes_defaults_and_tracks_quantization() {
        let mut implicit = tiny_args();
        implicit.dim_feedforward = None;
        implicit.depformer_dim_feedforward = None;
        implicit.depformer_context = None;
        implicit.depformer_max_period = None;
        let mut explicit = implicit.clone();
        explicit.dim_feedforward = Some(explicit.dim * 4);
        explicit.depformer_dim_feedforward = Some(explicit.depformer_dim * 4);
        explicit.depformer_context = Some(explicit.dep_q);
        explicit.depformer_max_period = Some(8.0);
        assert_eq!(
            MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &implicit),
            MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &explicit)
        );

        explicit.quantization =
            Some(crate::runtime::checkpoint::quantization::AffineQuantization::default().into());
        assert_ne!(
            MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &implicit),
            MlxRealtimeExecutionIdentity::new(RealtimeModelKind::Moshi, &explicit)
        );
    }

    #[test]
    fn released_session_rejects_a_different_same_geometry_model() {
        let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let first_model = tiny_model(context.stream());
        let second_model = tiny_model(context.stream());
        let first_identity = first_model.backend().model_identity(first_model.model());
        let second_identity = second_model.backend().model_identity(second_model.model());
        assert_ne!(first_identity, second_identity);

        let request = RequestId::new(7);
        let mut first_scheduler =
            RealtimeScheduler::new(&first_model, SchedulerLimits::new(1, 1).unwrap()).unwrap();
        first_scheduler
            .register_request(&first_model, request, RealtimeSampling::greedy())
            .unwrap();
        let session = first_scheduler.release_request(request).unwrap();

        let mut second_scheduler =
            RealtimeScheduler::new(&second_model, SchedulerLimits::new(1, 1).unwrap()).unwrap();
        let error = second_scheduler
            .register_request_with_session(&second_model, request, session)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("checkpoint artifact fingerprint"));
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn realtime_session_handoff_after_cancellation_uses_generic_lifecycle() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let model = tiny_model(stream);
        let mut scheduler =
            RealtimeScheduler::new(&model, SchedulerLimits::new(2, 2).unwrap()).unwrap();
        let first = RequestId::new(11);
        let second = RequestId::new(22);
        scheduler
            .register_request(&model, first, RealtimeSampling::greedy())
            .unwrap();
        scheduler
            .register_request(&model, second, RealtimeSampling::greedy())
            .unwrap();

        let frame = Array::from_slice(&[1i32, 2], &[1, 2]);
        assert!(scheduler
            .enqueue_batch(
                &model,
                first,
                vec![
                    MlxRealtimeInput::encoded_audio(&frame),
                    MlxRealtimeInput::encoded_audio(&frame),
                    MlxRealtimeInput::encoded_audio(&frame),
                ],
            )
            .unwrap_err()
            .to_string()
            .contains("queue capacity"));
        assert_eq!(scheduler.report().queued_work, 0);
        assert_eq!(scheduler.report().submitted_work, 0);
        scheduler
            .enqueue(&model, first, MlxRealtimeInput::encoded_audio(&frame))
            .unwrap();
        scheduler
            .enqueue(&model, second, MlxRealtimeInput::encoded_audio(&frame))
            .unwrap();
        assert!(scheduler
            .enqueue(&model, first, MlxRealtimeInput::encoded_audio(&frame))
            .unwrap_err()
            .to_string()
            .contains("queue capacity"));
        scheduler.cancel_request(second).unwrap();
        let changed_batch = Array::from_slice(&[1i32, 2, 3, 4], &[2, 2]);
        assert!(scheduler
            .enqueue(
                &model,
                first,
                MlxRealtimeInput::encoded_audio(&changed_batch),
            )
            .unwrap_err()
            .to_string()
            .contains("changed batch size"));
        scheduler.finish_request(first).unwrap();
        assert_eq!(
            scheduler.request_status(first),
            Some(RequestStatus::Finished)
        );
        assert_eq!(
            scheduler.request_status(second),
            Some(RequestStatus::Cancelled)
        );
        let report = scheduler.report();
        assert_eq!(report.active_requests, 0);
        assert_eq!(report.discarded_work, 2);
        assert_eq!(report.peak_queued_work, 2);

        scheduler.forget_terminal_request(first).unwrap();
        scheduler
            .register_request(&model, first, RealtimeSampling::greedy())
            .unwrap();
        let session = scheduler.release_request(first).unwrap();
        assert_eq!(session.state().step(), 0);
        scheduler
            .register_request_with_session(&model, first, session)
            .unwrap();
    }

    #[test]
    #[ignore = "requires an MLX runtime with a Metal device"]
    fn realtime_scheduler_is_fair_and_matches_independent_reference_sessions() {
        let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = context.stream();
        let mut model = tiny_model(stream);
        let resident = match model.model_mut() {
            MlxRealtimeModel::Moshi(model) => model,
            _ => unreachable!(),
        };
        let first_frame = Array::from_slice(&[1i32, 2], &[1, 2]);
        let second_frame = Array::from_slice(&[3i32, 4], &[1, 2]);
        let mut first_state = resident.new_generation_state();
        let mut second_state = resident.new_generation_state();
        let mut first_text = DefaultSampler;
        let mut second_text = DefaultSampler;
        let mut first_audio = default_audio_samplers();
        let mut second_audio = default_audio_samplers();
        let first_zero = resident
            .generate_step(
                &mut first_state,
                &first_frame,
                &mut first_text,
                &mut first_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let first_one = resident
            .generate_step(
                &mut first_state,
                &first_frame,
                &mut first_text,
                &mut first_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let second_zero = resident
            .generate_step(
                &mut second_state,
                &second_frame,
                &mut second_text,
                &mut second_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let second_one = resident
            .generate_step(
                &mut second_state,
                &second_frame,
                &mut second_text,
                &mut second_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let forced_audio = Array::from_slice(&[5i32, 6], &[1, 2]);
        let forced_text = Array::from_slice(&[7i32], &[1, 1]);
        let first_forced = resident
            .generate_step_forced(
                &mut first_state,
                &first_frame,
                Some(&forced_audio),
                Some(&forced_text),
                &mut first_text,
                &mut first_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let second_forced = resident
            .generate_step_forced(
                &mut second_state,
                &second_frame,
                Some(&forced_audio),
                Some(&forced_text),
                &mut second_text,
                &mut second_audio,
                0.0,
                0.0,
                None,
                stream,
            )
            .unwrap();
        let references = [
            first_zero,
            second_zero,
            first_one,
            second_one,
            first_forced,
            second_forced,
        ];

        let first = RequestId::new(11);
        let second = RequestId::new(22);
        let limits = SchedulerLimits::with_execution_bounds(2, 6, 2, 2, 1, 1).unwrap();
        let mut scheduler = RealtimeScheduler::new(&model, limits).unwrap();
        for request in [first, second] {
            scheduler
                .register_request(&model, request, RealtimeSampling::greedy())
                .unwrap();
        }
        for (request, frame) in [
            (first, &first_frame),
            (first, &first_frame),
            (first, &first_frame),
            (second, &second_frame),
            (second, &second_frame),
            (second, &second_frame),
        ] {
            let sequence = scheduler.queued_for_request(request);
            let input = if sequence == 2 {
                MlxRealtimeInput::encoded_audio(frame)
                    .with_forced_generated_audio(&forced_audio)
                    .with_forced_text(&forced_text)
            } else {
                MlxRealtimeInput::encoded_audio(frame)
            };
            scheduler.enqueue(&model, request, input).unwrap();
        }
        let mut output = Vec::new();
        while output.len() < 6 {
            output.extend(scheduler.run_bounded(&mut model, 2).unwrap());
            std::thread::yield_now();
        }
        assert_eq!(
            output
                .iter()
                .map(|output| (output.work().request().value(), output.work().sequence()))
                .collect::<Vec<_>>(),
            vec![(11, 0), (22, 0), (11, 1), (22, 1), (11, 2), (22, 2)]
        );
        for (expected, actual) in references.iter().zip(&output) {
            assert_tokens_equal(&expected.text_token, &actual.output().text_token, stream);
            assert_tokens_equal(
                &expected.sampled_audio_tokens,
                &actual.output().sampled_audio_tokens,
                stream,
            );
        }
        assert!((3..=6).contains(&scheduler.report().drain_cycles));
        assert_eq!(scheduler.release_request(first).unwrap().state().step(), 3);
        assert_eq!(scheduler.release_request(second).unwrap().state().step(), 3);
    }

    fn assert_tokens_equal(expected: &Array, actual: &Array, stream: &Stream) {
        let equal = expected
            .eq(actual, stream)
            .unwrap()
            .all(None, stream)
            .unwrap();
        assert!(equal.item::<bool>(stream));
    }
}
