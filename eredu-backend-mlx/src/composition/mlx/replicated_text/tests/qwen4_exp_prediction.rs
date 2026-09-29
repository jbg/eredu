//! Ordinary explicit prediction admission through native typed construction.
use super::*;
use crate::composition::mlx::{
    prepared_speculative::{MlxEmbeddedExecutorContinuation, MlxEmbeddedExecutorTypes},
    speculative::SpeculativeExecutionStreams,
    MlxModelInput,
};
use eredu_core::SpeculativeExecutor;

struct Probe<'a> {
    stream: &'a Stream,
    prompt: &'a [u32],
    media: Option<(&'a [input::InputPart], usize)>,
    logits: Vec<f32>,
    draft: Vec<f32>,
}
impl MlxEmbeddedExecutorContinuation for Probe<'_> {
    fn execute(
        &mut self,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
        executor: &mut eredu_architectures::speculative_execution::DynEmbeddedExecutor<
            '_,
            MlxEmbeddedExecutorTypes,
        >,
    ) -> Result<eredu_core::SpeculativeGenerationBatchOutput, Error> {
        assert_eq!(
            selected.requirements().strategy().proposal_capacity().get(),
            1
        );
        let streams = SpeculativeExecutionStreams::single(self.stream);
        let tokens = Array::from_slice(self.prompt, &[1, self.prompt.len() as i32]);
        let text_parts = [input::token_ids_part(&tokens)?];
        let (parts, positions) = self.media.unwrap_or((&text_parts, self.prompt.len()));
        let prepared = eredu_runtime::PreparedModelInput::new(parts.to_vec(), |tensor| {
            eredu_runtime::PreparedInputInspector::identity(&input::MlxInputInspector, tensor)
        })
        .unwrap();
        let identity = prepared
            .cache_identity(format!("miniature-qwen4-prediction-{:?}", self.prompt))
            .unwrap();
        let input = MlxModelInput::from(input::ModelInput::with_cache_identity(parts, &identity));
        let mut cache = executor.new_cache()?;
        let (logits, state, evaluated) = executor.prefill(input, &mut cache, streams)?.into_parts();
        assert_eq!(evaluated, positions);
        self.logits = last_logits(logits);
        assert!(self.logits.iter().any(|v| v.abs() > 1e-5));
        let (saved, saved_state) = executor
            .control_snapshot(&cache, &state, streams)
            .map_err(|e| Error::ArchitectureModel(e.to_string()))?
            .expect("selected snapshot mechanism");
        let mut draft = executor.begin_proposal(&state, 7, 1, streams)?;
        self.draft = last_logits(executor.proposal_logits(&mut draft, 7, streams)?);
        assert!(self.draft.iter().any(|v| v.abs() > 1e-5));
        let restored = executor
            .restore_control_snapshot(&mut cache, &saved, &saved_state, streams)
            .map_err(|e| Error::ArchitectureModel(e.to_string()))?
            .expect("selected restore mechanism");
        let mut replay = executor.begin_proposal(&restored, 7, 1, streams)?;
        close(
            &last_logits(executor.proposal_logits(&mut replay, 7, streams)?),
            &self.draft,
            "native prediction snapshot replay",
        );
        Ok(eredu_core::SpeculativeGenerationBatchOutput::new(
            vec![],
            Default::default(),
        ))
    }
}

#[test]
fn ordinary_qwen4_embedded_prediction_native_residencies_and_snapshot_replay() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights_stream = Stream::new_with_device(&device);
    let mut reference: Option<(Vec<f32>, Vec<f32>)> = None;
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(
            eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 25), Some(1 << 26), 1).unwrap(),
            )
            .with_max_cached_shards(1),
        ),
        LayerWeightResidency::DenseDiskStream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 25, 1 << 26, 1, 1).unwrap(),
        ),
    ] {
        for (path, companion) in [
            (&fixtures.safetensors_path, None),
            (&fixtures.gguf_path, Some(&fixtures.safetensors_path)),
        ] {
            for paged in [false, true] {
                let state = if paged {
                    CacheResidencyPolicy::Paged(
                        PagedCacheOptions::new(2, 1 << 23, 1 << 23, 1)
                            .unwrap()
                            .with_full_attention(true),
                    )
                } else {
                    CacheResidencyPolicy::Device
                };
                let request = NormalizedLoadRequest::default()
                    .with_weight_residency(eredu_runtime::WeightResidency::with_layers(residency))
                    .with_state_residency(state)
                    .with_required_session_capabilities(eredu_core::SessionCapabilities::new(
                        true, true, true,
                    ))
                    .with_drafting(eredu_runtime::DraftingLoadRequest::embedded(1).unwrap());
                let request = if let Some(companion) = companion {
                    request.with_prediction_source(companion.clone())
                } else {
                    request
                };
                let mut executable = bind_prediction(path, request, &stream, &weights_stream);
                assert!(executable.erased_mut().has_embedded_prediction());
                let mut probe = Probe {
                    stream: &stream,
                    prompt: &PROMPT,
                    media: None,
                    logits: vec![],
                    draft: vec![],
                };
                executable
                    .erased_mut()
                    .with_embedded_prediction(&mut probe)
                    .unwrap()
                    .unwrap();
                if let Some((logits, draft)) = &reference {
                    close(
                        &probe.logits,
                        logits,
                        "native prediction target residency parity",
                    );
                    close(
                        &probe.draft,
                        draft,
                        "native prediction proposal residency parity",
                    );
                } else {
                    reference = Some((probe.logits, probe.draft));
                }
            }
        }
    }
}

fn bind_prediction(
    path: &std::path::Path,
    request: NormalizedLoadRequest,
    stream: &Stream,
    weights_stream: &Stream,
) -> crate::composition::mlx::Executable {
    let mut inspection = eredu_architectures::configuration::inspect_artifact(path).unwrap();
    if matches!(
        request.media_execution(),
        eredu_runtime::MediaLoadRequest::Required(_)
    ) && inspection
        .architecture_plan()
        .required_gguf_special_tokens()
        .is_some()
    {
        use eredu_architectures::processor_plan::{GgufSpecialTokenIds, QwenMediaTokenIds};
        inspection
            .architecture_plan_mut()
            .bind_gguf_special_token_ids(GgufSpecialTokenIds::Qwen(QwenMediaTokenIds {
                image_token_id: 12,
                video_token_id: 13,
                vision_start_token_id: 14,
                vision_end_token_id: 15,
            }))
            .unwrap();
    }
    let selected = crate::composition::mlx::loading::select_preparation(
        &inspection,
        crate::MlxLoadRequest::from_normalized(request),
    )
    .unwrap();
    let plan = eredu_core::ModelPreparationPlan::from_retained_admission(
        inspection,
        selected.neutral().admission(),
    )
    .unwrap();
    let (sources, rank) =
        crate::composition::mlx::loading::prepare_selected_sources(plan, selected).unwrap();
    assert!(rank.is_none());
    crate::composition::mlx::loading::materialize_model_plan(sources, None, stream, weights_stream)
        .unwrap()
        .into_executable()
}

// Broad fixture ceilings isolate the semantic invocation boundary. No payload
// grows with these workspace allowances; only the miniature model is loaded.
fn long_policy(chunk: i32) -> eredu_runtime::BoundedExecutionPolicy {
    use eredu_runtime::*;
    BoundedExecutionPolicy::new(
        InvocationLimits::new(1, chunk, 1024).unwrap(),
        TiledSelectionLimits::new(32, 1 << 20, 1 << 28).unwrap(),
        RowLookupLoadPolicy::new(
            RowLookupLimits {
                requests: 2048,
                rows_per_acquisition: 32,
                acquisition_bytes: 8192,
                host_bytes: 1 << 20,
                output_bytes: 1 << 20,
            },
            ParameterBankLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(0), 1).unwrap(),
                8 << 20,
                8 << 20,
            )
            .unwrap(),
            16,
        )
        .unwrap(),
        AppendStreamLoadPolicy::new(
            AppendStreamLimits {
                entries: 512,
                page_entries: 32,
                read_entries: 32,
            },
            1 << 20,
            1 << 20,
            1 << 20,
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn ordinary_qwen4_embedded_prediction_long_prefill_preserves_shifted_capture_and_replay() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    eredu_evaluation::qwen4_exp::set_context_length(&fixtures.safetensors_path, 1024).unwrap();
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights_stream = Stream::new_with_device(&device);
    let prompt: Vec<_> = PROMPT.into_iter().cycle().take(515).collect();
    let mut reference: Option<(Vec<f32>, Vec<f32>)> = None;
    // Reference runs as one invocation. Default512 and odd17-token chunks must
    // retain the cross-chunk shifted MTP pair and incomplete QSA tail exactly.
    for policy in [Some(long_policy(1024)), None, Some(long_policy(17))] {
        let mut request = NormalizedLoadRequest::default()
            .with_required_session_capabilities(eredu_core::SessionCapabilities::new(
                true, true, true,
            ))
            .with_drafting(eredu_runtime::DraftingLoadRequest::embedded(1).unwrap());
        if let Some(policy) = policy {
            request = request.with_bounded_execution(policy);
        }
        let mut executable = bind_prediction(
            &fixtures.safetensors_path,
            request,
            &stream,
            &weights_stream,
        );
        let mut probe = Probe {
            stream: &stream,
            prompt: &prompt,
            media: None,
            logits: vec![],
            draft: vec![],
        };
        executable
            .erased_mut()
            .with_embedded_prediction(&mut probe)
            .unwrap()
            .unwrap();
        if let Some((logits, draft)) = &reference {
            close(&probe.logits, logits, "chunked long target prefill");
            close(
                &probe.draft,
                draft,
                "chunked long shifted prediction prefill",
            );
        } else {
            reference = Some((probe.logits, probe.draft));
        }
    }
}

#[cfg(feature = "image")]
#[test]
fn ordinary_qwen4_media_prediction_native_residencies_and_snapshot_replay() {
    use eredu_core::{InputMetadataKey as Key, InputModality as Modality};
    use eredu_runtime::{MediaExecutionPolicy, MediaLoadRequest, ProcessorSelectionRequest};
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    eredu_evaluation::qwen4_exp::add_vision_weights(&fixtures.safetensors_path, 2);
    eredu_evaluation::qwen4_exp::write_vision_projector(&directory.path().join("mmproj.gguf"), 2);
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights_stream = Stream::new_with_device(&device);
    let text = |ids: &[u32]| {
        input::token_ids_part(&Array::from_slice(ids, &[1, ids.len() as i32])).unwrap()
    };
    let media = |modality, patches: i32, h, w| {
        let values: Vec<f32> = (0..patches * 24)
            .map(|i| ((i * 11 % 53) as f32 - 26.) / 64.)
            .collect();
        input::input_part(
            modality,
            input::InputPayload::Tensor(Array::from_slice(&values, &[patches, 24])),
            [(Key::PatchGrid, Array::from_slice(&[1i32, h, w], &[1, 3]))],
            [],
        )
        .unwrap()
    };
    let parts = [
        text(&[3, 14]),
        media(Modality::Image, 16, 4, 4),
        text(&[15, 0, 4, 14]),
        media(Modality::Video, 8, 4, 2),
        text(&[15, 6]),
    ];
    let mut reference: Option<(Vec<f32>, Vec<f32>)> = None;
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(
            eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 25), Some(1 << 26), 1).unwrap(),
            )
            .with_max_cached_shards(1),
        ),
        LayerWeightResidency::DenseDiskStream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 25, 1 << 26, 1, 1).unwrap(),
        ),
    ] {
        for (path, companion) in [
            (&fixtures.safetensors_path, None),
            (&fixtures.gguf_path, Some(&fixtures.safetensors_path)),
        ] {
            for paged in [false, true] {
                let state = if paged {
                    CacheResidencyPolicy::Paged(
                        PagedCacheOptions::new(2, 1 << 23, 1 << 23, 1)
                            .unwrap()
                            .with_full_attention(true),
                    )
                } else {
                    CacheResidencyPolicy::Device
                };
                let media = MediaExecutionPolicy::new(
                    ProcessorSelectionRequest::new([
                        Modality::Text,
                        Modality::Image,
                        Modality::Video,
                    ])
                    .with_projected_embeddings(true)
                    .with_available_raw_media(true),
                )
                .unwrap();
                let mut request = NormalizedLoadRequest::default()
                    .with_weight_residency(eredu_runtime::WeightResidency::with_layers(residency))
                    .with_state_residency(state)
                    .with_media_execution(MediaLoadRequest::Required(media))
                    .with_required_session_capabilities(eredu_core::SessionCapabilities::new(
                        true, true, true,
                    ))
                    .with_drafting(eredu_runtime::DraftingLoadRequest::embedded(1).unwrap());
                if let Some(companion) = companion {
                    request = request.with_prediction_source(companion.clone());
                }
                let mut executable = bind_prediction(path, request, &stream, &weights_stream);
                let mut probe = Probe {
                    stream: &stream,
                    prompt: &[3],
                    media: Some((&parts, 14)),
                    logits: vec![],
                    draft: vec![],
                };
                executable
                    .erased_mut()
                    .with_embedded_prediction(&mut probe)
                    .unwrap()
                    .unwrap();
                if let Some((logits, draft)) = &reference {
                    close(
                        &probe.logits,
                        logits,
                        "native media prediction target residency parity",
                    );
                    close(
                        &probe.draft,
                        draft,
                        "native media prediction proposal residency parity",
                    );
                } else {
                    reference = Some((probe.logits, probe.draft));
                }
            }
        }
    }
}
