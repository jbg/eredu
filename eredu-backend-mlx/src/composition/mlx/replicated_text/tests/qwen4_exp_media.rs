//! Native conditional cursor conformance through ordinary artifact admission and loading.
use super::*;
use crate::backend::runtime::media::input;
use eredu_core::{InputMetadataKey as Key, InputModality as Modality};
use eredu_evaluation::qwen4_exp::{add_vision_weights, write_vision_projector};
use eredu_runtime::*;

fn load_policy(residency: LayerWeightResidency, chunk: usize) -> NormalizedLoadRequest {
    let policy = eredu_evaluation::qwen4_exp::bounded_policy();
    let config = eredu_architectures::qwen4_exp::config::Config::from_json(
        &eredu_evaluation::qwen4_exp::configuration(),
    )
    .unwrap();
    let rows_per_token = ((config.ngram.order - 1) * config.ngram.heads) as usize;
    let row_width = config.ngram.embedding_dim as usize / rows_per_token;
    let mut rows = policy.rows().limits();
    rows.requests = chunk * rows_per_token;
    // The generic row contract admits three simultaneous output buffers:
    // completed row chunks, concatenation, and request-order restoration.
    rows.output_bytes = (rows.requests * row_width * std::mem::size_of::<f32>() * 3) as u64;
    let row_policy = RowLookupLoadPolicy::new(
        rows,
        policy.rows().bank(),
        policy.rows().retained_scalar_bytes(),
    )
    .unwrap();
    let policy = BoundedExecutionPolicy::new(
        InvocationLimits::new(1, chunk as i32, 128).unwrap(),
        policy.selection(),
        row_policy,
        policy.append(),
    )
    .unwrap();
    assert_eq!(policy.invocation().batch(), 1);
    assert_eq!(policy.invocation().invocation_tokens(), chunk);
    assert_eq!(policy.rows().limits().requests, chunk * rows_per_token);
    assert_eq!(
        policy.rows().limits().output_bytes,
        (chunk * config.ngram.embedding_dim as usize * 4 * 3) as u64
    );
    if chunk < 48 {
        assert!(
            policy.rows().limits().requests < 48 * rows_per_token,
            "table lookup admission must cover only one invocation, not the full media prompt"
        );
    }
    let media = MediaExecutionPolicy::new(
        ProcessorSelectionRequest::new([Modality::Text, Modality::Image, Modality::Video])
            .with_projected_embeddings(true)
            .with_available_raw_media(true),
    )
    .unwrap();
    NormalizedLoadRequest::default()
        .with_bounded_execution(policy)
        .with_media_execution(MediaLoadRequest::Required(media))
        .with_weight_residency(WeightResidency::with_layers(residency))
        .with_required_session_capabilities(eredu_core::SessionCapabilities::new(false, true, true))
}

fn bind(
    artifact: &std::path::Path,
    residency: LayerWeightResidency,
    chunk: usize,
    processor_budget: Option<processor_resources::ProcessorRequestBudget>,
    stream: &Stream,
    weights_stream: &Stream,
) -> (
    crate::composition::mlx::MlxModel,
    eredu_checkpoint::store::SharedCheckpointSource,
    Option<eredu_checkpoint::store::SharedCheckpointSource>,
) {
    let mut inspection = eredu_architectures::configuration::inspect_artifact(artifact).unwrap();
    if inspection
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
    let mut load = load_policy(residency, chunk);
    // A supplied GGUF projector is sufficient media intent for ordinary loading.
    if processor_budget.is_none() && !artifact.is_dir() {
        load = load.with_media_execution(MediaLoadRequest::ArchitectureDefault);
    }
    if let Some(budget) = processor_budget {
        let MediaLoadRequest::Required(policy) = load.media_execution() else {
            unreachable!()
        };
        let media = MediaExecutionPolicy::new(policy.processor().clone().with_raw_media(true))
            .unwrap()
            .with_processor_budget(budget);
        load = load.with_media_execution(MediaLoadRequest::Required(media));
    }
    let selected = crate::composition::mlx::loading::select_preparation(
        &inspection,
        crate::MlxLoadRequest::from_normalized(load),
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
    let primary = sources.primary().clone();
    let projector = sources
        .companion(&eredu_core::GgufCompanionRole::MediaProjector)
        .cloned();
    assert_eq!(
        primary.source_diagnostics().unwrap().physical_reads,
        if artifact.is_dir() { 3 } else { 0 }
    );
    assert_eq!(projector.is_some(), !artifact.is_dir());
    if let Some(source) = &projector {
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    }
    let model = crate::composition::mlx::loading::materialize_model_plan(
        sources,
        None,
        stream,
        weights_stream,
    )
    .unwrap();
    (model, primary, projector)
}

fn prompt(stream: &Stream) -> Vec<input::InputPart> {
    fn text(ids: &[u32], _stream: &Stream) -> input::InputPart {
        input::token_ids_part(&Array::from_slice(ids, &[1, ids.len() as i32])).unwrap()
    }
    fn media(
        modality: Modality,
        patches: i32,
        h: i32,
        w: i32,
        _stream: &Stream,
    ) -> input::InputPart {
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
    }
    vec![
        text(&[3, 14], stream),
        media(Modality::Image, 16, 4, 4, stream),
        text(&[15, 4, 14], stream),
        media(Modality::Video, 8, 4, 2, stream),
        text(&[15, 5, 14], stream),
        media(Modality::Video, 8, 4, 2, stream),
        text(&[15, 5], stream),
        text(&[3; 32], stream),
    ]
}
#[derive(Default)]
struct ForwardCount(usize);
impl ActivationObserver<Array, Error> for ForwardCount {
    fn observe(&mut self, _: &str, _: &Array) -> Result<(), Error> {
        self.0 += 1;
        Ok(())
    }
}

#[derive(Default)]
struct EncoderCount(usize);
impl ActivationObserver<Array, Error> for EncoderCount {
    fn observe(&mut self, path: &str, _: &Array) -> Result<(), Error> {
        if path == "model.visual.blocks.1.output" {
            self.0 += 1;
        }
        Ok(())
    }
}
fn logits(value: Array) -> Vec<f32> {
    value.evaluated().unwrap().as_slice::<f32>().to_vec()
}
fn close(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f32;
    for (a, b) in actual.iter().zip(expected) {
        maximum = maximum.max((a - b).abs());
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-4 + 2e-4 * b.abs(),
            "{a} != {b}"
        );
    }
    maximum
}
#[test]
fn qwen4_conditional_native_composite_cursor_encoder_once_and_cached_decode() {
    let started = std::time::Instant::now();
    let dir = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(dir.path()).unwrap();
    let path = dir.path().join("mmproj.gguf");
    write_vision_projector(&path, 2);
    add_vision_weights(&fixtures.safetensors_path, 2);
    let st_config = eredu_architectures::qwen4_exp::config::Config::from_json(
        &serde_json::from_slice(
            &std::fs::read(fixtures.safetensors_path.join("config.json")).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let gguf_vision = eredu_architectures::qwen4_exp::prepared::GgufVisionPlan::prepare(
        fixtures.gguf_header.text_plan().config(),
        &eredu_gguf::Checkpoint::open(&path).unwrap(),
    )
    .unwrap();
    assert_eq!(
        st_config.vision.as_ref().unwrap(),
        gguf_vision.config(),
        "fixture containers must declare identical vision equations"
    );
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights = Stream::new_with_device(&device);
    let input = prompt(&stream); // 50 decoder positions, including eight media placeholders.
    let mut expected = None::<Vec<Vec<f32>>>;
    for st in [false, true] {
        for residency in [
            LayerWeightResidency::FullyResident,
            LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
            )),
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
            ),
        ] {
            for chunk in [64, 32, 8, 4] {
                let case_started = std::time::Instant::now();
                let (loaded, source, projector) = bind(
                    if st {
                        &fixtures.safetensors_path
                    } else {
                        &fixtures.gguf_path
                    },
                    residency,
                    chunk,
                    None,
                    &stream,
                    &weights,
                );
                let mut executable = loaded.into_executable();
                let model = executable.erased_mut();
                let bind_elapsed = case_started.elapsed();
                let mut oversized = input.clone();
                oversized.push(
                    input::token_ids_part(&Array::from_slice(&[3u32; 129], &[1, 129])).unwrap(),
                );
                let initial = model.state_snapshot();
                let mut denied_observer = ForwardCount::default();
                assert!(matches!(
                    model.prefill_with_observer(
                        input::ModelInput::new(&oversized),
                        None,
                        &stream,
                        &mut denied_observer,
                    ),
                    Err(Error::ModelStatePreserved(_))
                ));
                assert_eq!(
                    denied_observer.0, 0,
                    "history rejection precedes native forward work"
                );
                assert_eq!(model.state_snapshot(), initial);
                let prefill_started = std::time::Instant::now();
                let mut observer = EncoderCount::default();
                let first = model
                    .prefill_with_observer(
                        input::ModelInput::new(&input),
                        None,
                        &stream,
                        &mut observer,
                    )
                    .unwrap();
                assert_eq!(observer.0, 1, "encoder runs once for chunk {chunk}");
                let mut outputs = vec![logits(first)];
                let prefill_elapsed = prefill_started.elapsed();
                let mut saved = model.capture_native_control_state().unwrap();
                let decode_started = std::time::Instant::now();
                for step in 0..16u32 {
                    let token = Array::from_slice(&[step % 12], &[1, 1]);
                    outputs.push(logits(
                        model
                            .decode_result_with_observer(Ok(&token), &stream, &mut observer)
                            .unwrap(),
                    ));
                }
                let decode_elapsed = decode_started.elapsed();
                assert_eq!(observer.0, 1, "cached decode must not rerun encoder");
                let mut maximum = 0.0_f32;
                assert!(outputs.iter().flatten().any(|v| v.abs() > 1e-3));
                if let Some(reference) = &expected {
                    for (a, b) in outputs.iter().zip(reference) {
                        maximum = maximum.max(close(a, b));
                    }
                } else {
                    expected = Some(outputs.clone());
                }
                model.exchange_native_control_state(saved.as_mut()).unwrap();
                let token = Array::from_slice(&[0u32], &[1, 1]);
                maximum = maximum.max(close(
                    &logits(model.decode(&token, &stream).unwrap()),
                    &outputs[1],
                ));
                let banks = model.parameter_bank_report().unwrap().unwrap();
                let rows = banks
                    .rows()
                    .unwrap()
                    .residency()
                    .offload()
                    .peak_resident_bytes();
                assert!(rows.get(eredu_core::residency::MemoryTier::Device) <= 4096);
                assert_eq!(rows.get(eredu_core::residency::MemoryTier::Host), 0);
                let primary_reads = source.source_diagnostics().unwrap();
                let projector_reads = projector.as_ref().map(|s| s.source_diagnostics().unwrap());
                // Both admitted tiny artifacts fit in the selected caches. Repeated
                // decode must not repeatedly reload payloads or grow reader caches.
                let companion_bytes = projector_reads
                    .as_ref()
                    .map_or(0, |d| d.physical_read_bytes);
                let companion_reads = projector_reads.as_ref().map_or(0, |d| d.physical_reads);
                assert!(primary_reads.physical_read_bytes + companion_bytes <= 512 * 1024);
                assert!(primary_reads.physical_reads + companion_reads <= 128);
                assert!(primary_reads.currently_cached_shards <= 4);
                assert!(projector_reads
                    .as_ref()
                    .is_none_or(|d| d.currently_cached_shards <= 4));
                eprintln!(
                    "native media st={st} residency={residency:?} chunk={chunk}: bind={bind_elapsed:?}, prefill50={prefill_elapsed:?}, decode16={decode_elapsed:?}, total={:?}, max_abs={maximum}, row_pool_peak={rows:?}, primary_reads={primary_reads:?}, projector_reads={projector_reads:?}",
                    case_started.elapsed()
                );
            }
        }
    }
    eprintln!("native media complete: {:?}", started.elapsed());
}

#[cfg(feature = "image")]
fn raw_processor_budget() -> processor_resources::ProcessorRequestBudget {
    processor_resources::ProcessorRequestBudget {
        decoded_input_bytes: 1 << 20,
        host_buffer_bytes: 1 << 24,
        output_tensor_bytes: 1 << 20,
        planning_items: 4096,
        decoder_positions: 64,
    }
}

#[cfg(feature = "image")]
fn raw_prompt() -> eredu_core::TokenizedMultimodalRequest {
    use eredu_core::{
        Media, MultimodalRequest, MultimodalSegment as Segment, RgbImage, Video, VideoSampling,
    };
    let image = |width, height, seed| {
        RgbImage::new(
            (0..width * height * 3)
                .map(|i| ((i * 7 + seed) % 251) as u8)
                .collect(),
            width,
            height,
        )
        .unwrap()
    };
    MultimodalRequest::new(vec![
        Segment::TokenIds(vec![3]),
        Segment::Media(Media::Image(image(8, 8, 9))),
        Segment::TokenIds(vec![4]),
        Segment::Media(Media::Video(
            Video::new(
                (0..4).map(|i| image(8, 4, i * 17 + 1)).collect(),
                Some(1.0),
                VideoSampling::All,
            )
            .unwrap(),
        )),
        Segment::TokenIds(vec![3; 32]),
    ])
    .unwrap()
    .tokenize::<std::convert::Infallible>(|_| unreachable!())
    .unwrap()
}

#[cfg(feature = "image")]
#[derive(Default)]
struct ProcessorOutputs(usize);
#[cfg(feature = "image")]
impl ActivationObserver<Array, safemlx::error::Exception> for ProcessorOutputs {
    fn observe(&mut self, _: &str, value: &Array) -> Result<(), safemlx::error::Exception> {
        value.evaluated()?;
        self.0 += 1;
        Ok(())
    }
}

#[cfg(feature = "image")]
#[test]
fn qwen4_conditional_native_raw_processor_budget_and_ordinary_loading() {
    use crate::backend::runtime::media::ProcessorPreparationError;
    use processor_resources::ProcessorResourceError;
    let started = std::time::Instant::now();
    let dir = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(dir.path()).unwrap();
    write_vision_projector(&dir.path().join("mmproj.gguf"), 2);
    add_vision_weights(&fixtures.safetensors_path, 2);
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights = Stream::new_with_device(&device);
    let request = raw_prompt();
    // Tokenization of video timestamp framing remains facade policy; the fixture
    // supplies one explicit token for each timestamp in either container.
    let mut encode = |_: &str| Ok::<_, std::convert::Infallible>(vec![6]);
    let mut expected = None::<Vec<Vec<f32>>>;
    for artifact in [&fixtures.gguf_path, &fixtures.safetensors_path] {
        let mut zero = raw_processor_budget();
        zero.decoded_input_bytes = 0;
        let (mut denied, _, _) = bind(
            artifact,
            LayerWeightResidency::FullyResident,
            64,
            Some(zero),
            &stream,
            &weights,
        );
        let denied = denied
            .take_processor()
            .expect("ordinary loading retains the selected raw processor");
        let mut observed = ProcessorOutputs::default();
        for result in [
            denied.prepare_portable_input(&request, &mut encode),
            denied.prepare_portable_input_with_observer(&request, &mut encode, &mut observed),
        ] {
            assert!(matches!(
                result,
                Err(ProcessorPreparationError::Backend(
                    Error::ProcessorResources(ProcessorResourceError::Exhausted { limit: 0, .. })
                ))
            ));
        }
        assert_eq!(
            observed.0, 0,
            "budget denial precedes processor output publication"
        );
        let mut large_processor_budget = raw_processor_budget();
        large_processor_budget.decoder_positions = 256;
        let (mut history_model, _, _) = bind(
            artifact,
            LayerWeightResidency::FullyResident,
            8,
            Some(large_processor_budget),
            &stream,
            &weights,
        );
        let history_processor = history_model.take_processor().unwrap();
        let history_model = history_model.into_executable();
        let initial = history_model.erased().state_snapshot();
        let overhistory = eredu_core::MultimodalRequest::new(vec![
            eredu_core::MultimodalSegment::TokenIds(vec![3; 129]),
            eredu_core::MultimodalSegment::Media(eredu_core::Media::Image(
                eredu_core::RgbImage::new(vec![7; 4 * 4 * 3], 4, 4).unwrap(),
            )),
        ])
        .unwrap()
        .tokenize::<std::convert::Infallible>(|_| unreachable!())
        .unwrap();
        let mut observed = ProcessorOutputs::default();
        for result in [
            history_processor.prepare_portable_input(&overhistory, &mut encode),
            history_processor.prepare_portable_input_with_observer(
                &overhistory,
                &mut encode,
                &mut observed,
            ),
        ] {
            assert!(
                matches!(
                    result,
                    Err(ProcessorPreparationError::Backend(
                        Error::ProcessorResources(ProcessorResourceError::Exhausted {
                            resource: "decoder positions",
                            limit: 128,
                            ..
                        })
                    ))
                ),
                "retained sequence history must constrain a larger caller processor budget"
            );
        }
        assert_eq!(observed.0, 0);
        assert_eq!(history_model.erased().state_snapshot(), initial);
        for residency in [
            LayerWeightResidency::FullyResident,
            LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
            )),
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
            ),
        ] {
            for chunk in [64, 32, 8, 4] {
                let case = std::time::Instant::now();
                let (mut loaded, source, projector) = bind(
                    artifact,
                    residency,
                    chunk,
                    (chunk != 64).then(raw_processor_budget),
                    &stream,
                    &weights,
                );
                let processor = loaded
                    .take_processor()
                    .expect("ordinary loading retains a processor with optional limits");
                let mut executable = loaded.into_executable();
                let processing = std::time::Instant::now();
                let ordinary = processor
                    .prepare_portable_input(&request, &mut encode)
                    .unwrap();
                assert!(ordinary.cache_identity().is_some());
                let mut observed = ProcessorOutputs::default();
                let inspected = processor
                    .prepare_portable_input_with_observer(&request, &mut encode, &mut observed)
                    .unwrap();
                assert!(observed.0 > 0);
                assert!(inspected.cache_identity().is_none());
                assert_eq!(ordinary.identity(), inspected.identity());
                for (a, b) in ordinary.wire_arrays().iter().zip(inspected.wire_arrays()) {
                    assert_eq!(a.shape(), b.shape());
                    assert_eq!(a.dtype(), b.dtype());
                    // Numeric conversion preserves these small exact integers and f32 pixels.
                    close(
                        a.as_dtype(Dtype::Float32, &stream)
                            .unwrap()
                            .evaluated()
                            .unwrap()
                            .as_slice::<f32>(),
                        b.as_dtype(Dtype::Float32, &stream)
                            .unwrap()
                            .evaluated()
                            .unwrap()
                            .as_slice::<f32>(),
                    );
                }
                let processing = processing.elapsed();
                let input = if chunk == 64 { &ordinary } else { &inspected };
                let model = executable.erased_mut();
                let mut encoder = EncoderCount::default();
                let prefill = std::time::Instant::now();
                let first = model
                    .prefill_with_observer(
                        input.cache_identity().map_or_else(
                            || input::ModelInput::new(input.input_parts()),
                            |identity| {
                                input::ModelInput::with_cache_identity(
                                    input.input_parts(),
                                    identity,
                                )
                            },
                        ),
                        None,
                        &stream,
                        &mut encoder,
                    )
                    .unwrap();
                let prefill = prefill.elapsed();
                assert_eq!(encoder.0, 1);
                // This fixture declares a 2x2 spatial merger. Count original
                // text IDs plus merged media rows from the prepared description.
                let positions: usize = input
                    .identity()
                    .parts()
                    .iter()
                    .map(|part| {
                        let shape = part.payload().shape();
                        match part.modality() {
                            Modality::Text => shape.iter().product(),
                            Modality::Image | Modality::Video => {
                                assert_eq!(shape.len(), 2);
                                assert_eq!(shape[0] % 4, 0);
                                shape[0] / 4
                            }
                            _ => panic!("unexpected raw fixture modality"),
                        }
                    })
                    .sum();
                // Video's 64-pixel volume ceiling resizes the four frames to
                // 4x4: two temporal groups each contribute one merged position.
                assert_eq!(positions, 48);
                let mut outputs = vec![logits(first)];
                let mut snapshot = model.capture_native_control_state().unwrap();
                let decode = std::time::Instant::now();
                for step in 0..16u32 {
                    outputs.push(logits(
                        model
                            .decode_result_with_observer(
                                Ok(&Array::from_slice(&[step % 12], &[1, 1])),
                                &stream,
                                &mut encoder,
                            )
                            .unwrap(),
                    ));
                }
                let decode = decode.elapsed();
                assert_eq!(encoder.0, 1);
                let mut maximum = 0.0_f32;
                assert!(outputs.iter().flatten().any(|v| v.abs() > 1e-3));
                if let Some(reference) = &expected {
                    for (a, b) in outputs.iter().zip(reference) {
                        maximum = maximum.max(close(a, b));
                    }
                } else {
                    expected = Some(outputs.clone());
                }
                model
                    .exchange_native_control_state(snapshot.as_mut())
                    .unwrap();
                maximum = maximum.max(close(
                    &logits(
                        model
                            .decode(&Array::from_slice(&[0u32], &[1, 1]), &stream)
                            .unwrap(),
                    ),
                    &outputs[1],
                ));
                let row_peak = model
                    .parameter_bank_report()
                    .unwrap()
                    .unwrap()
                    .rows()
                    .unwrap()
                    .residency()
                    .offload()
                    .peak_resident_bytes();
                assert!(row_peak.get(eredu_core::residency::MemoryTier::Device) <= 4096);
                assert_eq!(row_peak.get(eredu_core::residency::MemoryTier::Host), 0);
                let primary_reads = source.source_diagnostics().unwrap();
                let projector_reads = projector.as_ref().map(|s| s.source_diagnostics().unwrap());
                let total_bytes = primary_reads.physical_read_bytes
                    + projector_reads
                        .as_ref()
                        .map_or(0, |d| d.physical_read_bytes);
                let total_reads = primary_reads.physical_reads
                    + projector_reads.as_ref().map_or(0, |d| d.physical_reads);
                assert!(total_bytes <= 512 * 1024);
                assert!(total_reads <= 128);
                assert!(primary_reads.currently_cached_shards <= 4);
                assert!(projector_reads
                    .as_ref()
                    .is_none_or(|d| d.currently_cached_shards <= 4));
                eprintln!("native raw media st={} residency={residency:?} chunk={chunk}: processing_twice={processing:?}, prefill{positions}={prefill:?}, decode16={decode:?}, total={:?}, max_abs={maximum}, row_peak={row_peak:?}, physical_reads={total_reads}, read_bytes={total_bytes}",
                    artifact.is_dir(), case.elapsed());
            }
        }
    }
    eprintln!("native raw media complete: {:?}", started.elapsed());
}
