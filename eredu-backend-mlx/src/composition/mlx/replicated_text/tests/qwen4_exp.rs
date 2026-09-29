//! Flash-Next through ordinary artifact admission, native construction and session drivers.
use super::*;
use eredu_runtime::{LayerWeightResidency, NormalizedLoadRequest};

#[path = "qwen4_exp_prediction.rs"]
mod prediction;

const PROMPT: [u32; 19] = [3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
// Frozen before the native fixture was evaluated; this is mechanism conformance,
// separate from released-checkpoint/reference-equation validation.
const ABS_TOLERANCE: f32 = 2e-4;
const REL_TOLERANCE: f32 = 2e-4;

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    assert!(actual.iter().all(|v| v.is_finite()), "{label}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= ABS_TOLERANCE + REL_TOLERANCE * expected.abs(),
            "{label} scalar {index}: {actual} != {expected}"
        );
    }
}

fn bind(
    artifact: &std::path::Path,
    residency: LayerWeightResidency,
    paged: bool,
    stream: &Stream,
    weights_stream: &Stream,
) -> (
    crate::composition::mlx::Executable,
    eredu_checkpoint::store::SharedCheckpointSource,
) {
    let cache = if paged {
        CacheResidencyPolicy::Paged(
            PagedCacheOptions::new(2, 1 << 23, 1 << 23, 1)
                .unwrap()
                .with_full_attention(true),
        )
    } else {
        CacheResidencyPolicy::Device
    };
    // Ordinary callers supply residency intent, not family workspace arithmetic.
    let load = NormalizedLoadRequest::default()
        .with_weight_residency(eredu_runtime::WeightResidency::with_layers(residency))
        .with_state_residency(cache.clone())
        .with_required_session_capabilities(eredu_core::SessionCapabilities::new(true, true, true))
        .with_prompt_cache_persistence(paged);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact).unwrap();
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
    assert!(std::sync::Arc::ptr_eq(sources.primary(), sources.target()));
    assert!(std::sync::Arc::ptr_eq(
        sources.primary(),
        sources.complete()
    ));
    let source = sources.primary().clone();
    // Source construction binds only the three exact integer SafeTensors controls.
    // GGUF retains those controls in metadata; table and ordinary weights stay lazy.
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        if artifact.is_dir() { 3 } else { 0 },
    );
    let model = crate::composition::mlx::loading::materialize_model_plan(
        sources,
        None,
        stream,
        weights_stream,
    )
    .unwrap();
    (model.into_executable(), source)
}

fn last_logits(output: Array) -> Vec<f32> {
    // The session indexes the declared text-output position before publication.
    assert_eq!(output.shape(), &[1, 16]);
    assert_eq!(output.dtype(), Dtype::Float32);
    let output = output.evaluated().unwrap();
    let values = output.as_slice::<f32>();
    assert_eq!(values.len(), 16);
    values.to_vec()
}

fn trajectory(
    model: &mut dyn ErasedReplicatedTextExecutable,
    chunks: &[usize],
    stream: &Stream,
) -> (Vec<Vec<f32>>, std::time::Duration, std::time::Duration) {
    let prefill_started = std::time::Instant::now();
    let mut offset = 0;
    let mut last = vec![];
    for &length in chunks {
        let tokens = Array::from_slice(&PROMPT[offset..offset + length], &[1, length as i32]);
        let parts = [input::token_ids_part(&tokens).unwrap()];
        last = last_logits(
            model
                .prefill(input::ModelInput::new(&parts), stream)
                .unwrap(),
        );
        offset += length;
    }
    assert_eq!(offset, PROMPT.len());
    let prefill_elapsed = prefill_started.elapsed();
    let decode_started = std::time::Instant::now();
    let mut outputs = vec![last];
    for token in 0..16u32 {
        outputs.push(last_logits(
            model
                .decode(&Array::from_slice(&[token], &[1, 1]), stream)
                .unwrap(),
        ));
    }
    assert_eq!(model.forecast_state_offset().unwrap(), Some(35));
    (outputs, prefill_elapsed, decode_started.elapsed())
}

struct FailAfterLogits {
    reached: bool,
}
impl eredu_runtime::ActivationObserver<Array, Error> for FailAfterLogits {
    fn observe(&mut self, path: &str, value: &Array) -> Result<(), Error> {
        if path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH {
            // Force all native work before failing after the complete state update.
            value.evaluated()?;
            self.reached = true;
            return Err(Error::ArchitectureModel(
                "injected late logits failure".into(),
            ));
        }
        Ok(())
    }
}

fn lifecycle(model: &mut dyn ErasedReplicatedTextExecutable, paged: bool, stream: &Stream) {
    let mut snapshot = model.capture_native_control_state().unwrap();
    let mut branch = model.copy_native_control_state(snapshot.as_ref()).unwrap();
    let saved_presence = model.state_snapshot();
    let saved_fixed = model.fixed_numeric_state_snapshot().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let prefix = PROMPT.into_iter().chain(0..16u32).collect::<Vec<_>>();
    let persisted = paged.then(|| {
        let descriptor = PromptCacheDescriptor::from_model_identity(
            model.prompt_cache_model_identity().clone(),
            "miniature-qwen4-native",
            "original-ids-with-eos",
            1,
        )
        .unwrap();
        let destination = cache_root.path().join("cache");
        model
            .save_prompt_cache(
                &destination,
                descriptor.clone(),
                &prefix,
                &PromptCacheOptions::default(),
            )
            .unwrap();
        (descriptor, destination)
    });
    let token = Array::from_slice(&[7u32], &[1, 1]);
    let expected = last_logits(model.decode(&token, stream).unwrap());
    model
        .exchange_native_control_state(snapshot.as_mut())
        .unwrap();
    assert_eq!(model.state_snapshot(), saved_presence);
    assert_eq!(model.fixed_numeric_state_snapshot().unwrap(), saved_fixed);
    let mut failing = FailAfterLogits { reached: false };
    assert!(model
        .decode_result_with_observer(Ok(&token), stream, &mut failing)
        .is_err());
    assert!(failing.reached);
    assert_eq!(model.state_snapshot(), saved_presence);
    assert_eq!(model.fixed_numeric_state_snapshot().unwrap(), saved_fixed);
    close(
        &last_logits(model.decode(&token, stream).unwrap()),
        &expected,
        "rollback replay",
    );
    model
        .exchange_native_control_state(branch.as_mut())
        .unwrap();
    assert_eq!(model.state_snapshot(), saved_presence);
    // The fork remains at the captured prefix while the other branch has advanced.
    close(
        &last_logits(model.decode(&token, stream).unwrap()),
        &expected,
        "isolated branch",
    );
    if let Some((descriptor, destination)) = persisted {
        model.reset_cache().unwrap();
        model
            .load_prompt_cache(&destination, &descriptor, &prefix)
            .unwrap();
        assert_eq!(model.state_snapshot(), saved_presence);
        assert_eq!(model.fixed_numeric_state_snapshot().unwrap(), saved_fixed);
        close(
            &last_logits(model.decode(&token, stream).unwrap()),
            &expected,
            "persisted state",
        );
    }
}

#[test]
fn ordinary_qwen4_target_native_containers_residencies_state_and_controls() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    let golden: serde_json::Value =
        serde_json::from_str(eredu_evaluation::qwen4_exp::DENSE_TRAJECTORY_JSON).unwrap();
    let golden_logits: Vec<Vec<f32>> = serde_json::from_value(golden["logits"].clone()).unwrap();
    assert_eq!(
        serde_json::from_value::<Vec<u32>>(golden["prefill"].clone()).unwrap(),
        PROMPT
    );
    assert_eq!(
        serde_json::from_value::<Vec<u32>>(golden["decode"].clone()).unwrap(),
        (0..16).collect::<Vec<_>>()
    );
    assert_eq!(
        golden["absolute_tolerance"].as_f64().unwrap() as f32,
        ABS_TOLERANCE
    );
    assert_eq!(
        golden["relative_tolerance"].as_f64().unwrap() as f32,
        REL_TOLERANCE
    );
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights_stream = Stream::new_with_device(&device);
    let residencies = [
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
    ];
    let mut reference: Option<Vec<Vec<f32>>> = None;
    for (container, artifact) in [
        ("safetensors", fixtures.safetensors_path.as_path()),
        ("gguf", fixtures.gguf_path.as_path()),
    ] {
        for residency in residencies {
            for paged in [false, true] {
                let label = format!("{container} {residency:?} paged={paged}");
                eprintln!("native ordinary target: {label}");
                let started = std::time::Instant::now();
                let (mut executable, source) =
                    bind(artifact, residency, paged, &stream, &weights_stream);
                let model = executable.erased_mut();
                let bind_elapsed = started.elapsed();
                assert_eq!(model.effective_model_type(), "qwen4_exp_text");
                assert_eq!(model.selected_residency(), residency);
                assert!(model.supports_chunked_prefill());
                let (whole, prefill_elapsed, decode_elapsed) = trajectory(model, &[19], &stream);
                let whole_elapsed = started.elapsed() - bind_elapsed;
                assert_eq!(whole.len(), golden_logits.len());
                for (actual, expected) in whole.iter().zip(&golden_logits) {
                    close(actual, expected, "frozen neutral mechanism trajectory");
                }
                let golden_max_error = whole
                    .iter()
                    .flatten()
                    .zip(golden_logits.iter().flatten())
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0_f32, f32::max);
                assert!(whole.iter().flatten().any(|v| v.abs() > 1e-5));
                if let Some(reference) = &reference {
                    for (actual, expected) in whole.iter().zip(reference) {
                        close(actual, expected, &label);
                    }
                } else {
                    reference = Some(whole.clone());
                }
                lifecycle(model, paged, &stream);
                model.reset_cache().unwrap();
                let (chunked, chunked_prefill_elapsed, warm_decode_elapsed) =
                    trajectory(model, &[3, 2, 4, 10], &stream);
                for (actual, expected) in chunked.iter().zip(&whole) {
                    close(actual, expected, &label);
                }
                let banks = model.parameter_bank_report().unwrap().unwrap();
                let rows = banks.rows().unwrap();
                let row_peak = rows.residency().offload().peak_resident_bytes();
                assert!(row_peak.get(eredu_core::residency::MemoryTier::Device) <= 4096);
                assert_eq!(row_peak.get(eredu_core::residency::MemoryTier::Host), 0);
                if residency != LayerWeightResidency::FullyResident {
                    let layers = model.residency_report().unwrap().unwrap();
                    let peak = layers.offload().peak_resident_bytes();
                    assert!(peak.get(eredu_core::residency::MemoryTier::Device) <= 1 << 25);
                    assert!(peak.get(eredu_core::residency::MemoryTier::Host) <= 1 << 26);
                }
                let cache = model.cache_residency_report().unwrap();
                assert_eq!(cache.is_some(), paged);
                if let Some(cache) = cache {
                    assert!(cache.peak_device_bytes <= 1 << 23);
                    assert!(cache.peak_host_bytes <= 1 << 23);
                    assert!(cache.append_stream_scratch_peak_bytes <= 65536);
                    assert!(cache.selected_attention_blocks > 0);
                }
                let reads_after = source.source_diagnostics().unwrap();
                eprintln!("{label}: bind={bind_elapsed:?}, prefill+16decode={whole_elapsed:?}, prefill19={prefill_elapsed:?}, decode16={decode_elapsed:?}, chunked_prefill19={chunked_prefill_elapsed:?}, warm_decode16={warm_decode_elapsed:?}, total={:?}, physical_reads={}, read_bytes={}, row_pool_peak={row_peak:?}, golden_max_abs={golden_max_error}",
                    started.elapsed(), reads_after.physical_reads,
                    reads_after.physical_read_bytes);
            }
        }
    }
}
