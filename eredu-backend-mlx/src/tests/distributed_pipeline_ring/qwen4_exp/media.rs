//! Ordinary image/video admission through native distributed prepared execution.
use super::*;
use crate::backend::runtime::media::input;
use eredu_core::{InputMetadataKey as Key, InputModality as Modality};
use eredu_runtime::{
    BoundedExecutionPolicy, InvocationLimits, LayerWeightResidency, MediaExecutionPolicy,
    MediaLoadRequest, NormalizedLoadRequest, ProcessorSelectionRequest, RowLookupLoadPolicy,
};

pub(super) fn is_fixture(checkpoint: &Path) -> bool {
    if checkpoint.is_dir() {
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap())
                .unwrap();
        config.get("vision_config").is_some()
    } else {
        checkpoint.parent().unwrap().join("mmproj.gguf").exists()
    }
}

fn run(gguf: bool, axes: &'static str, residency: WorkerResidency) {
    assert!(distributed::is_available(Backend::Ring));
    let directory = tempfile::tempdir().unwrap();
    let fixture = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_vision_weights(&fixture.safetensors_path, 2);
    eredu_evaluation::qwen4_exp::write_vision_projector(&directory.path().join("mmproj.gguf"), 2);
    // The ordinary inspected-artifact fixture helper supplies facade-owned IDs.
    std::fs::write(
        directory.path().join("component-media-fixture.json"),
        serde_json::to_vec(&serde_json::json!({
            "patch_width":24, "image_token_id":12, "video_token_id":13,
            "vision_start_token_id":14, "vision_end_token_id":15,
        }))
        .unwrap(),
    )
    .unwrap();
    let path = if gguf {
        fixture.gguf_path
    } else {
        fixture.safetensors_path
    };
    run_ring_pipeline_processes(
        residency,
        FixtureFamily::Qwen4Exp,
        WorkerMode::OpaqueSession,
        directory,
        path,
        Some(axes),
    );
}

#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_media_tp_safetensors_resident() {
    run(false, "tp", WorkerResidency::FullyResident);
}
#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_media_pp_gguf_host() {
    run(true, "pp", WorkerResidency::LayerwiseHost);
}
#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_media_ep_safetensors_disk() {
    run(false, "ep", WorkerResidency::DenseDiskStream);
}
#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_media_tp_pp_safetensors_host() {
    run(false, "tp-pp", WorkerResidency::LayerwiseHost);
}
#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_media_tp_ep_gguf_resident() {
    run(true, "tp-ep", WorkerResidency::FullyResident);
}
#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_media_pp_ep_gguf_disk() {
    run(true, "pp-ep", WorkerResidency::DenseDiskStream);
}
#[test]
#[ignore = "requires eight local MLX Ring ranks and loopback sockets"]
fn ring_media_tp_pp_ep_safetensors_host() {
    run(false, "tp-pp-ep", WorkerResidency::LayerwiseHost);
}
#[test]
#[ignore = "requires eight local MLX Ring ranks and loopback sockets"]
fn ring_media_tp_pp_ep_gguf_disk() {
    run(true, "tp-pp-ep", WorkerResidency::DenseDiskStream);
}

pub(super) fn load_policy(layers: LayerWeightResidency, chunk: usize) -> NormalizedLoadRequest {
    let policy = eredu_evaluation::qwen4_exp::bounded_policy();
    let config = eredu_architectures::qwen4_exp::config::Config::from_json(
        &eredu_evaluation::qwen4_exp::configuration(),
    )
    .unwrap();
    let rows_per_token = ((config.ngram.order - 1) * config.ngram.heads) as usize;
    let row_width = config.ngram.embedding_dim as usize / rows_per_token;
    let bank = ParameterBankLoadOptions::new(
        OffloadConfig::new(Some(1 << 25), Some(1 << 26), 1).unwrap(),
        1 << 25,
        1 << 24,
    )
    .unwrap();
    let mut rows = policy.rows().limits();
    rows.requests = chunk * rows_per_token;
    rows.output_bytes = (rows.requests * row_width * 4 * 3) as u64;
    let rows = RowLookupLoadPolicy::new(rows, bank, policy.rows().retained_scalar_bytes()).unwrap();
    let policy = BoundedExecutionPolicy::new(
        InvocationLimits::new(1, chunk as i32, 128).unwrap(),
        policy.selection(),
        rows,
        policy.append(),
    )
    .unwrap();
    let weights = match layers {
        LayerWeightResidency::FullyResident => WeightResidency::with_layers(layers),
        LayerWeightResidency::LayerwiseHost(options) => {
            WeightResidency::with_independent_parameter_banks(
                OrdinaryWeightResidency::LayerwiseHost(options),
                bank,
            )
        }
        LayerWeightResidency::DenseDiskStream(options) => {
            WeightResidency::with_independent_parameter_banks(
                OrdinaryWeightResidency::DenseDiskStream(options),
                bank,
            )
        }
        _ => unreachable!("fixture selects resident, host or disk"),
    };
    NormalizedLoadRequest::default()
        .with_bounded_execution(policy)
        .with_media_execution(MediaLoadRequest::Required(
            MediaExecutionPolicy::new(
                ProcessorSelectionRequest::new([Modality::Text, Modality::Image, Modality::Video])
                    .with_projected_embeddings(true)
                    .with_available_raw_media(true),
            )
            .unwrap(),
        ))
        .with_weight_residency(weights)
        .with_state_residency(CacheResidencyPolicy::Paged(
            PagedCacheOptions::new(2, 1 << 23, 1 << 23, 1)
                .unwrap()
                .with_full_attention(true),
        ))
        .with_required_session_capabilities(eredu_core::SessionCapabilities::new(false, true, true))
}

pub(super) fn prompt() -> MlxModelInput {
    let text = |ids: &[u32]| {
        input::token_ids_part(&Array::from_slice(ids, &[1, ids.len() as i32])).unwrap()
    };
    let media = |modality, patches: i32, height, width| {
        let values: Vec<f32> = (0..patches * 24)
            .map(|index| ((index * 11 % 53) as f32 - 26.) / 64.)
            .collect();
        input::input_part(
            modality,
            input::InputPayload::Tensor(Array::from_slice(&values, &[patches, 24])),
            [(
                Key::PatchGrid,
                Array::from_slice(&[1i32, height, width], &[1, 3]),
            )],
            [],
        )
        .unwrap()
    };
    let parts = [
        text(&[3; 29]),
        text(&[3, 14]),
        media(Modality::Image, 16, 4, 4),
        text(&[15, 4, 14]),
        media(Modality::Video, 8, 4, 2),
        text(&[15, 5, 14]),
        media(Modality::Video, 8, 4, 2),
        text(&[15, 5]),
        text(&[3; 3]),
    ];
    input::ModelInput::new(&parts).into()
}

fn trajectory(runtime: &mut ModelRuntime<MlxBackend<'_>>, owner: bool) -> Vec<Option<Vec<f32>>> {
    let first = runtime.prefill(prompt()).unwrap().wait().unwrap();
    assert_eq!(first.logits().is_some(), owner);
    let mut outputs = vec![values(first.logits())];
    for step in 0..16u32 {
        let output = runtime
            .decode(Array::from_slice(&[step % 12], &[1, 1]))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(output.logits().is_some(), owner);
        outputs.push(values(output.logits()));
    }
    outputs
}

pub(super) fn worker(
    checkpoint: &Path,
    topology: eredu_core::ParallelRankTopology,
    device: DeviceAssignment,
    group: &safemlx::distributed::Group,
    stream: &Stream,
    weights_stream: &Stream,
) {
    let started = Instant::now();
    let backend = crate::native::backend(stream, weights_stream);
    let ordinary = eredu_core::prepare_inspected_model(
        &backend,
        component_fixture_inspection(checkpoint),
        MlxLoadRequest::from_normalized(load_policy(LayerWeightResidency::FullyResident, 64)),
    )
    .unwrap();
    let mut ordinary = ModelRuntime::from_prepared(backend, ordinary).unwrap();
    let expected = trajectory(&mut ordinary, true);
    assert!(expected
        .iter()
        .flatten()
        .flatten()
        .any(|value| value.abs() > 1e-3));
    drop(ordinary);
    let layers = if std::env::var_os(DENSE_STREAM).is_some() {
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 25, 1 << 26, 1, 1).unwrap(),
        )
    } else if std::env::var_os(LAYERWISE_HOST).is_some() {
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(1 << 25), Some(1 << 26), 1).unwrap(),
        ))
    } else {
        LayerWeightResidency::FullyResident
    };
    let request = MlxLoadRequest::from_normalized(load_policy(layers, 32))
        .with_parallel_topology(
            topology,
            device,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            32,
            ring_completion_policy(),
        )
        .unwrap();
    let backend = crate::native::distributed_backend(stream, weights_stream, group);
    let model = eredu_core::prepare_inspected_model(
        &backend,
        component_fixture_inspection(checkpoint),
        request,
    )
    .unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let owner = topology.tensor_parallel_rank() == 0
        && topology.expert_parallel_rank() == 0
        && topology.pipeline_parallel_rank() + 1 == topology.pipeline_parallel_size();
    let actual = trajectory(&mut runtime, owner);
    for (actual, expected) in actual.iter().zip(&expected) {
        if owner {
            close(actual.as_ref().unwrap(), expected.as_ref().unwrap());
        } else {
            assert!(actual.is_none());
        }
    }
    let (_, session) = runtime.parts_mut();
    let banks = session.parameter_bank_report().unwrap();
    assert_eq!(
        banks.as_ref().and_then(|report| report.rows()).is_some(),
        topology.tensor_parallel_rank() == 0 && topology.pipeline_parallel_rank() == 0,
        "only the injection stage's tensor owner acquires lexical rows"
    );
    eprintln!("Flash-Next native media rank={} topology={:?} residency={layers:?}: prefill50+decode16 baseline64/distributed32 {:?}",
        topology.global_rank(), topology.topology(), started.elapsed());
}
