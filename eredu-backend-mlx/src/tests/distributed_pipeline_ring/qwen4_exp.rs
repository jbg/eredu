//! Ordinary Flash-Next source binding through the existing local Ring launcher.
use super::*;
mod encoded;
mod media;
mod prediction;

const PROMPT: [u32; 19] = [3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const ABS_TOLERANCE: f32 = 2e-4;
const REL_TOLERANCE: f32 = 2e-4;

fn run(gguf: bool, axes: &'static str, residency: WorkerResidency) {
    assert!(distributed::is_available(Backend::Ring));
    let directory = tempfile::tempdir().unwrap();
    let fixture = if axes == "tp4" {
        eredu_evaluation::qwen4_exp::Fixture::tensor_parallel_four()
            .prepare(directory.path())
            .unwrap()
    } else {
        eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap()
    };
    if axes == "tp4" {
        for rank in 0..4 {
            let local = fixture.safetensors.tensor_partition(rank, 4).unwrap();
            for projection in ["k_proj", "v_proj"] {
                assert_eq!(
                    local.partition().parameter_selections(&format!(
                        "model.layers.1.self_attn.{projection}.weight"
                    )),
                    Some(
                        [eredu_checkpoint::store::TensorSelection::Range {
                            axis: 0,
                            start: rank / 2 * 8,
                            end: (rank / 2 + 1) * 8,
                        }]
                        .as_slice()
                    ),
                    "two tensor ranks share each K/V head"
                );
            }
        }
    }
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
fn ring_two_process_qwen4_exp_tensor_safetensors_resident() {
    run(false, "tp", WorkerResidency::FullyResident);
}

#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_four_process_qwen4_exp_tp4_safetensors_resident() {
    run(false, "tp4", WorkerResidency::FullyResident);
}

#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_four_process_qwen4_exp_tp4_gguf_disk() {
    run(true, "tp4", WorkerResidency::DenseDiskStream);
}

#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_two_process_qwen4_exp_tensor_gguf_host() {
    run(true, "tp", WorkerResidency::LayerwiseHost);
}

#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_two_process_qwen4_exp_pipeline_safetensors_disk() {
    run(false, "pp", WorkerResidency::DenseDiskStream);
}

#[test]
#[ignore = "requires two local MLX Ring ranks and loopback sockets"]
fn ring_two_process_qwen4_exp_pipeline_gguf_resident() {
    run(true, "pp", WorkerResidency::FullyResident);
}

#[test]
#[ignore = "requires four local MLX Ring ranks and loopback sockets"]
fn ring_four_process_qwen4_exp_tensor_pipeline_safetensors_host() {
    run(false, "tp-pp", WorkerResidency::LayerwiseHost);
}

#[test]
#[ignore = "requires 2 local MLX Ring ranks and loopback sockets"]
fn ring_2_process_qwen4_exp_ep_safetensors_host() {
    run(false, "ep", WorkerResidency::LayerwiseHost);
}

#[test]
#[ignore = "requires 2 local MLX Ring ranks and loopback sockets"]
fn ring_2_process_qwen4_exp_ep_gguf_disk() {
    run(true, "ep", WorkerResidency::DenseDiskStream);
}

#[test]
#[ignore = "requires 4 local MLX Ring ranks and loopback sockets"]
fn ring_4_process_qwen4_exp_tp_ep_safetensors_host() {
    run(false, "tp-ep", WorkerResidency::LayerwiseHost);
}

#[test]
#[ignore = "requires 4 local MLX Ring ranks and loopback sockets"]
fn ring_4_process_qwen4_exp_tp_ep_gguf_disk() {
    run(true, "tp-ep", WorkerResidency::DenseDiskStream);
}

#[test]
#[ignore = "requires 4 local MLX Ring ranks and loopback sockets"]
fn ring_4_process_qwen4_exp_tp_ep_resident_safetensors() {
    run(false, "tp-ep", WorkerResidency::FullyResident);
}

#[test]
#[ignore = "requires 4 local MLX Ring ranks and loopback sockets"]
fn ring_4_process_qwen4_exp_tp_ep_resident_gguf() {
    run(true, "tp-ep", WorkerResidency::FullyResident);
}

#[test]
#[ignore = "requires 4 local MLX Ring ranks and loopback sockets"]
fn ring_4_process_qwen4_exp_pp_ep_safetensors_host() {
    run(false, "pp-ep", WorkerResidency::LayerwiseHost);
}

#[test]
#[ignore = "requires 4 local MLX Ring ranks and loopback sockets"]
fn ring_4_process_qwen4_exp_pp_ep_gguf_disk() {
    run(true, "pp-ep", WorkerResidency::DenseDiskStream);
}

#[test]
#[ignore = "requires 8 local MLX Ring ranks and loopback sockets"]
fn ring_8_process_qwen4_exp_tp_pp_ep_safetensors_host() {
    run(false, "tp-pp-ep", WorkerResidency::LayerwiseHost);
}

#[test]
#[ignore = "requires 8 local MLX Ring ranks and loopback sockets"]
fn ring_8_process_qwen4_exp_tp_pp_ep_gguf_disk() {
    run(true, "tp-pp-ep", WorkerResidency::DenseDiskStream);
}

fn input(tokens: &[u32]) -> MlxModelInput {
    let tokens = Array::from_slice(tokens, &[1, tokens.len() as i32]);
    let parts = [crate::backend::runtime::media::input::token_ids_part(&tokens).unwrap()];
    crate::backend::runtime::media::input::ModelInput::new(&parts).into()
}

fn values(value: Option<&MlxTensor>) -> Option<Vec<f32>> {
    value.map(|value| {
        assert_eq!(value.as_array().shape(), &[1, 16]);
        assert_eq!(value.as_array().dtype(), MlxDtype::Float32);
        let values = value.as_array().evaluated().unwrap();
        let values = values.as_slice::<f32>().to_vec();
        assert!(values.iter().all(|value| value.is_finite()));
        values
    })
}

fn trajectory(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    chunks: &[usize],
    owner: bool,
) -> Vec<Option<Vec<f32>>> {
    let mut offset = 0;
    let mut last = None;
    for &length in chunks {
        let output = runtime
            .prefill(input(&PROMPT[offset..offset + length]))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(output.logits().is_some(), owner);
        last = values(output.logits());
        offset += length;
    }
    assert_eq!(offset, PROMPT.len());
    let mut result = vec![last];
    for token in 0..16u32 {
        let output = runtime
            .decode(Array::from_slice(&[token], &[1, 1]))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(output.logits().is_some(), owner);
        result.push(values(output.logits()));
    }
    result
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= ABS_TOLERANCE + REL_TOLERANCE * expected.abs(),
            "scalar {index}: {actual} != {expected}"
        );
    }
}

pub(super) fn worker(
    checkpoint: &Path,
    cache_root: &Path,
    topology: eredu_core::ParallelRankTopology,
    device: DeviceAssignment,
    group: &safemlx::distributed::Group,
    stream: &Stream,
    weights_stream: &Stream,
) {
    if prediction::is_fixture(checkpoint) {
        return prediction::worker(checkpoint, topology, device, group, stream, weights_stream);
    }
    if media::is_fixture(checkpoint) {
        return media::worker(checkpoint, topology, device, group, stream, weights_stream);
    }
    // Every process computes an ordinary baseline without collectives. Alternate
    // exports use their matching canonical weights to validate conversion too.
    let backend = crate::native::backend(stream, weights_stream);
    let baseline = if topology.tensor_parallel_size() == 4
        && checkpoint.is_file()
        && !encoded::is_fixture(checkpoint)
    {
        // Validate the independent GGUF recurrent export/reorder against its
        // canonical SafeTensors source as well as single-rank execution.
        checkpoint.parent().unwrap().join("safetensors")
    } else {
        encoded::baseline_path(checkpoint)
    };
    let model = load_model(&backend, &baseline, MlxLoadRequest::default()).unwrap();
    let mut ordinary = ModelRuntime::from_prepared(backend, model).unwrap();
    let expected = trajectory(&mut ordinary, &[PROMPT.len()], true);
    drop(ordinary);
    if topology.tensor_parallel_size() != 4 && !encoded::is_fixture(checkpoint) {
        let golden: serde_json::Value =
            serde_json::from_str(eredu_evaluation::qwen4_exp::DENSE_TRAJECTORY_JSON).unwrap();
        let golden: Vec<Vec<f32>> = serde_json::from_value(golden["logits"].clone()).unwrap();
        for (expected, golden) in expected.iter().zip(&golden) {
            close(expected.as_ref().unwrap(), golden);
        }
    }

    let layers = if std::env::var_os(DENSE_STREAM).is_some() {
        eredu_runtime::LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 25, 1 << 26, 1, 1).unwrap(),
        )
    } else if std::env::var_os(LAYERWISE_HOST).is_some() {
        eredu_runtime::LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(1 << 25), Some(1 << 26), 1).unwrap(),
        ))
    } else {
        eredu_runtime::LayerWeightResidency::FullyResident
    };
    let weights = match layers {
        eredu_runtime::LayerWeightResidency::FullyResident => WeightResidency::with_layers(layers),
        _ => {
            let ordinary = match layers {
                eredu_runtime::LayerWeightResidency::LayerwiseHost(options) => {
                    eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(options)
                }
                eredu_runtime::LayerWeightResidency::DenseDiskStream(options) => {
                    eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(options)
                }
                _ => unreachable!(),
            };
            WeightResidency::with_independent_parameter_banks(
                ordinary,
                eredu_runtime::ParameterBankLoadOptions::new(
                    OffloadConfig::new(Some(1 << 25), Some(1 << 26), 1).unwrap(),
                    1 << 25,
                    1 << 24,
                )
                .unwrap(),
            )
        }
    };
    let load = eredu_runtime::NormalizedLoadRequest::default()
        .with_weight_residency(weights)
        .with_state_residency(CacheResidencyPolicy::Paged(
            PagedCacheOptions::new(2, 1 << 23, 1 << 23, 1)
                .unwrap()
                .with_full_attention(true),
        ))
        .with_required_session_capabilities(eredu_core::SessionCapabilities::new(true, true, true))
        .with_prompt_cache_persistence(true);
    let request = MlxLoadRequest::from_normalized(load)
        .with_parallel_topology(
            topology,
            device,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            64,
            ring_completion_policy(),
        )
        .unwrap();
    let backend = crate::native::distributed_backend(stream, weights_stream, group);
    let model = load_model(&backend, checkpoint, request).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let owner = topology.tensor_parallel_rank() == 0
        && topology.expert_parallel_rank() == 0
        && topology.pipeline_parallel_rank() + 1 == topology.pipeline_parallel_size();
    let actual = trajectory(&mut runtime, &[3, 5, 11], owner);
    for (actual, expected) in actual.iter().zip(&expected) {
        if owner {
            close(actual.as_ref().unwrap(), expected.as_ref().unwrap());
        } else {
            assert!(actual.is_none());
        }
    }
    // Persist all native recurrent, convolution, index-summary, token-ID and KV
    // components; restoring must reproduce a cached decode bit for bit per rank.
    let prefix: Vec<u32> = PROMPT.into_iter().chain(0..16u32).collect();
    let (backend, session) = runtime.parts_mut();
    let banks = session.parameter_bank_report().unwrap();
    assert_eq!(
        banks.as_ref().and_then(|report| report.rows()).is_some(),
        topology.tensor_parallel_rank() == 0 && topology.pipeline_parallel_rank() == 0,
        "only the stage owning the fixture's injection acquires table rows"
    );
    if !matches!(layers, eredu_runtime::LayerWeightResidency::FullyResident) {
        let banks = banks.as_ref().expect("independent expert bank report");
        let experts = eredu_core::balanced_contiguous_range(
            3,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .unwrap();
        assert!(!banks.banks().is_empty());
        for bank in banks.banks().values() {
            let members: std::collections::BTreeSet<_> = bank
                .placements()
                .iter()
                .map(|(key, _)| key.member())
                .collect();
            assert_eq!(
                members,
                experts.clone().collect(),
                "cache retains only owned expert members"
            );
            assert!(bank.bulk().requested_selections() > 0);
            assert!(bank.incremental().requested_selections() > 0);
        }
    }
    let descriptor = PromptCacheDescriptor::from_model_identity(
        session.prompt_cache_model_identity().unwrap(),
        "qwen4-ring",
        "original-ids",
        1,
    )
    .unwrap();
    let path = cache_root.join(format!("rank-{}", topology.global_rank()));
    session
        .save_prompt_cache(
            backend,
            &path,
            descriptor.clone(),
            &prefix,
            &PromptCacheOptions::default(),
        )
        .unwrap();
    let token = Array::from_slice(&[7u32], &[1, 1]);
    let first = session
        .decode(backend, token.clone())
        .unwrap()
        .wait()
        .unwrap();
    let first = values(first.logits());
    session
        .load_prompt_cache(backend, &path, &descriptor, &prefix)
        .unwrap();
    let replay = session.decode(backend, token).unwrap().wait().unwrap();
    assert_eq!(first, values(replay.logits()));
}
