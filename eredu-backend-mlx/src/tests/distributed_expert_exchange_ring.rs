#![cfg(unix)]

use std::{
    net::TcpListener,
    process::{Child, Command, Output, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use crate::native::{
    distributed::{self, Backend},
    ops::concatenate_axis,
    transforms::{async_eval_with_event, eval},
    Array, Device, DeviceType, Stream,
};
use crate::{
    backend::error::Error,
    backend::nn::moe::{PackedGatedProductExperts, PackedRelu2Experts},
    backend::runtime::{
        distributed::expert::{
            dispatch_replicated_with, dispatch_sharded, profile_expert_parallel_timings,
            AllToAllVPlan, DispatchedRoutes, ExpertAssignment, LocalExpertBank, RoutedTransport,
            ShardedRouteBlocks,
        },
        residency::expert_cache::{ExpertCache, ExpertCatalogEntry, ExpertRouteBatch},
    },
    module::Param,
};
use eredu_checkpoint::store::{SafetensorsWeightStore, TensorSelection};
use eredu_core::{residency::OffloadConfig, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    ExpertCacheLoadOptions, ExpertIdentity, ExpertPass, OffloadUnit, WeightBinding,
};
use safetensors::tensor::{serialize_to_file, Dtype as TensorDtype, TensorView};

const WORKER_RANK: &str = "EREDU_EXPERT_EXCHANGE_RING_WORKER";
const PAYLOAD_FILE: &str = "EREDU_EXPERT_EXCHANGE_PAYLOAD";

fn balanced_assignment(global_experts: usize, rank: usize) -> ExpertAssignment {
    let topology = ParallelTopology::new(1, 1, 2, 1).unwrap();
    let rank = ParallelRankTopology::new(topology, rank).unwrap();
    let group = eredu_runtime::ExecutionGroupId::new("test").unwrap();
    let plan = eredu_architectures::ExpertRealizationPlan::balanced(
        global_experts,
        rank,
        std::collections::BTreeMap::from([((group, 0), ())]),
    )
    .unwrap();
    ExpertAssignment::from_realization(&plan).unwrap()
}

fn f32_array(values: &[f32], shape: &[i32], stream: &Stream) -> Array {
    Array::from_slice(values, shape).copy(stream).unwrap()
}

fn i32_array(values: &[i32], shape: &[i32], stream: &Stream) -> Array {
    Array::from_slice(values, shape).copy(stream).unwrap()
}

fn assert_f32_close(actual: &Array, expected: &[f32]) {
    eval([actual]).unwrap();
    let actual = actual.evaluated().unwrap();
    assert_eq!(actual.as_slice::<f32>().len(), expected.len());
    for (index, (actual, expected)) in actual.as_slice::<f32>().iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "dispatch output {index} was {actual}, expected {expected}"
        );
    }
}

fn full_dispatch_blocks(rank: usize, stream: &Stream) -> ShardedRouteBlocks {
    let (hidden, global_expert_ids, original_route_indices, weights) = if rank == 0 {
        (
            [vec![2.0, 1.0], vec![1.0, 2.0]],
            [vec![1, 0], vec![3, 2]],
            [vec![3, 1], vec![0, 2]],
            [vec![0.2, 0.25], vec![0.5, 0.1]],
        )
    } else {
        (
            [vec![4.0, 3.0], vec![3.0, 4.0]],
            [vec![0, 1], vec![2, 3]],
            [vec![2, 0], vec![1, 3]],
            [vec![0.25, 0.2], vec![0.5, 0.1]],
        )
    };
    ShardedRouteBlocks {
        hidden: hidden
            .iter()
            .map(|values| f32_array(values, &[2, 1], stream))
            .collect(),
        global_expert_ids: global_expert_ids
            .iter()
            .map(|values| i32_array(values, &[2], stream))
            .collect(),
        original_route_indices: original_route_indices
            .iter()
            .map(|values| i32_array(values, &[2], stream))
            .collect(),
        weights: weights
            .iter()
            .map(|values| f32_array(values, &[2], stream))
            .collect(),
        top_k: 2,
        source_tokens: 2,
    }
}

fn relu2_bank(stream: &Stream) -> PackedRelu2Experts {
    let mut bank = PackedRelu2Experts::new(2, 1, 1, [None, None], stream).unwrap();
    bank.up_proj = Param::new(f32_array(&[1.0, 2.0], &[2, 1, 1], stream));
    bank.down_proj = Param::new(f32_array(&[1.0, 10.0], &[2, 1, 1], stream));
    bank
}

struct ScaledGatedProductExperts;

impl LocalExpertBank for ScaledGatedProductExperts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let shape = hidden.shape().to_vec();
        eval([hidden, local_expert_ids])?;
        let hidden = hidden.evaluated()?;
        let ids = local_expert_ids.evaluated()?;
        let values = hidden.as_slice::<f32>();
        let ids = ids.as_slice::<i32>();
        let output = values
            .iter()
            .zip(ids)
            .map(|(hidden, expert)| {
                let scale = if *expert == 0 { 1.0 } else { 2.0 };
                let gate = scale * hidden;
                let silu = gate / (1.0 + (-gate).exp());
                silu * (scale * hidden) * scale
            })
            .collect::<Vec<_>>();
        Ok(Array::from_slice(&output, &shape).copy(stream)?)
    }
}

fn scaled_route_output(hidden: f32, local_expert: usize, weight: f32) -> f32 {
    let scale = if local_expert == 0 { 1.0 } else { 2.0 };
    let gate = scale * hidden;
    let silu = gate / (1.0 + (-gate).exp());
    weight * silu * (scale * hidden) * scale
}

fn execute_cached_qwen_routes(
    cache: &ExpertCache,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(cache.execute_routes_bounded(
        ExpertRouteBatch::new(
            0,
            &routes.hidden,
            &routes.global_expert_ids,
            &routes.weights,
            pass,
        ),
        stream,
        |hidden, acquired, _weights, stream| {
            let started = Instant::now();
            let mut bank = PackedGatedProductExperts {
                policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
                num_experts: acquired.identities().len() as i32,
                hidden_dim: 1,
                intermediate_dim: 1,
                gate_up_affine: None,
                down_affine: None,
                gate_up_iquant: None,
                down_iquant: None,
                native_fp8_e8m0: false,
                gate_up_proj: Param::new(acquired.compact_binding("gate_up_proj", stream)?),
                gate_up_proj_bias: Param::new(None),
                gate_up_proj_scales: Param::new(None),
                gate_up_proj_biases: Param::new(None),
                down_proj: Param::new(acquired.compact_binding("down_proj", stream)?),
                down_proj_bias: Param::new(None),
                down_proj_scales: Param::new(None),
                down_proj_biases: Param::new(None),
            };
            cache.record_compact_bank(pass, acquired.scratch_bytes(), started.elapsed())?;
            let compact_routes = acquired.compact_routes().reshape(&[-1, 1], stream)?;
            let unit_weights =
                crate::native::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
            Ok(bank.forward(hidden, &compact_routes, &unit_weights, stream)?)
        },
    )?)
}

#[test]
fn expert_exchange_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let group = crate::backend::runtime::distributed::Group::native(
        &distributed::init(true, Backend::Ring).unwrap(),
    );
    assert_eq!(group.rank(), expected_rank);
    assert_eq!(group.size(), 2);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let arrays = Array::load_safetensors(std::env::var_os(PAYLOAD_FILE).unwrap(), &stream).unwrap();
    let blocks = (0..2)
        .map(|destination| arrays[&format!("r{expected_rank}d{destination}")].clone())
        .collect::<Vec<_>>();
    let _profiling = profile_expert_parallel_timings();
    let send_counts = blocks
        .iter()
        .map(|block| block.dim(0) as usize)
        .collect::<Vec<_>>();
    let block_refs = blocks.iter().collect::<Vec<_>>();
    let compact = concatenate_axis(&block_refs, 0, &stream).unwrap();
    let plan = AllToAllVPlan::new(&send_counts, &group, &stream).unwrap();
    let exchanged = plan.exchange(&compact, &group, &stream).unwrap();
    eval([&exchanged.received]).unwrap();
    let received = exchanged.received.evaluated().unwrap();
    if expected_rank == 0 {
        assert_eq!(exchanged.source_counts, vec![1, 0]);
        assert_eq!(received.as_slice::<i32>(), &[10]);
    } else {
        assert_eq!(exchanged.source_counts, vec![2, 1]);
        assert_eq!(received.as_slice::<i32>(), &[11, 12, 21]);
    }
    assert_eq!(exchanged.statistics.padding_routes, 0);
    assert_eq!(exchanged.statistics.padding_bytes, 0);
    assert_eq!(
        exchanged.statistics.useful_sent_bytes,
        12 - expected_rank * 8
    );
    assert_eq!(
        exchanged.statistics.routed_transport,
        RoutedTransport::Native
    );
    assert_eq!(plan.recv_counts(), exchanged.source_counts);
    assert!(exchanged.statistics.payload_exchange_time > Duration::ZERO);

    let assignment = balanced_assignment(4, expected_rank);
    let mut relu2 = relu2_bank(&stream);
    let dispatched = dispatch_sharded(
        full_dispatch_blocks(expected_rank, &stream),
        &assignment,
        &mut relu2,
        &group,
        &stream,
    )
    .unwrap();
    if expected_rank == 0 {
        assert_f32_close(&dispatched.output, &[20.25, 32.4]);
    } else {
        assert_f32_close(&dispatched.output, &[76.5, 68.0]);
    }
    assert_eq!(dispatched.statistics.total_routes, 4);
    assert_eq!(dispatched.statistics.sent_routes, 4);
    assert_eq!(dispatched.statistics.received_routes, 4);
    assert_eq!(dispatched.statistics.count_consensus_count, 1);
    assert_eq!(dispatched.statistics.padding_routes, 0);
    assert_eq!(dispatched.statistics.padding_bytes, 0);
    assert_eq!(
        dispatched.statistics.routed_transport,
        RoutedTransport::Native
    );

    let empty_hidden = f32_array(&[], &[0, 1], &stream);
    let empty_i32 = i32_array(&[], &[0], &stream);
    let empty_f32 = f32_array(&[], &[0], &stream);
    let empty_blocks = if expected_rank == 0 {
        ShardedRouteBlocks {
            hidden: vec![empty_hidden.clone(), empty_hidden.clone()],
            global_expert_ids: vec![empty_i32.clone(), empty_i32.clone()],
            original_route_indices: vec![empty_i32.clone(), empty_i32.clone()],
            weights: vec![empty_f32.clone(), empty_f32.clone()],
            top_k: 2,
            source_tokens: 1,
        }
    } else {
        ShardedRouteBlocks {
            hidden: vec![empty_hidden, f32_array(&[2.0], &[1, 1], &stream)],
            global_expert_ids: vec![empty_i32.clone(), i32_array(&[2], &[1], &stream)],
            original_route_indices: vec![empty_i32, i32_array(&[1], &[1], &stream)],
            weights: vec![empty_f32, f32_array(&[0.5], &[1], &stream)],
            top_k: 2,
            source_tokens: 1,
        }
    };
    let empty_dispatched =
        dispatch_sharded(empty_blocks, &assignment, &mut relu2, &group, &stream).unwrap();
    assert_f32_close(
        &empty_dispatched.output,
        if expected_rank == 0 { &[0.0] } else { &[2.0] },
    );
    assert_eq!(
        empty_dispatched.statistics.total_routes,
        usize::from(expected_rank == 1)
    );
    assert_eq!(
        empty_dispatched.statistics.received_routes,
        usize::from(expected_rank == 1)
    );
    assert_eq!(empty_dispatched.statistics.count_consensus_count, 1);
    assert_eq!(empty_dispatched.statistics.padding_routes, 0);

    let mut scaled = ScaledGatedProductExperts;
    let scaled_dispatched = dispatch_sharded(
        full_dispatch_blocks(expected_rank, &stream),
        &assignment,
        &mut scaled,
        &group,
        &stream,
    )
    .unwrap();
    let expected = if expected_rank == 0 {
        [
            scaled_route_output(1.0, 1, 0.5) + scaled_route_output(1.0, 0, 0.25),
            scaled_route_output(2.0, 0, 0.1) + scaled_route_output(2.0, 1, 0.2),
        ]
    } else {
        [
            scaled_route_output(3.0, 1, 0.2) + scaled_route_output(3.0, 0, 0.5),
            scaled_route_output(4.0, 0, 0.25) + scaled_route_output(4.0, 1, 0.1),
        ]
    };
    assert_f32_close(&scaled_dispatched.output, &expected);
    assert_eq!(scaled_dispatched.statistics.total_routes, 4);
    assert_eq!(scaled_dispatched.statistics.sent_routes, 4);
    assert_eq!(scaled_dispatched.statistics.received_routes, 4);
    assert_eq!(scaled_dispatched.statistics.count_consensus_count, 1);
    assert_eq!(scaled_dispatched.statistics.padding_bytes, 0);

    let qwen_gate_up = [1.0f32, 1.0, 2.0, 1.0, 1.0, 2.0, 0.5, 3.0];
    let qwen_down = [1.0f32, 1.5, 2.0, 0.5];
    let qwen_hidden = f32_array(&[1.0, 2.0], &[2, 1], &stream);
    let qwen_ids = i32_array(&[0, 1, 2, 3], &[2, 2], &stream);
    let qwen_weights = f32_array(&[0.25, 0.75, 0.4, 0.6], &[2, 2], &stream);
    let mut full_qwen = PackedGatedProductExperts {
        policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
        num_experts: 4,
        hidden_dim: 1,
        intermediate_dim: 1,
        gate_up_affine: None,
        down_affine: None,
        gate_up_iquant: None,
        down_iquant: None,
        native_fp8_e8m0: false,
        gate_up_proj: Param::new(f32_array(&qwen_gate_up, &[4, 2, 1], &stream)),
        gate_up_proj_bias: Param::new(None),
        gate_up_proj_scales: Param::new(None),
        gate_up_proj_biases: Param::new(None),
        down_proj: Param::new(f32_array(&qwen_down, &[4, 1, 1], &stream)),
        down_proj_bias: Param::new(None),
        down_proj_scales: Param::new(None),
        down_proj_biases: Param::new(None),
    };
    let expected_qwen = full_qwen
        .forward(&qwen_hidden, &qwen_ids, &qwen_weights, &stream)
        .unwrap();
    let qwen_assignment = balanced_assignment(4, expected_rank);
    let store =
        Arc::new(SafetensorsWeightStore::open(std::env::var_os(PAYLOAD_FILE).unwrap()).unwrap());
    let entries = qwen_assignment
        .local_global_expert_ids()
        .iter()
        .copied()
        .map(|expert| {
            let identity = ExpertIdentity::new(0, expert);
            let bindings = [
                WeightBinding::new(
                    "gate_up_proj",
                    "qwen_gate_up",
                    TensorSelection::Range {
                        axis: 0,
                        start: expert,
                        end: expert + 1,
                    },
                    8,
                )
                .unwrap(),
                WeightBinding::new(
                    "down_proj",
                    "qwen_down",
                    TensorSelection::Range {
                        axis: 0,
                        start: expert,
                        end: expert + 1,
                    },
                    4,
                )
                .unwrap(),
            ];
            let unit = OffloadUnit::new(identity.unit_id(), bindings).unwrap();
            ExpertCatalogEntry::new(identity, unit, 12).unwrap()
        })
        .collect::<Vec<_>>();
    let cache = ExpertCache::new(
        store,
        entries,
        ExpertCacheLoadOptions::new(OffloadConfig::new(Some(24), Some(24), 1).unwrap(), 24, 24)
            .unwrap(),
        stream.clone(),
        stream.clone(),
    )
    .unwrap();
    let cached_qwen = dispatch_replicated_with(
        &qwen_hidden,
        &qwen_ids,
        &qwen_weights,
        &qwen_assignment,
        &group,
        &stream,
        |routes, stream| execute_cached_qwen_routes(&cache, routes, ExpertPass::Prefill, stream),
    )
    .unwrap();
    async_eval_with_event([&expected_qwen, &cached_qwen.reduced_output])
        .unwrap()
        .synchronize()
        .unwrap();
    let expected_qwen = expected_qwen.evaluated().unwrap();
    assert_f32_close(&cached_qwen.reduced_output, expected_qwen.as_slice::<f32>());

    let cached_decode = dispatch_replicated_with(
        &qwen_hidden,
        &qwen_ids,
        &qwen_weights,
        &qwen_assignment,
        &group,
        &stream,
        |routes, stream| execute_cached_qwen_routes(&cache, routes, ExpertPass::Decode, stream),
    )
    .unwrap();
    assert_f32_close(
        &cached_decode.reduced_output,
        expected_qwen.as_slice::<f32>(),
    );
    let report = cache.report().unwrap();
    assert_eq!(report.owned_experts, 2);
    assert_eq!(report.prefill.distinct_experts, 2);
    assert_eq!(report.decode.distinct_experts, 2);
    assert_eq!(report.decode.device.hits, 2);
}

struct ChildGuard(Vec<Child>);

impl ChildGuard {
    fn finish(mut self) -> Vec<Output> {
        self.0
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
        }
        for child in &mut self.0 {
            let _ = child.wait();
        }
    }
}

fn reserve_ports() -> (TcpListener, TcpListener, u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let second = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let first_port = first.local_addr().unwrap().port();
    let second_port = second.local_addr().unwrap().port();
    (first, second, first_port, second_port)
}

/// Run with:
/// `cargo test -p eredu-backend-mlx --lib tests::distributed_expert_exchange_ring::ring_two_process_all_to_all_v_and_dispatch_sharded -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_all_to_all_v_and_dispatch_sharded() {
    assert!(distributed::is_available(Backend::Ring));
    let fixture = tempfile::tempdir().unwrap();
    let payload = fixture.path().join("payload.safetensors");
    let i32_bytes = |values: &[i32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let r0d0 = i32_bytes(&[10]);
    let r0d1 = i32_bytes(&[11, 12]);
    let r1d0 = Vec::<u8>::new();
    let r1d1 = i32_bytes(&[21]);
    let f32_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let qwen_gate_up = f32_bytes(&[1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.5, 3.0]);
    let qwen_down = f32_bytes(&[1.0, 1.5, 2.0, 0.5]);
    serialize_to_file(
        [
            (
                "r0d0",
                TensorView::new(TensorDtype::I32, vec![1, 1], &r0d0).unwrap(),
            ),
            (
                "r0d1",
                TensorView::new(TensorDtype::I32, vec![2, 1], &r0d1).unwrap(),
            ),
            (
                "r1d0",
                TensorView::new(TensorDtype::I32, vec![0, 1], &r1d0).unwrap(),
            ),
            (
                "r1d1",
                TensorView::new(TensorDtype::I32, vec![1, 1], &r1d1).unwrap(),
            ),
            (
                "qwen_gate_up",
                TensorView::new(TensorDtype::F32, vec![4, 2, 1], &qwen_gate_up).unwrap(),
            ),
            (
                "qwen_down",
                TensorView::new(TensorDtype::F32, vec![4, 1, 1], &qwen_down).unwrap(),
            ),
        ],
        None,
        &payload,
    )
    .unwrap();
    let (first, second, first_port, second_port) = reserve_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    drop(first);
    drop(second);
    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard(Vec::with_capacity(2));
    for rank in 0..2 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "tests::distributed_expert_exchange_ring::expert_exchange_ring_worker",
                    "--nocapture",
                ])
                .env(WORKER_RANK, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env(PAYLOAD_FILE, &payload)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| {
            format!(
                "exchange Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process all-to-all-v and sharded dispatch failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}
