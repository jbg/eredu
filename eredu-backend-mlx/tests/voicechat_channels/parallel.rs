//! Guarded fused ingress uses the retained failure authority before vocabulary collectives.
use super::*;
use eredu_architectures::voicechat::backbone_execution::{BackboneInput, BackboneRequest};
use eredu_nn::VocabularyParallelRange;
use eredu_runtime::replicated_execution::ReplicatedCoordinator;

const WORKER: &str = "EREDU_CHANNEL_INGRESS_TP_WORKER";

#[test]
fn native_guarded_channel_ingress_worker() {
    if std::env::var_os(WORKER).is_none() {
        return;
    }
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), 2);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut coordinator = component_coordinator::coordinator(&world, &stream);
    let parallel = coordinator.parallel_group().clone();
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let range = VocabularyParallelRange {
        global_vocabulary: 24,
        local: world.rank() * 12..(world.rank() + 1) * 12,
    };
    let mut modules = StaticModules::<Backend>::from_parallel_spec(
        specification(&fixture),
        range.clone(),
        Some(range.clone()),
        &stream,
    )
    .unwrap();
    modules.visit_parameters_mut(&mut Bind {
        values: &fixture["parameters"],
        dtype: TensorElementType::F32,
        stream: &stream,
        rows: Some(range.local.start as i32..range.local.end as i32),
    });
    let case = &fixture["f32"];
    let audio = tensor(&case["audio_input_f32"]);
    let agent = ids(&case["agent_tokens"], &stream);
    let function = ids(&case["function_tokens"], &stream);
    let fusion = fusion();
    let mut prepare = |request| {
        BackboneInput::prepare_parallel_guarded::<Backend, _>(
            &fusion,
            &mut modules,
            request,
            &parallel,
            &stream,
            |descriptor, success, context| {
                coordinator
                    .agree(descriptor, success, context)
                    .map_err(ChannelError::from)
            },
        )
    };
    for _ in 0..2 {
        let fused = prepare(BackboneRequest::Audio {
            projected_audio: &audio,
            agent_tokens: &agent,
            function_tokens: &function,
        })
        .unwrap();
        compare(
            fused.embedded(),
            &case["fused_input"],
            &fixture["tolerance"],
            false,
            &stream,
        );
    }
    for prompt in fixture["prompts"].as_array().unwrap() {
        let fused = prepare(BackboneRequest::Prompt {
            tokens: &[5, 7],
            policy: PromptPolicy {
                repeat: prompt["repeat_n"].as_u64().unwrap() as usize,
                max_tokens: 8,
            },
        })
        .unwrap();
        compare(
            fused.embedded(),
            &prompt["fused_input"],
            &fixture["tolerance"],
            false,
            &stream,
        );
        assert_eq!(
            fused.tokens().to_i32_vec(&stream).unwrap(),
            prompt["prompt_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_i64().unwrap() as i32)
                .collect::<Vec<_>>()
        );
    }
    // One rank rejects feedback geometry before either rank enters lookup/reduction.
    let invalid_audio = MlxTensor::full_f32(0., &[1, 1, 5], &stream).unwrap();
    let malformed = if world.rank() == 0 {
        &invalid_audio
    } else {
        &audio
    };
    assert!(
        prepare(BackboneRequest::Audio {
            projected_audio: malformed,
            agent_tokens: &agent,
            function_tokens: &function,
        })
        .is_err()
    );
    // Prompt framing fails on one rank, before allocation can lead into lookup.
    assert!(
        prepare(BackboneRequest::Prompt {
            tokens: &[5, 7],
            policy: PromptPolicy {
                repeat: 2,
                max_tokens: if world.rank() == 0 { 7 } else { 8 }
            },
        })
        .is_err()
    );
    // Valid peers also reject a distinct invocation; equal tensor payloads cannot
    // allow prompt and live-audio collective sequences to cross.
    let mismatched = if world.rank() == 0 {
        BackboneRequest::Prompt {
            tokens: &[5, 7],
            policy: PromptPolicy {
                repeat: 1,
                max_tokens: 8,
            },
        }
    } else {
        BackboneRequest::Audio {
            projected_audio: &audio,
            agent_tokens: &agent,
            function_tokens: &function,
        }
    };
    assert!(prepare(mismatched).is_err());
    // Reusing the retained authority after agreed rejection cannot leave an
    // unmatched lookup collective that corrupts the next valid request.
    let retry = prepare(BackboneRequest::Audio {
        projected_audio: &audio,
        agent_tokens: &agent,
        function_tokens: &function,
    })
    .unwrap();
    compare(
        retry.embedded(),
        &case["fused_input"],
        &fixture["tolerance"],
        false,
        &stream,
    );
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_guarded_channel_ingress_tp2_matches_reference_and_rejects_peer_failures() {
    use std::{
        net::TcpListener,
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };
    struct Children(Vec<Child>);
    impl Drop for Children {
        fn drop(&mut self) {
            for child in &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let sockets = (0..2)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let mut children = Children(Vec::new());
    for rank in 0..2 {
        children.0.push(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "parallel::native_guarded_channel_ingress_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(WORKER, "1")
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        if Instant::now() >= deadline || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                let _ = child.kill();
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut succeeded = true;
    let mut diagnostics = String::new();
    for (rank, child) in children.0.drain(..).enumerate() {
        let output = child.wait_with_output().unwrap();
        succeeded &= output.status.success();
        diagnostics.push_str(&format!(
            "rank {rank}: {}\n{}\n{}\n",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    assert!(succeeded, "guarded ingress workers failed:\n{diagnostics}");
}
