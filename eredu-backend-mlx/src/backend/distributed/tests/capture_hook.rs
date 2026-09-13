use super::*;
use eredu_runtime::capture::partition::{PartitionCaptureHookTransport, PartitionCaptureTransport};
use std::{
    net::TcpListener,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const WORKER: &str = "EREDU_CAPTURE_HOOK_RING_RANK";

#[test]
fn worker() {
    let Ok(rank) = std::env::var(WORKER) else {
        return;
    };
    let rank = rank.parse::<usize>().unwrap();
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stage = rank / 2;
    let members = vec![stage * 2, stage * 2 + 1];
    let groups = vec![eredu_runtime::CommunicationGroupDescriptor::new(
        CollectiveGroupId::new(stage as u32 + 61),
        0,
        members,
        Some(rank % 2),
        eredu_runtime::CommunicationGroupRequirements::new([
            eredu_runtime::CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap(),
    )
    .unwrap()];
    let manifest = eredu_runtime::CommunicationManifest::new(4, rank, groups, vec![])
        .unwrap()
        .with_completion_policy(
            eredu_runtime::CommunicationCompletionPolicy::new(
                Duration::from_secs(10),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let session = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
    let check_identity = |session: &MlxDistributedSession| {
        let words = session
            .session_identity()
            .bytes()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        let BoundedSubmissionOutcome::Completed(output) = session
            .submit_all_gather_words(&words)
            .unwrap()
            .wait_bounded(session.capture_wait().unwrap())
            .unwrap()
        else {
            panic!("setup identity agreement exceeded deadline")
        };
        let gathered = session.resolve_all_gather_words(output).unwrap();
        assert_eq!(gathered.len(), 4 * words.len());
        for peer in gathered.chunks_exact(words.len()) {
            assert_eq!(peer, words);
        }
    };
    check_identity(&session);
    let repeated = MlxDistributedSession::from_manifest(&manifest, &world, &stream).unwrap();
    assert_ne!(session.session_identity(), repeated.session_identity());
    assert_eq!(
        session.session_identity(),
        session.clone().session_identity()
    );
    check_identity(&repeated);
    for stage in 0..2 {
        assert_eq!(
            session
                .selected_group(CollectiveGroupId::new(stage as u32 + 61))
                .is_some(),
            rank / 2 == stage,
            "retained remote facts must not realize remote groups"
        );
    }
    assert!(session.estimate_capture_hook(&[0, 2]).is_err());
    for (phase, members) in [vec![0, 1], vec![2, 3]].into_iter().enumerate() {
        let bound = session.estimate_capture_hook(&members).unwrap();
        assert!(bound.retained_bytes > 0 && bound.host_bytes > 0);
        for unanimous in [true, false, true] {
            if members.contains(&rank) {
                let submitted = session
                    .submit_capture_hook(&members, unanimous || rank == members[0])
                    .unwrap();
                assert!(submitted.completion.retained_arrays() >= 2);
                let BoundedSubmissionOutcome::Completed(output) = submitted
                    .wait_bounded(session.capture_wait().unwrap())
                    .unwrap()
                else {
                    panic!("active-stage vote exceeded deadline")
                };
                assert_eq!(session.resolve_capture_hook(output).unwrap(), unanimous);
            }
            // Inactive ranks may reach this common boundary first. They never
            // enter the active stage's status collective or native tensor work.
            let BoundedSubmissionOutcome::Completed(output) = session
                .submit_all_gather_words(&[phase as u32])
                .unwrap()
                .wait_bounded(session.capture_wait().unwrap())
                .unwrap()
            else {
                panic!("common boundary exceeded deadline")
            };
            assert_eq!(
                session.resolve_all_gather_words(output).unwrap(),
                vec![phase as u32; 4]
            );
            session.ensure_capture_active().unwrap();
        }
    }
}

struct Children(Vec<Child>);
impl Drop for Children {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[test]
#[ignore = "spawns four local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_capture_hooks_exclude_inactive_pipeline_ranks_and_allow_reuse() {
    let sockets = (0..4)
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
    for rank in 0..4 {
        children.0.push(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "backend::distributed::tests::capture_hook::worker",
                    "--nocapture",
                ])
                .env(WORKER, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(45);
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
    for (rank, child) in children.0.drain(..).enumerate() {
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
