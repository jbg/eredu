use super::*;

const RANK: &str = "EREDU_MEMBER_STATUS_WORKER";
const DIRECTORY: &str = "EREDU_MEMBER_STATUS_DIRECTORY";

#[test]
#[ignore = "spawns four local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_member_status_agrees_without_inactive_rank_collectives() {
    let sockets = (0..4)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![socket.local_addr().unwrap().to_string()])
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::new());
    for rank in 0..4 {
        children.0.push(Command::new(&executable)
            .args(["--exact", "backend::runtime::distributed::communication_tests::status::member_status_worker", "--nocapture"])
            .env(RANK, rank.to_string()).env(DIRECTORY, directory.path())
            .env("MLX_RANK", rank.to_string()).env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap());
    }
    let deadline = Instant::now() + Duration::from_secs(45);
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
                "rank {rank}: {}\n{}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(
        !timed_out && failures.is_empty(),
        "member status timed_out={timed_out}: {}",
        failures.join("\n")
    );
}

#[test]
fn member_status_worker() {
    let Ok(rank) = std::env::var(RANK) else {
        return;
    };
    let rank = rank.parse::<usize>().unwrap();
    let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).unwrap());
    let native = distributed::init(true, Backend::Ring).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for (phase, members) in [
        vec![0, 1, 2],
        vec![0, 2, 3],
        vec![1, 2, 3],
        vec![0, 3],
        vec![2],
    ]
    .into_iter()
    .enumerate()
    {
        super::super::reset_native_collective_submissions();
        if let Some(local) = members.iter().position(|member| *member == rank) {
            let descriptor = CommunicationGroupDescriptor::new(
                CollectiveGroupId::new(71),
                0,
                members.clone(),
                Some(local),
                CommunicationGroupRequirements::new([
                    CommunicationOperationRequirement::failure_agreement(true),
                ])
                .unwrap(),
            )
            .unwrap();
            let group = Group::uncontracted(&native)
                .logical_subgroup(&members)
                .unwrap()
                .with_manifest_contract(&descriptor, completion_policy())
                .unwrap();
            // No world-wave proof is attached. Nonmembers perform no native work.
            for failed in [None, Some(members[0]), Some(*members.last().unwrap())] {
                let submission =
                    MlxNeuralBackend::agree_success(failed != Some(rank), &group, &stream).unwrap();
                assert_eq!(submission.completion.retained_arrays(), 2);
                assert_eq!(submission.completion.retained_groups(), 1);
                assert_eq!(submission.completion.retained_streams(), 1);
                let BoundedSubmissionOutcome::Completed(output) = submission
                    .wait_bounded(completion_policy().bounded_wait())
                    .unwrap()
                else {
                    panic!("bounded member status deadline");
                };
                assert_eq!(
                    MlxNeuralBackend::resolve_failure_agreement(output).unwrap(),
                    failed.is_none()
                );
            }
            assert_eq!(super::super::group::native_collective_submissions(), 3);
        } else {
            assert_eq!(super::super::group::native_collective_submissions(), 0);
        }
        // A host fixture barrier separates rounds without making inactive ranks
        // enter a native collective on the same Ring streams.
        std::fs::write(directory.join(format!("finished-{phase}-{rank}")), []).unwrap();
        let deadline = Instant::now() + Duration::from_secs(35);
        while !(0..4).all(|rank| directory.join(format!("finished-{phase}-{rank}")).exists()) {
            assert!(
                Instant::now() < deadline,
                "fixture peer failed phase {phase}"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }
}
