use super::*;

#[test]
#[ignore = "spawns four local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_overlapping_tp_pp_subgroups_use_exact_logical_membership() {
    assert!(distributed::is_available(Backend::Ring));
    let sockets = (0..4)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let ports = sockets
        .iter()
        .map(|socket| socket.local_addr().unwrap().port())
        .collect::<Vec<_>>();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!(
            "[[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"]]",
            ports[0], ports[1], ports[2], ports[3]
        ),
    )
    .unwrap();
    drop(sockets);

    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::with_capacity(4));
    for rank in 0..4 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "backend::runtime::distributed::communication_tests::subgroup_wave_worker",
                    "--nocapture",
                ])
                .env(SUBGROUP_WAVE_WORKER_RANK, rank.to_string())
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
                "overlapping subgroup Ring rank {rank} exited with {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "four-process overlapping subgroup Ring failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}

#[test]
#[ignore = "spawns four local Ring ranks and opens loopback sockets; run explicitly"]
fn ring_overlapping_tp_ep_variable_all_to_all_uses_one_world_wave() {
    assert!(distributed::is_available(Backend::Ring));
    let sockets = (0..4)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let ports = sockets
        .iter()
        .map(|socket| socket.local_addr().unwrap().port())
        .collect::<Vec<_>>();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!(
            "[[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"],[\"127.0.0.1:{}\"]]",
            ports[0], ports[1], ports[2], ports[3]
        ),
    )
    .unwrap();
    drop(sockets);

    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::with_capacity(4));
    for rank in 0..4 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "backend::runtime::distributed::communication_tests::variable_subgroup_wave_worker",
                    "--nocapture",
                ])
                .env(VARIABLE_SUBGROUP_WORKER_RANK, rank.to_string())
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
                "overlapping TP/EP VariableAllToAll rank {rank} exited with {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "four-process overlapping TP/EP VariableAllToAll failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}
