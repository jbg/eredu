use std::{
    fs::File,
    io::{Read, Seek},
    net::TcpListener,
    process::{Child, Command},
    thread,
    time::{Duration, Instant},
};

struct Worker {
    child: Child,
    stdout: File,
    stderr: File,
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_log(file: &mut File) -> String {
    let mut bytes = Vec::new();
    file.rewind().unwrap();
    file.read_to_end(&mut bytes).unwrap();
    String::from_utf8_lossy(&bytes).into_owned()
}

pub(super) fn run(test: &str, timeout: Duration, mut configure: impl FnMut(&mut Command, usize)) {
    assert!(safemlx::distributed::is_available(
        safemlx::distributed::Backend::Ring
    ));
    let listeners = [(); 2].map(|()| TcpListener::bind(("127.0.0.1", 0)).unwrap());
    let hosts = listeners
        .iter()
        .map(|listener| [listener.local_addr().unwrap().to_string()])
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(listeners);

    let executable = std::env::current_exe().unwrap();
    let mut workers = Vec::with_capacity(hosts.len());
    for rank in 0..hosts.len() {
        let stdout = tempfile::tempfile().unwrap();
        let stderr = tempfile::tempfile().unwrap();
        let mut command = Command::new(&executable);
        command
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(stdout.try_clone().unwrap())
            .stderr(stderr.try_clone().unwrap());
        configure(&mut command, rank);
        let child = command.spawn().unwrap();
        workers.push(Worker {
            child,
            stdout,
            stderr,
        });
    }

    let deadline = Instant::now() + timeout;
    let timed_out = loop {
        let statuses = workers
            .iter_mut()
            .map(|worker| worker.child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses
            .iter()
            .all(|status| status.is_some_and(|status| status.success()))
        {
            return;
        }
        if statuses.iter().flatten().any(|status| !status.success()) {
            break false;
        }
        if Instant::now() >= deadline {
            break true;
        }
        thread::sleep(Duration::from_millis(20));
    };

    for worker in &mut workers {
        let _ = worker.child.kill();
    }
    let diagnostics = workers
        .iter_mut()
        .enumerate()
        .map(|(rank, worker)| {
            let status = worker.child.wait().unwrap();
            let stdout = read_log(&mut worker.stdout);
            let stderr = read_log(&mut worker.stderr);
            format!("rank {rank} exited with {status}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}")
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    panic!("Ring test {test} failed (timed_out={timed_out}):\n{diagnostics}");
}
