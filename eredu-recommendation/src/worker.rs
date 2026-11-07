use crate::{
    catalogue::Profile,
    forecast::{self, Inspection},
};
use anyhow::{ensure, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use eredu::api::{
    ArtifactMetadata, CheckpointMetadata, GgufCompanionMetadata, GgufHeader, MetadataProvenance,
    SafetensorsHeader,
};
use eredu_core::artifact::ArtifactCompanionRole;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Serialize, Deserialize)]
struct Header {
    member: String,
    bytes: String,
    file_len: u64,
}

// Only immutable metadata crosses the process boundary; workers cannot fetch weights.
#[derive(Serialize, Deserialize)]
struct Input {
    provenance: MetadataProvenance,
    config: Option<String>,
    index: Option<String>,
    headers: Vec<Header>,
    projector: Vec<Header>,
    sidecars: BTreeMap<String, String>,
}

impl Input {
    fn encode(metadata: ArtifactMetadata) -> Result<Self> {
        fn gguf(headers: Vec<GgufHeader>) -> Vec<Header> {
            headers
                .into_iter()
                .map(|h| Header {
                    member: h.member,
                    bytes: STANDARD.encode(h.bytes),
                    file_len: h.file_len,
                })
                .collect()
        }
        let (config, index, headers, projector) = match metadata.checkpoint {
            CheckpointMetadata::SafeTensors {
                config,
                index,
                headers,
            } => (
                Some(STANDARD.encode(config)),
                index.map(|bytes| STANDARD.encode(bytes)),
                headers
                    .into_iter()
                    .map(|h| Header {
                        member: h.member,
                        bytes: STANDARD.encode(h.bytes),
                        file_len: h.file_len,
                    })
                    .collect(),
                vec![],
            ),
            CheckpointMetadata::Gguf {
                headers,
                companions,
            } => {
                let mut projector = Vec::new();
                for companion in companions {
                    ensure!(
                        companion.role == ArtifactCompanionRole::MediaProjector
                            && projector.is_empty(),
                        "Unexpected catalogue companion"
                    );
                    projector = gguf(companion.headers);
                }
                (None, None, gguf(headers), projector)
            }
        };
        Ok(Self {
            provenance: metadata.provenance,
            config,
            index,
            headers,
            projector,
            sidecars: metadata
                .sidecars
                .into_iter()
                .map(|(key, bytes)| (key, STANDARD.encode(bytes)))
                .collect(),
        })
    }

    fn decode(self) -> Result<ArtifactMetadata> {
        fn gguf(headers: Vec<Header>) -> Result<Vec<GgufHeader>> {
            headers
                .into_iter()
                .map(|h| {
                    Ok(GgufHeader {
                        member: h.member,
                        bytes: STANDARD.decode(h.bytes)?,
                        file_len: h.file_len,
                    })
                })
                .collect()
        }
        let checkpoint = match self.config {
            Some(config) => CheckpointMetadata::SafeTensors {
                config: STANDARD.decode(config)?,
                index: self.index.map(|bytes| STANDARD.decode(bytes)).transpose()?,
                headers: self
                    .headers
                    .into_iter()
                    .map(|h| {
                        Ok(SafetensorsHeader {
                            member: h.member,
                            bytes: STANDARD.decode(h.bytes)?,
                            file_len: h.file_len,
                        })
                    })
                    .collect::<Result<_>>()?,
            },
            None => CheckpointMetadata::Gguf {
                headers: gguf(self.headers)?,
                companions: if self.projector.is_empty() {
                    vec![]
                } else {
                    vec![GgufCompanionMetadata {
                        role: ArtifactCompanionRole::MediaProjector,
                        headers: gguf(self.projector)?,
                    }]
                },
            },
        };
        Ok(ArtifactMetadata {
            provenance: self.provenance,
            checkpoint,
            sidecars: self
                .sidecars
                .into_iter()
                .map(|(key, bytes)| Ok((key, STANDARD.decode(bytes)?)))
                .collect::<Result<_>>()?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WorkerLimits {
    pub timeout: Duration,
    pub max_memory_bytes: u64,
    pub max_io_bytes: u64,
}

impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            max_memory_bytes: 2 * 1024 * 1024 * 1024,
            max_io_bytes: 128 * 1024 * 1024,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct WorkerInput {
    profile: Profile,
    metadata: Input,
}

#[derive(Serialize, Deserialize)]
struct WorkerOutput {
    result: std::result::Result<Inspection, String>,
}

struct CappedWriter<W> {
    inner: W,
    remaining: u64,
}

impl<W: Write> Write for CappedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.remaining {
            return Err(std::io::Error::other("Worker input exceeds limit"));
        }
        let written = self.inner.write(bytes)?;
        self.remaining -= written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// The executable implements this crate's `--worker` JSON protocol.
/// The caller selects it explicitly; the hosting application is never reexecuted.
pub fn inspect(
    executable: &Path,
    profile: &Profile,
    metadata: ArtifactMetadata,
    limits: WorkerLimits,
) -> Result<Inspection> {
    ensure!(
        !limits.timeout.is_zero() && limits.max_memory_bytes > 0 && limits.max_io_bytes > 0,
        "Worker limits must be positive"
    );
    profile.validate()?;
    let mut input = tempfile::tempfile()?;
    serde_json::to_writer(
        CappedWriter {
            inner: &mut input,
            remaining: limits.max_io_bytes,
        },
        &WorkerInput {
            profile: profile.clone(),
            metadata: Input::encode(metadata)?,
        },
    )?;
    ensure!(
        input.metadata()?.len() <= limits.max_io_bytes,
        "Worker input exceeds limit"
    );
    input.seek(SeekFrom::Start(0))?;
    let mut output = tempfile::tempfile()?;
    let diagnostics = tempfile::tempfile()?;
    let mut child = Command::new(executable)
        .arg("--worker")
        .stdin(Stdio::from(input))
        .stdout(Stdio::from(output.try_clone()?))
        .stderr(Stdio::from(diagnostics.try_clone()?))
        .spawn()?;
    let started = Instant::now();
    let pid = sysinfo::Pid::from_u32(child.id());
    let mut system = sysinfo::System::new();
    let outcome = (|| -> Result<()> {
        loop {
            if let Some(status) = child.try_wait()? {
                ensure!(status.success(), "Metadata worker exited with {status}");
                break;
            }
            ensure!(
                started.elapsed() < limits.timeout,
                "Metadata inspection exceeded time limit"
            );
            ensure!(
                output.metadata()?.len() <= limits.max_io_bytes
                    && diagnostics.metadata()?.len() <= limits.max_io_bytes,
                "Worker output exceeds limit"
            );
            system.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                true,
                sysinfo::ProcessRefreshKind::nothing().with_memory(),
            );
            if let Some(process) = system.process(pid) {
                ensure!(
                    process.memory() <= limits.max_memory_bytes,
                    "Metadata inspection exceeded memory limit"
                );
            } else if started.elapsed() > Duration::from_secs(1) {
                // Recheck exit before treating an absent process observation as a monitor failure.
                if child.try_wait()?.is_none() {
                    anyhow::bail!("Worker memory observation unavailable");
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        ensure!(
            output.metadata()?.len() <= limits.max_io_bytes,
            "Worker output exceeds limit"
        );
        Ok(())
    })();
    if outcome.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    outcome?;
    output.seek(SeekFrom::Start(0))?;
    let response: WorkerOutput = serde_json::from_reader(output)?;
    response.result.map_err(anyhow::Error::msg)
}

/// Internal CLI protocol. No network or device discovery runs in this process.
pub fn run() -> Result<()> {
    const LIMIT: u64 = 128 * 1024 * 1024;
    let mut bytes = Vec::new();
    std::io::stdin().take(LIMIT + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= LIMIT,
        "Worker input exceeds protocol limit"
    );
    let input: WorkerInput = serde_json::from_slice(&bytes)?;
    let result = (|| -> Result<Inspection> {
        let metadata = input.metadata.decode()?;
        let options = input.profile.inspection_options()?;
        Ok(forecast::inspect(
            &metadata,
            &options,
            &input.profile.memory_options()?,
            &eredu::api::ForecastCalibration::default(),
        )?)
    })()
    .map_err(|e| e.to_string());
    serde_json::to_writer(std::io::stdout().lock(), &WorkerOutput { result })?;
    std::io::stdout().flush()?;
    Ok(())
}
