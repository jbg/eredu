//! Local, digest-checked artifact preparation and selected native startup.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use eredu::api::realtime::{
    prepare_mlx_realtime, prepare_realtime_model, LoadedMlxRealtime, MlxVoiceChatSession,
    RealtimeArtifactRequest, RealtimeArtifactSource as Source, RealtimeAssetDirectories,
    RealtimeLoadOptions, SpeechSessionOptions, VoiceChatDecodeLimits, VoiceChatInputConfig,
    VoiceChatPhrasePolicy, VoiceChatPublicationCapacity, VoiceChatPublicationLimits,
    VoiceChatSpeechOptions, VoiceChatStartOptions, VoiceChatTextLimits, VoiceChatToolLimits,
    VoiceChatTurnLimits, VoiceChatTurnPolicy,
};
use eredu_architectures::voicechat::{
    agent_sampling::AgentSamplingPolicy,
    channels::PromptPolicy,
    component_residency::VoiceChatResidencyPolicy,
    frontend::FrontendOptions,
    perception_rnnt::PerceptionRnntLimits,
    provider_residency::{SpeechProvider, SpeechResidencyPolicy},
    rnnt::RnntLimits,
    schedule::ScheduleLimits,
};
use eredu_backend_mlx::native::ExecutionContext;
use eredu_core::{residency::OffloadConfig, scheduler::RequestId};
use eredu_runtime::{
    CacheResidencyPolicy, DenseDiskStreamLoadOptions, LayerWeightResidency, LayerwiseLoadOptions,
};
use serde::Deserialize;

/// A local request. Relative artifact paths resolve against this JSON file's directory.
/// Required byte limits are logical admission limits, not allocator or process limits.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub speech_weights: SpeechFloatChoice,
    #[serde(default)]
    pub speech_compute: SpeechFloatChoice,
    pub official_assets: PathBuf,
    pub base_assets: PathBuf,
    pub source: Source,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    /// Fixed choice spoken during tool-call serialization, overriding random selection.
    #[serde(default)]
    pub acknowledgement: Option<String>,
    /// Select from the released pool using `seed` when no fixed phrase is supplied.
    #[serde(default = "default_random_phrase")]
    pub random_acknowledgement: bool,
    /// Fixed fallback initial reminder; per-call scripted text takes precedence.
    #[serde(default)]
    pub initial_reminder: Option<String>,
    /// Select from the released initial reminder pool when no fixed text is supplied.
    #[serde(default = "default_random_phrase")]
    pub random_initial_reminder: bool,
    #[serde(default)]
    pub device: Device,
    #[serde(default)]
    pub parameter_residency: ParameterResidency,
    pub device_parameter_bytes: u64,
    pub host_parameter_bytes: u64,
    pub max_branch_bytes: u64,
    pub max_total_copy_bytes: u64,
    #[serde(default = "default_frames")]
    pub max_frames: u64,
    #[serde(default = "default_steps")]
    pub max_model_steps: u64,
    /// Consecutive RNNT nonblank frames required for barge-in (80 ms each).
    /// Omission retains the pinned streaming policy's 40-frame threshold.
    #[serde(default)]
    pub interruption_speech_frames: Option<u64>,
    #[serde(default)]
    pub seed: u64,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    #[default]
    Cpu,
    Gpu,
}

/// Portable speech projection precision, configured independently of the device.
#[derive(Debug, Default, Clone, Copy, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SpeechFloatChoice {
    #[default]
    F32,
    F16,
    Bf16,
}
impl From<SpeechFloatChoice> for eredu_architectures::voicechat::precision::SpeechFloat {
    fn from(value: SpeechFloatChoice) -> Self {
        match value {
            SpeechFloatChoice::F32 => Self::F32,
            SpeechFloatChoice::F16 => Self::F16,
            SpeechFloatChoice::Bf16 => Self::Bf16,
        }
    }
}

/// Every speech provider has its own finite pool; the enclosing totals include all ten.
#[derive(Debug, Default, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParameterResidency {
    #[default]
    Resident,
    Host {
        backbone_device_bytes: u64,
        backbone_host_bytes: u64,
        speech_device_bytes: u64,
        speech_host_bytes: u64,
        #[serde(default = "default_depth")]
        depth: usize,
    },
    Disk {
        backbone_device_bytes: u64,
        backbone_host_bytes: u64,
        speech_device_bytes: u64,
        speech_host_bytes: u64,
    },
}

fn default_frames() -> u64 {
    4096
}
fn default_steps() -> u64 {
    8192
}
fn default_depth() -> usize {
    1
}
fn default_random_phrase() -> bool {
    true
}

impl Config {
    pub fn resolve_paths(&mut self, directory: &Path) {
        fn resolve(path: &mut PathBuf, directory: &Path) {
            if path.is_relative() {
                *path = directory.join(&*path);
            }
        }
        resolve(&mut self.official_assets, directory);
        resolve(&mut self.base_assets, directory);
        match &mut self.source {
            Source::SafeTensors { path } | Source::Gguf { path } => resolve(path, directory),
            Source::SplitGguf {
                perception,
                backbone,
                function_head,
                tts,
                companion,
            } => {
                for path in [perception, backbone, function_head, tts, companion] {
                    resolve(path, directory);
                }
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.official_assets.is_dir(),
            "official_assets must be a local snapshot directory"
        );
        ensure!(
            self.base_assets.is_dir(),
            "base_assets must be a local snapshot directory"
        );
        ensure!(
            self.system_prompt.is_ascii(),
            "VoiceChat system_prompt must be ASCII"
        );
        for (role, text) in [
            ("acknowledgement", &self.acknowledgement),
            ("initial_reminder", &self.initial_reminder),
        ] {
            if let Some(text) = text {
                ensure!(text.is_ascii(), "VoiceChat {role} must be ASCII");
                ensure!(
                    !text.trim().is_empty() && text.len() <= 65_536,
                    "VoiceChat {role} must contain text within the 65536-byte limit"
                );
            }
        }
        ensure!(
            self.device_parameter_bytes > 0 && self.host_parameter_bytes > 0,
            "parameter admission requires positive device and host limits"
        );
        ensure!(
            self.max_branch_bytes > 0 && self.max_total_copy_bytes > 0,
            "state branch and total copy limits must be positive"
        );
        ensure!(
            self.max_frames > 0 && self.max_model_steps > 0,
            "frame and model-step limits must be positive"
        );
        ensure!(
            self.interruption_speech_frames != Some(0),
            "interruption_speech_frames must be positive"
        );
        // Check integer representations before any large artifact is opened.
        usize::try_from(self.max_frames)?;
        usize::try_from(self.max_model_steps)?;
        self.max_model_steps
            .checked_add(8194)
            .context("backbone position capacity overflows")?;
        self.max_model_steps
            .checked_add(128)
            .context("speech position capacity overflows")?;
        match &self.source {
            Source::SafeTensors { path } => ensure!(
                path.exists(),
                "SafeTensors source does not exist: {}",
                path.display()
            ),
            Source::Gguf { path } => ensure!(
                path.is_file(),
                "GGUF source is not a file: {}",
                path.display()
            ),
            Source::SplitGguf {
                perception,
                backbone,
                function_head,
                tts,
                companion,
            } => {
                for path in [perception, backbone, function_head, tts] {
                    ensure!(
                        path.is_file(),
                        "split GGUF component is not a file: {}",
                        path.display()
                    );
                }
                ensure!(
                    companion.exists(),
                    "explicit complete companion does not exist: {}",
                    companion.display()
                );
            }
        }
        let _ = self.residency(2)?;
        Ok(())
    }

    fn residency(&self, cached_shards: usize) -> Result<VoiceChatResidencyPolicy> {
        let host = |device, host, depth| -> Result<_> {
            Ok(LayerWeightResidency::LayerwiseHost(
                LayerwiseLoadOptions::new(OffloadConfig::new(Some(device), Some(host), depth)?)
                    .with_max_cached_shards(cached_shards),
            ))
        };
        let disk = |device, host| -> Result<_> {
            let slots = usize::from(host != 0);
            Ok(LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(device, host, slots, slots)?
                    .with_max_cached_shards(cached_shards),
            ))
        };
        let (backbone, speech) = match self.parameter_residency {
            ParameterResidency::Resident => (
                LayerWeightResidency::FullyResident,
                LayerWeightResidency::FullyResident,
            ),
            ParameterResidency::Host {
                backbone_device_bytes,
                backbone_host_bytes,
                speech_device_bytes,
                speech_host_bytes,
                depth,
            } => (
                host(backbone_device_bytes, backbone_host_bytes, depth)?,
                host(speech_device_bytes, speech_host_bytes, depth)?,
            ),
            ParameterResidency::Disk {
                backbone_device_bytes,
                backbone_host_bytes,
                speech_device_bytes,
                speech_host_bytes,
            } => (
                disk(backbone_device_bytes, backbone_host_bytes)?,
                disk(speech_device_bytes, speech_host_bytes)?,
            ),
        };
        let speech = SpeechResidencyPolicy::new(
            SpeechProvider::ALL
                .into_iter()
                .map(|provider| (provider, speech))
                .collect(),
            self.device_parameter_bytes,
            self.host_parameter_bytes,
        )?;
        Ok(VoiceChatResidencyPolicy::new(
            backbone,
            speech,
            self.device_parameter_bytes,
            self.host_parameter_bytes,
        ))
    }
}

pub fn load(
    config: &Config,
    quantization: eredu_architectures::voicechat::quantization::SpeechQuantizationPolicy,
    device: eredu::api::LocalDevice,
    cached_shards: usize,
    sampling: AgentSamplingPolicy,
) -> Result<MlxVoiceChatSession> {
    config.validate()?;
    let artifact = prepare_realtime_model(&RealtimeArtifactRequest {
        source: config.source.clone(),
        assets: RealtimeAssetDirectories {
            model: Some(config.official_assets.clone()),
            text_backbone: Some(config.base_assets.clone()),
        },
    })?;
    let prepared = prepare_mlx_realtime(
        &artifact,
        &RealtimeLoadOptions::Speech(SpeechSessionOptions {
            quantization,
            conversation: start_options(config, sampling)?,
            parameters: config.residency(cached_shards)?,
            state: CacheResidencyPolicy::Device,
            max_branch_bytes: config.max_branch_bytes,
            max_total_copy_bytes: config.max_total_copy_bytes,
            seed: config.seed,
        }),
    )?;
    if let Some(report) = prepared.speech_residency_report() {
        eprintln!(
            "parameter admission: device={} host={} materialization={} execution_cache={} bytes; mutable state and activations are separate",
            report.device_parameter_bytes,
            report.host_parameter_bytes,
            report.host_materialization_bytes,
            report.speech.execution_cache_device_bytes
        );
    }
    let weights = ExecutionContext::cpu()?;
    let execution = match device {
        eredu::api::LocalDevice::Cpu => ExecutionContext::cpu()?,
        eredu::api::LocalDevice::Accelerator(index) => {
            ExecutionContext::accelerator(usize::try_from(index)?)?
        }
    };
    match prepared.load(&weights, &execution)? {
        LoadedMlxRealtime::Speech(session) => Ok(session),
        LoadedMlxRealtime::Categorical { .. } => {
            anyhow::bail!("WAV input requires a PCM speech artifact")
        }
    }
}

fn start_options(config: &Config, sampling: AgentSamplingPolicy) -> Result<VoiceChatStartOptions> {
    let frame_capacity = usize::try_from(config.max_frames)?;
    let capacity = VoiceChatPublicationCapacity {
        speech_samples: 8192,
        transcription_tokens: 128,
        agent_text_bytes: 65_536,
        transcription_text_bytes: 65_536,
        audio_codes: 128,
        tool_events: 32,
        tool_bytes: 131_072,
        diagnostics: 128,
    };
    Ok(VoiceChatStartOptions {
        speech_precision: eredu_architectures::voicechat::precision::SpeechPrecision {
            weights: config.speech_weights.into(),
            compute: config.speech_compute.into(),
        },
        request: RequestId::new(1),
        system_prompt: config.system_prompt.clone(),
        tools: config.tools.clone(),
        acknowledgement: match &config.acknowledgement {
            Some(text) => VoiceChatPhrasePolicy::Fixed(text.clone()),
            None if config.random_acknowledgement => {
                VoiceChatPhrasePolicy::Streaming { seed: config.seed }
            }
            None => VoiceChatPhrasePolicy::Disabled,
        },
        initial_reminder: match &config.initial_reminder {
            Some(text) => VoiceChatPhrasePolicy::Fixed(text.clone()),
            None if config.random_initial_reminder => VoiceChatPhrasePolicy::Streaming {
                // Independent retained host stream, domain-separated by "REMINDER".
                seed: config.seed ^ 0x5245_4d49_4e44_4552,
            },
            None => VoiceChatPhrasePolicy::Disabled,
        },
        text_limits: VoiceChatTextLimits {
            max_bytes: 65_536,
            max_tokens: 8192,
        },
        prompt_policy: PromptPolicy {
            repeat: 1,
            max_tokens: 8194,
        },
        tool_limits: VoiceChatToolLimits {
            max_payload_tokens: 8192,
            max_payload_bytes: 65_536,
            max_calls: 8,
        },
        schedule_limits: ScheduleLimits {
            max_audio_frames: config.max_frames,
            max_backbone_positions: config
                .max_model_steps
                .checked_add(8194)
                .context("backbone position capacity overflows")?,
            max_tts_positions: config
                .max_model_steps
                .checked_add(128)
                .context("TTS position capacity overflows")?,
            max_codec_frames: config
                .max_model_steps
                .checked_add(128)
                .context("codec position capacity overflows")?,
        },
        sampling,
        turn_policy: VoiceChatTurnPolicy {
            bou_frames: config
                .interruption_speech_frames
                .unwrap_or_else(|| VoiceChatTurnPolicy::default().bou_frames),
            ..VoiceChatTurnPolicy::default()
        },
        turn_limits: VoiceChatTurnLimits {
            max_audio_frames: config.max_frames,
            max_model_steps: config.max_model_steps,
            max_history: 40,
        },
        decode_limits: VoiceChatDecodeLimits::default(),
        transcription_pending_bytes: 65_536,
        frontend: FrontendOptions::default(),
        input_config: VoiceChatInputConfig::default(),
        input_limits: PerceptionRnntLimits {
            frame_capacity: config.max_frames,
            rnnt: RnntLimits {
                max_symbols_per_frame: 10,
                hypothesis_capacity: frame_capacity
                    .checked_mul(10)
                    .context("RNNT hypothesis capacity overflows")?,
            },
        },
        max_character_cells: 65_536,
        max_prompt_positions: 37,
        publication_limits: VoiceChatPublicationLimits {
            max_publications: 1,
            capacity,
        },
        publication_capacity: capacity,
        clock_origin: Instant::now(),
        start_at: Duration::ZERO,
        speech: VoiceChatSpeechOptions {
            force_silence_on_eos: true,
            force_silence_on_pad: true,
        },
    })
}
