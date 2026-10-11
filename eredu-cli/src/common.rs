//! Arguments and portable normalization shared by text and speech commands.
use super::{CliDevice, LoadQuantizationMode};
use anyhow::{bail, Result};
use eredu_core::{QuantizationRequest, WeightTransformationPlan};
use eredu_runtime::NormalizedLoadRequest;

#[derive(Debug, Clone, PartialEq, clap::Args)]
pub struct CommonOptions {
    /// Model directory, GGUF file, or cached Hugging Face model identifier.
    /// Append `:QUANT` to select a cached GGUF quantization.
    #[arg(short, long, value_name = "PATH_OR_ID")]
    pub model: Option<String>,

    /// Cached Hugging Face revision (a ref such as `main` or a commit hash).
    #[arg(long, value_name = "REVISION", requires = "model")]
    pub revision: Option<String>,

    /// Main model execution device: `cpu` or `gpu:N`.
    #[arg(long, value_name = "DEVICE")]
    pub device: Option<CliDevice>,

    /// Process-global MLX allocator-cache limit in bytes; zero disables caching.
    #[arg(long, value_name = "BYTES")]
    pub mlx_cache_limit_bytes: Option<u64>,

    /// Quantize eligible dense weights to this bit width while loading.
    #[arg(long, value_name = "BITS")]
    pub quantize: Option<i32>,

    /// Encoding used for load-time quantization. MXFP4 requires --quantize 4.
    #[arg(long, value_enum, default_value_t = LoadQuantizationMode::Affine)]
    pub quantization_mode: LoadQuantizationMode,

    /// Number of adjacent weights sharing quantization parameters.
    #[arg(long, default_value_t = 64, value_name = "WEIGHTS")]
    pub quantization_group_size: i32,

    /// Keep repeated model layers on the host and use a bounded device window.
    #[arg(long)]
    pub layerwise_host: bool,

    /// Stream ordinary execution layers through bounded disk, host, and device caches.
    #[arg(long)]
    pub dense_disk_stream: bool,

    /// Maximum repeated layers resident on the execution device.
    #[arg(long, default_value_t = 1, value_name = "LAYERS")]
    pub device_layer_window: usize,

    /// Optional logical device parameter budget in bytes.
    #[arg(long, value_name = "BYTES")]
    pub device_budget_bytes: Option<u64>,

    /// Optional logical host parameter budget in bytes.
    #[arg(long, value_name = "BYTES")]
    pub host_budget_bytes: Option<u64>,

    /// Maximum simultaneously cached SafeTensors shards or GGUF readers.
    #[arg(long, default_value_t = 4, value_name = "SHARDS")]
    pub cached_shards: usize,

    /// Sampling temperature. Omission preserves the command's sampling policy.
    #[arg(short = 't', long, value_name = "FLOAT")]
    pub temperature: Option<f32>,

    /// Nucleus probability. Omission preserves the command's sampling policy.
    #[arg(long, value_name = "FLOAT")]
    pub top_p: Option<f32>,

    /// Penalty for repeating a token. Omission preserves the command's sampling policy.
    #[arg(long, value_name = "FLOAT")]
    pub repeat_penalty: Option<f32>,

    /// Random seed used when temperature is non-zero.
    #[arg(long)]
    pub seed: Option<u64>,
}
impl CommonOptions {
    pub fn configure_runtime(&self) -> Result<()> {
        let runtime = self
            .mlx_cache_limit_bytes
            .map(|bytes| {
                usize::try_from(bytes).map(|limit| {
                    eredu::api::LocalRuntimeConfiguration::default()
                        .with_allocator_cache_limit(limit)
                })
            })
            .transpose()?
            .unwrap_or_default();
        eredu::api::configure_local_runtime(&runtime)?;
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !(self.revision.is_some()
                && self
                    .model
                    .as_deref()
                    .is_some_and(|model| std::path::Path::new(model).exists())),
            "--revision can only be used with a Hugging Face model identifier"
        );

        anyhow::ensure!(
            self.cached_shards > 0,
            "--cached-shards must be greater than zero"
        );
        anyhow::ensure!(
            self.device_layer_window > 0,
            "--device-layer-window must be greater than zero"
        );
        anyhow::ensure!(
            !(self.layerwise_host && self.dense_disk_stream),
            "--dense-disk-stream conflicts with --layerwise-host"
        );
        self.quantization_request()?;
        Ok(())
    }
    pub fn quantization_request(&self) -> Result<Option<QuantizationRequest>> {
        let request = match (self.quantize, self.quantization_mode) {
            (Some(bits), LoadQuantizationMode::Affine) => Some(QuantizationRequest::Affine {
                group_size: u32::try_from(self.quantization_group_size)?,
                bits: u8::try_from(bits)?,
            }),
            (Some(4), LoadQuantizationMode::Mxfp4) => Some(QuantizationRequest::MxFp4),
            (Some(_), LoadQuantizationMode::Mxfp4) => {
                bail!("--quantization-mode mxfp4 requires --quantize 4")
            }
            (None, LoadQuantizationMode::Mxfp4) => bail!("--quantization-mode requires --quantize"),
            (None, LoadQuantizationMode::Affine) => None,
        };
        if let Some(request) = request {
            NormalizedLoadRequest::with_quantization(request).weight_quantization()?;
        }
        Ok(request)
    }
    pub fn weight_transformation(&self) -> Result<WeightTransformationPlan> {
        Ok(match self.quantization_request()? {
            None => WeightTransformationPlan::PreserveCheckpoint,
            Some(QuantizationRequest::Affine { bits, group_size }) => {
                WeightTransformationPlan::Affine {
                    bits: i32::from(bits),
                    group_size: i32::try_from(group_size)?,
                }
            }
            Some(QuantizationRequest::MxFp4) => WeightTransformationPlan::MxFp4,
            Some(_) => unreachable!("CLI declares only affine and MXFP4 transformations"),
        })
    }
    pub fn weight_quantization(&self) -> Result<Option<eredu_checkpoint::WeightQuantization>> {
        let request = eredu_runtime::execution_plan_quantization(self.weight_transformation()?)?;
        request
            .map(|q| NormalizedLoadRequest::with_quantization(q).weight_quantization())
            .transpose()
            .map(Option::flatten)
            .map_err(Into::into)
    }
}
