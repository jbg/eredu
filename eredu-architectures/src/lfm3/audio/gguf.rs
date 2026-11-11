//! Physical admission for the released LFM3 Audio GGUF backbone and projector.

use eredu_checkpoint::{
    schema::{CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint},
    validation::{resolve_gguf_plan, CheckpointValidation, ResolvedCheckpointPlan},
};
use eredu_gguf::{Checkpoint, GgmlType, Limits, MetadataValue, QuantizedTensorRepresentation};
use std::path::Path;

mod speech;
pub use speech::{AudioGgufSpeechBindings, AudioGgufSpeechSources};

/// Physical layouts published for either part of LFM3 Audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioGgufEncoding {
    /// BF16 weights with F32 companion tensors.
    Bf16,
    /// F16 weights with F32 companion tensors.
    F16,
    /// Q4_0 matrices, with a Q6_K vocabulary in the backbone.
    Q4_0,
    /// Mixed Q4_K and Q6_K matrices.
    Q4KM,
    /// Q8_0 matrices.
    Q8_0,
}

impl AudioGgufEncoding {
    fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self, AudioGgufError> {
        let value = checkpoint
            .metadata()
            .get("general.file_type")
            .and_then(MetadataValue::as_i64);
        match value {
            Some(32) => Ok(Self::Bf16),
            Some(1) => Ok(Self::F16),
            Some(2) => Ok(Self::Q4_0),
            Some(15) => Ok(Self::Q4KM),
            Some(7) => Ok(Self::Q8_0),
            _ => Err(AudioGgufError::Metadata("general.file_type")),
        }
    }

    fn matrix(self, backbone_embedding: bool) -> GgufTypeConstraint {
        use GgmlType::*;
        use GgufTypeConstraint::{Exact, OneOf};
        match self {
            Self::Bf16 => Exact(Bf16),
            Self::F16 => Exact(F16),
            Self::Q4_0 if backbone_embedding => Exact(Q6K),
            Self::Q4_0 => Exact(Q4_0),
            Self::Q4KM => OneOf(vec![Q4K, Q6K]),
            Self::Q8_0 => Exact(Q8_0),
        }
    }
}

/// Failure to admit the published Audio GGUF pair.
#[derive(Debug, thiserror::Error)]
pub enum AudioGgufError {
    /// Container structure or storage failed validation.
    #[error(transparent)]
    Container(#[from] eredu_gguf::Error),
    /// The retained source cannot realize its admitted tensor catalog.
    #[error(transparent)]
    Source(#[from] eredu_checkpoint::store::StoreError),
    /// Required role, geometry or encoding metadata differs.
    #[error("invalid LFM3 Audio GGUF metadata {0}")]
    Metadata(&'static str),
    /// The backbone or projector declaration is invalid.
    #[error("invalid LFM3 Audio GGUF declaration: {0}")]
    Declaration(String),
    /// The physical tensors differ from the declared layout.
    #[error("LFM3 Audio GGUF tensor schema differs: {0:?}")]
    Schema(CheckpointValidation),
}

/// A physically validated backbone and projector pair with retained container metadata.
/// This contract describes admission. The projector supplies the speech input encoder,
/// depth decoder and LiquidCodec decoder. Codec encoding requires a separate source.
pub struct AdmittedAudioGguf {
    backbone: Checkpoint,
    projector: Checkpoint,
    declaration: super::super::Declaration,
    backbone_layout: ResolvedCheckpointPlan,
    projector_layout: ResolvedCheckpointPlan,
    backbone_encoding: AudioGgufEncoding,
    projector_encoding: AudioGgufEncoding,
}

impl AdmittedAudioGguf {
    /// Opens the two explicitly supplied GGUF files and validates the backbone's rotary companion.
    pub fn open(backbone: &Path, projector: &Path) -> Result<Self, AudioGgufError> {
        let open = |path| {
            Checkpoint::open_with_representation(
                path,
                Limits::default(),
                QuantizedTensorRepresentation::DecodedF32,
            )
        };
        let admitted = Self::from_checkpoints(open(backbone)?, open(projector)?)?;
        super::super::checkpoint::validate_gguf_rope(
            admitted.backbone(),
            admitted.declaration.args(),
        )
        .map_err(AudioGgufError::Declaration)?;
        Ok(admitted)
    }

    /// Validates retained container metadata without reading tensor payloads.
    /// A metadata-only checkpoint keeps its original access authority.
    pub fn from_checkpoints(
        backbone: Checkpoint,
        projector: Checkpoint,
    ) -> Result<Self, AudioGgufError> {
        validate_roles(&backbone, &projector)?;
        let args = backbone_args(&backbone)?;
        let backbone_encoding = AudioGgufEncoding::from_checkpoint(&backbone)?;
        let projector_encoding = AudioGgufEncoding::from_checkpoint(&projector)?;
        let declaration = super::super::Declaration::new(args)
            .map_err(|error| AudioGgufError::Declaration(error.to_string()))?;
        let mut plan = super::super::checkpoint::gguf_plan(&declaration)
            .map_err(AudioGgufError::Declaration)?;
        plan.catalog_policy = CatalogPolicy::strict();
        for tensor in &mut plan.common_tensors {
            tensor.encoding =
                if tensor.shape.len() == 1 || tensor.key.ends_with("shortconv.conv.weight") {
                    GgufTypeConstraint::Exact(GgmlType::F32)
                } else {
                    backbone_encoding.matrix(tensor.key == "token_embd.weight")
                };
        }
        let backbone_layout =
            resolve_gguf_plan(&backbone, &plan).map_err(AudioGgufError::Schema)?;
        let projector_layout = resolve_gguf_plan(&projector, &projector_plan(projector_encoding)?)
            .map_err(AudioGgufError::Schema)?;
        Ok(Self {
            backbone,
            projector,
            declaration,
            backbone_layout,
            projector_layout,
            backbone_encoding,
            projector_encoding,
        })
    }

    /// Retained backbone, including tokenizer and chat-template metadata.
    pub fn backbone(&self) -> &Checkpoint {
        &self.backbone
    }

    /// Retained projector, including speech sampling metadata.
    pub fn projector(&self) -> &Checkpoint {
        &self.projector
    }

    /// Portable language declaration resolved from this backbone.
    pub fn declaration(&self) -> &super::super::Declaration {
        &self.declaration
    }

    /// Resolved physical schemas in backbone and projector order.
    pub fn layouts(&self) -> (&ResolvedCheckpointPlan, &ResolvedCheckpointPlan) {
        (&self.backbone_layout, &self.projector_layout)
    }

    /// Independently selected encodings in backbone and projector order.
    pub fn encodings(&self) -> (AudioGgufEncoding, AudioGgufEncoding) {
        (self.backbone_encoding, self.projector_encoding)
    }
}

fn backbone_args(checkpoint: &Checkpoint) -> Result<super::super::ModelArgs, AudioGgufError> {
    let metadata = checkpoint
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let args = super::super::model_args_from_gguf_catalog(checkpoint, &metadata)
        .map_err(|error| AudioGgufError::Declaration(error.to_string()))?;
    if args.vocab_size != 128_000
        || args.hidden_size != 2048
        || args.num_hidden_layers != 24
        || args.max_position_embeddings != 128_000
        || !args.tie_word_embeddings
    {
        return Err(AudioGgufError::Metadata("LFM3 Audio backbone geometry"));
    }
    Ok(args)
}

fn validate_roles(backbone: &Checkpoint, projector: &Checkpoint) -> Result<(), AudioGgufError> {
    for (checkpoint, architecture, role) in
        [(backbone, "lfm3", "model"), (projector, "clip", "mmproj")]
    {
        for (key, value) in [
            ("general.architecture", architecture),
            ("general.type", role),
            ("general.basename", "LFM3-Audio"),
        ] {
            if checkpoint
                .metadata()
                .get(key)
                .and_then(MetadataValue::as_str)
                != Some(value)
            {
                return Err(AudioGgufError::Metadata(key));
            }
        }
    }
    for key in [
        "tokenizer.chat_template",
        "tokenizer.chat_template.asr",
        "tokenizer.chat_template.tts",
    ] {
        if backbone
            .metadata()
            .get(key)
            .and_then(MetadataValue::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(AudioGgufError::Metadata(key));
        }
    }
    if backbone
        .metadata()
        .get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_strings)
        .is_none_or(|tokens| tokens.len() != 128_000)
    {
        return Err(AudioGgufError::Metadata("tokenizer.ggml.tokens"));
    }
    let meta = projector.metadata();
    for key in ["clip.has_audio_encoder", "clip.has_gen_audio_encoder"] {
        if meta.get(key).and_then(MetadataValue::as_bool) != Some(true) {
            return Err(AudioGgufError::Metadata(key));
        }
    }
    for (key, value) in [
        ("clip.audio.projector_type", "lfm3a"),
        ("clip.gen.audio.projector_type", "lfm3a_gen"),
    ] {
        if meta.get(key).and_then(MetadataValue::as_str) != Some(value) {
            return Err(AudioGgufError::Metadata(key));
        }
    }
    for (key, value) in [
        ("clip.audio.projection_dim", 2048),
        ("clip.audio.embedding_length", 768),
        ("clip.audio.feed_forward_length", 1792),
        ("clip.audio.block_count", 17),
        ("clip.audio.attention.head_count", 8),
        ("clip.audio.attention.head_count_kv", 4),
        ("clip.audio.num_mel_bins", 128),
        ("clip.audio.window_size", 32),
        ("clip.gen.audio.projection_dim", 2048),
        ("clip.gen.audio.embedding_length", 1024),
        ("clip.gen.audio.feed_forward_length", 2816),
        ("clip.gen.audio.block_count", 3),
        ("clip.gen.audio.attention.head_count", 32),
        ("clip.gen.audio.attention.head_count_kv", 8),
    ] {
        if meta.get(key).and_then(MetadataValue::as_i64) != Some(value) {
            return Err(AudioGgufError::Metadata(key));
        }
    }
    for key in [
        "clip.audio.attention.layer_norm_epsilon",
        "clip.gen.audio.attention.layer_norm_epsilon",
    ] {
        if meta.get(key).and_then(MetadataValue::as_f32) != Some(1e-5) {
            return Err(AudioGgufError::Metadata(key));
        }
    }
    Ok(())
}

/// Physical shapes below follow GGUF axis order, fastest first.
fn projector_plan(encoding: AudioGgufEncoding) -> Result<GgufCheckpointPlan, AudioGgufError> {
    let mut tensors = Vec::new();
    let mut add = |name: String, shape: &[usize], matrix: bool| {
        let shape = shape.iter().rev().copied().collect::<Vec<_>>();
        let dtype = if matrix {
            encoding.matrix(false)
        } else {
            GgufTypeConstraint::Exact(GgmlType::F32)
        };
        tensors.push(GgufTensorConstraint::required(name, shape, dtype));
    };

    for (layer, kernel, input) in [(0, 3, 1), (2, 3, 1), (3, 1, 256), (5, 3, 1), (6, 1, 256)] {
        add(
            format!("a.conv1d.{layer}.weight"),
            &[kernel, kernel, input, 256],
            false,
        );
        add(format!("a.conv1d.{layer}.bias"), &[1, 1, 256], false);
    }
    let input_attention = [2, 5, 8, 11, 14, 15, 16];
    let codec_attention = [2, 4, 6];
    for (prefix, layers, width, ffn, conv, heads, kv, attention, sandwich, gate) in [
        (
            "a.blk",
            17,
            768,
            1792,
            3,
            8,
            4,
            input_attention.as_slice(),
            false,
            false,
        ),
        (
            "a.gen.code.blk",
            3,
            1024,
            2816,
            0,
            32,
            8,
            &[0, 1, 2],
            false,
            true,
        ),
        (
            "a.gen.wav.tfm.blk",
            8,
            512,
            2304,
            7,
            8,
            2,
            codec_attention.as_slice(),
            true,
            true,
        ),
    ] {
        for index in 0..layers {
            let mut parameter = |name, shape: &[usize], matrix| {
                add(format!("{prefix}.{index}.{name}.weight"), shape, matrix)
            };
            for name in ["ln1", "ln2"] {
                parameter(name, &[width], false);
            }
            for name in ["ffn_gate", "ffn_up"] {
                parameter(name, &[width, ffn], true);
            }
            parameter("ffn_down", &[ffn, width], true);
            if sandwich {
                for name in ["attn_post_norm", "ffn_post_norm"] {
                    parameter(name, &[width], false);
                }
            }
            if attention.contains(&index) {
                for name in ["attn_q", "attn_out"] {
                    parameter(name, &[width, width], true);
                }
                for name in ["attn_k", "attn_v"] {
                    parameter(name, &[width, width / heads * kv], true);
                }
                for name in ["attn_q_norm", "attn_k_norm"] {
                    parameter(name, &[width / heads], false);
                }
                if gate {
                    parameter("attn_gate", &[width, heads], true);
                }
            } else {
                parameter("conv_dw", &[conv, width], false);
                parameter("conv_pw1", &[width, 3 * width], true);
                parameter("conv_pw2", &[width, width], true);
            }
        }
    }
    for (name, shape, matrix) in [
        ("a.post_ln.weight", &[768][..], false),
        ("a.pre_encode.out.weight", &[4096, 768], true),
        ("a.pre_encode.out.bias", &[768], false),
        ("mm.a.mlp.0.weight", &[768], false),
        ("mm.a.mlp.0.bias", &[768], false),
        ("mm.a.mlp.1.weight", &[768, 2048], true),
        ("mm.a.mlp.1.bias", &[2048], false),
        ("mm.a.mlp.3.weight", &[2048, 2048], true),
        ("mm.a.mlp.3.bias", &[2048], false),
        ("a.gen.code.embd.weight", &[1024, 2017, 12], true),
        ("a.gen.code.head_norm.weight", &[1024, 12], false),
        ("a.gen.code.mask_embd", &[1024], false),
        ("a.gen.code.out_embd.weight", &[2048, 24204], true),
        ("a.gen.code.output_norm.weight", &[1024], false),
        ("a.gen.code.proj_in.weight", &[2048, 12288], true),
        ("a.gen.code.proj_in.bias", &[12288], false),
        ("a.gen.wav.quant.first.codebook.weight", &[4, 2016], false),
        ("a.gen.wav.quant.first.out_proj.weight", &[48, 256], false),
        ("a.gen.wav.quant.first.out_proj.bias", &[256], false),
        ("a.gen.wav.tfm.in_proj.weight", &[256, 512], true),
        ("a.gen.wav.tfm.in_proj.bias", &[512], false),
        ("a.gen.wav.tfm.out_proj.weight", &[512, 1282], true),
        ("a.gen.wav.tfm.out_proj.bias", &[1282], false),
        ("a.gen.wav.tfm.output_norm.weight", &[512], false),
    ] {
        add(name.into(), shape, matrix);
    }
    GgufCheckpointPlan::new(
        "LFM3 Audio GGUF projector",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| AudioGgufError::Declaration(error.to_string()))
}
