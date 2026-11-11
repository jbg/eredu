//! Published LiquidCodec decoder layout in the speech projector GGUF.

use super::{invalid, Config, Declaration};
use crate::Error;
use eredu_checkpoint::{
    gguf_store::GgufCatalog,
    recipe::DerivedWeightRecipe,
    store::{
        CheckpointLease, CompositeCheckpointSource, MemoryWeightStore, ReadPolicy,
        SharedCheckpointSource, TensorReadRequest, TensorSelection,
    },
    StoredDtype,
};
use eredu_gguf::{ConvertedTensor, DenseDtype};
use eredu_nn::DeclareLogicalParameters;
use std::{collections::BTreeMap, sync::Arc};

const CODE_TABLE: &str = "a.gen.wav.quant.first.codebook.weight";
const WINDOW: &str = "decoder.istft.window";

/// Canonical decoder-only projection of a published speech GGUF.
pub struct GgufDecoderBindings {
    config: Config,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
}

impl GgufDecoderBindings {
    /// Checks the canonical projection and full decoder tensor coverage from metadata.
    pub fn compile(declaration: &Declaration, catalog: &GgufCatalog) -> Result<Self, Error> {
        let config = declaration.config();
        if config.fsq_levels != [8, 7, 6, 6]
            || config.n_q != 12
            || config.quantizer_dim != 256
            || config.encoder_dim != 512
            || config.stft_n_fft != 1280
            || config.stft_win_length != 1280
            || config.stft_hop_length != 320
            || config.stft_frame_stack != 6
            || config.decoder_window != 30
            || config.sampling_rate != 24_000
        {
            return Err(invalid(
                "LiquidCodec geometry differs from the GGUF decoder",
            ));
        }
        let table = source(CODE_TABLE)
            .infer(catalog)
            .map_err(|error| invalid(error.to_string()))?;
        if table.shape != [2016, 4] || table.dtype != StoredDtype::F32.into() {
            return Err(invalid("invalid LiquidCodec GGUF scalar code table layout"));
        }
        let mut recipes = BTreeMap::new();
        for parameter in declaration.logical_parameters()? {
            let key = parameter.parameter.id.as_str();
            if !key.starts_with("decoder.")
                && !key.starts_with("from_quantizer.")
                && !key.starts_with("quantizer.project_out.")
            {
                continue;
            }
            let recipe = decoder_recipe(key, &parameter.shape)
                .ok_or_else(|| invalid(format!("no LiquidCodec GGUF projection for {key}")))?;
            if key != WINDOW {
                let inferred = recipe
                    .infer(catalog)
                    .map_err(|error| invalid(error.to_string()))?;
                if inferred.shape != parameter.shape {
                    return Err(invalid(format!(
                        "{key}: projected shape {:?}, expected {:?}",
                        inferred.shape, parameter.shape,
                    )));
                }
            }
            recipes.insert(key.to_owned(), recipe);
        }
        let covered = recipes
            .values()
            .flat_map(DerivedWeightRecipe::source_keys)
            .filter(|key| *key != WINDOW)
            .chain([CODE_TABLE])
            .collect::<std::collections::BTreeSet<_>>();
        let keys = catalog.keys();
        if covered
            != keys
                .iter()
                .map(String::as_str)
                .filter(|key| key.starts_with("a.gen.wav."))
                .collect()
        {
            return Err(invalid(
                "LiquidCodec GGUF decoder projection has incomplete coverage",
            ));
        }
        Ok(Self {
            config: config.clone(),
            recipes,
        })
    }

    /// Decoder recipes keyed by their canonical parameter names.
    pub fn recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.recipes
    }

    /// Verifies the derived scalar table and adds the reconstructed synthesis window.
    /// The decoder's learned parameters still use the original bounded GGUF leases.
    pub fn retain(
        self,
        projector: SharedCheckpointSource,
    ) -> Result<
        (
            SharedCheckpointSource,
            BTreeMap<String, DerivedWeightRecipe>,
        ),
        Error,
    > {
        let lease = projector
            .acquire_lease(TensorReadRequest {
                key: CODE_TABLE.into(),
                selection: TensorSelection::Full,
                policy: ReadPolicy::RequireBounded,
            })
            .map_err(|error| invalid(error.to_string()))?;
        let CheckpointLease::Gguf(lease) = lease else {
            return Err(invalid("expected a GGUF scalar code table lease"));
        };
        let ConvertedTensor::Dense(table) = lease
            .materialize_portable()
            .map_err(|error| invalid(error.to_string()))?
            .into_converted()
        else {
            return Err(invalid("expected a dense LiquidCodec scalar code table"));
        };
        let expected = (0..self.config.codebook_size()?)
            .flat_map(|index| self.config.scalar_code(index))
            .flat_map(f32::to_ne_bytes);
        if table.dtype != DenseDtype::F32 || !table.data.into_iter().eq(expected) {
            return Err(invalid(
                "LiquidCodec GGUF scalar code table differs from its geometry",
            ));
        }

        let length = self.config.stft_win_length as usize;
        // mtmd uses a periodic Hann with a double-precision angle passed to cosf.
        let bytes = (0..length)
            .flat_map(|index| {
                let angle = (std::f64::consts::TAU * index as f64 / length as f64) as f32;
                ((0.5 * (1.0 - f64::from(angle.cos()))) as f32).to_le_bytes()
            })
            .collect();
        let window: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([(
                WINDOW.into(),
                safetensors::Dtype::F32,
                vec![length],
                bytes,
            )])
            .map_err(|error| invalid(error.to_string()))?,
        );
        let source = CompositeCheckpointSource::new([projector, window])
            .map_err(|error| invalid(error.to_string()))?;
        Ok((Arc::new(source), self.recipes))
    }
}

fn source(key: impl Into<String>) -> DerivedWeightRecipe {
    DerivedWeightRecipe::source(key, TensorSelection::Full)
}

fn decoder_recipe(key: &str, shape: &[usize]) -> Option<DerivedWeightRecipe> {
    if key == WINDOW {
        return Some(source(WINDOW));
    }
    if key == "decoder.lfm.embedding_norm.weight" {
        return Some(source("a.gen.wav.tfm.output_norm.weight"));
    }
    for (canonical, physical) in [
        ("quantizer.project_out.", "a.gen.wav.quant.first.out_proj"),
        ("from_quantizer.", "a.gen.wav.tfm.in_proj"),
        ("decoder.lin.", "a.gen.wav.tfm.out_proj"),
    ] {
        if let Some(field @ ("weight" | "bias")) = key.strip_prefix(canonical) {
            return Some(source(format!("{physical}.{field}")));
        }
    }
    let (layer, field) = key.strip_prefix("decoder.lfm.layers.")?.split_once('.')?;
    let physical = match field.strip_suffix(".weight")? {
        "operator_norm" => "ln1",
        "post_operator_norm" => "attn_post_norm",
        "ffn_norm" => "ln2",
        "post_ffn_norm" => "ffn_post_norm",
        "self_attn.q_proj" => "attn_q",
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_proj" => "attn_k",
        "self_attn.k_norm" => "attn_k_norm",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_out",
        "self_attn.gate_proj" => "attn_gate",
        "conv.in_proj" => "conv_pw1",
        "conv.conv" => "conv_dw",
        "conv.out_proj" => "conv_pw2",
        "feed_forward.w1" => "ffn_gate",
        "feed_forward.w2" => "ffn_down",
        "feed_forward.w3" => "ffn_up",
        _ => return None,
    };
    let weight = source(format!("a.gen.wav.tfm.blk.{layer}.{physical}.weight"));
    Some(if physical == "conv_dw" {
        DerivedWeightRecipe::Reshape {
            input: Box::new(weight),
            shape: shape.to_vec(),
        }
    } else {
        weight
    })
}
