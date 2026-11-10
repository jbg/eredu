//! Complete SafeTensors physical contracts, separate from executable admission.
use super::SafetensorsTableSourcePlan;
use crate::qwen4_exp::config::Config;
use crate::qwen4_exp::feed_forward::FeedForwardSublayerSpec;
use crate::qwen4_exp::mtp::declarations::PredictionParameters;
use crate::qwen4_exp::target::declarations::{MixerParameters, TargetParameters, UnitParameters};
use eredu_checkpoint::{
    schema::{
        matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, LayoutVariant,
        MatrixScaleNames, SafetensorsCheckpointPlan, SafetensorsTensorConstraint as Constraint,
        StoredDtypeConstraint,
    },
    BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat, StoredDtype,
};
use eredu_nn::DeclareLogicalParameters;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Declared physical encoding and exact module exclusions, retained by preparation.
#[derive(Debug, Clone)]
pub struct SafetensorsEncoding {
    block: Option<BlockFp8Format>,
    excluded: BTreeSet<String>,
}

/// Normalizes only the released namespace aliases, never arbitrary suffix matches.
pub fn canonical_name(name: &str) -> String {
    if let Some(relative) = name.strip_prefix("model.language_model.") {
        format!("model.{relative}")
    } else if name.starts_with("layers.")
        || name.starts_with("embed_tokens.")
        || name.starts_with("hyper_connection_mixer.")
    {
        format!("model.{name}")
    } else {
        name.into()
    }
}
pub(in crate::qwen4_exp) fn aliases(name: &str) -> Vec<String> {
    if let Some(relative) = name
        .strip_prefix("model.")
        .filter(|s| !s.starts_with("visual."))
    {
        vec![format!("model.language_model.{relative}"), relative.into()]
    } else {
        Vec::new()
    }
}
impl SafetensorsEncoding {
    /// Parses the published block-FP8 policy without routing table scales through it.
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let Some(q) = root.get("quantization_config").filter(|v| !v.is_null()) else {
            return Ok(Self {
                block: None,
                excluded: BTreeSet::new(),
            });
        };
        if q.get("quant_method").and_then(Value::as_str) != Some("fp8")
            || q.get("activation_scheme").and_then(Value::as_str) != Some("dynamic")
            || q.get("weight_block_size") != Some(&serde_json::json!([128, 128]))
            || q.get("weight_per_tensor").and_then(Value::as_bool) != Some(false)
            || q.get("act_per_tensor").and_then(Value::as_bool) != Some(false)
        {
            return Err("qwen4_exp requires the released dynamic 128x128 E4M3 policy".into());
        }
        let excluded = q
            .get("modules_to_not_convert")
            .and_then(Value::as_array)
            .ok_or("FP8 policy omitted exact module exclusions")?
            .iter()
            .map(|v| {
                let name = v
                    .as_str()
                    .filter(|s| !s.is_empty() && !s.contains('*'))
                    .ok_or("FP8 exclusions must be literal module names")?;
                Ok(canonical_name(name))
            })
            .collect::<Result<BTreeSet<_>, String>>()?;
        if q.get("modules_to_convert")
            != Some(&serde_json::json!(["ple.ple_embedding.ngram_embedding"]))
        {
            return Err("FP8 policy must declare the separate n-gram table conversion".into());
        }
        Ok(Self {
            block: Some(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint)
                    .map_err(|e| e.to_string())?,
            ),
            excluded,
        })
    }
    /// Exact parameter format; exclusions apply to a module and its descendants.
    pub fn linear_format(&self, parameter: &str) -> LinearFormat {
        let canonical = canonical_name(parameter);
        let module = canonical.strip_suffix(".weight").unwrap_or(&canonical);
        if self.excluded.iter().any(|name| {
            module == name
                || module
                    .strip_prefix(name)
                    .is_some_and(|suffix| suffix.starts_with('.'))
        }) {
            LinearFormat::Dense
        } else {
            self.block
                .map_or(LinearFormat::Dense, LinearFormat::E4M3BlockFp8)
        }
    }
    /// Table rows use one shared scalar, independent of expert blocks.
    pub fn scalar_fp8_table(&self) -> bool {
        self.block.is_some()
    }
}

fn dense(tensors: &mut Vec<Constraint>, name: String, shape: Vec<usize>) {
    tensors.push(
        Constraint::required(&name, shape, StoredDtypeConstraint::Floating)
            .with_aliases(aliases(&name)),
    );
}
fn matrix(
    encoding: &SafetensorsEncoding,
    tensors: &mut Vec<Constraint>,
    name: String,
    shape: Vec<usize>,
) -> Result<(), String> {
    let scale = format!("{name}_scale_inv");
    tensors.extend(
        matrix_for_linear_format(
            &name,
            aliases(&name),
            shape,
            encoding.linear_format(&name),
            Some(MatrixScaleNames {
                key: scale.clone(),
                aliases: aliases(&scale),
            }),
        )
        .map_err(|e| e.to_string())?,
    );
    Ok(())
}
fn ordinary_parameters(
    spec: &impl DeclareLogicalParameters,
    encoding: &SafetensorsEncoding,
    tensors: &mut Vec<Constraint>,
) -> Result<(), String> {
    for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
        let name = parameter.parameter.id.to_string();
        if parameter.linear_format.is_some() {
            matrix(encoding, tensors, name, parameter.shape)?;
        } else {
            dense(tensors, name, parameter.shape);
        }
    }
    Ok(())
}

fn block(
    mixer: &MixerParameters,
    feed_forward: &FeedForwardSublayerSpec,
    encoding: &SafetensorsEncoding,
    tensors: &mut Vec<Constraint>,
    groups: &mut Vec<AlternativeLayoutGroup<Constraint>>,
) -> Result<(), String> {
    use super::experts::{sources, SourceLayout};
    ordinary_parameters(mixer, encoding, tensors)?;
    ordinary_parameters(&feed_forward.residual, encoding, tensors)?;
    let parameters = &feed_forward.feed_forward.parameters;
    let experts = &parameters.routed.experts;
    let packed = experts.packed_parameters().map_err(|e| e.to_string())?;
    let root = super::experts::source_root(&packed)?;
    let expert_slots = experts
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|d| d.parameter.id.clone())
        .collect::<BTreeSet<_>>();
    for parameter in parameters.logical_parameters().map_err(|e| e.to_string())? {
        if !expert_slots.contains(&parameter.parameter.id) {
            matrix(
                encoding,
                tensors,
                parameter.parameter.id.to_string(),
                parameter.shape,
            )?;
        }
    }
    let mut variants = Vec::new();
    for (id, layout) in [
        ("packed", SourceLayout::Packed),
        ("separate", SourceLayout::Separate),
    ] {
        let mut tensors = Vec::new();
        for source in sources(&parameters.routed.experts, layout)? {
            if !expert_slots.contains(&source.logical) {
                return Err(format!(
                    "source {} references unresolved expert {}",
                    source.name, source.logical
                ));
            }
            matrix(encoding, &mut tensors, source.name, source.shape)?;
        }
        variants.push(LayoutVariant {
            id: id.into(),
            discriminator_keys: tensors.iter().map(|t| t.key.clone()).collect(),
            tensors,
        });
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{root} storage"),
        required: true,
        variants,
    });
    Ok(())
}

/// Builds the full target, embedded prediction, vision and exact physical table
/// catalog. Table plans validate headers only; literal hash values are acquired
/// after the complete artifact has passed this contract.
pub(in crate::qwen4_exp) fn safetensors_plan(
    config: &Config,
    encoding: &SafetensorsEncoding,
    tables: &BTreeMap<usize, SafetensorsTableSourcePlan>,
    parameters: &TargetParameters,
    prediction: Option<&PredictionParameters>,
) -> Result<SafetensorsCheckpointPlan, String> {
    if tables.keys().copied().collect::<Vec<_>>() != config.ngram.layers {
        return Err("prepared tables do not match injection layers".into());
    }
    safetensors_plan_inner(config, encoding, Some(tables), parameters, prediction)
}

/// Preliminary configuration contract. Physical table geometry is added only
/// when the actual catalog is admitted; this schema cannot authorize loading.
pub(crate) fn safetensors_config_plan(
    declaration: &crate::qwen4_exp::Declaration,
    encoding: &SafetensorsEncoding,
) -> Result<SafetensorsCheckpointPlan, String> {
    safetensors_plan_inner(
        declaration.configuration(),
        encoding,
        None,
        &declaration.target_parameters,
        declaration.prediction_parameters.as_deref(),
    )
}

fn safetensors_plan_inner(
    config: &Config,
    encoding: &SafetensorsEncoding,
    tables: Option<&BTreeMap<usize, SafetensorsTableSourcePlan>>,
    parameters: &TargetParameters,
    prediction: Option<&PredictionParameters>,
) -> Result<SafetensorsCheckpointPlan, String> {
    let mut tensors = Vec::new();
    let mut groups = Vec::new();
    parameters.logical_parameters().map_err(|e| e.to_string())?;
    ordinary_parameters(&parameters.static_parameters, encoding, &mut tensors)?;
    // Source layout order is independent of execution order (lexical injection runs first).
    let mut units = parameters.units.iter().collect::<Vec<_>>();
    units.sort_by_key(|unit| match unit {
        UnitParameters::Decoder { layer, .. } => (*layer, 0),
        UnitParameters::Lexical { layer, .. } => (*layer, 1),
    });
    for unit in units {
        if let UnitParameters::Decoder {
            mixer,
            feed_forward,
            ..
        } = unit
        {
            block(mixer, feed_forward, encoding, &mut tensors, &mut groups)?;
        }
        if let UnitParameters::Lexical { layer, parameters } = unit {
            let root = format!("model.layers.{layer}");
            let table = tables.and_then(|tables| tables.get(layer));
            if table.is_some_and(|table| table.scalar_fp8() != encoding.scalar_fp8_table()) {
                return Err("table scalar encoding differs from the checkpoint policy".into());
            }
            ordinary_parameters(parameters, encoding, &mut tensors)?;
            let prefix = format!("{root}.ple.ple_embedding");
            for (name, count) in [
                ("layer_multipliers", config.ngram.order as usize),
                (
                    "ngram_heads_offsets",
                    ((config.ngram.order - 1) * config.ngram.heads) as usize,
                ),
                (
                    "ngram_heads_vocab_sizes",
                    ((config.ngram.order - 1) * config.ngram.heads) as usize,
                ),
            ] {
                let name = format!("{prefix}.{name}");
                tensors.push(
                    Constraint::required(
                        &name,
                        vec![count],
                        StoredDtypeConstraint::Exact(StoredDtype::I64),
                    )
                    .with_aliases(aliases(&name)),
                );
            }
            if let Some(table) = table {
                for name in table.shards.iter().chain(table.scale_name.iter()) {
                    let meta = &table.catalog[name];
                    let canonical = canonical_name(name);
                    tensors.push(
                        Constraint::required(
                            &canonical,
                            meta.shape.clone(),
                            StoredDtypeConstraint::Exact(meta.stored_dtype.clone()),
                        )
                        .with_aliases(aliases(&canonical)),
                    );
                }
            }
        }
    }
    prediction_tensors(prediction, encoding, &mut tensors, &mut groups)?;
    if let Some(vision) = &config.vision {
        if vision.root() != "model.visual" {
            return Err("conditional vision namespace differs from component".into());
        }
        // Vision retains its declared dense formats under the released source policy.
        let parameters = vision.as_ref();
        for parameter in parameters
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            if parameter.linear_format.is_some()
                && encoding.linear_format(parameter.parameter.id.as_str()) != LinearFormat::Dense
            {
                return Err(format!(
                    "visual parameter {} must retain its declared dense encoding",
                    parameter.parameter.id
                ));
            }
        }
        let vision = crate::qwen::vision::safetensors_plan(&parameters)?;
        tensors.extend(vision.common_tensors);
        groups.extend(vision.layout_groups);
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_suffixes.push("rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("qwen4_exp SafeTensors", tensors, groups, policy)
        .map_err(|e| e.to_string())
}

fn prediction_tensors(
    parameters: Option<&PredictionParameters>,
    encoding: &SafetensorsEncoding,
    tensors: &mut Vec<Constraint>,
    groups: &mut Vec<AlternativeLayoutGroup<Constraint>>,
) -> Result<(), String> {
    if let Some(parameters) = parameters {
        parameters.logical_parameters().map_err(|e| e.to_string())?;
        ordinary_parameters(&parameters.shared, encoding, tensors)?;
        for unit in &parameters.units {
            let UnitParameters::Decoder {
                mixer,
                feed_forward,
                ..
            } = unit
            else {
                return Err("prediction cannot own lexical parameters".into());
            };
            block(mixer, feed_forward, encoding, tensors, groups)?;
        }
    }
    Ok(())
}

/// Exact prediction role, usable for a standalone MTP artifact or a restricted
/// view of a complete official checkpoint. Vocabulary weights belong to the target.
pub(in crate::qwen4_exp) fn prediction_safetensors_plan(
    parameters: &PredictionParameters,
    encoding: &SafetensorsEncoding,
) -> Result<SafetensorsCheckpointPlan, String> {
    let mut tensors = Vec::new();
    let mut groups = Vec::new();
    prediction_tensors(Some(parameters), encoding, &mut tensors, &mut groups)?;
    let mut policy = CatalogPolicy::strict();
    policy.allowed_suffixes.push("rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("qwen4_exp prediction SafeTensors", tensors, groups, policy)
        .map_err(|e| e.to_string())
}

/// Source-free logical target authority; physical formats project after exact admission.
pub(in crate::qwen4_exp) fn target_parameters(config: &Config) -> Result<TargetParameters, String> {
    TargetParameters::from_config(
        config,
        |layer, _| {
            crate::qwen4_exp::feed_forward::expert_spec(
                config,
                &format!("model.layers.{layer}.mlp.experts"),
                |_, _| eredu_nn::LinearFormatSpec::unscaled(LinearFormat::Dense),
            )
        },
        |_| eredu_nn::LinearFormatSpec::unscaled(LinearFormat::Dense),
    )
    .map_err(|e| e.to_string())
}

pub(in crate::qwen4_exp) fn prediction_parameters(
    config: &Config,
) -> Result<PredictionParameters, String> {
    PredictionParameters::from_config(
        config,
        |depth| {
            crate::qwen4_exp::feed_forward::expert_spec(
                config,
                &format!("mtp.layers.{depth}.mlp.experts"),
                |_, _| eredu_nn::LinearFormatSpec::unscaled(LinearFormat::Dense),
            )
        },
        |_| eredu_nn::LinearFormatSpec::unscaled(LinearFormat::Dense),
    )
    .map_err(|e| e.to_string())
}
