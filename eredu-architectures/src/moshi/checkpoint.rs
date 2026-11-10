//! Strict physical catalogs and canonical recipes for Moshi-family checkpoints.

use eredu_checkpoint::{
    recipe::{
        AtomicRecipeSet, DerivedWeightRecipe, MatrixRecipeMember, RecipeAlias, RecipeCatalog,
    },
    schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    },
    store::TensorSelection,
    AtomicMatrixRecipeFamily, StoredDtype, WeightQuantization,
};

use super::{CheckpointLayout, MoshiConfig};

/// Builds the strict physical SafeTensors catalog selected by normalized metadata.
pub fn safetensors_plan(
    contract: &super::MoshiParameterContract,
) -> Result<SafetensorsCheckpointPlan, String> {
    let config = contract.configuration();
    let declarations = contract.declarations();
    let mut physical = std::collections::BTreeMap::new();
    for extension in storage_extensions(config, declarations)? {
        if let Some(source) = extension.source {
            if let Some(previous) = physical.insert(source.key.clone(), source.clone()) {
                if previous != source {
                    return Err(format!(
                        "conflicting Moshi physical declarations for {}",
                        source.key
                    ));
                }
            }
        }
    }
    let mut tensors = Vec::new();
    for source in physical.into_values() {
        if source.matrix {
            matrix_constraint(
                config,
                &mut tensors,
                &source.key,
                source.aliases,
                &source.companion_prefix,
                source.shape,
            )?;
        } else {
            tensors.push(vector(source.key, source.shape[0], source.flattened));
        }
    }
    SafetensorsCheckpointPlan::new(
        match config.checkpoint_layout() {
            CheckpointLayout::MoshiSafetensors => "Moshi SafeTensors",
            CheckpointLayout::PersonaPlexPytorch => "PersonaPlex released PyTorch SafeTensors",
        },
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|e| e.to_string())
}

/// Maps either physical layout to one canonical Moshi parameter publication.
/// No output or alias escapes until every source and companion has validated.
pub fn canonical_recipes<C: RecipeCatalog + ?Sized>(
    contract: &super::MoshiParameterContract,
    catalog: &C,
) -> Result<AtomicRecipeSet, String> {
    let config = contract.configuration();
    let declarations = contract.declarations();
    let mut builder = RecipeBuilder::new(config, catalog);
    for extension in storage_extensions(config, declarations)? {
        let target = extension.parameter.id.as_str();
        if let Some(owner) = &extension.parameter.alias_of {
            builder.alias(target, owner.as_str());
            continue;
        }
        let source = extension
            .source
            .expect("owned Moshi parameter has a source");
        let names = std::iter::once(source.key.clone()).chain(source.aliases.clone());
        if source.matrix {
            builder.matrix(target, names, &source.companion_prefix, extension.rows)?;
        } else {
            builder.vector(target, names, source.shape[0])?;
        }
    }
    builder.publish()
}
#[derive(Clone, Debug, Eq, PartialEq)]
struct PhysicalSource {
    key: String,
    aliases: Vec<String>,
    companion_prefix: String,
    shape: Vec<usize>,
    matrix: bool,
    flattened: bool,
}
struct StorageExtension<'a> {
    parameter: &'a eredu_nn::ParameterSpec,
    source: Option<PhysicalSource>,
    rows: Option<std::ops::Range<usize>>,
}

fn storage_extensions<'a>(
    config: &MoshiConfig,
    declarations: &'a super::declarations::Declarations,
) -> Result<Vec<StorageExtension<'a>>, String> {
    use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};
    declarations
        .logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|declaration| {
            let parameter = declaration.parameter;
            if parameter.alias_of.is_some() {
                return Ok(StorageExtension {
                    parameter,
                    source: None,
                    rows: None,
                });
            }
            let canonical = parameter.id.as_str();
            let matrix = matches!(
                declaration.role,
                LogicalParameterRole::LinearWeight | LogicalParameterRole::Embedding
            );
            if !matrix && declaration.role != LogicalParameterRole::Normalization {
                return Err(format!("unsupported Moshi parameter role for {canonical}"));
            }
            let mut source = PhysicalSource {
                key: canonical.into(),
                aliases: Vec::new(),
                companion_prefix: canonical.trim_end_matches(".weight").into(),
                shape: declaration.shape,
                matrix,
                flattened: false,
            };
            let mut rows = None;
            if config.checkpoint_layout() == CheckpointLayout::PersonaPlexPytorch {
                if canonical == "out_norm.weight" {
                    source.key = "out_norm.alpha".into();
                    source.flattened = true;
                } else if let Some(local) = canonical.strip_prefix("audio_embs.") {
                    source.key = format!("emb.{local}");
                } else if let Some(local) = canonical.strip_prefix("depformer.slices.") {
                    let (slice, local) = local
                        .split_once('.')
                        .ok_or_else(|| format!("invalid Moshi depth identity {canonical}"))?;
                    let slice: usize = slice
                        .parse()
                        .map_err(|_| format!("invalid Moshi depth slice {canonical}"))?;
                    source.key = match local {
                        "emb.weight" if slice == 0 => "depformer_text_emb.weight".into(),
                        "emb.weight" => format!("depformer_emb.{}.weight", slice - 1),
                        "linear_in.weight" => format!("depformer_in.{slice}.weight"),
                        "linear_out.weight" => format!("linears.{slice}.weight"),
                        _ => {
                            let local =
                                local.strip_prefix("transformer.layers.").ok_or_else(|| {
                                    format!("unknown Moshi depth parameter {canonical}")
                                })?;
                            let (layer, local) = local
                                .split_once('.')
                                .ok_or_else(|| format!("invalid Moshi depth block {canonical}"))?;
                            let root = format!("depformer.layers.{layer}");
                            match local {
                                "norm1.weight" | "norm2.weight" => {
                                    source.flattened = true;
                                    format!("{root}.{}.alpha", local.trim_end_matches(".weight"))
                                }
                                "self_attn.in_proj.weight" | "self_attn.out_proj.weight" => {
                                    let width = source.shape[0];
                                    let start = mul(slice, width, "depth projection row offset")?;
                                    rows =
                                        Some(start..add(start, width, "depth projection row end")?);
                                    source.shape[0] = mul(
                                        declarations.depth.len(),
                                        width,
                                        "packed depth projection rows",
                                    )?;
                                    format!("{root}.{local}")
                                }
                                "gating.linear_in.weight" | "gating.linear_out.weight" => format!(
                                    "{root}.gating.{slice}.{}",
                                    local.strip_prefix("gating.").unwrap()
                                ),
                                _ => {
                                    return Err(format!(
                                        "unknown Moshi depth parameter {canonical}"
                                    ));
                                }
                            }
                        }
                    };
                }
                // Physical alpha tensors and packed QKV spellings are source-format rules.
                if source.key.starts_with("transformer.layers.") && !matrix {
                    source.key = format!("{}.alpha", source.key.trim_end_matches(".weight"));
                    source.flattened = true;
                }
                if source.key.ends_with(".self_attn.in_proj.weight") {
                    let dotted = source.key.clone();
                    source.key = format!("{}_weight", dotted.trim_end_matches(".weight"));
                    source.aliases.push(dotted);
                }
                source.companion_prefix = source
                    .key
                    .strip_suffix("_weight")
                    .filter(|_| source.key.ends_with(".in_proj_weight"))
                    .unwrap_or_else(|| source.key.trim_end_matches(".weight"))
                    .into();
            }
            Ok(StorageExtension {
                parameter,
                source: Some(source),
                rows,
            })
        })
        .collect()
}

fn matrix_constraint(
    config: &MoshiConfig,
    out: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    aliases: impl IntoIterator<Item = String>,
    companion_prefix: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    let aliases = aliases.into_iter().collect::<Vec<_>>();
    let Some(quantization) = config.native_quantization() else {
        out.push(
            SafetensorsTensorConstraint::required(name, shape, StoredDtypeConstraint::Floating)
                .with_aliases(aliases)
                .with_linear_format(eredu_checkpoint::LinearFormat::Dense),
        );
        return Ok(());
    };
    if matches!(quantization, WeightQuantization::GgufIQuant { .. }) {
        return Err("Moshi SafeTensors cannot use GGUF quantization".into());
    }
    let dtype = match config.checkpoint_layout() {
        CheckpointLayout::PersonaPlexPytorch => StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ]),
        CheckpointLayout::MoshiSafetensors => match quantization {
            WeightQuantization::Affine(_) => StoredDtypeConstraint::Floating,
            WeightQuantization::MxFp4 => StoredDtypeConstraint::Exact(StoredDtype::U8),
            WeightQuantization::NvFp4 => return Err("Moshi NVFP4 layout is not implemented".into()),
            WeightQuantization::GgufIQuant { .. } => unreachable!(),
        },
    };
    let mut constraints =
        eredu_checkpoint::schema::matrix_for_packed_words(name, aliases, shape, quantization, None)
            .map_err(|error| error.to_string())?;
    for constraint in &mut constraints {
        match &constraint.linear_companion {
            None => constraint.linear_format = Some(quantization.into()),
            Some(companion) => {
                use eredu_checkpoint::schema::LinearCompanionKind;
                let suffix = match companion.kind {
                    LinearCompanionKind::Scale => "scales",
                    LinearCompanionKind::AffineBias => "biases",
                    LinearCompanionKind::GlobalScale => {
                        return Err("Moshi packed source has no global scale companion".into());
                    }
                };
                // Published companion names and dtypes are source-layout rules.
                // Weight aliases do not grant additional companion spellings.
                constraint.key = format!("{companion_prefix}.{suffix}");
                constraint.aliases.clear();
                constraint.dtype = dtype.clone();
            }
        }
    }
    out.extend(constraints);
    Ok(())
}

fn vector(
    name: impl Into<String>,
    elements: usize,
    flattened: bool,
) -> SafetensorsTensorConstraint {
    let value = SafetensorsTensorConstraint::required(
        name,
        vec![elements],
        StoredDtypeConstraint::Floating,
    );
    if flattened {
        value.with_element_count(elements)
    } else {
        value
    }
}

struct RecipeBuilder<'a, C: ?Sized> {
    config: &'a MoshiConfig,
    catalog: &'a C,
    outputs: Vec<(String, DerivedWeightRecipe)>,
    aliases: Vec<RecipeAlias>,
}

impl<'a, C: RecipeCatalog + ?Sized> RecipeBuilder<'a, C> {
    fn new(config: &'a MoshiConfig, catalog: &'a C) -> Self {
        Self {
            config,
            catalog,
            outputs: Vec::new(),
            aliases: Vec::new(),
        }
    }
    fn vector<S: Into<String>>(
        &mut self,
        target: &str,
        sources: impl IntoIterator<Item = S>,
        elements: usize,
    ) -> Result<(), String> {
        let source = resolve_source(self.catalog, sources.into_iter().map(Into::into))?;
        let recipe = DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::source(source, TensorSelection::Full)),
            shape: vec![elements],
        };
        recipe
            .infer(self.catalog)
            .map_err(|error| error.to_string())?;
        self.outputs.push((target.into(), recipe));
        Ok(())
    }
    fn matrix<S: Into<String>>(
        &mut self,
        target: &str,
        sources: impl IntoIterator<Item = S>,
        companion_prefix: &str,
        rows: Option<std::ops::Range<usize>>,
    ) -> Result<(), String> {
        let source = resolve_source(self.catalog, sources.into_iter().map(Into::into))?;
        let target_prefix = target.strip_suffix(".weight").unwrap_or(target);
        let weight = MatrixRecipeMember::new(
            target,
            DerivedWeightRecipe::source(source, TensorSelection::Full),
        );
        let (scales, biases) = match self.config.native_quantization() {
            None => (None, None),
            Some(quantization) => (
                Some(MatrixRecipeMember::new(
                    format!("{target_prefix}.scales"),
                    DerivedWeightRecipe::source(
                        format!("{companion_prefix}.scales"),
                        TensorSelection::Full,
                    ),
                )),
                quantization.has_biases().then(|| {
                    MatrixRecipeMember::new(
                        format!("{target_prefix}.biases"),
                        DerivedWeightRecipe::source(
                            format!("{companion_prefix}.biases"),
                            TensorSelection::Full,
                        ),
                    )
                }),
            ),
        };
        let family = AtomicMatrixRecipeFamily::new(self.catalog, weight, scales, biases)
            .map_err(|error| error.to_string())?;
        let family = if let Some(rows) = rows {
            family
                .select_leading_axis(
                    self.catalog,
                    TensorSelection::Range {
                        axis: 0,
                        start: rows.start,
                        end: rows.end,
                    },
                )
                .map_err(|error| error.to_string())?
        } else {
            family
        };
        self.outputs.extend(
            family
                .publish(self.catalog, std::iter::empty())
                .map_err(|error| error.to_string())?
                .into_outputs(),
        );
        Ok(())
    }
    fn alias(&mut self, alias: impl Into<String>, destination: impl Into<String>) {
        self.aliases.push(RecipeAlias::new(alias, destination));
    }
    fn publish(self) -> Result<AtomicRecipeSet, String> {
        AtomicRecipeSet::new_with_aliases(self.catalog, self.outputs, self.aliases)
            .map_err(|error| error.to_string())
    }
}

fn resolve_source<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    sources: impl IntoIterator<Item = String>,
) -> Result<String, String> {
    let sources = sources.into_iter().collect::<Vec<_>>();
    let present = sources
        .iter()
        .filter(|key| catalog.tensor_metadata(key).is_ok())
        .cloned()
        .collect::<Vec<_>>();
    match present.as_slice() {
        [source] => Ok(source.clone()),
        [] => Err(format!(
            "checkpoint is missing required tensor; expected one of {sources:?}"
        )),
        _ => Err(format!(
            "checkpoint contains colliding physical aliases {present:?}"
        )),
    }
}

fn add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Moshi {name} geometry overflows"))
}
fn mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Moshi {name} geometry overflows"))
}
