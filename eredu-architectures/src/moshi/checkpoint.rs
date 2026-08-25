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
pub fn safetensors_plan(config: &MoshiConfig) -> Result<SafetensorsCheckpointPlan, String> {
    let g = Geometry::new(config)?;
    let mut tensors = Vec::new();
    match config.checkpoint_layout() {
        CheckpointLayout::MoshiSafetensors => native_catalog(config, g, &mut tensors)?,
        CheckpointLayout::PersonaPlexPytorch => personaplex_catalog(config, g, &mut tensors)?,
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
    .map_err(|error| error.to_string())
}

/// Maps either physical layout to one canonical Moshi parameter publication.
/// No output or alias escapes until every source and companion has validated.
pub fn canonical_recipes<C: RecipeCatalog + ?Sized>(
    config: &MoshiConfig,
    catalog: &C,
) -> Result<AtomicRecipeSet, String> {
    let g = Geometry::new(config)?;
    let mut builder = RecipeBuilder::new(config, catalog);
    match config.checkpoint_layout() {
        CheckpointLayout::MoshiSafetensors => native_recipes(g, &mut builder)?,
        CheckpointLayout::PersonaPlexPytorch => personaplex_recipes(g, &mut builder)?,
    }
    builder.publish()
}

#[derive(Clone, Copy)]
struct Geometry {
    temporal_dim: usize,
    temporal_hidden: usize,
    temporal_layers: usize,
    depth_dim: usize,
    depth_hidden: usize,
    depth_layers: usize,
    text_vocab: usize,
    audio_vocab: usize,
    audio_codebooks: usize,
    depth_slices: usize,
}

impl Geometry {
    fn new(config: &MoshiConfig) -> Result<Self, String> {
        Ok(Self {
            temporal_dim: dim(config.temporal().hidden_size(), "temporal hidden size")?,
            temporal_hidden: dim(
                config.temporal().gated_hidden_size(),
                "temporal gated width",
            )?,
            temporal_layers: dim(
                config.temporal().num_hidden_layers(),
                "temporal layer count",
            )?,
            depth_dim: dim(config.depth_template().hidden_size(), "depth hidden size")?,
            depth_hidden: dim(
                config.depth_template().gated_hidden_size(),
                "depth gated width",
            )?,
            depth_layers: dim(
                config.depth_template().num_hidden_layers(),
                "depth layer count",
            )?,
            text_vocab: dim(config.text_vocabulary_size(), "text vocabulary size")?,
            audio_vocab: dim(config.audio_vocabulary_size(), "audio vocabulary size")?,
            audio_codebooks: config.frame_schedule().total_audio_codebooks(),
            depth_slices: config.frame_schedule().depth_audio_codebooks(),
        })
    }
    fn text_input(self) -> Result<usize, String> {
        add(self.text_vocab, 1, "text input vocabulary")
    }
    fn audio_input(self) -> Result<usize, String> {
        add(self.audio_vocab, 1, "audio input vocabulary")
    }
}

fn native_catalog(
    config: &MoshiConfig,
    g: Geometry,
    out: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    matrix_constraint(
        config,
        out,
        "text_emb.weight",
        [],
        "text_emb",
        vec![g.text_input()?, g.temporal_dim],
    )?;
    matrix_constraint(
        config,
        out,
        "text_linear.weight",
        [],
        "text_linear",
        vec![g.text_vocab, g.temporal_dim],
    )?;
    out.push(vector("out_norm.weight", g.temporal_dim, false));
    for codebook in 0..g.audio_codebooks {
        let name = format!("audio_embs.{codebook}.weight");
        matrix_constraint(
            config,
            out,
            &name,
            [],
            name.trim_end_matches(".weight"),
            vec![g.audio_input()?, g.temporal_dim],
        )?;
    }
    for layer in 0..g.temporal_layers {
        block_catalog(
            config,
            out,
            &format!("transformer.layers.{layer}"),
            g.temporal_dim,
            g.temporal_hidden,
            false,
        )?;
    }
    for slice in 0..g.depth_slices {
        let root = format!("depformer.slices.{slice}");
        for (local, shape) in [
            (
                "emb",
                vec![
                    if slice == 0 {
                        g.text_input()?
                    } else {
                        g.audio_input()?
                    },
                    g.depth_dim,
                ],
            ),
            ("linear_in", vec![g.depth_dim, g.temporal_dim]),
            ("linear_out", vec![g.audio_vocab, g.depth_dim]),
        ] {
            let name = format!("{root}.{local}.weight");
            matrix_constraint(
                config,
                out,
                &name,
                [],
                name.trim_end_matches(".weight"),
                shape,
            )?;
        }
        for layer in 0..g.depth_layers {
            block_catalog(
                config,
                out,
                &format!("{root}.transformer.layers.{layer}"),
                g.depth_dim,
                g.depth_hidden,
                false,
            )?;
        }
    }
    Ok(())
}

fn personaplex_catalog(
    config: &MoshiConfig,
    g: Geometry,
    out: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    matrix_constraint(
        config,
        out,
        "text_emb.weight",
        [],
        "text_emb",
        vec![g.text_input()?, g.temporal_dim],
    )?;
    matrix_constraint(
        config,
        out,
        "text_linear.weight",
        [],
        "text_linear",
        vec![g.text_vocab, g.temporal_dim],
    )?;
    out.push(vector("out_norm.alpha", g.temporal_dim, true));
    for codebook in 0..g.audio_codebooks {
        let name = format!("emb.{codebook}.weight");
        matrix_constraint(
            config,
            out,
            &name,
            [],
            name.trim_end_matches(".weight"),
            vec![g.audio_input()?, g.temporal_dim],
        )?;
    }
    for layer in 0..g.temporal_layers {
        block_catalog(
            config,
            out,
            &format!("transformer.layers.{layer}"),
            g.temporal_dim,
            g.temporal_hidden,
            true,
        )?;
    }
    for slice in 0..g.depth_slices {
        let embedding = if slice == 0 {
            "depformer_text_emb.weight".into()
        } else {
            format!("depformer_emb.{}.weight", slice - 1)
        };
        for (name, shape) in [
            (
                embedding,
                vec![
                    if slice == 0 {
                        g.text_input()?
                    } else {
                        g.audio_input()?
                    },
                    g.depth_dim,
                ],
            ),
            (
                format!("depformer_in.{slice}.weight"),
                vec![g.depth_dim, g.temporal_dim],
            ),
            (
                format!("linears.{slice}.weight"),
                vec![g.audio_vocab, g.depth_dim],
            ),
        ] {
            matrix_constraint(
                config,
                out,
                &name,
                [],
                name.trim_end_matches(".weight"),
                shape,
            )?;
        }
    }
    for layer in 0..g.depth_layers {
        let root = format!("depformer.layers.{layer}");
        out.push(vector(format!("{root}.norm1.alpha"), g.depth_dim, true));
        out.push(vector(format!("{root}.norm2.alpha"), g.depth_dim, true));
        matrix_constraint(
            config,
            out,
            &format!("{root}.self_attn.in_proj_weight"),
            [format!("{root}.self_attn.in_proj.weight")],
            &format!("{root}.self_attn.in_proj"),
            vec![
                mul(
                    mul(g.depth_slices, 3, "depth QKV count")?,
                    g.depth_dim,
                    "packed depth QKV rows",
                )?,
                g.depth_dim,
            ],
        )?;
        let output = format!("{root}.self_attn.out_proj.weight");
        matrix_constraint(
            config,
            out,
            &output,
            [],
            output.trim_end_matches(".weight"),
            vec![
                mul(g.depth_slices, g.depth_dim, "packed depth output rows")?,
                g.depth_dim,
            ],
        )?;
        for slice in 0..g.depth_slices {
            for (local, shape) in [
                (
                    "linear_in",
                    vec![mul(2, g.depth_hidden, "depth gated rows")?, g.depth_dim],
                ),
                ("linear_out", vec![g.depth_dim, g.depth_hidden]),
            ] {
                let name = format!("{root}.gating.{slice}.{local}.weight");
                matrix_constraint(
                    config,
                    out,
                    &name,
                    [],
                    name.trim_end_matches(".weight"),
                    shape,
                )?;
            }
        }
    }
    Ok(())
}

fn block_catalog(
    config: &MoshiConfig,
    out: &mut Vec<SafetensorsTensorConstraint>,
    root: &str,
    hidden: usize,
    gated: usize,
    pytorch: bool,
) -> Result<(), String> {
    let suffix = if pytorch { "alpha" } else { "weight" };
    out.push(vector(format!("{root}.norm1.{suffix}"), hidden, pytorch));
    out.push(vector(format!("{root}.norm2.{suffix}"), hidden, pytorch));
    let input = if pytorch {
        format!("{root}.self_attn.in_proj_weight")
    } else {
        format!("{root}.self_attn.in_proj.weight")
    };
    let aliases = pytorch.then(|| format!("{root}.self_attn.in_proj.weight"));
    matrix_constraint(
        config,
        out,
        &input,
        aliases,
        &format!("{root}.self_attn.in_proj"),
        vec![mul(3, hidden, "QKV rows")?, hidden],
    )?;
    for (local, shape) in [
        ("self_attn.out_proj", vec![hidden, hidden]),
        (
            "gating.linear_in",
            vec![mul(2, gated, "gated rows")?, hidden],
        ),
        ("gating.linear_out", vec![hidden, gated]),
    ] {
        let name = format!("{root}.{local}.weight");
        matrix_constraint(
            config,
            out,
            &name,
            [],
            name.trim_end_matches(".weight"),
            shape,
        )?;
    }
    Ok(())
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
                .with_aliases(aliases),
        );
        return Ok(());
    };
    if matches!(quantization, WeightQuantization::GgufIQuant { .. }) {
        return Err("Moshi SafeTensors cannot use GGUF quantization".into());
    }
    let input = *shape
        .last()
        .ok_or_else(|| format!("matrix {name:?} is scalar"))?;
    let group = dim(quantization.group_size(), "quantization group size")?;
    let bits = dim(quantization.bits(), "quantization bit width")?;
    let packed_bits = mul(input, bits, "packed input")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!("quantized Moshi tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"));
    }
    let mut packed = shape.clone();
    *packed.last_mut().unwrap() = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().unwrap() = input / group;
    out.push(
        SafetensorsTensorConstraint::required(
            name,
            packed,
            StoredDtypeConstraint::Exact(StoredDtype::U32),
        )
        .with_aliases(aliases),
    );
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
            WeightQuantization::GgufIQuant { .. } => unreachable!(),
        },
    };
    out.push(
        SafetensorsTensorConstraint::required(
            format!("{companion_prefix}.scales"),
            companion.clone(),
            dtype.clone(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        out.push(
            SafetensorsTensorConstraint::required(
                format!("{companion_prefix}.biases"),
                companion,
                dtype,
            )
            .companion(),
        );
    }
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

fn native_recipes<C: RecipeCatalog + ?Sized>(
    g: Geometry,
    b: &mut RecipeBuilder<'_, C>,
) -> Result<(), String> {
    b.matrix("text_emb.weight", ["text_emb.weight"], "text_emb", None)?;
    b.matrix(
        "text_linear.weight",
        ["text_linear.weight"],
        "text_linear",
        None,
    )?;
    b.vector("out_norm.weight", ["out_norm.weight"], g.temporal_dim)?;
    for codebook in 0..g.audio_codebooks {
        let name = format!("audio_embs.{codebook}.weight");
        b.matrix(&name, [&name], name.trim_end_matches(".weight"), None)?;
    }
    for layer in 0..g.temporal_layers {
        let root = format!("transformer.layers.{layer}");
        direct_block(g.temporal_dim, &root, b)?;
    }
    for slice in 0..g.depth_slices {
        let root = format!("depformer.slices.{slice}");
        for local in ["emb", "linear_in", "linear_out"] {
            let name = format!("{root}.{local}.weight");
            b.matrix(&name, [&name], name.trim_end_matches(".weight"), None)?;
        }
        for layer in 0..g.depth_layers {
            direct_block(
                g.depth_dim,
                &format!("{root}.transformer.layers.{layer}"),
                b,
            )?;
        }
    }
    Ok(())
}

fn personaplex_recipes<C: RecipeCatalog + ?Sized>(
    g: Geometry,
    b: &mut RecipeBuilder<'_, C>,
) -> Result<(), String> {
    b.matrix("text_emb.weight", ["text_emb.weight"], "text_emb", None)?;
    b.matrix(
        "text_linear.weight",
        ["text_linear.weight"],
        "text_linear",
        None,
    )?;
    b.vector("out_norm.weight", ["out_norm.alpha"], g.temporal_dim)?;
    for codebook in 0..g.audio_codebooks {
        b.matrix(
            &format!("audio_embs.{codebook}.weight"),
            [format!("emb.{codebook}.weight")],
            &format!("emb.{codebook}"),
            None,
        )?;
    }
    for layer in 0..g.temporal_layers {
        let root = format!("transformer.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            b.vector(
                &format!("{root}.{norm}.weight"),
                [format!("{root}.{norm}.alpha")],
                g.temporal_dim,
            )?;
        }
        let canonical = format!("{root}.self_attn.in_proj.weight");
        let underscored = format!("{root}.self_attn.in_proj_weight");
        b.matrix(
            &canonical,
            [underscored.clone(), canonical.clone()],
            &format!("{root}.self_attn.in_proj"),
            None,
        )?;
        for local in [
            "self_attn.out_proj",
            "gating.linear_in",
            "gating.linear_out",
        ] {
            let name = format!("{root}.{local}.weight");
            b.matrix(&name, [&name], name.trim_end_matches(".weight"), None)?;
        }
    }
    for slice in 0..g.depth_slices {
        let root = format!("depformer.slices.{slice}");
        let embedding = if slice == 0 {
            "depformer_text_emb.weight".into()
        } else {
            format!("depformer_emb.{}.weight", slice - 1)
        };
        for (local, source) in [
            ("emb", embedding),
            ("linear_in", format!("depformer_in.{slice}.weight")),
            ("linear_out", format!("linears.{slice}.weight")),
        ] {
            b.matrix(
                &format!("{root}.{local}.weight"),
                [source.clone()],
                source.trim_end_matches(".weight"),
                None,
            )?;
        }
    }
    for layer in 0..g.depth_layers {
        let physical = format!("depformer.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            let owner = format!("depformer.slices.0.transformer.layers.{layer}.{norm}.weight");
            b.vector(&owner, [format!("{physical}.{norm}.alpha")], g.depth_dim)?;
            for slice in 1..g.depth_slices {
                b.alias(
                    format!("depformer.slices.{slice}.transformer.layers.{layer}.{norm}.weight"),
                    owner.clone(),
                );
            }
        }
        for slice in 0..g.depth_slices {
            let canonical = format!("depformer.slices.{slice}.transformer.layers.{layer}");
            b.matrix(
                &format!("{canonical}.self_attn.in_proj.weight"),
                [
                    format!("{physical}.self_attn.in_proj_weight"),
                    format!("{physical}.self_attn.in_proj.weight"),
                ],
                &format!("{physical}.self_attn.in_proj"),
                Some(slice * 3 * g.depth_dim..(slice + 1) * 3 * g.depth_dim),
            )?;
            b.matrix(
                &format!("{canonical}.self_attn.out_proj.weight"),
                [format!("{physical}.self_attn.out_proj.weight")],
                &format!("{physical}.self_attn.out_proj"),
                Some(slice * g.depth_dim..(slice + 1) * g.depth_dim),
            )?;
            for local in ["linear_in", "linear_out"] {
                let source = format!("{physical}.gating.{slice}.{local}.weight");
                b.matrix(
                    &format!("{canonical}.gating.{local}.weight"),
                    [source.clone()],
                    source.trim_end_matches(".weight"),
                    None,
                )?;
            }
        }
    }
    Ok(())
}

fn direct_block<C: RecipeCatalog + ?Sized>(
    hidden: usize,
    root: &str,
    b: &mut RecipeBuilder<'_, C>,
) -> Result<(), String> {
    for norm in ["norm1", "norm2"] {
        let name = format!("{root}.{norm}.weight");
        b.vector(&name, [name.clone()], hidden)?;
    }
    for local in [
        "self_attn.in_proj",
        "self_attn.out_proj",
        "gating.linear_in",
        "gating.linear_out",
    ] {
        let name = format!("{root}.{local}.weight");
        b.matrix(
            &name,
            [name.clone()],
            name.trim_end_matches(".weight"),
            None,
        )?;
    }
    Ok(())
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

fn dim(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Moshi {name} must be positive, got {value}"))
}
fn add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Moshi {name} geometry overflows"))
}
fn mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Moshi {name} geometry overflows"))
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Mutex};

    use eredu_checkpoint::{
        recipe::{BoundedRecipeSource, RecipeError},
        store::{StoreError, TensorMetadata},
    };

    use super::*;

    #[derive(Default)]
    struct Catalog {
        tensors: BTreeMap<String, TensorMetadata>,
        reads: Mutex<Vec<(String, TensorSelection)>>,
    }

    impl Catalog {
        fn from_plan(plan: &SafetensorsCheckpointPlan) -> Self {
            let tensors = plan
                .common_tensors
                .iter()
                .map(|tensor| {
                    let dtype = match &tensor.dtype {
                        StoredDtypeConstraint::Exact(dtype) => dtype.clone(),
                        StoredDtypeConstraint::Floating => StoredDtype::F32,
                        StoredDtypeConstraint::OneOf(dtypes) => dtypes[0].clone(),
                    };
                    (
                        tensor.key.clone(),
                        TensorMetadata {
                            name: tensor.key.clone(),
                            logical_shape: tensor.shape.clone(),
                            physical_shape: tensor.shape.clone(),
                            stored_dtype: dtype,
                            encoded_byte_len: 0,
                            backing_shard: Some("model.safetensors".into()),
                        },
                    )
                })
                .collect();
            Self {
                tensors,
                reads: Mutex::new(Vec::new()),
            }
        }
    }

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.tensors
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    impl BoundedRecipeSource for Catalog {
        fn verify_bounded_source(
            &self,
            key: &str,
            selection: TensorSelection,
        ) -> Result<(), StoreError> {
            self.tensor_metadata(key)?;
            self.reads.lock().unwrap().push((key.into(), selection));
            Ok(())
        }
    }

    fn personaplex(quantization: Option<&str>) -> MoshiConfig {
        let quantization = quantization
            .map(|value| format!(r#", "quantization": {value}"#))
            .unwrap_or_default();
        MoshiConfig::from_json(&format!(
            r#"{{"model_type":"personaplex","version":"7b-v1"{quantization}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn native_catalog_and_recipes_cover_the_complete_topology() {
        let config = MoshiConfig::native_v0_1().unwrap();
        let plan = safetensors_plan(&config).unwrap();
        assert!(plan.catalog_policy.strict);
        assert_eq!(plan.common_tensors.len(), 523);
        let recipes = canonical_recipes(&config, &Catalog::from_plan(&plan)).unwrap();
        assert_eq!(recipes.iter().count(), 523);
        assert_eq!(recipes.aliases().count(), 0);
        assert!(recipes
            .get("depformer.slices.7.transformer.layers.5.self_attn.in_proj.weight")
            .is_some());
    }

    #[test]
    fn personaplex_catalog_maps_to_canonical_names_and_shared_owners() {
        let config = personaplex(None);
        let plan = safetensors_plan(&config).unwrap();
        assert!(plan.catalog_policy.strict);
        assert_eq!(plan.common_tensors.len(), 475);
        let qkv = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "transformer.layers.0.self_attn.in_proj_weight")
            .unwrap();
        assert_eq!(
            qkv.aliases,
            ["transformer.layers.0.self_attn.in_proj.weight"]
        );
        assert_eq!(
            plan.common_tensors
                .iter()
                .find(|tensor| tensor.key == "out_norm.alpha")
                .unwrap()
                .element_count,
            Some(4_096)
        );
        let recipes = canonical_recipes(&config, &Catalog::from_plan(&plan)).unwrap();
        assert_eq!(recipes.iter().count(), 655);
        assert_eq!(recipes.aliases().count(), 180);
        let (owner, _) = recipes
            .get_resolved("depformer.slices.15.transformer.layers.5.norm2.weight")
            .unwrap();
        assert_eq!(
            owner,
            "depformer.slices.0.transformer.layers.5.norm2.weight"
        );
        assert!(recipes
            .aliases()
            .all(|(alias, _)| !alias.contains("in_proj_weight")));
    }

    #[test]
    fn affine_depth_slice_and_rank_selection_remain_coherent_and_bounded() {
        let config = personaplex(Some(r#"{"group_size":32,"bits":4,"mode":"affine"}"#));
        let plan = safetensors_plan(&config).unwrap();
        let catalog = Catalog::from_plan(&plan);
        let recipes = canonical_recipes(&config, &catalog).unwrap();
        let root = "depformer.slices.3.transformer.layers.0.self_attn.in_proj";
        let member = |suffix: &str| {
            MatrixRecipeMember::new(
                format!("rank.{suffix}"),
                recipes.get(&format!("{root}.{suffix}")).unwrap().clone(),
            )
        };
        let family = AtomicMatrixRecipeFamily::new(
            &catalog,
            member("weight"),
            Some(member("scales")),
            Some(member("biases")),
        )
        .unwrap();
        let rank = family
            .select_leading_axis(
                &catalog,
                TensorSelection::Range {
                    axis: 0,
                    start: 1_024,
                    end: 2_048,
                },
            )
            .unwrap();
        for recipe in [
            &rank.weight().recipe,
            &rank.scales().unwrap().recipe,
            &rank.biases().unwrap().recipe,
        ] {
            assert!(matches!(
                recipe,
                DerivedWeightRecipe::Source {
                    selection: TensorSelection::Range {
                        axis: 0,
                        start: 10_240,
                        end: 11_264
                    },
                    ..
                }
            ));
            recipe.preflight_bounded(&catalog).unwrap();
        }
        assert!(catalog
            .reads
            .lock()
            .unwrap()
            .iter()
            .all(|(_, selection)| matches!(
                selection,
                TensorSelection::Range {
                    axis: 0,
                    start: 10_240,
                    end: 11_264
                }
            )));
    }

    #[test]
    fn mxfp4_requires_scales_without_affine_biases() {
        let config = personaplex(Some(r#"{"group_size":32,"bits":4,"mode":"mxfp4"}"#));
        let plan = safetensors_plan(&config).unwrap();
        assert!(plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "text_emb.scales"));
        assert!(!plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "text_emb.biases"));
        let recipes = canonical_recipes(&config, &Catalog::from_plan(&plan)).unwrap();
        assert!(recipes.get("text_emb.scales").is_some());
        assert!(recipes.get("text_emb.biases").is_none());
    }

    #[test]
    fn missing_or_malformed_companions_fail_before_publication() {
        let config = personaplex(Some(r#"{"group_size":32,"bits":4,"mode":"affine"}"#));
        let plan = safetensors_plan(&config).unwrap();
        let mut missing = Catalog::from_plan(&plan);
        missing.tensors.remove("text_emb.scales");
        assert!(canonical_recipes(&config, &missing)
            .unwrap_err()
            .contains("text_emb.scales"));
        let mut malformed = Catalog::from_plan(&plan);
        malformed
            .tensors
            .get_mut("text_emb.scales")
            .unwrap()
            .logical_shape[0] -= 1;
        let error = canonical_recipes(&config, &malformed).unwrap_err();
        assert!(error.contains("matrix-family scales geometry"), "{error}");
    }

    #[test]
    fn physical_spellings_are_exclusive_and_preserve_source_provenance() {
        let config = personaplex(None);
        let plan = safetensors_plan(&config).unwrap();
        let primary = "transformer.layers.0.self_attn.in_proj_weight";
        let alias = "transformer.layers.0.self_attn.in_proj.weight";
        let mut alternate = Catalog::from_plan(&plan);
        let mut metadata = alternate.tensors.remove(primary).unwrap();
        metadata.name = alias.into();
        alternate.tensors.insert(alias.into(), metadata);
        let recipes = canonical_recipes(&config, &alternate).unwrap();
        assert_eq!(
            recipes
                .get("transformer.layers.0.self_attn.in_proj.weight")
                .unwrap()
                .source_keys(),
            [alias]
        );
        let mut collision = Catalog::from_plan(&plan);
        let mut metadata = collision.tensors[primary].clone();
        metadata.name = alias.into();
        collision.tensors.insert(alias.into(), metadata);
        assert!(canonical_recipes(&config, &collision)
            .unwrap_err()
            .contains("colliding physical aliases"));
    }

    #[test]
    fn bounded_reads_use_one_shared_owner_and_alias_validation_is_fail_closed() {
        let config = personaplex(None);
        let plan = safetensors_plan(&config).unwrap();
        let catalog = Catalog::from_plan(&plan);
        let recipes = canonical_recipes(&config, &catalog).unwrap();
        for (_, recipe) in recipes.iter() {
            recipe.preflight_bounded(&catalog).unwrap();
        }
        assert_eq!(
            catalog
                .reads
                .lock()
                .unwrap()
                .iter()
                .filter(|(key, _)| key == "depformer.layers.0.norm1.alpha")
                .count(),
            1
        );
        let owner = "depformer.slices.0.transformer.layers.0.norm1.weight";
        let output = vec![(owner.into(), recipes.get(owner).unwrap().clone())];
        let cycle = AtomicRecipeSet::new_with_aliases(
            &catalog,
            output.clone(),
            [RecipeAlias::new("a", "b"), RecipeAlias::new("b", "a")],
        );
        assert!(matches!(cycle, Err(RecipeError::AliasCycle { .. })));
        let collision =
            AtomicRecipeSet::new_with_aliases(&catalog, output, [RecipeAlias::new(owner, owner)]);
        assert!(matches!(
            collision,
            Err(RecipeError::AliasOutputCollision { .. })
        ));
        let invalid = AtomicRecipeSet::new_with_aliases(
            &catalog,
            [(owner.into(), recipes.get(owner).unwrap().clone())],
            [RecipeAlias::new("orphan", "missing.owner")],
        );
        assert!(matches!(
            invalid,
            Err(RecipeError::InvalidAliasDestination { .. })
        ));
    }
}
