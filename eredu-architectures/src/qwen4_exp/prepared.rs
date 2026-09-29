//! Exact retained target sources, parameter formats and independent bank owners.
//!
//! This is artifact preparation for the shared construction driver, not public
//! executable admission. Prediction and vision are separate preparation roles.
use super::{
    checkpoint::{
        self,
        recipes::{self, ParameterScope},
        schema::SafetensorsEncoding,
        PreparedNGramTable,
    },
    config::Config,
    target::{BoundTargetSpec, TargetLimits, TargetSpec, UnitSpec},
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{
        PreparedCheckpointSource, PreparedTensorSource, ResolvedCheckpointSource,
        RestrictedCheckpointSource, SharedCheckpointSource, StoreError,
    },
    validation::resolve_safetensors_plan,
    LinearFormat,
};
use eredu_core::residency::ResidencyPolicy;
use eredu_nn::{
    Error, GatedProductGroupLayout, GatedProductPolicy, GroupedGatedProductSpec,
    GroupedProjectionSpec, LinearFormatSpec, ParameterSpec,
};
use eredu_runtime::{PreparedRowLookup, RowLookupLimits, RowResidencyRange};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Invalid artifacts fail before native allocation or ordinary weight reads.
#[derive(Debug, thiserror::Error)]
pub enum PreparationError {
    /// The artifact configuration declares no embedded prediction weights.
    #[error("qwen4_exp artifact does not declare embedded prediction weights")]
    MissingPrediction,
    /// The artifact has no declared vision tower.
    #[error("qwen4_exp artifact does not declare vision weights")]
    MissingVision,
    /// The projector or caller-resolved media identities differ from the target.
    #[error("qwen4_exp vision source differs from target in {field}")]
    VisionMismatch {
        /// Architecture-owned declaration which did not match.
        field: &'static str,
    },
    /// An existing embedded or separately prepared prediction cannot be replaced.
    #[error("qwen4_exp target already declares prediction weights")]
    PredictionAlreadyPresent,
    /// Independent prediction configuration differs from its target contract.
    #[error("qwen4_exp prediction source differs from target in {field}")]
    PredictionMismatch {
        /// Architecture-owned contract field which did not match.
        field: &'static str,
    },
    /// Exact header contract or declared owner differs from the artifact.
    #[error("invalid qwen4_exp prepared target: {0}")]
    Contract(String),
    /// Retained metadata/provenance changed or a source is unavailable.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// Literal integer controls or table companions are invalid.
    #[error(transparent)]
    Table(#[from] checkpoint::NGramArtifactError),
    /// Row geometry or admitted workspace is invalid.
    #[error(transparent)]
    Rows(#[from] eredu_runtime::RowLookupError),
    /// No exact row mechanism or admitted bank resources cover this target.
    #[error(transparent)]
    RowSelection(#[from] eredu_runtime::RowLookupSelectionError),
    /// Named stream geometry or finite read/storage bounds are invalid.
    #[error(transparent)]
    Streams(#[from] eredu_runtime::AppendStreamError),
    /// Portable context or state-memory projection is invalid.
    #[error(transparent)]
    Capability(#[from] eredu_core::CapabilityError),
    /// Portable module specification is invalid.
    #[error(transparent)]
    Neural(#[from] Error),
}

/// Recipes for one ordinary owner or one expert, with no authority over other sources.
#[derive(Clone)]
pub struct PreparedParameters {
    source: SharedCheckpointSource,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
}
impl PreparedParameters {
    pub(in crate::qwen4_exp) fn new(
        source: SharedCheckpointSource,
        recipes: BTreeMap<String, DerivedWeightRecipe>,
    ) -> Result<Self, PreparationError> {
        let keys = recipes
            .values()
            .flat_map(|r| r.source_keys().into_iter().map(str::to_owned))
            .collect();
        let source = retain(source, keys)?;
        for recipe in recipes.values() {
            recipe
                .infer(source.as_ref())
                .map_err(|e| PreparationError::Contract(e.to_string()))?;
        }
        Ok(Self { source, recipes })
    }
    /// Exact restricted source. Deferred binding cannot reopen or rediscover artifacts.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Canonical parameter identities mapped to exact physical recipes.
    pub fn recipes(&self) -> &BTreeMap<String, DerivedWeightRecipe> {
        &self.recipes
    }
}
fn retain(
    source: SharedCheckpointSource,
    keys: BTreeSet<String>,
) -> Result<SharedCheckpointSource, StoreError> {
    let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
        source,
        "qwen4_exp prepared owner",
        keys,
    )?);
    let catalog = source
        .source_keys()
        .into_iter()
        .map(|key| {
            Ok((
                key.clone(),
                PreparedTensorSource {
                    metadata: source.source_metadata(&key)?,
                    provenance: source.source_provenance(&key)?,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, StoreError>>()?;
    Ok(Arc::new(PreparedCheckpointSource::new(source, catalog)?))
}

/// A complete target's source-format construction and retained artifact ownership.
/// Table payloads are never concatenated or decoded during this preparation.
#[derive(Clone)]
pub struct PreparedTarget {
    artifact: SharedCheckpointSource,
    formats: ParameterFormats,
    expert_banks: BTreeMap<String, Arc<PreparedExpertBank>>,
    spec: TargetSpec,
    tables: BTreeMap<usize, PreparedNGramTable>,
    static_parameters: PreparedParameters,
    units: Vec<PreparedParameters>,
}
impl PreparedTarget {
    /// Validates the complete official SafeTensors catalog and prepares exact target
    /// owners. Only bounded integer controls and optional table scalars are read.
    pub fn safetensors(
        source: SharedCheckpointSource,
        config: Config,
        encoding: SafetensorsEncoding,
        limits: TargetLimits,
    ) -> Result<Self, PreparationError> {
        // Bind admission to this exact source before any bounded literal read.
        let source = retain(source.clone(), source.source_keys().into_iter().collect())?;
        SafetensorsTargetPlan::prepare(source.as_ref(), config, encoding)?.bind(source, limits)
    }
    /// Exact construction specs; callers do not select family names or equations.
    pub fn spec(&self) -> &TargetSpec {
        &self.spec
    }
    /// Geometry bound to this artifact's exact lexical controls for executable construction.
    pub fn bound_spec(&self) -> Result<BoundTargetSpec, Error> {
        self.bind_spec(self.spec.clone())
    }
    fn bind_spec(&self, spec: TargetSpec) -> Result<BoundTargetSpec, Error> {
        BoundTargetSpec::new(
            spec,
            self.tables
                .iter()
                .map(|(&layer, table)| (layer, table.hash.clone()))
                .collect(),
        )
    }
    /// Retained complete artifact for separate prediction/vision preparation.
    pub fn artifact(&self) -> &SharedCheckpointSource {
        &self.artifact
    }
    /// Embedding, output and final residual mixer, excluding extension/media weights.
    pub fn static_parameters(&self) -> &PreparedParameters {
        &self.static_parameters
    }
    /// Exact ordinary source for one execution ordinal (not checkpoint layer).
    pub fn unit(&self, ordinal: usize) -> Result<&PreparedParameters, PreparationError> {
        self.units
            .get(ordinal)
            .ok_or_else(|| PreparationError::Contract("execution owner is outside target".into()))
    }
    /// Independently addressable one-expert recipes; packed physical banks retain a
    /// bounded leading-axis selection, including all encoding companions.
    pub fn expert(
        &self,
        ordinal: usize,
        expert: usize,
    ) -> Result<PreparedParameters, PreparationError> {
        let Some(UnitSpec::Decoder { layer, .. }) = self.spec.units.get(ordinal) else {
            return Err(PreparationError::Contract(
                "expert owner is not a decoder".into(),
            ));
        };
        let root = format!("model.layers.{layer}.mlp.experts");
        let recipes = self.expert_recipes(&root, expert)?;
        PreparedParameters::new(
            self.artifact.clone(),
            recipes
                .into_iter()
                .map(|(name, r)| (format!("{root}.{name}"), r))
                .collect(),
        )
    }
    fn expert_recipes(
        &self,
        root: &str,
        expert: usize,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, PreparationError> {
        select_expert(&self.expert_banks, root, expert)
    }
    /// Iterates the small table set, keyed by its original checkpoint layer.
    pub fn tables(&self) -> &BTreeMap<usize, PreparedNGramTable> {
        &self.tables
    }
    /// Selects generic row-provider request bounds and compact residency metadata.
    /// Prepares all lexical tables as one compact binding set. The execution
    /// ordinal and source recipes remain architecture-owned; table residency is
    /// controlled by the generic shared parameter-bank pool.
    pub fn row_lookups(
        &self,
        limits: RowLookupLimits,
        policy: ResidencyPolicy,
    ) -> Result<eredu_runtime::PreparedRowLookups, PreparationError> {
        let entries = self
            .tables
            .keys()
            .map(|layer| self.row_lookup(*layer, limits, policy))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = eredu_runtime::PreparedRowLookups::new(entries, self.spec.units.len())?;
        rows.validate_other_banks([0])?;
        Ok(rows)
    }

    /// The shared bank controller still owns aggregate host/device/transfer budgets.
    pub fn row_lookup(
        &self,
        layer: usize,
        limits: RowLookupLimits,
        policy: ResidencyPolicy,
    ) -> Result<PreparedRowLookup, PreparationError> {
        let table = self
            .tables
            .get(&layer)
            .ok_or_else(|| PreparationError::Contract("layer has no table".into()))?;
        if limits.requests < self.spec.limits.lookup_rows {
            return Err(PreparationError::Rows(
                eredu_runtime::RowLookupError::Budget {
                    resource: "target lookup requests",
                    required: self.spec.limits.lookup_rows as u64,
                    limit: limits.requests as u64,
                },
            ));
        }
        let metadata = table.rows.metadata();
        let range = checkpoint::table_row_range(&table.lookup, metadata, policy)?;
        let range =
            RowResidencyRange::new(range, table.rows.clone(), table.lookup.parameter.as_str())
                .map_err(|e| PreparationError::Contract(e.to_string()))?;
        Ok(PreparedRowLookup::new(
            range,
            table.lookup.clone(),
            table.scale.clone(),
            limits,
        )?)
    }
}
struct PreparedExpertBank {
    recipes: BTreeMap<String, DerivedWeightRecipe>,
    members: Vec<BTreeMap<String, DerivedWeightRecipe>>,
}
impl PreparedExpertBank {
    fn new<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
        source: &C,
        recipes: BTreeMap<String, DerivedWeightRecipe>,
        count: usize,
    ) -> Result<Self, PreparationError> {
        let mut members = vec![BTreeMap::new(); count];
        for (name, recipe) in &recipes {
            let selected = recipe
                .select_bounded_members(source)
                .map_err(|e| PreparationError::Contract(e.to_string()))?;
            if selected.len() != count {
                return Err(PreparationError::Contract(
                    "prepared expert count mismatch".into(),
                ));
            }
            for (member, recipe) in members.iter_mut().zip(selected) {
                member.insert(name.clone(), recipe);
            }
        }
        Ok(Self { recipes, members })
    }
}
fn select_expert(
    banks: &BTreeMap<String, Arc<PreparedExpertBank>>,
    root: &str,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, PreparationError> {
    banks
        .get(root)
        .and_then(|b| b.members.get(expert))
        .cloned()
        .ok_or_else(|| {
            PreparationError::Contract(format!("expert {expert} is outside prepared bank {root}"))
        })
}

/// Exact formats retained after container-specific preparation. Missing identities
/// are errors; native construction cannot reconstruct a checkpoint policy.
#[derive(Clone)]
struct ParameterFormats(
    BTreeMap<String, LinearFormat>,
    BTreeMap<String, eredu_nn::LinearRowLayout>,
);
impl ParameterFormats {
    fn get(&self, name: &str) -> Result<LinearFormat, Error> {
        self.0
            .get(name)
            .copied()
            .ok_or_else(|| Error::backend(format!("missing prepared parameter format {name}")))
    }
    fn ordinary(&self, name: &str) -> Result<LinearFormatSpec, Error> {
        crate::linear_format::standard_linear_format(name, self.get(name)?)
    }
}
fn expert_spec(
    config: &Config,
    formats: &ParameterFormats,
    root: &str,
) -> Result<GroupedGatedProductSpec, Error> {
    let projection = |name: &str| {
        let name = format!("{root}.{name}");
        GroupedProjectionSpec::new(
            ParameterSpec::trainable(&name).map_err(Error::backend)?,
            None,
            crate::linear_format::standard_expert_format(&name, formats.get(&name)?)?
                .with_row_layout(
                    formats
                        .1
                        .get(&name)
                        .copied()
                        .unwrap_or(eredu_nn::LinearRowLayout::Contiguous),
                )?,
        )
    };
    GroupedGatedProductSpec::new(
        config.experts.count,
        config.hidden_size,
        config.experts.intermediate,
        config.hidden_size,
        GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: projection("gate_up_proj")?,
            down: projection("down_proj")?,
        },
    )
}

mod tensor_partition;
pub use tensor_partition::PreparedTensorTarget;

mod partition_selection;
pub(crate) use partition_selection::load_partition_request;
pub use partition_selection::{
    PreparedTargetPartition, SelectedTargetPartitionExecution, TargetPartitionExecutionPlan,
};

mod gguf_target;
pub use gguf_target::{GgufTargetExecutionPlan, GgufTargetPlan, SelectedGgufTargetExecution};

mod safetensors_target;
pub use safetensors_target::{
    SafetensorsTargetExecutionPlan, SafetensorsTargetPlan, SelectedSafetensorsTargetExecution,
};

mod load_defaults;
mod load_policy;
pub(crate) use load_defaults::normalize_load_request;
mod requirements;
pub use load_policy::TargetLoadError;

mod execution;
pub use execution::{SelectedTargetExecution, TargetExecutionPlan, TargetSelectionError};

mod prediction;
pub use prediction::{PreparedPrediction, PreparedTensorPrediction};
mod prediction_source;
pub use prediction_source::{PreparedPredictionSource, SafetensorsPredictionPlan};

mod prediction_plan;
pub use prediction_plan::PreparedPredictionExecution;

mod graph;
mod observations;
mod resources;
pub use observations::{PredictionDiscovery, PredictionObservationBinding};
pub use resources::TargetResourceReport;

fn prepare_safetensors_formats(
    artifact: &SharedCheckpointSource,
    config: &Config,
    encoding: &SafetensorsEncoding,
    roots: impl IntoIterator<Item = String>,
) -> Result<(ParameterFormats, BTreeMap<String, Arc<PreparedExpertBank>>), PreparationError> {
    let (formats, recipes) = safetensors_format_recipes(
        artifact.as_ref(),
        &artifact.source_keys(),
        config,
        encoding,
        roots,
    )?;
    let banks = recipes
        .into_iter()
        .map(|(root, recipes)| {
            Ok((
                root,
                Arc::new(PreparedExpertBank::new(
                    artifact.as_ref(),
                    recipes,
                    config.experts.count as usize,
                )?),
            ))
        })
        .collect::<Result<_, PreparationError>>()?;
    Ok((formats, banks))
}

fn safetensors_format_recipes<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    catalog: &C,
    keys: &[String],
    config: &Config,
    encoding: &SafetensorsEncoding,
    roots: impl IntoIterator<Item = String>,
) -> Result<
    (
        ParameterFormats,
        BTreeMap<String, BTreeMap<String, DerivedWeightRecipe>>,
    ),
    PreparationError,
> {
    // A fused logical bank must preserve every physical member's encoding.
    // The released policy is homogeneous; do not silently reinterpret a
    // custom per-expert exclusion as a block-scaled packed parameter.
    for key in keys {
        let canonical = checkpoint::schema::canonical_name(&key);
        let Some((root, member)) = canonical.split_once(".mlp.experts.") else {
            continue;
        };
        if (!root.starts_with("model.layers.") && !root.starts_with("mtp.layers."))
            || member.ends_with("_scale_inv")
        {
            continue;
        }
        let projection = if member == "down_proj" || member.ends_with(".down_proj.weight") {
            "down_proj"
        } else {
            "gate_up_proj"
        };
        let logical = format!("{root}.mlp.experts.{projection}");
        if encoding.linear_format(&canonical) != encoding.linear_format(&logical) {
            return Err(PreparationError::Contract(format!(
                "physical expert encoding for {key} differs from logical bank {logical}"
            )));
        }
    }
    let mut formats = ParameterFormats(
        keys.iter()
            .map(|key| {
                let name = checkpoint::schema::canonical_name(key);
                let format = encoding.linear_format(&name);
                (name, format)
            })
            .collect(),
        BTreeMap::new(),
    );
    let mut expert_banks = BTreeMap::new();
    for root in roots {
        let bank = crate::shared_routed::checkpoint::gated_expert_recipes(
            catalog,
            &root,
            0..config.experts.count as usize,
            eredu_checkpoint::store::TensorSelection::Full,
            checkpoint::schema::aliases,
        )
        .map_err(PreparationError::Contract)?;
        if !matches!(
            bank.get("gate_up_proj"),
            Some(DerivedWeightRecipe::Source { .. })
        ) {
            formats.1.insert(
                format!("{root}.gate_up_proj"),
                eredu_nn::LinearRowLayout::equal_partitions(2)?,
            );
        }
        for name in ["gate_up_proj", "down_proj"] {
            let name = format!("{root}.{name}");
            formats
                .0
                .insert(name.clone(), encoding.linear_format(&name));
        }
        expert_banks.insert(root, bank);
    }
    Ok((formats, expert_banks))
}

mod vision;
pub use vision::PreparedVision;
mod vision_plan;
pub use vision_plan::{GgufVisionPlan, GgufVisionSourcePlan, VisionPlan};
mod conditional;
pub use conditional::{ConditionalExecutionPlan, SelectedConditionalExecution};
mod conditional_header;
pub use conditional_header::{ConditionalHeaderExecutionPlan, SelectedConditionalHeaderExecution};
mod conditional_partition;
pub use conditional_partition::{
    ConditionalPartitionExecutionPlan, PreparedConditionalPartition,
    SelectedConditionalPartitionExecution,
};

pub(crate) use graph::join_prediction_descriptor;
