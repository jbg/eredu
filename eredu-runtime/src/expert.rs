//! Runtime ownership boundary for routed expert acquisition and residency.

use eredu_nn::{
    DistributedNeuralBackend, GroupSelection, GroupedGatedProductOperator, GroupedNeuralBackend,
    GroupedRelu2Operator, Tensor, TensorParallelGroupedOutput,
};

use crate::ExpertPass;
use crate::{
    observe_and_intervene, ActivationObserver, ParameterBankAccess, ParameterBankKey,
    ReplicatedTextMaterializationTask, ReplicatedTextParameterOwner, RoutingObservation,
    WeightLoweringKind,
};

/// One exact selected parameter in an independently addressable bank member.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AddressableBankParameter {
    binding_name: String,
    task: ReplicatedTextMaterializationTask,
    recipe: eredu_checkpoint::recipe::DerivedWeightRecipe,
    source_output: eredu_checkpoint::recipe::RecipeMetadata,
    selected_bytes: u64,
    quantization_companions: Option<crate::QuantizationCompanionBindings>,
}

impl AddressableBankParameter {
    /// Retains and validates one selected task and its member-local recipe.
    pub fn new(
        binding_name: impl Into<String>,
        task: ReplicatedTextMaterializationTask,
        recipe: eredu_checkpoint::recipe::DerivedWeightRecipe,
        source_output: eredu_checkpoint::recipe::RecipeMetadata,
        selected_bytes: u64,
        quantization_companions: Option<crate::QuantizationCompanionBindings>,
    ) -> Result<Self, AddressableBankMemberError> {
        let binding_name = binding_name.into();
        if binding_name.trim().is_empty() {
            return Err(AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "addressable binding name is empty".into(),
            });
        }
        task.source_recipe()
            .map_err(|error| AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: error.to_string(),
            })?;
        let descriptor = task.lowering_descriptor();
        if descriptor.source() != task.source_encoding()
            || descriptor.executable() != task.executable()
            || descriptor.physical_shape() != task.physical_shape()
            || descriptor.logical_shape() != task.logical_shape()
        {
            return Err(AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "selected source, executable, or lowering descriptor drifted".into(),
            });
        }
        let declared_sources = task
            .sources()
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let recipe_sources = recipe
            .source_keys()
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        if recipe_sources.is_empty() || !recipe_sources.is_subset(&declared_sources) {
            return Err(AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "member recipe consumes sources outside the selected task".into(),
            });
        }
        if source_output.byte_len() == 0 {
            return Err(AddressableBankMemberError::ZeroSourceBytes {
                parameter: task.name().to_owned(),
            });
        }
        if selected_bytes == 0 {
            return Err(AddressableBankMemberError::ZeroSelectedBytes {
                parameter: task.name().to_owned(),
            });
        }
        let transforms = matches!(
            task.lowering(),
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
        );
        if !transforms && quantization_companions.is_some() {
            return Err(AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "non-transform lowering declared local transform companions".into(),
            });
        }
        if transforms
            && quantization_companions.is_none()
            && source_output.dtype() != &eredu_checkpoint::recipe::RecipeDtype::F4
        {
            return Err(AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "floating transform omitted its local output companions".into(),
            });
        }
        if transforms
            && source_output.dtype() != &eredu_checkpoint::recipe::RecipeDtype::F4
            && task.output_companions().is_empty()
        {
            return Err(AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "floating transform omitted its exact selected output companions".into(),
            });
        }
        if let Some(companions) = quantization_companions.as_ref() {
            let declared_roles = task
                .output_companions()
                .iter()
                .map(|companion| companion.role())
                .collect::<std::collections::BTreeSet<_>>();
            let mut bound_roles =
                std::collections::BTreeSet::from([eredu_nn::LinearCompanionRole::Scale]);
            if companions.affine_bias().is_some() {
                bound_roles.insert(eredu_nn::LinearCompanionRole::AffineBias);
            }
            if declared_roles != bound_roles {
                return Err(AddressableBankMemberError::InvalidParameter {
                    parameter: task.name().to_owned(),
                    detail: "selected quantization companion roles differ from exact outputs"
                        .into(),
                });
            }
            for companion in task.output_companions() {
                let local = match companion.role() {
                    eredu_nn::LinearCompanionRole::Scale => companions.scale(),
                    eredu_nn::LinearCompanionRole::AffineBias => companions
                        .affine_bias()
                        .expect("validated affine-bias role has one binding"),
                };
                if companion.name() != local && !companion.name().ends_with(&format!(".{local}")) {
                    return Err(AddressableBankMemberError::InvalidParameter {
                        parameter: task.name().to_owned(),
                        detail: format!(
                            "local companion {local:?} differs from selected output {:?}",
                            companion.name()
                        ),
                    });
                }
                let owner_matches = match (task.owner(), companion.owner()) {
                    (
                        ReplicatedTextParameterOwner::ExecutionUnit { group, unit },
                        crate::ParameterGroupOwner::ExecutionUnit {
                            group: companion_group,
                            global_unit,
                        },
                    ) => group == companion_group.as_str() && unit == global_unit,
                    (
                        ReplicatedTextParameterOwner::StaticRole(role),
                        crate::ParameterGroupOwner::StaticRole(companion_role),
                    ) => role == companion_role,
                    _ => false,
                };
                if !owner_matches {
                    return Err(AddressableBankMemberError::InvalidParameter {
                        parameter: task.name().to_owned(),
                        detail: format!(
                            "selected companion {:?} has a different owner",
                            companion.name()
                        ),
                    });
                }
            }
        }
        let expected = selected_addressable_parameter_bytes(&task, &source_output)?;
        if selected_bytes != expected {
            return Err(AddressableBankMemberError::SelectedByteMismatch {
                parameter: task.name().to_owned(),
                expected,
                actual: selected_bytes,
            });
        }
        Ok(Self {
            binding_name,
            task,
            recipe,
            source_output,
            selected_bytes,
            quantization_companions,
        })
    }

    /// Returns the local grouped-operator binding name.
    pub fn binding_name(&self) -> &str {
        &self.binding_name
    }

    /// Returns the complete authoritative selected materialization task.
    pub const fn task(&self) -> &ReplicatedTextMaterializationTask {
        &self.task
    }

    /// Returns the exact member-local source recipe.
    pub const fn recipe(&self) -> &eredu_checkpoint::recipe::DerivedWeightRecipe {
        &self.recipe
    }

    /// Returns admitted metadata for the member-local source recipe.
    pub const fn source_output(&self) -> &eredu_checkpoint::recipe::RecipeMetadata {
        &self.source_output
    }

    /// Returns source bytes before an optional lowering.
    pub const fn source_bytes(&self) -> u64 {
        self.source_output.byte_len()
    }

    /// Returns executable bytes after the selected lowering.
    pub const fn selected_bytes(&self) -> u64 {
        self.selected_bytes
    }

    /// Returns exact local scale and affine-bias binding names for a transform.
    pub const fn quantization_companions(&self) -> Option<&crate::QuantizationCompanionBindings> {
        self.quantization_companions.as_ref()
    }
}

/// Exact generic storage member projected from an architecture-owned bank catalog.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AddressableBankMember {
    key: ParameterBankKey,
    placement: AddressableBankMemberPlacement,
    parameters: Vec<AddressableBankParameter>,
    source_bytes: u64,
    selected_bytes: u64,
}

/// Neutral placement class for one independently addressable member.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum AddressableBankDistribution {
    /// The member is present on every execution rank.
    Replicated,
    /// The member follows an architecture-selected expert partition.
    ExpertParallel,
}

/// Architecture-selected ownership retained with an addressable member.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AddressableBankMemberPlacement {
    owner_group: crate::ExecutionGroupId,
    owner_unit: usize,
    unit_path: String,
    distribution: AddressableBankDistribution,
    owner_rank: Option<usize>,
}

impl AddressableBankMemberPlacement {
    /// Creates exact architecture-global member placement.
    pub fn new(
        owner_group: crate::ExecutionGroupId,
        owner_unit: usize,
        unit_path: impl Into<String>,
        distribution: AddressableBankDistribution,
    ) -> Result<Self, AddressableBankMemberError> {
        let unit_path = unit_path.into();
        if unit_path.trim().is_empty() {
            return Err(AddressableBankMemberError::InvalidPlacement(
                "addressable member unit path is empty".into(),
            ));
        }
        Ok(Self {
            owner_group,
            owner_unit,
            unit_path,
            distribution,
            owner_rank: None,
        })
    }

    /// Binds this selected member projection to one global partition rank.
    pub fn with_owner_rank(mut self, owner_rank: usize) -> Self {
        self.owner_rank = Some(owner_rank);
        self
    }

    /// Returns the architecture execution group that owns this member.
    pub const fn owner_group(&self) -> &crate::ExecutionGroupId {
        &self.owner_group
    }
    /// Returns the architecture-global execution-unit index.
    pub const fn owner_unit(&self) -> usize {
        self.owner_unit
    }
    /// Returns the stable architecture path of the owning unit.
    pub fn unit_path(&self) -> &str {
        &self.unit_path
    }
    /// Returns the architecture-selected distribution class.
    pub const fn distribution(&self) -> AddressableBankDistribution {
        self.distribution
    }
    /// Returns the global partition rank after rank-local projection.
    pub const fn owner_rank(&self) -> Option<usize> {
        self.owner_rank
    }
}

impl AddressableBankMember {
    /// Validates one atomic member and all of its selected parameter tasks.
    pub fn new(
        key: ParameterBankKey,
        placement: AddressableBankMemberPlacement,
        parameters: impl IntoIterator<Item = AddressableBankParameter>,
    ) -> Result<Self, AddressableBankMemberError> {
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        if parameters.is_empty() {
            return Err(AddressableBankMemberError::EmptyMember { key });
        }
        if placement.owner_unit() != key.unit() {
            return Err(AddressableBankMemberError::InvalidPlacement(format!(
                "addressable member unit {} differs from placement unit {}",
                key.unit(),
                placement.owner_unit()
            )));
        }
        let mut bindings = std::collections::BTreeSet::new();
        let mut targets = std::collections::BTreeSet::new();
        let mut source_bytes = 0u64;
        let mut selected_bytes = 0u64;
        for parameter in &parameters {
            if !bindings.insert(parameter.binding_name())
                || !targets.insert(parameter.task().name())
            {
                return Err(AddressableBankMemberError::DuplicateParameter { key });
            }
            if !matches!(
                parameter.task().owner(),
                ReplicatedTextParameterOwner::ExecutionUnit { group, unit }
                    if *unit == placement.owner_unit()
                        && group == placement.owner_group().as_str()
            ) {
                return Err(AddressableBankMemberError::InvalidParameter {
                    parameter: parameter.task().name().to_owned(),
                    detail: "selected task has a non-bank owner".into(),
                });
            }
            source_bytes = source_bytes
                .checked_add(parameter.source_bytes())
                .ok_or(AddressableBankMemberError::SourceByteOverflow { key })?;
            selected_bytes = selected_bytes
                .checked_add(parameter.selected_bytes())
                .ok_or(AddressableBankMemberError::SelectedByteOverflow { key })?;
        }
        Ok(Self {
            key,
            placement,
            parameters,
            source_bytes,
            selected_bytes,
        })
    }

    /// Returns the generic bank key selected by neutral composition.
    pub const fn key(&self) -> ParameterBankKey {
        self.key
    }

    /// Exact group/unit/distribution/rank ownership selected for this member.
    pub const fn placement(&self) -> &AddressableBankMemberPlacement {
        &self.placement
    }

    /// Retains the global rank selected by a partition projection.
    pub fn with_owner_rank(mut self, owner_rank: usize) -> Self {
        self.placement = self.placement.with_owner_rank(owner_rank);
        self
    }

    /// Returns every exact selected parameter in deterministic binding order.
    pub fn parameters(&self) -> &[AddressableBankParameter] {
        &self.parameters
    }

    /// Returns admitted source bytes before optional lowerings.
    pub const fn source_bytes(&self) -> u64 {
        self.source_bytes
    }

    /// Returns executable bytes after selected lowerings.
    pub const fn selected_bytes(&self) -> u64 {
        self.selected_bytes
    }
}

/// Backend-neutral transform retained for one addressable binding.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AddressableBindingTransform {
    quantization: eredu_checkpoint::WeightQuantization,
    companion_dtype: eredu_checkpoint::recipe::RecipeDtype,
}

impl AddressableBindingTransform {
    /// Selected packed executable format.
    pub const fn quantization(&self) -> eredu_checkpoint::WeightQuantization {
        self.quantization
    }
    /// Selected scale and affine-bias scalar representation.
    pub const fn companion_dtype(&self) -> &eredu_checkpoint::recipe::RecipeDtype {
        &self.companion_dtype
    }
}

/// Canonical binding and transformation plan for one addressable member.
#[derive(Debug, Clone)]
pub struct AddressableBankBindingPlan {
    key: ParameterBankKey,
    bindings: Vec<crate::WeightBinding>,
    transformations: std::collections::BTreeMap<String, AddressableBindingTransform>,
    selected_bytes: u64,
    placement: AddressableBankMemberPlacement,
}

impl AddressableBankBindingPlan {
    /// Generic member identity.
    pub const fn key(&self) -> ParameterBankKey {
        self.key
    }
    /// Source-side canonical bindings.
    pub fn bindings(&self) -> &[crate::WeightBinding] {
        &self.bindings
    }
    /// Per-binding transforms selected by architecture admission.
    pub const fn transformations(
        &self,
    ) -> &std::collections::BTreeMap<String, AddressableBindingTransform> {
        &self.transformations
    }
    /// Exact executable bytes after all transforms.
    pub const fn selected_bytes(&self) -> u64 {
        self.selected_bytes
    }
    /// Exact rank-local architecture placement.
    pub const fn placement(&self) -> &AddressableBankMemberPlacement {
        &self.placement
    }
    /// Consumes the complete canonical member plan.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        ParameterBankKey,
        Vec<crate::WeightBinding>,
        std::collections::BTreeMap<String, AddressableBindingTransform>,
        u64,
        AddressableBankMemberPlacement,
    ) {
        (
            self.key,
            self.bindings,
            self.transformations,
            self.selected_bytes,
            self.placement,
        )
    }
}

/// Validates exact addressable tasks and derives their singular canonical binding plans.
pub fn plan_addressable_bank_bindings<L, E>(
    members: &[AddressableBankMember],
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    mut lower_mxfp4: L,
) -> Result<Vec<AddressableBankBindingPlan>, AddressableBankMemberError>
where
    L: FnMut(
        &ReplicatedTextMaterializationTask,
        eredu_checkpoint::recipe::DerivedWeightRecipe,
        &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<eredu_checkpoint::recipe::DerivedWeightRecipe, E>,
    E: std::fmt::Display,
{
    let mut plans = Vec::with_capacity(members.len());
    for member in members {
        let mut bindings = Vec::with_capacity(member.parameters().len());
        let mut transformations = std::collections::BTreeMap::new();
        for parameter in member.parameters() {
            let task = parameter.task();
            let declared = task
                .sources()
                .iter()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>();
            let physical = task
                .physical_sources()
                .iter()
                .map(|item| item.catalog_key())
                .collect::<std::collections::BTreeSet<_>>();
            if declared != physical || physical.len() != task.physical_sources().len() {
                return Err(AddressableBankMemberError::InvalidParameter {
                    parameter: task.name().to_owned(),
                    detail: "selected physical provenance does not exactly cover task sources"
                        .into(),
                });
            }
            for admitted in task.physical_sources() {
                let actual = source
                    .source_provenance(admitted.catalog_key())
                    .map_err(|error| AddressableBankMemberError::InvalidParameter {
                        parameter: task.name().to_owned(),
                        detail: error.to_string(),
                    })?;
                let metadata = source
                    .source_metadata(admitted.catalog_key())
                    .map_err(|error| AddressableBankMemberError::InvalidParameter {
                        parameter: task.name().to_owned(),
                        detail: error.to_string(),
                    })?;
                if actual.catalog_key != admitted.catalog_key()
                    || actual.physical_tensor != admitted.tensor()
                    || actual.output != admitted.output()
                    || actual.backing_shard.as_deref() != Some(admitted.shard())
                    || actual.source_encoding != *admitted.source_encoding()
                    || metadata.encoded_byte_len != admitted.encoded_byte_len()
                {
                    return Err(AddressableBankMemberError::InvalidParameter {
                        parameter: task.name().to_owned(),
                        detail: format!(
                            "source {:?} differs from admitted provenance",
                            admitted.catalog_key()
                        ),
                    });
                }
            }
            let mut recipe = parameter.recipe().clone();
            let inferred = recipe.infer(source).map_err(|error| {
                AddressableBankMemberError::InvalidParameter {
                    parameter: task.name().to_owned(),
                    detail: error.to_string(),
                }
            })?;
            if &inferred != parameter.source_output() {
                return Err(AddressableBankMemberError::InvalidParameter {
                    parameter: task.name().to_owned(),
                    detail: "member-local recipe output drifted".into(),
                });
            }
            if task.executable() == eredu_checkpoint::LinearFormat::MxFp4
                && inferred.dtype() == &eredu_checkpoint::recipe::RecipeDtype::F4
                && parameter.quantization_companions().is_none()
            {
                recipe = lower_mxfp4(task, recipe, source).map_err(|error| {
                    AddressableBankMemberError::InvalidParameter {
                        parameter: task.name().to_owned(),
                        detail: error.to_string(),
                    }
                })?;
            }
            let metadata = recipe.infer(source).map_err(|error| {
                AddressableBankMemberError::InvalidParameter {
                    parameter: task.name().to_owned(),
                    detail: error.to_string(),
                }
            })?;
            let mut binding = crate::WeightBinding::from_recipe(
                parameter.binding_name(),
                recipe,
                metadata.byte_len(),
            )
            .and_then(|binding| binding.with_logical_target(task.name()))
            .map_err(|error| AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: error.to_string(),
            })?;
            if let Some(companions) = parameter.quantization_companions() {
                let quantization = task.executable().weight_quantization().ok_or_else(|| {
                    AddressableBankMemberError::InvalidParameter {
                        parameter: task.name().to_owned(),
                        detail: "transformed task has no packed format".into(),
                    }
                })?;
                transformations.insert(
                    parameter.binding_name().to_owned(),
                    AddressableBindingTransform {
                        quantization,
                        companion_dtype: parameter.source_output().dtype().clone(),
                    },
                );
                binding = binding
                    .with_quantization_companions(
                        companions.scale(),
                        companions.affine_bias().map(str::to_owned),
                    )
                    .map_err(|error| AddressableBankMemberError::InvalidParameter {
                        parameter: task.name().to_owned(),
                        detail: error.to_string(),
                    })?;
            }
            bindings.push(binding);
        }
        crate::WeightBindingPlan::new(&bindings).map_err(|error| {
            AddressableBankMemberError::InvalidParameter {
                parameter: format!("{:?}", member.key()),
                detail: error.to_string(),
            }
        })?;
        plans.push(AddressableBankBindingPlan {
            key: member.key(),
            bindings,
            transformations,
            selected_bytes: member.selected_bytes(),
            placement: member.placement().clone(),
        });
    }
    Ok(plans)
}

/// Computes executable storage bytes for one admitted member-local task output.
///
/// A task's own derived output describes whole-parameter derivation before a
/// member projection. `metadata` instead describes the exact member-local
/// recipe retained by [`AddressableBankParameter`], including rank-local
/// sharding. This function is the single neutral authority for their selected
/// executable byte geometry.
pub fn selected_addressable_parameter_bytes(
    task: &ReplicatedTextMaterializationTask,
    metadata: &eredu_checkpoint::recipe::RecipeMetadata,
) -> Result<u64, AddressableBankMemberError> {
    if !matches!(
        task.lowering(),
        WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
    ) {
        return Ok(metadata.byte_len());
    }
    let quantization = task.executable().weight_quantization().ok_or_else(|| {
        AddressableBankMemberError::InvalidParameter {
            parameter: task.name().to_owned(),
            detail: "transform lowering has no packed executable format".into(),
        }
    })?;
    if matches!(
        quantization,
        eredu_checkpoint::WeightQuantization::GgufIQuant { .. }
    ) {
        return Err(AddressableBankMemberError::InvalidParameter {
            parameter: task.name().to_owned(),
            detail: "load-time transform selected checkpoint-native GGUF encoding".into(),
        });
    }
    if task.lowering_descriptor().packed_axis() != metadata.shape().len().checked_sub(1) {
        return Err(AddressableBankMemberError::InvalidParameter {
            parameter: task.name().to_owned(),
            detail: "transform packed axis is not the final logical matrix axis".into(),
        });
    }
    let shape = metadata.shape();
    let (&columns, row_shape) =
        shape
            .split_last()
            .ok_or_else(|| AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "transform target is not a matrix".into(),
            })?;
    let rows = row_shape
        .iter()
        .try_fold(1u64, |total, dimension| {
            total.checked_mul(*dimension as u64)
        })
        .ok_or_else(|| AddressableBankMemberError::InvalidParameter {
            parameter: task.name().to_owned(),
            detail: "transform row geometry overflowed".into(),
        })?;
    let group = usize::try_from(quantization.group_size()).map_err(|_| {
        AddressableBankMemberError::InvalidParameter {
            parameter: task.name().to_owned(),
            detail: "transform group size is invalid".into(),
        }
    })?;
    if group == 0 || !columns.is_multiple_of(group) || !columns.is_multiple_of(32) {
        return Err(AddressableBankMemberError::InvalidParameter {
            parameter: task.name().to_owned(),
            detail: "transform geometry is incompatible with its packed format".into(),
        });
    }
    let groups = (columns / group) as u64;
    let packed = (columns as u64)
        .checked_mul(quantization.bits() as u64)
        .and_then(|bits| bits.checked_div(8))
        .ok_or_else(|| AddressableBankMemberError::InvalidParameter {
            parameter: task.name().to_owned(),
            detail: "packed row byte geometry overflowed".into(),
        })?;
    let scalar_bytes = metadata.dtype().bit_width().map_err(|error| {
        AddressableBankMemberError::InvalidParameter {
            parameter: task.name().to_owned(),
            detail: error.to_string(),
        }
    })? / 8;
    let companion = if matches!(quantization, eredu_checkpoint::WeightQuantization::MxFp4) {
        groups
    } else {
        groups.checked_mul(scalar_bytes).ok_or_else(|| {
            AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "scale byte geometry overflowed".into(),
            }
        })?
    };
    let bias = if quantization.has_biases() {
        companion
    } else {
        0
    };
    rows.checked_mul(
        packed
            .checked_add(companion)
            .and_then(|bytes| bytes.checked_add(bias))
            .ok_or_else(|| AddressableBankMemberError::InvalidParameter {
                parameter: task.name().to_owned(),
                detail: "selected row byte geometry overflowed".into(),
            })?,
    )
    .ok_or_else(|| AddressableBankMemberError::InvalidParameter {
        parameter: task.name().to_owned(),
        detail: "selected byte geometry overflowed".into(),
    })
}

/// Invalid generic addressable-bank member projection.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum AddressableBankMemberError {
    /// Member placement was empty or disagreed with task ownership.
    #[error("invalid addressable bank member placement: {0}")]
    InvalidPlacement(String),
    /// A member contained no parameter task.
    #[error("addressable bank member {key:?} is empty")]
    EmptyMember {
        /// Invalid member identity.
        key: ParameterBankKey,
    },
    /// A member repeated a local binding or logical task.
    #[error("addressable bank member {key:?} repeats a parameter")]
    DuplicateParameter {
        /// Invalid member identity.
        key: ParameterBankKey,
    },
    /// One parameter's exact task closure was inconsistent.
    #[error("invalid addressable bank parameter {parameter:?}: {detail}")]
    InvalidParameter {
        /// Invalid logical parameter.
        parameter: String,
        /// Exact validation failure.
        detail: String,
    },
    /// A source recipe selected no bytes.
    #[error("addressable bank parameter {parameter:?} source byte geometry is zero")]
    ZeroSourceBytes {
        /// Invalid logical parameter.
        parameter: String,
    },
    /// Source binding byte accounting overflowed.
    #[error("addressable bank member {key:?} source byte geometry overflowed")]
    SourceByteOverflow {
        /// Invalid member.
        key: ParameterBankKey,
    },
    /// Selected executable byte accounting overflowed.
    #[error("addressable bank member {key:?} selected byte geometry overflowed")]
    SelectedByteOverflow {
        /// Invalid member identity.
        key: ParameterBankKey,
    },
    /// Selected executable storage was empty.
    #[error("addressable bank parameter {parameter:?} selected byte geometry is zero")]
    ZeroSelectedBytes {
        /// Invalid parameter.
        parameter: String,
    },
    /// Selected executable byte geometry differed from the task.
    #[error("addressable bank parameter {parameter:?} selected bytes differ: expected {expected}, got {actual}")]
    SelectedByteMismatch {
        /// Invalid logical parameter.
        parameter: String,
        /// Authoritative computed byte total.
        expected: u64,
        /// Supplied selected byte total.
        actual: u64,
    },
}

/// Generic indexed tensor movement required by bounded grouped execution.
///
/// Implementations expose integer-index discovery and tensor movement without
/// receiving architecture plans, bank meaning, or text lifecycle policy.
pub trait IndexedMovement<B>
where
    B: GroupedNeuralBackend,
{
    /// Indexed movement failure.
    type Error;

    /// Returns deterministic demand counts for integer indices below `upper_bound`.
    fn index_demands(
        &mut self,
        indices: &B::Tensor,
        upper_bound: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<(usize, u64)>, Self::Error>;

    /// Rewrites source indices through one exact source-to-compact mapping.
    fn remap_indices(
        &mut self,
        indices: &B::Tensor,
        mapping: &[(usize, usize)],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Selects a contiguous range along the leading row axis.
    fn select_rows(
        &mut self,
        value: &B::Tensor,
        start: usize,
        end: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Concatenates row partitions in their original order.
    fn concatenate_rows(
        &mut self,
        values: &[B::Tensor],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Backend-neutral tensor movement needed by an expert-exchange protocol.
///
/// Architecture code supplies already validated row and flattened-route
/// indices. Implementations retain tensor storage and completion ownership;
/// they do not receive expert identities, topology, or model-family policy.
pub trait ExpertRouteTensorMovement<T> {
    /// Tensor movement failure.
    type Error;

    /// Returns the logical tensor shape without materializing its values.
    fn shape(&self, value: &T) -> Vec<usize>;

    /// Duplicates and reorders leading-axis rows in the supplied order.
    fn gather_rows(&mut self, value: &T, rows: &[usize]) -> Result<T, Self::Error>;

    /// Selects flattened route scalars and returns them as `[routes, 1]`.
    fn gather_route_values(
        &mut self,
        value: &T,
        flattened_routes: &[usize],
    ) -> Result<T, Self::Error>;

    /// Additively combines route rows into their architecture source rows.
    ///
    /// Every input row must be consumed exactly once. Repeated destination
    /// rows are intentional and implement weighted routed-expert summation.
    fn scatter_add_rows(
        &mut self,
        value: T,
        destination_rows: &[usize],
        output_rows: usize,
    ) -> Result<T, Self::Error>;
}

/// Opaque variable-count transport used by architecture-owned expert routing.
///
/// Implementations must preserve peer-block and within-block order, validate
/// every tensor against the selected communication requirement, and retain all
/// native resources until the exact completion has finished.
pub trait ExpertRouteExchange<T> {
    /// Communication or metadata transport failure.
    type Error;

    /// Exchanges one tensor whose leading rows match the supplied peer counts.
    fn exchange_tensor(
        &mut self,
        counts: &crate::CommunicationPeerCounts,
        value: T,
    ) -> Result<T, Self::Error>;

    /// Exchanges one unsigned metadata value per leading tensor row.
    fn exchange_indices(
        &mut self,
        counts: &crate::CommunicationPeerCounts,
        values: Vec<usize>,
    ) -> Result<Vec<usize>, Self::Error>;
}

/// Architecture-selected combination for one expert-exchange batch.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertRouteCombination {
    /// Apply each route coefficient once, then add routes targeting one token.
    CoefficientWeightedSum,
}

/// One owner-local grouped batch submitted after expert exchange.
pub struct AddressableExpertRouteRequest<'a, T> {
    /// Global execution unit containing the addressable expert bank.
    pub unit: usize,
    /// Rows received from every source peer.
    pub input: &'a T,
    /// Checkpoint-global expert identity for every received row.
    ///
    /// Addressable storage keys must be derived from this identity. It is
    /// deliberately kept separate from `owner_local_experts`, whose values
    /// are valid only as indices into the rank-local grouped operator.
    pub global_experts: &'a [usize],
    /// Dense owner-local expert identity for every received row.
    pub owner_local_experts: &'a [usize],
    /// Selected router scores aligned one-for-one with received rows.
    pub selected_scores: &'a T,
    /// Final route coefficients aligned one-for-one with received rows.
    pub coefficients: &'a T,
    /// Prefill or decode execution classification.
    pub pass: ExpertPass,
    /// Storage access classification derived from `pass`.
    pub access: ParameterBankAccess,
    /// Architecture-declared route combination.
    pub combination: ExpertRouteCombination,
}

impl<T> AddressableExpertRouteRequest<'_, T> {
    /// Returns the only valid addressable-bank key for one routed row.
    ///
    /// The owner-local ID is intentionally not accepted here: it addresses the
    /// compact grouped operator, not checkpoint-global storage.
    pub fn addressable_bank_key(&self, row: usize) -> Option<ParameterBankKey> {
        self.global_experts
            .get(row)
            .copied()
            .map(|global| ParameterBankKey::new(self.unit, global))
    }

    /// Returns the rank-local grouped-operator ID for one routed row.
    pub fn owner_local_execution_id(&self, row: usize) -> Option<usize> {
        self.owner_local_experts.get(row).copied()
    }
}

/// Local addressable grouped execution used by expert exchange.
///
/// The provider must consume every submitted row exactly once, select its
/// corresponding owner-local expert, and apply its route coefficient exactly
/// once. Acquired bank resources remain provider-owned until the returned
/// tensor is natively complete.
pub trait AddressableExpertRouteProvider<T> {
    /// Acquisition or grouped execution failure.
    type Error;

    /// Executes one owner-local grouped batch.
    fn execute_addressable_routes(
        &mut self,
        request: AddressableExpertRouteRequest<'_, T>,
    ) -> Result<T, Self::Error>;

    /// Executes one owner-local grouped batch while retaining tensor-parallel
    /// reduction structure.
    ///
    /// Providers without rank-local TP work inherit complete-output behavior.
    /// A TP provider overrides this method and returns its reducible activation
    /// contribution plus the optional selection-weighted post-reduction bias.
    /// The exchange protocol returns both values to their source-token order;
    /// it must not add the bias before the caller's tensor all-sum.
    fn execute_addressable_routes_tensor_parallel(
        &mut self,
        request: AddressableExpertRouteRequest<'_, T>,
    ) -> Result<RoutedExpertTensorParallelOutput<T>, Self::Error> {
        self.execute_addressable_routes(request)
            .map(RoutedExpertTensorParallelOutput::Complete)
    }
}

/// Exact generic request for an independently addressable bank acquisition.
#[derive(Debug, Clone, Copy)]
pub struct ParameterBankAcquisition<'a> {
    entries: &'a [(ParameterBankKey, u64)],
    access: ParameterBankAccess,
}

impl<'a> ParameterBankAcquisition<'a> {
    /// Creates one deterministic acquisition request in compact-bank order.
    pub const fn new(entries: &'a [(ParameterBankKey, u64)], access: ParameterBankAccess) -> Self {
        Self { entries, access }
    }

    /// Returns generic bank keys and duplicate-preserving demand counts.
    pub const fn entries(&self) -> &'a [(ParameterBankKey, u64)] {
        self.entries
    }

    /// Returns the selected generic storage access class.
    pub const fn access(&self) -> ParameterBankAccess {
        self.access
    }
}

/// Generic addressable storage and grouped-operator construction mechanisms.
///
/// The mechanism receives already translated bank keys, compact specifications,
/// and access classes. Architecture identity, routing policy, global identity
/// mapping, chunking, and text-session behavior remain outside this contract.
pub trait AddressableGroupedBank<B>
where
    B: GroupedNeuralBackend,
{
    /// Live native storage retained across grouped execution.
    type Acquisition;
    /// Generic bank telemetry snapshot.
    type Report;
    /// Storage, transfer, lowering, or construction failure.
    type Error;

    /// Returns the selected byte geometry for one admitted bank member.
    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64>;

    /// Acquires exact generic keys in caller-supplied compact order.
    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Acquisition, Self::Error>;

    /// Constructs one compact gated-product operator from acquired bindings.
    fn gated_product_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::GatedProductGroups, Self::Error>;

    /// Constructs one compact ReLU-squared operator from acquired bindings.
    fn relu2_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedRelu2Spec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Relu2Groups, Self::Error>;

    /// Retains acquired storage until the grouped output is natively complete.
    fn complete(
        &mut self,
        acquisition: Self::Acquisition,
        output: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Returns generic key, byte, tier, acquisition, and eviction telemetry.
    fn report(&self) -> Result<Self::Report, Self::Error>;
}

/// Mechanism-only lookup of one grouped operator in an addressable parameter bank.
pub trait AddressableGatedProductBank<B>
where
    B: GroupedNeuralBackend,
{
    /// Bank lookup or construction failure.
    type Error;

    /// Resolves one generic bank key and exact grouped construction specification.
    fn acquire(
        &mut self,
        key: ParameterBankKey,
        spec: &eredu_nn::GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<&mut B::GatedProductGroups, Self::Error>;
}

/// One architecture route batch submitted to a runtime expert provider.
pub struct RoutedExpertRequest<'a, T> {
    /// Global decoder layer requesting experts.
    pub layer: usize,
    /// Flattened token rows submitted to the selected experts.
    pub input: &'a T,
    /// Backend-native selected expert IDs, scores, and weights.
    pub routes: &'a GroupSelection<T>,
    /// Whether this route batch belongs to prefill or decode.
    pub pass: ExpertPass,
}

impl<T> RoutedExpertRequest<'_, T> {
    /// Projects architecture execution semantics into the storage workload
    /// class exposed to backend parameter-bank mechanisms.
    pub const fn parameter_bank_access(&self) -> ParameterBankAccess {
        self.pass.parameter_bank_access()
    }
}

/// Provider result that distinguishes complete outputs from rank-local TP work.
pub enum RoutedExpertTensorParallelOutput<T> {
    /// Provider already completed every required collective and bias addition.
    Complete(T),
    /// Caller must all-sum `reducible`, then add `post_reduce` exactly once.
    Partial(TensorParallelGroupedOutput<T>),
}

/// Completes one rank-local expert output with one all-sum and one post-bias add.
pub fn reduce_tensor_parallel_expert_output<B>(
    output: TensorParallelGroupedOutput<B::Tensor>,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, eredu_nn::Error>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
{
    let reduced = B::sum_parallel(output.reducible().clone(), parallel, context)?;
    match output.post_reduce().cloned() {
        Some(bias) => reduced.add(&bias, context),
        None => Ok(reduced),
    }
}

/// Combines two rank-local expert partials without introducing another collective.
pub fn combine_tensor_parallel_expert_outputs<B>(
    left: TensorParallelGroupedOutput<B::Tensor>,
    right: TensorParallelGroupedOutput<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<TensorParallelGroupedOutput<B::Tensor>, eredu_nn::Error>
where
    B: GroupedNeuralBackend,
{
    let post_reduce = match (left.post_reduce().cloned(), right.post_reduce().cloned()) {
        (Some(left), Some(right)) => Some(left.add(&right, context)?),
        (Some(bias), None) | (None, Some(bias)) => Some(bias),
        (None, None) => None,
    };
    Ok(TensorParallelGroupedOutput::new(
        left.reducible().add(right.reducible(), context)?,
        post_reduce,
    ))
}

/// Combines routed/shared provider outputs while requiring one coherent TP mode.
pub fn combine_routed_expert_tensor_parallel<B>(
    left: RoutedExpertTensorParallelOutput<B::Tensor>,
    right: RoutedExpertTensorParallelOutput<B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, eredu_nn::Error>
where
    B: GroupedNeuralBackend,
{
    match (left, right) {
        (
            RoutedExpertTensorParallelOutput::Complete(left),
            RoutedExpertTensorParallelOutput::Complete(right),
        ) => Ok(RoutedExpertTensorParallelOutput::Complete(
            left.add(&right, context)?,
        )),
        (
            RoutedExpertTensorParallelOutput::Partial(left),
            RoutedExpertTensorParallelOutput::Partial(right),
        ) => combine_tensor_parallel_expert_outputs::<B>(left, right, context)
            .map(RoutedExpertTensorParallelOutput::Partial),
        _ => Err(eredu_nn::Error::backend(
            "provider mixed complete and rank-local expert outputs in one block",
        )),
    }
}

/// Completes a provider TP result while preserving provider-owned collectives.
pub fn reduce_routed_expert_tensor_parallel<B>(
    output: RoutedExpertTensorParallelOutput<B::Tensor>,
    parallel: &B::ParallelContext,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, eredu_nn::Error>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
{
    match output {
        RoutedExpertTensorParallelOutput::Complete(output) => Ok(output),
        RoutedExpertTensorParallelOutput::Partial(output) => {
            reduce_tensor_parallel_expert_output::<B>(output, parallel, context)
        }
    }
}

/// Runtime boundary for resident or independently cached routed experts.
///
/// Implementations own identity ordering, acquisition, leases, chunking,
/// budgets, and residency reports. They keep every lease alive until the
/// backend-native routed result is safe to return. The backend retains tensor
/// storage, transfers, compact-bank construction, and execution kernels.
pub trait RoutedExpertProvider<B>
where
    B: GroupedNeuralBackend,
{
    /// Provider-specific acquisition or execution failure.
    type Error;

    /// Executes one typed route batch while retaining its acquired resources.
    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Executes destination-local rows that were already expanded to one
    /// owner-local expert per row by the neutral expert exchange.
    ///
    /// The compact request deliberately has route cardinality one; providers
    /// must not compare it with the architecture's original top-k cardinality.
    fn forward_compact_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_grouped(resident_bank, request, context)
    }

    /// Executes one ReLU-squared route batch through the same residency boundary.
    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Additive provider mechanism for tensor-parallel grouped partials.
pub trait TensorParallelRoutedExpertProvider<B>: RoutedExpertProvider<B>
where
    B: GroupedNeuralBackend,
{
    /// Executes a rank-local gated-product contribution.
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>;

    /// Executes destination-local, one-expert-per-row contributions while
    /// preserving the backend's TP reduction and post-bias structure.
    fn forward_compact_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.forward_grouped_tensor_parallel(resident_bank, request, partitions, context)
    }

    /// Executes a rank-local ReLU-squared contribution.
    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error>;
}

/// Stable routing metadata supplied by an architecture composition at one
/// canonical unit boundary.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoutedObservationPoint {
    path: String,
    expert_count: i32,
}

impl RoutedObservationPoint {
    /// Creates one routed observation point.
    pub fn new(path: impl Into<String>, expert_count: i32) -> Self {
        Self {
            path: path.into(),
            expert_count,
        }
    }

    /// Returns the stable routed-module path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the total number of routed experts.
    pub const fn expert_count(&self) -> i32 {
        self.expert_count
    }
}

/// Failure from either canonical expert execution or its observation hook.
#[derive(Debug)]
pub enum ObservedExpertProviderError<P, O> {
    /// The wrapped provider rejected or failed the expert request.
    Provider(P),
    /// The observer rejected the normalized routing event.
    Observer(O),
}

impl<P, O> std::fmt::Display for ObservedExpertProviderError<P, O>
where
    P: std::fmt::Display,
    O: std::fmt::Display,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(error) => write!(formatter, "routed expert provider failed: {error}"),
            Self::Observer(error) => write!(formatter, "routed expert observer failed: {error}"),
        }
    }
}

impl<P, O> std::error::Error for ObservedExpertProviderError<P, O>
where
    P: std::error::Error + 'static,
    O: std::error::Error + 'static,
{
}

/// Decorates a routed provider with normalized routing observation.
///
/// The decorator sees the exact request and output of canonical provider
/// execution. It therefore adds observation without reimplementing a model
/// family's block, routing, shape, or residency lifecycle. Tensor-parallel
/// requests are delegated without an event because their provider result may
/// still require an architecture-owned reduction before it is observable.
pub struct ObservedExpertProvider<'a, P, O: ?Sized, E> {
    provider: &'a mut P,
    observer: &'a mut O,
    point: RoutedObservationPoint,
    error: std::marker::PhantomData<fn() -> E>,
}

impl<'a, P, O: ?Sized, E> ObservedExpertProvider<'a, P, O, E> {
    /// Wraps `provider` for one canonical routed module invocation.
    pub fn new(provider: &'a mut P, observer: &'a mut O, point: RoutedObservationPoint) -> Self {
        Self {
            provider,
            observer,
            point,
            error: std::marker::PhantomData,
        }
    }

    fn observe<T, ObservationError>(
        &mut self,
        routes: &eredu_nn::GroupSelection<T>,
        output: &T,
    ) -> Result<T, ObservationError>
    where
        T: Clone,
        O: ActivationObserver<T, ObservationError>,
    {
        self.observer.observe_routing(RoutingObservation {
            path: self.point.path(),
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: output,
            local_routed_output: None,
            reduced_routed_output: None,
            shared_output: None,
            combined_output: None,
            expert_count: self.point.expert_count(),
        })?;
        observe_and_intervene(
            self.observer,
            &format!("{}.output", self.point.path()),
            output,
        )
    }
}

impl<B, P, O, E> RoutedExpertProvider<B> for ObservedExpertProvider<'_, P, O, E>
where
    B: GroupedNeuralBackend,
    P: RoutedExpertProvider<B>,
    O: ActivationObserver<B::Tensor, E> + ?Sized,
{
    type Error = ObservedExpertProviderError<P::Error, E>;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let routes = request.routes;
        let output = self
            .provider
            .forward_grouped(resident_bank, request, context)
            .map_err(ObservedExpertProviderError::Provider)?;
        self.observe(routes, &output)
            .map_err(ObservedExpertProviderError::Observer)
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let routes = request.routes;
        let output = self
            .provider
            .forward_relu2_routed(resident_bank, request, context)
            .map_err(ObservedExpertProviderError::Provider)?;
        self.observe(routes, &output)
            .map_err(ObservedExpertProviderError::Observer)
    }
}

impl<B, P, O, E> TensorParallelRoutedExpertProvider<B> for ObservedExpertProvider<'_, P, O, E>
where
    B: GroupedNeuralBackend,
    P: TensorParallelRoutedExpertProvider<B>,
    O: ActivationObserver<B::Tensor, E> + ?Sized,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.provider
            .forward_grouped_tensor_parallel(resident_bank, request, partitions, context)
            .map_err(ObservedExpertProviderError::Provider)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        self.provider
            .forward_relu2_routed_tensor_parallel(resident_bank, request, partitions, context)
            .map_err(ObservedExpertProviderError::Provider)
    }
}

/// Provider for a fully resident expert bank.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResidentExpertProvider;

impl<B> RoutedExpertProvider<B> for ResidentExpertProvider
where
    B: GroupedNeuralBackend,
{
    type Error = eredu_nn::Error;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        resident_bank.forward_grouped(request.input, request.routes, context)
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        resident_bank.forward_grouped(request.input, request.routes, context)
    }
}

impl<B> TensorParallelRoutedExpertProvider<B> for ResidentExpertProvider
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut B::GatedProductGroups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        B::gated_product_groups_tensor_parallel(
            resident_bank,
            request.input,
            request.routes,
            partitions,
            context,
        )
        .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut B::Relu2Groups,
        request: RoutedExpertRequest<'_, B::Tensor>,
        partitions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedExpertTensorParallelOutput<B::Tensor>, Self::Error> {
        B::relu2_groups_tensor_parallel(
            resident_bank,
            request.input,
            request.routes,
            partitions,
            context,
        )
        .map(RoutedExpertTensorParallelOutput::Partial)
    }
}
