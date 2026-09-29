//! Tensor-rank source authority derived from already admitted canonical owners.
use super::*;
use crate::qwen4_exp::target::TargetTensorPartition;
use eredu_checkpoint::store::TensorSelection;

/// Local target parameters and geometry over the original admitted source.
/// Preparation reads no parameter/table payload and does not reopen artifacts.
/// Vocabulary, residual mixers and lexical parameters retain their replicated owners.
#[derive(Clone)]
pub struct PreparedTensorTarget {
    source: PreparedTarget,
    source_bound: BoundTargetSpec,
    bound: BoundTargetSpec,
    partition: TargetTensorPartition,
    static_parameters: PreparedParameters,
    units: Vec<PreparedParameters>,
    expert_banks: BTreeMap<String, Arc<PreparedExpertBank>>,
}
impl PreparedTarget {
    /// Binds one tensor rank to exact canonical source recipes and integer controls.
    /// Selections apply after container-authored layout transformations; they use
    /// the admitted output representation, including packed bytes and companions.
    pub fn tensor_partition(
        &self,
        rank: usize,
        ranks: usize,
    ) -> Result<PreparedTensorTarget, PreparationError> {
        let partition = self.spec.tensor_partition(rank, ranks)?;
        self.bind_tensor_partition(partition)
    }

    pub(super) fn bind_tensor_partition(
        &self,
        partition: TargetTensorPartition,
    ) -> Result<PreparedTensorTarget, PreparationError> {
        let source_bound = self.bound_spec()?;
        let bound = source_bound.tensor_partition(&partition)?;
        let static_parameters = partition_owner(&self.static_parameters, &partition)?;
        let units = self
            .units
            .iter()
            .map(|owner| partition_owner(owner, &partition))
            .collect::<Result<_, _>>()?;
        let expert_banks = self
            .expert_banks
            .iter()
            .map(|(root, bank)| {
                let recipes = bank
                    .recipes
                    .iter()
                    .map(|(name, recipe)| {
                        let canonical = format!("{root}.{name}");
                        Ok((
                            name.clone(),
                            partition_recipe(
                                self.artifact.as_ref(),
                                &canonical,
                                recipe,
                                partition.parameter_selections(&canonical),
                            )?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, PreparationError>>()?;
                // Partition the complete canonical group axes before deriving
                // each bounded expert member. Singleton group axes stay intact.
                let bank =
                    PreparedExpertBank::new(self.artifact.as_ref(), recipes, bank.members.len())?;
                Ok((root.clone(), Arc::new(bank)))
            })
            .collect::<Result<_, PreparationError>>()?;
        Ok(PreparedTensorTarget {
            source: self.clone(),
            source_bound,
            bound,
            partition,
            static_parameters,
            units,
            expert_banks,
        })
    }
}
impl PreparedTensorTarget {
    /// Exact rank and physical selections retained by this source binding.
    pub fn partition(&self) -> &TargetTensorPartition {
        &self.partition
    }
    /// Global source geometry and controls used to validate the retained rank plan.
    pub fn source_bound_spec(&self) -> &BoundTargetSpec {
        &self.source_bound
    }
    /// Local geometry with the original controls and rank-specific state identity.
    pub fn bound_spec(&self) -> &BoundTargetSpec {
        &self.bound
    }
    /// Rank-local unit and state geometry; request vocabulary/policy remain global.
    pub fn spec(&self) -> &TargetSpec {
        self.bound.geometry()
    }
    /// Replicated vocabulary and final residual collapse parameters.
    pub fn static_parameters(&self) -> &PreparedParameters {
        &self.static_parameters
    }
    /// Exact rank-local ordinary recipes for one execution ordinal.
    pub fn unit(&self, ordinal: usize) -> Result<&PreparedParameters, PreparationError> {
        self.units.get(ordinal).ok_or_else(|| {
            PreparationError::Contract("execution owner is outside tensor target".into())
        })
    }
    /// One independently acquired expert, preserving its singleton group axis.
    pub fn expert(
        &self,
        ordinal: usize,
        expert: usize,
    ) -> Result<PreparedParameters, PreparationError> {
        let Some(UnitSpec::Decoder { layer, .. }) = self.spec().units.get(ordinal) else {
            return Err(PreparationError::Contract(
                "expert owner is not a decoder".into(),
            ));
        };
        let root = format!("model.layers.{layer}.mlp.experts");
        let recipes = select_expert(&self.expert_banks, &root, expert)?;
        PreparedParameters::new(
            self.source.artifact.clone(),
            recipes
                .into_iter()
                .map(|(name, recipe)| (format!("{root}.{name}"), recipe))
                .collect(),
        )
    }
    /// Original compact tables and exact hash controls; no row storage is replicated here.
    pub fn tables(&self) -> &BTreeMap<usize, PreparedNGramTable> {
        self.source.tables()
    }
    /// Retains original table authority for the generic owner-only row provider.
    pub fn row_lookups(
        &self,
        limits: RowLookupLimits,
        policy: ResidencyPolicy,
    ) -> Result<eredu_runtime::PreparedRowLookups, PreparationError> {
        self.source.row_lookups(limits, policy)
    }
    /// Retains one original table and its scalar companion without payload acquisition.
    pub fn row_lookup(
        &self,
        layer: usize,
        limits: RowLookupLimits,
        policy: ResidencyPolicy,
    ) -> Result<PreparedRowLookup, PreparationError> {
        self.source.row_lookup(layer, limits, policy)
    }
}
fn partition_owner(
    owner: &PreparedParameters,
    partition: &TargetTensorPartition,
) -> Result<PreparedParameters, PreparationError> {
    let recipes = owner
        .recipes
        .iter()
        .map(|(name, recipe)| {
            Ok((
                name.clone(),
                partition_recipe(
                    owner.source.as_ref(),
                    name,
                    recipe,
                    partition.parameter_selections(name),
                )?,
            ))
        })
        .collect::<Result<_, PreparationError>>()?;
    PreparedParameters::new(owner.source.clone(), recipes)
}
pub(super) fn partition_recipe<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
    catalog: &C,
    name: &str,
    source: &DerivedWeightRecipe,
    selections: Option<&[TensorSelection]>,
) -> Result<DerivedWeightRecipe, PreparationError> {
    let invalid = |error| PreparationError::Contract(format!("tensor parameter {name}: {error}"));
    let Some(selections) = selections else {
        source.infer(catalog).map_err(invalid)?;
        return Ok(source.clone());
    };
    let mut pieces = selections
        .iter()
        .map(|selection| {
            source
                .select_bounded(catalog, selection.clone())
                .map_err(invalid)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let recipe = if pieces.len() == 1 {
        pieces.pop().expect("one selected piece")
    } else {
        let Some(TensorSelection::Range { axis, .. }) = selections.first() else {
            return Err(PreparationError::Contract(format!(
                "tensor parameter {name} has no range selection"
            )));
        };
        DerivedWeightRecipe::Concatenate {
            axis: *axis,
            inputs: pieces,
        }
    };
    // Check the composed output before publishing the owner. PreparedParameters
    // retains the same exact metadata/provenance authority after selection pushdown.
    recipe.infer(catalog).map_err(invalid)?;
    Ok(recipe)
}
