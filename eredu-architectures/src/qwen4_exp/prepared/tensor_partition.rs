//! Tensor-rank source authority derived from already admitted canonical owners.
use super::*;
use crate::qwen4_exp::target::TargetTensorPartition;

/// Local target parameters and geometry over the original admitted source.
/// Preparation reads no parameter/table payload and does not reopen artifacts.
/// Vocabulary, residual mixers and lexical parameters retain their replicated owners.
#[derive(Clone)]
pub struct PreparedTensorTarget {
    source: PreparedTarget,
    source_bound: BoundTargetSpec,
    bound: BoundTargetSpec,
    partition: TargetTensorPartition,
    static_parameters: PreparedParameterRecipes,
    units: Vec<PreparedParameterRecipes>,
    expert_banks: BTreeMap<String, Arc<ParameterBankRecipes>>,
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
        let partition_owner = |owner: &PreparedParameterRecipes| {
            owner
                .clone()
                .map_recipes("qwen4_exp prepared owner", |name, recipe, source| {
                    eredu_runtime::select_parameter_recipe(
                        source,
                        name,
                        recipe,
                        partition.parameter_selections(name),
                    )
                    .map_err(PreparationError::from)
                })
        };
        let static_parameters = partition_owner(&self.static_parameters)?;
        let units = self
            .units
            .iter()
            .map(partition_owner)
            .collect::<Result<_, _>>()?;
        let expert_banks = self
            .expert_banks
            .iter()
            .map(|(root, bank)| {
                let bank = bank.map_recipes(self.artifact.as_ref(), |name, recipe| {
                    let canonical = format!("{root}.{name}");
                    eredu_runtime::select_parameter_recipe(
                        self.artifact.as_ref(),
                        &canonical,
                        recipe,
                        partition.parameter_selections(&canonical),
                    )
                    .map_err(PreparationError::from)
                })?;
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
    pub fn static_parameters(&self) -> &PreparedParameterRecipes {
        &self.static_parameters
    }
    /// Exact rank-local ordinary recipes for one execution ordinal.
    pub fn unit(&self, ordinal: usize) -> Result<&PreparedParameterRecipes, PreparationError> {
        self.units.get(ordinal).ok_or_else(|| {
            PreparationError::Contract("execution owner is outside tensor target".into())
        })
    }
    /// One independently acquired expert, preserving its singleton group axis.
    pub fn expert(
        &self,
        ordinal: usize,
        expert: usize,
    ) -> Result<PreparedParameterRecipes, PreparationError> {
        let Some(UnitSpec::Decoder { layer, .. }) = self.spec().units.get(ordinal) else {
            return Err(PreparationError::Contract(
                "expert owner is not a decoder".into(),
            ));
        };
        let root = format!("model.layers.{layer}.mlp.experts");
        let recipes = select_expert(&self.expert_banks, &root, expert)?;
        PreparedParameterRecipes::new(
            "qwen4_exp prepared owner",
            self.source.artifact.clone(),
            recipes
                .into_iter()
                .map(|(name, recipe)| (format!("{root}.{name}"), recipe))
                .collect(),
        )
        .map_err(PreparationError::from)
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
