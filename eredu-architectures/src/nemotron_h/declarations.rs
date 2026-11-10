//! Retained target and prediction scopes, independent of mutable state and placement.
use super::{LayerGeometry, LayerPolicy, ModelArgs};
use eredu_nn::{
    DeclareLogicalParameters, Error, LogicalParameterDeclaration, NormalizationConstructionSpec,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) enum TargetOperatorParameters {
    Mamba(super::mamba::Mamba2Spec),
    Attention(super::attention::AttentionParameters),
    Dense(super::mlp::DenseParameters),
    Sparse(super::mlp::SparseMoeSpec),
}
#[derive(Debug, Clone)]
pub(crate) struct TargetBlockParameters {
    pub norm: NormalizationConstructionSpec,
    pub operator: TargetOperatorParameters,
}
impl TargetBlockParameters {
    pub fn new(args: &ModelArgs, layer: usize) -> Result<Self, Error> {
        let policy = *args
            .layer_schedule
            .get(layer)
            .ok_or_else(|| Error::backend("target layer is outside its schedule"))?;
        Self::at(args, &format!("model.layers.{layer}"), policy, layer, false)
    }
    pub fn at(
        args: &ModelArgs,
        root: &str,
        policy: LayerPolicy,
        identity: usize,
        prediction: bool,
    ) -> Result<Self, Error> {
        let operator = match policy {
            LayerPolicy::Mamba => TargetOperatorParameters::Mamba(super::mamba::mixer_spec(
                args,
                &format!("{root}.mamba"),
                args.mamba_num_heads,
                args.n_groups,
            )?),
            LayerPolicy::SelfAttention(_) => {
                TargetOperatorParameters::Attention(super::attention::AttentionParameters::new(
                    args,
                    &format!("{root}.{}", if prediction { "mixer" } else { "attention" }),
                    args.num_attention_heads,
                    args.num_key_value_heads,
                )?)
            }
            LayerPolicy::DenseMlp => {
                TargetOperatorParameters::Dense(super::mlp::DenseParameters::new(
                    args,
                    &format!("{root}.mlp"),
                    args.intermediate_size,
                )?)
            }
            LayerPolicy::SparseMoe => TargetOperatorParameters::Sparse(
                super::mlp::SparseMoeSpec::for_layer(args, identity)?,
            ),
        };
        let spec = Self {
            norm: super::block::normalization_spec(args, &format!("{root}.norm.weight"))?,
            operator,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
    pub(crate) fn geometry(&self) -> LayerGeometry {
        match &self.operator {
            TargetOperatorParameters::Mamba(spec) => LayerGeometry::Mamba {
                heads: spec.geometry.heads(),
                groups: spec.geometry.groups(),
            },
            TargetOperatorParameters::Attention(spec) => LayerGeometry::Attention {
                query_heads: spec.projection.geometry.query_heads(),
                kv_heads: spec.projection.geometry.key_value_heads(),
            },
            TargetOperatorParameters::Dense(spec) => LayerGeometry::DenseMlp {
                intermediate: spec.network.up.output,
            },
            TargetOperatorParameters::Sparse(spec) => LayerGeometry::SparseMoe {
                routed: spec.experts.intermediate_dimensions(),
                shared: spec.shared.network.up.output,
            },
        }
    }
    pub(crate) fn state_policy(
        &self,
        policy: LayerPolicy,
    ) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        match (&self.operator, policy) {
            (TargetOperatorParameters::Mamba(spec), LayerPolicy::Mamba) => spec.state_policy(),
            (TargetOperatorParameters::Attention(spec), LayerPolicy::SelfAttention(attention)) => {
                spec.state_policy(attention)
            }
            (TargetOperatorParameters::Dense(_), LayerPolicy::DenseMlp)
            | (TargetOperatorParameters::Sparse(_), LayerPolicy::SparseMoe) => {
                Ok(eredu_core::cache::LayerCachePolicy::NoState)
            }
            _ => Err(Error::backend(
                "state policy differs from its retained operator scope",
            )),
        }
    }
    pub fn local(
        mut self,
        geometry: LayerGeometry,
        routed: Option<eredu_nn::GroupedRelu2Spec>,
    ) -> Result<Self, Error> {
        self.operator = match (self.operator, geometry) {
            (TargetOperatorParameters::Mamba(spec), LayerGeometry::Mamba { heads, groups }) => {
                TargetOperatorParameters::Mamba(spec.local(heads, groups)?)
            }
            (
                TargetOperatorParameters::Attention(mut spec),
                LayerGeometry::Attention {
                    query_heads,
                    kv_heads,
                },
            ) => {
                let old = spec.projection.geometry;
                if query_heads > old.query_heads() || kv_heads > old.key_value_heads() {
                    return Err(Error::backend("local attention exceeds retained geometry"));
                }
                let g = eredu_nn::AttentionHeadGeometry::new(
                    query_heads,
                    kv_heads,
                    old.key_dimensions(),
                    old.value_dimensions(),
                )?;
                spec.projection.query.output = g.query_width();
                spec.projection.key.output = g.key_width();
                spec.projection.value.output = g.value_width();
                spec.projection.output.input = g.attended_width();
                spec.projection.geometry = g;
                TargetOperatorParameters::Attention(spec)
            }
            (
                TargetOperatorParameters::Dense(mut spec),
                LayerGeometry::DenseMlp { intermediate },
            ) => {
                if intermediate <= 0 || intermediate > spec.network.up.output {
                    return Err(Error::backend("local MLP exceeds retained geometry"));
                }
                spec.network.up.output = intermediate;
                spec.network.down.input = intermediate;
                TargetOperatorParameters::Dense(spec)
            }
            (
                TargetOperatorParameters::Sparse(spec),
                LayerGeometry::SparseMoe {
                    routed: width,
                    shared,
                },
            ) => {
                let experts = match routed {
                    Some(experts) => experts,
                    None => spec
                        .experts
                        .clone()
                        .with_group_geometry(spec.experts.group_count(), width)?,
                };
                TargetOperatorParameters::Sparse(spec.with_local_geometry(experts, shared)?)
            }
            _ => {
                return Err(Error::backend(
                    "target scope and selected operator geometry differ",
                ))
            }
        };
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(self)
    }
    pub(crate) fn with_formats(mut self, args: &ModelArgs) -> Result<Self, Error> {
        let project = |spec: &mut eredu_nn::LinearSpec| -> Result<(), Error> {
            spec.format = crate::linear_format::project_matrix_format(
                &spec.weight,
                &spec.format,
                args.weight_quantization_for(spec.weight.id.as_str()).into(),
            )?;
            Ok(())
        };
        match &mut self.operator {
            TargetOperatorParameters::Mamba(spec) => {
                project(&mut spec.in_proj)?;
                project(&mut spec.out_proj)?;
            }
            TargetOperatorParameters::Attention(spec) => {
                project(&mut spec.projection.query)?;
                project(&mut spec.projection.key)?;
                project(&mut spec.projection.value)?;
                project(&mut spec.projection.output)?;
            }
            TargetOperatorParameters::Dense(spec) => {
                project(&mut spec.network.up)?;
                project(&mut spec.network.down)?;
            }
            TargetOperatorParameters::Sparse(spec) => {
                let name = spec.selector.weight().id.as_str();
                let format = crate::linear_format::project_matrix_format(
                    spec.selector.weight(),
                    spec.selector.format(),
                    args.weight_quantization_for(name).into(),
                )?;
                spec.selector = spec.selector.clone().with_format(format)?;
                spec.experts = spec.experts.map_projections(|p| {
                    crate::linear_format::project_expert_format(
                        p,
                        args.weight_quantization_for(p.weight().id.as_str()).into(),
                        eredu_nn::LinearRowLayout::Contiguous,
                    )
                })?;
                project(&mut spec.shared.network.up)?;
                project(&mut spec.shared.network.down)?;
            }
        }
        self.logical_parameters()?;
        Ok(self)
    }
}
impl DeclareLogicalParameters for TargetBlockParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.norm.logical_parameters()?;
        p.extend(match &self.operator {
            TargetOperatorParameters::Mamba(s) => s.logical_parameters()?,
            TargetOperatorParameters::Attention(s) => s.logical_parameters()?,
            TargetOperatorParameters::Dense(s) => s.logical_parameters()?,
            TargetOperatorParameters::Sparse(s) => s.logical_parameters()?,
        });
        Ok(p)
    }
}

/// Complete Nemotron parameter scopes with independent target and prediction ownership.
#[derive(Debug, Clone)]
pub struct Declaration {
    args: ModelArgs,
    pub(crate) statics: crate::decoder::StaticModuleConstruction,
    pub(crate) blocks: Vec<TargetBlockParameters>,
    pub(crate) predictions: Vec<super::mtp::PredictionUnitParameters>,
    definition: Arc<eredu_runtime::ComponentDefinition>,
}
impl Declaration {
    /// Declares target and prediction components without source values or backend allocation.
    pub fn new(args: ModelArgs) -> Result<Self, Error> {
        args.validate().map_err(Error::backend_source)?;
        let statics = super::static_spec(&args).construction_specs()?;
        let blocks = (0..args.num_hidden_layers as usize)
            .map(|i| TargetBlockParameters::new(&args, i))
            .collect::<Result<_, _>>()?;
        let predictions = super::mtp::PredictionUnitParameters::all(&args)?;
        Self::from_parts(args, statics, blocks, predictions)
    }
    fn from_parts(
        args: ModelArgs,
        statics: crate::decoder::StaticModuleConstruction,
        blocks: Vec<TargetBlockParameters>,
        predictions: Vec<super::mtp::PredictionUnitParameters>,
    ) -> Result<Self, Error> {
        let parameters = super::parameters::parameter_description_with_scopes(
            &args,
            &statics,
            &blocks,
            &predictions,
        )
        .map_err(Error::backend_source)?;
        let policies = blocks
            .iter()
            .zip(args.layer_schedule.iter().copied())
            .map(|(block, policy)| block.state_policy(policy))
            .chain(
                predictions
                    .iter()
                    .map(|unit| unit.block.state_policy(unit.policy)),
            )
            .collect::<Result<Vec<_>, _>>()?;
        let state = super::config::state_layout_from_policies(blocks.len(), policies)
            .map_err(Error::backend_source)?;
        let definition = Arc::new(super::parameters::component_definition(parameters, state)?);
        let scope = Self {
            args,
            statics,
            blocks,
            predictions,
            definition,
        };
        eredu_nn::validate_logical_parameter_topology(&scope.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let formats = scope
            .args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.iter())
            .map(|(name, format)| (name.as_str(), Some((*format).into())));
        let selected = scope
            .args
            .quantized_weights
            .iter()
            .flat_map(|names| names.iter())
            .map(|name| {
                (
                    name.as_str(),
                    scope.args.weight_quantization_for(name).map(Into::into),
                )
            });
        crate::format_references::validate(
            &scope,
            formats.chain(selected),
            || Ok(Vec::new()),
            || scope.safetensors_plan(),
        )?;
        Ok(scope)
    }
    pub(crate) fn block(&self, identity: usize) -> Option<&TargetBlockParameters> {
        if identity < self.blocks.len() {
            self.blocks.get(identity)
        } else {
            self.predictions
                .get(identity - self.blocks.len())
                .map(|unit| &unit.block)
        }
    }
    /// Routed banks keyed by independent invocation ordinal, borrowing canonical owners.
    pub fn expert_banks(&self) -> std::collections::BTreeMap<usize, &eredu_nn::GroupedRelu2Spec> {
        self.blocks
            .iter()
            .chain(self.predictions.iter().map(|unit| &unit.block))
            .enumerate()
            .filter_map(|(identity, block)| match &block.operator {
                TargetOperatorParameters::Sparse(spec) => Some((identity, &spec.experts)),
                _ => None,
            })
            .collect()
    }
    pub(crate) fn expert_bank(
        &self,
        identity: usize,
    ) -> Result<&eredu_nn::GroupedRelu2Spec, Error> {
        match self.block(identity).map(|block| &block.operator) {
            Some(TargetOperatorParameters::Sparse(spec)) => Ok(&spec.experts),
            _ => Err(Error::backend(
                "invocation has no declared Nemotron expert bank",
            )),
        }
    }
    /// Projects state from retained operator scopes at explicit local geometry.
    pub fn state_layout_for(
        &self,
        geometry: &[LayerGeometry],
    ) -> Result<eredu_runtime::StateLayout, Error> {
        let scopes = self
            .blocks
            .iter()
            .zip(self.args.layer_schedule.iter().copied())
            .chain(
                self.predictions
                    .iter()
                    .map(|unit| (&unit.block, unit.policy)),
            );
        if geometry.len() != self.blocks.len() + self.predictions.len() {
            return Err(Error::backend(
                "state geometry differs from the declared invocation count",
            ));
        }
        let policies = scopes
            .zip(geometry)
            .map(
                |((block, policy), geometry)| match (&block.operator, geometry) {
                    (TargetOperatorParameters::Dense(_), LayerGeometry::DenseMlp { .. })
                    | (TargetOperatorParameters::Sparse(_), LayerGeometry::SparseMoe { .. }) => {
                        block.state_policy(policy)
                    }
                    _ => block.clone().local(*geometry, None)?.state_policy(policy),
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        super::config::state_layout_from_policies(self.blocks.len(), policies)
            .map_err(Error::backend_source)
    }
    /// Validated family equations and storage selection.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }
    /// Parameter, invocation, state and placement semantics from the retained slots.
    pub fn definition(&self) -> &Arc<eredu_runtime::ComponentDefinition> {
        &self.definition
    }
    /// Projects ordinary target ownership while retaining its exact neural scopes.
    pub fn prediction_target(&self) -> Result<Self, Error> {
        let args = self
            .args
            .prediction_target()
            .map_err(Error::backend_source)?;
        Self::from_parts(args, self.statics.clone(), self.blocks.clone(), Vec::new())
    }
    /// Projects exact physical formats without reconstructing logical geometry or identities.
    pub fn with_formats(&self, args: ModelArgs) -> Result<Self, Error> {
        fn logical(mut a: ModelArgs) -> ModelArgs {
            a.weight_quantization = None;
            a.quantized_weights = None;
            a.quantized_weight_configs = None;
            a
        }
        args.validate().map_err(Error::backend_source)?;
        if logical(args.clone()) != logical(self.args.clone()) {
            return Err(Error::backend(
                "target format projection changes retained architecture policy",
            ));
        }
        let mut embedding = self.statics.embedding_spec().clone();
        embedding.format = crate::linear_format::project_matrix_format(
            &embedding.weight,
            &embedding.format,
            args.weight_quantization_for(embedding.weight.id.as_str())
                .into(),
        )?;
        let head = self
            .statics
            .head_spec()
            .cloned()
            .map(|mut head| {
                head.format = crate::linear_format::project_matrix_format(
                    &head.weight,
                    &head.format,
                    args.weight_quantization_for(head.weight.id.as_str()).into(),
                )?;
                Ok::<_, Error>(head)
            })
            .transpose()?;
        let statics = crate::decoder::StaticModuleConstruction::new(
            embedding,
            head,
            self.statics.normalization_spec().clone(),
        )?;
        let blocks = self
            .blocks
            .iter()
            .cloned()
            .map(|b| b.with_formats(&args))
            .collect::<Result<_, _>>()?;
        let predictions = self
            .predictions
            .iter()
            .cloned()
            .map(|p| p.with_formats(&args))
            .collect::<Result<_, _>>()?;
        Self::from_parts(args, statics, blocks, predictions)
    }
    /// Projects released SafeTensors aliases and physical alternatives from retained slots.
    pub fn safetensors_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, String> {
        super::checkpoint::safetensors_plan_from_declaration(self)
    }
    /// Projects the published GGUF target catalog without changing its prediction exclusions.
    pub fn gguf_plan(&self) -> Result<eredu_checkpoint::schema::GgufCheckpointPlan, String> {
        super::checkpoint::gguf_plan_from_declaration(self)
    }
    /// Strict canonical recipe-output validation after physical source normalization.
    pub fn canonical_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, String> {
        let mut tensors = Vec::new();
        for p in self.logical_parameters().map_err(|e| e.to_string())? {
            super::checkpoint::add_safe_declaration(&mut tensors, p.parameter.id.as_str(), &p)?;
        }
        let mut policy = eredu_checkpoint::schema::CatalogPolicy::strict();
        policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
        eredu_checkpoint::schema::SafetensorsCheckpointPlan::new(
            "Nemotron-H canonical parameters",
            tensors,
            vec![],
            policy,
        )
        .map_err(|e| e.to_string())
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.statics.logical_parameters()?;
        for b in &self.blocks {
            p.extend(b.logical_parameters()?);
        }
        for prediction in &self.predictions {
            p.extend(prediction.logical_parameters()?);
        }
        Ok(p)
    }
}
impl eredu_runtime::ComponentPreparationDeclaration for Declaration {
    fn parameters(&self) -> &eredu_runtime::ArchitectureParameterDescription {
        self.definition.parameters()
    }
}
