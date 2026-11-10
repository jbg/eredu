//! Decoder parameter ownership composed from typed neural construction scopes.
use super::*;
use eredu_nn::DeclareLogicalParameters;

/// Validated common decoder scopes, with integrated or provider-owned feed-forward slots.
/// Invocation policy remains supplied by the exact configuration used to declare them.
#[derive(Debug, Clone)]
pub struct DecoderDeclaration {
    identity: String,
    pub(super) statics: StaticModuleConstruction,
    pub(super) blocks: Vec<topology::BlockParameters>,
    state: StateLayout,
}

impl DecoderDeclaration {
    /// Declares statics and dense blocks once, retaining independent invocation state.
    pub fn new(config: &impl Config) -> Result<Self, Error> {
        Self::declare(config, topology::BlockParameters::new)
    }

    /// Declares common attention, normalization and static slots for provider-owned stages.
    pub fn without_feed_forward(config: &impl Config) -> Result<Self, Error> {
        Self::declare(config, |config, layer| {
            topology::BlockParameters::with_feed_forward(config, layer, None)
        })
    }

    fn declare<C: Config>(
        config: &C,
        block: impl Fn(&C, usize) -> Result<topology::BlockParameters, Error>,
    ) -> Result<Self, Error> {
        config.validate_config()?;
        let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
        let statics = StaticModuleSpec::from_config(config).construction_specs()?;
        let blocks = (0..layers)
            .map(|layer| block(config, layer))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_blocks(config, statics, blocks)
    }

    /// Composes already declared common slots without regenerating family geometry.
    pub(crate) fn from_blocks(
        config: &impl Config,
        statics: StaticModuleConstruction,
        blocks: Vec<topology::BlockParameters>,
    ) -> Result<Self, Error> {
        config.validate_config()?;
        let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
        if blocks.len() != layers {
            return Err(Error::backend(
                "decoder common scopes change layer coverage",
            ));
        }
        let geometry = (0..layers)
            .map(|layer| {
                blocks
                    .get(config.block_parameter_owner(layer))
                    .map(|block| block.attention.geometry)
                    .ok_or_else(|| {
                        Error::backend("decoder state references an undeclared block owner")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let state = StateLayout::new(cache_layout_with_attention_geometry(config, geometry)?)
            .map_err(Error::backend)?;
        let declaration = Self {
            identity: config.architecture_fingerprint(),
            statics,
            blocks,
            state,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        config.validate_parameter_formats(&declaration)?;
        Ok(declaration)
    }

    /// Rejects a configuration that changes retained geometry, identity or execution policy.
    pub fn validate_for(&self, config: &impl Config) -> Result<(), Error> {
        config.validate_config()?;
        if self.identity != config.architecture_fingerprint() {
            return Err(Error::backend(
                "decoder declaration differs from its configuration",
            ));
        }
        Ok(())
    }

    /// Projects integrated gated geometry; provider families supply their replacement groups.
    pub fn parameter_description(
        &self,
        config: &impl Config,
        encoding: ParameterEncoding,
    ) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
        self.validate_for(config)
            .map_err(|error| ParallelPlanError::InvalidGroup(error.to_string()))?;
        parameter_description_with_encoding(
            config,
            |_| Ok(None),
            true,
            ParameterConstruction::Decoder(self),
            encoding,
        )
    }

    /// Projects provider-owned dense slots into a temporary common view.
    pub(crate) fn with_feed_forward_parameters(
        &self,
        config: &impl Config,
        mut parameters: impl FnMut(usize) -> Result<Option<eredu_nn::GatedProductSpec>, Error>,
    ) -> Result<Self, Error> {
        self.validate_for(config)?;
        let mut projected = self.clone();
        for (layer, block) in projected.blocks.iter_mut().enumerate() {
            block.feed_forward = parameters(layer)?;
        }
        eredu_nn::validate_logical_parameter_topology(&projected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(projected)
    }
    /// Combines common slots with family-owned placement groups.
    pub(crate) fn parameter_description_with_groups(
        &self,
        config: &impl Config,
        replacement: impl Fn(usize) -> Result<Option<Vec<ParameterGroupSpec>>, ParallelPlanError>,
    ) -> Result<ArchitectureParameterDescription, ParallelPlanError> {
        self.validate_for(config)
            .map_err(|e| ParallelPlanError::InvalidGroup(e.to_string()))?;
        parameter_description_with_encoding(
            config,
            replacement,
            true,
            ParameterConstruction::Decoder(self),
            ParameterEncoding::Stored,
        )
    }

    /// Projects invocation memory topology from the same attention, gated and static slots.
    pub fn execution_topology(
        &self,
        config: &impl Config,
    ) -> Result<eredu_runtime::execution_topology::TextExecutionTopology, Error> {
        topology::text_from_declaration(config, self)
    }

    /// Local configuration may change only the dimensions owned by partition geometry.
    pub(crate) fn validate_local_policy<C: PartitionedConfig>(
        &self,
        global: &C,
        local: &C,
    ) -> Result<(), Error> {
        self.validate_for(global)?;
        let mut policy = local.clone();
        policy.restore_global_geometry(global)?;
        if policy.architecture_fingerprint() != self.identity {
            return Err(Error::backend(
                "local decoder projection changes retained execution policy",
            ));
        }
        Ok(())
    }

    /// Projects rank-local body geometry while retaining parameter identities and equations.
    pub fn local_projection(
        &self,
        config: &impl Config,
        geometry: &[(eredu_nn::AttentionHeadGeometry, i32)],
    ) -> Result<DecoderLocalProjection, Error> {
        self.validate_for(config)?;
        if geometry.len() != self.blocks.len() {
            return Err(Error::backend(
                "local decoder geometry changes invocation coverage",
            ));
        }
        let blocks = self
            .blocks
            .iter()
            .zip(geometry)
            .map(|(block, (heads, intermediate))| {
                block.with_geometry(
                    *heads,
                    block.feed_forward.as_ref().map(|_| *intermediate),
                    config.query_key_norm_per_head_weights(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let heads = (0..blocks.len())
            .map(|layer| {
                blocks
                    .get(config.block_parameter_owner(layer))
                    .map(|block| block.attention.geometry)
                    .ok_or_else(|| Error::backend("local decoder state has an undeclared owner"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let state = StateLayout::new(cache_layout_with_attention_geometry(config, heads)?)
            .map_err(Error::backend)?;
        let projection = DecoderLocalProjection { blocks, state };
        // Shared invocations must retain compatible projected parameter geometry.
        eredu_nn::validate_logical_parameter_scope(&projection.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(projection)
    }

    /// Composes a selected declaration from retained local slots and unchanged static ownership.
    pub(crate) fn local_declaration(
        &self,
        global: &impl Config,
        selected: &impl Config,
        geometry: &[(eredu_nn::AttentionHeadGeometry, i32)],
    ) -> Result<Self, Error> {
        let local = self.local_projection(global, geometry)?;
        Self::from_blocks(selected, self.statics.clone(), local.blocks)
    }

    /// Applies encoding extensions after the family proves unchanged non-format policy.
    pub(crate) fn with_formats(&self, config: &impl Config) -> Result<Self, Error> {
        config.validate_config()?;
        let mut projected = self.clone();
        let format = |slot: &eredu_nn::ParameterSpec, old: &eredu_nn::LinearFormatSpec| {
            crate::linear_format::project_matrix_format(
                slot,
                old,
                config.linear_format(slot.id.as_str()),
            )
        };
        let affine = |spec: &mut eredu_nn::LinearSpec| -> Result<(), Error> {
            spec.format = format(&spec.weight, &spec.format)?;
            Ok(())
        };
        projected.statics.embedding.format = format(
            &projected.statics.embedding.weight,
            &projected.statics.embedding.format,
        )?;
        if let Some(head) = &mut projected.statics.head {
            head.format = format(&head.weight, &head.format)?;
        }
        for block in &mut projected.blocks {
            match &mut block.attention.input {
                topology::AttentionInputParameters::Split { query, key, value } => {
                    affine(query)?;
                    affine(key)?;
                    if let Some(value) = value {
                        affine(value)?;
                    }
                }
                topology::AttentionInputParameters::Fused { projection, .. } => affine(projection)?,
            }
            if let Some((gate, _)) = &mut block.attention.gate {
                affine(gate)?;
            }
            affine(&mut block.attention.output)?;
            if let Some(feed_forward) = &mut block.feed_forward {
                match feed_forward {
                    eredu_nn::GatedProductSpec::Split(spec) => {
                        affine(&mut spec.gate)?;
                        affine(&mut spec.up)?;
                        affine(&mut spec.down)?;
                    }
                    eredu_nn::GatedProductSpec::Fused(spec) => {
                        affine(&mut spec.gate_up)?;
                        affine(&mut spec.down)?;
                    }
                }
            }
        }
        projected.identity = config.architecture_fingerprint();
        eredu_nn::validate_logical_parameter_topology(&projected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        config.validate_parameter_formats(&projected)?;
        Ok(projected)
    }

    /// Declared slots for one logical invocation, including its sharing references.
    pub(crate) fn block_parameters(&self, layer: usize) -> &topology::BlockParameters {
        &self.blocks[layer]
    }

    /// Retained ordinary static slots used by cold binding projections.
    pub(crate) fn static_construction(&self) -> &StaticModuleConstruction {
        &self.statics
    }

    /// Independent mutable attention state for every logical layer invocation.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state
    }

    /// Projects parameter ownership, invocation boundaries and state into execution contracts.
    pub fn component_definition(
        &self,
        config: &impl Config,
    ) -> Result<eredu_runtime::ComponentDefinition, Error> {
        component_definition(
            config,
            self.parameter_description(config, ParameterEncoding::Stored)
                .map_err(Error::backend_source)?,
            self.state.clone(),
        )
    }
}

impl DeclareLogicalParameters for DecoderDeclaration {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.statics.logical_parameters()?;
        parameters.extend(self.blocks[..].logical_parameters()?);
        Ok(parameters)
    }
}

/// Rank-local common body scopes with independent complete invocation-state geometry.
/// Vocabulary placement remains an explicit projection of the source static scopes.
#[derive(Debug, Clone)]
pub struct DecoderLocalProjection {
    blocks: Vec<topology::BlockParameters>,
    state: StateLayout,
}
impl DecoderLocalProjection {
    /// Complete local-head state, before pipeline slicing.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state
    }
}
impl DeclareLogicalParameters for DecoderLocalProjection {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.blocks[..].logical_parameters()
    }
}

/// Architecture-owned projection slots for canonical parameter owners.
/// Independent execution invocations and mutable state remain in the decoder schedule.
#[derive(Debug, Clone)]
pub struct ProjectionDeclarations<S> {
    identity: String,
    owners: std::collections::BTreeMap<usize, S>,
}
impl<S: eredu_nn::DeclareLogicalParameters> ProjectionDeclarations<S> {
    /// Declares each shared body owner once, without backend contexts or tensors.
    pub fn new(
        config: &impl Config,
        mut declare: impl FnMut(usize) -> Result<S, Error>,
    ) -> Result<Self, Error> {
        config.validate_config()?;
        let schedule = DecoderSchedule::new(config)?;
        let mut owners = std::collections::BTreeMap::new();
        for (owner, _) in schedule.resources() {
            if let DecoderOperation::Body { layer } = schedule.operation(owner)? {
                owners.insert(layer, declare(layer)?);
            }
        }
        let declarations = Self {
            identity: config.architecture_fingerprint(),
            owners,
        };
        eredu_nn::validate_logical_parameter_scope(&declarations.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(declarations)
    }
    /// Rejects use under a different normalized architecture or execution policy.
    pub fn validate_for(&self, config: &impl Config) -> Result<(), Error> {
        config.validate_config()?;
        if self.identity != config.architecture_fingerprint() {
            return Err(Error::backend(
                "projection declarations differ from their configuration",
            ));
        }
        Ok(())
    }
    /// Joins typed slot references to declared geometry, sharing, encoding and ownership.
    pub fn validate_against(
        &self,
        config: &impl Config,
        description: &eredu_runtime::ArchitectureParameterDescription,
    ) -> Result<(), Error> {
        self.validate_for(config)?;
        let schedule = DecoderSchedule::new(config)?;
        let group = eredu_runtime::ExecutionGroupId::new(TEXT_DECODER_EXECUTION_GROUP)
            .map_err(Error::backend)?;
        let members = description
            .groups()
            .iter()
            .flat_map(|owned| {
                owned
                    .group()
                    .members()
                    .iter()
                    .map(move |member| (member.target(), (owned.owner(), member)))
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        for (layer, parameters) in &self.owners {
            for slot in parameters.logical_parameters()? {
                let (owner, member) = members
                    .get(slot.parameter.id.as_str())
                    .ok_or_else(|| Error::backend("unresolved projection parameter reference"))?;
                if member.logical_shape() != Some(slot.shape.as_slice())
                    || member.logical_role() != Some(slot.role)
                    || member.alias_of() != slot.parameter.alias_of.as_ref()
                    || member.logical_linear_format() != slot.linear_format
                    || !owner.has_unit_consumer(&group, schedule.body_operation(*layer))
                {
                    return Err(Error::backend(
                        "projection declaration differs from its parameter ownership or geometry",
                    ));
                }
            }
        }
        Ok(())
    }
    /// Canonical parameter-layer identities, independent of selected invocation ranges.
    pub fn owner_layers(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.owners.keys().copied()
    }
    /// Exact typed parameters of one declared canonical body owner.
    pub fn parameters(&self, owner: usize) -> Result<&S, Error> {
        self.owners
            .get(&owner)
            .ok_or_else(|| Error::backend("undeclared projection owner"))
    }
}
impl<S: eredu_nn::DeclareLogicalParameters> eredu_nn::DeclareLogicalParameters
    for ProjectionDeclarations<S>
{
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        for owner in self.owners.values() {
            parameters.extend(owner.logical_parameters()?);
        }
        Ok(parameters)
    }
}
