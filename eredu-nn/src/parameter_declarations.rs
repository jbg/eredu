//! Logical parameter projections from neural construction specifications.

use crate::{
    EmbeddingSpec, Error, GatedProductGroupLayout, GroupedGatedProductSpec, GroupedProjectionSpec,
    GroupedRelu2Spec, LinearSpec, NormalizationConstructionSpec, NormalizationScale, ParameterSpec,
    TopKGroupSelectorSpec,
};

/// Neural meaning of a logical construction slot, independent of its encoding.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LogicalParameterRole {
    /// Affine projection weight, including grouped matrices and structured patch kernels.
    LinearWeight,
    /// Additive affine output bias.
    LinearBias,
    /// Additive learned normalization offset.
    NormalizationBias,
    /// Lookup table whose rows represent tokens.
    Embedding,
    /// Mean-vector bank indexed by discrete codes, including stacked residual depths.
    Codebook,
    /// Retained scalar variance associated with a discrete mean-vector codebook.
    CodebookVariance,
    /// Stored initialization status for a codebook.
    CodebookInitialization,
    /// EMA usage statistics associated with codebook entries.
    CodebookUsage,
    /// EMA vector sums from which codebook means are derived.
    CodebookSum,
    /// Learned normalization scale or scale offset.
    Normalization,
    /// Bias used to rank routes independently of mixture coefficients.
    RoutingCorrection,
    /// Learned bias applied to a recurrent update or decay coefficient.
    RecurrentBias,
    /// Learned recurrent transition magnitude or rate.
    RecurrentTransition,
    /// Learned direct skip coefficient in a recurrent mixer.
    RecurrentSkip,
    /// Integer lookup selecting routes for each input token.
    RoutingTable,
    /// Integer lookup translating source token identities into target token identities.
    TokenMapping,
    /// Learned routing coefficient multiplier.
    RoutingScale,
    /// Learned attention logit for a head's non-token sink.
    AttentionSink,
    /// Learned per-dimension attention scaling coefficient.
    AttentionScale,
    /// Learned head vector added to queries before scoring content keys.
    AttentionContentBias,
    /// Learned head vector added to queries before scoring relative-position keys.
    AttentionPositionBias,
    /// Additive position-dependent logits.
    PositionBias,
    /// Learned spatial position vectors added to features.
    PositionEmbedding,
    /// Learned or frozen conditioning features introduced at explicit invocation boundaries.
    ConditioningEmbedding,
    /// Learned logits controlling featurewise mixing of separate streams.
    FeatureGate,
    /// Learned feature center subtracted before standardization.
    FeatureCenter,
    /// Learned multiplier applied to centered features.
    FeatureScale,
    /// Learned scalar lower bound applied before a projection.
    ClipInputMinimum,
    /// Learned scalar upper bound applied before a projection.
    ClipInputMaximum,
    /// Learned scalar lower bound applied after a projection.
    ClipOutputMinimum,
    /// Learned scalar upper bound applied after a projection.
    ClipOutputMaximum,
    /// Learned multiplier applied to the complete residual stream.
    ResidualScale,
    /// Projection producing residual-stream mixing or collapse coefficients.
    ResidualMixWeight,
    /// Additive base for residual-stream coefficients.
    ResidualMixBias,
    /// Learned multiplier for residual-stream coefficient logits.
    ResidualMixScale,
}

/// Logical geometry projected from a typed neural construction specification.
/// Physical encodings and companions remain storage extensions. Construction
/// specs may retain a typed reference to the exact extension for this slot.
#[derive(Debug, Clone)]
pub struct LogicalParameterDeclaration<'a> {
    /// Authoritative construction slot, including sharing metadata.
    pub parameter: &'a ParameterSpec,
    /// Exact logical shape before encoding or partitioning.
    pub shape: Vec<usize>,
    /// Meaning declared by the neural operator.
    pub role: LogicalParameterRole,
    /// Exact construction-owned linear encoding, independent of logical geometry.
    /// Absence adds no encoded-linear expansion or backend-support claim.
    pub linear_format: Option<&'a crate::LinearFormatSpec>,
}

impl<'a> LogicalParameterDeclaration<'a> {
    /// Validates positive dimensions and checked element count without allocation
    /// of tensor storage. Scalars have an empty shape.
    pub fn new(
        parameter: &'a ParameterSpec,
        shape: &[i32],
        role: LogicalParameterRole,
    ) -> Result<Self, Error> {
        let shape = shape
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension)
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| {
                        Error::backend(format!(
                            "parameter {} has invalid dimension {dimension}",
                            parameter.id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or_else(|| {
                Error::backend(format!(
                    "parameter {} element count overflows",
                    parameter.id
                ))
            })?;
        Ok(Self {
            parameter,
            shape,
            role,
            linear_format: None,
        })
    }

    /// References the retained storage extension for this exact primary identity.
    pub fn with_linear_format(
        mut self,
        format: &'a crate::LinearFormatSpec,
    ) -> Result<Self, Error> {
        format.validate_for_weight(self.parameter)?;
        self.linear_format = Some(format);
        Ok(self)
    }
}

/// Source-free logical parameters consumed by construction and cold projections.
/// This does not declare observation hooks, placement, or backend support.
pub trait DeclareLogicalParameters {
    /// Projects exact parameter identities, sharing and geometry from this spec.
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error>;
}

/// A collected projection retains its references to the construction-owned slots.
impl DeclareLogicalParameters for Vec<LogicalParameterDeclaration<'_>> {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(self.clone())
    }
}

impl<S: DeclareLogicalParameters> DeclareLogicalParameters for [S] {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        for spec in self {
            parameters.extend(spec.logical_parameters()?);
        }
        Ok(parameters)
    }
}

impl DeclareLogicalParameters for LinearSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.format.validate_for_weight(&self.weight)?;
        let mut parameters = vec![LogicalParameterDeclaration::new(
            &self.weight,
            &[self.output, self.input],
            LogicalParameterRole::LinearWeight,
        )?
        .with_linear_format(&self.format)?];
        if let Some(bias) = &self.bias {
            parameters.push(LogicalParameterDeclaration::new(
                bias,
                &[self.output],
                LogicalParameterRole::LinearBias,
            )?);
        }
        Ok(parameters)
    }
}

impl DeclareLogicalParameters for EmbeddingSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.format.validate_for_weight(&self.weight)?;
        Ok(vec![LogicalParameterDeclaration::new(
            &self.weight,
            &[self.vocabulary, self.dimensions],
            LogicalParameterRole::Embedding,
        )?
        .with_linear_format(&self.format)?])
    }
}

impl DeclareLogicalParameters for NormalizationConstructionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate()?;
        match &self.scale {
            NormalizationScale::Learned(weight)
            | NormalizationScale::LearnedOffset { weight, .. } => {
                Ok(vec![LogicalParameterDeclaration::new(
                    weight,
                    &[self.dimensions],
                    LogicalParameterRole::Normalization,
                )?])
            }
            NormalizationScale::Unit => Ok(vec![]),
        }
    }
}

impl DeclareLogicalParameters for TopKGroupSelectorSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate()?;
        let groups = self.selection().group_count();
        let mut parameters = vec![LogicalParameterDeclaration::new(
            self.weight(),
            &[groups, self.input_dimensions()],
            LogicalParameterRole::LinearWeight,
        )?
        .with_linear_format(self.format())?];
        for (parameter, role) in [
            (self.bias(), LogicalParameterRole::LinearBias),
            (
                self.correction_bias(),
                LogicalParameterRole::RoutingCorrection,
            ),
            (self.coefficient_scale(), LogicalParameterRole::RoutingScale),
        ] {
            if let Some(parameter) = parameter {
                parameters.push(LogicalParameterDeclaration::new(
                    parameter,
                    &[groups],
                    role,
                )?);
            }
        }
        if let Some(transform) = self.input_transform() {
            parameters.push(LogicalParameterDeclaration::new(
                transform.scale(),
                &[self.input_dimensions()],
                LogicalParameterRole::Normalization,
            )?);
        }
        Ok(parameters)
    }
}

impl GroupedProjectionSpec {
    pub(crate) fn logical_parameters_with_shape(
        &self,
        shape: &[i32],
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = vec![LogicalParameterDeclaration::new(
            self.weight(),
            shape,
            LogicalParameterRole::LinearWeight,
        )?
        .with_linear_format(self.format())?];
        if let Some(bias) = self.bias() {
            parameters.push(LogicalParameterDeclaration::new(
                bias,
                &shape[..shape.len() - 1],
                LogicalParameterRole::LinearBias,
            )?);
        }
        Ok(parameters)
    }
}
/// Borrowed logical slots of a component-major packed gated-product bank.
/// Geometry and encodings come from the construction specification; source
/// layouts, ownership and execution invocations remain separate contracts.
#[derive(Debug, Clone)]
pub struct PackedGatedProductParameters<'a> {
    /// Fused gate/up matrix, with group, component-major rows and input axes.
    gate_up: LogicalParameterDeclaration<'a>,
    gate_up_bias: Option<LogicalParameterDeclaration<'a>>,
    /// Down matrix, with group, output and intermediate axes.
    down: LogicalParameterDeclaration<'a>,
    down_bias: Option<LogicalParameterDeclaration<'a>>,
}
impl<'a> PackedGatedProductParameters<'a> {
    /// Fused gate/up primary, retaining the construction-owned identity and encoding.
    pub fn gate_up(&self) -> &LogicalParameterDeclaration<'a> {
        &self.gate_up
    }
    /// Down primary, retaining the construction-owned identity and encoding.
    pub fn down(&self) -> &LogicalParameterDeclaration<'a> {
        &self.down
    }
    /// Optional additive bias of the fused gate/up projection.
    pub fn gate_up_bias(&self) -> Option<&LogicalParameterDeclaration<'a>> {
        self.gate_up_bias.as_ref()
    }
    /// Optional additive bias of the down projection.
    pub fn down_bias(&self) -> Option<&LogicalParameterDeclaration<'a>> {
        self.down_bias.as_ref()
    }
    /// Logical shape of either unfused gate or up bias across all groups.
    pub fn unfused_bias_shape(&self) -> Option<Vec<usize>> {
        self.gate_up_bias().map(|bias| {
            let mut shape = bias.shape.clone();
            shape[1] /= 2;
            shape
        })
    }
    /// Number of independently addressable parameter groups.
    pub fn group_count(&self) -> usize {
        self.gate_up.shape[0]
    }
    /// Logical shape of either unfused gate or up matrix across all groups.
    pub fn unfused_shape(&self) -> Vec<usize> {
        let mut shape = self.gate_up.shape.clone();
        shape[1] /= 2;
        shape
    }
    /// Exact selected encodings of the two primary projections.
    pub fn encodings(&self) -> Result<[eredu_checkpoint::LinearFormat; 2], Error> {
        let encoding = |parameter: &LogicalParameterDeclaration<'_>| {
            parameter
                .linear_format
                .map(|format| format.encoding())
                .ok_or_else(|| {
                    Error::backend(format!(
                        "grouped projection {} has no declared encoding",
                        parameter.parameter.id
                    ))
                })
        };
        Ok([encoding(&self.gate_up)?, encoding(&self.down)?])
    }
}
impl GroupedGatedProductSpec {
    /// Projects exact packed primary and bias slots, rejecting independent layouts.
    pub fn packed_parameters(&self) -> Result<PackedGatedProductParameters<'_>, Error> {
        let GatedProductGroupLayout::Packed { gate_up, down } = self.layout() else {
            return Err(Error::backend(
                "packed parameter projection requires a packed gated-product bank",
            ));
        };
        let parameters = self.logical_parameters()?;
        validate_logical_parameter_scope(&parameters).map_err(Error::backend_source)?;
        let find = |target: &ParameterSpec| {
            parameters
                .iter()
                .find(|parameter| parameter.parameter.id == target.id)
                .cloned()
                .ok_or_else(|| {
                    Error::backend(format!("undeclared grouped projection {}", target.id))
                })
        };
        Ok(PackedGatedProductParameters {
            gate_up: find(gate_up.weight())?,
            gate_up_bias: gate_up.bias().map(find).transpose()?,
            down: find(down.weight())?,
            down_bias: down.bias().map(find).transpose()?,
        })
    }
}

/// Logical slots of an addressable member, retaining the bank's parameter identities.
/// Packed slots keep a singleton group axis; independent slots keep their ordinary axes.
/// This projection does not declare storage recipes, residency or execution support.
pub trait DeclareGroupedParameters: DeclareLogicalParameters {
    /// Projects one member and rejects an index outside the retained bank.
    fn logical_member_parameters(
        &self,
        member: usize,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error>;
}

fn checked_member_range(groups: i32, member: usize) -> Result<std::ops::Range<usize>, Error> {
    if member >= usize::try_from(groups).map_err(Error::backend)? {
        return Err(Error::backend(format!(
            "grouped member {member} is outside bank of {groups}"
        )));
    }
    Ok(member..member + 1)
}

impl DeclareLogicalParameters for GroupedGatedProductSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.logical_group_parameters(
            0..usize::try_from(self.group_count()).map_err(Error::backend)?,
        )
    }
}

impl DeclareGroupedParameters for GroupedGatedProductSpec {
    fn logical_member_parameters(
        &self,
        member: usize,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.logical_group_parameters(checked_member_range(self.group_count(), member)?)
    }
}

impl GroupedGatedProductSpec {
    fn logical_group_parameters(
        &self,
        groups: std::ops::Range<usize>,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate()?;
        let group_count = i32::try_from(groups.len()).map_err(Error::backend)?;
        let intermediate = self.intermediate_dimensions();
        match self.layout() {
            GatedProductGroupLayout::Packed { gate_up, down } => {
                let doubled = intermediate
                    .checked_mul(2)
                    .ok_or_else(|| Error::backend("grouped gate/up width overflow"))?;
                let mut parameters = gate_up.logical_parameters_with_shape(&[
                    group_count,
                    doubled,
                    self.input_dimensions(),
                ])?;
                parameters.extend(down.logical_parameters_with_shape(&[
                    group_count,
                    self.output_dimensions(),
                    intermediate,
                ])?);
                Ok(parameters)
            }
            GatedProductGroupLayout::Independent(bank) => {
                let mut parameters = Vec::new();
                for group in &bank[groups] {
                    parameters.extend(
                        group.gate().logical_parameters_with_shape(&[
                            intermediate,
                            self.input_dimensions(),
                        ])?,
                    );
                    parameters.extend(
                        group.up().logical_parameters_with_shape(&[
                            intermediate,
                            self.input_dimensions(),
                        ])?,
                    );
                    parameters.extend(group.down().logical_parameters_with_shape(&[
                        self.output_dimensions(),
                        intermediate,
                    ])?);
                }
                Ok(parameters)
            }
        }
    }
}

impl DeclareLogicalParameters for GroupedRelu2Spec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.logical_parameters_with_groups(self.group_count())
    }
}

impl DeclareGroupedParameters for GroupedRelu2Spec {
    fn logical_member_parameters(
        &self,
        member: usize,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        checked_member_range(self.group_count(), member)?;
        self.logical_parameters_with_groups(1)
    }
}

impl GroupedRelu2Spec {
    fn logical_parameters_with_groups(
        &self,
        groups: i32,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate()?;
        let mut parameters = self.up().logical_parameters_with_shape(&[
            groups,
            self.intermediate_dimensions(),
            self.hidden_dimensions(),
        ])?;
        parameters.extend(self.down().logical_parameters_with_shape(&[
            groups,
            self.hidden_dimensions(),
            self.intermediate_dimensions(),
        ])?);
        Ok(parameters)
    }
}

/// Invalid source-free logical parameter declaration or sharing reference.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum LogicalParameterTopologyError {
    /// Duplicate identity, unresolved sharing target or non-authoritative target.
    #[error(transparent)]
    Identity(#[from] crate::ParameterTopologyError),
    /// A logical shape contains zero or its element count overflows.
    #[error("invalid logical geometry for parameter {0}")]
    InvalidGeometry(crate::ParameterId),
    /// Shared slots must describe exactly the same logical tensor.
    #[error(
        "parameter {alias} geometry {shape:?} differs from sharing target {destination} geometry {destination_shape:?}"
    )]
    IncompatibleSharing {
        /// Logical alias identity.
        alias: crate::ParameterId,
        /// Authoritative shared parameter.
        destination: crate::ParameterId,
        /// Declared alias geometry.
        shape: Vec<usize>,
        /// Declared authoritative geometry.
        destination_shape: Vec<usize>,
    },
}

/// Validates a complete logical declaration scope before storage projection or
/// backend allocation. Sharing references use typed identities; different neural
/// roles may share one exact geometry, as with tied embedding and output slots.
pub fn validate_logical_parameter_topology(
    declarations: &[LogicalParameterDeclaration<'_>],
) -> Result<(), LogicalParameterTopologyError> {
    validate_logical_parameter_scope_with_destinations(declarations, true)
}

/// Validates geometry and identities within a partial parameter scope.
/// Sharing targets outside this scope remain typed references. Preparation must
/// validate the complete topology before allocating backend values; a group or
/// component alone cannot establish that an external target exists.
pub fn validate_logical_parameter_scope(
    declarations: &[LogicalParameterDeclaration<'_>],
) -> Result<(), LogicalParameterTopologyError> {
    validate_logical_parameter_scope_with_destinations(declarations, false)
}

fn validate_logical_parameter_scope_with_destinations(
    declarations: &[LogicalParameterDeclaration<'_>],
    require_destinations: bool,
) -> Result<(), LogicalParameterTopologyError> {
    crate::validate_parameter_identities(
        declarations.iter().map(|declaration| {
            (
                &declaration.parameter.id,
                declaration.parameter.alias_of.as_ref(),
            )
        }),
        require_destinations,
    )?;
    let parameters = declarations
        .iter()
        .map(|declaration| (&declaration.parameter.id, declaration))
        .collect::<std::collections::BTreeMap<_, _>>();
    for declaration in declarations {
        if declaration.shape.contains(&0)
            || declaration
                .shape
                .iter()
                .try_fold(1usize, |count, &width| count.checked_mul(width))
                .is_none()
        {
            return Err(LogicalParameterTopologyError::InvalidGeometry(
                declaration.parameter.id.clone(),
            ));
        }
        if let Some(target) = declaration
            .parameter
            .alias_of
            .as_ref()
            .and_then(|id| parameters.get(id))
        {
            let destination = &target.parameter.id;
            if declaration.shape != target.shape {
                return Err(LogicalParameterTopologyError::IncompatibleSharing {
                    alias: declaration.parameter.id.clone(),
                    destination: destination.clone(),
                    shape: declaration.shape.clone(),
                    destination_shape: target.shape.clone(),
                });
            }
        }
    }
    Ok(())
}

impl DeclareLogicalParameters for crate::CausalDepthwiseConvolutionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate()?;
        let mut parameters = vec![LogicalParameterDeclaration::new(
            &self.weight,
            &[self.channels, 1, self.kernel_size],
            LogicalParameterRole::LinearWeight,
        )?];
        if let Some(bias) = &self.bias {
            parameters.push(LogicalParameterDeclaration::new(
                bias,
                &[self.channels],
                LogicalParameterRole::LinearBias,
            )?);
        }
        Ok(parameters)
    }
}

impl DeclareLogicalParameters for crate::GatedShortConvolutionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut declarations = self.input_projection.logical_parameters()?;
        declarations.extend(self.convolution.logical_parameters()?);
        declarations.extend(self.output_projection.logical_parameters()?);
        validate_logical_parameter_topology(&declarations).map_err(Error::backend_source)?;
        Ok(declarations)
    }
}

impl DeclareLogicalParameters for crate::LowRankProjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = Vec::new();
        if let Some(first) = &self.first {
            parameters.extend(first.logical_parameters()?);
        }
        parameters.extend(self.normalization.logical_parameters()?);
        parameters.extend(self.second.logical_parameters()?);
        validate_logical_parameter_topology(&parameters).map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

impl DeclareLogicalParameters for crate::HyperConnectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let width = self
            .streams
            .checked_mul(self.hidden_size)
            .ok_or_else(|| Error::backend("hyper-connection input width overflowed"))?;
        let mixing = self
            .streams
            .checked_add(2)
            .and_then(|n| n.checked_mul(self.streams))
            .ok_or_else(|| Error::backend("hyper-connection mixing width overflowed"))?;
        residual_mix_parameters(&self.function, &self.base, &self.scale, mixing, width, 3)
    }
}

impl DeclareLogicalParameters for crate::HyperHeadSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let width = self
            .streams
            .checked_mul(self.hidden_size)
            .ok_or_else(|| Error::backend("hyper-head input width overflowed"))?;
        residual_mix_parameters(
            &self.function,
            &self.base,
            &self.scale,
            self.streams,
            width,
            1,
        )
    }
}

fn residual_mix_parameters<'a>(
    function: &'a ParameterSpec,
    base: &'a ParameterSpec,
    scale: &'a ParameterSpec,
    outputs: i32,
    input: i32,
    scales: i32,
) -> Result<Vec<LogicalParameterDeclaration<'a>>, Error> {
    let parameters = vec![
        LogicalParameterDeclaration::new(
            function,
            &[outputs, input],
            LogicalParameterRole::ResidualMixWeight,
        )?,
        LogicalParameterDeclaration::new(base, &[outputs], LogicalParameterRole::ResidualMixBias)?,
        LogicalParameterDeclaration::new(scale, &[scales], LogicalParameterRole::ResidualMixScale)?,
    ];
    validate_logical_parameter_topology(&parameters).map_err(Error::backend_source)?;
    Ok(parameters)
}

impl DeclareLogicalParameters for crate::recurrent::LstmLayerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate()?;
        let mut parameters = self.input.logical_parameters()?;
        parameters.extend(self.recurrent.logical_parameters()?);
        Ok(parameters)
    }
}

impl DeclareLogicalParameters for crate::residual_streams::GatedResidualSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate_geometry()?;
        let mut parameters = self.normalization.logical_parameters()?;
        for projection in [&self.down, &self.up]
            .into_iter()
            .chain(self.injection.iter())
        {
            let mut projected = projection.logical_parameters()?;
            for parameter in &mut projected {
                parameter.role = LogicalParameterRole::ResidualMixWeight;
            }
            parameters.extend(projected);
        }
        validate_logical_parameter_topology(&parameters).map_err(Error::backend_source)?;
        Ok(parameters)
    }
}

impl DeclareLogicalParameters for crate::GroupedLinearSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.logical_parameters_with_groups(self.group_count())
    }
}

impl DeclareGroupedParameters for crate::GroupedLinearSpec {
    fn logical_member_parameters(
        &self,
        member: usize,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        checked_member_range(self.group_count(), member)?;
        self.logical_parameters_with_groups(1)
    }
}

impl crate::GroupedLinearSpec {
    fn logical_parameters_with_groups(
        &self,
        groups: i32,
    ) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.validate()?;
        if groups == 0 {
            return Ok(Vec::new());
        }
        self.projection().logical_parameters_with_shape(&[
            groups,
            self.output_dimensions(),
            self.input_dimensions(),
        ])
    }
}
