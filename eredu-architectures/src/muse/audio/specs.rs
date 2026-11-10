//! Typed waveform construction, parameter ownership, and encoding projections.

use eredu_nn::{
    CausalDepthwiseConvolutionSpec, ConvolutionActivation, DeclareLogicalParameters, Error,
    LinearSpec, LogicalParameterDeclaration, NormalizationConstructionSpec, ParameterSpec,
};
use eredu_runtime::{MemberSharding, ParameterGroupSpec, ParameterRole};

use super::{AudioPolicy, AudioUnitKind};

#[derive(Debug, Clone)]
pub(crate) struct PatchSpec {
    pub projection: LinearSpec,
    pub norm: NormalizationConstructionSpec,
    pub patch_size: i32,
    pub input_channels: i32,
    pub input_precision: crate::muse::config::MediaInputPrecision,
}
#[derive(Debug, Clone)]
pub(crate) struct ConvSpec {
    pub convolution: CausalDepthwiseConvolutionSpec,
    pub norm: NormalizationConstructionSpec,
    pub network: eredu_nn::UnaryFeedForwardSpec,
}
#[derive(Debug, Clone)]
pub(crate) struct ProjectionSpec {
    pub norm: NormalizationConstructionSpec,
    pub projection: LinearSpec,
    pub post_norm: NormalizationConstructionSpec,
}
impl DeclareLogicalParameters for PatchSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.patch_size <= 0
            || self.input_channels <= 0
            || self.input_channels.checked_mul(self.patch_size) != Some(self.projection.input)
            || self.norm.dimensions != self.projection.output
        {
            return Err(Error::backend("incompatible Muse audio patch geometry"));
        }
        let mut parameters = self.projection.logical_parameters()?;
        parameters.extend(self.norm.logical_parameters()?);
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for ConvSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let width = self.convolution.channels;
        if self.norm.dimensions != width
            || self.network.up.input != width
            || self.network.down.output != width
            || self.network.activation != eredu_nn::UnaryActivation::Gelu
            || self.network.up.bias.is_none()
            || self.network.down.bias.is_none()
            || self.convolution.bias.is_none()
        {
            return Err(Error::backend(
                "incompatible Muse audio residual geometry or GELU policy",
            ));
        }
        let mut parameters = self.convolution.logical_parameters()?;
        parameters.extend(self.norm.logical_parameters()?);
        parameters.extend(self.network.logical_parameters()?);
        Ok(parameters)
    }
}
impl DeclareLogicalParameters for ProjectionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if self.norm.dimensions != self.projection.input
            || self.post_norm.dimensions != self.projection.output
        {
            return Err(Error::backend(
                "incompatible Muse audio projection normalization geometry",
            ));
        }
        let mut parameters = self.norm.logical_parameters()?;
        parameters.extend(self.projection.logical_parameters()?);
        parameters.extend(self.post_norm.logical_parameters()?);
        Ok(parameters)
    }
}
#[derive(Debug, Clone)]
pub(crate) enum UnitSpec {
    Patch(PatchSpec),
    Convolution(ConvSpec),
    Projection(ProjectionSpec),
}

fn linear(
    config: &AudioPolicy,
    name: &str,
    input: i32,
    output: i32,
    arithmetic: eredu_nn::LinearArithmetic,
) -> Result<LinearSpec, Error> {
    let weight = format!("{name}.weight");
    Ok(LinearSpec {
        input,
        output,
        weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
        bias: Some(ParameterSpec::trainable(format!("{name}.bias")).map_err(Error::backend)?),
        format: crate::linear_format::standard_linear_format(
            &weight,
            config.linear_format_for(&weight),
        )?
        .with_arithmetic(arithmetic)?,
    })
}
fn norm(
    config: &AudioPolicy,
    name: &str,
    width: i32,
) -> Result<NormalizationConstructionSpec, Error> {
    NormalizationConstructionSpec::learned(
        width,
        config.rms_norm_eps,
        ParameterSpec::trainable(format!("{name}.weight")).map_err(Error::backend)?,
    )
    .with_groups(1)
}
impl UnitSpec {
    fn new(config: &AudioPolicy, kind: AudioUnitKind) -> Result<Self, Error> {
        use eredu_nn::LinearArithmetic::{ChannelMajorAffine, RowMajorAffine};
        let root = kind.path();
        Ok(match kind {
            AudioUnitKind::Patch { stage } => {
                let patch_size = config.patch_sizes[stage];
                let input_channels = if stage == 0 {
                    1
                } else {
                    config.hidden_sizes[stage - 1]
                };
                Self::Patch(PatchSpec {
                    projection: linear(
                        config,
                        &format!("{root}.proj"),
                        input_channels
                            .checked_mul(patch_size)
                            .ok_or_else(|| Error::backend("Muse audio patch width exceeds i32"))?,
                        config.hidden_sizes[stage],
                        RowMajorAffine,
                    )?,
                    norm: norm(config, &format!("{root}.norm"), config.hidden_sizes[stage])?,
                    patch_size,
                    input_channels,
                    input_precision: if stage == 0 {
                        config.input_precision
                    } else {
                        crate::muse::config::MediaInputPrecision::Preserve
                    },
                })
            }
            AudioUnitKind::Convolution { stage, .. } => {
                let width = config.hidden_sizes[stage];
                let intermediate = config.intermediate_size(stage)?;
                Self::Convolution(ConvSpec {
                    convolution: CausalDepthwiseConvolutionSpec {
                        arithmetic: eredu_nn::ConvolutionArithmetic::FourLaneFloat32Affine,
                        dilation: 1,
                        channels: width,
                        kernel_size: config.conv_kernel_size,
                        weight: ParameterSpec::trainable(format!("{root}.conv.weight"))
                            .map_err(Error::backend)?,
                        bias: Some(
                            ParameterSpec::trainable(format!("{root}.conv.bias"))
                                .map_err(Error::backend)?,
                        ),
                        activation: ConvolutionActivation::Identity,
                    },
                    norm: NormalizationConstructionSpec::learned(
                        width,
                        config.rms_norm_eps,
                        ParameterSpec::trainable(format!("{root}.norm.weight"))
                            .map_err(Error::backend)?,
                    )
                    .with_reduction(eredu_nn::NormalizationReduction::ChannelMajor)?,
                    // Four ordered FP32 dot sums precede product rounding and bias.
                    network: eredu_nn::UnaryFeedForwardSpec {
                        up: linear(
                            config,
                            &format!("{root}.mlp.fc1"),
                            width,
                            intermediate,
                            ChannelMajorAffine,
                        )?,
                        down: linear(
                            config,
                            &format!("{root}.mlp.fc2"),
                            intermediate,
                            width,
                            RowMajorAffine,
                        )?,
                        activation: eredu_nn::UnaryActivation::Gelu,
                    },
                })
            }
            AudioUnitKind::Projection => {
                let width = *config.hidden_sizes.last().expect("validated audio stages");
                Self::Projection(ProjectionSpec {
                    norm: norm(config, "model.audio_tower.norm", width)?,
                    projection: linear(
                        config,
                        "model.audio_tower.proj",
                        width,
                        config.out_hidden_size,
                        RowMajorAffine,
                    )?,
                    post_norm: norm(
                        config,
                        "model.audio_tower.post_proj_norm",
                        config.out_hidden_size,
                    )?,
                })
            }
        })
    }
    pub fn projections(&self) -> Vec<&LinearSpec> {
        match self {
            Self::Patch(spec) => vec![&spec.projection],
            Self::Convolution(spec) => spec.network.projections().to_vec(),
            Self::Projection(spec) => vec![&spec.projection],
        }
    }
    pub fn parameter_groups(&self, kind: AudioUnitKind) -> Result<Vec<ParameterGroupSpec>, Error> {
        Ok(vec![eredu_runtime::logical_parameter_group(
            kind.path(),
            ParameterRole::Replicated,
            None,
            self,
            |_| MemberSharding::Replicated,
        )
        .map_err(Error::backend_source)?])
    }
}
impl DeclareLogicalParameters for UnitSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Patch(spec) => spec.logical_parameters(),
            Self::Convolution(spec) => spec.logical_parameters(),
            Self::Projection(spec) => spec.logical_parameters(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Declarations {
    pub units: Vec<(AudioUnitKind, UnitSpec)>,
}
impl Declarations {
    pub(super) fn new(config: &AudioPolicy) -> Result<Self, Error> {
        let units = config
            .execution_units()?
            .into_iter()
            .map(|kind| Ok((kind, UnitSpec::new(config, kind)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        let declarations = Self { units };
        eredu_nn::validate_logical_parameter_topology(&declarations.logical_parameters()?)
            .map_err(Error::backend_source)?;
        declarations.validate_format_references(config)?;
        Ok(declarations)
    }
    pub fn unit(&self, kind: AudioUnitKind) -> Result<&UnitSpec, Error> {
        self.units
            .iter()
            .find(|(unit, _)| *unit == kind)
            .map(|(_, spec)| spec)
            .ok_or_else(|| Error::backend("Muse audio unit is outside its configured pyramid"))
    }
    pub fn project_formats(&mut self, policy: &AudioPolicy) -> Result<(), Error> {
        for (_, unit) in &mut self.units {
            let projections = match unit {
                UnitSpec::Patch(s) => vec![&mut s.projection],
                UnitSpec::Convolution(s) => vec![&mut s.network.up, &mut s.network.down],
                UnitSpec::Projection(s) => vec![&mut s.projection],
            };
            for spec in projections {
                let mut format = crate::linear_format::standard_linear_format(
                    spec.weight.id.as_str(),
                    policy.linear_format_for(spec.weight.id.as_str()),
                )?
                .with_arithmetic(spec.format.arithmetic())?;
                if spec.format.exact_zero_inputs() {
                    format = format.with_exact_zero_inputs();
                }
                spec.format = format;
            }
        }
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        self.validate_format_references(policy)
    }
    fn validate_format_references(&self, policy: &AudioPolicy) -> Result<(), Error> {
        crate::format_references::validate_matrices(
            self.logical_parameters()?,
            policy
                .linear_formats
                .iter()
                .map(|(name, format)| (name.as_str(), Some(*format))),
        )
    }
    pub fn owners(&self) -> Result<std::collections::BTreeMap<String, usize>, Error> {
        let mut owners = std::collections::BTreeMap::new();
        for (index, (kind, spec)) in self.units.iter().enumerate() {
            for group in spec.parameter_groups(*kind)? {
                for member in group.members() {
                    if owners.insert(member.target().to_owned(), index).is_some() {
                        return Err(Error::backend(format!(
                            "conflicting Muse audio owner for {}",
                            member.target()
                        )));
                    }
                }
            }
        }
        Ok(owners)
    }
}
impl DeclareLogicalParameters for Declarations {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = Vec::new();
        for (_, unit) in &self.units {
            parameters.extend(unit.logical_parameters()?);
        }
        Ok(parameters)
    }
}
