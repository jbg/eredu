//! Typed EAR neural parameter owners and canonical convolution storage layouts.
use crate::{
    ear_decoder::{invalid, DecoderConfig, DecoderUnitSpec},
    ear_encoder::{EncoderConfig, EncoderUnitSpec},
    streaming::{Conv1dSpec, ConvTranspose1dSpec, TransposeTail},
    Error,
};
use eredu_nn::{
    DeclareLogicalParameters, LayerNormalizationArithmetic, LayerNormalizationConstructionSpec,
    LogicalParameterDeclaration, LogicalParameterRole, PadMode, Parameter, ParameterSpec, Tensor,
};
use eredu_runtime::{logical_parameter_members, MemberSharding, ParameterGroupSpec, ParameterRole};
fn parameter(name: String, group: &str) -> Result<ParameterSpec, Error> {
    let mut p =
        ParameterSpec::trainable(name).map_err(|_| invalid("invalid EAR parameter identity"))?;
    p.trainable = false;
    p.group = Some(group.into());
    Ok(p)
}
fn validate(spec: &impl DeclareLogicalParameters) -> Result<(), Error> {
    eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
        .map_err(|_| invalid("invalid EAR parameter topology"))
}
/// Canonical direction of a learned one-dimensional convolution kernel.
#[derive(Debug, Clone, Copy)]
pub enum KernelLayout {
    /// Output, input-per-group, kernel taps.
    Forward,
    /// Input, output-per-group, kernel taps.
    Transposed,
}
/// Frozen convolution slots paired with the portable convolution geometry.
#[derive(Debug, Clone)]
pub struct ConvolutionSpec {
    pub(crate) geometry: Conv1dSpec,
    weight: ParameterSpec,
    bias: Option<ParameterSpec>,
}
impl ConvolutionSpec {
    fn new(root: &str, group: &str, geometry: Conv1dSpec) -> Result<Self, Error> {
        let g = geometry;
        if g.input_channels <= 0
            || g.output_channels <= 0
            || g.groups <= 0
            || g.input_channels % g.groups != 0
            || g.output_channels % g.groups != 0
            || g.kernel_size <= 0
            || g.stride <= 0
            || g.dilation <= 0
        {
            return Err(invalid("invalid EAR convolution geometry"));
        }
        let s = Self {
            geometry,
            weight: parameter(format!("{root}.weight"), group)?,
            bias: geometry
                .bias
                .then(|| parameter(format!("{root}.bias"), group))
                .transpose()?,
        };
        validate(&s)?;
        Ok(s)
    }
    fn pointwise(
        root: &str,
        group: &str,
        input: i32,
        output: i32,
        bias: bool,
    ) -> Result<Self, Error> {
        Self::new(
            root,
            group,
            Conv1dSpec {
                input_channels: input,
                output_channels: output,
                kernel_size: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                bias,
                pad_mode: PadMode::Constant,
            },
        )
    }
    /// Portable input/output, grouping and stride semantics.
    pub fn geometry(&self) -> Conv1dSpec {
        self.geometry
    }
    /// Canonical stored axis orientation, independent of native kernel layout.
    pub fn layout(&self) -> KernelLayout {
        KernelLayout::Forward
    }
    /// Declared kernel identity and ownership metadata.
    pub fn weight(&self) -> &ParameterSpec {
        &self.weight
    }
    /// Declared output bias, when present.
    pub fn bias(&self) -> Option<&ParameterSpec> {
        self.bias.as_ref()
    }
    /// Exact canonical kernel dimensions before any backend layout transform.
    pub fn weight_shape(&self) -> [i32; 3] {
        let g = self.geometry;
        [
            g.output_channels,
            g.input_channels / g.groups,
            g.kernel_size,
        ]
    }

    pub(crate) fn weight_parameter<T: Tensor>(
        &self,
        context: &T::Context,
    ) -> Result<Parameter<T>, Error> {
        Ok(Parameter::unloaded(
            self.weight.clone(),
            &self.weight_shape(),
            context,
        )?)
    }
    pub(crate) fn bias_parameter<T: Tensor>(
        &self,
        context: &T::Context,
    ) -> Result<Option<Parameter<T>>, Error> {
        self.bias
            .as_ref()
            .map(|p| {
                Ok(Parameter::unloaded(
                    p.clone(),
                    &[self.geometry.output_channels],
                    context,
                )?)
            })
            .transpose()
    }
}
impl DeclareLogicalParameters for ConvolutionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        let mut p = vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.weight_shape(),
            LogicalParameterRole::LinearWeight,
        )?];
        if let Some(b) = &self.bias {
            p.push(LogicalParameterDeclaration::new(
                b,
                &[self.geometry.output_channels],
                LogicalParameterRole::LinearBias,
            )?);
        }
        Ok(p)
    }
}
/// Frozen IOK transposed kernel paired with its output-stride contract.
#[derive(Debug, Clone)]
pub struct TransposeSpec {
    pub(crate) geometry: ConvTranspose1dSpec,
    weight: ParameterSpec,
}
impl TransposeSpec {
    fn new(root: &str, group: &str, geometry: ConvTranspose1dSpec) -> Result<Self, Error> {
        let g = geometry;
        if g.input_channels <= 0
            || g.output_channels <= 0
            || g.groups <= 0
            || g.input_channels % g.groups != 0
            || g.output_channels % g.groups != 0
            || g.stride <= 0
            || g.kernel_size != g.stride
            || g.bias
        {
            return Err(invalid(
                "EAR transposed convolution requires a nonoverlapping biasless kernel",
            ));
        }
        let spec = Self {
            geometry,
            weight: parameter(format!("{root}.weight"), group)?,
        };
        validate(&spec)?;
        Ok(spec)
    }
    /// Portable transposed convolution and finite-tail geometry.
    pub fn geometry(&self) -> ConvTranspose1dSpec {
        self.geometry
    }
    /// Canonical IOK axis orientation.
    pub fn layout(&self) -> KernelLayout {
        KernelLayout::Transposed
    }
    /// Declared kernel identity and ownership metadata.
    pub fn weight(&self) -> &ParameterSpec {
        &self.weight
    }
    /// Input, output-per-group and kernel-tap dimensions.
    pub fn weight_shape(&self) -> [i32; 3] {
        let g = self.geometry;
        [
            g.input_channels,
            g.output_channels / g.groups,
            g.kernel_size,
        ]
    }
    pub(crate) fn weight_parameter<T: Tensor>(
        &self,
        context: &T::Context,
    ) -> Result<Parameter<T>, Error> {
        Ok(Parameter::unloaded(
            self.weight.clone(),
            &self.weight_shape(),
            context,
        )?)
    }
}
impl DeclareLogicalParameters for TransposeSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        Ok(vec![LogicalParameterDeclaration::new(
            &self.weight,
            &self.weight_shape(),
            LogicalParameterRole::LinearWeight,
        )?])
    }
}
/// One causal depthwise convolution, affine layer norm, and exact-GELU MLP.
#[derive(Debug, Clone)]
pub struct ConvNextSpec {
    pub(crate) depthwise: ConvolutionSpec,
    pub(crate) norm: LayerNormalizationConstructionSpec,
    pub(crate) expand: ConvolutionSpec,
    pub(crate) project: ConvolutionSpec,
}
impl ConvNextSpec {
    pub(crate) fn causal_geometry(width: i32, kernel: i32) -> Conv1dSpec {
        Conv1dSpec {
            input_channels: width,
            output_channels: width,
            kernel_size: kernel,
            stride: 1,
            dilation: 1,
            groups: width,
            bias: true,
            pad_mode: PadMode::Constant,
        }
    }
    fn new(root: &str, width: i32, intermediate: i32, kernel: i32) -> Result<Self, Error> {
        let s = Self {
            depthwise: ConvolutionSpec::new(
                &format!("{root}.dwconv"),
                root,
                Self::causal_geometry(width, kernel),
            )?,
            norm: LayerNormalizationConstructionSpec {
                dimensions: width,
                epsilon: 1e-6,
                weight: Some(parameter(format!("{root}.norm.weight"), root)?),
                bias: Some(parameter(format!("{root}.norm.bias"), root)?),
                arithmetic: LayerNormalizationArithmetic::Native,
            },
            expand: ConvolutionSpec::pointwise(
                &format!("{root}.pwconv1"),
                root,
                width,
                intermediate,
                true,
            )?,
            project: ConvolutionSpec::pointwise(
                &format!("{root}.pwconv2"),
                root,
                intermediate,
                width,
                true,
            )?,
        };
        validate(&s)?;
        Ok(s)
    }
    fn local(&mut self, width: i32) {
        self.expand.geometry.output_channels = width;
        self.project.geometry.input_channels = width;
    }
    /// Canonical depthwise slots.
    pub fn depthwise(&self) -> &ConvolutionSpec {
        &self.depthwise
    }
    /// Affine normalization declarations and arithmetic.
    pub fn normalization(&self) -> &LayerNormalizationConstructionSpec {
        &self.norm
    }
    /// Intermediate expansion slots.
    pub fn expand(&self) -> &ConvolutionSpec {
        &self.expand
    }
    /// Residual projection slots.
    pub fn project(&self) -> &ConvolutionSpec {
        &self.project
    }
    fn members(&self) -> Result<Vec<eredu_runtime::ParameterMemberSpec>, Error> {
        Ok(logical_parameter_members(self, |p| {
            if p.parameter.id == self.expand.weight.id
                || self
                    .expand
                    .bias
                    .as_ref()
                    .is_some_and(|b| p.parameter.id == b.id)
            {
                MemberSharding::Partitioned { axis: 0 }
            } else if p.parameter.id == self.project.weight.id {
                MemberSharding::Partitioned { axis: 1 }
            } else {
                MemberSharding::Replicated
            }
        })?)
    }
}
impl DeclareLogicalParameters for ConvNextSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        let mut p = self.depthwise.logical_parameters()?;
        p.extend(self.norm.logical_parameters()?);
        p.extend(self.expand.logical_parameters()?);
        p.extend(self.project.logical_parameters()?);
        Ok(p)
    }
}
/// Parameter mechanism of one encoder invocation.
#[derive(Debug, Clone)]
pub enum EncoderParameters {
    /// Initial spectral pointwise projection.
    Input(ConvolutionSpec),
    /// Causal ConvNeXt residual block.
    Block(ConvNextSpec),
    /// Grouped strided downsampling.
    Downsample(ConvolutionSpec),
}
/// Parameter mechanism of one decoder invocation.
#[derive(Debug, Clone)]
pub enum DecoderParameters {
    /// Nonoverlapping transposed convolution.
    Upsample(TransposeSpec),
    /// Causal ConvNeXt residual block.
    Block(ConvNextSpec),
    /// Final spectral pointwise projection.
    Output(ConvolutionSpec),
}
/// Complete ownership for one independently materializable encoder invocation.
#[derive(Debug, Clone)]
pub struct EncoderParameterSpec {
    pub(crate) unit: EncoderUnitSpec,
    pub(crate) prefix: String,
    pub(crate) local: Option<crate::ear_encoder_layered::EncoderUnitLocalDimensions>,
    pub(crate) parameters: EncoderParameters,
}
/// Complete ownership for one independently materializable decoder invocation.
#[derive(Debug, Clone)]
pub struct DecoderParameterSpec {
    pub(crate) unit: DecoderUnitSpec,
    pub(crate) prefix: String,
    pub(crate) local: Option<crate::ear_decoder_layered::DecoderUnitLocalDimensions>,
    pub(crate) parameters: DecoderParameters,
}
fn local_range(start: i32, width: i32, domain: i32) -> Result<(), Error> {
    if start < 0 || width <= 0 || start.checked_add(width).is_none_or(|end| end > domain) {
        return Err(invalid("invalid EAR parameter partition"));
    }
    Ok(())
}
impl EncoderParameterSpec {
    /// Declares one global owner from validated encoder execution geometry.
    pub fn new(unit: EncoderUnitSpec, prefix: &str) -> Result<Self, Error> {
        unit.validate()?;
        if prefix.is_empty() {
            return Err(invalid("empty EAR encoder parameter prefix"));
        }
        let root = format!("{prefix}.layers.{}", unit.layer());
        let parameters = match unit {
            EncoderUnitSpec::Input { input, output, .. } => EncoderParameters::Input(
                ConvolutionSpec::pointwise(&root, &root, input, output, false)?,
            ),
            EncoderUnitSpec::Block {
                width,
                intermediate,
                kernel,
                ..
            } => EncoderParameters::Block(ConvNextSpec::new(&root, width, intermediate, kernel)?),
            EncoderUnitSpec::Downsample {
                input,
                output,
                rate,
                groups,
                ..
            } => EncoderParameters::Downsample(ConvolutionSpec::new(
                &root,
                &root,
                Conv1dSpec {
                    input_channels: input,
                    output_channels: output,
                    kernel_size: rate,
                    stride: rate,
                    dilation: 1,
                    groups,
                    bias: false,
                    pad_mode: PadMode::Constant,
                },
            )?),
        };
        let s = Self {
            unit,
            prefix: prefix.into(),
            local: None,
            parameters,
        };
        validate(&s)?;
        Ok(s)
    }
    /// Selects coherent output or intermediate channels; grouped outputs retain all groups.
    pub fn local(
        mut self,
        local: crate::ear_encoder_layered::EncoderUnitLocalDimensions,
    ) -> Result<Self, Error> {
        let domain = match self.unit {
            EncoderUnitSpec::Input { output, .. } => output,
            EncoderUnitSpec::Block { intermediate, .. } => intermediate,
            EncoderUnitSpec::Downsample { output, groups, .. } => output / groups,
        };
        local_range(local.start, local.width, domain)?;
        match &mut self.parameters {
            EncoderParameters::Input(c) => c.geometry.output_channels = local.width,
            EncoderParameters::Downsample(c) => {
                c.geometry.output_channels = local.width * c.geometry.groups
            }
            EncoderParameters::Block(b) => b.local(local.width),
        }
        self.local = Some(local);
        validate(&self)?;
        Ok(self)
    }
    /// Global invocation geometry, independent of local parameter shape.
    pub fn unit(&self) -> EncoderUnitSpec {
        self.unit
    }
    /// Typed parameter mechanisms of this invocation.
    pub fn parameters(&self) -> &EncoderParameters {
        &self.parameters
    }
    pub(crate) fn group(&self) -> Result<ParameterGroupSpec, Error> {
        let (role, size, members) = match &self.parameters {
            EncoderParameters::Input(c) => (
                ParameterRole::Channels,
                c.geometry.output_channels,
                logical_parameter_members(c, |_| MemberSharding::Partitioned { axis: 0 })?,
            ),
            EncoderParameters::Downsample(c) => {
                let g = c.geometry;
                let width = g.output_channels / g.groups;
                (
                    ParameterRole::Channels,
                    width,
                    logical_parameter_members(c, |_| MemberSharding::PartitionedSegments {
                        axis: 0,
                        segments: (0..g.groups)
                            .map(|n| (n * width) as usize..((n + 1) * width) as usize)
                            .collect(),
                    })?,
                )
            }
            EncoderParameters::Block(b) => (
                ParameterRole::FeedForwardIntermediate,
                b.expand.geometry.output_channels,
                b.members()?,
            ),
        };
        Ok(ParameterGroupSpec::partitioned(
            format!("{}.layers.{}", self.prefix, self.unit.layer()),
            role,
            size as usize,
            members,
        )?)
    }
}
impl DecoderParameterSpec {
    /// Declares one global owner from validated decoder execution geometry.
    pub fn new(unit: DecoderUnitSpec, prefix: &str) -> Result<Self, Error> {
        unit.validate()?;
        if prefix.is_empty() {
            return Err(invalid("empty EAR decoder parameter prefix"));
        }
        let root = format!("{prefix}.layers.{}", unit.layer());
        let parameters = match unit {
            DecoderUnitSpec::Output { input, output, .. } => DecoderParameters::Output(
                ConvolutionSpec::pointwise(&root, &root, input, output, false)?,
            ),
            DecoderUnitSpec::Block {
                width,
                intermediate,
                kernel,
                ..
            } => DecoderParameters::Block(ConvNextSpec::new(&root, width, intermediate, kernel)?),
            DecoderUnitSpec::Upsample {
                input,
                output,
                rate,
                groups,
                ..
            } => DecoderParameters::Upsample(TransposeSpec::new(
                &root,
                &root,
                ConvTranspose1dSpec {
                    input_channels: input,
                    output_channels: output,
                    kernel_size: rate,
                    stride: rate,
                    groups,
                    bias: false,
                    tail: TransposeTail::Emit,
                },
            )?),
        };
        let s = Self {
            unit,
            prefix: prefix.into(),
            local: None,
            parameters,
        };
        validate(&s)?;
        Ok(s)
    }
    /// Selects coherent intermediate channels or transposed-kernel output columns.
    pub fn local(
        mut self,
        local: crate::ear_decoder_layered::DecoderUnitLocalDimensions,
    ) -> Result<Self, Error> {
        let domain = match self.unit {
            DecoderUnitSpec::Output { output, .. } => output,
            DecoderUnitSpec::Block { intermediate, .. } => intermediate,
            DecoderUnitSpec::Upsample { output, groups, .. } => output / groups,
        };
        local_range(local.start, local.width, domain)?;
        match &mut self.parameters {
            DecoderParameters::Output(c) => c.geometry.output_channels = local.width,
            DecoderParameters::Upsample(c) => {
                c.geometry.output_channels = local.width * c.geometry.groups
            }
            DecoderParameters::Block(b) => b.local(local.width),
        }
        self.local = Some(local);
        validate(&self)?;
        Ok(self)
    }
    /// Global invocation geometry, independent of local parameter shape.
    pub fn unit(&self) -> DecoderUnitSpec {
        self.unit
    }
    /// Typed parameter mechanisms of this invocation.
    pub fn parameters(&self) -> &DecoderParameters {
        &self.parameters
    }
    pub(crate) fn group(&self) -> Result<ParameterGroupSpec, Error> {
        let (role, size, members) = match &self.parameters {
            DecoderParameters::Output(c) => (
                ParameterRole::Channels,
                c.geometry.output_channels,
                logical_parameter_members(c, |_| MemberSharding::Partitioned { axis: 0 })?,
            ),
            DecoderParameters::Upsample(c) => (
                ParameterRole::Channels,
                c.geometry.output_channels / c.geometry.groups,
                logical_parameter_members(c, |_| MemberSharding::Partitioned { axis: 1 })?,
            ),
            DecoderParameters::Block(b) => (
                ParameterRole::FeedForwardIntermediate,
                b.expand.geometry.output_channels,
                b.members()?,
            ),
        };
        Ok(ParameterGroupSpec::partitioned(
            format!("{}.layers.{}", self.prefix, self.unit.layer()),
            role,
            size as usize,
            members,
        )?)
    }
}
impl DeclareLogicalParameters for EncoderParameterSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        match &self.parameters {
            EncoderParameters::Input(c) | EncoderParameters::Downsample(c) => {
                c.logical_parameters()
            }
            EncoderParameters::Block(b) => b.logical_parameters(),
        }
    }
}
impl DeclareLogicalParameters for DecoderParameterSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        match &self.parameters {
            DecoderParameters::Output(c) => c.logical_parameters(),
            DecoderParameters::Upsample(c) => c.logical_parameters(),
            DecoderParameters::Block(b) => b.logical_parameters(),
        }
    }
}
/// Complete finite encoder parameter ownership in execution order.
#[derive(Debug, Clone)]
pub struct EncoderSpec {
    pub(crate) units: Vec<EncoderParameterSpec>,
}
impl EncoderSpec {
    /// Validates the complete encoder parameter topology before allocation.
    pub fn new(config: &EncoderConfig, prefix: &str) -> Result<Self, Error> {
        let s = Self {
            units: config
                .unit_specs()?
                .into_iter()
                .map(|u| EncoderParameterSpec::new(u, prefix))
                .collect::<Result<_, _>>()?,
        };
        validate(&s)?;
        Ok(s)
    }
    /// Independently materializable owners in encoder execution order.
    pub fn units(&self) -> &[EncoderParameterSpec] {
        &self.units
    }
    /// Projects exact ownership and placement semantics from these declarations.
    pub fn parameter_description(
        &self,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Error> {
        crate::ear_encoder_layered::declared_parameters(self)
    }
}
/// Complete streaming decoder parameter ownership, separate from mutable carry.
#[derive(Debug, Clone)]
pub struct DecoderSpec {
    pub(crate) units: Vec<DecoderParameterSpec>,
}
impl DecoderSpec {
    /// Validates the complete decoder parameter topology before allocation.
    pub fn new(config: &DecoderConfig, prefix: &str) -> Result<Self, Error> {
        let s = Self {
            units: config
                .unit_specs()?
                .into_iter()
                .map(|u| DecoderParameterSpec::new(u, prefix))
                .collect::<Result<_, _>>()?,
        };
        validate(&s)?;
        Ok(s)
    }
    /// Independently materializable owners in decoder execution order.
    pub fn units(&self) -> &[DecoderParameterSpec] {
        &self.units
    }
    /// Projects exact ownership and placement semantics from these declarations.
    pub fn parameter_description(
        &self,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Error> {
        crate::ear_decoder_layered::declared_parameters(self)
    }
}
impl DeclareLogicalParameters for EncoderSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        self.units[..].logical_parameters()
    }
}
impl DeclareLogicalParameters for DecoderSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, eredu_nn::Error> {
        self.units[..].logical_parameters()
    }
}
