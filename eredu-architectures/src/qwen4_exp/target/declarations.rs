//! Source-free parameter assembly and logical invocations, independent of state bounds.
use super::*;
use crate::qwen4_exp::{indexed::IndexedParameters, ple::LexicalParameters};
use eredu_nn::residual_streams::GatedResidualSpec;

/// Logical invocation identity; its ordinal is independent of checkpoint layer identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetInvocation {
    Lexical { layer: usize },
    Decoder { layer: usize, kind: LayerKind },
}
impl TargetInvocation {
    pub(crate) fn schedule(config: &Config) -> impl Iterator<Item = Self> + '_ {
        config.layers.iter().enumerate().flat_map(|(layer, &kind)| {
            config
                .ngram
                .layers
                .contains(&layer)
                .then_some(Self::Lexical { layer })
                .into_iter()
                .chain(std::iter::once(Self::Decoder { layer, kind }))
        })
    }
    pub(crate) fn layer(self) -> usize {
        match self {
            Self::Lexical { layer } | Self::Decoder { layer, .. } => layer,
        }
    }
    pub(crate) fn parameter_scope(self) -> super::super::checkpoint::recipes::ParameterScope {
        use super::super::checkpoint::recipes::ParameterScope;
        match self {
            Self::Lexical { layer } => ParameterScope::Lexical(layer),
            Self::Decoder { layer, .. } => ParameterScope::Target(layer),
        }
    }
}

/// Mixer parameters become an invocation only after explicit state/workspace admission.
#[derive(Debug, Clone)]
pub(crate) enum MixerParameters {
    Recurrent(RecurrentSublayerSpec),
    Indexed {
        residual: GatedResidualSpec,
        parameters: IndexedParameters,
    },
}
impl MixerParameters {
    pub(crate) fn from_config(
        config: &Config,
        kind: LayerKind,
        root: &str,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        Ok(match kind {
            LayerKind::Recurrent => {
                Self::Recurrent(RecurrentSublayerSpec::from_config(config, root, format)?)
            }
            LayerKind::Indexed => Self::Indexed {
                residual: super::super::residual::mixer_spec(
                    config,
                    &format!("{root}.attn_hyper_connection"),
                    true,
                    &mut format,
                )?,
                parameters: IndexedParameters::from_config(config, root, format)
                    .map_err(Error::backend_source)?,
            },
        })
    }
    pub(crate) fn into_spec(
        self,
        config: &Config,
        tile_blocks: i32,
        selection_workspace: u64,
        limits: QsaExecutionLimits,
        element: TensorElementType,
    ) -> Result<MixerSpec, Error> {
        Ok(match self {
            Self::Recurrent(spec) => MixerSpec::Recurrent(spec),
            Self::Indexed {
                residual,
                parameters,
            } => MixerSpec::Indexed(
                IndexedSublayerSpec::from_parameters(
                    config,
                    residual,
                    parameters,
                    tile_blocks,
                    selection_workspace,
                    limits,
                    element,
                    TensorElementType::F32,
                )
                .map_err(Error::backend_source)?,
            ),
        })
    }
}
impl DeclareLogicalParameters for MixerParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Recurrent(spec) => spec.logical_parameters(),
            Self::Indexed {
                residual,
                parameters,
            } => {
                let mut slots = residual.logical_parameters()?;
                slots.extend(parameters.logical_parameters()?);
                Ok(slots)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum UnitParameters {
    Lexical {
        layer: usize,
        parameters: LexicalParameters,
    },
    Decoder {
        layer: usize,
        mixer: MixerParameters,
        feed_forward: FeedForwardSublayerSpec,
    },
}
impl UnitParameters {
    pub(crate) fn expert_bank(&self) -> Option<&GroupedGatedProductSpec> {
        match self {
            Self::Lexical { .. } => None,
            Self::Decoder { feed_forward, .. } => {
                Some(&feed_forward.feed_forward.parameters.routed.experts)
            }
        }
    }
}
impl DeclareLogicalParameters for UnitParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut slots = match self {
            Self::Lexical { parameters, .. } => parameters.logical_parameters()?,
            Self::Decoder { mixer, .. } => mixer.logical_parameters()?,
        };
        if let Self::Decoder { feed_forward, .. } = self {
            slots.extend(feed_forward.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_topology(&slots).map_err(Error::backend_source)?;
        Ok(slots)
    }
}

/// One assembly supplies both cold source requirements and bounded executable construction.
#[derive(Debug, Clone)]
pub(crate) struct TargetParameters {
    pub(crate) static_parameters: StaticParameters,
    pub(crate) units: Vec<UnitParameters>,
}
impl TargetParameters {
    pub(crate) fn decoder(
        &self,
        layer: usize,
    ) -> Option<(&MixerParameters, &FeedForwardSublayerSpec)> {
        self.units.iter().find_map(|unit| match unit {
            UnitParameters::Decoder {
                layer: owner,
                mixer,
                feed_forward,
            } if *owner == layer => Some((mixer, feed_forward)),
            _ => None,
        })
    }

    /// Projects storage choices while retaining logical slots, equations and invocation owners.
    pub(crate) fn with_formats(
        &self,
        mut format: impl FnMut(&ParameterSpec, &LinearFormatSpec) -> Result<LinearFormatSpec, Error>,
        mut experts: impl FnMut(
            usize,
            &GroupedGatedProductSpec,
        ) -> Result<GroupedGatedProductSpec, Error>,
    ) -> Result<Self, Error> {
        let mut selected = self.clone();
        let static_parameters = &mut selected.static_parameters;
        static_parameters.embedding.format = storage_format(
            &static_parameters.embedding.format,
            format(
                &static_parameters.embedding.weight,
                &static_parameters.embedding.format,
            )?,
        )?;
        if let Some(head) = &mut static_parameters.head {
            linear_formats(head, &mut format)?;
        }
        residual_formats(&mut static_parameters.readout, &mut format)?;
        unit_formats(&mut selected.units, &mut format, &mut experts)?;
        validate_format_projection(self, &selected)?;
        Ok(selected)
    }

    pub(crate) fn from_config(
        config: &Config,
        mut experts: impl FnMut(usize, usize) -> Result<GroupedGatedProductSpec, Error>,
        mut format: impl FnMut(&str) -> Result<LinearFormatSpec, Error>,
    ) -> Result<Self, Error> {
        let static_parameters = StaticParameters::from_config(config, &mut format)?;
        let units = TargetInvocation::schedule(config)
            .enumerate()
            .map(|(ordinal, invocation)| {
                let layer = invocation.layer();
                let root = format!("model.layers.{layer}");
                Ok(match invocation {
                    TargetInvocation::Lexical { .. } => UnitParameters::Lexical {
                        layer,
                        parameters: LexicalParameters::from_config(
                            config,
                            layer,
                            &root,
                            &mut format,
                        )?,
                    },
                    TargetInvocation::Decoder { kind, .. } => UnitParameters::Decoder {
                        layer,
                        mixer: MixerParameters::from_config(config, kind, &root, &mut format)?,
                        feed_forward: FeedForwardSublayerSpec::from_config(
                            config,
                            ordinal,
                            &root,
                            experts(layer, ordinal)?,
                            &mut format,
                        )?,
                    },
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let declaration = Self {
            static_parameters,
            units,
        };
        declaration.logical_parameters()?;
        Ok(declaration)
    }
}
impl DeclareLogicalParameters for TargetParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut slots = self.static_parameters.logical_parameters()?;
        for unit in &self.units {
            slots.extend(unit.logical_parameters()?);
        }
        eredu_nn::validate_logical_parameter_topology(&slots).map_err(Error::backend_source)?;
        Ok(slots)
    }
}

/// Retains operator policy when projecting physical storage and its companions.
pub(crate) fn storage_format(
    source: &LinearFormatSpec,
    selected: LinearFormatSpec,
) -> Result<LinearFormatSpec, Error> {
    let mut selected = selected.with_arithmetic(source.arithmetic())?;
    if source.exact_zero_inputs() {
        selected = selected.with_exact_zero_inputs();
    }
    Ok(selected)
}

pub(crate) fn linear_formats(
    spec: &mut LinearSpec,
    format: &mut impl FnMut(&ParameterSpec, &LinearFormatSpec) -> Result<LinearFormatSpec, Error>,
) -> Result<(), Error> {
    spec.format = storage_format(&spec.format, format(&spec.weight, &spec.format)?)?;
    Ok(())
}
pub(crate) fn residual_formats(
    spec: &mut GatedResidualSpec,
    format: &mut impl FnMut(&ParameterSpec, &LinearFormatSpec) -> Result<LinearFormatSpec, Error>,
) -> Result<(), Error> {
    linear_formats(&mut spec.down, format)?;
    linear_formats(&mut spec.up, format)?;
    if let Some(injection) = &mut spec.injection {
        linear_formats(injection, format)?;
    }
    Ok(())
}

pub(crate) fn unit_formats(
    units: &mut [UnitParameters],
    format: &mut impl FnMut(&ParameterSpec, &LinearFormatSpec) -> Result<LinearFormatSpec, Error>,
    experts: &mut impl FnMut(usize, &GroupedGatedProductSpec) -> Result<GroupedGatedProductSpec, Error>,
) -> Result<(), Error> {
    for (ordinal, unit) in units.iter_mut().enumerate() {
        match unit {
            UnitParameters::Lexical { parameters, .. } => {
                linear_formats(&mut parameters.key, format)?;
                linear_formats(&mut parameters.value, format)?;
            }
            UnitParameters::Decoder {
                mixer,
                feed_forward,
                ..
            } => {
                match mixer {
                    MixerParameters::Recurrent(spec) => {
                        residual_formats(&mut spec.residual, format)?;
                        for projection in [
                            &mut spec.mixer.input_qkv,
                            &mut spec.mixer.input_gate,
                            &mut spec.mixer.input_beta,
                            &mut spec.mixer.input_decay,
                            &mut spec.mixer.output,
                        ] {
                            linear_formats(projection, format)?;
                        }
                    }
                    MixerParameters::Indexed {
                        residual: boundary,
                        parameters,
                    } => {
                        residual_formats(boundary, format)?;
                        for projection in [
                            &mut parameters.projection.query,
                            &mut parameters.projection.key,
                            &mut parameters.projection.value,
                            &mut parameters.projection.output,
                            &mut parameters.index_projection,
                        ] {
                            linear_formats(projection, format)?;
                        }
                    }
                }
                residual_formats(&mut feed_forward.residual, format)?;
                let parameters = &mut feed_forward.feed_forward.parameters;
                let router_format = storage_format(
                    parameters.routed.selector.format(),
                    format(
                        parameters.routed.selector.weight(),
                        parameters.routed.selector.format(),
                    )?,
                )?;
                parameters.routed.selector = parameters
                    .routed
                    .selector
                    .clone()
                    .with_format(router_format)?;
                let bank = experts(ordinal, &parameters.routed.experts)?;
                if bank.policy() != parameters.routed.experts.policy() {
                    return Err(Error::backend(
                        "selected Qwen4 storage changes expert equations",
                    ));
                }
                parameters.routed.experts =
                    bank.with_reduction(parameters.routed.experts.reduction());
                for projection in [
                    &mut parameters.shared.gate,
                    &mut parameters.shared.up,
                    &mut parameters.shared.down,
                    &mut parameters.shared_gate,
                ] {
                    linear_formats(projection, format)?;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_format_projection(
    source: &impl DeclareLogicalParameters,
    projected: &impl DeclareLogicalParameters,
) -> Result<(), Error> {
    let original = source.logical_parameters()?;
    let projected = projected.logical_parameters()?;
    if original.len() != projected.len()
        || original.iter().zip(&projected).any(|(a, b)| {
            let mut logical = a.parameter.clone();
            // Independent encoded row blocks belong to the selected storage extension.
            logical.linear_row_layout = b.parameter.linear_row_layout;
            &logical != b.parameter
                || a.shape != b.shape
                || a.role != b.role
                || a.linear_format
                    .map(|f| (f.arithmetic(), f.exact_zero_inputs()))
                    != b.linear_format
                        .map(|f| (f.arithmetic(), f.exact_zero_inputs()))
        })
    {
        return Err(Error::backend(
            "selected Qwen4 storage changes declared geometry, ownership or operator policy",
        ));
    }
    Ok(())
}
