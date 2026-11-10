//! Immutable common and provider parameter scopes for Qwen text decoders.
use super::{FeedForwardParameters, ModelArgs};
use crate::decoder::{Config, DecoderDeclaration, ProjectionDeclarations};
use eredu_nn::{DeclareLogicalParameters, Error};
use std::sync::Arc;

/// Complete text parameter declaration, independent of backend resources and placement.
#[derive(Debug, Clone)]
pub struct Declaration {
    args: ModelArgs,
    common: Arc<DecoderDeclaration>,
    projections: Option<ProjectionDeclarations<FeedForwardParameters>>,
}
impl Declaration {
    /// Declares common slots and each canonical provider owner once.
    pub fn new(args: ModelArgs) -> Result<Self, Error> {
        let common = Arc::new(if args.is_moe() {
            DecoderDeclaration::without_feed_forward(&args)?
        } else {
            DecoderDeclaration::new(&args)?
        });
        let projections = args
            .is_moe()
            .then(|| {
                ProjectionDeclarations::new(&args, |layer| {
                    Ok(FeedForwardParameters::Routed(super::moe::routed_spec(
                        &args, layer,
                    )?))
                })
            })
            .transpose()?;
        Self::from_scopes(args, common, projections)
    }
    pub(crate) fn from_scopes(
        args: ModelArgs,
        common: Arc<DecoderDeclaration>,
        projections: Option<ProjectionDeclarations<FeedForwardParameters>>,
    ) -> Result<Self, Error> {
        common.validate_for(&args)?;
        if args.is_moe() != projections.is_some() {
            return Err(Error::backend(
                "Qwen declaration changes its feed-forward equation",
            ));
        }
        for layer in 0..args.num_hidden_layers as usize {
            if common.block_parameters(layer).feed_forward.is_some() == args.is_moe() {
                return Err(Error::backend("Qwen common and provider scopes conflict"));
            }
        }
        if let Some(scopes) = &projections {
            scopes.validate_for(&args)?;
            for owner in scopes.owner_layers() {
                if !matches!(scopes.parameters(owner)?, FeedForwardParameters::Routed(_)) {
                    return Err(Error::backend("Qwen routed scope changes its equation"));
                }
            }
        }
        let declaration = Self {
            args,
            common,
            projections,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let args = declaration.args();
        let formats = args
            .quantized_weight_configs
            .iter()
            .flat_map(|m| m.iter())
            .map(|(name, format)| (name.as_str(), Some((*format).into())));
        let selected = args
            .quantized_weights
            .iter()
            .flat_map(|names| names.iter())
            .map(|name| {
                (
                    name.as_str(),
                    args.weight_quantization_for(name).map(Into::into),
                )
            });
        crate::format_references::validate(
            &declaration,
            formats.chain(selected),
            || {
                args.tie_word_embeddings
                    .then(|| {
                        let (name, _) = super::checkpoint::tied_output_source(&declaration)?;
                        let format = args.weight_quantization_for(&name).into();
                        Ok((name, format))
                    })
                    .transpose()
                    .map(|source| source.into_iter().collect())
            },
            || super::safetensors_plan(&declaration, true),
        )?;
        Ok(declaration)
    }
    /// Exact normalized source policy.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }
    /// Shared static/body scopes and independent invocation-state geometry.
    pub fn common(&self) -> &Arc<DecoderDeclaration> {
        &self.common
    }
    /// Provider stages, absent for integrated dense feed-forward scopes.
    pub fn projections(&self) -> Option<&ProjectionDeclarations<FeedForwardParameters>> {
        self.projections.as_ref()
    }
    pub(crate) fn routed(&self, layer: usize) -> Result<&eredu_nn::RoutedGatedProductSpec, Error> {
        match self
            .projections
            .as_ref()
            .ok_or_else(|| Error::backend("dense Qwen has no routed declaration"))?
            .parameters(layer)?
        {
            FeedForwardParameters::Routed(spec) => Ok(spec),
            _ => Err(Error::backend(
                "Qwen routed declaration contains dense provider slots",
            )),
        }
    }
    /// Exact packed bank of a declared canonical owner.
    pub fn expert_bank(&self, layer: usize) -> Result<&eredu_nn::GroupedGatedProductSpec, Error> {
        Ok(&self.routed(layer)?.experts)
    }
    /// Canonical bank geometry without backend or source-format traversal.
    pub fn expert_banks(
        &self,
    ) -> Result<std::collections::BTreeMap<usize, eredu_nn::GroupedGatedProductSpec>, Error> {
        let Some(scopes) = &self.projections else {
            return Ok(Default::default());
        };
        scopes
            .owner_layers()
            .map(|layer| Ok((layer, self.expert_bank(layer)?.clone())))
            .collect()
    }
    /// Projects encoding extensions while preserving all non-format policy.
    pub fn with_formats(&self, args: ModelArgs) -> Result<Self, Error> {
        fn logical(mut args: ModelArgs) -> ModelArgs {
            args.quantization = None;
            args.quantized_weights = None;
            args.quantized_weight_configs = None;
            args
        }
        args.validate().map_err(Error::backend_source)?;
        if logical(args.clone()).architecture_fingerprint()
            != logical(self.args.clone()).architecture_fingerprint()
        {
            return Err(Error::backend(
                "Qwen format projection changes retained architecture policy",
            ));
        }
        let common = Arc::new(self.common.with_formats(&args)?);
        let projections = self
            .projections
            .as_ref()
            .map(|_| {
                ProjectionDeclarations::new(&args, |layer| {
                    let mut spec = self.routed(layer)?.clone();
                    let name = spec.selector.weight().id.as_str();
                    spec.selector = spec.selector.clone().with_format(
                        crate::linear_format::project_matrix_format(
                            spec.selector.weight(),
                            spec.selector.format(),
                            args.weight_quantization_for(name).into(),
                        )?,
                    )?;
                    spec.experts = crate::linear_format::project_packed_gated_formats(
                        &spec.experts,
                        |name| args.weight_quantization_for(name).into(),
                    )?;
                    Ok(FeedForwardParameters::Routed(spec))
                })
            })
            .transpose()?;
        Self::from_scopes(args, common, projections)
    }
    /// Complete execution resources projected from the same common/provider slots.
    pub fn component_definition(&self) -> Result<eredu_runtime::ComponentDefinition, Error> {
        crate::decoder::component_definition(
            &self.args,
            super::parallel::parameter_description(self, crate::decoder::ParameterEncoding::Stored)
                .map_err(Error::backend_source)?,
            self.common.state_layout().clone(),
        )
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut slots = self.common.logical_parameters()?;
        if let Some(scopes) = &self.projections {
            slots.extend(scopes.logical_parameters()?);
        }
        Ok(slots)
    }
}
