//! Retained typed routing and expert declarations, independent of backend allocation.
use super::ModelArgs;
use crate::decoder::Config;
use eredu_nn::{DeclareLogicalParameters, Error};

/// Global provider slots shared by admission, placement and materialization.
#[derive(Debug, Clone)]
pub struct Declaration {
    args: ModelArgs,
    common: std::sync::Arc<crate::decoder::DecoderDeclaration>,
    projections: crate::decoder::ProjectionDeclarations<eredu_nn::RoutedGatedProductSpec>,
}
impl Declaration {
    /// Validates and retains one typed routing/expert scope per canonical body owner.
    pub fn new(args: ModelArgs) -> Result<Self, Error> {
        let projections = crate::decoder::ProjectionDeclarations::new(&args, |layer| {
            super::moe::routed_spec(&args, layer)
        })?;
        let common = std::sync::Arc::new(crate::decoder::DecoderDeclaration::without_feed_forward(
            &args,
        )?);
        Self::from_scopes(args, common, projections)
    }
    pub(crate) fn from_scopes(
        args: ModelArgs,
        common: std::sync::Arc<crate::decoder::DecoderDeclaration>,
        projections: crate::decoder::ProjectionDeclarations<eredu_nn::RoutedGatedProductSpec>,
    ) -> Result<Self, Error> {
        common.validate_for(&args)?;
        for layer in 0..args.num_hidden_layers as usize {
            if common.block_parameters(layer).feed_forward.is_some() {
                return Err(Error::backend(
                    "GPT-OSS common scope contains an integrated feed-forward stage",
                ));
            }
        }
        projections.validate_for(&args)?;
        let declaration = Self {
            args,
            common,
            projections,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let formats = declaration
            .args
            .quantized_weight_configs
            .iter()
            .flat_map(|m| m.iter())
            .map(|(name, format)| (name.as_str(), Some((*format).into())));
        crate::format_references::validate(
            &declaration,
            formats,
            || {
                let mut sources = Vec::new();
                for tensor in super::safetensors_expert_tensors(&declaration)? {
                    if tensor.linear_companion.is_none() {
                        if let Some(format) = tensor.linear_format {
                            sources.push((tensor.key, format));
                            sources.extend(tensor.aliases.into_iter().map(|name| (name, format)));
                        }
                    }
                }
                Ok(sources)
            },
            || super::safetensors_plan(&declaration),
        )?;
        Ok(declaration)
    }
    /// Exact normalized family configuration of these source slots.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }
    /// Shared static and common body slots, with independent invocation-state geometry.
    pub fn common(&self) -> &std::sync::Arc<crate::decoder::DecoderDeclaration> {
        &self.common
    }
    /// Canonical typed provider slots, independent of selected invocations.
    pub fn projections(
        &self,
    ) -> &crate::decoder::ProjectionDeclarations<eredu_nn::RoutedGatedProductSpec> {
        &self.projections
    }
    /// Applies only explicit ordinary-matrix encoding changes to retained slots.
    pub fn with_formats(&self, args: ModelArgs) -> Result<Self, Error> {
        fn logical(mut args: ModelArgs) -> ModelArgs {
            args.quantization = None;
            args.quantized_weight_configs = None;
            args
        }
        args.validate().map_err(Error::backend_source)?;
        if logical(args.clone()).architecture_fingerprint()
            != logical(self.args.clone()).architecture_fingerprint()
        {
            return Err(Error::backend(
                "GPT-OSS format projection changes retained architecture policy",
            ));
        }
        let projections = crate::decoder::ProjectionDeclarations::new(&args, |layer| {
            let mut spec = self.projections.parameters(layer)?.clone();
            let weight = spec.selector.weight().id.as_str();
            spec.selector =
                spec.selector
                    .clone()
                    .with_format(crate::linear_format::project_matrix_format(
                        spec.selector.weight(),
                        spec.selector.format(),
                        args.checkpoint_weight_quantization_for(weight).into(),
                    )?)?;
            Ok(spec)
        })?;
        let common = std::sync::Arc::new(self.common.with_formats(&args)?);
        Self::from_scopes(args, common, projections)
    }
    /// Complete execution contract projected from common and routed scopes.
    pub fn component_definition(&self) -> Result<eredu_runtime::ComponentDefinition, Error> {
        crate::decoder::component_definition(
            &self.args,
            super::parallel::parameter_description(self, crate::decoder::ParameterEncoding::Stored)
                .map_err(Error::backend_source)?,
            self.common.state_layout().clone(),
        )
    }
    /// One exact packed bank retained for a canonical body owner.
    pub fn expert_bank(&self, layer: usize) -> Result<&eredu_nn::GroupedGatedProductSpec, Error> {
        Ok(&self.projections.parameters(layer)?.experts)
    }
    /// Packed bank scopes projected from the retained routed parameters.
    pub fn expert_banks(
        &self,
    ) -> Result<std::collections::BTreeMap<usize, eredu_nn::GroupedGatedProductSpec>, Error> {
        self.projections
            .owner_layers()
            .map(|layer| Ok((layer, self.projections.parameters(layer)?.experts.clone())))
            .collect()
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut slots = self.common.logical_parameters()?;
        slots.extend(self.projections.logical_parameters()?);
        Ok(slots)
    }
}
