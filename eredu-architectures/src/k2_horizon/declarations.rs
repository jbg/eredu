//! Retained common, dense-prefix and independently routed K2 parameter scopes.
use super::{specs::BlockSpec, ModelArgs, ProjectionParameters};
use crate::decoder::{Config, DecoderDeclaration, ProjectionDeclarations};
use eredu_nn::{DeclareLogicalParameters, Error};
use std::sync::Arc;

/// Shared typed construction authority, retained independently of mutable local policy.
#[derive(Debug, Clone)]
pub(crate) struct ParameterScopes {
    common: Arc<DecoderDeclaration>,
    projections: Option<Arc<ProjectionDeclarations<ProjectionParameters>>>,
}

/// Complete K2 parameter authority, independent of source layout and selected rank.
#[derive(Debug, Clone)]
pub struct Declaration {
    args: ModelArgs,
    common: Arc<DecoderDeclaration>,
    projections: Option<Arc<ProjectionDeclarations<ProjectionParameters>>>,
}
impl Declaration {
    /// Declares every logical slot once, separating parameter owners from invocation state.
    pub fn new(args: ModelArgs) -> Result<Self, Error> {
        if let Some(retained) = args.retained_scopes.clone() {
            return Self::from_retained(args, retained);
        }
        if !args.is_moe() {
            let common = Arc::new(DecoderDeclaration::new(&args)?);
            return Self::from_scopes(args, common, None);
        }
        let mut blocks = (0..args.num_hidden_layers as usize)
            .map(|layer| BlockSpec::new(&args, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let projections = ProjectionDeclarations::new(&args, |layer| {
            let block = &mut blocks[layer];
            Ok(ProjectionParameters {
                dense: block.common.feed_forward.take(),
                routed: block.routed.take(),
                values: block.values.take(),
                points: args.routed_observation_points(&format!("model.layers.{layer}"), layer),
            })
        })?;
        let common = Arc::new(DecoderDeclaration::from_blocks(
            &args,
            crate::decoder::StaticModuleSpec::from_config(&args).construction_specs()?,
            blocks.into_iter().map(|block| block.common).collect(),
        )?);
        Self::from_scopes(args, common, Some(projections))
    }
    pub(crate) fn from_scopes(
        args: ModelArgs,
        common: Arc<DecoderDeclaration>,
        projections: Option<ProjectionDeclarations<ProjectionParameters>>,
    ) -> Result<Self, Error> {
        Self::from_retained(
            args,
            Arc::new(ParameterScopes {
                common,
                projections: projections.map(Arc::new),
            }),
        )
    }
    fn from_retained(mut args: ModelArgs, retained: Arc<ParameterScopes>) -> Result<Self, Error> {
        let common = Arc::clone(&retained.common);
        let projections = retained.projections.clone();
        common.validate_for(&args)?;
        if args.is_moe() != projections.is_some() {
            return Err(Error::backend("K2 declaration changes provider ownership"));
        }
        if projections.is_none() {
            for layer in 0..args.num_hidden_layers as usize {
                if common.block_parameters(layer).feed_forward.is_none() {
                    return Err(Error::backend(
                        "dense K2 scope omits its feed-forward stage",
                    ));
                }
            }
        }
        if let Some(scopes) = &projections {
            scopes.validate_for(&args)?;
            for layer in scopes.owner_layers() {
                let p = scopes.parameters(layer)?;
                if common.block_parameters(layer).feed_forward.is_some()
                    || p.routed.is_some() != args.is_sparse_layer(layer)
                    || p.dense.is_some() == args.is_sparse_layer(layer)
                    || p.values.is_some() != args.is_mova_layer(layer)
                {
                    return Err(Error::backend(
                        "K2 common/provider scopes change the invocation equation",
                    ));
                }
            }
        }
        args.retained_scopes = Some(retained);
        let declaration = Self {
            args,
            common,
            projections,
        };
        eredu_nn::validate_logical_parameter_topology(&declaration.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let mut identities = declaration
            .logical_parameters()?
            .into_iter()
            .filter(|p| p.shape.len() >= 2)
            .map(|p| p.parameter.id.to_string())
            .collect::<std::collections::BTreeSet<_>>();
        for layout in [super::SourceLayout::Safetensors, super::SourceLayout::Gguf] {
            identities.extend(
                super::source_parameters(&declaration, layout)
                    .map_err(Error::backend)?
                    .into_iter()
                    .filter(|p| p.shape.len() >= 2)
                    .map(|p| p.name),
            );
        }
        if let Some(name) = declaration
            .args
            .formats
            .keys()
            .find(|name| !identities.contains(*name))
        {
            return Err(Error::backend(format!(
                "undeclared K2 matrix format reference {name}"
            )));
        }
        Ok(declaration)
    }
    /// Exact global family policy.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }
    /// Shared attention/static/state declarations.
    pub fn common(&self) -> &Arc<DecoderDeclaration> {
        &self.common
    }
    /// Dense-prefix, routed/shared and value stages of a routed stack.
    pub fn projections(&self) -> Option<&ProjectionDeclarations<ProjectionParameters>> {
        self.projections.as_deref()
    }
    /// Complete typed view of one owner, without declaring geometry again.
    pub fn block(&self, layer: usize) -> Result<BlockSpec, Error> {
        if layer >= self.args.num_hidden_layers as usize {
            return Err(Error::backend("K2 block outside declared stack"));
        }
        let mut common = self.common.block_parameters(layer).clone();
        let (values, routed) = if let Some(scopes) = &self.projections {
            let p = scopes.parameters(layer)?;
            common.feed_forward = p.dense.clone();
            (p.values.clone(), p.routed.clone())
        } else {
            (None, None)
        };
        Ok(BlockSpec {
            common,
            values,
            routed,
        })
    }
    pub(crate) fn blocks(&self) -> Result<Vec<BlockSpec>, Error> {
        (0..self.args.num_hidden_layers as usize)
            .map(|layer| self.block(layer))
            .collect()
    }
    pub(crate) fn common_projection(&self) -> Result<DecoderDeclaration, Error> {
        if let Some(scopes) = &self.projections {
            self.common
                .with_feed_forward_parameters(&self.args, |layer| {
                    Ok(scopes.parameters(layer)?.dense.clone())
                })
        } else {
            Ok(self.common.as_ref().clone())
        }
    }
    /// Projects source encodings after proving unchanged non-format execution policy.
    pub fn with_formats(&self, args: ModelArgs) -> Result<Self, Error> {
        let mut policy = args.clone();
        policy.formats = self.args.formats.clone();
        args.validate().map_err(Error::backend_source)?;
        if policy.architecture_fingerprint() != self.args.architecture_fingerprint() {
            return Err(Error::backend(
                "K2 format projection changes execution policy",
            ));
        }
        if args.architecture_fingerprint() == self.args.architecture_fingerprint() {
            return Self::from_retained(
                args,
                self.args.retained_scopes.clone().ok_or_else(|| {
                    Error::backend("K2 source has no retained construction authority")
                })?,
            );
        }
        let common = Arc::new(self.common.with_formats(&args)?);
        let project_linear = |p: &mut eredu_nn::LinearSpec| -> Result<(), Error> {
            p.format = crate::linear_format::project_matrix_format(
                &p.weight,
                &p.format,
                args.linear_format_for(p.weight.id.as_str()),
            )?;
            Ok(())
        };
        let projections = self
            .projections
            .as_ref()
            .map(|scopes| {
                ProjectionDeclarations::new(&args, |layer| {
                    let mut p = scopes.parameters(layer)?.clone();
                    if let Some(dense) = &mut p.dense {
                        match dense {
                            eredu_nn::GatedProductSpec::Split(s) => {
                                project_linear(&mut s.gate)?;
                                project_linear(&mut s.up)?;
                                project_linear(&mut s.down)?;
                            }
                            eredu_nn::GatedProductSpec::Fused(s) => {
                                project_linear(&mut s.gate_up)?;
                                project_linear(&mut s.down)?;
                            }
                        }
                    }
                    let selector = |s: &eredu_nn::TopKGroupSelectorSpec| {
                        s.clone()
                            .with_format(crate::linear_format::project_matrix_format(
                                s.weight(),
                                s.format(),
                                args.linear_format_for(s.weight().id.as_str()),
                            )?)
                    };
                    if let Some(r) = &mut p.routed {
                        r.parameters.selector = selector(&r.parameters.selector)?;
                        r.parameters.experts = crate::linear_format::project_packed_gated_formats(
                            &r.parameters.experts,
                            |name| args.linear_format_for(name),
                        )?;
                        if let Some(s) = &mut r.shared {
                            project_linear(&mut s.gate)?;
                            project_linear(&mut s.up)?;
                            project_linear(&mut s.down)?;
                        }
                    }
                    if let Some(v) = &mut p.values {
                        v.router = selector(&v.router)?;
                        let bank = &v.experts;
                        let p = bank.projection();
                        v.experts = bank.clone().with_projection_format(
                            crate::linear_format::project_matrix_format(
                                p.weight(),
                                p.format(),
                                args.linear_format_for(p.weight().id.as_str()),
                            )?,
                        )?;
                    }
                    Ok(p)
                })
            })
            .transpose()?;
        Self::from_scopes(args, common, projections)
    }
    /// Memory topology from the actual dense/routed/shared feed-forward equations.
    pub fn execution_topology(
        &self,
    ) -> Result<eredu_runtime::execution_topology::TextExecutionTopology, Error> {
        use eredu_runtime::execution_topology::{FeedForwardTopology, ProjectionTopology};
        let gated = |spec: &eredu_nn::GatedProductSpec| -> Result<FeedForwardTopology, Error> {
            Ok(FeedForwardTopology::Gated {
                intermediate_size: spec.down().input as u64,
                projections: spec
                    .projections()
                    .map(ProjectionTopology::from_spec)
                    .collect::<Result<_, _>>()?,
            })
        };
        if self.projections.is_none() {
            return self.common.execution_topology(&self.args);
        }
        crate::decoder::topology::text_with_scopes(&self.args, &self.common, |layer| {
            let p = self
                .projections
                .as_ref()
                .expect("routed K2 stages")
                .parameters(layer)?;
            if let Some(dense) = &p.dense {
                return gated(dense);
            }
            let routed = p
                .routed
                .as_ref()
                .ok_or_else(|| Error::backend("missing K2 feed-forward equation"))?;
            let route = FeedForwardTopology::from_grouped_specs(
                &routed.parameters.selector,
                &routed.parameters.experts,
            )?;
            if let Some(shared) = &routed.shared {
                Ok(FeedForwardTopology::Additive {
                    branches: vec![
                        route,
                        gated(&eredu_nn::GatedProductSpec::Split(shared.clone()))?,
                    ],
                })
            } else {
                Ok(route)
            }
        })
    }
    /// Complete execution resources from the same common/provider slots.
    pub fn component_definition(&self) -> Result<eredu_runtime::ComponentDefinition, Error> {
        crate::decoder::component_definition(
            &self.args,
            super::parameter_description(self).map_err(Error::backend_source)?,
            self.common.state_layout().clone(),
        )
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut slots = self.common.logical_parameters()?;
        if let Some(p) = &self.projections {
            slots.extend(p.logical_parameters()?);
        }
        Ok(slots)
    }
}
