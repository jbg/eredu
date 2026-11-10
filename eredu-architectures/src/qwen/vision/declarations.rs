//! Retained typed parameter authority for the shared vision tower.
use super::{
    specs::{self, ParameterConstruction as P},
    VisionConfig,
};
use eredu_nn::{DeclareLogicalParameters, Error};

/// Immutable source slots, separate from vision invocations and selected placement.
#[derive(Debug, Clone)]
pub struct Declaration {
    config: VisionConfig,
    root: String,
    pub(super) slots: Vec<P>,
}
impl Declaration {
    /// Declares canonical parameters once at a component boundary.
    pub fn new(config: VisionConfig, root: &str) -> Result<Self, Error> {
        let slots = specs::parameter_construction(&config, root)?;
        let declaration = Self {
            config,
            root: root.into(),
            slots,
        };
        declaration.validate_formats()?;
        validate_storage(&declaration)?;
        Ok(declaration)
    }
    /// Complete invocation and format policy.
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }
    /// Canonical component parameter namespace.
    pub fn root(&self) -> &str {
        &self.root
    }
    pub(super) fn position(&self) -> &eredu_nn::EmbeddingSpec {
        let P::Position(spec) = &self.slots[0] else {
            unreachable!()
        };
        spec
    }
    pub(super) fn patch(&self) -> &specs::PatchSpec {
        let P::Patch(spec) = &self.slots[1] else {
            unreachable!()
        };
        spec
    }
    pub(super) fn block(&self, layer: usize) -> Result<&specs::BlockSpec, Error> {
        match self.slots.get(layer + 2) {
            Some(P::Block(spec)) => Ok(spec),
            _ => Err(Error::backend("undeclared vision block")),
        }
    }
    pub(super) fn merger(
        &self,
        index: usize,
    ) -> Result<
        (
            &eredu_nn::LayerNormalizationConstructionSpec,
            [&eredu_nn::LinearSpec; 2],
        ),
        Error,
    > {
        let offset = 2 + self.config.layer_count() + 3 * index;
        match self.slots.get(offset..offset + 3) {
            Some([P::Normalization(n), P::Linear(up), P::Linear(down)]) => Ok((n, [up, down])),
            _ => Err(Error::backend("undeclared vision merger")),
        }
    }
    pub(crate) fn local_block(
        &self,
        layer: usize,
        heads: i32,
        intermediate: i32,
    ) -> Result<specs::BlockSpec, Error> {
        let mut spec = self.block(layer)?.clone();
        if heads <= 0
            || heads > spec.attention.geometry.query_heads()
            || intermediate <= 0
            || intermediate > spec.feed_forward.up.output
        {
            return Err(Error::backend(
                "local vision geometry exceeds declared slots",
            ));
        }
        let old = spec.attention.geometry;
        let g = eredu_nn::AttentionHeadGeometry::new(
            heads,
            heads,
            old.key_dimensions(),
            old.value_dimensions(),
        )?;
        spec.attention.geometry = g;
        spec.attention.layout = eredu_nn::FusedProjectionLayout::new([
            eredu_nn::FusedProjectionSegment::new("query", g.query_width())?,
            eredu_nn::FusedProjectionSegment::new("key", g.key_width())?,
            eredu_nn::FusedProjectionSegment::new("value", g.value_width())?,
        ])?;
        spec.attention.qkv.output = spec.attention.layout.output_width();
        spec.attention.output.input = g.attended_width();
        spec.feed_forward.up.output = intermediate;
        spec.feed_forward.down.input = intermediate;
        spec.logical_parameters()?;
        validate_storage(&spec)?;
        Ok(spec)
    }
    pub(super) fn local_merger(
        &self,
        index: usize,
        intermediate: i32,
    ) -> Result<
        (
            eredu_nn::LayerNormalizationConstructionSpec,
            [eredu_nn::LinearSpec; 2],
        ),
        Error,
    > {
        let (normalization, [up, down]) = self.merger(index)?;
        if intermediate <= 0 || intermediate > up.output {
            return Err(Error::backend("local merger width exceeds declaration"));
        }
        let mut up = up.clone();
        let mut down = down.clone();
        up.output = intermediate;
        down.input = intermediate;
        let result = (normalization.clone(), [up, down]);
        result.0.logical_parameters()?;
        result.1.as_slice().logical_parameters()?;
        validate_storage(result.1.as_slice())?;
        Ok(result)
    }
    /// Selects aligned affine slots using retained logical input geometry.
    pub fn with_quantization(
        &self,
        quantization: eredu_checkpoint::WeightQuantization,
    ) -> Result<Self, Error> {
        quantization.validate().map_err(Error::backend_source)?;
        let mut config = self.config.clone();
        config.linear_formats.clear();
        let group_size = quantization.group_size() as usize;
        for (name, shape) in super::source_linear_shapes(self).map_err(Error::backend)? {
            let input = shape[1];
            if input % group_size == 0 && input % 32 == 0 {
                config
                    .linear_formats
                    .insert(self.relative(&name)?.to_owned(), quantization.into());
            }
        }
        self.with_formats(config)
    }
    fn relative<'a>(&self, name: &'a str) -> Result<&'a str, Error> {
        if self.root.is_empty() {
            Ok(name)
        } else {
            name.strip_prefix(&self.root)
                .and_then(|name| name.strip_prefix('.'))
                .ok_or_else(|| Error::backend("vision slot outside component namespace"))
        }
    }
    fn validate_formats(&self) -> Result<(), Error> {
        let matrices = super::source_linear_shapes(self).map_err(Error::backend)?;
        let identities = matrices
            .keys()
            .map(|name| self.relative(name))
            .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
        for name in self.config.linear_formats.keys() {
            let relative = name.strip_prefix("model.visual.").unwrap_or(name);
            if !identities.contains(relative) {
                return Err(Error::backend(format!(
                    "vision format references undeclared affine slot {name}"
                )));
            }
        }
        Ok(())
    }
    /// Projects storage encodings while retaining identities and logical geometry.
    pub fn with_formats(&self, config: VisionConfig) -> Result<Self, Error> {
        let mut policy = config.clone();
        policy.linear_formats = self.config.linear_formats.clone();
        if policy != self.config {
            return Err(Error::backend(
                "vision format projection changes invocation policy",
            ));
        }
        config.validate().map_err(Error::backend_source)?;
        let mut result = self.clone();
        let apply = |spec: &mut eredu_nn::LinearSpec| -> Result<(), Error> {
            let name = spec.weight.id.as_str();
            let relative = self.relative(name)?;
            spec.format = crate::linear_format::project_matrix_format(
                &spec.weight,
                &spec.format,
                config.linear_format(relative),
            )?;
            Ok(())
        };
        for slot in &mut result.slots {
            match slot {
                P::Linear(s) => apply(s)?,
                P::Block(s) => {
                    apply(&mut s.attention.qkv)?;
                    apply(&mut s.attention.output)?;
                    apply(&mut s.feed_forward.up)?;
                    apply(&mut s.feed_forward.down)?;
                }
                _ => {}
            }
        }
        result.config = config;
        result.validate_formats()?;
        eredu_nn::validate_logical_parameter_topology(&result.logical_parameters()?)
            .map_err(Error::backend_source)?;
        validate_storage(&result)?;
        Ok(result)
    }
}
fn validate_storage(spec: &(impl DeclareLogicalParameters + ?Sized)) -> Result<(), Error> {
    eredu_runtime::logical_parameter_group(
        "vision declaration",
        eredu_runtime::ParameterRole::Replicated,
        None,
        spec,
        |_| eredu_runtime::MemberSharding::Replicated,
    )
    .map_err(Error::backend_source)?;
    Ok(())
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.slots.as_slice().logical_parameters()
    }
}

impl std::ops::Deref for Declaration {
    type Target = VisionConfig;
    fn deref(&self) -> &Self::Target {
        &self.config
    }
}
