//! Static text parameter ownership shared by construction and cold consumers.
use crate::gemma4::ModelArgs;
use eredu_nn::{
    DeclareLogicalParameters, EmbeddingSpec, Error, LinearSpec, LogicalParameterDeclaration,
    NormalizationConstructionSpec, ParameterSpec,
};

/// Whole per-layer identity table, decoder projection, and per-channel gain.
#[derive(Debug, Clone)]
pub struct PerLayerSpec {
    /// Token identities across every decoder layer; materialized as one static resource.
    pub embedding: EmbeddingSpec,
    /// Complete decoder-to-layer-input projection before rank-local input selection.
    pub projection: LinearSpec,
    /// Gain shared by each layer's projected input channels.
    pub norm: NormalizationConstructionSpec,
}
impl PerLayerSpec {
    pub(crate) fn project_formats(&self, args: &ModelArgs) -> Result<Self, Error> {
        let mut projected = self.clone();
        projected.embedding.format = crate::linear_format::project_matrix_format(
            &self.embedding.weight,
            &self.embedding.format,
            args.linear_format_for(self.embedding.weight.id.as_str()),
        )?;
        projected.projection.format = crate::linear_format::project_matrix_format(
            &self.projection.weight,
            &self.projection.format,
            args.linear_format_for(self.projection.weight.id.as_str()),
        )?;
        eredu_nn::validate_logical_parameter_topology(&projected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(projected)
    }
    /// Declares the complete static layer-input source scope at its owning namespace.
    pub(crate) fn at(args: &ModelArgs, root: &str) -> Result<Option<Self>, Error> {
        Ok(if args.hidden_size_per_layer_input > 0 {
            let width = args.hidden_size_per_layer_input;
            let layers = i32::try_from(args.num_hidden_layers()).map_err(Error::backend)?;
            let combined = width
                .checked_mul(layers)
                .ok_or_else(|| Error::backend("Gemma combined per-layer width overflows"))?;
            let name = format!("{root}.embed_tokens_per_layer.weight");
            Some(Self {
                embedding: EmbeddingSpec {
                    vocabulary: args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
                    dimensions: combined,
                    weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        &name,
                        args.linear_format_for(&name),
                    )?,
                },
                projection: super::specs::linear_spec(
                    args,
                    &format!("{root}.per_layer_model_projection.weight"),
                    args.hidden_size,
                    combined,
                    false,
                )?,
                norm: NormalizationConstructionSpec::learned(
                    width,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{root}.per_layer_projection_norm.weight"))
                        .map_err(Error::backend)?,
                ),
            })
        } else {
            None
        })
    }
}
impl DeclareLogicalParameters for PerLayerSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.embedding.logical_parameters()?;
        p.extend(self.projection.logical_parameters()?);
        p.extend(self.norm.logical_parameters()?);
        Ok(p)
    }
}

/// Ordinary vocabulary ownership and optional static per-layer input resources.
#[derive(Debug, Clone)]
pub struct Declaration {
    pub(crate) ordinary: crate::decoder::StaticModuleConstruction,
    /// Independently owned per-layer input resources, when the equation uses them.
    pub per_layer: Option<PerLayerSpec>,
    root: &'static str,
}
impl Declaration {
    /// Gemma's static final gain uses native weighted normalization arithmetic.
    pub(crate) fn gemma(args: &ModelArgs) -> Result<Self, Error> {
        Self::build(args, "model.language_model", false)
    }
    /// Diffusion's static final gain follows its configured normalization arithmetic.
    pub(crate) fn diffusion(args: &ModelArgs) -> Result<Self, Error> {
        Self::build(args, "model.decoder", true)
    }
    fn build(args: &ModelArgs, root: &'static str, configured_gain: bool) -> Result<Self, Error> {
        let embedding_name = format!("{root}.embed_tokens.weight");
        let embedding = EmbeddingSpec {
            vocabulary: args.vocab_size,
            dimensions: args.hidden_size,
            weight: ParameterSpec::trainable(&embedding_name).map_err(Error::backend)?,
            format: crate::linear_format::standard_linear_format(
                &embedding_name,
                args.linear_format_for(&embedding_name),
            )?,
        };
        let head = (!args.tie_word_embeddings)
            .then(|| {
                super::specs::linear_spec(
                    args,
                    "lm_head.weight",
                    args.hidden_size,
                    args.vocab_size,
                    false,
                )
            })
            .transpose()?;
        let norm_weight =
            ParameterSpec::trainable(format!("{root}.norm.weight")).map_err(Error::backend)?;
        let norm = if configured_gain {
            args.normalization(args.hidden_size, norm_weight)
        } else {
            NormalizationConstructionSpec::learned(args.hidden_size, args.rms_norm_eps, norm_weight)
        };
        let ordinary = crate::decoder::StaticModuleConstruction::new(embedding, head, norm)?;
        let per_layer = PerLayerSpec::at(args, root)?;
        let declaration = Self {
            ordinary,
            per_layer,
            root,
        };
        declaration.validate_for(args)?;
        Ok(declaration)
    }

    /// Projects storage extensions onto retained owners without rebuilding geometry.
    pub(crate) fn project_formats(&self, args: &ModelArgs) -> Result<Self, Error> {
        fn format(
            weight: &ParameterSpec,
            old: &eredu_nn::LinearFormatSpec,
            args: &ModelArgs,
        ) -> Result<eredu_nn::LinearFormatSpec, Error> {
            let mut format = crate::linear_format::standard_linear_format(
                weight.id.as_str(),
                args.linear_format_for(weight.id.as_str()),
            )?
            .with_arithmetic(old.arithmetic())?;
            if old.exact_zero_inputs() {
                format = format.with_exact_zero_inputs();
            }
            Ok(format)
        }
        let mut embedding = self.ordinary.embedding_spec().clone();
        embedding.format = format(&embedding.weight, &embedding.format, args)?;
        let mut head = self.ordinary.head_spec().cloned();
        if let Some(head) = &mut head {
            head.format = format(&head.weight, &head.format, args)?;
        }
        let ordinary = crate::decoder::StaticModuleConstruction::new(
            embedding,
            head,
            self.normalization().clone(),
        )?;
        let per_layer = self
            .per_layer
            .as_ref()
            .map(|spec| spec.project_formats(args))
            .transpose()?;
        let declaration = Self {
            ordinary,
            per_layer,
            root: self.root,
        };
        declaration.validate_for(args)?;
        Ok(declaration)
    }
    /// Resolves pinned geometry and format references against the selected source policy.
    pub(crate) fn validate_for(&self, args: &ModelArgs) -> Result<(), Error> {
        let embedding = self.embedding();
        if embedding.vocabulary != args.vocab_size
            || embedding.dimensions != args.hidden_size
            || self.head().is_none() != args.tie_word_embeddings
            || self.normalization().epsilon != args.rms_norm_eps
            || self.per_layer.is_some() != (args.hidden_size_per_layer_input > 0)
        {
            return Err(Error::backend(
                "Gemma pinned source policy changes retained geometry or ownership",
            ));
        }
        if let Some(spec) = &self.per_layer {
            let combined = args
                .hidden_size_per_layer_input
                .checked_mul(i32::try_from(args.num_hidden_layers()).map_err(Error::backend)?)
                .ok_or_else(|| Error::backend("Gemma per-layer table width overflows"))?;
            if spec.embedding.dimensions != combined
                || spec.embedding.vocabulary
                    != args.vocab_size_per_layer_input.unwrap_or(args.vocab_size)
                || spec.projection.input != args.hidden_size
                || spec.projection.output != combined
                || spec.norm.dimensions != args.hidden_size_per_layer_input
            {
                return Err(Error::backend(
                    "Gemma per-layer source geometry is incompatible with its retained owner",
                ));
            }
        }
        let parameters = self.logical_parameters()?;
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(Error::backend_source)?;
        let embedding_name =
            crate::gemma4::config::canonical_weight_name(embedding.weight.id.as_str());
        let root = embedding_name
            .strip_suffix("embed_tokens.weight")
            .ok_or_else(|| Error::backend("invalid pinned embedding identity"))?;
        for name in args
            .quantized_weight_configs
            .iter()
            .flat_map(|f| f.keys())
            .chain(args.quantized_weights.iter().flatten())
        {
            let canonical = crate::gemma4::config::canonical_weight_name(name);
            let pinned = canonical.starts_with(&format!("{root}embed_tokens"))
                || canonical.starts_with(&format!("{root}per_layer_"))
                || canonical.starts_with(&format!("{root}norm."))
                || canonical.starts_with("lm_head.");
            if pinned {
                let parameter = parameters
                    .iter()
                    .find(|p| {
                        p.linear_format.is_some()
                            && crate::gemma4::config::canonical_weight_name(p.parameter.id.as_str())
                                == canonical
                    })
                    .ok_or_else(|| {
                        Error::backend(format!(
                            "Gemma pinned format references undeclared projection {name}"
                        ))
                    })?;
                if let Some(format) = args
                    .quantized_weight_configs
                    .as_ref()
                    .and_then(|formats| formats.get(name))
                {
                    if parameter
                        .linear_format
                        .expect("declared projection")
                        .encoding()
                        != (*format).into()
                    {
                        return Err(Error::backend(format!(
                            "Gemma pinned format alias is unresolved or conflicting: {name}"
                        )));
                    }
                }
                if args
                    .quantized_weights
                    .as_ref()
                    .is_some_and(|names| names.contains(name))
                    && name != parameter.parameter.id.as_str()
                {
                    return Err(Error::backend(format!(
                        "Gemma pinned selection requires the declared identity: {name}"
                    )));
                }
            }
        }
        Ok(())
    }
    /// Canonical component root; the optional independent readout remains outside this root.
    pub const fn root(&self) -> &'static str {
        self.root
    }
    /// Token table that also owns a tied readout's physical encoding.
    pub fn embedding(&self) -> &EmbeddingSpec {
        self.ordinary.embedding_spec()
    }
    /// Independent readout, absent when token embeddings own the output weights.
    pub fn head(&self) -> Option<&LinearSpec> {
        self.ordinary.head_spec()
    }
    /// Projects readout geometry from its actual owner without declaring another parameter.
    pub fn readout_projection(&self) -> std::borrow::Cow<'_, LinearSpec> {
        self.ordinary.output_projection_spec()
    }
    /// Exact final gain geometry and arithmetic.
    pub fn normalization(&self) -> &NormalizationConstructionSpec {
        self.ordinary.normalization_spec()
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.ordinary.logical_parameters()?;
        if let Some(spec) = &self.per_layer {
            p.extend(spec.logical_parameters()?)
        }
        Ok(p)
    }
}
