//! Separately prepared prediction authority; target vocabulary stays target-owned.
use super::safetensors_target::HeaderCatalog;
use super::*;
use crate::qwen4_exp::mtp::{PredictionLimits, PredictionSpec};
use eredu_checkpoint::{
    recipe::RecipeCatalog,
    validation::{ResolvedCheckpointPlan, SafetensorsCatalog},
};
use eredu_runtime::ReplicatedTextPhysicalSource;

/// Immutable prediction-only SafeTensors headers and physical provenance. A full
/// checkpoint is accepted, but target, vision and table tensors are not retained.
#[derive(Debug, Clone)]
pub struct SafetensorsPredictionPlan {
    config: Config,
    encoding: SafetensorsEncoding,
    catalog: HeaderCatalog,
    resolution: ResolvedCheckpointPlan,
    physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
}
impl SafetensorsPredictionPlan {
    /// Resolves prediction aliases and companions without reading any payload.
    pub fn prepare<C: SafetensorsCatalog + ?Sized>(
        catalog: &C,
        config: Config,
        encoding: SafetensorsEncoding,
        physical: BTreeMap<String, ReplicatedTextPhysicalSource>,
    ) -> Result<Self, PreparationError> {
        let mut plan = Self::prepare_metadata(catalog, config, encoding)?;
        plan.physical = plan
            .resolution
            .source_keys()
            .iter()
            .map(|key| {
                let entry = physical.get(key).ok_or_else(|| {
                    PreparationError::Contract(format!(
                        "missing prediction physical declaration {key}"
                    ))
                })?;
                let header = &plan.catalog.0[key];
                if entry.catalog_key() != key
                    || entry.source_encoding()
                        != &eredu_checkpoint::SourceTensorEncoding::Safetensors(
                            header.stored_dtype.clone(),
                        )
                    || Some(entry.encoded_byte_len()) != HeaderCatalog::tensor_bytes(header)
                {
                    return Err(PreparationError::Contract(format!(
                        "physical declaration differs from header {key}"
                    )));
                }
                Ok((key.clone(), entry.clone()))
            })
            .collect::<Result<_, PreparationError>>()?;
        Ok(plan)
    }

    // Already retained in-memory or generated sources need exact tensor metadata,
    // but have no file identity. Public cold plans additionally pin physical files.
    fn prepare_metadata<C: SafetensorsCatalog + ?Sized>(
        catalog: &C,
        config: Config,
        encoding: SafetensorsEncoding,
    ) -> Result<Self, PreparationError> {
        if config.prediction.is_none() {
            return Err(PreparationError::MissingPrediction);
        }
        let mut headers = BTreeMap::new();
        for key in catalog
            .keys()
            .into_iter()
            .filter(|key| key.starts_with("mtp."))
        {
            let metadata = catalog.metadata(&key).map_err(PreparationError::Contract)?;
            if headers.insert(key.clone(), metadata).is_some() {
                return Err(PreparationError::Contract(format!(
                    "duplicate header {key}"
                )));
            }
        }
        let catalog = HeaderCatalog(headers);
        let checkpoint = checkpoint::schema::prediction_safetensors_plan(&config, &encoding)
            .map_err(PreparationError::Contract)?;
        let resolution = resolve_safetensors_plan(&catalog, &checkpoint)
            .map_err(|error| PreparationError::Contract(format!("{error:?}")))?;
        Ok(Self {
            config,
            encoding,
            catalog,
            resolution,
            physical: BTreeMap::new(),
        })
    }

    /// Exact family geometry supplied with the prediction artifact.
    pub fn configuration(&self) -> &Config {
        &self.config
    }
    /// Admitted aliases and physical source keys for the prediction role.
    pub fn resolution(&self) -> &ResolvedCheckpointPlan {
        &self.resolution
    }
    /// Original physical provenance, retained across joint selection.
    pub fn physical_sources(&self) -> &BTreeMap<String, ReplicatedTextPhysicalSource> {
        &self.physical
    }
    /// Checks normalized equations against the independently admitted target.
    pub fn validate_target(&self, target: &Config) -> Result<(), PreparationError> {
        match_target(target, &self.config)
    }
    /// Header-derived predictor geometry without materialization.
    pub fn prediction_spec(
        &self,
        limits: PredictionLimits,
    ) -> Result<PredictionSpec, PreparationError> {
        self.parts(limits).map(|(spec, _, _)| spec)
    }
    pub(super) fn parts(
        &self,
        limits: PredictionLimits,
    ) -> Result<
        (
            PredictionSpec,
            ParameterFormats,
            BTreeMap<String, Arc<PreparedExpertBank>>,
        ),
        PreparationError,
    > {
        let prediction = self
            .config
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        let keys = self
            .resolution
            .source_keys()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let roots =
            (0..prediction.layers.len()).map(|depth| format!("mtp.layers.{depth}.mlp.experts"));
        let (formats, recipes) =
            safetensors_format_recipes(&self.catalog, &keys, &self.config, &self.encoding, roots)?;
        let banks = recipes
            .into_iter()
            .map(|(root, recipes)| {
                Ok((
                    root,
                    Arc::new(PreparedExpertBank::new(
                        &self.catalog,
                        recipes,
                        self.config.experts.count as usize,
                    )?),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, PreparationError>>()?;
        let bank = eredu_runtime::RoutedBankId::new(
            u32::try_from(self.config.ngram.layers.len() + 1).map_err(|_| {
                PreparationError::Contract("prediction bank identity exceeds u32".into())
            })?,
        );
        let spec = PredictionSpec::from_prepared(
            &self.config,
            limits,
            bank,
            |depth| {
                expert_spec(
                    &self.config,
                    &formats,
                    &format!("mtp.layers.{depth}.mlp.experts"),
                )
            },
            |name| formats.ordinary(name),
        )?;
        Ok((spec, formats, banks))
    }
    /// Pins the exact admitted prediction headers and provenance. Target payloads
    /// of a complete companion checkpoint are excluded from the retained source.
    pub fn bind(
        self,
        source: SharedCheckpointSource,
    ) -> Result<PreparedPredictionSource, PreparationError> {
        let keys: BTreeSet<_> = source
            .source_keys()
            .into_iter()
            .filter(|key| key.starts_with("mtp."))
            .collect();
        if keys != *self.resolution.source_keys()
            && keys != self.catalog.0.keys().cloned().collect()
        {
            return Err(PreparationError::Contract(
                "prediction source set differs from admitted headers".into(),
            ));
        }
        let artifact = retain(source, self.resolution.source_keys().clone())?;
        for key in self.resolution.source_keys() {
            let metadata = artifact.source_metadata(key)?;
            let admitted = &self.catalog.0[key];
            if metadata.logical_shape != admitted.shape
                || metadata.physical_shape != admitted.shape
                || metadata.stored_dtype != admitted.stored_dtype
                || Some(metadata.encoded_byte_len) != HeaderCatalog::tensor_bytes(admitted)
            {
                return Err(PreparationError::Contract(format!(
                    "prediction source changed for {key}"
                )));
            }
            if let Some(expected) = self.physical.get(key) {
                let actual = crate::replicated_text::exact_physical_source(artifact.as_ref(), key)
                    .map_err(|error| PreparationError::Contract(error.to_string()))?;
                if &actual != expected {
                    return Err(PreparationError::Contract(format!(
                        "prediction physical source changed for {key}"
                    )));
                }
            }
        }
        let prediction = self
            .config
            .prediction
            .as_ref()
            .ok_or(PreparationError::MissingPrediction)?;
        let roots =
            (0..prediction.layers.len()).map(|depth| format!("mtp.layers.{depth}.mlp.experts"));
        let (formats, expert_banks) =
            prepare_safetensors_formats(&artifact, &self.config, &self.encoding, roots)?;
        Ok(PreparedPredictionSource {
            config: self.config,
            artifact,
            formats,
            expert_banks,
        })
    }
}
impl RecipeCatalog for SafetensorsPredictionPlan {
    fn tensor_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        self.catalog.tensor_metadata(key)
    }
}

/// Validated MTP-only source, including exact physical formats and bounded experts.
/// Geometry matching cannot authenticate training lineage; callers must pair the
/// intended checkpoint revisions. Physical source provenance is retained unchanged.
#[derive(Clone)]
pub struct PreparedPredictionSource {
    config: Config,
    artifact: SharedCheckpointSource,
    formats: ParameterFormats,
    expert_banks: BTreeMap<String, Arc<PreparedExpertBank>>,
}
impl PreparedPredictionSource {
    /// Prepares either a standalone MTP checkpoint or only the prediction role of
    /// a complete SafeTensors checkpoint. No vocabulary, target, vision or table
    /// payload is acquired. Unknown or malformed prediction tensors are rejected.
    pub fn safetensors(
        source: SharedCheckpointSource,
        config: Config,
        encoding: SafetensorsEncoding,
    ) -> Result<Self, PreparationError> {
        SafetensorsPredictionPlan::prepare_metadata(source.as_ref(), config, encoding)?.bind(source)
    }

    /// Exact prediction-only source, retaining original physical provenance.
    pub fn artifact(&self) -> &SharedCheckpointSource {
        &self.artifact
    }
}

impl PreparedTarget {
    /// Attaches independently prepared prediction weights before cold selection.
    /// Existing prediction authority cannot be replaced. The original target's
    /// embedding/output parameters are reused even when their encoding differs.
    pub fn with_prediction_source(
        mut self,
        source: PreparedPredictionSource,
    ) -> Result<Self, PreparationError> {
        if self.spec.config.prediction.is_some() {
            return Err(PreparationError::PredictionAlreadyPresent);
        }
        match_target(&self.spec.config, &source.config)?;
        self.artifact = Arc::new(eredu_checkpoint::store::CompositeCheckpointSource::new([
            self.artifact,
            source.artifact,
        ])?);
        self.formats.0.extend(source.formats.0);
        self.formats.1.extend(source.formats.1);
        self.expert_banks.extend(source.expert_banks);
        self.spec.config.prediction = source.config.prediction;
        Ok(self)
    }
}

fn match_target(target: &Config, prediction: &Config) -> Result<(), PreparationError> {
    macro_rules! same {
        ($($field:ident),+ $(,)?) => { $(
            if target.$field != prediction.$field {
                return Err(PreparationError::PredictionMismatch { field: stringify!($field) });
            }
        )+ };
    }
    same!(
        vocabulary,
        hidden_size,
        max_positions,
        norm_epsilon,
        tied_embeddings,
        layers,
        residual,
        recurrent,
        experts
    );
    // Compare normalized rotary equations, not source-container spelling/defaults.
    let (a, b) = (&target.attention, &prediction.attention);
    if (
        a.heads,
        a.kv_heads,
        a.head_dim,
        a.rotary_dim,
        a.index_heads,
        a.index_kv_heads,
        a.index_head_dim,
        a.budget,
        a.ratio,
        a.bias,
    ) != (
        b.heads,
        b.kv_heads,
        b.head_dim,
        b.rotary_dim,
        b.index_heads,
        b.index_kv_heads,
        b.index_head_dim,
        b.budget,
        b.ratio,
        b.bias,
    ) || a.rotary != b.rotary
    {
        return Err(PreparationError::PredictionMismatch { field: "attention" });
    }
    let (a, b) = (&target.ngram, &prediction.ngram);
    if (&a.layers, a.order, a.heads, a.embedding_dim, a.kernel)
        != (&b.layers, b.order, b.heads, b.embedding_dim, b.kernel)
        || target.eos.first() != prediction.eos.first()
    {
        return Err(PreparationError::PredictionMismatch { field: "ngram" });
    }
    Ok(())
}
