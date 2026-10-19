//! Portable bounded quantization recipes and cold prepared-source admission.

use crate::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::{StoreError, TensorSelection},
    WeightQuantization,
};
use std::collections::BTreeSet;

/// One dense semantic weight that will be replaced by its packed representation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BoundedQuantizationTarget {
    pub(crate) weight_name: String,
    pub(crate) scales_name: String,
    pub(crate) biases_name: Option<String>,
    pub(crate) source: DerivedWeightRecipe,
    pub(crate) affine_companion_dtype: RecipeDtype,
}

impl BoundedQuantizationTarget {
    /// Floating storage selected for affine companions.
    pub fn affine_companion_dtype(&self) -> &RecipeDtype {
        &self.affine_companion_dtype
    }

    /// Quantizes a checkpoint tensor in place under exact output identities.
    pub fn direct(
        weight_name: impl Into<String>,
        scales_name: impl Into<String>,
        biases_name: Option<impl Into<String>>,
    ) -> Result<Self, StoreError> {
        let weight_name = weight_name.into();
        Self::from_recipe(
            weight_name.clone(),
            scales_name,
            biases_name,
            DerivedWeightRecipe::source(weight_name, TensorSelection::Full),
        )
    }

    /// Quantizes a semantic recipe under exact packed runtime identities.
    pub fn from_recipe(
        weight_name: impl Into<String>,
        scales_name: impl Into<String>,
        biases_name: Option<impl Into<String>>,
        source: DerivedWeightRecipe,
    ) -> Result<Self, StoreError> {
        let weight_name = weight_name.into();
        let scales_name = scales_name.into();
        let biases_name = biases_name.map(Into::into);
        validate_output_names(&weight_name, &scales_name, biases_name.as_deref())?;
        Ok(Self {
            weight_name,
            scales_name,
            biases_name,
            source,
            affine_companion_dtype: RecipeDtype::F32,
        })
    }

    /// Selects the runtime dtype required for affine scales and biases.
    pub fn with_affine_companion_dtype(mut self, dtype: RecipeDtype) -> Result<Self, StoreError> {
        if !matches!(
            dtype,
            RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
        ) {
            return Err(quantization_error(format!(
                "bounded affine companions require F16, BF16, or F32, got {dtype:?}"
            )));
        }
        self.affine_companion_dtype = dtype;
        Ok(self)
    }

    /// Returns the packed weight's canonical logical name.
    pub fn weight_name(&self) -> &str {
        &self.weight_name
    }

    /// Returns the semantic source recipe.
    pub fn source(&self) -> &DerivedWeightRecipe {
        &self.source
    }

    /// Returns the scalar width selected for affine scale and bias outputs.
    pub fn affine_companion_bytes(&self) -> u64 {
        match self.affine_companion_dtype {
            RecipeDtype::F16 | RecipeDtype::BF16 => 2,
            RecipeDtype::F32 => 4,
            _ => unreachable!("validated bounded affine companion dtype"),
        }
    }

    /// Returns the exact packed scale tensor name.
    pub fn scales_name(&self) -> &str {
        &self.scales_name
    }

    /// Returns the exact packed affine-bias tensor name, when declared.
    pub fn biases_name(&self) -> Option<&str> {
        self.biases_name.as_deref()
    }
}

/// A validated collection of source-bounded, file-backed transformations.
#[derive(Debug, Clone)]
pub struct BoundedQuantizationPlan {
    pub(crate) quantization: WeightQuantization,
    pub(crate) max_working_set_bytes: u64,
    pub(crate) targets: Vec<BoundedQuantizationTarget>,
    pub(crate) cache: crate::prepared_cache::PreparedCheckpointCachePolicy,
}

impl BoundedQuantizationPlan {
    /// Bounded persistent cache policy for these encoded outputs.
    pub fn prepared_cache(&self) -> &crate::prepared_cache::PreparedCheckpointCachePolicy {
        &self.cache
    }

    /// Creates a non-empty plan with an explicit conversion working-set bound.
    pub fn new(
        quantization: impl Into<WeightQuantization>,
        max_working_set_bytes: u64,
        targets: impl IntoIterator<Item = BoundedQuantizationTarget>,
    ) -> Result<Self, StoreError> {
        let quantization = quantization.into();
        quantization
            .validate()
            .map_err(|e| quantization_error(e.to_string()))?;
        if quantization == WeightQuantization::NvFp4 {
            return Err(quantization_error("NVFP4 load-time encoding is unavailable; retained NVFP4 sources use direct materialization"));
        }
        if quantization.gguf_iquant().is_some() {
            return Err(quantization_error(
                "checkpoint-native GGUF IQ encodings cannot be produced by load-time quantization",
            ));
        }
        if max_working_set_bytes == 0 {
            return Err(quantization_error(
                "bounded quantization working-set bytes must be nonzero",
            ));
        }
        let mut targets = targets.into_iter().collect::<Vec<_>>();
        if targets.is_empty() {
            return Err(quantization_error(
                "bounded quantization requires at least one target",
            ));
        }
        targets.sort_by(|left, right| left.weight_name.cmp(&right.weight_name));
        let mut output_names = BTreeSet::new();
        for target in &targets {
            if quantization.has_biases() && target.biases_name.is_none() {
                return Err(quantization_error(format!(
                    "bounded affine quantization target {:?} has no affine-bias identity",
                    target.weight_name
                )));
            }
            for name in output_names_for(target, quantization)? {
                if !output_names.insert(name.clone()) {
                    return Err(quantization_error(format!(
                        "bounded quantization output {name:?} is produced more than once"
                    )));
                }
            }
        }
        Ok(Self {
            quantization,
            max_working_set_bytes,
            targets,
            cache: Default::default(),
        })
    }

    /// Selects filesystem location and total retained prepared-checkpoint bytes.
    pub fn with_prepared_cache(
        mut self,
        policy: crate::prepared_cache::PreparedCheckpointCachePolicy,
    ) -> Self {
        self.cache = policy;
        self
    }

    /// Returns the final packed encoding.
    pub const fn quantization(&self) -> WeightQuantization {
        self.quantization
    }

    /// Returns the admitted conversion working-set bound.
    pub const fn max_working_set_bytes(&self) -> u64 {
        self.max_working_set_bytes
    }

    /// Returns targets in deterministic logical-name order.
    pub fn targets(&self) -> &[BoundedQuantizationTarget] {
        &self.targets
    }
}

pub(crate) fn output_names_for(
    target: &BoundedQuantizationTarget,
    quantization: WeightQuantization,
) -> Result<Vec<String>, StoreError> {
    let mut names = vec![target.weight_name.clone(), target.scales_name.clone()];
    if quantization.has_biases() {
        names.push(
            target
                .biases_name
                .clone()
                .expect("bounded plan validated affine-bias identity"),
        );
    }
    Ok(names)
}

pub(crate) fn validate_output_names(
    weight_name: &str,
    scales_name: &str,
    biases_name: Option<&str>,
) -> Result<(), StoreError> {
    if weight_name.trim().is_empty()
        || scales_name.trim().is_empty()
        || biases_name.is_some_and(|name| name.trim().is_empty())
    {
        return Err(quantization_error(
            "bounded quantization output identities must not be empty",
        ));
    }
    if weight_name == scales_name
        || biases_name.is_some_and(|name| name == weight_name || name == scales_name)
    {
        return Err(quantization_error(
            "bounded quantization output identities must be distinct",
        ));
    }
    Ok(())
}

fn quantization_error(message: impl Into<String>) -> StoreError {
    StoreError::InvalidPreparation(message.into())
}

use crate::{
    store::{
        CheckpointLease, CheckpointSource, SharedCheckpointSource, TensorMetadata,
        TensorReadRequest, WeightStoreDiagnostics,
    },
    StoredDtype,
};
use std::{collections::BTreeMap, sync::OnceLock};

impl BoundedQuantizationPlan {
    /// Exact encoded output geometry, inferred without reading source payloads.
    pub fn output_metadata(
        &self,
        source: &dyn CheckpointSource,
    ) -> Result<BTreeMap<String, TensorMetadata>, StoreError> {
        let mut outputs = BTreeMap::new();
        let source_keys = source.source_keys().into_iter().collect::<BTreeSet<_>>();
        for target in &self.targets {
            let input = target
                .source
                .infer(source)
                .map_err(|e| quantization_error(e.to_string()))?;
            if input.shape.len() < 2
                || !matches!(
                    input.dtype,
                    RecipeDtype::F32 | RecipeDtype::F16 | RecipeDtype::BF16
                )
            {
                return Err(quantization_error(format!(
                    "quantization requires a floating matrix: {}",
                    target.weight_name
                )));
            }
            for name in std::iter::once(&target.scales_name).chain(target.biases_name.iter()) {
                if source_keys.contains(name) {
                    return Err(quantization_error(format!(
                        "quantization companion already exists: {name}; implicit transcoding is unsupported"
                    )));
                }
            }
            outputs.extend(target.output_metadata(self.quantization, &input.shape)?);
        }
        Ok(outputs)
    }
}

impl BoundedQuantizationTarget {
    /// Deterministic output identities for the selected encoding.
    pub fn output_names(
        &self,
        quantization: WeightQuantization,
    ) -> Result<Vec<String>, StoreError> {
        output_names_for(self, quantization)
    }
    /// Exact encoded geometry shared by cold admission and native file writers.
    pub fn output_metadata(
        &self,
        quantization: WeightQuantization,
        input_shape: &[usize],
    ) -> Result<BTreeMap<String, TensorMetadata>, StoreError> {
        quantization
            .validate()
            .map_err(|e| quantization_error(e.to_string()))?;
        if quantization.gguf_iquant().is_some() {
            return Err(quantization_error(
                "load-time quantization cannot produce GGUF IQ encodings",
            ));
        }
        let columns = input_shape.last().copied().unwrap_or(0);
        let group = quantization.group_size() as usize;
        if input_shape.len() < 2
            || columns == 0
            || columns % group != 0
            || columns % 32 != 0
            || input_shape.contains(&0)
        {
            return Err(quantization_error(format!(
                "quantization input geometry is incompatible: {}",
                self.weight_name
            )));
        }
        let dtype = match self.affine_companion_dtype {
            RecipeDtype::F16 => StoredDtype::F16,
            RecipeDtype::BF16 => StoredDtype::BF16,
            RecipeDtype::F32 => StoredDtype::F32,
            _ => return Err(quantization_error("invalid affine companion dtype")),
        };
        let mut outputs = BTreeMap::new();
        let mut insert = |name: &str,
                          columns: usize,
                          dtype: StoredDtype,
                          width: u64|
         -> Result<(), StoreError> {
            let mut shape = input_shape.to_vec();
            *shape.last_mut().unwrap() = columns;
            let bytes = shape
                .iter()
                .try_fold(width, |n, &d| n.checked_mul(d as u64))
                .ok_or_else(|| quantization_error("quantized byte count overflow"))?;
            outputs.insert(
                name.to_owned(),
                TensorMetadata {
                    name: name.to_owned(),
                    logical_shape: shape.clone(),
                    physical_shape: shape,
                    stored_dtype: dtype,
                    encoded_byte_len: bytes,
                    backing_shard: None,
                },
            );
            Ok(())
        };
        insert(
            &self.weight_name,
            columns
                .checked_mul(quantization.bits() as usize)
                .ok_or_else(|| quantization_error("packed width overflow"))?
                / 32,
            StoredDtype::U32,
            4,
        )?;
        let mxfp = matches!(quantization, WeightQuantization::MxFp4);
        insert(
            &self.scales_name,
            columns / group,
            if mxfp { StoredDtype::U8 } else { dtype.clone() },
            if mxfp {
                1
            } else {
                self.affine_companion_bytes()
            },
        )?;
        if quantization.has_biases() {
            insert(
                self.biases_name
                    .as_deref()
                    .ok_or_else(|| quantization_error("affine target has no biases identity"))?,
                columns / group,
                dtype,
                self.affine_companion_bytes(),
            )?;
        }
        Ok(outputs)
    }
}

/// Immutable cold catalog with an explicitly pending bounded transformation.
/// A native mechanism publishes exact encoded storage before any transformed read.
/// Original source guards and metadata remain retained throughout preparation.
pub struct PreparedQuantizedSource {
    source: SharedCheckpointSource,
    plan: BoundedQuantizationPlan,
    outputs: BTreeMap<String, TensorMetadata>,
    ready: OnceLock<SharedCheckpointSource>,
    recipes: crate::recipe::RecipeInferenceCache,
}

impl PreparedQuantizedSource {
    /// Admits output shapes and retained source authority without acquiring payloads.
    pub fn new(
        source: SharedCheckpointSource,
        plan: BoundedQuantizationPlan,
    ) -> Result<Self, StoreError> {
        let outputs = plan.output_metadata(source.as_ref())?;
        Ok(Self {
            source,
            plan,
            outputs,
            ready: OnceLock::new(),
            recipes: Default::default(),
        })
    }
}

impl CheckpointSource for PreparedQuantizedSource {
    fn has_direct_encoded_leases(&self) -> bool {
        self.source.has_direct_encoded_leases()
            && self
                .ready
                .get()
                .is_some_and(|ready| ready.has_direct_encoded_leases())
    }

    fn retained_payload_bytes_upper_bound(&self) -> Option<u64> {
        // Publication retains both the original and the prepared source.
        // Before publication the native materializer chooses the output owner.
        self.source
            .retained_payload_bytes_upper_bound()?
            .checked_add(self.ready.get()?.retained_payload_bytes_upper_bound()?)
    }

    fn preparation_working_set_bytes(&self) -> u64 {
        self.plan.max_working_set_bytes
    }
    fn preparation(&self) -> Option<(SharedCheckpointSource, BoundedQuantizationPlan)> {
        self.ready
            .get()
            .is_none()
            .then(|| (self.source.clone(), self.plan.clone()))
    }
    fn publish_preparation(&self, ready: SharedCheckpointSource) -> Result<(), StoreError> {
        for (name, expected) in &self.outputs {
            let actual = ready.source_metadata(name)?;
            if actual.logical_shape != expected.logical_shape
                || actual.physical_shape != expected.physical_shape
                || actual.stored_dtype != expected.stored_dtype
                || actual.encoded_byte_len != expected.encoded_byte_len
            {
                return Err(quantization_error(format!(
                    "prepared output differs from admitted catalog: {name}"
                )));
            }
        }
        self.ready
            .set(ready)
            .map_err(|_| quantization_error("prepared source is already published"))
    }
    fn prepared_cache_policy(
        &self,
    ) -> Option<&crate::prepared_cache::PreparedCheckpointCachePolicy> {
        Some(&self.plan.cache)
    }
    fn recipe_cache(&self) -> Option<&crate::recipe::RecipeInferenceCache> {
        Some(&self.recipes)
    }
    fn source_keys(&self) -> Vec<String> {
        self.source
            .source_keys()
            .into_iter()
            .chain(self.outputs.keys().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.outputs
            .get(key)
            .cloned()
            .map_or_else(|| self.source.source_metadata(key), Ok)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        if self.outputs.contains_key(&request.key) {
            self.ready
                .get()
                .ok_or_else(|| StoreError::PreparationRequired(request.key.clone()))?
                .acquire_lease(request)
        } else {
            self.source.acquire_lease(request)
        }
    }
    fn prepare_encoded_read(
        &self,
        requests: &[TensorReadRequest],
    ) -> Result<Option<crate::store::EncodedReadBatch>, StoreError> {
        if requests.iter().all(|r| self.outputs.contains_key(&r.key)) {
            if let Some(ready) = self.ready.get() {
                ready.prepare_encoded_read(requests)
            } else {
                Ok(None)
            }
        } else if requests.iter().all(|r| !self.outputs.contains_key(&r.key)) {
            self.source.prepare_encoded_read(requests)
        } else {
            Ok(None)
        }
    }
    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.source.unclaimed_checkpoint_keys()
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.ready
            .get()
            .unwrap_or(&self.source)
            .source_diagnostics()
    }
    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.outputs.contains_key(key) || self.source.is_authoritative_materialized_key(key)
    }
    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.source.is_checkpoint_contract_resolved()
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<crate::store::TensorSourceProvenance, StoreError> {
        if self.outputs.contains_key(key) {
            let metadata = self.source_metadata(key)?;
            Ok(crate::store::TensorSourceProvenance {
                catalog_key: key.into(),
                physical_tensor: key.into(),
                output: key.into(),
                backing_shard: None,
                source_encoding: crate::SourceTensorEncoding::Safetensors(metadata.stored_dtype),
            })
        } else {
            self.source.source_provenance(key)
        }
    }
    fn materialized_source_keys(&self) -> Vec<String> {
        self.ready
            .get()
            .map_or_else(Vec::new, |s| s.materialized_source_keys())
    }
    fn materialized_source_shards(&self) -> Vec<std::path::PathBuf> {
        self.ready
            .get()
            .map_or_else(Vec::new, |s| s.materialized_source_shards())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryWeightStore;
    use safetensors::tensor::Dtype;

    #[test]
    fn prepared_quantization_requires_exact_publication_and_preserves_restricted_reads() {
        let dense: SharedCheckpointSource = std::sync::Arc::new(
            MemoryWeightStore::from_safetensors([
                (
                    "linear.weight".into(),
                    Dtype::F32,
                    vec![2, 64],
                    vec![0; 512],
                ),
                ("norm.weight".into(), Dtype::F32, vec![2], vec![0; 8]),
            ])
            .unwrap(),
        );
        let plan = BoundedQuantizationPlan::new(
            crate::AffineQuantization::new(32, 8).unwrap(),
            4096,
            [BoundedQuantizationTarget::direct(
                "linear.weight",
                "linear.scales",
                Some("linear.biases"),
            )
            .unwrap()],
        )
        .unwrap();
        let cold = PreparedQuantizedSource::new(dense.clone(), plan).unwrap();
        let expected = cold.source_metadata("linear.weight").unwrap();
        assert_eq!(expected.logical_shape, [2, 16]);
        assert_eq!(expected.encoded_byte_len, 128);
        assert_eq!(
            cold.source_metadata("linear.scales").unwrap().logical_shape,
            [2, 2]
        );
        assert_eq!(dense.source_diagnostics().unwrap().physical_reads, 0);
        assert!(matches!(
            cold.acquire_lease("linear.weight".into()),
            Err(StoreError::PreparationRequired(_))
        ));
        let storage = |bad| -> SharedCheckpointSource {
            std::sync::Arc::new(
                MemoryWeightStore::from_safetensors([
                    (
                        "linear.weight".into(),
                        Dtype::U32,
                        if bad { vec![2, 8] } else { vec![2, 16] },
                        vec![0; if bad { 64 } else { 128 }],
                    ),
                    ("linear.scales".into(), Dtype::F32, vec![2, 2], vec![0; 16]),
                    ("linear.biases".into(), Dtype::F32, vec![2, 2], vec![0; 16]),
                ])
                .unwrap(),
            )
        };
        assert!(cold.publish_preparation(storage(true)).is_err());
        assert!(cold.preparation().is_some());
        cold.publish_preparation(storage(false)).unwrap();
        assert!(cold.preparation().is_none());
        assert_eq!(cold.source_metadata("linear.weight").unwrap(), expected);
        assert!(cold.acquire_lease("linear.weight".into()).is_ok());
        assert!(cold.acquire_lease("norm.weight".into()).is_ok());
        assert!(cold.publish_preparation(storage(false)).is_err());
        let source: SharedCheckpointSource = std::sync::Arc::new(cold);
        let restricted = crate::store::RestrictedCheckpointSource::including(
            source,
            "selected",
            BTreeSet::from(["linear.scales".into()]),
        )
        .unwrap();
        assert!(matches!(
            restricted.acquire_lease("linear.biases".into()),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        assert!(restricted.acquire_lease("linear.scales".into()).is_ok());
    }
}
