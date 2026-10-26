//! Exact checkpoint authority for architecture-admitted execution requirements.
use crate::{ReplicatedTextParameterPresence, ReplicatedTextPhysicalSource, ReplicatedTextRequirements};
use std::{collections::BTreeSet, sync::Arc};

/// Checkpoint provenance or recipe facts disagree with the admitted handoff.
#[derive(Debug, thiserror::Error)]
pub enum SourceHandoffError {
    /// An admitted physical source is unavailable.
    #[error("selected source catalog key {key:?} is unavailable: {source}")]
    Provenance {
        /// Admitted catalog identity.
        key: String,
        /// Original checkpoint failure.
        #[source]
        source: eredu_checkpoint::store::StoreError,
    },
    /// Metadata for an admitted physical source is unavailable.
    #[error("selected source metadata for {key:?} is unavailable: {source}")]
    Metadata {
        /// Admitted catalog identity.
        key: String,
        /// Original checkpoint failure.
        #[source]
        source: eredu_checkpoint::store::StoreError,
    },
    /// An admitted derivation cannot be resolved against this source.
    #[error("selected recipe for {target:?} is invalid: {source}")]
    Recipe {
        /// Canonical recipe target.
        target: String,
        /// Original neutral recipe failure.
        #[source]
        source: eredu_checkpoint::recipe::RecipeError,
    },
    /// Restriction or physical metadata access failed.
    #[error(transparent)]
    Storage(#[from] eredu_checkpoint::store::StoreError),
    /// Exact admitted geometry, encoding, provenance or recipe membership differs.
    #[error("{0}")]
    Invalid(String),
}
impl From<String> for SourceHandoffError {
    fn from(reason: String) -> Self {
        Self::Invalid(reason)
    }
}

/// Source authority being validated; restricted primary views must not acquire auxiliaries.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum SourceHandoffScope {
    /// Only ordinary target parameters; auxiliary sources remain inaccessible.
    Primary,
    /// Ordinary and admitted auxiliary parameters share this view.
    Complete,
}

/// Projects an already admitted artifact onto the exact selected parameter role.
/// Other roles retain their own source leases; native binding must not gain their
/// catalog authority merely because they share an underlying checkpoint file.
pub fn restrict_checkpoint_handoff(
    requirements: &ReplicatedTextRequirements,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    scope: SourceHandoffScope,
) -> Result<eredu_checkpoint::store::SharedCheckpointSource, SourceHandoffError> {
    validate_checkpoint_handoff(requirements, store.as_ref(), scope)?;
    let keys = requirements
        .parameters()
        .iter()
        .chain(
            requirements
                .auxiliary_parameters()
                .iter()
                .filter(|_| scope == SourceHandoffScope::Complete),
        )
        .flat_map(|parameter| parameter.sources().iter().cloned())
        .collect::<BTreeSet<_>>();
    Ok(Arc::new(
        eredu_checkpoint::store::RestrictedCheckpointSource::including(
            store,
            "declared-execution-parameters",
            keys,
        )?,
    ))
}

/// Validates admitted provenance, geometry and recipe inputs without acquiring payloads.
/// Primary validation excludes auxiliary declarations and their recipes.
pub fn validate_checkpoint_handoff(
    requirements: &ReplicatedTextRequirements,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    scope: SourceHandoffScope,
) -> Result<(), SourceHandoffError> {
    for parameter in requirements.parameters().iter().chain(
        requirements
            .auxiliary_parameters()
            .iter()
            .filter(|_| scope == SourceHandoffScope::Complete),
    ) {
        let derived = matches!(
            parameter.presence(),
            ReplicatedTextParameterPresence::Derived { .. }
        );
        let mut lowering_key = None;
        let mut admitted_keys = BTreeSet::new();
        for source in parameter.physical_sources() {
            let key = source.catalog_key();
            let actual =
                store
                    .source_provenance(key)
                    .map_err(|source| SourceHandoffError::Provenance {
                        key: key.into(),
                        source,
                    })?;
            let metadata =
                store
                    .source_metadata(key)
                    .map_err(|source| SourceHandoffError::Metadata {
                        key: key.into(),
                        source,
                    })?;
            if actual.catalog_key != key
                || actual.physical_tensor != source.tensor()
                || actual.backing_shard.as_deref() != Some(source.shard())
                || actual.output != source.output()
                || &actual.source_encoding != source.source_encoding()
                || metadata.encoded_byte_len != source.encoded_byte_len()
            {
                return Err(format!(
                    "selected physical provenance for catalog key {key:?} differs from the admitted store"
                ).into());
            }
            if !derived
                && parameter
                    .source_encoding()
                    .is_some_and(|expected| expected != source.source_encoding())
            {
                return Err(format!(
                    "selected physical encoding for {:?} differs from the admitted store",
                    parameter.name()
                )
                .into());
            }
            lowering_key.get_or_insert(key);
            admitted_keys.insert(key);
        }
        let selected_keys = parameter
            .sources()
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if parameter.has_lowering_source() && selected_keys != admitted_keys {
            return Err(format!(
                "selected source catalog for {:?} differs from admitted physical provenance",
                parameter.name()
            )
            .into());
        }
        if !derived {
            if let Some(physical_shape) = parameter.physical_shape() {
                let key = lowering_key.ok_or_else(|| {
                    format!(
                        "selected physical facts for {:?} have no provenance",
                        parameter.name()
                    )
                })?;
                let metadata = store.source_metadata(key)?;
                if physical_shape != metadata.physical_shape.as_slice() {
                    return Err(format!(
                        "selected physical geometry for {:?} differs from the admitted store",
                        parameter.name()
                    )
                    .into());
                }
            }
        }
    }
    for (target, recipe) in requirements.derived_recipes() {
        if scope == SourceHandoffScope::Primary
            && requirements
                .auxiliary_parameters()
                .iter()
                .any(|parameter| parameter.name() == target)
        {
            continue;
        }
        let parameter = requirements
            .parameters()
            .iter()
            .chain(requirements.auxiliary_parameters())
            .find(|parameter| parameter.name() == target)
            .ok_or_else(|| format!("selected recipe target {target:?} is undeclared"))?;
        if !matches!(
            parameter.presence(),
            ReplicatedTextParameterPresence::Derived { .. }
        ) {
            return Err(
                format!("selected recipe target {target:?} is not declared as derived").into(),
            );
        }
        let inferred = recipe
            .infer(store)
            .map_err(|source| SourceHandoffError::Recipe {
                target: target.clone(),
                source,
            })?;
        let admitted_output = requirements
            .derived_recipe_outputs()
            .get(target)
            .expect("validated recipe contract has one output per target");
        if &inferred != admitted_output {
            return Err(format!(
                "selected recipe output for {target:?} differs from admitted metadata"
            )
            .into());
        }
        let admitted = parameter
            .physical_sources()
            .iter()
            .map(ReplicatedTextPhysicalSource::catalog_key)
            .collect::<BTreeSet<_>>();
        let consumed = recipe.source_keys().into_iter().collect::<BTreeSet<_>>();
        if consumed != admitted {
            return Err(format!(
                "selected recipe sources for {target:?} differ from admitted provenance"
            )
            .into());
        }
    }
    Ok(())
}
