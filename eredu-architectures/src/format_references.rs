//! Resolves storage extensions against typed owners and explicit checkpoint sources.
use eredu_checkpoint::{
    schema::{SafetensorsCheckpointPlan, TensorRole},
    LinearFormat,
};
use eredu_nn::{DeclareLogicalParameters, Error, LogicalParameterDeclaration};
use std::collections::BTreeMap;

#[derive(Debug, thiserror::Error)]
enum ReferenceError {
    #[error("unresolved parameter format reference {0}")]
    Unresolved(String),
    #[error("format {format:?} is incompatible with declared parameter or source {identity}")]
    Incompatible {
        identity: String,
        format: LinearFormat,
    },
    #[error("invalid parameter format source extension: {0}")]
    Extension(String),
    #[error("conflicting format source identity {0}")]
    Conflict(String),
    #[error("parameter {identity} and its sharing target {target} declare incompatible encodings")]
    Sharing { identity: String, target: String },
    #[error("parameter format reference {0} does not name a declared matrix")]
    NotMatrix(String),
}

/// Canonical references borrow logical owners; other references require explicit
/// source constraints. Companions and allowed-but-unowned artifact keys are not primaries.
pub(crate) fn validate<'a>(
    declaration: &impl DeclareLogicalParameters,
    references: impl IntoIterator<Item = (&'a str, Option<LinearFormat>)>,
    source_references: impl FnOnce() -> Result<Vec<(String, LinearFormat)>, String>,
    extensions: impl FnOnce() -> Result<SafetensorsCheckpointPlan, String>,
) -> Result<(), Error> {
    resolve(
        || declaration.logical_parameters(),
        references,
        false,
        source_references,
        || extensions().map(Some),
    )
}

/// Resolves a family whose format references name logical slots exclusively.
pub(crate) fn validate_logical<'a>(
    declaration: &impl DeclareLogicalParameters,
    references: impl IntoIterator<Item = (&'a str, Option<LinearFormat>)>,
) -> Result<(), Error> {
    resolve(
        || declaration.logical_parameters(),
        references,
        false,
        || Ok(Vec::new()),
        || Ok(None),
    )
}

/// Resolves matrix-only directives within an already projected ownership scope.
pub(crate) fn validate_matrices<'a, 'p>(
    parameters: Vec<LogicalParameterDeclaration<'p>>,
    references: impl IntoIterator<Item = (&'a str, Option<LinearFormat>)>,
) -> Result<(), Error> {
    validate_matrix_sources(parameters, references, || Ok(Vec::new()))
}

/// Resolves matrix directives and explicit physical source projections of their owners.
pub(crate) fn validate_matrix_sources<'a, 'p>(
    parameters: Vec<LogicalParameterDeclaration<'p>>,
    references: impl IntoIterator<Item = (&'a str, Option<LinearFormat>)>,
    sources: impl FnOnce() -> Result<Vec<(String, LinearFormat)>, String>,
) -> Result<(), Error> {
    resolve(|| Ok(parameters), references, true, sources, || Ok(None))
}

fn resolve<'a, 'p>(
    parameters: impl FnOnce() -> Result<Vec<LogicalParameterDeclaration<'p>>, Error>,
    references: impl IntoIterator<Item = (&'a str, Option<LinearFormat>)>,
    matrix_only: bool,
    source_references: impl FnOnce() -> Result<Vec<(String, LinearFormat)>, String>,
    extensions: impl FnOnce() -> Result<Option<SafetensorsCheckpointPlan>, String>,
) -> Result<(), Error> {
    let references = references.into_iter().collect::<Vec<_>>();
    if references.is_empty() {
        return Ok(());
    }
    let slots = parameters()?
        .into_iter()
        .map(|slot| (slot.parameter.id.as_str(), slot))
        .collect::<BTreeMap<_, _>>();
    for slot in slots.values() {
        if let Some(target) = slot
            .parameter
            .alias_of
            .as_ref()
            .and_then(|target| slots.get(target.as_str()))
        {
            if let (Some(format), Some(destination)) = (slot.linear_format, target.linear_format) {
                if format.encoding() != destination.encoding() {
                    return Err(Error::backend_source(ReferenceError::Sharing {
                        identity: slot.parameter.id.to_string(),
                        target: target.parameter.id.to_string(),
                    }));
                }
            }
        }
    }
    let mut source_references = Some(source_references);
    let mut declared_sources = None;
    let mut extensions = Some(extensions);
    let mut sources = None;
    for (name, requested) in references {
        if let Some(format) = requested {
            format.validate().map_err(Error::backend_source)?;
        }
        if let Some(slot) = slots.get(name) {
            if matrix_only && slot.linear_format.is_none() {
                return Err(Error::backend_source(ReferenceError::NotMatrix(
                    name.into(),
                )));
            }
            if let Some(format) = requested {
                let declared = slot
                    .linear_format
                    .map(|f| f.encoding())
                    .unwrap_or(LinearFormat::Dense);
                if format != declared {
                    return Err(Error::backend_source(ReferenceError::Incompatible {
                        identity: name.into(),
                        format,
                    }));
                }
            }
            continue;
        }
        if declared_sources.is_none() {
            let mut resolved = BTreeMap::new();
            for (name, format) in source_references
                .take()
                .expect("source references are prepared once")()
            .map_err(|reason| Error::backend_source(ReferenceError::Extension(reason)))?
            {
                format.validate().map_err(Error::backend_source)?;
                if resolved
                    .insert(name.clone(), format)
                    .is_some_and(|old| old != format)
                {
                    return Err(Error::backend_source(ReferenceError::Conflict(name)));
                }
            }
            declared_sources = Some(resolved);
        }
        if let Some(declared) = declared_sources
            .as_ref()
            .expect("prepared source references")
            .get(name)
        {
            if let Some(format) = requested.filter(|f| f != declared) {
                return Err(Error::backend_source(ReferenceError::Incompatible {
                    identity: name.into(),
                    format,
                }));
            }
            continue;
        }
        if sources.is_none() {
            sources = Some(
                extensions
                    .take()
                    .expect("source extensions are prepared once")()
                .map_err(|reason| Error::backend_source(ReferenceError::Extension(reason)))?,
            );
        }
        let Some(sources) = sources.as_ref().expect("prepared source extensions") else {
            return Err(Error::backend_source(ReferenceError::Unresolved(
                name.into(),
            )));
        };
        let mut found = false;
        let mut compatible = false;
        for source in sources.common_tensors.iter().chain(
            sources
                .layout_groups
                .iter()
                .flat_map(|group| &group.variants)
                .flat_map(|variant| &variant.tensors),
        ) {
            if source.linear_companion.is_some()
                || matches!(
                    source.role,
                    TensorRole::Companion | TensorRole::ArtifactMetadata
                )
            {
                continue;
            }
            if source.key == name || source.aliases.iter().any(|alias| alias == name) {
                found = true;
                compatible |= requested.is_none_or(|format| {
                    format == source.linear_format.unwrap_or(LinearFormat::Dense)
                });
            }
        }
        if !found {
            return Err(Error::backend_source(ReferenceError::Unresolved(
                name.into(),
            )));
        }
        if !compatible {
            return Err(Error::backend_source(ReferenceError::Incompatible {
                identity: name.into(),
                format: requested.expect("incompatible encoding is present"),
            }));
        }
    }
    Ok(())
}

/// Joins a split physical matrix encoding onto its typed executable owner.
/// The split is a format extension; it does not create a new logical parameter.
pub(crate) fn rehome_split_matrix_format(
    formats: &mut std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    owner: &eredu_nn::ParameterId,
    sources: [String; 2],
) -> Result<(), String> {
    let [gate, up] = sources;
    if (formats.contains_key(&gate) || formats.contains_key(&up))
        && formats.get(&gate) != formats.get(&up)
    {
        return Err(format!("split expert gate/up formats disagree for {owner}"));
    }
    rehome_matrix_format(formats, owner, &gate)?;
    formats.remove(&up);
    Ok(())
}

/// Resolves a physical matrix format reference onto its declared owner.
pub(crate) fn rehome_matrix_format(
    formats: &mut std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
    owner: &eredu_nn::ParameterId,
    source: &str,
) -> Result<(), String> {
    if let Some(format) = formats.get(source).copied() {
        if formats
            .get(owner.as_str())
            .is_some_and(|selected| *selected != format)
        {
            return Err(format!(
                "source format conflicts with declared owner {owner}"
            ));
        }
        formats.remove(source);
        formats.insert(owner.to_string(), format);
    }
    Ok(())
}
