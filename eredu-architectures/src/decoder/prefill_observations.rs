//! Explicit row semantics of the common decoder's actual hooks.
use super::module_metadata::ModuleMetadata;
use super::*;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::layered::{
    PrefillObservationDeclaration as Declaration, PrefillReadoutStage as Stage,
};

pub(super) fn declarations<C: Config>(
    config: &C,
    causal_blocks: bool,
    mut residual: impl FnMut(usize) -> &'static str,
    mut components: impl FnMut(&str, usize, &mut Vec<Declaration>) -> Result<(), Error>,
    metadata: Option<&WorkspaceContext>,
) -> Result<Vec<Declaration>, Error> {
    let destination = ModuleMetadata::destination(metadata);
    destination.borrowed_controls(&components)?;
    destination.borrowed_controls(&residual)?;
    destination.controls::<(
        &C,
        bool,
        Option<&WorkspaceContext>,
        usize,
        Vec<Declaration>,
        String,
        Option<eredu_runtime::RoutedObservationPoints>,
        Result<Vec<Declaration>, Error>,
    )>()?;
    let layers = if causal_blocks {
        usize::try_from(config.num_hidden_layers()).map_err(|cause| destination.source(cause))?
    } else {
        0
    };
    let mut declarations = ordinary_prefill_observation_declarations(
        (0..layers).map(|index| {
            destination.text(format_args!("{}.layers.{index}", config.parameter_root()))
        }),
        causal_blocks,
        metadata,
    )?;
    for index in 0..layers {
        let path = destination.text(format_args!("{}.layers.{index}", config.parameter_root()))?;
        append_decoder_component_prefill_observations(
            &mut declarations,
            &path,
            residual(index),
            metadata,
        )?;
        components(&path, index, &mut declarations)?;
        if let Some(points) = config.routed_observation_points(&path, index, metadata)? {
            append_routed_prefill_observations(&mut declarations, &points, metadata)?;
        }
    }
    Ok(declarations)
}

/// The shared TransformerBlock keeps these values in [batch, new-token, channel]
/// order. Attention has already applied its causal prefix and flattened heads;
/// normalization, output projection and residual addition preserve each row.
fn append_decoder_component_prefill_observations(
    declarations: &mut Vec<Declaration>,
    unit: &str,
    residual: &str,
    metadata: Option<&WorkspaceContext>,
) -> Result<(), Error> {
    let destination = ModuleMetadata::destination(metadata);
    destination.controls::<(
        &mut Vec<Declaration>,
        &str,
        &str,
        Option<&WorkspaceContext>,
        std::array::IntoIter<&str, 8>,
        std::array::IntoIter<&str, 2>,
        String,
        Declaration,
    )>()?;
    for component in [
        "attention.input",
        "attention.channels",
        "attention.write",
        "attention.output",
        "attention.residual",
        "feed_forward.input",
        residual,
        "feed_forward.residual",
    ] {
        for suffix in ["", ".effective"] {
            let path = destination.text(format_args!("{unit}.{component}{suffix}"))?;
            destination.push(
                declarations,
                Declaration::causal_ordinary_text(path, 1, Stage::BeforeReadout),
            )?;
        }
    }
    // The projection-input hook reports the actual selected affine input once;
    // it has no separate effective hook and does not reconstruct normalized data.
    let path = destination.text(format_args!("{unit}.attention.write_input"))?;
    destination.push(
        declarations,
        Declaration::causal_ordinary_text(path, 1, Stage::BeforeReadout),
    )
}

/// Actual row-local products and writes of a caller-identified dense MLP.
pub(crate) fn append_dense_component_prefill_observations(
    declarations: &mut Vec<Declaration>,
    component: &str,
    metadata: Option<&WorkspaceContext>,
) -> Result<(), Error> {
    let destination = ModuleMetadata::destination(metadata);
    destination.controls::<(
        &mut Vec<Declaration>,
        &str,
        Option<&WorkspaceContext>,
        std::array::IntoIter<&str, 5>,
        String,
        Declaration,
    )>()?;
    for suffix in [
        "units",
        "units.effective",
        "write",
        "write.effective",
        "write_input",
    ] {
        let path = destination.text(format_args!("{component}.{suffix}"))?;
        destination.push(
            declarations,
            Declaration::causal_ordinary_text(path, 1, Stage::BeforeReadout),
        )?;
    }
    Ok(())
}

/// Sparse companions of the same bank identities used by actual execution.
pub(crate) fn append_routed_prefill_observations(
    declarations: &mut Vec<Declaration>,
    points: &eredu_runtime::RoutedObservationPoints,
    metadata: Option<&WorkspaceContext>,
) -> Result<(), Error> {
    let destination = ModuleMetadata::destination(metadata);
    destination.controls::<(
        &mut Vec<Declaration>,
        &eredu_runtime::RoutedObservationPoints,
        Option<&WorkspaceContext>,
        Result<(), Error>,
    )>()?;
    let iter = points.iter();
    destination.borrowed_controls(&iter)?;
    for (_, point) in iter {
        append_routed_prefill_path(declarations, point.path(), metadata)?;
    }
    Ok(())
}

/// Original/effective hooks of one architecture-declared causal bank.
pub(crate) fn append_routed_prefill_path(
    declarations: &mut Vec<Declaration>,
    routing: &str,
    metadata: Option<&WorkspaceContext>,
) -> Result<(), Error> {
    let destination = ModuleMetadata::destination(metadata);
    destination.controls::<(
        &mut Vec<Declaration>,
        &str,
        Option<&WorkspaceContext>,
        std::array::IntoIter<&str, 2>,
        String,
        Declaration,
    )>()?;
    for suffix in ["units", "units.effective"] {
        let path = destination.text(format_args!("{routing}.{suffix}"))?;
        destination.push(declarations, Declaration::causal_routed_units(path))?;
    }
    Ok(())
}

/// Outer hooks of an actual ordinary target invocation and its causal proof.
pub(crate) fn ordinary_prefill_observation_declarations(
    unit_paths: impl IntoIterator<Item = Result<String, Error>>,
    causal_blocks: bool,
    metadata: Option<&WorkspaceContext>,
) -> Result<Vec<Declaration>, Error> {
    outer(unit_paths, causal_blocks, false, metadata)
}

/// Decoder hooks of an authenticated retained-media source. Encoder axes and
/// intervention equivalence remain independent architecture obligations.
pub(crate) fn media_prefill_observation_declarations(
    unit_paths: impl IntoIterator<Item = Result<String, Error>>,
    metadata: Option<&WorkspaceContext>,
) -> Result<Vec<Declaration>, Error> {
    outer(unit_paths, true, true, metadata)
}

fn outer(
    unit_paths: impl IntoIterator<Item = Result<String, Error>>,
    causal_blocks: bool,
    media: bool,
    metadata: Option<&WorkspaceContext>,
) -> Result<Vec<Declaration>, Error> {
    let destination = ModuleMetadata::destination(metadata);
    destination.borrowed_controls(&unit_paths)?;
    destination.controls::<(
        bool,
        bool,
        Option<&WorkspaceContext>,
        Vec<Declaration>,
        String,
        String,
        Stage,
        std::array::IntoIter<&str, 2>,
        std::array::IntoIter<&str, 3>,
        Result<Vec<Declaration>, Error>,
    )>()?;
    let mut entries = destination.vector(0)?;
    let mut add = |path: String, stage| {
        let declaration = if media {
            Declaration::prepared_media_decoder(path, 1, stage)
        } else {
            Declaration::causal_ordinary_text(path, 1, stage)
        };
        destination.push(&mut entries, declaration)
    };
    destination.borrowed_controls(&add)?;
    for path in ["readout.embedding", "readout.embedding.effective"] {
        add(
            destination.text(format_args!("{path}"))?,
            Stage::BeforeReadout,
        )?;
    }
    if !causal_blocks {
        return Ok(entries);
    }
    let iter = unit_paths.into_iter();
    destination.borrowed_controls(&iter)?;
    for unit in iter {
        let unit = unit?;
        for suffix in ["input", "output"] {
            let path = destination.text(format_args!("{unit}.{suffix}"))?;
            let effective = destination.text(format_args!("{path}.effective"))?;
            add(path, Stage::BeforeReadout)?;
            add(effective, Stage::BeforeReadout)?;
        }
    }
    for path in ["readout.residual", "readout.normalized"] {
        add(
            destination.text(format_args!("{path}"))?,
            Stage::ReadoutInput,
        )?;
        add(
            destination.text(format_args!("{path}.effective"))?,
            Stage::ReadoutInput,
        )?;
    }
    if !media {
        add(
            destination.text(format_args!("readout.projection_input"))?,
            Stage::ReadoutInput,
        )?;
    }
    for path in [
        "readout.linear",
        "readout.linear.effective",
        eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
    ] {
        add(
            destination.text(format_args!("{path}"))?,
            Stage::VocabularyScores,
        )?;
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn common_causal_component_rows_are_unique_and_keep_the_actual_sequence_axis() {
        let mut rows = Vec::new();
        append_decoder_component_prefill_observations(
            &mut rows,
            "model.layers.0",
            "feed_forward.output",
            None,
        )
        .unwrap();
        append_dense_component_prefill_observations(&mut rows, "model.layers.0.feed_forward", None)
            .unwrap();
        let unique = rows
            .iter()
            .map(|row| row.path())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), rows.len());
        for path in [
            "attention.channels",
            "attention.channels.effective",
            "attention.write_input",
            "feed_forward.input",
            "feed_forward.output.effective",
            "feed_forward.residual",
            "feed_forward.units",
            "feed_forward.write_input",
        ] {
            let name = format!("model.layers.0.{path}");
            let declaration = rows.iter().find(|row| row.path() == name).unwrap();
            assert_eq!(declaration.sequence_axis(), 1);
            assert_eq!(declaration.readout_stage(), Stage::BeforeReadout);
            assert!(!declaration.flattens_batch_tokens());
        }
    }
}
