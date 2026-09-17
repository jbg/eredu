//! Explicit row semantics of the common ordinary decoder's actual hooks.
use super::*;
use eredu_runtime::layered::{
    PrefillObservationDeclaration as Declaration, PrefillReadoutStage as Stage,
};

pub(super) fn declarations<C: Config>(
    config: &C,
    causal_blocks: bool,
    mut components: impl FnMut(&str, usize, &mut Vec<Declaration>),
) -> Result<Vec<Declaration>, Error> {
    let layers = if causal_blocks {
        usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?
    } else {
        0
    };
    let mut declarations = ordinary_prefill_observation_declarations(
        (0..layers).map(|index| Ok(format!("{}.layers.{index}", config.parameter_root()))),
        causal_blocks,
    )?;
    // The same Config method supplies the bank identities to actual observed
    // execution. CAUSAL_PREFILL_ROWS is the block factory's equation proof.
    for index in 0..layers {
        let path = format!("{}.layers.{index}", config.parameter_root());
        components(&path, index, &mut declarations);
        if let Some(points) = config.routed_observation_points(&path, index) {
            append_routed_prefill_observations(&mut declarations, &points);
        }
    }
    Ok(declarations)
}

/// Actual unit products and write terms of a caller-identified dense MLP.
/// Both equations act independently per request row. The caller supplies the
/// selected ordinary/TP worker and its causal input proof; a path alone is not
/// enough. Spatial completion and additive write provenance remain separate.
pub(crate) fn append_dense_component_prefill_observations(
    declarations: &mut Vec<Declaration>, component: &str,
) {
    for suffix in ["units", "units.effective", "write", "write.effective"] {
        declarations.push(Declaration::causal_ordinary_text(
            format!("{component}.{suffix}"), 1, Stage::BeforeReadout));
    }
}

/// Sparse companions of actual causal, row-local bank invocations. The caller
/// supplies equation applicability and the same bank points used by execution.
/// Projection, routing mutation and native admission remain independent.
pub(crate) fn append_routed_prefill_observations(
    declarations: &mut Vec<Declaration>,
    points: &eredu_runtime::RoutedObservationPoints,
) {
    for (_, point) in points.iter() {
        append_routed_prefill_path(declarations, point.path());
    }
}

/// Original/effective sparse hooks of one architecture-declared causal bank.
/// The caller supplies its actual invocation identity and sparse schedule.
pub(crate) fn append_routed_prefill_path(declarations: &mut Vec<Declaration>, routing: &str) {
    for path in [format!("{routing}.units"), format!("{routing}.units.effective")] {
        declarations.push(Declaration::causal_routed_units(path));
    }
}

/// Shared outer row hooks for an architecture's actual ordinary target units.
/// The calling architecture supplies the causal equation proof and exact paths;
/// this helper neither selects an execution owner nor declares internal hooks.
pub(crate) fn ordinary_prefill_observation_declarations(
    unit_paths: impl IntoIterator<Item = Result<String, Error>>,
    causal_blocks: bool,
) -> Result<Vec<Declaration>, Error> {
    let mut entries = Vec::new();
    let mut add =
        |path: String, stage| entries.push(Declaration::causal_ordinary_text(path, 1, stage));
    // Actual scaled embedding observation is emitted before layers/readout.
    // Its effective companion is unchanged in this intervention-free invocation.
    for path in ["readout.embedding", "readout.embedding.effective"] {
        add(path.into(), Stage::BeforeReadout);
    }
    if !causal_blocks {
        return Ok(entries);
    }
    for unit in unit_paths {
        let unit = unit?;
        for point in [
            eredu_core::UnitObservation::Input,
            eredu_core::UnitObservation::Output,
        ] {
            let path = point.path(&unit);
            add(path.clone(), Stage::BeforeReadout);
            add(format!("{path}.effective"), Stage::BeforeReadout);
        }
    }
    // These identities are the actual ComponentInstrumentation/primary head
    // hooks. They are declared here, never classified later by path prefix.
    for path in ["readout.residual", "readout.normalized"] {
        add(path.into(), Stage::ReadoutInput);
        add(format!("{path}.effective"), Stage::ReadoutInput);
    }
    add("readout.projection_input".into(), Stage::ReadoutInput);
    for path in [
        "readout.linear",
        "readout.linear.effective",
        eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
    ] {
        add(path.into(), Stage::VocabularyScores);
    }
    Ok(entries)
}

/// Exact decoder hooks of an architecture-validated retained-media source.
/// The architecture supplies causal/cache/position equivalence; no encoder
/// paths or generated projection factories are admitted by this helper.
pub(crate) fn media_prefill_observation_declarations(
    unit_paths: impl IntoIterator<Item = Result<String, Error>>,
) -> Result<Vec<Declaration>, Error> {
    let mut entries = Vec::new();
    let mut add =
        |path: String, stage| entries.push(Declaration::prepared_media_decoder(path, 1, stage));
    for path in ["readout.embedding", "readout.embedding.effective"] {
        add(path.into(), Stage::BeforeReadout);
    }
    for unit in unit_paths {
        let unit = unit?;
        for point in [
            eredu_core::UnitObservation::Input,
            eredu_core::UnitObservation::Output,
        ] {
            let path = point.path(&unit);
            add(path.clone(), Stage::BeforeReadout);
            add(format!("{path}.effective"), Stage::BeforeReadout);
        }
    }
    for path in ["readout.residual", "readout.normalized"] {
        add(path.into(), Stage::ReadoutInput);
        add(format!("{path}.effective"), Stage::ReadoutInput);
    }
    for path in [
        "readout.linear",
        "readout.linear.effective",
        eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
    ] {
        add(path.into(), Stage::VocabularyScores);
    }
    Ok(entries)
}
