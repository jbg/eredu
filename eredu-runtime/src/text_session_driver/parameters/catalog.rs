//! Bounded exchange of exact loaded placement and capability facts.

use super::*;
use eredu_core::capture::{CaptureBudget, CaptureError};
use sha2::{Digest, Sha256};

/// Catalog agreed by all ranks, with the corresponding rank-local native layouts.
pub(super) struct PartitionedParameters<L> {
    /// Loaded global parameter descriptors.
    pub(super) discovery: ParameterDiscovery,
    /// Rank-local layouts. A rank may own none of the requested parameter.
    pub(super) layouts: BTreeMap<String, L>,
    /// Validated global ownership.
    pub(super) global: PartitionParameterCatalog,
}

const DISCOVERY_LIMIT: CaptureUsage = CaptureUsage {
    captures: 1,
    retained_bytes: 256 << 20,
    host_bytes: 1 << 30,
    encoded_bytes: 128 << 20,
};

fn parameter_outputs(
    task: &crate::ReplicatedTextMaterializationTask,
) -> impl Iterator<Item = (&str, &[usize], eredu_checkpoint::LinearFormat)> {
    std::iter::once(task.name())
        .chain(task.aliases().iter().map(String::as_str))
        .map(move |name| (name, task.logical_shape(), task.executable()))
        .chain(task.output_companions().iter().map(|companion| {
            (
                companion.name(),
                companion.logical_shape(),
                eredu_checkpoint::LinearFormat::Dense,
            )
        }))
}

/// Logical placement and effective storage format of a selected native slot.
pub struct ParameterSlotLayout {
    /// Effective format, after checkpoint source selection.
    pub format: eredu_checkpoint::LinearFormat,
    /// Effective local tensor shape.
    pub local_shape: Vec<u64>,
    /// Global logical tensor shape.
    pub global_shape: Vec<u64>,
}

struct CatalogPreparation<L> {
    artifact: String,
    layouts: BTreeMap<String, L>,
    facts: Vec<LocalParameterFacts>,
    geometry: [u8; 32],
    max_bytes: usize,
}

fn prepare_catalog<B, N>(
    session: &Session<B, N>,
    transport: &N::ControlTransport,
    budget: &mut ParameterBudget,
) -> Result<CatalogPreparation<EffectiveLayout>, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    session
        .ensure_no_submission_in_flight()
        .map_err(parameter_failure)?;
    let world = transport.parameter_setup().participant_count();
    let prepared = session.model_discovery().ok_or_else(|| {
        ParameterError::Unsupported("no retained prepared parameter identity".into())
    })?;
    let (primary, auxiliary) = prepared.partition_parameter_tasks().ok_or_else(|| {
        ParameterError::Unsupported("no global prepared parameter declaration".into())
    })?;
    let tasks = primary.iter().chain(auxiliary);
    let shared_name_bytes = tasks
        .clone()
        .flat_map(parameter_outputs)
        .map(|(name, _, _)| name.len())
        .max()
        .unwrap_or(0) as u64;
    let mut max_bytes = 1024u64;
    for (id, shape, _) in tasks.clone().flat_map(parameter_outputs) {
        max_bytes = add(
            max_bytes,
            add(
                1024,
                add(
                    mul(add(id.len() as u64, shared_name_bytes)?, 6)?,
                    mul(shape.len() as u64, 64)?,
                )?,
            )?,
        )?;
    }
    // The metadata protocol has finite per-rank and padded-world bounds.
    if max_bytes > 16 << 20 || mul(max_bytes, world as u64)? > 128 << 20 {
        return Err(CaptureError::Limit {
            budget: CaptureBudget::Host,
            cumulative: false,
        }
        .into());
    }
    budget.reserve_quota(CaptureUsage {
        captures: 1,
        // Include the temporary index of borrowed task names and formats.
        host_bytes: mul(max_bytes, 9)?,
        encoded_bytes: max_bytes,
        ..Default::default()
    })?;
    let artifact = prepared.capture()?.artifact_identity;
    let mut digest = Sha256::new();
    digest.update(b"eredu-loaded-parameter-catalog-v1\0");
    for value in [&artifact, prepared.execution_identity()] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }

    let payload = &**session.resources();
    let slots = payload.model.as_ref().prepared_parameter_slots();
    let mut formats = BTreeMap::new();
    for (name, _, format) in tasks.flat_map(parameter_outputs) {
        formats.entry(name).or_insert(format);
    }
    let mut placements = BTreeMap::new();
    for slot in slots {
        let id = slot.parameter.id.as_str();
        let format = formats
            .get(id)
            .copied()
            .ok_or_else(|| ParameterError::Missing(id.into()))?;
        let placement = prepared
            .parameter_partition_layout_for_rank(id, transport.parameter_rank(), budget)?
            .ok_or_else(|| ParameterError::Unsupported("no effective local placement".into()))?;
        let coordinates = placement.coordinates().ok_or_else(|| {
            ParameterError::Invalid("prepared slot has no selected rank ownership".into())
        })?;
        placements.insert(
            id,
            ParameterSlotLayout {
                format,
                local_shape: coordinates.local_shape().to_vec(),
                global_shape: placement.global_shape().to_vec(),
            },
        );
    }
    let state_reset = payload
        .model
        .as_ref()
        .control_state()
        .and_then(|state| state.estimate_parameter_reset())
        .is_some();
    let originals = payload
        .parameter_overlay
        .as_ref()
        .map(InstalledTextParameterOverlay::originals);
    let (facts, layouts) = effective::partition_parameter_slots::<B, N>(
        originals.unwrap_or(&BTreeMap::new()),
        slots,
        placements,
        state_reset,
    )?;
    Ok(CatalogPreparation {
        artifact,
        layouts,
        facts,
        geometry: digest.finalize().into(),
        max_bytes: usize::try_from(max_bytes).map_err(|_| ParameterError::Overflow)?,
    })
}

pub(super) fn partition_catalog<B, N>(
    session: &Session<B, N>,
    limits: Option<CaptureUsage>,
) -> Result<PartitionedParameters<EffectiveLayout>, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    let epoch = session.parameter_epoch;
    let active = session.parameter_overlay_identity();
    let transport =
        N::control_transport(&session.resources().retained).expect("selected partition transport");
    let (owner, binding) = session.parameter_catalog_operation(epoch)?;
    let mut budget = session.parameter_budget(
        limits.unwrap_or(UNLIMITED),
        limits.is_none().then_some(DISCOVERY_LIMIT),
    );
    let prepared = prepare_catalog::<B, N>(session, transport, &mut budget);
    let (local, admission) = match prepared {
        Ok(local) => {
            let admission =
                ParameterReadPreparation::new((), local.geometry, local.max_bytes.div_ceil(4) + 2);
            (Some(local), Ok(admission))
        }
        Err(error) => (None, Err(error)),
    };
    let mut encoding_budget = budget.clone();
    let mut assembly_budget = budget.clone();
    let result = owner.read(
        transport,
        binding,
        &[0x63; 32],
        ParameterOperationKind::Catalog,
        admission,
        &mut budget,
        |_| {
            let local = local.as_ref().expect("admitted local catalogue");
            encode_parameter_catalog(&local.facts, local.max_bytes, &mut encoding_budget)
        },
        |rows| {
            let local = local.as_ref().expect("admitted local catalogue");
            assembly_budget.reserve_quota(CaptureUsage {
                host_bytes: mul(rows.len() as u64, 64)?,
                ..Default::default()
            })?;
            let facts = rows
                .iter()
                .map(|row| decode_parameter_catalog(row, local.max_bytes, &mut assembly_budget))
                .collect::<Result<Vec<_>, _>>()?;
            let prepared = session
                .model_discovery()
                .expect("admitted prepared discovery");
            PartitionParameterCatalog::new(
                &facts,
                |rank, parameter, budget| {
                    // Sharing is supplied by actual prepared slot metadata.
                    // Distinct tasks can intentionally share a module parameter;
                    // runtime validates the canonical class and matching shape.
                    let layout = prepared
                        .parameter_partition_layout_for_rank(&parameter.id, rank, budget)?
                        .ok_or_else(|| {
                            ParameterError::Unsupported(
                                "retained parameter placement is unavailable".into(),
                            )
                        })?;
                    if layout.global_shape() != parameter.shape {
                        return Err(ParameterError::Invalid(
                            "loaded global parameter shape differs from retained task".into(),
                        ));
                    }
                    Ok(layout.into_coordinates())
                },
                &mut assembly_budget,
            )
        },
    );
    let global = finish_control::<B, N, _>(session, transport, result)?;
    let local = local.expect("completed catalogue preparation");
    // The initial metadata allowance includes this public descriptor copy
    // as well as the local maps/facts; no parameter tensor is retained.

    let discovery = ParameterDiscovery {
        identity: format!("{}:parameters:{epoch}", session.intervention_identity()),
        artifact_identity: local.artifact,
        overlay_identity: active.map(str::to_owned),
        parameters: global.parameters().to_vec(),
        usage: session.parameter_usage.get(),
        coordination_usage: owner.usage()?,
    };
    Ok(PartitionedParameters {
        discovery,
        layouts: local.layouts,
        global,
    })
}
