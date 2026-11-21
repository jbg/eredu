//! Admission, publication and rollback of distributed parameter edits.

use super::*;
use eredu_core::capture::elements;
use sha2::{Digest, Sha256};

struct PreparedPartitionOverlay<V> {
    previous: BTreeMap<String, V>,
    next: BTreeMap<String, V>,
    state: OverlayState,
    reset_estimate: Option<SnapshotEstimate>,
}

struct OverlayState {
    state: super::super::ErasedTextState,
    exchanged: bool,
}

pub(super) fn invalidate_snapshots<B: TextGenerationBackend, N: ParameterMechanisms<B>>(
    model: &mut Model<B, N>,
) {
    if let Some((state, _)) = model.as_mut().control_state_mut() {
        state.invalidate_parameter_snapshots();
    }
}

pub(super) fn publish_tensors<B: TextGenerationBackend, N: ParameterMechanisms<B>>(
    model: &mut Model<B, N>,
    tensors: &ParameterTensors<B, N>,
    active: bool,
) -> Result<(), B::Error> {
    if !model
        .as_mut()
        .publish_parameter_replacements(tensors, active)?
    {
        return Err(N::parameter_access_error(ParameterError::Unsupported(
            "prepared parameter publication is unavailable".into(),
        )));
    }
    Ok(())
}

fn prepare_overlay_state<B: TextGenerationBackend, N: ParameterMechanisms<B>>(
    model: &mut Model<B, N>,
) -> Result<OverlayState, B::Error> {
    let (state, context) = model.as_mut().control_state_mut().ok_or_else(|| {
        N::parameter_access_error(ParameterError::Unsupported(
            "parameter state reset is unavailable".into(),
        ))
    })?;
    let state = state
        .prepare_parameter_reset(context)
        .map_err(N::preparation_error)?;
    Ok(OverlayState {
        state,
        exchanged: false,
    })
}

fn exchange_overlay_state<B: TextGenerationBackend, N: ParameterMechanisms<B>>(
    model: &mut Model<B, N>,
    prepared: &mut OverlayState,
) -> Result<(), B::Error> {
    let (state, _) = model.as_mut().control_state_mut().ok_or_else(|| {
        N::parameter_access_error(ParameterError::Unsupported(
            "parameter state exchange is unavailable".into(),
        ))
    })?;
    state
        .exchange_parameter_reset(&mut prepared.state)
        .map_err(N::preparation_error)?;
    prepared.exchanged = !prepared.exchanged;
    Ok(())
}

fn publish_partition_overlay<B: TextGenerationBackend, N: ParameterMechanisms<B>>(
    model: &mut Model<B, N>,
    prepared: &mut OverlayState,
    tensors: &ParameterTensors<B, N>,
    active: bool,
    rollback: bool,
) -> Result<(), B::Error> {
    // The outer coordinator performs agreement after this local transaction.
    if !rollback {
        exchange_overlay_state::<B, N>(model, prepared)?;
    }
    publish_tensors::<B, N>(model, tensors, active)?;
    if rollback && prepared.exchanged {
        exchange_overlay_state::<B, N>(model, prepared)?;
    }
    Ok(())
}

pub(super) fn partition_activation<B, N>(
    session: &mut Session<B, N>,
    overlay: &AdmittedParameterOverlay,
    limits: CaptureUsage,
    context: &Context<B, N>,
) -> Result<ParameterDiscovery, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    let catalog = partition_catalog::<B, N>(session, Some(limits))?;
    let intent = Sha256::digest(overlay.intent_identity().as_bytes()).into();
    partition_edit::<B, N, _>(
        session,
        catalog,
        intent,
        Some(overlay),
        |session, catalog, rank| {
            overlay.validate(&catalog.discovery)?;
            if catalog.discovery.overlay_identity.is_some() {
                return Err(ParameterError::Unsupported(
                    "remove the active overlay before installing another transaction".into(),
                ));
            }
            let mut budget = session.parameter_budget(limits, None);
            let estimate = session
                .resources()
                .model
                .as_ref()
                .control_state()
                .and_then(|state| state.estimate_parameter_reset())
                .ok_or_else(|| {
                    ParameterError::Unsupported(
                        "bounded fresh parameter state is unavailable".into(),
                    )
                })?;
            budget.reserve_quota(CaptureUsage {
                captures: 1,
                retained_bytes: add(4096, estimate.retained_bytes)?,
                host_bytes: add(4096, estimate.copy_bytes)?,
                encoded_bytes: 4096,
            })?;
            let targets = prepare_targets::<B, N>(catalog, overlay, rank, &mut budget)?;
            Ok((targets, estimate))
        },
        |model, layouts, (targets, estimate)| {
            let edits = targets.iter().map(|target| {
                (
                    target.id.as_str(),
                    target.edits.iter().map(|edit| (&edit.region, &edit.update)),
                )
            });
            let (original, replacement) = match super::loans::prepare_replacements::<B, N, _>(
                model, layouts, edits, context,
            )? {
                Ok(pair) => pair,
                Err(error) => return Ok(Err(error)),
            };
            Ok(Ok(PreparedPartitionOverlay {
                previous: original,
                next: replacement,
                state: prepare_overlay_state::<B, N>(model)?,
                reset_estimate: Some(estimate),
            }))
        },
    )
}

pub(super) fn partition_removal<B, N>(
    session: &mut Session<B, N>,
    identity: &str,
) -> Result<ParameterDiscovery, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    let catalog = partition_catalog::<B, N>(session, None)?;
    let mut digest = Sha256::new();
    digest.update(b"eredu-remove-parameter-overlay-v1\0");
    if let Some(active) = &catalog.discovery.overlay_identity {
        digest.update(active.as_bytes());
    }
    partition_edit::<B, N, _>(
        session,
        catalog,
        digest.finalize().into(),
        None,
        |session, catalog, _| {
            if catalog.discovery.identity != identity
                || catalog.discovery.overlay_identity.is_none()
            {
                return Err(ParameterError::StaleIdentity);
            }
            let Some(InstalledTextParameterOverlay::Partitioned {
                originals,
                published,
                reset_estimate,
            }) = &session.resources().parameter_overlay
            else {
                return Err(ParameterError::StaleIdentity);
            };
            let mut cost = CaptureUsage {
                captures: 1,
                retained_bytes: add(4096, reset_estimate.retained_bytes)?,
                host_bytes: add(4096, reset_estimate.copy_bytes)?,
                encoded_bytes: 4096,
            };
            for id in originals.keys() {
                cost.host_bytes = add(cost.host_bytes, add(512, mul(id.len() as u64, 4)?)?)?;
            }
            session
                .parameter_budget(UNLIMITED, None)
                .reserve_quota(cost)?;
            Ok((published.clone(), originals.clone()))
        },
        |model, _, (previous, next)| {
            Ok(Ok(PreparedPartitionOverlay {
                previous,
                next,
                state: prepare_overlay_state::<B, N>(model)?,
                reset_estimate: None,
            }))
        },
    )
}

fn partition_edit<B, N, P>(
    session: &mut Session<B, N>,
    catalog: PartitionedParameters<EffectiveLayout>,
    intent: [u8; 32],
    overlay: Option<&AdmittedParameterOverlay>,
    admit: impl FnOnce(
        &Session<B, N>,
        &PartitionedParameters<EffectiveLayout>,
        usize,
    ) -> Result<P, ParameterError>,
    prepare: impl FnOnce(
        &mut Model<B, N>,
        &BTreeMap<String, EffectiveLayout>,
        P,
    ) -> Result<
        Result<PreparedPartitionOverlay<N::ParameterValue>, ParameterError>,
        B::Error,
    >,
) -> Result<ParameterDiscovery, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    let retained = &session.resources().retained;
    let transport = N::control_transport(retained)
        .expect("completed partition catalog")
        .clone();
    let epoch = session.parameter_epoch;
    let (owner, binding) = session.parameter_operation(epoch)?;
    let active = overlay.is_some();
    let kind = if active {
        ParameterOperationKind::Activation
    } else {
        ParameterOperationKind::Removal
    };
    let local = epoch
        .checked_add(1)
        .ok_or(ParameterError::Overflow)
        .and_then(|next| Ok((admit(session, &catalog, transport.parameter_rank())?, next)));
    let PartitionedParameters {
        mut discovery,
        layouts,
        ..
    } = catalog;
    let result = {
        let session = RefCell::new(&mut *session);
        owner.transaction(
            &transport,
            binding,
            &intent,
            kind,
            local,
            |(value, epoch)| {
                let prepared = session
                    .borrow_mut()
                    .with_model_transaction(|model| prepare(model, &layouts, value))
                    .map_err(parameter_failure)??;
                Ok((prepared, epoch))
            },
            |(prepared, _)| {
                session
                    .borrow_mut()
                    .with_model_transaction(|model| {
                        publish_partition_overlay::<B, N>(
                            model,
                            &mut prepared.state,
                            &prepared.next,
                            active,
                            false,
                        )
                    })
                    .map_err(parameter_failure)
            },
            |(prepared, _)| {
                session
                    .borrow_mut()
                    .with_model_transaction(|model| {
                        publish_partition_overlay::<B, N>(
                            model,
                            &mut prepared.state,
                            &prepared.previous,
                            !active,
                            true,
                        )
                    })
                    .map_err(parameter_failure)
            },
        )
    };
    let (prepared, epoch) = finish_control::<B, N, _>(session, &transport, result)?;
    let payload = session
        .exclusive_resources()
        .expect("completed parameter transaction");
    let payload = &mut **payload;
    invalidate_snapshots::<B, N>(&mut payload.model);
    payload.parameter_overlay = if active {
        Some(InstalledTextParameterOverlay::Partitioned {
            originals: prepared.previous,
            published: prepared.next,
            reset_estimate: prepared
                .reset_estimate
                .expect("admitted activation reset bound"),
        })
    } else {
        None
    };
    drop(prepared.state);
    let next = if let Some(overlay) = overlay {
        let mut baseline_transforms = BTreeMap::new();
        let mut precision = None;
        for parameter in &mut discovery.parameters {
            if overlay.shared_targets().contains(&parameter.shared_id) {
                let transform = std::mem::replace(
                    &mut parameter.input_transform,
                    ProjectionInputTransform::Identity,
                );
                baseline_transforms.insert(parameter.id.clone(), transform);
                precision = precision.max(parameter_width(parameter.dtype));
            }
        }
        Some(ActiveParameterOverlay {
            identity: overlay.intent_identity().into(),
            precision,
            baseline_transforms,
        })
    } else {
        let mut baseline_transforms = session
            .parameter_overlay
            .take()
            .expect("admitted overlay removal")
            .baseline_transforms;
        for parameter in &mut discovery.parameters {
            if let Some(transform) = baseline_transforms.remove(&parameter.id) {
                parameter.input_transform = transform;
            }
        }
        None
    };
    session.commit_parameter_overlay(&mut discovery, epoch, next);
    discovery.coordination_usage = owner.usage()?;
    Ok(discovery)
}

/// An update restricted to rank-local effective parameter coordinates.
pub struct LocalParameterEdit {
    /// Exact region of this tensor.
    pub region: ParameterRegion,
    /// Update values sliced to that region.
    pub update: ParameterUpdate,
}

/// One local tensor whose parameter class is affected by an overlay.
pub struct LocalParameterTarget {
    /// Effective slot identifier.
    pub id: String,
    /// Edits intersecting the local shard, in plan order.
    pub edits: Vec<LocalParameterEdit>,
}

fn prepare_targets<B, N>(
    catalog: &PartitionedParameters<EffectiveLayout>,
    overlay: &AdmittedParameterOverlay,
    rank: usize,
    budget: &mut ParameterBudget,
) -> Result<Vec<LocalParameterTarget>, ParameterError>
where
    B: TextGenerationBackend,
    N: ParameterMechanisms<B>,
{
    let mut targets = Vec::new();
    for target in &catalog.discovery.parameters {
        if !overlay.shared_targets().contains(&target.shared_id) {
            continue;
        }
        if !target.access().replacement {
            return Err(ParameterError::Unsupported(target.condition.clone()));
        }
        // Global baseline transforms and return metadata are retained even on
        // nonowning ranks. Native storage is charged only to actual local owners.
        budget.reserve_quota(CaptureUsage {
            host_bytes: add(512, mul(target.id.len() as u64, 4)?)?,
            ..Default::default()
        })?;
        for (edit, shared) in overlay.plan().edits.iter().zip(overlay.shared_targets()) {
            if shared == &target.shared_id
                && (edit.parameter_shape != target.shape || Some(edit.dtype) != target.dtype)
            {
                return Err(ParameterError::Invalid(
                    "shared parameter slots have incompatible geometry or dtype".into(),
                ));
            }
        }
        let map = catalog
            .global
            .coordinates(&target.id)
            .and_then(|maps| maps.get(rank))
            .and_then(Option::as_ref);
        let Some(map) = map else {
            continue;
        };
        if map.local_shape().contains(&0) {
            continue;
        }
        let layout = catalog.layouts.get(&target.id).ok_or_else(|| {
            ParameterError::Incomplete("local edit owner has no effective layout".into())
        })?;
        let (mut base_cost, per_edit) =
            N::local_overlay_cost(&layout.physical, elements(&layout.physical.shape)?)?;
        base_cost.host_bytes = add(
            base_cost.host_bytes,
            layout.selection_metadata_bytes(&target.id)?,
        )?;
        budget.reserve_quota(layout.price_loan(base_cost)?)?;
        let mut edits = Vec::new();
        for (edit, shared) in overlay.plan().edits.iter().zip(overlay.shared_targets()) {
            if shared != &target.shared_id {
                continue;
            }
            let projected = map.project_region(&edit.region, 65_536, budget)?;
            for (index, fragment) in projected.fragments().iter().enumerate() {
                let values = elements(&fragment.local().shape)?;
                budget.reserve_quota(CaptureUsage {
                    retained_bytes: add(per_edit, mul(values, 16)?)?,
                    host_bytes: add(
                        512,
                        add(
                            mul(values, 8)?,
                            mul(fragment.local().shape.len() as u64, 32)?,
                        )?,
                    )?,
                    ..Default::default()
                })?;
                edits.push(LocalParameterEdit {
                    region: fragment.local().clone(),
                    update: projected.project_update(index, &edit.update, budget)?,
                });
            }
        }
        targets.push(LocalParameterTarget {
            id: target.id.clone(),
            edits,
        });
    }
    Ok(targets)
}
