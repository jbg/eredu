//! Admission and execution of parameter operations for a single local owner.

use super::*;
use eredu_core::{capture::elements, intervention::InterventionDtype};
use std::collections::BTreeSet;

pub(super) fn local_catalog<B, N>(
    session: &mut Session<B, N>,
) -> Result<(ParameterDiscovery, BTreeMap<String, EffectiveLayout>), ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    session
        .ensure_no_submission_in_flight()
        .map_err(parameter_failure)?;
    if session
        .resources()
        .model
        .as_ref()
        .partition_parameter_description()
        .is_some()
    {
        return Err(ParameterError::Unsupported(
            "partitioned parameter operations require the retained collective communication owner"
                .into(),
        ));
    }
    let artifact_identity = session
        .model_discovery()
        .ok_or_else(|| ParameterError::Unsupported("no retained prepared identity".into()))?
        .capture()?
        .artifact_identity;
    let epoch = session.parameter_epoch;
    let active = session.parameter_overlay_identity();
    let discovery_identity = format!("{}:parameters:{epoch}", session.intervention_identity());
    let overlay_identity = active.map(str::to_owned);
    let payload = session
        .exclusive_resources()
        .ok_or_else(|| ParameterError::Unsupported("native payload still retained".into()))?;
    let payload = &mut **payload;
    let originals = payload
        .parameter_overlay
        .as_ref()
        .map(InstalledTextParameterOverlay::originals);
    let (parameters, layouts) = effective::local_parameter_slots::<B, N>(
        &mut payload.model,
        originals.unwrap_or(&BTreeMap::new()),
    )?;
    Ok((
        ParameterDiscovery {
            identity: discovery_identity,
            artifact_identity,
            overlay_identity,
            parameters,
            usage: session.parameter_usage.get(),
            coordination_usage: Default::default(),
        },
        layouts,
    ))
}

pub(super) fn local_read<B, N>(
    session: &mut Session<B, N>,
    identity: &str,
    parameter: &str,
    region: &ParameterRegion,
    projection: Option<&ParameterProjection>,
    limits: CaptureUsage,
    context: &Context<B, N>,
) -> Result<(InterventionDtype, Vec<u64>, Vec<f32>), ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    let (discovery, layouts) = local_catalog::<B, N>(session)?;
    if identity != discovery.identity {
        return Err(ParameterError::StaleIdentity);
    }
    let descriptor = discovery
        .parameters
        .iter()
        .find(|p| p.id == parameter)
        .ok_or_else(|| ParameterError::Missing(parameter.into()))?;
    if !descriptor.supported {
        return Err(ParameterError::Unsupported(descriptor.condition.clone()));
    }
    let (shape, selected, output) = match projection {
        Some(projection) => {
            let shape = projection.output_shape(&descriptor.shape)?;
            let output = elements(&shape)?;
            (shape, elements(&region.shape)?, output)
        }
        None => {
            let count = region.validate(&descriptor.shape)?;
            (region.shape.clone(), count, count)
        }
    };
    let layout = &layouts[parameter];
    let mut cost = layout.price_loan(N::local_read_cost(
        &layout.physical,
        elements(&descriptor.shape)?,
        selected,
        output,
        projection.map(|p| p.coefficients.len() as u64),
    )?)?;
    cost.captures = 1;
    cost.encoded_bytes = add(
        mul(output, 32)?,
        add(2048, mul((parameter.len() + identity.len()) as u64, 6)?)?,
    )?;
    session.parameter_budget(limits, None).reserve_quota(cost)?;
    let values = session
        .with_model_transaction(|model| {
            super::loans::with_parameter_values::<B, N, _>(
                model,
                layout,
                parameter,
                context,
                |selected| {
                    N::read_parameter_values(
                        selected,
                        &layout.physical,
                        parameter,
                        region,
                        projection,
                        context,
                    )
                },
            )
        })
        .map_err(parameter_failure)?;
    if values.iter().any(|value| !value.is_finite()) {
        let message = if projection.is_some() {
            "non-finite effective parameter projection"
        } else {
            "non-finite effective parameter"
        };
        return Err(ParameterError::Invalid(message.into()));
    }
    Ok((
        descriptor.dtype.expect("supported floating dtype"),
        shape,
        values,
    ))
}

pub(super) fn local_activation<B, N>(
    session: &mut Session<B, N>,
    overlay: &AdmittedParameterOverlay,
    limits: CaptureUsage,
    context: &Context<B, N>,
) -> Result<ParameterDiscovery, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    let (mut discovery, layouts) = local_catalog::<B, N>(session)?;
    overlay.validate(&discovery)?;
    if discovery.overlay_identity.is_some() {
        return Err(ParameterError::Unsupported(
            "remove the active overlay before installing another transaction".into(),
        ));
    }
    let epoch = next_epoch(session)?;
    let shared: BTreeSet<_> = overlay.shared_targets().iter().collect();
    let mut cost = CaptureUsage {
        captures: 1,
        retained_bytes: 4096,
        host_bytes: 4096,
        encoded_bytes: 4096,
    };
    let mut targets = Vec::new();
    let mut precision = None;
    for target in &discovery.parameters {
        if !shared.contains(&target.shared_id) {
            continue;
        }
        if !target.supported {
            return Err(ParameterError::Unsupported(
                "shared target cannot be edited by the selected native mechanism".into(),
            ));
        }
        let (base, additional_destination) =
            N::local_overlay_cost(&layouts[&target.id].physical, elements(&target.shape)?)?;
        cost = cost.checked_add(layouts[&target.id].price_loan(base)?)?;
        for (edit, class) in overlay.plan().edits.iter().zip(overlay.shared_targets()) {
            if class == &target.shared_id {
                if edit.parameter_shape != target.shape || Some(edit.dtype) != target.dtype {
                    return Err(ParameterError::Invalid(
                        "shared parameter slots have incompatible geometry or dtype".into(),
                    ));
                }
                cost.retained_bytes = add(cost.retained_bytes, additional_destination)?;
            }
        }
        targets.push(target);
        precision = precision.max(parameter_width(target.dtype));
    }
    for edit in &overlay.plan().edits {
        let values = edit.update.values().len() as u64;
        cost.host_bytes = add(cost.host_bytes, mul(values, 8)?)?;
        cost.retained_bytes = add(cost.retained_bytes, mul(values, 16)?)?;
    }
    session.parameter_budget(limits, None).reserve_quota(cost)?;
    // Each local slot consumes the same admitted edit for its shared class. Borrow
    // the plan until submission to avoid cloning the host values for each alias.
    let targets = targets.into_iter().map(|target| {
        (
            target.id.as_str(),
            overlay
                .plan()
                .edits
                .iter()
                .zip(overlay.shared_targets())
                .filter(|(_, class)| *class == &target.shared_id)
                .map(|(edit, _)| (&edit.region, &edit.update)),
        )
    });
    let original = session
        .with_model_transaction(|model| {
            let (original, replacement) = match super::loans::prepare_replacements::<B, N, _>(
                model, &layouts, targets, context,
            )? {
                Ok(handles) => handles,
                Err(error) => return Ok(Err(error)),
            };
            model.as_mut().reset_local()?;
            super::overlay::publish_tensors::<B, N>(model, &replacement, true)?;
            super::overlay::invalidate_snapshots::<B, N>(model);
            Ok(Ok(original))
        })
        .map_err(parameter_failure)??;
    for parameter in &mut discovery.parameters {
        if shared.contains(&parameter.shared_id) {
            parameter.input_transform = ProjectionInputTransform::Identity;
        }
    }
    let payload = session
        .exclusive_resources()
        .expect("completed parameter transaction");
    payload.parameter_overlay = Some(InstalledTextParameterOverlay::Local {
        originals: original,
    });
    session.commit_parameter_overlay(
        &mut discovery,
        epoch,
        Some(ActiveParameterOverlay {
            identity: overlay.identity().into(),
            precision,
            baseline_transforms: BTreeMap::new(),
        }),
    );
    Ok(discovery)
}

pub(super) fn local_removal<B, N>(
    session: &mut Session<B, N>,
    identity: &str,
) -> Result<ParameterDiscovery, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    let (mut discovery, layouts) = local_catalog::<B, N>(session)?;
    if discovery.identity != identity || discovery.overlay_identity.is_none() {
        return Err(ParameterError::StaleIdentity);
    }
    let epoch = next_epoch(session)?;
    let Some(InstalledTextParameterOverlay::Local { originals }) =
        &session.resources().parameter_overlay
    else {
        return Err(ParameterError::StaleIdentity);
    };
    let original = originals.clone();
    session
        .with_model_transaction(|model| {
            model.as_mut().reset_local()?;
            super::overlay::publish_tensors::<B, N>(model, &original, false)?;
            super::overlay::invalidate_snapshots::<B, N>(model);
            Ok(())
        })
        .map_err(parameter_failure)?;
    for parameter in &mut discovery.parameters {
        if let Some(layout) = layouts.get(&parameter.id) {
            if let Some(transform) = original.get(&parameter.id).map(|value| {
                N::parameter_input_transform(&layout.physical, &N::parameter_dtype(value))
            }) {
                parameter.input_transform = transform;
            }
        }
    }
    let payload = session
        .exclusive_resources()
        .expect("completed parameter transaction");
    payload.parameter_overlay = None;
    session.commit_parameter_overlay(&mut discovery, epoch, None);
    Ok(discovery)
}

fn next_epoch<R, N>(session: &TextSessionDriver<R, N>) -> Result<u64, ParameterError> {
    session
        .parameter_epoch
        .checked_add(1)
        .ok_or(ParameterError::Overflow)
}
