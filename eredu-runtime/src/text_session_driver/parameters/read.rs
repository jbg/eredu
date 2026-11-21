//! Rank-local reads and contractions followed by checked global assembly.

use super::*;
use eredu_core::{
    capture::{elements, CaptureQuota},
    intervention::InterventionDtype,
};

struct PreparedRead {
    plan: PartitionParameterReadPlan,
    fragments: Vec<ParameterReadFragment>,
    dtype: InterventionDtype,
    assembly: RefCell<CaptureQuota>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_read<B, N>(
    catalog: &PartitionedParameters<EffectiveLayout>,
    identity: &str,
    parameter: &str,
    region: &ParameterRegion,
    projection: Option<&ParameterProjection>,
    rank: usize,
    budget: &mut ParameterBudget,
) -> Result<PreparedRead, ParameterError>
where
    B: TextGenerationBackend<Error: Send + Sync + 'static>,
    N: ParameterMechanisms<B>,
{
    if identity != catalog.discovery.identity {
        return Err(ParameterError::StaleIdentity);
    }
    if parameter.len() > 4096 || region.shape.len() > 32 || region.starts.len() > 32 {
        return Err(ParameterError::Invalid(
            "bounded parameter identity or geometry".into(),
        ));
    }
    let descriptor = catalog
        .discovery
        .parameters
        .iter()
        .find(|p| p.id == parameter)
        .ok_or_else(|| ParameterError::Missing(parameter.into()))?;
    let access = descriptor.access();
    if !(if projection.is_some() {
        access.projection
    } else {
        access.query
    }) {
        return Err(ParameterError::Unsupported(descriptor.condition.clone()));
    }
    let maps = catalog
        .global
        .coordinates(parameter)
        .ok_or_else(|| ParameterError::Incomplete("loaded parameter owners are absent".into()))?;
    budget.reserve_quota(CaptureUsage {
        host_bytes: add(512, mul(maps.len() as u64, 16)?)?,
        ..Default::default()
    })?;
    let maps = maps.iter().map(Option::as_ref).collect::<Vec<_>>();
    let plan = PartitionParameterReadPlan::new(
        &descriptor.shape,
        region,
        projection,
        &maps,
        65_536,
        budget,
    )?;
    let output = elements(plan.output_shape())?;
    let total = plan
        .rank_counts()
        .iter()
        .try_fold(0u64, |sum, n| add(sum, *n as u64))?;
    let assembly = budget.reserve_quota(CaptureUsage {
        host_bytes: add(256, add(mul(output, 12)?, mul(maps.len() as u64, 8)?)?)?,
        ..Default::default()
    })?;
    // Reserve producer words, host decoding and the final result envelope.
    let mut cost = CaptureUsage {
        captures: 1,
        retained_bytes: 4096,
        host_bytes: add(
            add(512, mul((identity.len() + parameter.len()) as u64, 4)?)?,
            add(mul(total, 4)?, mul(maps.len() as u64, 32)?)?,
        )?,
        encoded_bytes: add(
            mul(output, 32)?,
            add(2048, mul((identity.len() + parameter.len()) as u64, 6)?)?,
        )?,
    };
    let mut fragments = Vec::new();
    for fragment in plan
        .fragments()
        .iter()
        .filter(|fragment| fragment.rank() == rank)
    {
        budget.reserve_quota(CaptureUsage {
            host_bytes: add(256, mul(region.shape.len() as u64, 32)?)?,
            ..Default::default()
        })?;
        let projection = projection
            .map(|projection| {
                fragment
                    .source()
                    .project_contraction(projection, &descriptor.shape, budget)
            })
            .transpose()?;
        fragments.push(ParameterReadFragment {
            region: fragment.source().local().clone(),
            projection,
            output_elements: elements(&fragment.destination().shape)?,
        });
    }

    if !fragments.is_empty() {
        let layout = catalog.layouts.get(parameter).ok_or_else(|| {
            ParameterError::Incomplete("local producer has no effective layout".into())
        })?;
        let mut native = N::parameter_read_cost(&layout.physical, &fragments)?;
        native.host_bytes = add(
            native.host_bytes,
            layout.selection_metadata_bytes(parameter)?,
        )?;
        cost = cost.checked_add(layout.price_loan(native)?)?;
    }
    budget.reserve_quota(cost)?;
    Ok(PreparedRead {
        plan,
        fragments,
        dtype: descriptor.dtype.expect("supported floating parameter"),
        assembly: RefCell::new(assembly),
    })
}

pub(super) fn partition_read<B, N>(
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
    let catalog = partition_catalog::<B, N>(session, Some(limits))?;
    let retained = &session.resources().retained;
    let transport = N::control_transport(retained)
        .expect("completed partition catalog")
        .clone();
    let (owner, binding) = session.parameter_operation(session.parameter_epoch)?;
    let mut budget = session.parameter_budget(limits, None);
    let prepared = prepare_read::<B, N>(
        &catalog,
        identity,
        parameter,
        region,
        projection,
        transport.parameter_rank(),
        &mut budget,
    );
    let (prepared, admission) = match prepared {
        Ok(prepared) => {
            let admission = ParameterReadPreparation::new(
                (),
                *prepared.plan.identity(),
                prepared.plan.max_rank_words(),
            );
            (Some(prepared), Ok(admission))
        }
        Err(error) => (None, Err(error)),
    };
    let result = owner.read(
        &transport,
        binding,
        &parameter_read_intent(parameter, region, projection),
        if projection.is_some() {
            ParameterOperationKind::Projection
        } else {
            ParameterOperationKind::Query
        },
        admission,
        &mut budget,
        |_| {
            let prepared = prepared.as_ref().expect("admitted query");
            if prepared.fragments.is_empty() {
                return Ok(Vec::new());
            }
            session
                .with_model_transaction(|model| {
                    let layout = &catalog.layouts[parameter];
                    super::loans::with_parameter_values::<B, N, _>(
                        model,
                        layout,
                        parameter,
                        context,
                        |selected| {
                            N::read_parameter_fragments(
                                selected,
                                &layout.physical,
                                parameter,
                                &prepared.fragments,
                                prepared.plan.rank_counts()[transport.parameter_rank()],
                                context,
                            )
                        },
                    )
                })
                .map_err(parameter_failure)
        },
        |rows| {
            let prepared = prepared.as_ref().expect("admitted query");
            let rows = rows
                .iter()
                .map(|row| row.iter().copied().map(f32::from_bits).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            prepared
                .plan
                .assemble(&rows, &mut *prepared.assembly.borrow_mut())
        },
    );
    let values = finish_control::<B, N, _>(session, &transport, result)?;
    let prepared = prepared.expect("completed query");

    Ok((
        prepared.dtype,
        prepared.plan.output_shape().to_vec(),
        values,
    ))
}
