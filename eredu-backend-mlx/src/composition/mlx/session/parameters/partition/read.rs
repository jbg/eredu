//! Public bounded queries and contractions over actual global loaded ownership.
use super::*;
use std::cell::RefCell;

struct PreparedRead {
    plan: PartitionParameterReadPlan,
    fragments: Vec<(ParameterRegion, Option<ParameterProjectionFragment>)>,
    dtype: InterventionDtype,
    assembly: RefCell<CaptureQuota>,
}
struct ReadOutput {
    dtype: InterventionDtype,
    shape: Vec<u64>,
    values: Vec<f32>,
}

impl MlxModelSession {
    pub(in super::super) fn query_partition_parameter(
        &mut self,
        identity: &str,
        parameter: &str,
        region: ParameterRegion,
        limits: CaptureUsage,
        stream: &Stream,
    ) -> Result<ParameterValues, ParameterError> {
        let output =
            self.read_partition_parameter(identity, parameter, &region, None, limits, stream)?;
        Ok(ParameterValues {
            identity: identity.into(),
            parameter: parameter.into(),
            dtype: output.dtype,
            region,
            values: output.values,
            usage: self.payload.parameter_state.usage.get(),
        })
    }
    pub(in super::super) fn project_partition_parameter(
        &mut self,
        identity: &str,
        parameter: &str,
        projection: ParameterProjection,
        limits: CaptureUsage,
        stream: &Stream,
    ) -> Result<ParameterProjectionValues, ParameterError> {
        let output = self.read_partition_parameter(
            identity,
            parameter,
            &projection.region,
            Some(&projection),
            limits,
            stream,
        )?;
        Ok(ParameterProjectionValues {
            identity: identity.into(),
            parameter: parameter.into(),
            source_dtype: output.dtype,
            shape: output.shape,
            values: output.values,
            usage: self.payload.parameter_state.usage.get(),
        })
    }
    fn read_partition_parameter(
        &mut self,
        identity: &str,
        parameter: &str,
        region: &ParameterRegion,
        projection: Option<&ParameterProjection>,
        limits: CaptureUsage,
        stream: &Stream,
    ) -> Result<ReadOutput, ParameterError> {
        let catalog = self.partition_parameter_catalog(Some(limits))?;
        let transport = self
            .payload
            .distributed
            .clone()
            .expect("completed distributed catalogue");
        let owner = transport.parameter_operations()?;
        let binding = self.parameter_operation_binding()?;
        let mut budget = NativeParameterBudget {
            total: Rc::clone(&self.payload.parameter_state.usage),
            limit: limits,
            operation: None,
        };
        let prepared = prepare_read(
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
                let layout = &catalog.layouts[parameter];
                let mut ids = BTreeSet::from([parameter.to_owned()]);
                layout.extend_dependencies(&mut ids);
                self.with_model_operation(|model| {
                    with_selected_parameter_values(model.erased_mut(), &ids, stream, |selected| {
                        let tensor = layout.effective(parameter, selected, stream)?;
                        let mut output = Vec::with_capacity(
                            prepared.plan.rank_counts()[transport.parameter_rank()],
                        );
                        for (region, projection) in &prepared.fragments {
                            let values = match projection {
                                None => encoding::read_effective(&tensor, region, stream)?,
                                Some(projection) => encoding::project_effective(
                                    &tensor,
                                    projection.projection(),
                                    stream,
                                )?,
                            };
                            output.extend(values.into_iter().map(f32::to_bits));
                        }
                        Ok(output)
                    })
                })
                .map_err(failure)
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
        let values = self.finish_parameter_control(&transport, result)?;
        let prepared = prepared.expect("completed query");
        Ok(ReadOutput {
            dtype: prepared.dtype,
            shape: prepared.plan.output_shape().to_vec(),
            values,
        })
    }
}

fn prepare_read(
    catalog: &GlobalParameterCatalog,
    identity: &str,
    parameter: &str,
    region: &ParameterRegion,
    projection: Option<&ParameterProjection>,
    rank: usize,
    budget: &mut NativeParameterBudget,
) -> Result<PreparedRead, ParameterError> {
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
    // Producer words, rank-wise F32 decoding, final envelope and selected native
    // temporaries are paid before any parameter loan or producer submission.
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
        let selected = elements(&fragment.source().local().shape)?;
        let count = elements(&fragment.destination().shape)?;
        let directions = projection
            .as_ref()
            .map_or(0, |p| p.projection().coefficients.len() as u64);
        cost.retained_bytes = add(
            cost.retained_bytes,
            add(
                mul(selected, 32)?,
                add(mul(count, 16)?, mul(directions, 8)?)?,
            )?,
        )?;
        cost.host_bytes = add(cost.host_bytes, add(mul(count, 16)?, mul(directions, 8)?)?)?;
        fragments.push((fragment.source().local().clone(), projection));
    }
    if !fragments.is_empty() {
        let layout = catalog.layouts.get(parameter).ok_or_else(|| {
            ParameterError::Incomplete("local producer has no effective layout".into())
        })?;
        let source = elements(&layout.shape)?;
        if source > i32::MAX as u64 {
            return Err(ParameterError::Unsupported(
                "local native parameter exceeds i32 indexing".into(),
            ));
        }
        cost.retained_bytes = add(
            cost.retained_bytes,
            add(
                mul(
                    source,
                    if layout.format == eredu_checkpoint::LinearFormat::Dense {
                        8
                    } else {
                        32
                    },
                )?,
                layout.conversion_and_loan_bytes()?,
            )?,
        )?;
        cost.host_bytes = add(cost.host_bytes, layout.host_conversion_bytes()?)?;
        cost.host_bytes = add(cost.host_bytes, layout.selection_metadata_bytes(parameter)?)?;
    }
    budget.reserve_quota(cost)?;
    Ok(PreparedRead {
        plan,
        fragments,
        dtype: descriptor.dtype.expect("supported floating parameter"),
        assembly: RefCell::new(assembly),
    })
}
