//! Public bounded queries and contractions over actual global loaded ownership.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataAllocation};
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
    result: super::super::result::PreparedResult,
}

impl MlxModelSession {
    fn completed_parameter_fragments(
        &mut self,
        parameter: &str,
        prepared: &PreparedRead,
        rank: usize,
        host: &eredu_core::HostPreparationAuthority,
        funding: &HostMetadataFunding,
    ) -> Result<Option<Vec<u32>>, ParameterError> {
        if prepared
            .fragments
            .iter()
            .any(|(_, projection)| projection.is_some())
        {
            return Ok(None);
        }
        let mut words = funding
            .metadata_vec(prepared.plan.rank_counts()[rank])
            .map_err(|cause| failure(Error::Neural(cause)))?;
        for (region, _) in &prepared.fragments {
            let count =
                usize::try_from(elements(&region.shape)?).map_err(|_| ParameterError::Overflow)?;
            let controls = WorkspaceContext::metadata_vec_bytes::<f32>(count)
                .ok_or(ParameterError::Overflow)?;
            funding
                .reserve_metadata(controls)
                .map_err(|cause| failure(Error::WorkspacePlanning(cause)))?;
            let Some(values) = self.query_completed_parameter(parameter, region, host)? else {
                return Ok(None);
            };
            if words
                .len()
                .checked_add(values.len())
                .is_none_or(|length| length > words.capacity())
            {
                return Err(ParameterError::Overflow);
            }
            words.extend(values.into_iter().map(f32::to_bits));
        }
        Ok(Some(words))
    }

    pub(in super::super) fn query_partition_parameter(
        &mut self,
        identity: &str,
        parameter: &str,
        region: ParameterRegion,
        limits: CaptureUsage,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
    ) -> Result<SharedParameterValues, ParameterError> {
        let output =
            self.read_partition_parameter(identity, parameter, &region, None, limits, environment)?;
        output.result.finish_query(ParameterValues {
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
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
    ) -> Result<SharedParameterProjectionValues, ParameterError> {
        let output = self.read_partition_parameter(
            identity,
            parameter,
            &projection.region,
            Some(&projection),
            limits,
            environment,
        )?;
        output.result.finish_projection(ParameterProjectionValues {
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
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
    ) -> Result<ReadOutput, ParameterError> {
        let catalog = self.partition_parameter_catalog(Some(limits))?;
        let transport = self
            .payload
            .distributed
            .clone()
            .expect("completed distributed catalogue");
        let owner = transport.parameter_operations()?;
        let prepared_transport = self.prepared_parameter_transport(&transport)?;
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
        let prepared = prepared.and_then(|prepared| {
            let result = if projection.is_some() {
                super::super::result::PreparedResult::projection(
                    &self.payload.memory_ledger,
                    identity,
                    parameter,
                    prepared.plan.output_shape(),
                    prepared.plan.output_shape().len(),
                )?
            } else {
                super::super::result::PreparedResult::query(
                    &self.payload.memory_ledger,
                    identity,
                    parameter,
                    region,
                )?
            };
            Ok((prepared, result))
        });
        let (prepared, result_funding, admission) = match prepared {
            Ok((prepared, result)) => {
                let admission = ParameterReadPreparation::new(
                    (),
                    *prepared.plan.identity(),
                    prepared.plan.max_rank_words(),
                );
                (Some(prepared), Some(result), Ok(admission))
            }
            Err(error) => (None, None, Err(error)),
        };
        let result = owner.read(
            &prepared_transport,
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
                if let Some(words) = self.completed_parameter_fragments(
                    parameter,
                    prepared,
                    transport.parameter_rank(),
                    result_funding
                        .as_ref()
                        .expect("admitted result")
                        .host_authority(),
                    prepared_transport.funding(),
                )? {
                    return Ok(words);
                }
                let layout = &catalog.layouts[parameter];
                let mut output = prepared_transport
                    .funding()
                    .metadata_vec(prepared.plan.rank_counts()[transport.parameter_rank()])
                    .map_err(|cause| failure(Error::Neural(cause)))?;
                for (region, projection) in &prepared.fragments {
                    let request = match projection {
                        None => super::super::numerical::Request::Read(region),
                        Some(projection) => {
                            super::super::numerical::Request::Project(projection.projection())
                        }
                    };
                    let values = self
                        .read_resident_effective_parameter(parameter, layout, request, environment)
                        .map_err(failure)?
                        .ok_or_else(|| {
                            failure(Error::PrefillControl(
                                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                            ))
                        })?;
                    if output
                        .len()
                        .checked_add(values.len())
                        .is_none_or(|n| n > output.capacity())
                    {
                        return Err(ParameterError::Overflow);
                    }
                    output.extend(values.into_iter().map(f32::to_bits));
                }
                Ok(output)
            },
            |rows| {
                let prepared = prepared.as_ref().expect("admitted query");
                let funding = prepared_transport.funding();
                let mut decoded = funding
                    .metadata_vec(rows.len())
                    .map_err(|cause| failure(Error::Neural(cause)))?;
                for row in rows {
                    let mut values = funding
                        .metadata_vec(row.len())
                        .map_err(|cause| failure(Error::Neural(cause)))?;
                    values.extend(row.iter().copied().map(f32::from_bits));
                    decoded.push(values);
                }
                prepared
                    .plan
                    .assemble(&decoded, &mut *prepared.assembly.borrow_mut())
            },
        );
        let values = self.finish_parameter_control(&transport, result)?;
        let prepared = prepared.expect("completed query");
        Ok(ReadOutput {
            dtype: prepared.dtype,
            shape: prepared.plan.output_shape().to_vec(),
            values,
            result: result_funding.expect("completed result allocation grant"),
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
