//! Exact selected local-route stages, with descriptive native pair envelopes.
use super::*;
use eredu_nn::workspace::WorkspaceAllocationPopulation;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertLocalStage {
    Slice { value: usize },
    Count { value: usize, step: usize },
    Data { value: usize, step: usize },
    Receive { value: usize, step: usize },
    Join,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpertLocalStageBound {
    pub(crate) kind: ExpertLocalStage,
    pub(crate) numerical: Option<ExpertReorderEnvelope>,
    capacity: BoundaryStageCapacity,
    births: usize,
    rows: usize,
    width: i32,
    dtype: safemlx::Dtype,
    ordinary: Option<OrdinaryParallelControls>,
}
impl ExpertLocalStageBound {
    pub(crate) fn covers_pair(
        self,
        send: &[i32],
        receive: &[i32],
        dtype: safemlx::Dtype,
        graph: usize,
        records: usize,
        backing: usize,
        births: usize,
    ) -> bool {
        let shapes = match self.kind {
            ExpertLocalStage::Count { .. } => {
                send == [1] && receive == [1] && dtype == safemlx::Dtype::Int32
            }
            ExpertLocalStage::Data { .. } => [send, receive].into_iter().all(|shape| {
                shape.len() == 2
                    && shape[1] == self.width
                    && usize::try_from(shape[0]).is_ok_and(|rows| rows <= self.rows)
            }),
            _ => false,
        };
        self.numerical.is_none()
            && shapes
            && dtype == self.dtype
            && graph <= self.capacity.graph
            && records <= self.capacity.records
            && backing <= self.capacity.backing
            && births <= self.births
    }
}
pub(crate) struct ExpertLocalTransport {
    stages: Vec<ExpertLocalStageBound>,
    stage_scratch: Vec<Vec<Option<WorkspaceAllocationPopulation>>>,
    pub(crate) ordinary_count_scratch: Option<WorkspaceAllocationPopulation>,
    pub(crate) capacity: BoundaryStageCapacity,
    pub(crate) births: usize,
    ordinary_caller: Option<usize>,
}
impl ExpertLocalTransport {
    pub(super) fn prepare(
        source: &OriginalParallelSource,
        mechanism: ResidentExecutionMechanisms,
        order: usize,
        width: i32,
        dtype: safemlx::Dtype,
        send_rows: usize,
        receive_rows: usize,
    ) -> Result<Self, Error> {
        let context =
            WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())?;
        context.charge_metadata(size_of::<(
            Self,
            Result<Self, Error>,
            ExpertLocalStageBound,
            Vec<Vec<Option<WorkspaceAllocationPopulation>>>,
            Vec<Option<WorkspaceAllocationPopulation>>,
            WorkspaceContext,
            [usize; 3],
            [i32; 2],
            [i32; 2],
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "expert local transport has no exact route source"
            ))
        };
        let actual = source
            .communication_source()
            .map_err(|cause| source.neural_error(cause))?;
        let (group, descriptor, _) = actual.group(order).ok_or_else(invalid)?;
        let plan = group
            .logical_variable_route_plan()
            .map_err(|_| invalid())?
            .ok_or_else(invalid)?;
        if !std::ptr::eq(plan.group(), group)
            || plan.len() != descriptor.members().len()
            || descriptor.local_index() != Some(group.rank())
            || width <= 0
        {
            return Err(invalid());
        }
        // The ordinary pair creates one eager I32 wire value and its stream
        // Copy alias. The prepared-input mechanism has its own retained source.
        let mut ordinary_count_scratch = None;
        let ordinary_count = if mechanism.allocation().original_storage {
            None
        } else {
            context.charge_metadata(size_of::<(
                WorkspaceTraceReport,
                SpeculativeNumericalRecipe,
                WorkspaceTensor,
                [i32; 1],
                Option<SpeculativeNumericalRecipe>,
                Option<WorkspaceAllocationPopulation>,
            )>())?;
            context.begin_span();
            let seed = WorkspaceTensor::from_i32_slice(&[0], &[1], &context)?;
            ordinary_count_scratch = context.new_allocation_scratch()?;
            let report = context.finish_report(&[seed])?;
            Some(super::super::numerical(&report, 1, mechanism, &context)?)
        };
        let exchanges = plan
            .values()
            .try_fold(0usize, |count, route| {
                let hops = (0..route.steps())
                    .filter(|&step| route.exchange(step).is_some())
                    .count();
                count.checked_add(hops)
            })
            .ok_or_else(invalid)?;
        let count = exchanges
            .checked_mul(3)
            .and_then(|n| n.checked_add(plan.len()))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(invalid)?;
        let ordinary_caller =
            crate::backend::runtime::distributed::ordinary_local_variable_control_bytes(
                plan.len(),
                exchanges,
            );
        let mut stages = context.metadata_vec(count)?;
        let mut stage_scratch = context.metadata_vec(count)?;
        let mut capacity = BoundaryStageCapacity {
            graph: 0,
            records: 0,
            backing: 0,
        };
        let mut births = 0usize;
        let mut push = |stage: ExpertLocalStageBound,
                        scratch: Vec<Option<WorkspaceAllocationPopulation>>|
         -> Result<(), Error> {
            if stages.len() == count {
                return Err(invalid());
            }
            capacity.graph = capacity.graph.max(stage.capacity.graph);
            capacity.records = capacity.records.max(stage.capacity.records);
            capacity.backing = capacity
                .backing
                .checked_add(stage.capacity.backing)
                .ok_or_else(invalid)?;
            births = births.checked_add(stage.births).ok_or_else(invalid)?;
            stages.push(stage);
            stage_scratch.push(scratch);
            Ok(())
        };
        context.charge_metadata(std::mem::size_of_val(&push))?;
        // A single matrix edge is bounded by both its global sender and receiver.
        // The pair worker branches only on zero versus positive extent; include
        // singleton and upper geometry independently in both directions.
        let edge_rows = send_rows.min(receive_rows);
        for (value, route) in plan.values().enumerate() {
            let (slice, slice_scratch) = ExpertReorderEnvelope::prepare_with_scratch(
                source,
                mechanism,
                send_rows,
                width,
                dtype,
                plan.len(),
            )?;
            push(
                ExpertLocalStageBound {
                    kind: ExpertLocalStage::Slice { value },
                    numerical: Some(slice),
                    capacity: slice.capacity,
                    births: slice.births,
                    rows: send_rows,
                    width,
                    dtype,
                    ordinary: slice.ordinary,
                },
                slice_scratch,
            )?;
            for step in 0..route.steps() {
                if route.exchange(step).is_none() {
                    continue;
                }
                for count in [true, false] {
                    let kind = if count {
                        ExpertLocalStage::Count { value, step }
                    } else {
                        ExpertLocalStage::Data { value, step }
                    };
                    let mut bound = ExpertLocalStageBound {
                        kind,
                        numerical: None,
                        capacity: BoundaryStageCapacity {
                            graph: 0,
                            records: 0,
                            backing: 0,
                        },
                        births: 0,
                        rows: edge_rows,
                        width,
                        dtype: if count { safemlx::Dtype::Int32 } else { dtype },
                        ordinary: Some(OrdinaryParallelControls::default()),
                    };
                    let candidates = if count {
                        [1, 1, 1]
                    } else {
                        [0, edge_rows.min(1), edge_rows]
                    };
                    for (a, &sent) in candidates.iter().enumerate() {
                        if candidates[..a].contains(&sent) {
                            continue;
                        }
                        for (b, &received) in candidates.iter().enumerate() {
                            if candidates[..b].contains(&received) {
                                continue;
                            }
                            let send = [i32::try_from(sent).map_err(|_| invalid())?, width];
                            let receive = [i32::try_from(received).map_err(|_| invalid())?, width];
                            let (send, receive) = if count {
                                (&send[..1], &receive[..1])
                            } else {
                                (&send[..], &receive[..])
                            };
                            let queried = actual
                                .group_variable_route_layout(
                                    order,
                                    value,
                                    step,
                                    send,
                                    receive,
                                    bound.dtype,
                                )
                                .map_err(|cause| source.neural_error(cause))?
                                .ok_or_else(invalid)?;
                            let owned = queried
                                .try_into_owned(source)
                                .map_err(|cause| source.neural_error(cause))?;
                            bound.capacity.graph = bound.capacity.graph.max(owned.graph_capacity());
                            bound.capacity.records =
                                bound.capacity.records.max(owned.record_capacity());
                            bound.capacity.backing =
                                bound.capacity.backing.max(owned.backing_capacity());
                            bound.births = bound
                                .births
                                .max(owned.maximum_backing_births().ok_or_else(invalid)?);
                            bound.ordinary = match (
                                bound.ordinary,
                                owned
                                    .ordinary_controls()
                                    .and_then(OrdinaryParallelControls::group),
                            ) {
                                (Some(prior), Some(value)) => Some(prior.union(value)),
                                _ => None,
                            };
                        }
                    }
                    let native_scratch = super::super::scratch::native_cpu(
                        &context,
                        mechanism,
                        bound.capacity.backing,
                        bound.births,
                    )?;
                    if count {
                        if let Some(seed) = ordinary_count {
                            bound.capacity.backing = bound
                                .capacity
                                .backing
                                .checked_add(
                                    usize::try_from(seed.storage.mutable_bytes())
                                        .map_err(|_| invalid())?,
                                )
                                .ok_or_else(invalid)?;
                            bound.births = bound
                                .births
                                .checked_add(seed.storage.maximum_births())
                                .ok_or_else(invalid)?;
                            bound.ordinary = bound.ordinary.and_then(|value| value.append(
                                OrdinaryParallelControls::constructed(seed)?.metadata(
                                    crate::backend::runtime::distributed::completion::ordinary_completed_i32_scalar_control_bytes()?
                                )?
                            ));
                        }
                    }
                    let mut source_rows = context.metadata_vec(2)?;
                    source_rows.push((&native_scratch, 1));
                    if count {
                        if let Some(seed) = &ordinary_count_scratch {
                            source_rows.push((seed, 1));
                        }
                    }
                    let population = super::super::scratch::simultaneous(&context, &source_rows)?;
                    let mut alternatives = context.metadata_vec(1)?;
                    alternatives.push(Some(population));
                    push(bound, alternatives)?;
                    if count {
                        let (empty, empty_scratch) =
                            ExpertReorderEnvelope::prepare_empty_with_scratch(
                                source, mechanism, edge_rows, width, dtype,
                            )?;
                        push(
                            ExpertLocalStageBound {
                                kind: ExpertLocalStage::Receive { value, step },
                                numerical: Some(empty),
                                capacity: empty.capacity,
                                births: empty.births,
                                rows: edge_rows,
                                width,
                                dtype,
                                ordinary: empty.ordinary,
                            },
                            empty_scratch,
                        )?;
                    }
                }
            }
        }
        let (join, join_scratch) = ExpertReorderEnvelope::prepare_join_with_scratch(
            source,
            mechanism,
            receive_rows,
            width,
            dtype,
            plan.len(),
        )?;
        push(
            ExpertLocalStageBound {
                kind: ExpertLocalStage::Join,
                numerical: Some(join),
                capacity: join.capacity,
                births: join.births,
                rows: receive_rows,
                width,
                dtype,
                ordinary: join.ordinary,
            },
            join_scratch,
        )?;
        drop(push);
        if stages.len() != count {
            return Err(invalid());
        }
        Ok(Self {
            stages,
            stage_scratch,
            ordinary_count_scratch,
            capacity,
            births,
            ordinary_caller,
        })
    }
    /// Every stage retains its actual allocation source. Neural branches are
    /// resolved per domain; native CPU branches share their established route.
    pub(crate) fn scratch(
        &self,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceAllocationPopulation>, Error> {
        let mut stages = context.metadata_vec(self.stage_scratch.len())?;
        for rows in &self.stage_scratch {
            let Some(source) = super::super::scratch::alternatives(context, rows)? else {
                return Ok(None);
            };
            stages.push(source);
        }
        let mut sources = context.metadata_vec(stages.len())?;
        sources.extend(stages.iter().map(|source| (source, 1)));
        super::super::scratch::simultaneous(context, &sources).map(Some)
    }
    /// Actual source alternatives for one executed itinerary stage.
    pub(crate) fn stage_scratch(
        &self,
        index: usize,
    ) -> Option<&[Option<WorkspaceAllocationPopulation>]> {
        self.stage_scratch.get(index).map(Vec::as_slice)
    }
    pub(crate) fn stages(&self) -> impl Iterator<Item = &ExpertLocalStageBound> {
        self.stages.iter()
    }
    pub(crate) fn len(&self) -> usize {
        self.stages.len()
    }
    pub(crate) fn stage(&self, index: usize) -> Option<ExpertLocalStageBound> {
        self.stages.get(index).copied()
    }
    pub(crate) fn ordinary_controls(&self) -> Option<OrdinaryParallelControls> {
        self.stages
            .iter()
            .try_fold(OrdinaryParallelControls::default(), |total, stage| {
                total.append(stage.ordinary?)
            })?
            .metadata(self.ordinary_caller?)
    }
}
