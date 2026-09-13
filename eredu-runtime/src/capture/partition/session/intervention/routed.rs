//! Sparse operations use the same admitted owner, epoch, votes and outcome frame.
use super::*;
use eredu_core::intervention::*;
use std::collections::BTreeSet;

pub(super) struct SparseWork {
    routing: String,
    geometry: RoutedUnitGeometry,
    pub(super) source_tokens: u64,
    slice: ResolvedCaptureSlice,
    local: Option<SparseLocal>,
    state: State,
    pub(super) affected: u64,
    seen: BTreeSet<(Option<u64>, u64, u64)>,
}
impl SparseWork {
    pub(super) fn maximum_affected(&self) -> Result<u64, CaptureError> {
        mul(
            mul(self.source_tokens, self.geometry.routes_per_token)?,
            self.geometry.units_per_expert,
        )
    }
}
struct SparseLocal {
    ownership: RoutedUnitCaptureOwnership,
    input_width: u64,
    reports: bool,
}
#[derive(Clone, Copy)]
enum State {
    Pending,
    Active { rows: u64, end: u64 },
    Complete,
    Failed,
}

impl<T: PartitionCaptureHookTransport> SessionPartitionIntervention<'_, T> {
    pub(in crate::capture::partition) fn routed_path(&self) -> Option<&str> {
        self.routed.as_ref().map(|work| work.routing.as_str())
    }
}

impl CaptureSession {
    // Called only after the ordinary operation preparation checks fresh authority,
    // retained artifact, active schedule and no completed common coordination.
    pub(super) fn prepare_partition_routed_intervention<'a, B, T, L>(
        &mut self,
        transport: &'a T,
        layout: &L,
        backend: &B,
        index: usize,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<SessionPartitionIntervention<'a, T>, PartitionCaptureExchangeError>
    where
        B: InterventionBackend,
        T: PartitionCaptureHookTransport,
        L: PartitionActivationLayout + PartitionCaptureLayout,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let epoch = self.partition_epoch()?;
        let world = self
            .partition
            .as_ref()
            .expect("binding checked")
            .identity
            .world_size;
        let rank = transport.capture_rank();
        let wait = transport.capture_wait()?;
        if !T::Completion::supports_cancellation(wait.cancellation()) {
            return Err(CaptureError::Unsupported(
                "sparse intervention cancellation policy".into(),
            )
            .into());
        }
        transport.ensure_capture_active()?;
        let run = self.interventions.as_ref().expect("admission checked");
        let operation = &run.plan.plan().operations[index];
        let point = &run.plan.points()[index];
        let routed = point.routed_units.as_ref().expect("sparse dispatch");
        let geometry = routed.geometry;
        if operation.evidence != InterventionEvidence::None {
            return Err(CaptureError::Unsupported(
                "sparse operation evidence requires separate routed captures".into(),
            )
            .into());
        }
        let shape = run
            .plan
            .geometry_at(self.phase, self.prediction, self.invocation)?
            .resolve(&point.observation_geometry())?
            .ok_or_else(|| invalid("unknown sparse intervention shape"))?;
        let slice = run.plan.validate_at(
            index,
            self.phase,
            self.prediction,
            self.invocation,
            &shape,
            operation.action.dtype(),
        )?;
        if shape.len() != 2 || shape[1] != geometry.components()? || limits.max_producers == 0 {
            return Err(invalid("invalid sparse intervention geometry or member bound").into());
        }
        let source_tokens = shape[0];
        let mut charged = CaptureUsage {
            host_bytes: mul(
                add(8192, mul(limits.max_producers as u64, 1024)?)?,
                world as u64,
            )?,
            ..Default::default()
        };
        self.ledger.reserve_quota(charged)?;
        let projections =
            layout.routed_activation_members(&run.plan, index, limits.max_producers)?;
        let members = projections
            .iter()
            .map(|member| member.rank)
            .collect::<Vec<_>>();
        if members.is_empty()
            || members.len() > limits.max_producers
            || members.windows(2).any(|pair| pair[0] >= pair[1])
            || members.iter().any(|rank| *rank >= world)
        {
            return Err(invalid("invalid sparse invocation membership").into());
        }
        let first = projections[0].ownership;
        let mut digest = Sha256::new();
        digest.update(b"eredu-partition-sparse-intervention-v1\0");
        digest.update(run.plan.intent_identity().as_bytes());
        digest.update((index as u64).to_le_bytes());
        let mut local = None;
        let mut reported: Vec<&eredu_core::component::RoutedComponentCoordinateMap> = Vec::new();
        let mut coverage = 0u64;
        for member in projections {
            let owned = member.ownership;
            owned.validate(geometry)?;
            if member.input_width == 0
                || owned.source_peer != first.source_peer
                || owned.source_peers != first.source_peers
            {
                return Err(invalid("sparse intervention source geometry disagrees").into());
            }
            // Repeated physical coordinate replicas all execute, but only one
            // contributes counts. Every EP source is edited; only publication
            // peer rows count toward the logical single-sequence outcome.
            let reports = !reported.contains(&&owned.coordinates);
            if reports {
                if reported.iter().any(|other| {
                    super::super::super::routed::maps_overlap(
                        other.experts(),
                        owned.coordinates.experts(),
                    ) && super::super::super::routed::maps_overlap(
                        other.units(),
                        owned.coordinates.units(),
                    )
                }) {
                    return Err(invalid("sparse intervention ownership partially overlaps").into());
                }
                coverage = add(
                    coverage,
                    mul(
                        owned.coordinates.experts().local_count() as u64,
                        owned.coordinates.units().local_count() as u64,
                    )?,
                )?;
                reported.push(&owned.coordinates);
            }
            let rows = owned.maximum_source_rows(source_tokens, geometry.routes_per_token)?;
            let width = if owned.source_peer.is_some() {
                1
            } else {
                geometry.routes_per_token
            };
            let units = owned.coordinates.units().local_count() as u64;
            let native = run.estimator.partition_routed_unit_usage(
                geometry,
                source_tokens,
                owned,
                &slice,
                &operation.action,
            )?;
            if native.captures != 0 || native.encoded_bytes != 0 {
                return Err(invalid("sparse edit estimate includes unrelated records").into());
            }
            let dependencies = backend
                .estimate_partition_source(&[rows, member.input_width], wait)?
                .checked_add(
                    backend
                        .estimate_partition_source(&[1, mul(width, units)?], wait)?
                        .checked_mul(rows)?,
                )?;
            let metadata = CaptureUsage {
                host_bytes: mul(
                    super::super::super::routed::ownership_bytes(owned)?,
                    world as u64,
                )?,
                ..Default::default()
            };
            let usage = native.checked_add(dependencies)?.checked_add(metadata)?;
            self.ledger.reserve_quota(usage)?;
            charged = charged.checked_add(usage)?;
            digest.update((member.rank as u64).to_le_bytes());
            digest.update(member.input_width.to_le_bytes());
            digest.update([u8::from(reports)]);
            digest.update(
                serde_json::to_vec(owned).map_err(|_| invalid("sparse ownership encoding"))?,
            );
            hash_usage(&mut digest, usage);
            if member.rank == rank {
                local = Some(SparseLocal {
                    ownership: owned.clone(),
                    input_width: member.input_width,
                    reports,
                });
            }
        }
        if coverage != geometry.components()? {
            return Err(invalid("sparse intervention ownership is incomplete").into());
        }
        let votes = transport
            .estimate_capture_hook(&members)?
            .checked_mul(mul(members.len() as u64, 2)?)?;
        let receipt = super::super::super::exchange::gather_usage(transport, world, RECEIPT_WORDS)?
            .checked_mul(world as u64)?;
        let coordination = votes.checked_add(receipt)?;
        self.ledger.reserve_quota(coordination)?;
        charged = charged.checked_add(coordination)?;
        hash_usage(&mut digest, charged);
        let descriptor = digest.finalize().into();
        let plan_identity = run.plan.identity().to_owned();
        charged =
            charged.checked_add(run.records.as_ref().expect("active record")[index].charged)?;
        self.partition
            .as_mut()
            .expect("binding checked")
            .intervention_claimed
            .insert(index, descriptor);
        Ok(SessionPartitionIntervention {
            owner: Arc::clone(&self.owner),
            epoch,
            index,
            plan_identity,
            transport,
            members,
            wait,
            local: None,
            attempted: false,
            accepted: false,
            descriptor,
            charged,
            evidence: vec![],
            routed: Some(SparseWork {
                routing: routed.routing.clone(),
                geometry,
                source_tokens,
                slice,
                local,
                state: State::Pending,
                affected: 0,
                seen: BTreeSet::new(),
            }),
        })
    }

    /// Preflight the actual provider input before native work. Every member casts
    /// its prepaid source vote, including idle receivers and earlier failures.
    pub fn begin_partition_routed_intervention<B, T>(
        &mut self,
        work: &mut SessionPartitionIntervention<'_, T>,
        backend: &mut B,
        invocation: &crate::RoutedUnitInvocation<'_, B::Tensor>,
        success: bool,
    ) -> Result<(), PartitionCaptureObserverError<B::Error>>
    where
        B: InterventionBackend,
        T: PartitionCaptureHookTransport,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_intervention(work)
            .map_err(CaptureExecutionError::Admission)?;
        let valid: Result<u64, CaptureExecutionError<B::Error>> = (|| {
            let sparse = work
                .routed
                .as_mut()
                .ok_or_else(|| invalid("not a sparse operation"))?;
            if work.attempted || !matches!(sparse.state, State::Pending) {
                return Err(invalid("sparse operation already attempted").into());
            }
            work.attempted = true;
            sparse.state = State::Failed;
            let local = sparse
                .local
                .as_ref()
                .ok_or_else(|| invalid("sparse invocation is not local"))?;
            let shape = backend
                .shape(invocation.input)
                .map_err(CaptureExecutionError::Backend)?;
            let rows = super::super::routed_input_rows(&shape)?;
            if !success
                || shape.last() != Some(&local.input_width)
                || rows
                    > local.ownership.maximum_source_rows(
                        sparse.source_tokens,
                        sparse.geometry.routes_per_token,
                    )?
                || (local.ownership.coordinates.experts().local_count() == 0 && rows != 0)
                || invocation
                    .unit_coordinates
                    .is_some_and(|units| units != local.ownership.coordinates.units())
                || !matches!(
                    backend.source_dtype(invocation.input),
                    Some(
                        TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32 | TensorDtype::F64
                    )
                )
            {
                return Err(invalid("sparse input differs from admitted source").into());
            }
            validate_origins(
                sparse,
                rows,
                invocation
                    .origins
                    .map(|origins| origins.capture_coordinates()),
            )?;
            Ok(rows)
        })();
        let agreed = hook::agree(work.transport, &work.members, work.wait, valid.is_ok());
        let result = (|| {
            let rows = valid?;
            if !agreed? {
                return Err(PartitionCaptureObserverError::HookRejected);
            }
            prepare_value(work, backend, invocation.input)?;
            work.routed.as_mut().expect("sparse").state = State::Active { rows, end: 0 };
            Ok(())
        })();
        self.record_sparse_failure(work.index, &result);
        result
    }

    /// Apply one native chunk in original operation order. All source peers and
    /// replicas are updated; outcome counting follows retained publication ownership.
    pub fn apply_partition_routed_intervention<B, T>(
        &mut self,
        work: &mut SessionPartitionIntervention<'_, T>,
        backend: &mut B,
        source: &PartitionRoutedUnitCaptureSource<'_, B::Tensor>,
    ) -> Result<Option<B::Tensor>, PartitionCaptureObserverError<B::Error>>
    where
        B: InterventionBackend,
        T: PartitionCaptureHookTransport,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_intervention(work)
            .map_err(CaptureExecutionError::Admission)?;
        let mut sparse = work
            .routed
            .take()
            .ok_or_else(|| CaptureExecutionError::Admission(invalid("not a sparse operation")))?;
        let state = std::mem::replace(&mut sparse.state, State::Failed);
        let result = (|| {
            let State::Active { rows, end } = state else {
                return Err(CaptureExecutionError::Admission(invalid(
                    "sparse chunk has no active authority",
                ))
                .into());
            };
            let local = sparse
                .local
                .as_ref()
                .ok_or_else(|| invalid("missing local sparse work"))?;
            let owned = &local.ownership;
            validate_origins(&sparse, rows, source.origins)?;
            let width = if source.origins.is_some() {
                1
            } else {
                sparse.geometry.routes_per_token
            };
            let units = owned.coordinates.units().local_count() as u64;
            let shape = backend
                .shape(source.source.values)
                .map_err(CaptureExecutionError::Backend)?;
            let coefficients = backend
                .shape(source.source.coefficients)
                .map_err(CaptureExecutionError::Backend)?;
            let groups = backend
                .shape(source.source.source_groups)
                .map_err(CaptureExecutionError::Backend)?;
            let next = add(end, coefficients.first().copied().unwrap_or(0))?;
            let action = &self
                .interventions
                .as_ref()
                .expect("plan")
                .plan
                .plan()
                .operations[work.index]
                .action;
            let dtype = backend
                .intervention_dtype(source.source.values)
                .map_err(CaptureExecutionError::Backend)?;
            if source.unit_coordinates != owned.coordinates.units()
                || dtype != action.dtype().ok_or_else(|| invalid("sparse dtype"))?
                || coefficients.len() != 2
                || coefficients[1] != width
                || groups != [rows, width]
                || shape != [mul(next - end, width)?, units]
                || source.source.token_offset != end
                || next <= end
                || next > rows
            {
                return Err(CaptureExecutionError::Admission(invalid(
                    "sparse chunk geometry, dtype or coverage changed",
                ))
                .into());
            }
            if units == 0 {
                sparse.state = State::Active { rows, end: next };
                return Ok(None);
            }
            let unavailable = || {
                CaptureError::Unsupported(
                    "partition sparse edit native primitive unavailable".into(),
                )
            };
            let locations = backend
                .partition_routed_unit_locations(source, sparse.geometry)
                .ok_or_else(unavailable)?
                .map_err(CaptureExecutionError::Backend)?;
            if locations.source_token_range != [end, next]
                || locations.rows.len() as u64 != shape[0]
                || locations.rows.iter().any(|row| {
                    (owned.source_peer.is_some() != row.source_peer.is_some())
                        || row
                            .source_peer
                            .is_some_and(|peer| peer >= owned.source_peers)
                        || !sparse.seen.insert((row.source_peer, row.token, row.slot))
                })
            {
                return Err(CaptureExecutionError::Admission(invalid(
                    "sparse native coordinates changed range or repeated a route",
                ))
                .into());
            }
            let lowered = crate::intervention::lower_partition_routed_intervention(
                sparse.geometry,
                sparse.source_tokens,
                &locations.rows,
                &owned.coordinates,
                &sparse.slice,
                action,
            )?;
            let affected = if local.reports && units > 0 {
                lowered
                    .indices
                    .iter()
                    .filter(|index| {
                        locations.rows[(**index / units) as usize].source_peer == owned.source_peer
                    })
                    .count() as u64
            } else {
                0
            };
            let output = if let Some(action) = lowered.action {
                let input = source.source.values;
                let selected = backend
                    .select_elements(input, &lowered.indices)
                    .ok_or_else(unavailable)?
                    .map_err(CaptureExecutionError::Backend)?;
                let n = lowered.indices.len() as u64;
                if backend
                    .shape(&selected)
                    .map_err(CaptureExecutionError::Backend)?
                    != [n]
                    || backend
                        .intervention_dtype(&selected)
                        .map_err(CaptureExecutionError::Backend)?
                        != dtype
                {
                    return Err(CaptureExecutionError::Admission(invalid(
                        "sparse gather changed shape or dtype",
                    ))
                    .into());
                }
                let slice = ResolvedCaptureSlice {
                    starts: vec![0],
                    ends: vec![n],
                    strides: vec![1],
                    shape: vec![n],
                };
                let replacement =
                    crate::intervention::apply_activation(backend, &selected, &action, &slice)?;
                let output = backend
                    .update_elements(input, &lowered.indices, &replacement)
                    .ok_or_else(unavailable)?
                    .map_err(CaptureExecutionError::Backend)?;
                if backend
                    .shape(&output)
                    .map_err(CaptureExecutionError::Backend)?
                    != shape
                    || backend
                        .intervention_dtype(&output)
                        .map_err(CaptureExecutionError::Backend)?
                        != dtype
                {
                    return Err(CaptureExecutionError::Admission(invalid(
                        "sparse edit changed shape or dtype",
                    ))
                    .into());
                }
                prepare_value(work, backend, &output)?;
                Some(output)
            } else {
                None
            };
            sparse.affected = add(sparse.affected, affected)?;
            sparse.state = State::Active { rows, end: next };
            Ok(output)
        })();
        work.routed = Some(sparse);
        self.record_sparse_failure(work.index, &result);
        result
    }

    /// Require all native chunks and provider success before the final prepaid
    /// invocation vote permits reverse exchange or downstream collectives.
    pub fn finish_partition_routed_intervention<B, T>(
        &mut self,
        work: &mut SessionPartitionIntervention<'_, T>,
        success: bool,
    ) -> Result<(), PartitionCaptureObserverError<B>>
    where
        B: std::error::Error + 'static,
        T: PartitionCaptureHookTransport,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_intervention(work)
            .map_err(CaptureExecutionError::Admission)?;
        let sparse = work
            .routed
            .as_mut()
            .ok_or_else(|| CaptureExecutionError::Admission(invalid("not a sparse operation")))?;
        let previous = std::mem::replace(&mut sparse.state, State::Failed);
        let valid = success
            && work.attempted
            && matches!(previous, State::Active { rows, end } if rows == end);
        let agreed = hook::agree(work.transport, &work.members, work.wait, valid);
        let result = (|| {
            if !valid {
                return Err(CaptureExecutionError::Admission(invalid(
                    "sparse operation failed or omitted chunks",
                ))
                .into());
            }
            if !agreed? {
                return Err(PartitionCaptureObserverError::HookRejected);
            }
            work.routed.as_mut().expect("sparse").state = State::Complete;
            work.accepted = true;
            Ok(())
        })();
        if result.is_err() {
            work.routed.as_mut().expect("sparse").affected = 0;
        }
        self.record_sparse_failure(work.index, &result);
        result
    }

    fn record_sparse_failure<O, E: std::error::Error>(
        &mut self,
        index: usize,
        result: &Result<O, E>,
    ) {
        if let Err(error) = result {
            if let Some(record) = self
                .interventions
                .as_mut()
                .and_then(|run| run.records.as_mut())
                .and_then(|records| records.get_mut(index))
            {
                if !matches!(record.outcome, InterventionOutcome::Failed { .. }) {
                    record.outcome = InterventionOutcome::Failed {
                        message: bounded_diagnostic(error),
                    };
                }
            }
        }
    }
}

fn validate_origins(
    sparse: &SparseWork,
    rows: u64,
    origins: Option<eredu_core::capture::RoutedUnitOrigins<'_>>,
) -> Result<(), CaptureError> {
    let owned = &sparse
        .local
        .as_ref()
        .ok_or_else(|| invalid("no local sparse source"))?
        .ownership;
    let valid = match origins {
        Some(origins) => {
            owned.source_peer.is_some()
                && origins.row_count() as u64 == rows
                && origins.peer_count() as u64 == owned.source_peers
                && origins.routes_per_token() as u64 == sparse.geometry.routes_per_token
        }
        None => owned.source_peer.is_none() && rows == sparse.source_tokens,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid("sparse source origin topology changed"))
    }
}
