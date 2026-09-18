//! Sparse chunks spend the same live authority as ordinary partition fragments.
use super::*;

pub(super) enum Producers {
    Dense(Vec<PartitionCaptureProducer>),
    Sum(Vec<PartitionCaptureProducer>),
    Routed(Vec<PartitionRoutedCaptureProducer>),
}

impl Producers {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Dense(producers) | Self::Sum(producers) => producers.len(),
            Self::Routed(producers) => producers.len(),
        }
    }

    pub(super) fn preparation_usage(&self) -> Result<CaptureUsage, CaptureError> {
        match self {
            Self::Dense(producers) | Self::Sum(producers) => {
                PartitionCaptureReceiptPlan::preparation_usage(producers)
            }
            Self::Routed(producers) => {
                PartitionCaptureReceiptPlan::routed_preparation_usage(producers)
            }
        }
    }

    pub(super) fn admit(
        self,
        plan: SharedCapturePlan,
        context: PartitionCaptureContext,
        world: usize,
        limits: PartitionCaptureReceiptLimits,
        quota: &mut CaptureQuota,
    ) -> Result<PartitionCaptureReceiptPlan, PartitionCaptureMergeError> {
        match self {
            Self::Dense(producers) => PartitionCaptureReceiptPlan::new_source(
                plan, context, producers, world, limits, quota,
            ),
            Self::Sum(producers) => PartitionCaptureReceiptPlan::new_sum_source(
                plan, context, producers, world, limits, quota,
            ),
            Self::Routed(producers) => PartitionCaptureReceiptPlan::new_routed_source(
                plan, context, producers, world, limits, quota,
            ),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum RoutedInvocation {
    Pending,
    Active { rows: u64, end: u64 },
    Complete,
    Failed,
}

pub(super) fn geometry(
    receipt: &PartitionCaptureReceiptPlan,
) -> Result<RoutedUnitGeometry, CaptureError> {
    match receipt.plan.points()[receipt.context.selection_index].value_type {
        eredu_core::ObservationValueType::RoutedUnits { geometry, .. } => Ok(geometry),
        _ => Err(invalid("partition selection is not routed-unit capture")),
    }
}

pub(super) fn prepare_fragments<T: PartitionCaptureTransport>(
    work: &mut SessionPartitionCapture<'_, T>,
    phase: CapturePhase,
    prediction: u64,
    costs: Vec<CaptureUsage>,
) -> Result<(), CaptureError>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    let receipt = work.exchange.receipt_plan();
    let Some(projection) = receipt.producer(work.rank) else {
        return Ok(());
    };
    let bank = geometry(receipt)?;
    let selection = &work.plan.plan().selections[work.index];
    let point = &work.plan.points()[work.index];
    if costs.len() != projection.fragments().len() {
        return Err(invalid("sparse fragment reservations changed"));
    }
    // Spend all native and retained-record credits before allocating accumulators.
    // Repeated chunks cannot obtain additional allowance or refund failed work.
    let total = costs
        .iter()
        .try_fold(CaptureUsage::default(), |sum, cost| sum.checked_add(*cost))?;
    if let Some(CaptureSkipReason::Limit { budget, cumulative }) = work.quota.reserve(total)? {
        return Err(CaptureError::Limit { budget, cumulative });
    }
    work.fragments = Some(
        projection
            .fragments()
            .iter()
            .zip(costs)
            .map(|(fragment, charged)| CapturedPartitionFragment {
                invocation: receipt.context().invocation,
                combination: PartitionCaptureCombination::Disjoint,
                plan_identity: work.plan.identity().into(),
                selection_index: work.index,
                phase,
                prediction,
                producer_rank: work.rank,
                axis: projection.axis(),
                global_shape: projection.global_shape().to_vec(),
                global_slice: projection.global_slice().clone(),
                geometry: fragment.clone(),
                record: CaptureRecord {
                    schema_version: CAPTURE_SCHEMA_VERSION,
                    selection_id: selection.id.clone(),
                    path: selection.path.clone(),
                    node_id: point.node_id.clone(),
                    position: point.position,
                    source_shape: Some(projection.local_shape().to_vec()),
                    source_dtype: None,
                    selected_shape: Some(fragment.local().shape.clone()),
                    outcome: CaptureOutcome::Missing,
                    payload: Some(CapturePayload::RoutedUnits(RoutedUnitCapture {
                        geometry: bank,
                        source_token_ranges: Vec::new(),
                        rows: Vec::new(),
                    })),
                    charged,
                },
            })
            .collect(),
    );
    Ok(())
}

impl CaptureSession {
    /// Prepares sparse receipt ownership and all producer/receiver costs within
    /// this run's ordinary ledger. Common coordination is still required before
    /// invoking a provider or collecting any native chunk.
    pub fn prepare_partition_routed_capture<'a, T: PartitionCaptureTransport>(
        &mut self,
        transport: &'a T,
        index: usize,
        producers: Vec<PartitionRoutedCaptureProducer>,
        limits: PartitionCaptureReceiptLimits,
        mut estimate: impl FnMut(
            &PartitionRoutedUnitCaptureRequest<'_>,
        ) -> Result<CaptureUsage, CaptureError>,
    ) -> Result<SessionPartitionCapture<'a, T>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.prepare_partition_work(
            transport,
            self.plan.clone(),
            PartitionCaptureKey::Observation(index),
            index,
            Producers::Routed(producers),
            limits,
            |receipt, rank, index| {
                let projection = receipt.producer(rank).expect("retained producer");
                let slice = super::super::routed::global_fragment_slice(projection, index)?;
                let request = PartitionRoutedUnitCaptureRequest {
                    geometry: geometry(receipt)?,
                    source_tokens: projection.global_shape()[0],
                    ownership: receipt
                        .routed_producer(rank)
                        .expect("retained sparse ownership"),
                    slice: &slice,
                };
                request.validate()?;
                Ok(PartitionCaptureNativeEstimate {
                    capture: estimate(&request)?,
                    generated_creation_bytes: 0,
                })
            },
        )
    }

    /// Pins one actual invocation's native row extent, dtype, unit map and
    /// exchange topology before its first chunk. An idle EP owner supplies zero
    /// rows explicitly; absence of a callback never establishes an idle owner.
    /// The enclosing invocation driver must complete its ordinary source work
    /// and call `finish_partition_routed_capture` only after provider success.
    pub fn begin_partition_routed_capture<T: PartitionCaptureTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        rows: u64,
        dtype: TensorDtype,
        units: &eredu_core::component::ComponentCoordinateMap,
        origins: Option<eredu_core::capture::RoutedUnitOrigins<'_>>,
    ) -> Result<(), CaptureError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_work(work)?;
        if !matches!(work.routed, Some(RoutedInvocation::Pending)) {
            return Err(invalid("sparse invocation requires unused live authority"));
        }
        work.routed = Some(RoutedInvocation::Failed);
        if work.source_accepted == Some(false) {
            return Err(invalid("partition dependencies have not completed"));
        }
        let receipt = work.exchange.receipt_plan();
        if let Some(owned) = receipt.routed_producer(work.rank) {
            validate_invocation(receipt, owned, rows, units, origins)?;
        }
        if !matches!(
            dtype,
            TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32 | TensorDtype::F64
        ) {
            return Err(invalid("unsupported sparse invocation dtype"));
        }
        for fragment in work.fragments.iter_mut().flatten() {
            fragment.record.source_dtype = Some(dtype.clone());
        }
        work.source_dtype = Some(dtype);
        work.routed = Some(RoutedInvocation::Active { rows, end: 0 });
        Ok(())
    }

    /// Accumulates a real provider chunk under its prepaid fragment allowance.
    /// Every accepted chunk must advance the exact native invocation extent.
    pub fn observe_partition_routed_units<B: CaptureBackend, T: PartitionCaptureTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        backend: &mut B,
        source: &PartitionRoutedUnitCaptureSource<'_, B::Tensor>,
    ) -> Result<(), CaptureExecutionError<B::Error>>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_work(work)?;
        let Some(RoutedInvocation::Active { rows, end }) = work.routed else {
            return Err(invalid("sparse chunk has no active invocation").into());
        };
        work.routed = Some(RoutedInvocation::Failed);
        let receipt = work.exchange.receipt_plan();
        let Some(owned) = receipt.routed_producer(work.rank) else {
            work.routed = Some(RoutedInvocation::Active { rows, end });
            return Ok(());
        };
        validate_invocation(
            receipt,
            owned,
            rows,
            source.unit_coordinates,
            source.origins,
        )?;
        let bank = geometry(receipt)?;
        let width = if owned.source_peer.is_some() {
            1
        } else {
            bank.routes_per_token
        };
        let groups = backend
            .shape(source.source.source_groups)
            .map_err(CaptureExecutionError::Backend)?;
        let coefficients = backend
            .shape(source.source.coefficients)
            .map_err(CaptureExecutionError::Backend)?;
        let next=eredu_core::capture::PartitionRoutedUnitCaptureLayout::advance_native_rows(
            rows,end,source.source.token_offset,coefficients.first().copied().unwrap_or(0))
            .map_err(|cause|match cause {eredu_core::capture::RoutedUnitValidationError::Overflow=>CaptureError::Overflow,
                _=>invalid("sparse chunk changed its invocation geometry or dtype")})?;
        if backend.source_dtype(source.source.values) != work.source_dtype
            || groups != [rows, width]
            || coefficients.len() != 2
            || coefficients[1] != width
        {
            return Err(invalid("sparse chunk changed its invocation geometry or dtype").into());
        }
        let projection = receipt.producer(work.rank).expect("retained producer");
        for (index, fragment) in work
            .fragments
            .as_mut()
            .expect("prepaid sparse fragments")
            .iter_mut()
            .enumerate()
        {
            let slice = super::super::routed::global_fragment_slice(projection, index)?;
            let request = PartitionRoutedUnitCaptureRequest {
                geometry: bank,
                source_tokens: projection.global_shape()[0],
                ownership: owned,
                slice: &slice,
            };
            let mut chunk = backend
                .capture_partition_routed_units(source, &request)
                .ok_or_else(|| {
                    CaptureError::Unsupported("partition routed collector unavailable".into())
                })?
                .map_err(CaptureExecutionError::Backend)?;
            if chunk.source_token_ranges != [[end, next]] {
                return Err(invalid("collector changed actual native chunk coverage").into());
            }
            chunk.validate_rows(&slice)?;
            if chunk.geometry != bank
                || chunk.rows.iter().any(|row| {
                    row.source_peer != owned.source_peer
                        || usize::try_from(row.expert)
                            .ok()
                            .and_then(|expert| owned.coordinates.experts().global_to_local(expert))
                            .is_none()
                        || (owned.source_peer.is_none() && (row.token < end || row.token >= next))
                })
            {
                return Err(invalid("collector changed routed ownership").into());
            }
            let Some(CapturePayload::RoutedUnits(payload)) = &mut fragment.record.payload else {
                return Err(invalid("sparse accumulator changed payload kind").into());
            };
            if add(payload.rows.len() as u64, chunk.rows.len() as u64)?
                > mul(slice.shape[0], slice.shape[1])?
            {
                return Err(invalid("sparse accumulator exceeds selected route bound").into());
            }
            payload.rows.append(&mut chunk.rows);
            payload
                .source_token_ranges
                .append(&mut chunk.source_token_ranges);
        }
        work.routed = Some(RoutedInvocation::Active { rows, end: next });
        Ok(())
    }

    /// Certifies exact chunk completion after the actual provider returned.
    /// Failure poisons this authority. Success still needs the invocation hook
    /// vote, ordinary all-rank delivery, and the enclosing transaction's commit.
    pub fn finish_partition_routed_capture<T: PartitionCaptureTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        success: bool,
    ) -> Result<(), CaptureError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.validate_partition_work(work)?;
        let previous = work.routed.replace(RoutedInvocation::Failed);
        let Some(RoutedInvocation::Active { rows, end }) = previous else {
            return Err(invalid(
                "sparse invocation has not started or has already finished",
            ));
        };
        if !success
            || (work
                .fragments
                .as_ref()
                .is_some_and(|fragments| !fragments.is_empty())
                && end != rows)
        {
            return Err(invalid("sparse invocation failed or omitted native chunks"));
        }
        let receipt = work.exchange.receipt_plan();
        for (index, fragment) in work.fragments.iter_mut().flatten().enumerate() {
            let Some(CapturePayload::RoutedUnits(payload)) = &mut fragment.record.payload else {
                return Err(invalid("sparse accumulator changed payload kind"));
            };
            super::super::routed::validate_payload(
                receipt,
                work.rank,
                receipt.producer(work.rank).expect("producer"),
                index,
                payload,
            )?;
            fragment.record.outcome = CaptureOutcome::Captured;
            let mut sink = CountingWriter {
                written: 0,
                limit: fragment.record.charged.encoded_bytes,
            };
            serde_json::to_writer(&mut sink, &fragment.record)
                .map_err(|_| invalid("sparse collector underestimated encoded size"))?;
        }
        work.routed = Some(RoutedInvocation::Complete);
        work.observed = true;
        Ok(())
    }
}

fn validate_invocation(
    receipt: &PartitionCaptureReceiptPlan,
    owned: &RoutedUnitCaptureOwnership,
    rows: u64,
    units: &eredu_core::component::ComponentCoordinateMap,
    origins: Option<eredu_core::capture::RoutedUnitOrigins<'_>>,
) -> Result<(), CaptureError> {
    eredu_core::capture::PartitionRoutedUnitCaptureLayout {
        geometry: geometry(receipt)?, source_tokens: receipt.global_shape[0], ownership: owned,
    }.validate_invocation(rows, units, origins)
}
