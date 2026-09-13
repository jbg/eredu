//! Prepaid dependency execution on every rank of an observed invocation.
use super::*;
use eredu_core::{BoundedCompletionOutcome, BoundedCompletionWait};

/// Move-only authority for one source preflight and bounded preparation. Distinct
/// from the final hook vote: agreeing geometry does not certify completed work.
pub struct SessionPartitionSource<'a, T: PartitionCaptureHookTransport> {
    vote: SessionPartitionHook<'a, T>,
    shape: Option<Vec<u64>>,
    routed: Option<RoutedSourceGeometry>,
    wait: BoundedCompletionWait,
}

struct RoutedSourceGeometry {
    ownership: RoutedUnitCaptureOwnership,
    geometry: RoutedUnitGeometry,
    source_tokens: u64,
}

impl CaptureSession {
    /// Reserves ordinary source dependencies for every executing rank, including
    /// nonexporting replicas. Empty global selections need no source execution.
    /// All shapes and costs join the common digest before model work begins.
    pub fn prepare_partition_source<'a, B: CaptureBackend, T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'a, T>,
        backend: &B,
        members: Vec<usize>,
        shapes: Vec<Vec<u64>>,
    ) -> Result<Option<SessionPartitionSource<'a, T>>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        if shapes.len() != members.len() || shapes.iter().any(|shape| shape.len() > 32) {
            return Err(invalid("partition source shapes do not match hook membership").into());
        }
        for (rank, projection) in work.exchange.receipt_plan().producers() {
            if members
                .iter()
                .position(|member| *member == rank)
                .is_none_or(|index| shapes[index] != projection.local_shape())
            {
                return Err(invalid("partition producer and source geometry disagree").into());
            }
        }
        if work
            .exchange
            .receipt_plan()
            .producers()
            .all(|(_, projection)| projection.fragments().is_empty())
        {
            return Ok(None);
        }
        self.prepare_partition_source_work(work, backend, members, shapes, None)
    }

    /// Prepays the worst-case provider input on every sparse invocation member.
    /// Actual receive rows may be smaller, including zero, but input width and
    /// original peer/route topology are fixed before model work begins.
    pub fn prepare_partition_routed_source<
        'a,
        B: CaptureBackend,
        T: PartitionCaptureHookTransport,
    >(
        &mut self,
        work: &mut SessionPartitionCapture<'a, T>,
        backend: &B,
        sources: Vec<PartitionRoutedCaptureSource>,
    ) -> Result<Option<SessionPartitionSource<'a, T>>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let receipt = work.exchange.receipt_plan();
        let geometry = super::routed::geometry(receipt)?;
        let tokens = receipt.global_shape[0];
        let first = receipt
            .producers()
            .next()
            .and_then(|(rank, _)| receipt.routed_producer(rank))
            .ok_or_else(|| invalid("sparse source has no retained producer"))?;
        let mut shapes = Vec::with_capacity(sources.len());
        for source in &sources {
            source.ownership.validate(geometry)?;
            if source.input_width == 0
                || source.ownership.source_peer != first.source_peer
                || source.ownership.source_peers != first.source_peers
                || receipt
                    .routed_producer(source.rank)
                    .is_some_and(|owned| owned != &source.ownership)
            {
                return Err(
                    invalid("sparse source differs from retained producer geometry").into(),
                );
            }
            shapes.push(vec![
                source
                    .ownership
                    .maximum_source_rows(tokens, geometry.routes_per_token)?,
                source.input_width,
            ]);
        }
        let members = sources.iter().map(|source| source.rank).collect();
        let routed = sources
            .into_iter()
            .map(|source| RoutedSourceGeometry {
                ownership: source.ownership,
                geometry,
                source_tokens: tokens,
            })
            .collect();
        self.prepare_partition_source_work(work, backend, members, shapes, Some(routed))
    }

    fn prepare_partition_source_work<'a, B: CaptureBackend, T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'a, T>,
        backend: &B,
        members: Vec<usize>,
        shapes: Vec<Vec<u64>>,
        routed: Option<Vec<RoutedSourceGeometry>>,
    ) -> Result<Option<SessionPartitionSource<'a, T>>, PartitionCaptureExchangeError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let vote = self.prepare_partition_vote(work, members.clone(), true)?;
        let wait = work.exchange.transport().capture_wait()?;
        let world = work.exchange.transport().participant_count() as u64;
        let mut usage = CaptureUsage::default();
        let mut descriptor = Sha256::new();
        descriptor.update(b"eredu-partition-capture-source-v2\0");
        descriptor.update([u8::from(routed.is_some())]);
        descriptor.update(self.partition.as_ref().expect("validated binding").claimed[&work.key]);
        for (index, (rank, shape)) in members.iter().zip(&shapes).enumerate() {
            if let Some(routed) = &routed {
                usage = usage.checked_add(CaptureUsage {
                    host_bytes: mul(
                        super::super::routed::ownership_bytes(&routed[index].ownership)?,
                        world,
                    )?,
                    ..Default::default()
                })?;
                descriptor.update(
                    serde_json::to_vec(&routed[index].ownership)
                        .map_err(|_| invalid("sparse source ownership cannot be encoded"))?,
                );
            }
            let native = backend.estimate_partition_source(shape, wait)?;
            usage = usage.checked_add(native)?.checked_add(CaptureUsage {
                // Sparse inputs may retain leading batch dimensions even though
                // their cold bound is flattened. Cover both actual shape reads
                // (preflight and invocation begin), each bounded to 32 axes.
                host_bytes: mul(
                    add(
                        64,
                        mul(
                            if routed.is_some() {
                                64
                            } else {
                                shape.len() as u64
                            },
                            8,
                        )?,
                    )?,
                    world,
                )?,
                ..Default::default()
            })?;
            descriptor.update((*rank as u64).to_le_bytes());
            descriptor.update((shape.len() as u64).to_le_bytes());
            for dimension in shape {
                descriptor.update(dimension.to_le_bytes());
            }
            hash_usage(&mut descriptor, native);
        }
        // The consumed ticket authorizes fixed source work; fragment factories
        // cannot spend these credits, and failure/restore cannot refund them.
        self.ledger.reserve_quota(usage)?;
        self.partition
            .as_mut()
            .expect("validated binding")
            .claimed
            .insert(work.key, descriptor.finalize().into());
        let local = members.iter().position(|rank| *rank == work.rank);
        let shape = local.map(|index| shapes.into_iter().nth(index).expect("matching lengths"));
        let routed =
            local.and_then(|index| routed.and_then(|values| values.into_iter().nth(index)));
        Ok(Some(SessionPartitionSource {
            vote,
            shape,
            routed,
            wait,
        }))
    }

    /// Validates local geometry on all active members before any dependency
    /// graph is submitted, then completes each ordinary source without export.
    pub fn ready_partition_source<B: CaptureBackend, T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        source: SessionPartitionSource<'_, T>,
        backend: &mut B,
        tensor: &B::Tensor,
    ) -> Result<(), PartitionCaptureObserverError<B::Error>>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.ready_partition_source_work(work, source, backend, tensor, None, true)
    }

    /// Checks actual routed input/maps/origins on every member before any source
    /// is submitted. `local_success = false` propagates an earlier preparation
    /// failure through the prepaid vote without submitting native source work.
    pub fn ready_partition_routed_source<B: CaptureBackend, T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        source: SessionPartitionSource<'_, T>,
        backend: &mut B,
        invocation: &crate::RoutedUnitInvocation<'_, B::Tensor>,
        local_success: bool,
    ) -> Result<(), PartitionCaptureObserverError<B::Error>>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.ready_partition_source_work(
            work,
            source,
            backend,
            invocation.input,
            Some(invocation),
            local_success,
        )
    }

    fn ready_partition_source_work<B: CaptureBackend, T: PartitionCaptureHookTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
        source: SessionPartitionSource<'_, T>,
        backend: &mut B,
        tensor: &B::Tensor,
        invocation: Option<&crate::RoutedUnitInvocation<'_, B::Tensor>>,
        local_success: bool,
    ) -> Result<(), PartitionCaptureObserverError<B::Error>>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let local = (|| {
            self.validate_partition_work(work)?;
            if work.observed || work.source_accepted != Some(false) || !source.vote.participates() {
                return Err(invalid("partition source requires unused local authority").into());
            }
            let shape = backend
                .shape(tensor)
                .map_err(CaptureExecutionError::Backend)?;
            if !local_success {
                return Err(invalid("an earlier routed source preparation failed").into());
            }
            match (&source.routed, invocation) {
                (None, None) if source.shape.as_ref() == Some(&shape) => {}
                (Some(routed), Some(invocation)) => {
                    let bound = source.shape.as_ref().expect("participating source");
                    let rows = routed_input_rows(&shape)?;
                    if shape.last() != bound.last()
                        || rows > bound[0]
                        || (routed.ownership.coordinates.experts().local_count() == 0 && rows != 0)
                        || invocation
                            .unit_coordinates
                            .is_some_and(|units| units != routed.ownership.coordinates.units())
                        || !matches!(
                            backend.source_dtype(tensor),
                            Some(
                                TensorDtype::F16
                                    | TensorDtype::Bf16
                                    | TensorDtype::F32
                                    | TensorDtype::F64
                            )
                        )
                        || match invocation.origins {
                            Some(origins) => {
                                let origins = origins.capture_coordinates();
                                routed.ownership.source_peer.is_none()
                                    || origins.row_count() as u64 != rows
                                    || origins.peer_count() as u64 != routed.ownership.source_peers
                                    || origins.routes_per_token() as u64
                                        != routed.geometry.routes_per_token
                            }
                            None => {
                                routed.ownership.source_peer.is_some()
                                    || rows != routed.source_tokens
                            }
                        }
                    {
                        return Err(
                            invalid("actual routed input differs from its prepaid source").into(),
                        );
                    }
                }
                _ => {
                    return Err(invalid(
                        "partition source shape or kind differs from its live authority",
                    )
                    .into())
                }
            }
            Ok::<_, CaptureExecutionError<B::Error>>(())
        })();
        let agreed = self.agree_partition_vote(work, source.vote, local.is_ok());
        // Still execute and fence the vote when local validation fails, retaining
        // the original cause if transport also fails.
        local?;
        if !agreed? {
            return Err(PartitionCaptureObserverError::HookRejected);
        }
        let result = backend.prepare_partition_source(tensor, source.wait);
        match result {
            Ok(BoundedCompletionOutcome::Completed) => {
                work.source_accepted = Some(true);
                Ok(())
            }
            Ok(BoundedCompletionOutcome::DeadlineExceeded { cancellation }) => {
                let error = PartitionCaptureExchangeError::Deadline { cancellation };
                work.exchange.transport().fail_capture_exchange(&error);
                Err(error.into())
            }
            Err(error) => {
                // The native cause remains the returned error. Fence transport
                // because a failed dependency may leave a model collective live.
                work.exchange.transport().fail_capture_exchange(
                    &PartitionCaptureExchangeError::Protocol("capture source preparation failed"),
                );
                Err(CaptureExecutionError::Backend(error).into())
            }
        }
    }

    pub(in crate::capture::partition) fn failed_partition_source<T: PartitionCaptureTransport>(
        &mut self,
        work: &mut SessionPartitionCapture<'_, T>,
    ) {
        // Allows the prepaid final error vote after failed preflight/preparation;
        // neither source completion nor successful capture is certified here.
        work.observed = true;
    }
}

/// Row extent of a provider input; leading batch dimensions are flattened by the
/// existing expert operator, while the final dimension remains the read width.
pub(in crate::capture::partition) fn routed_input_rows(shape: &[u64]) -> Result<u64, CaptureError> {
    if !(2..=32).contains(&shape.len()) || shape.last() == Some(&0) {
        return Err(invalid("routed provider input has invalid dimensions"));
    }
    shape[..shape.len() - 1]
        .iter()
        .try_fold(1, |rows, width| mul(rows, *width))
}
