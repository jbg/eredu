//! Eager numerical backend: gathers actual sorted rows after shared admission.
use super::*;

pub(super) struct Collector {
    pub(super) fault: Option<&'static str>,
    pub(super) calls: Arc<[AtomicUsize; 3]>,
}
impl CaptureBackend for Collector {
    type Tensor = NumericTensor;
    type Error = Error;
    fn shape(&self, value: &NumericTensor) -> Result<Vec<u64>, Error> {
        Ok(value.shape.iter().map(|n| *n as u64).collect())
    }
    fn source_dtype(&self, _: &NumericTensor) -> Option<eredu_core::checkpoint::TensorDtype> {
        Some(eredu_core::checkpoint::TensorDtype::F32)
    }
    fn estimate(
        &self,
        _: &NumericTensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        panic!("dense estimate for sparse units")
    }
    fn transform(
        &mut self,
        _: &NumericTensor,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Error> {
        panic!("dense transform for sparse units")
    }
    fn estimate_partition_source(
        &self,
        shape: &[u64],
        _: BoundedCompletionWait,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            retained_bytes: shape.iter().product::<u64>() * 4,
            ..Default::default()
        })
    }
    fn prepare_partition_source(
        &mut self,
        value: &NumericTensor,
        _: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Error> {
        self.calls[0].fetch_add(1, Ordering::SeqCst);
        if self.fault
            == Some(if value.data.is_empty() {
                "source-idle"
            } else {
                "source-active"
            })
        {
            self.calls[2].fetch_add(1, Ordering::SeqCst);
            return Err(Error::backend_retained_source(UnitSentinel));
        }
        Ok(BoundedCompletionOutcome::Completed)
    }
    fn estimate_partition_routed_units(
        &self,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        request.validate()?;
        let values = request.native_shape()?.iter().product::<u64>();
        let rows = request.source_tokens * request.geometry.routes_per_token;
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: values * 4,
            host_bytes: 4096 + rows * 1024 + values * 32,
            encoded_bytes: 4096 + rows * 1024 + values * 32,
        })
    }
    fn capture_partition_routed_units(
        &mut self,
        input: &PartitionRoutedUnitCaptureSource<'_, NumericTensor>,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Option<Result<RoutedUnitCapture, Error>> {
        self.calls[1].fetch_add(1, Ordering::SeqCst);
        if self.fault == Some("collector") {
            self.calls[2].fetch_add(1, Ordering::SeqCst);
            return Some(Err(Error::backend_retained_source(UnitSentinel)));
        }
        let source = &input.source;
        let width = input.unit_coordinates.local_count();
        let route_width = *source.coefficients.shape.last().unwrap() as usize;
        let group_width = *source.source_groups.shape.last().unwrap() as usize;
        let count = source.coefficients.data.len() / route_width;
        let mut rows = Vec::new();
        for row in 0..source.values.shape[0] as usize {
            let native = source.token_indices.data[row] as usize;
            let selection = source.selection_indices.data[row] as usize;
            let slot = selection % route_width;
            let source_token = source.token_offset as usize + native;
            let (peer, token, slot) = if let Some(origins) = input.origins {
                let origin = origins.resolve(source_token).unwrap();
                (origin.source_peer, origin.token, origin.slot)
            } else {
                (None, source_token, slot)
            };
            let slice = request.slice;
            if peer.map(|p| p as u64) != request.ownership.source_peer
                || [token as u64, slot as u64]
                    .iter()
                    .enumerate()
                    .any(|(axis, v)| {
                        *v < slice.starts[axis]
                            || *v >= slice.ends[axis]
                            || !(*v - slice.starts[axis]).is_multiple_of(slice.strides[axis])
                    })
            {
                continue;
            }
            let local_group = source.source_groups.data
                [source_token * group_width + selection % route_width]
                as usize;
            let expert = source
                .global_groups
                .map_or(local_group, |groups| groups[local_group]);
            assert!(request
                .ownership
                .coordinates
                .experts()
                .global_to_local(expert)
                .is_some());
            let data = (0..slice.shape[2])
                .map(|unit| {
                    let global = slice.starts[2] + unit * slice.strides[2];
                    let local = input
                        .unit_coordinates
                        .global_to_local(global as usize)
                        .unwrap();
                    source.values.data[row * width + local]
                })
                .collect::<Vec<_>>();
            rows.push(RoutedUnitCaptureRow {
                source_peer: peer.map(|p| p as u64),
                token: token as u64,
                slot: slot as u64,
                expert: expert as u64,
                coefficient: source.coefficients.data[selection],
                unit_start: slice.starts[2],
                unit_stride: slice.strides[2],
                values: TensorObservation::new(vec![data.len()], TensorObservationData::F32(data))
                    .unwrap(),
            });
        }
        Some(Ok(RoutedUnitCapture {
            geometry: request.geometry,
            source_token_ranges: vec![[source.token_offset, source.token_offset + count as u64]],
            rows,
        }))
    }
}
