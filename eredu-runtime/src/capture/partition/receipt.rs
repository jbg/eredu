//! Expected-producer receipt admission and bounded host-record encoding/decoding.
use super::*;
use eredu_core::checkpoint::TensorDtype;
use std::collections::BTreeMap;
use std::sync::Arc;

/// One retained producer projection for an exact global selection and forward.
/// Architecture composition supplies invocation ownership and scalar coordinates.
#[derive(Debug, Clone)]
pub struct PartitionCaptureProducer {
    /// Expected world rank, independently checked against the transport sender.
    pub rank: usize,
    /// Original global selection projected onto this producer's actual tensor.
    pub projection: CaptureSlicePartition,
}

/// Mechanism bounds applied before receipt storage or decoding allocations.
#[derive(Debug, Clone, Copy)]
pub struct PartitionCaptureReceiptLimits {
    /// Maximum distinct producer count for this selection.
    pub max_producers: usize,
    /// Maximum total native fragments across those producers.
    pub max_fragments: usize,
    /// Maximum bytes in one encoded producer receipt.
    pub max_record_bytes: u64,
}

/// Immutable expected-producer authority for one selection in one forward.
/// This is derived geometry and receipt authority, not admission to native work.
/// Session identity, global work quotas and native completion must be established
/// by the enclosing distributed session before it lends these facts.
#[derive(Debug)]
pub struct PartitionCaptureReceiptPlan {
    pub(super) combination: PartitionCaptureCombination,
    pub(super) plan: Arc<AdmittedCapturePlan>,
    pub(super) context: PartitionCaptureContext,
    producers: BTreeMap<usize, CaptureSlicePartition>,
    pub(super) global_shape: Vec<u64>,
    pub(super) routed: BTreeMap<usize, RoutedUnitCaptureOwnership>,
    limits: PartitionCaptureReceiptLimits,
    fragments: usize,
    identity: String,
    world_size: usize,
}

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}

fn reserve(ledger: &mut dyn CaptureReservation, usage: CaptureUsage) -> Result<(), CaptureError> {
    ledger.reserve_quota(usage)?;
    Ok(())
}

impl PartitionCaptureReceiptPlan {
    /// Cold host storage bound before retaining these already bounded declarations.
    pub fn preparation_usage(
        producers: &[PartitionCaptureProducer],
    ) -> Result<CaptureUsage, CaptureError> {
        Self::projection_preparation_usage(producers.iter().map(|producer| &producer.projection))
    }

    /// Cold bound including retained sparse ownership and temporary projection
    /// validation. A live session can prepay this before calling `new_routed`.
    pub fn routed_preparation_usage(
        producers: &[PartitionRoutedCaptureProducer],
    ) -> Result<CaptureUsage, CaptureError> {
        Self::projection_preparation_usage(producers.iter().map(|producer| &producer.projection))?
            .checked_add(Self::routed_ownership_usage(producers)?)
    }

    fn routed_ownership_usage(
        producers: &[PartitionRoutedCaptureProducer],
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            host_bytes: producers.iter().try_fold(4096u64, |bytes, producer| {
                add(bytes, super::routed::ownership_bytes(&producer.ownership)?)
            })?,
            ..Default::default()
        })
    }

    fn projection_preparation_usage<'a>(
        mut projections: impl ExactSizeIterator<Item = &'a CaptureSlicePartition>,
    ) -> Result<CaptureUsage, CaptureError> {
        let producers = projections.len() as u64;
        let first = projections
            .next()
            .ok_or_else(|| invalid("partition receipt has no producers"))?;
        let rank = first.global_shape().len();
        let fragments = projections
            .try_fold(first.fragments().len() as u64, |sum, projection| {
                add(sum, projection.fragments().len() as u64)
            })?;
        Ok(CaptureUsage {
            host_bytes: add(
                4096,
                add(
                    mul(producers, 2048)?,
                    mul(fragments, add(1024, mul(rank as u64, 256)?)?)?,
                )?,
            )?,
            ..Default::default()
        })
    }
    /// Validates complete, disjoint ownership before any capture work. Producers
    /// with zero overlap are retained so their explicit receipt cannot be confused
    /// with a missing peer. Empty global selections still require a producer.
    /// `ledger` prices cold host admission/storage; native work uses its separately
    /// admitted producer quotas, not an untrusted receipt's charged field.
    pub fn new(
        plan: impl Into<Arc<AdmittedCapturePlan>>,
        context: PartitionCaptureContext,
        producers: Vec<PartitionCaptureProducer>,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureMergeError> {
        Self::new_with_ownership(
            plan.into(),
            context,
            producers,
            BTreeMap::new(),
            PartitionCaptureCombination::Disjoint,
            world_size,
            limits,
            ledger,
        )
    }

    /// Admits complete selected floating terms from every named producer. Unlike
    /// disjoint assembly, missing terms can never be filled by another producer's
    /// overlapping coverage. Raw selected values are priced before native work;
    /// nonlinear transforms follow bounded, compensated host summation.
    pub fn new_sum(
        plan: impl Into<Arc<AdmittedCapturePlan>>,
        context: PartitionCaptureContext,
        producers: Vec<PartitionCaptureProducer>,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureMergeError> {
        Self::new_with_ownership(
            plan.into(),
            context,
            producers,
            BTreeMap::new(),
            PartitionCaptureCombination::SumF64ToF32,
            world_size,
            limits,
            ledger,
        )
    }

    /// Admits dynamic routed rows against exact expert/unit ownership. This uses
    /// the ordinary receipt transport and completion protocol; no native work or
    /// publication is authorized by this host declaration alone.
    pub fn new_routed(
        plan: impl Into<Arc<AdmittedCapturePlan>>,
        context: PartitionCaptureContext,
        producers: Vec<PartitionRoutedCaptureProducer>,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureMergeError> {
        if producers.is_empty() || producers.len() > limits.max_producers {
            return Err(invalid("sparse producer count exceeds its bound").into());
        }
        // Price temporary maps/vectors and retained ownership before allocating.
        reserve(ledger, Self::routed_ownership_usage(&producers)?)?;
        let mut ownership = BTreeMap::new();
        let mut projections = Vec::with_capacity(producers.len());
        for producer in producers {
            if ownership
                .insert(producer.rank, producer.ownership)
                .is_some()
            {
                return Err(invalid("duplicate sparse producer").into());
            }
            projections.push(PartitionCaptureProducer {
                rank: producer.rank,
                projection: producer.projection,
            });
        }
        Self::new_with_ownership(
            plan.into(),
            context,
            projections,
            ownership,
            PartitionCaptureCombination::Disjoint,
            world_size,
            limits,
            ledger,
        )
    }

    fn new_with_ownership(
        plan: Arc<AdmittedCapturePlan>,
        context: PartitionCaptureContext,
        producers: Vec<PartitionCaptureProducer>,
        routed: BTreeMap<usize, RoutedUnitCaptureOwnership>,
        combination: PartitionCaptureCombination,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureMergeError> {
        context.validate()?;
        if context.capture_plan_identity != plan.identity() {
            return Err(invalid("partition receipt context changed the capture admission").into());
        }
        let selection = plan
            .plan()
            .selections
            .get(context.selection_index)
            .ok_or_else(|| invalid("unknown partition receipt selection"))?;
        if context.prediction >= plan.request().max_predictions
            || !selection
                .schedule
                .includes(context.phase, context.prediction)
        {
            return Err(invalid("partition receipt is outside its admitted schedule").into());
        }
        if world_size == 0
            || producers.is_empty()
            || producers.len() > limits.max_producers
            || limits.max_record_bytes == 0
        {
            return Err(invalid("partition receipt producer or record bound is invalid").into());
        }
        let first = &producers[0].projection;
        if super::vocabulary::is_vocabulary(&selection.transform) && producers.len() != 1 {
            return Err(CaptureError::Unsupported(
                "vocabulary reduction requires exactly one global producer".into(),
            )
            .into());
        }
        let point = &plan.points()[context.selection_index];
        // Sparse declaration validation reconstructs temporary projections. Price
        // their storage, retained maps and pending slots before that allocation.
        if !routed.is_empty() {
            reserve(ledger, Self::preparation_usage(&producers)?)?;
        }
        let routed_geometry = super::routed::validate_declarations(
            point,
            selection,
            &producers,
            &routed,
            limits.max_fragments,
        )?;
        plan.geometry_at(context.phase, context.prediction, context.invocation)?
            .validate_actual(point, first.global_shape())?;
        let slice = resolve_slice(point, selection, first.global_shape())?;
        let mut fragments = 0usize;
        let mut covered = 0u64;
        for (index, producer) in producers.iter().enumerate() {
            let projection = &producer.projection;
            if combination == PartitionCaptureCombination::SumF64ToF32 {
                super::sum::validate_projection(selection, point, projection)?;
            }
            super::vocabulary::validate_projection(&selection.transform, projection)?;
            if producer.rank >= world_size
                || producers[..index].iter().any(|p| p.rank == producer.rank)
                || projection.global_shape() != first.global_shape()
                || projection.axis() != first.axis()
                || projection.global_slice() != &slice
            {
                return Err(
                    invalid("partition receipt producer identity or geometry disagrees").into(),
                );
            }
            fragments = fragments
                .checked_add(projection.fragments().len())
                .ok_or(CaptureError::Overflow)?;
            if fragments > limits.max_fragments {
                return Err(invalid("partition receipt fragment count exceeds its bound").into());
            }
            for geometry in projection.fragments() {
                for previous in &producers[..index] {
                    for other in previous.projection.fragments() {
                        if combination == PartitionCaptureCombination::Disjoint
                            && geometry.overlaps_destination(other)?
                            && super::routed::owners_overlap(&routed, producer.rank, previous.rank)
                        {
                            return Err(PartitionCaptureMergeError::Overlap);
                        }
                    }
                }
                let experts = routed
                    .get(&producer.rank)
                    .map_or(1, |owned| owned.coordinates.experts().local_count() as u64);
                covered = add(
                    covered,
                    mul(elements(&geometry.destination().shape)?, experts)?,
                )?;
            }
        }
        let expected = mul(
            elements(&slice.shape)?,
            if combination == PartitionCaptureCombination::SumF64ToF32 {
                producers.len() as u64
            } else {
                routed_geometry.map_or(1, |geometry| geometry.experts)
            },
        )?;
        if covered != expected {
            return Err(PartitionCaptureMergeError::Incomplete {
                expected_elements: expected,
                received_elements: covered,
            });
        }
        if routed.is_empty() {
            reserve(ledger, Self::preparation_usage(&producers)?)?;
        }
        let global_shape = first.global_shape().to_vec();
        let producers = producers
            .into_iter()
            .map(|p| (p.rank, p.projection))
            .collect();
        let identity = receipt_identity(
            &context,
            &producers,
            &routed,
            combination,
            world_size,
            limits,
        )?;
        Ok(Self {
            combination,
            plan,
            context,
            producers,
            global_shape,
            routed,
            limits,
            fragments,
            identity,
            world_size,
        })
    }

    /// Stable digest of this exact producer geometry, context and delivery bound.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Retained assembly equation, also authenticated by the receipt identity.
    pub const fn combination(&self) -> PartitionCaptureCombination {
        self.combination
    }

    /// Exact participant topology used to admit the producer ranks.
    pub const fn world_size(&self) -> usize {
        self.world_size
    }

    /// Exact expected context for this one selection and submission.
    pub const fn context(&self) -> &PartitionCaptureContext {
        &self.context
    }
    /// Expected native geometry for an admitted producer.
    pub fn producer(&self, rank: usize) -> Option<&CaptureSlicePartition> {
        self.producers.get(&rank)
    }
    /// Exact sparse expert/source ownership, separate from the unit rectangle.
    pub fn routed_producer(&self, rank: usize) -> Option<&RoutedUnitCaptureOwnership> {
        self.routed.get(&rank)
    }
    /// Maximum encoded bytes reserved/accepted for one producer.
    pub const fn max_record_bytes(&self) -> u64 {
        self.limits.max_record_bytes
    }

    /// Expected producers, including empty selections, in canonical world order.
    pub fn producers(&self) -> impl Iterator<Item = (usize, &CaptureSlicePartition)> {
        self.producers
            .iter()
            .map(|(&rank, projection)| (rank, projection))
    }

    /// Prepaid producer encoding bound, without constructing or serializing a value.
    pub fn encoding_usage(&self, rank: usize) -> Result<CaptureUsage, CaptureError> {
        let projection = self
            .producer(rank)
            .ok_or_else(|| invalid("unexpected partition producer"))?;
        Ok(CaptureUsage {
            host_bytes: add(
                add(4096, mul(projection.fragments().len() as u64, 1024)?)?,
                self.limits.max_record_bytes,
            )?,
            encoded_bytes: self.limits.max_record_bytes,
            ..Default::default()
        })
    }

    /// Complete receiver decoding and assembly bound for the admitted byte caps.
    /// It is charged before native producer work by the enclosing live session.
    pub fn delivery_usage(&self) -> Result<CaptureUsage, CaptureError> {
        let selection = &self.plan.plan().selections[self.context.selection_index];
        let point = &self.plan.points()[self.context.selection_index];
        let slice = resolve_slice(point, selection, &self.global_shape)?;
        self.decoding_usage(self.limits.max_record_bytes)?
            .checked_mul(self.producers.len() as u64)?
            .checked_add(CaptureUsage {
                host_bytes: add(
                    mul(
                        self.fragments as u64,
                        std::mem::size_of::<CapturedPartitionFragment>() as u64,
                    )?,
                    mul(self.producers.len() as u64, 8)?,
                )?,
                ..Default::default()
            })?
            .checked_add(super::assembly::assembly_metadata_usage(
                selection,
                point,
                slice.shape.len(),
                self.fragments,
            )?)?
            .checked_add(
                if self.combination == PartitionCaptureCombination::SumF64ToF32 {
                    super::sum::payload_usage(&selection.transform, elements(&slice.shape)?)?
                } else if matches!(selection.transform, CaptureTransform::RoutedUnits) {
                    super::routed::payload_usage(self, &slice)?
                } else {
                    super::assembly::assembly_payload_usage(
                        &selection.transform,
                        elements(&slice.shape)?,
                        self.fragments == 0,
                    )?
                },
            )
    }

    /// Conservative parser/decoded-record storage for an exact received byte count.
    /// Distributed preflight can price this without parsing or native work.
    pub fn decoding_usage(&self, encoded_bytes: u64) -> Result<CaptureUsage, CaptureError> {
        if encoded_bytes > self.limits.max_record_bytes {
            return Err(invalid(
                "partition capture receipt exceeds its encoded bound",
            ));
        }
        Ok(CaptureUsage {
            host_bytes: add(8192, mul(encoded_bytes, 128)?)?,
            encoded_bytes,
            ..Default::default()
        })
    }

    /// Encodes actual local capture evidence after reserving host/encoded storage.
    /// No sender-provided geometry can replace the expected producer projection.
    pub fn encode_producer(
        &self,
        producer_rank: usize,
        source_dtype: Option<TensorDtype>,
        fragments: Vec<CapturedPartitionFragment>,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Vec<u8>, CaptureError> {
        self.validate_routed_precision(source_dtype.as_ref())?;
        let projection = self
            .producer(producer_rank)
            .ok_or_else(|| invalid("unexpected partition capture producer"))?;
        if fragments.len() != projection.fragments().len() {
            return Err(invalid(
                "partition producer did not supply every expected fragment",
            ));
        }
        for (index, fragment) in fragments.iter().enumerate() {
            if fragment.combination != self.combination
                || fragment.producer_rank != producer_rank
                || fragment.plan_identity != self.plan.identity()
                || fragment.selection_index != self.context.selection_index
                || fragment.phase != self.context.phase
                || fragment.prediction != self.context.prediction
                || fragment.invocation != self.context.invocation
                || fragment.axis != projection.axis()
                || fragment.global_shape != self.global_shape
                || &fragment.global_slice != projection.global_slice()
                || fragment.geometry != projection.fragments()[index]
            {
                return Err(invalid(
                    "partition producer fragment differs from retained receipt authority",
                ));
            }
            if fragment.record.source_dtype != source_dtype {
                return Err(invalid(
                    "partition producer source precision disagrees with its fragments",
                ));
            }
            self.validate_record(producer_rank, projection, index, &fragment.record)?;
        }
        reserve(
            ledger,
            CaptureUsage {
                host_bytes: add(4096, mul(fragments.len() as u64, 1024)?)?,
                ..Default::default()
            },
        )?;
        let record = PartitionCaptureProducerRecord {
            schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
            combination: self.combination,
            receipt_plan_identity: self.identity.clone(),
            context: self.context.clone(),
            producer_rank,
            source_dtype,
            fragments: fragments
                .into_iter()
                .enumerate()
                .map(
                    |(fragment_index, fragment)| PartitionCaptureFragmentRecord {
                        fragment_index,
                        record: fragment.record,
                    },
                )
                .collect(),
        };
        // Count before allocating a JSON buffer. The counting sink enforces the
        // same byte limit as the subsequent bounded writer.
        let mut count = CountingWriter {
            written: 0,
            limit: self.limits.max_record_bytes,
        };
        serde_json::to_writer(&mut count, &record)
            .map_err(|_| invalid("partition producer record exceeds its encoded bound"))?;
        reserve(
            ledger,
            CaptureUsage {
                host_bytes: count.written,
                encoded_bytes: count.written,
                ..Default::default()
            },
        )?;
        let capacity = usize::try_from(count.written).map_err(|_| CaptureError::Overflow)?;
        let mut bytes = Vec::with_capacity(capacity);
        serde_json::to_writer(&mut bytes, &record)
            .map_err(|_| invalid("partition producer record could not be encoded"))?;
        if bytes.len() != capacity {
            return Err(invalid(
                "partition producer record size changed while encoding",
            ));
        }
        Ok(bytes)
    }

    fn validate_routed_precision(&self, dtype: Option<&TensorDtype>) -> Result<(), CaptureError> {
        if self.combination == PartitionCaptureCombination::SumF64ToF32 {
            super::sum::validate_precision(dtype)?;
        }
        if !self.routed.is_empty()
            && !matches!(
                dtype,
                None | Some(
                    TensorDtype::F16
                        | TensorDtype::Bf16
                        | TensorDtype::F32
                        | TensorDtype::F64
                        | TensorDtype::Encoded(_)
                )
            )
        {
            return Err(invalid("sparse receipt has non-floating source precision"));
        }
        Ok(())
    }

    fn validate_record(
        &self,
        producer_rank: usize,
        projection: &CaptureSlicePartition,
        index: usize,
        record: &CaptureRecord,
    ) -> Result<(), CaptureError> {
        let geometry = projection
            .fragments()
            .get(index)
            .ok_or_else(|| invalid("unexpected partition fragment ordinal"))?;
        let selection = &self.plan.plan().selections[self.context.selection_index];
        let point = &self.plan.points()[self.context.selection_index];
        if record.schema_version != CAPTURE_SCHEMA_VERSION
            || record.selection_id != selection.id
            || record.path != selection.path
            || record.node_id != point.node_id
            || record.position != point.position
            || record.source_shape.as_deref() != Some(projection.local_shape())
            || record.selected_shape.as_ref() != Some(&geometry.local().shape)
        {
            return Err(invalid(
                "partition fragment record differs from its admitted selection or geometry",
            ));
        }
        if record
            .charged
            .exceeded(self.plan.plan().limits.per_step)
            .is_some()
            || record
                .charged
                .exceeded(self.plan.plan().limits.cumulative)
                .is_some()
        {
            return Err(invalid(
                "partition fragment reports work beyond its global admission",
            ));
        }
        let native_transform;
        let transform = if self.combination == PartitionCaptureCombination::SumF64ToF32 {
            native_transform = super::sum::native_transform(&selection.transform);
            &native_transform
        } else {
            &selection.transform
        };
        match &record.outcome {
            outcome
                if *outcome
                    == completed_capture_outcome(transform, elements(&geometry.local().shape)?) =>
            {
                match (transform, &record.payload) {
                    (CaptureTransform::RoutedUnits, Some(CapturePayload::RoutedUnits(payload))) => {
                        super::routed::validate_payload(
                            self,
                            producer_rank,
                            projection,
                            index,
                            payload,
                        )?;
                    }
                    (
                        CaptureTransform::FullTensor | CaptureTransform::Slice,
                        Some(CapturePayload::Tensor(value)),
                    ) => {
                        if !source_representation_matches(
                            record.source_dtype.as_ref(),
                            value.data(),
                        ) {
                            return Err(invalid(
                            "partition raw payload representation disagrees with source precision",
                        ));
                        }
                        if value.shape().len() != geometry.local().shape.len()
                            || value
                                .shape()
                                .iter()
                                .zip(&geometry.local().shape)
                                .any(|(a, b)| *a as u64 != *b)
                        {
                            return Err(invalid(
                                "partition tensor payload differs from its selected geometry",
                            ));
                        }
                    }
                    (
                        CaptureTransform::Preview { max_elements },
                        Some(CapturePayload::Tensor(value)),
                    ) => {
                        if !source_representation_matches(
                            record.source_dtype.as_ref(),
                            value.data(),
                        ) || value.shape().len() != 1
                            || value.shape()[0] as u64
                                != (*max_elements).min(elements(&geometry.local().shape)?)
                        {
                            return Err(invalid(
                                "partition preview payload differs from its bounded source prefix",
                            ));
                        }
                    }
                    (CaptureTransform::Summary, Some(CapturePayload::Summary(_)))
                    | (CaptureTransform::Histogram { .. }, Some(CapturePayload::Histogram(_))) => {}
                    (
                        CaptureTransform::TokenScores { .. }
                        | CaptureTransform::TopCandidates { .. },
                        Some(payload),
                    ) => {
                        super::vocabulary::validate_payload(
                            selection,
                            point.position,
                            projection.global_shape(),
                            payload,
                        )?;
                    }
                    _ => {
                        return Err(invalid(
                            "partition payload does not match its admitted transform",
                        ))
                    }
                }
            }
            CaptureOutcome::Skipped { .. }
            | CaptureOutcome::Failed { .. }
            | CaptureOutcome::Missing
                if record.payload.is_none() => {}
            _ => return Err(invalid("partition fragment outcome and payload disagree")),
        }
        Ok(())
    }

    /// Starts one move-only collection. Consumption does not authorize a second
    /// delivery for the same forward; the enclosing run owns epoch uniqueness.
    pub fn into_delivery(self) -> PartitionCaptureDelivery {
        PartitionCaptureDelivery {
            plan: self,
            receipts: BTreeMap::new(),
            precision: None,
        }
    }
}

/// Bounded receipt collection for exactly one retained forward selection.
/// A failed receipt does not refund decoding or previously completed capture work.
#[derive(Debug)]
pub struct PartitionCaptureDelivery {
    plan: PartitionCaptureReceiptPlan,
    receipts: BTreeMap<usize, Vec<CapturedPartitionFragment>>,
    precision: Option<Option<TensorDtype>>,
}

impl PartitionCaptureDelivery {
    /// Validates the independently known sender before allocating decoded data.
    /// The encoded-byte bound and conservative parser storage are reserved first.
    /// This receives only completed transport bytes; native completion ownership
    /// remains with the enclosing distributed forward/transport driver.
    pub fn receive(
        &mut self,
        sender_rank: usize,
        bytes: &[u8],
        ledger: &mut dyn CaptureReservation,
    ) -> Result<(), CaptureError> {
        let projection = self
            .plan
            .producer(sender_rank)
            .ok_or_else(|| invalid("receipt came from an unexpected capture producer"))?;
        if self.receipts.contains_key(&sender_rank) {
            return Err(invalid("capture producer receipt was already delivered"));
        }
        if bytes.len() as u64 > self.plan.limits.max_record_bytes {
            return Err(invalid(
                "partition capture receipt exceeds its encoded bound",
            ));
        }
        // serde's tagged enums and nonfinite-value representation can temporarily
        // retain decoded content as well as final values. Price both buffers,
        // allocation slack, and fixed record/map metadata before parsing.
        reserve(ledger, self.plan.decoding_usage(bytes.len() as u64)?)?;
        let receipt: PartitionCaptureProducerRecord = serde_json::from_slice(bytes)
            .map_err(|_| invalid("malformed partition capture receipt"))?;
        self.plan
            .validate_routed_precision(receipt.source_dtype.as_ref())?;
        if receipt.schema_version != PARTITION_CAPTURE_SCHEMA_VERSION
            || receipt.combination != self.plan.combination
            || receipt.receipt_plan_identity != self.plan.identity
            || receipt.context != self.plan.context
            || receipt.producer_rank != sender_rank
            || receipt.fragments.len() != projection.fragments().len()
        {
            return Err(invalid(
                "partition capture receipt context, producer or cardinality differs",
            ));
        }
        if self.plan.fragments == 0
            && self
                .precision
                .as_ref()
                .is_some_and(|precision| precision != &receipt.source_dtype)
        {
            return Err(invalid(
                "partition producer receipts disagree on source precision",
            ));
        }
        let mut seen = vec![false; projection.fragments().len()];
        for fragment in &receipt.fragments {
            let Some(slot) = seen.get_mut(fragment.fragment_index) else {
                return Err(invalid("unexpected partition capture fragment ordinal"));
            };
            if std::mem::replace(slot, true) {
                return Err(invalid("partition capture fragment ordinal was repeated"));
            }
            if fragment.record.source_dtype != receipt.source_dtype {
                return Err(invalid(
                    "partition receipt source precision disagrees with its fragments",
                ));
            }
            self.plan.validate_record(
                sender_rank,
                projection,
                fragment.fragment_index,
                &fragment.record,
            )?;
        }
        self.precision = Some(receipt.source_dtype);
        let fragments = receipt
            .fragments
            .into_iter()
            .map(|fragment| CapturedPartitionFragment {
                invocation: self.plan.context.invocation,
                combination: self.plan.combination,
                plan_identity: self.plan.plan.identity().into(),
                selection_index: self.plan.context.selection_index,
                phase: self.plan.context.phase,
                prediction: self.plan.context.prediction,
                producer_rank: sender_rank,
                axis: projection.axis(),
                global_shape: projection.global_shape().to_vec(),
                global_slice: projection.global_slice().clone(),
                geometry: projection.fragments()[fragment.fragment_index].clone(),
                record: fragment.record,
            })
            .collect();
        self.receipts.insert(sender_rank, fragments);
        Ok(())
    }

    /// Expected producers whose explicit receipts have not arrived. A producer
    /// with no selected values remains missing until it acknowledges that fact.
    pub fn missing_producers(&self) -> impl Iterator<Item = usize> + '_ {
        self.plan
            .producers
            .keys()
            .copied()
            .filter(|rank| !self.receipts.contains_key(rank))
    }

    /// Consumes complete producer receipts and validates global payload coverage.
    /// Publication still requires the session's native completion/failure agreement.
    pub fn finish(
        self,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<ReceivedPartitionCapture, PartitionCaptureMergeError> {
        if let Some(producer_rank) = self.missing_producers().next() {
            return Err(PartitionCaptureMergeError::MissingProducer { producer_rank });
        }
        reserve(
            ledger,
            CaptureUsage {
                host_bytes: add(
                    mul(
                        self.plan.fragments as u64,
                        std::mem::size_of::<CapturedPartitionFragment>() as u64,
                    )?,
                    mul(self.receipts.len() as u64, 8)?,
                )?,
                ..Default::default()
            },
        )?;
        let producers = self.receipts.keys().copied().collect();
        let context = &self.plan.context;
        let selection = &self.plan.plan.plan().selections[context.selection_index];
        let capture = if self.plan.combination == PartitionCaptureCombination::SumF64ToF32
            && self.plan.fragments != 0
        {
            super::sum::assemble(
                &self.plan,
                self.receipts.into_values().flatten().collect(),
                ledger,
            )?
        } else if matches!(selection.transform, CaptureTransform::RoutedUnits) {
            let mut capture = super::routed::assemble(
                &self.plan,
                self.receipts.into_values().flatten().collect(),
                ledger,
            )?;
            if self.plan.fragments == 0 {
                capture.record.source_dtype = self.precision.flatten();
            }
            capture
        } else if self.plan.fragments == 0 {
            self.plan.empty_capture(self.precision.flatten(), ledger)?
        } else {
            let fragments = self.receipts.into_values().flatten().collect();
            match selection.transform {
                CaptureTransform::RoutedUnits => unreachable!("handled before dense assembly"),
                CaptureTransform::Slice
                | CaptureTransform::FullTensor
                | CaptureTransform::Preview { .. } => assemble_tensor_fragments(
                    &self.plan.plan,
                    context.selection_index,
                    context.phase,
                    context.prediction,
                    &self.plan.global_shape,
                    fragments,
                    ledger,
                )?,
                CaptureTransform::Summary | CaptureTransform::Histogram { .. } => {
                    assemble_reduced_fragments(
                        &self.plan.plan,
                        context.selection_index,
                        context.phase,
                        context.prediction,
                        &self.plan.global_shape,
                        fragments,
                        ledger,
                    )?
                }
                CaptureTransform::TokenScores { .. } | CaptureTransform::TopCandidates { .. } => {
                    super::assembly::assemble_vocabulary_fragment(
                        &self.plan.plan,
                        context.selection_index,
                        context.phase,
                        context.prediction,
                        &self.plan.global_shape,
                        fragments,
                        ledger,
                    )?
                }
            }
        };
        Ok(ReceivedPartitionCapture {
            receipt_plan_identity: self.plan.identity,
            context: self.plan.context,
            producers,
            capture,
        })
    }
}

/// Complete received selection with its execution/branch/overlay/epoch provenance
/// and every producer acknowledgment, including producers contributing no values.
/// This remains evidence pending the enclosing session's native completion agreement.
#[derive(Debug)]
pub struct ReceivedPartitionCapture {
    receipt_plan_identity: String,
    context: PartitionCaptureContext,
    producers: Vec<usize>,
    capture: AssembledPartitionCapture,
}
impl ReceivedPartitionCapture {
    /// Validated expected-producer geometry and delivery-bound identity.
    pub fn receipt_plan_identity(&self) -> &str {
        &self.receipt_plan_identity
    }
    /// Exact context preserved through assembly and public delivery composition.
    pub const fn context(&self) -> &PartitionCaptureContext {
        &self.context
    }
    /// Every expected producer whose receipt was accepted.
    pub fn producers(&self) -> &[usize] {
        &self.producers
    }
    /// Global value and its contributing native fragment evidence.
    pub const fn capture(&self) -> &AssembledPartitionCapture {
        &self.capture
    }
    /// Moves context, explicit acknowledgments and global capture evidence together.
    pub fn into_parts(
        self,
    ) -> (
        PartitionCaptureContext,
        Vec<usize>,
        AssembledPartitionCapture,
        String,
    ) {
        (
            self.context,
            self.producers,
            self.capture,
            self.receipt_plan_identity,
        )
    }
}

impl PartitionCaptureReceiptPlan {
    fn empty_capture(
        &self,
        precision: Option<TensorDtype>,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<AssembledPartitionCapture, PartitionCaptureMergeError> {
        let selection = &self.plan.plan().selections[self.context.selection_index];
        let point = &self.plan.points()[self.context.selection_index];
        let slice = resolve_slice(point, selection, &self.global_shape)?;
        if elements(&slice.shape)? != 0 {
            return Err(invalid("nonempty capture has no native fragments").into());
        }
        let charged = metadata_reservation(selection, point)?.checked_add(
            super::assembly::assembly_payload_usage(&selection.transform, 0, true)?,
        )?;
        reserve(ledger, charged)?;
        let payload =
            crate::capture::empty_payload(&selection.transform, &slice.shape, precision.as_ref())?;
        Ok(AssembledPartitionCapture {
            combination: self.combination,
            plan_identity: self.plan.identity().into(),
            phase: self.context.phase,
            prediction: self.context.prediction,
            record: CaptureRecord {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selection_id: selection.id.clone(),
                path: selection.path.clone(),
                node_id: point.node_id.clone(),
                position: point.position,
                source_shape: Some(self.global_shape.clone()),
                source_dtype: precision,
                selected_shape: Some(slice.shape),
                outcome: CaptureOutcome::Captured,
                payload: Some(payload),
                charged,
            },
            contributions: Vec::new(),
        })
    }
}

fn source_representation_matches(
    dtype: Option<&TensorDtype>,
    data: &eredu_core::TensorObservationData,
) -> bool {
    use eredu_core::TensorObservationData as Data;
    match dtype {
        None | Some(TensorDtype::Encoded(_)) => true,
        Some(TensorDtype::Bool) => matches!(data, Data::Bool(_)),
        Some(TensorDtype::I8 | TensorDtype::I16 | TensorDtype::I32 | TensorDtype::I64) => {
            matches!(data, Data::I64(_))
        }
        Some(TensorDtype::U8 | TensorDtype::U16 | TensorDtype::U32 | TensorDtype::U64) => {
            matches!(data, Data::U64(_))
        }
        Some(TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32 | TensorDtype::F64) => {
            matches!(data, Data::F32(_))
        }
        Some(TensorDtype::Complex64) => false,
    }
}

fn receipt_identity(
    context: &PartitionCaptureContext,
    producers: &BTreeMap<usize, CaptureSlicePartition>,
    routed: &BTreeMap<usize, RoutedUnitCaptureOwnership>,
    combination: PartitionCaptureCombination,
    world_size: usize,
    limits: PartitionCaptureReceiptLimits,
) -> Result<String, CaptureError> {
    use sha2::{Digest, Sha256};
    fn word(hash: &mut Sha256, value: u64) {
        hash.update(value.to_le_bytes());
    }
    fn text(hash: &mut Sha256, value: &str) {
        word(hash, value.len() as u64);
        hash.update(value.as_bytes());
    }
    fn numbers(hash: &mut Sha256, values: &[u64]) {
        word(hash, values.len() as u64);
        for value in values {
            word(hash, *value);
        }
    }
    fn slice(hash: &mut Sha256, value: &ResolvedCaptureSlice) {
        numbers(hash, &value.starts);
        numbers(hash, &value.ends);
        numbers(hash, &value.strides);
        numbers(hash, &value.shape);
    }
    let mut hash = Sha256::new();
    hash.update(b"eredu.partition.capture.receipt.v2");
    word(
        &mut hash,
        match combination {
            PartitionCaptureCombination::Disjoint => 0,
            PartitionCaptureCombination::SumF64ToF32 => 1,
        },
    );
    for value in [
        &context.artifact_identity,
        &context.execution_identity,
        &context.run_identity,
        &context.capture_plan_identity,
    ] {
        text(&mut hash, value);
    }
    word(&mut hash, u64::from(context.overlay_identity.is_some()));
    if let Some(value) = &context.overlay_identity {
        text(&mut hash, value);
    }
    for value in [
        context.selection_index,
        world_size,
        limits.max_producers,
        limits.max_fragments,
        producers.len(),
    ] {
        word(
            &mut hash,
            u64::try_from(value).map_err(|_| CaptureError::Overflow)?,
        );
    }
    word(&mut hash, limits.max_record_bytes);
    word(
        &mut hash,
        match context.phase {
            CapturePhase::Prefill => 0,
            CapturePhase::Decode => 1,
        },
    );
    word(&mut hash, context.prediction);
    word(&mut hash, context.forward_epoch);
    for (rank, projection) in producers {
        word(&mut hash, *rank as u64);
        word(&mut hash, projection.axis() as u64);
        numbers(&mut hash, projection.global_shape());
        numbers(&mut hash, projection.local_shape());
        slice(&mut hash, projection.global_slice());
        word(&mut hash, projection.fragments().len() as u64);
        for geometry in projection.fragments() {
            slice(&mut hash, geometry.local());
            slice(&mut hash, geometry.destination());
        }
    }
    if !routed.is_empty() {
        hash.update(b"routed.ownership.v1");
        for (rank, owned) in routed {
            word(&mut hash, *rank as u64);
            word(&mut hash, owned.source_peers);
            word(&mut hash, u64::from(owned.source_peer.is_some()));
            word(&mut hash, owned.source_peer.unwrap_or(0));
            for map in [owned.coordinates.experts(), owned.coordinates.units()] {
                word(&mut hash, map.global_count() as u64);
                if let Some(range) = map.contiguous_range() {
                    word(&mut hash, 0);
                    word(&mut hash, range.start as u64);
                    word(&mut hash, range.end as u64);
                } else {
                    word(&mut hash, 1);
                    word(&mut hash, map.local_count() as u64);
                    for index in 0..map.local_count() {
                        word(
                            &mut hash,
                            map.local_to_global(index).expect("checked map") as u64,
                        );
                    }
                }
            }
        }
    }
    use std::fmt::Write;
    let mut identity = String::with_capacity(64);
    for byte in hash.finalize() {
        write!(&mut identity, "{byte:02x}").expect("writing into String");
    }
    Ok(identity)
}
