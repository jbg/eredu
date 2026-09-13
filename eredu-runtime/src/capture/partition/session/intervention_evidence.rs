//! Per-operation evidence using the ordinary bounded partition capture engine.
use super::*;
use crate::intervention::{evidence_selections, PartitionActivationLayout};
use eredu_core::{
    DescriptionCompleteness, ObservationCatalog, ObservationSupport, ObservationSupportReport,
};

impl CaptureSession {
    pub(super) fn prepare_intervention_evidence<'a, T, L>(
        &mut self,
        transport: &'a T,
        layout: &L,
        operation: usize,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<Vec<SessionPartitionCapture<'a, T>>, PartitionCaptureExchangeError>
    where
        T: PartitionCaptureTransport,
        L: PartitionCaptureLayout + PartitionActivationLayout,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let run = self
            .interventions
            .as_ref()
            .ok_or_else(|| invalid("no intervention admission"))?;
        let point = &run.plan.points()[operation];
        let operation_plan = &run.plan.plan().operations[operation];
        if operation_plan.evidence == eredu_core::intervention::InterventionEvidence::None {
            return Ok(Vec::new());
        }
        // Price the temporary selections/catalogs, admitted copies and canonical
        // serialization workspace before constructing any evidence DTO. These
        // contain no activation payload. The multiplier bounds escaping and the
        // simultaneously live before/after admissions; spare allocator capacity
        // is outside this logical storage contract.
        let heap = crate::execution_control::storage::heap_bytes;
        let point_bytes = heap(point).ok_or(CaptureError::Overflow)?;
        let slices = crate::execution_control::storage::heap_bytes(&operation_plan.slices)
            .ok_or(CaptureError::Overflow)?;
        let preparation = CaptureUsage {
            host_bytes: mul(
                mul(
                    add(
                        8192,
                        add(point_bytes, add(slices, operation_plan.id.len() as u64)?)?,
                    )?,
                    16,
                )?,
                transport.participant_count() as u64,
            )?,
            ..Default::default()
        };
        self.ledger.reserve_quota(preparation)?;
        let entries = evidence_selections(&run.plan.plan().operations[operation], point);
        let estimator = Arc::clone(&run.estimator);
        let mut plans = Vec::with_capacity(entries.len());
        for (selection, geometry) in entries {
            // This private derived admission retains the exact admitted global
            // operation geometry. It never replaces the parent's capture plan.
            // The live operation key and shared intent bind it to that authority.
            let catalog = ObservationCatalog {
                schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
                completeness: DescriptionCompleteness::Complete,
                points: vec![geometry],
            };
            let support = ObservationSupportReport {
                schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
                points: vec![ObservationSupport {
                    path: selection.path.clone(),
                    prefill: point.prefill.clone(),
                    decode: point.decode.clone(),
                    floating_to_f32: true,
                }],
                capture: CaptureCapabilities {
                    transformations: vec![selection.transform.kind()],
                    ..Default::default()
                },
            };
            let evidence = CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections: vec![selection],
                limits: self.plan.plan().limits.clone(),
            };
            plans.push(Arc::new(match self.plan.invocation_bounds() {
                Some(bounds) => {
                    evidence.admit_invocations(&catalog, &support, &support.capture, bounds)?
                }
                None => {
                    evidence.admit(&catalog, &support, &support.capture, self.plan.request())?
                }
            }));
        }
        let mut work = Vec::with_capacity(plans.len());
        for (evidence, plan) in plans.into_iter().enumerate() {
            let placement = layout.capture_placement_at(
                &plan,
                0,
                self.phase,
                self.prediction,
                self.invocation,
                limits,
            )?;
            let selection = &plan.plan().selections[0];
            let point = &plan.points()[0];
            let combination = layout.capture_combination(&selection.path)?;
            let native_selection = super::super::sum::reserve_native_selection(
                &plan,
                0,
                combination,
                transport.participant_count(),
                &mut self.ledger,
            )?;
            let mut bound = 0;
            for producer in &placement.producers {
                let mut bytes = 16_384;
                for fragment in producer.projection.fragments() {
                    bytes = add(
                        bytes,
                        add(
                            fragment_metadata_usage(
                                selection,
                                point,
                                producer.projection.global_shape().len(),
                            )?
                            .encoded_bytes,
                            estimator
                                .capture_usage(
                                    producer.projection.local_shape(),
                                    &native_selection,
                                    fragment.local(),
                                )?
                                .encoded_bytes,
                        )?,
                    )?;
                }
                bound = bound.max(bytes);
            }
            if bound > limits.max_record_bytes {
                return Err(CaptureError::Limit {
                    budget: CaptureBudget::Encoded,
                    cumulative: false,
                }
                .into());
            }
            let limits = PartitionCaptureReceiptLimits {
                max_record_bytes: bound,
                ..limits
            };
            drop(native_selection);
            work.push(self.prepare_partition_selection(
                transport,
                plan,
                PartitionCaptureKey::InterventionEvidence {
                    operation,
                    evidence,
                },
                0,
                placement.producers,
                combination,
                limits,
                |shape, selection, slice| {
                    Ok(PartitionCaptureNativeEstimate {
                        capture: estimator.capture_usage(shape, selection, slice)?,
                        generated_creation_bytes: 0,
                    })
                },
            )?);
        }
        Ok(work)
    }
}
