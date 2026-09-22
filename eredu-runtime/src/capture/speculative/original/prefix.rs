//! Closed logical prefix; native and physical ownership remain separate.
use super::*;

/// Already spent metadata and aggregate quota for one exact original invocation.
///
/// The source owner creates this only after committing the same request lineage.
/// Cold inspection and the funded collector resume its step without reserving the
/// prefix twice. It conveys no native permission, host funding, or completion.
#[derive(Clone, Debug)]
pub struct OriginalSpeculativeCapturePrefix {
    source: eredu_core::SharedStorageIdentity,
    active: Active,
    step: CaptureUsage,
    total: CaptureUsage,
}
impl OriginalSpeculativeCapturePrefix {
    pub(super) fn from_ledger(
        source: &SharedCapturePlan,
        active: Active,
        ledger: &CaptureLedger,
    ) -> Self {
        Self {
            source: source.storage_identity().clone(),
            active,
            step: ledger.step(),
            total: ledger.total(),
        }
    }
    pub(in crate::capture) fn validate_invocation(
        &self,
        invocation: OriginalSpeculativeCaptureInvocation<'_>,
    ) -> Result<(), CaptureProtocolError> {
        if self.source != *invocation.source().plan().storage_identity()
            || self.active != invocation.active
            || invocation.lineage().is_none()
        {
            return Err(CaptureProtocolError::Invocation);
        }
        Ok(())
    }
    /// Reconstruct the same logical step after the caller authenticates this
    /// source against its exact request lineage. Current totals must match; a
    /// consumed or stale prefix cannot reset cumulative spending. This changes
    /// only the returned diagnostic/collector ledger, not the source lineage.
    pub fn resume(
        &self,
        source: &SharedCapturePlan,
        current: CaptureUsage,
    ) -> Result<CaptureLedger, CaptureProtocolError> {
        if self.source != *source.storage_identity() || current != self.total {
            return Err(CaptureProtocolError::Invocation);
        }
        let base = CaptureUsage {
            captures: self
                .total
                .captures
                .checked_sub(self.step.captures)
                .ok_or(CaptureProtocolError::Geometry)?,
            retained_bytes: self
                .total
                .retained_bytes
                .checked_sub(self.step.retained_bytes)
                .ok_or(CaptureProtocolError::Geometry)?,
            host_bytes: self
                .total
                .host_bytes
                .checked_sub(self.step.host_bytes)
                .ok_or(CaptureProtocolError::Geometry)?,
            encoded_bytes: self
                .total
                .encoded_bytes
                .checked_sub(self.step.encoded_bytes)
                .ok_or(CaptureProtocolError::Geometry)?,
        };
        let mut ledger = CaptureLedger::with_inherited_usage(source.admission(), base)
            .map_err(|_| CaptureProtocolError::Geometry)?;
        ledger
            .reserve_quota(self.step)
            .map_err(|_| CaptureProtocolError::Geometry)?;
        Ok(ledger)
    }
    /// Exact fixed reconstruction/copy controls; no payload or native bytes.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<CaptureLedger>(),
            size_of::<[CaptureUsage; 3]>(),
            size_of::<CaptureProtocolError>(),
            size_of::<Result<CaptureLedger, CaptureProtocolError>>(),
            size_of::<(&Self, &SharedCapturePlan, CaptureUsage)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

impl OriginalSpeculativeCapture {
    pub(super) fn reserve_prefix(
        &self,
        active: Active,
        ledger: &mut CaptureLedger,
    ) -> Result<(), super::super::reductions::construction::ConstructionError> {
        let policy = CaptureObservationStep::with_invocation(
            self.source.plan().admission(),
            phase(active.phase),
            active.origin.prediction as u64,
            Some(CaptureInvocationShape {
                batch: 1,
                sequence: active.sequence,
                context: None,
            }),
        )
        .map_err(|_| {
            super::super::reductions::construction::ConstructionError::Invalid(
                "aggregate prefix invocation differs",
            )
        })?;
        ledger.begin_step();
        policy.reserve_invocation_metadata(ledger)?;
        policy.reserve_metadata(ledger)?;
        if let Some(edits) = &self.interventions {
            edits.source.reserve_capture_metadata(ledger)?;
        }
        crate::capture::policy::reserve_required(
            ledger,
            super::super::envelope_usage(self.identity.len(), active.span.is_some())?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> SharedCapturePlan {
        SharedCapturePlan::new(
            CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections: Vec::new(),
                limits: CaptureLimits {
                    per_step: CaptureUsage {
                        captures: u64::MAX,
                        retained_bytes: u64::MAX,
                        host_bytes: 1000,
                        encoded_bytes: u64::MAX,
                    },
                    cumulative: CaptureUsage {
                        captures: u64::MAX,
                        retained_bytes: u64::MAX,
                        host_bytes: 2000,
                        encoded_bytes: u64::MAX,
                    },
                    on_limit: CaptureLimitPolicy::Fail,
                },
            }
            .admit_invocations(
                &eredu_core::ObservationCatalog {
                    schema_version: 1,
                    points: Vec::new(),
                    completeness: eredu_core::DescriptionCompleteness::Complete,
                },
                &eredu_core::ObservationSupportReport {
                    schema_version: 1,
                    capture: Default::default(),
                    points: Vec::new(),
                },
                &CaptureCapabilities {
                    transformations: Vec::new(),
                    max_histogram_bins: 0,
                    conditions: Vec::new(),
                },
                CaptureInvocationBounds {
                    batch: 1,
                    max_sequence: 3,
                    max_context: None,
                    max_predictions: 2,
                },
            )
            .unwrap(),
        )
    }
    #[test]
    fn exact_prefix_preserves_step_total_and_rejects_stale_or_equal_foreign_source() {
        let source = source();
        let active = Active {
            invocation: 7,
            origin: SpeculativeActivationOrigin {
                request: SpeculativeRequestId::new(93),
                committed_tokens: 0,
                prediction: 0,
                prefix_digest: [0; 32],
                optimistic: false,
            },
            phase: SpeculativeActivationPhase::TargetPrefill,
            sequence: 3,
            span: None,
        };
        let mut logical = CaptureLedger::with_inherited_usage(
            source.admission(),
            CaptureUsage {
                host_bytes: 100,
                ..Default::default()
            },
        )
        .unwrap();
        logical
            .reserve_quota(CaptureUsage {
                host_bytes: 500,
                ..Default::default()
            })
            .unwrap();
        let prefix = OriginalSpeculativeCapturePrefix::from_ledger(&source, active, &logical);
        let mut resumed = prefix.resume(&source, logical.total()).unwrap();
        assert_eq!(resumed.step(), logical.step());
        assert_eq!(resumed.total(), logical.total());
        assert!(matches!(
            resumed.reserve(CaptureUsage {
                host_bytes: 501,
                ..Default::default()
            }),
            Err(CaptureError::Limit {
                budget: CaptureBudget::Host,
                cumulative: false
            })
        ));
        assert_eq!(resumed.total(), logical.total());
        resumed
            .reserve_quota(CaptureUsage {
                host_bytes: 7,
                ..Default::default()
            })
            .unwrap();
        assert!(prefix.resume(&source, resumed.total()).is_err());
        let foreign = SharedCapturePlan::new(source.admission().clone());
        assert!(prefix.resume(&foreign, logical.total()).is_err());
    }
}
