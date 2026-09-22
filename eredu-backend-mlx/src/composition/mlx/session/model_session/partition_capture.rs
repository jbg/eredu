//! Native composition of retained architecture layouts with the shared collector.
use super::*;
use eredu_core::{ObservationSupportStatus, capture::*};
use eredu_runtime::capture::{CaptureSession, partition::*};
mod owner;
pub(super) use owner::Publication;

/// All aliases are closed: the final shared allocation retires before its
/// payload and the account that funded its original construction.
#[derive(Clone)]
pub(in crate::composition::mlx) struct LoadedPartitionCapture(Publication<PartitionCaptureData>);
pub(in crate::composition::mlx) struct PartitionCaptureData {
    pub(super) discovery: CaptureDiscovery,
    execution: String,
    setup: eredu_runtime::CommunicationSessionIdentity,
    layouts: eredu_architectures::prepared_sources::PreparedComponentPartitionSource,
}
impl std::ops::Deref for LoadedPartitionCapture {
    type Target = PartitionCaptureData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl LoadedPartitionCapture {
    pub(in crate::composition::mlx) fn layouts(
        &self,
    ) -> &eredu_architectures::component_partition::ComponentPartitionLayouts {
        self.layouts.layouts()
    }
    pub(in crate::composition::mlx) fn source_labels(
        &self,
    ) -> (&str, &str, eredu_runtime::CommunicationSessionIdentity) {
        (
            &self.discovery.artifact_identity,
            &self.execution,
            self.setup,
        )
    }
    pub(super) fn identity(
        &self,
        overlay: Option<&str>,
    ) -> Result<PartitionCaptureIdentity, CaptureError> {
        PartitionCaptureIdentity::for_session(
            self.discovery.artifact_identity.clone(),
            self.execution.clone(),
            self.setup,
            overlay.map(str::to_owned),
        )
    }
}
#[derive(Debug, thiserror::Error)]
enum ConstructionError {
    #[error("{0}")]
    Capture(#[from] CaptureError),
    #[error("{0}")]
    Component(#[from] eredu_architectures::prepared_sources::ComponentPartitionSourceError),
    #[error("{0}")]
    Discovery(#[from] eredu_architectures::prepared_sources::CaptureDiscoverySourceError),
}
impl ConstructionError {
    fn native(self, funding: Option<&eredu_core::HostMetadataFunding>) -> Error {
        let refusal = match &self {
            Self::Capture(CaptureError::AdmissionStorage(
                CaptureAdmissionStorageError::Funding(error),
            )) => Some(*error),
            Self::Component(error) => error.funding_error(),
            Self::Discovery(error) => error.funding_error(),
            _ => None,
        };
        if let Some(error) = refusal {
            return Error::WorkspacePlanning(error);
        }
        match funding {
            Some(funding) => {
                crate::composition::mlx::model::retain_planning_error(self, funding.clone())
            }
            None => Error::Other(Box::new(self)),
        }
    }
}
impl MlxModelSession {
    /// Ring fixtures exercise this exact source producer with their loaded pool
    /// and transport, then continue their ordinary native capture oracle.
    #[cfg(test)]
    pub(crate) fn verify_original_partition_capture_publication(&self, prompt_tokens: u64) {
        assert!(self.payload.distributed.is_some());
        self.loaded_partition_capture().unwrap().unwrap();
        let ordinary = self
            .partition_capture
            .borrow_mut()
            .take()
            .expect("fixture compiled its capture plan through ordinary discovery");
        assert!(!ordinary.is_funded());
        let usage = CaptureUsage {
            captures: 100_000,
            retained_bytes: 512 << 20,
            host_bytes: 4 << 30,
            encoded_bytes: 512 << 20,
        };
        let precompiled = CapturePlan {
            schema_version: 1,
            selections: vec![CaptureSelection {
                id: "cold-logits".into(),
                path: "model.logits".into(),
                schedule: CaptureSchedule::default(),
                slices: vec![],
                transform: CaptureTransform::FullTensor,
            }],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage.checked_mul(3).unwrap(),
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &ordinary.discovery.catalog,
            &ordinary.discovery.support,
            &ordinary.discovery.support.capture,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens,
                max_predictions: 3,
            },
        )
        .unwrap();
        precompiled.revalidate(&ordinary.discovery).unwrap();
        let pool = &self.payload.memory_ledger;
        assert_eq!(
            pool.unquoted_owner_count().unwrap(),
            0,
            "cold source fixture must run before ordinary parameter or state mutation"
        );
        let baseline = pool.fixture_host_charge().unwrap();
        let capacity = baseline.checked_add(256 << 20).unwrap();

        // The real pool refuses before constructing or publishing a cold source.
        assert!(matches!(
            self.prepare_original_partition_capture(crate::memory_fixture::resolved_limits(
                baseline
            )),
            Err(Error::WorkspacePlanning(_))
        ));
        assert!(self.partition_capture.borrow().is_none());
        assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
        self.prepare_original_partition_capture(crate::memory_fixture::resolved_limits(capacity))
            .unwrap();
        let cold = self.partition_capture_source().unwrap();
        assert!(cold.0.is_funded());
        precompiled.revalidate(&cold.discovery).unwrap();
        let retained = pool.fixture_host_charge().unwrap();
        assert!(retained > baseline);
        self.prepare_original_partition_capture(crate::memory_fixture::resolved_limits(capacity))
            .unwrap();
        assert_eq!(
            pool.fixture_host_charge().unwrap(),
            retained,
            "funded lookup creates no account"
        );
        let cached = self.partition_capture_source().unwrap();
        assert!(std::ptr::eq(&*cold, &*cached));
        drop(cached);
        self.partition_capture.borrow_mut().take();
        assert_eq!(
            pool.fixture_host_charge().unwrap(),
            retained,
            "escaping source retains its account"
        );
        drop(cold);
        assert_eq!(pool.fixture_host_charge().unwrap(), baseline);

        // Keep the actual old publication alive across refusal and replacement.
        *self.partition_capture.borrow_mut() = Some(ordinary.clone());
        assert!(matches!(
            self.prepare_original_partition_capture(crate::memory_fixture::resolved_limits(
                baseline
            )),
            Err(Error::WorkspacePlanning(_))
        ));
        let unchanged = self.partition_capture_source().unwrap();
        assert!(std::ptr::eq(&*ordinary, &*unchanged));
        drop(unchanged);
        self.prepare_original_partition_capture(crate::memory_fixture::resolved_limits(capacity))
            .unwrap();
        let original = self.partition_capture_source().unwrap();
        assert!(original.0.is_funded());
        assert!(!std::ptr::eq(&*ordinary, &*original));
        precompiled.revalidate(&original.discovery).unwrap();
        precompiled.revalidate(&ordinary.discovery).unwrap();
        let observed = self.loaded_partition_capture().unwrap().unwrap();
        assert!(std::ptr::eq(&*original, &*observed));
        drop((observed, original));
        // The following numerical oracle intentionally creates an ordinary
        // reference model. Finish this funded source's lifetime before that
        // unquoted load, preserving the original discovery for the oracle.
        let retired = self.partition_capture.borrow_mut().replace(ordinary);
        drop(retired);
        assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
    }
    pub(in crate::composition::mlx) fn partition_capture_source(
        &self,
    ) -> Option<LoadedPartitionCapture> {
        self.partition_capture
            .borrow()
            .as_ref()
            .map(|source| LoadedPartitionCapture(source.clone()))
    }
    /// All original capture inputs, including already compiled declarations,
    /// qualify the retained source before cold semantic revalidation. Raw
    /// declaration compilation may already have constructed that same source.
    pub(super) fn prepare_original_partition_capture(
        &self,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<(), Error> {
        if self.payload.distributed.is_none()
            || self
                .partition_capture
                .borrow()
                .as_ref()
                .is_some_and(Publication::is_funded)
        {
            return Ok(());
        }
        let funding = self
            .payload
            .memory_ledger
            .prepare_workspace_metadata(
                self.payload.model.erased().inference_execution_identity(),
                capacity,
            )
            .map_err(Error::WorkspacePlanning)?;
        funding
            .reserve_metadata(std::mem::size_of::<(
                &Self,
                u64,
                eredu_core::HostMetadataFunding,
                Result<Option<LoadedPartitionCapture>, Error>,
            )>())
            .map_err(Error::WorkspacePlanning)?;
        self.original_partition_capture(&funding)?;
        Ok(())
    }
    pub(super) fn loaded_partition_capture(
        &self,
    ) -> Result<Option<LoadedPartitionCapture>, CaptureError> {
        self.partition_capture_with_funding(None)
            .map_err(|error| match error {
                ConstructionError::Capture(cause) => cause,
                ConstructionError::Component(cause) => CaptureError::Unsupported(cause.to_string()),
                ConstructionError::Discovery(
                    eredu_architectures::prepared_sources::CaptureDiscoverySourceError::Capture(
                        cause,
                    ),
                ) => cause,
                ConstructionError::Discovery(cause) => CaptureError::Unsupported(cause.to_string()),
            })
    }
    pub(super) fn original_partition_capture(
        &self,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<Option<LoadedPartitionCapture>, Error> {
        self.partition_capture_with_funding(Some(funding))
            .map_err(|error| error.native(Some(funding)))
    }
    fn partition_capture_with_funding(
        &self,
        funding: Option<&eredu_core::HostMetadataFunding>,
    ) -> Result<Option<LoadedPartitionCapture>, ConstructionError> {
        let construction = CaptureSourceConstruction::new(funding);
        construction.controls(std::mem::size_of::<(
            &Self,
            Option<&eredu_core::HostMetadataFunding>,
            Option<LoadedPartitionCapture>,
            std::cell::Ref<'_, Option<LoadedPartitionCapture>>,
        )>())?;
        let Some(transport) = &self.payload.distributed else {
            return Ok(None);
        };
        Publication::cached(&self.partition_capture, construction, || {
            self.construct_partition_capture(transport, construction)
                .map(|source| source.0)
        })
        .map(|source| Some(LoadedPartitionCapture(source)))
    }
    fn construct_partition_capture(
        &self,
        transport: &MlxDistributedSession,
        construction: CaptureSourceConstruction<'_>,
    ) -> Result<LoadedPartitionCapture, ConstructionError> {
        use std::mem::size_of;
        construction.controls(size_of::<(
            &Self,
            &MlxDistributedSession,
            PartitionCaptureData,
            LoadedPartitionCapture,
            Result<LoadedPartitionCapture, ConstructionError>,
        )>())?;
        let prepared = self.capture_discovery.as_ref().ok_or_else(|| {
            match construction.text("session has no retained capture catalog") {
                Ok(text) => CaptureError::Unsupported(text),
                Err(error) => error,
            }
        })?;
        let layouts = prepared
            .compile_component_partition_source(1024, None, construction)?
            .ok_or_else(|| {
                match construction
                    .text("distributed capture has no retained architecture placement")
                {
                    Ok(text) => CaptureError::Invalid(text),
                    Err(error) => error,
                }
            })?;
        let setup = transport.session_identity();
        if layouts.layouts().topology().world_size() != setup.participant_count() {
            return Err(CaptureError::Invalid(
                construction.text("capture placement differs from native setup")?,
            )
            .into());
        }
        transport.capture_wait_source(construction)?;
        construction.controls(size_of::<(
            usize,
            CaptureUsage,
            Result<CaptureUsage, CaptureError>,
        )>())?;
        transport.estimate_capture_gather(16)?;
        let collector = |point: &eredu_core::ObservationPoint,
                         construction: CaptureSourceConstruction<'_>| {
            let Some(members) = layouts
                .layouts()
                .capture_hook_members_source(&point.path, construction)?
            else {
                return Ok(ObservationSupportStatus::Unverified(construction.text(
                    "global observation ownership is not yet implemented for this point",
                )?));
            };
            match transport.estimate_capture_hook_source(&members, construction) {
                Ok(_) => Ok(ObservationSupportStatus::Supported),
                // A funding refusal cannot become a semantic unsupported point.
                Err(error @ CaptureError::AdmissionStorage(_)) => Err(error),
                Err(error) => Ok(ObservationSupportStatus::Unverified(
                    construction.format(format_args!("{error}"))?,
                )),
            }
        };
        let mut discovery =
            prepared.capture_with_partition_source(layouts.layouts(), construction, collector)?;
        discovery.support.capture.transformations.retain(|kind| {
            matches!(
                kind,
                CaptureTransformKind::Preview
                    | CaptureTransformKind::Slice
                    | CaptureTransformKind::FullTensor
                    | CaptureTransformKind::Summary
                    | CaptureTransformKind::Histogram
                    | CaptureTransformKind::TokenScores
                    | CaptureTransformKind::TopCandidates
                    | CaptureTransformKind::RoutedUnits
            )
        });
        discovery.support.capture.conditions = construction.texts(&[
            "Component, normalized-input and readout observations with retained global ownership and independent invocation-group failure agreement; one token-ID sequence",
            "Every model rank participates in the same capture plan and forward; records publish after shared completion and commit",
            "Vocabulary reductions require the complete ordered final logits on the retained publication owner; probabilities normalize the full model vocabulary",
            "Global producer, transport, decoder and assembly credits are reserved before model work; restore refunds neither credits nor epochs",
            "Skip-on-limit decisions and consumed credits are agreed before model work; record envelopes and coordination remain mandatory",
            "Generated projection-input logical buffers use the shared reconstruction plan; selected native allocation and workspace bounds require separate admission",
        ])?;
        let execution = construction.text(prepared.execution_identity())?;
        Ok(LoadedPartitionCapture(Publication::new(
            PartitionCaptureData {
                discovery,
                execution,
                setup,
                layouts,
            },
            construction,
        )?))
    }
}

pub(super) fn estimate(
    shape: &[u64],
    selection: &CaptureSelection,
    slice: &ResolvedCaptureSlice,
) -> Result<PartitionCaptureNativeEstimate, CaptureError> {
    let width = shape
        .last()
        .copied()
        .ok_or_else(|| CaptureError::Invalid("scalar observation capture".into()))?;
    let rows = elements(&shape[..shape.len() - 1])?;
    let reconstruction_shape = [
        i32::try_from(rows).map_err(|_| CaptureError::Overflow)?,
        i32::try_from(width).map_err(|_| CaptureError::Overflow)?,
    ];
    let reconstruction = eredu_nn::BlockFp8InputReconstructionPlan::new(&reconstruction_shape)
        .map_err(|_| CaptureError::Overflow)?;
    Ok(PartitionCaptureNativeEstimate {
        capture: super::super::bounded_capture::estimate_shape(shape, selection, slice)?,
        generated_creation_bytes: reconstruction
            .logical_capture_source()
            .map_err(|_| CaptureError::Overflow)?
            .creation_bytes,
    })
}

fn observer<'a>(
    capture: &'a mut CaptureSession,
    stream: &'a Stream,
    domain: Option<CaptureTokenDomain<'a>>,
    prediction: u64,
    transport: &'a MlxDistributedSession,
    loaded: &'a LoadedPartitionCapture,
    identity: PartitionCaptureIdentity,
) -> impl RuntimeActivationObserver<MlxTensor, Error> + 'a {
    let max_record_bytes = capture.plan().plan().limits.per_step.encoded_bytes;
    PartitionCaptureObserver::for_step(
        capture,
        super::super::bounded_capture::NativeCapture {
            partition: Some(transport),
            stream,
            domain,
        },
        transport,
        loaded.layouts(),
        prediction,
        PartitionCaptureReceiptLimits {
            max_producers: loaded.setup.participant_count(),
            max_fragments: 65_536,
            max_record_bytes,
        },
        estimate,
        |error: PartitionCaptureObserverError<Error>| match error {
            PartitionCaptureObserverError::Capture(error) => {
                super::super::bounded_capture::capture_error(error)
            }
            error => Error::observation(error),
        },
    )
    .with_session_identity(identity)
    .with_interventions()
}

struct RejectedCapture(CaptureError, Option<eredu_core::HostPreparationAuthority>);
impl RuntimeActivationObserver<MlxTensor, Error> for RejectedCapture {
    fn transactional(&self) -> bool {
        true
    }
    fn prepare_transaction(
        &mut self,
        _: eredu_core::DistributedCommitEpoch,
        _: eredu_runtime::ExpertPass,
    ) -> Result<(), Error> {
        Err(Error::observation(self.0.clone()).retain_ordinary_capture(self.1.clone()))
    }
    fn observe(&mut self, _: &str, _: &MlxTensor) -> Result<(), Error> {
        Err(Error::observation(self.0.clone()).retain_ordinary_capture(self.1.clone()))
    }
}

/// Keeps native and layout owners alive for the entire shared forward. Any
/// cold composition failure becomes local observer preparation failure so peers
/// can agree rejection before their hooks or mutable model execution.
pub(super) fn with_observer<R>(
    session: &mut MlxModelSession,
    capture: &mut CaptureSession,
    stream: &Stream,
    domain: Option<CaptureTokenDomain<'_>>,
    prediction: u64,
    submit: impl FnOnce(
        &mut MlxModelSession,
        &mut dyn RuntimeActivationObserver<MlxTensor, Error>,
    ) -> Result<R, Error>,
) -> Result<R, Error> {
    let host = capture.ordinary_error_custody().cloned();
    let loaded = session.loaded_partition_capture();
    let transport = session.payload.distributed.clone();
    let mut observer: Box<dyn RuntimeActivationObserver<MlxTensor, Error> + '_> =
        match (transport.as_ref(), loaded.as_ref()) {
            (None, Ok(None)) => Box::new(super::super::bounded_capture::observer(
                capture, stream, domain, prediction,
            )),
            (Some(transport), Ok(Some(loaded))) => {
                match loaded.identity(session.payload.parameter_state.active.as_deref()) {
                    Ok(identity) => Box::new(observer(
                        capture, stream, domain, prediction, transport, loaded, identity,
                    )),
                    Err(error) => Box::new(RejectedCapture(error, host.clone())),
                }
            }
            (_, Err(error)) => Box::new(RejectedCapture(error.clone(), host.clone())),
            _ => Box::new(RejectedCapture(
                CaptureError::Invalid(
                    "native capture owner and retained placement disagree".into(),
                ),
                host.clone(),
            )),
        };
    let result = submit(session, &mut *observer);
    drop(observer);
    result.map_err(|error| error.retain_ordinary_capture(host))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
