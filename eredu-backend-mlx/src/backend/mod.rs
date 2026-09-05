//! MLX backend adapter.

pub(crate) mod compaction;
/// Session-owned MLX communicators, transfers, and collectives.
pub mod distributed;
/// Errors produced by MLX model loading and execution.
pub mod error;
mod execution;
#[cfg(any(feature = "image", feature = "audio"))]
mod media;
/// Reusable MLX neural-network building blocks.
pub mod nn;
/// Stateful random-key ownership for backend sessions.
pub mod random;
/// MLX allocator observations for neutral residency telemetry.
pub mod residency;
/// MLX-only tensor, checkpoint, execution, and residency infrastructure.
pub mod runtime;
/// MLX process-local device binding for composition-owned rank topology.
pub(crate) mod topology;
pub(crate) use distributed::MlxDistributedSession;
pub use execution::ExecutionContext;
pub use topology::{DeviceAssignment, MlxRankContext};

use eredu_core::backend::{
    BackendDescriptor, BackendProvider, Completion, DeviceCapabilities, DeviceDescriptor,
    ModelLoadingBackend, PreparedModel, SessionCapabilities, Submission,
};
use std::num::NonZeroU8;

use safemlx::{transforms::async_eval_with_event, Array, Device, DeviceType, Event, Stream};

#[cfg(any(feature = "image", feature = "audio"))]
use crate::composition::mlx::ModelProcessor;
use crate::{
    backend::error::Error,
    composition::mlx::{Executable, MlxModelSession},
    MlxLoadRequest,
};

fn device_capabilities(has_world: bool) -> DeviceCapabilities {
    DeviceCapabilities::new(true, true, has_world)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum MlxAcceleratorFamily {
    Metal,
    Cuda,
}

impl MlxAcceleratorFamily {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub(crate) const fn is_compiled(self) -> bool {
        match self {
            Self::Metal => cfg!(all(feature = "metal", target_vendor = "apple")),
            Self::Cuda => cfg!(feature = "cuda"),
        }
    }

    pub(crate) fn is_available(self) -> Result<bool, Error> {
        match self {
            Self::Metal => {
                #[cfg(all(feature = "metal", target_vendor = "apple"))]
                {
                    safemlx::metal::is_available().map_err(Into::into)
                }
                #[cfg(not(all(feature = "metal", target_vendor = "apple")))]
                {
                    Ok(false)
                }
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    safemlx::cuda::is_available().map_err(Into::into)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok(false)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct MlxDeviceIdentity {
    kind: DeviceType,
    family: &'static str,
    index: i32,
}

impl MlxDeviceIdentity {
    pub(crate) fn from_realized_device(
        device: &Device,
        accelerator_family: Option<MlxAcceleratorFamily>,
    ) -> Result<Self, Error> {
        let kind = device.get_type()?;
        let family = match (kind, accelerator_family) {
            (DeviceType::Cpu, None) => "cpu",
            (DeviceType::Gpu, Some(family)) => family.as_str(),
            (DeviceType::Cpu, Some(family)) => {
                return Err(Error::AutomaticPlanning(format!(
                    "realized CPU device cannot have accelerator family {}",
                    family.as_str()
                )))
            }
            (DeviceType::Gpu, None) => {
                return Err(Error::AutomaticPlanning(
                    "realized GPU device is missing its concrete accelerator family".into(),
                ))
            }
        };
        Ok(Self {
            kind,
            family,
            index: device.get_index()?,
        })
    }

    fn validate_device(&self, device: &Device) -> Result<(), Error> {
        let kind = device.get_type()?;
        let index = device.get_index()?;
        if kind != self.kind || index != self.index {
            return Err(Error::AutomaticPlanning(format!(
                "realized device identity {}:{} does not match backend stream device {kind:?}:{index}",
                self.family, self.index
            )));
        }
        Ok(())
    }

    fn descriptor(&self) -> DeviceDescriptor {
        DeviceDescriptor::new(
            format!("{}:{}", self.family, self.index),
            format!("MLX {} {}", self.family, self.index),
            self.family,
            None,
        )
    }
}

fn infer_native_device_identity(device: &Device) -> Result<MlxDeviceIdentity, Error> {
    if device.get_type()? == DeviceType::Cpu {
        return MlxDeviceIdentity::from_realized_device(device, None);
    }

    let mut available = Vec::new();
    for family in [MlxAcceleratorFamily::Metal, MlxAcceleratorFamily::Cuda] {
        if family.is_compiled() && family.is_available()? {
            available.push(family);
        }
    }
    let family = available.first().copied().ok_or_else(|| {
        Error::AutomaticPlanning(
            "cannot identify native MLX GPU stream because no compiled accelerator family is available"
                .into(),
        )
    })?;
    if available.len() != 1 {
        return Err(Error::AutomaticPlanning(
            "cannot identify native MLX GPU stream because multiple accelerator families are available"
                .into(),
        ));
    }
    MlxDeviceIdentity::from_realized_device(device, Some(family))
}

/// Opaque MLX executable selected for one complete model session.
///
/// Replicated and axis-partitioned materializations share
/// this type. Architecture-specific rank-local executables are deliberately
/// not exposed through the public loading API.
pub struct MlxModel {
    executable: Executable,
    floating_state_dtype_bytes: NonZeroU8,
    state_residency: eredu_runtime::CacheResidencyPolicy,
    distributed: Option<MlxDistributedSession>,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
}

impl MlxModel {
    pub(crate) fn new(
        model: Executable,
        floating_state_dtype_bytes: NonZeroU8,
        state_residency: eredu_runtime::CacheResidencyPolicy,
    ) -> Self {
        Self {
            executable: model,
            floating_state_dtype_bytes,
            state_residency,
            distributed: None,
            #[cfg(any(feature = "image", feature = "audio"))]
            processor: None,
        }
    }

    pub(crate) const fn floating_state_dtype_bytes(&self) -> NonZeroU8 {
        self.floating_state_dtype_bytes
    }

    pub(crate) const fn state_residency(&self) -> &eredu_runtime::CacheResidencyPolicy {
        &self.state_residency
    }

    /// Reports speculative-weight readiness to backend integration tests.
    #[cfg(test)]
    pub fn speculative_capability_for_test(&self) -> eredu_core::SpeculativeCapability {
        self.executable.speculative_capability()
    }

    pub(crate) fn into_executable(self) -> Executable {
        self.executable
    }

    pub(crate) fn take_distributed(&mut self) -> Option<MlxDistributedSession> {
        self.distributed.take()
    }

    pub(crate) fn with_distributed(mut self, distributed: MlxDistributedSession) -> Self {
        self.distributed = Some(distributed);
        self
    }

    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn take_processor(&mut self) -> Option<ModelProcessor> {
        self.processor.take()
    }

    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn with_processor(mut self, processor: Option<ModelProcessor>) -> Self {
        self.processor = processor;
        self
    }

    #[cfg(test)]
    pub(crate) fn effective_model_type(&self) -> &str {
        self.executable.effective_model_type()
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        self.executable.residency_report()
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.executable.dense_stream_report()
    }

    /// Returns load-time weight transformation telemetry when present.
    pub fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.executable.materialization_report()
    }

    /// Returns addressable parameter-bank residency telemetry when enabled.
    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        self.executable.parameter_bank_report()
    }
}

/// MLX backend selected for a complete model/session.
pub struct MlxBackend<'a> {
    stream: Stream,
    weights_stream: Stream,
    realized_device: Option<MlxDeviceIdentity>,
    world: Option<&'a safemlx::distributed::Group>,
}

impl MlxBackend<'static> {
    pub(crate) fn new(stream: &Stream, weights_stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            realized_device: None,
            world: None,
        }
    }

    pub(crate) fn for_execution_plan(
        stream: &Stream,
        weights_stream: &Stream,
        realized_device: MlxDeviceIdentity,
    ) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            realized_device: Some(realized_device),
            world: None,
        }
    }
}

impl<'a> MlxBackend<'a> {
    pub(crate) fn with_distributed_world(
        stream: &Stream,
        weights_stream: &Stream,
        world: &'a safemlx::distributed::Group,
    ) -> Self {
        Self {
            stream: stream.clone(),
            weights_stream: weights_stream.clone(),
            realized_device: None,
            world: Some(world),
        }
    }
    pub(crate) const fn stream(&self) -> &Stream {
        &self.stream
    }

    pub(crate) const fn weights_stream(&self) -> &Stream {
        &self.weights_stream
    }

    fn realize_selected_communication(
        &self,
        manifest: Option<&eredu_runtime::CommunicationManifest>,
        rank: Option<crate::backend::MlxRankContext>,
    ) -> Result<Option<MlxDistributedSession>, Error> {
        let Some(manifest) = manifest else {
            return match rank {
                None => Ok(None),
                Some(_) => Err(Error::Parallel(
                    "distributed MLX preparation has no architecture communication manifest".into(),
                )),
            };
        };
        let rank = rank.ok_or_else(|| {
            Error::Parallel("communication manifest has no MLX rank/device context".into())
        })?;
        #[cfg(test)]
        crate::composition::mlx::path_instrumentation::communication_realization_attempt();
        let world = self.world.ok_or_else(|| {
            Error::Parallel(
                "distributed model preparation requires native::distributed_backend".into(),
            )
        })?;
        rank.validate_execution_stream(&self.stream)?;
        #[cfg(test)]
        crate::composition::mlx::path_instrumentation::manifest_communication_realization_attempt();
        MlxDistributedSession::from_manifest(manifest, world, &self.stream).map(Some)
    }

    fn materialize_after_communication(
        &self,
        capabilities: SessionCapabilities,
        manifest: Option<&eredu_runtime::CommunicationManifest>,
        rank: Option<crate::backend::MlxRankContext>,
        materialize: impl FnOnce(Option<MlxDistributedSession>) -> Result<MlxModel, Error>,
    ) -> Result<PreparedModel<MlxModel>, Error> {
        let distributed = self.realize_selected_communication(manifest, rank)?;
        materialize(distributed).map(|model| PreparedModel::new(model, capabilities))
    }

    /// Waits for all work submitted to this backend's execution queue.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.stream.synchronize().map_err(Into::into)
    }
}

impl<'a> BackendProvider for MlxBackend<'a> {
    type ModelConfig = crate::composition::mlx::loading::MlxModelConfig;
    type Model = MlxModel;
    type Session = MlxModelSession;
    type Error = Error;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("mlx", env!("CARGO_PKG_VERSION"))
    }

    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        let device = self.stream.get_device()?;
        let identity = match &self.realized_device {
            Some(identity) => {
                identity.validate_device(&device)?;
                identity.clone()
            }
            None => infer_native_device_identity(&device)?,
        };
        Ok(vec![(
            identity.descriptor(),
            device_capabilities(self.world.is_some()),
        )])
    }

    fn prepare_model(
        &self,
        config: Self::ModelConfig,
    ) -> Result<PreparedModel<Self::Model>, Self::Error> {
        let capabilities = config.selected.session_capabilities();
        if config.plan.admitted_session_capabilities() != capabilities {
            return Err(Error::ArchitectureModel(
                "selected session facilities differ from the admitted preparation plan".into(),
            ));
        }
        let rank = config.selected.rank_context();
        let manifest = config.selected.realized_communication_manifest().cloned();
        self.materialize_after_communication(capabilities, manifest.as_ref(), rank, |distributed| {
            crate::composition::mlx::loading::materialize_model_plan(
                config.plan,
                config.selected,
                distributed,
                &self.stream,
                &self.weights_stream,
            )
        })
    }

    fn create_session(
        &self,
        model: PreparedModel<Self::Model>,
    ) -> Result<Self::Session, Self::Error> {
        let admitted = model.capabilities();
        MlxModelSession::from_model(model.into_inner(), admitted)
    }

    fn session_capability_mismatch(
        &self,
        admitted: SessionCapabilities,
        realized: SessionCapabilities,
    ) -> Self::Error {
        Error::ArchitectureModel(format!(
            "realized MLX session capabilities {realized:?} do not match pre-materialization admission {admitted:?}"
        ))
    }
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "backend-selection tests stay adjacent to the selection implementation"
)]
mod tests {
    use super::{device_capabilities, MlxBackend, MlxDeviceIdentity, MlxModel};
    use crate::backend::ExecutionContext;
    use crate::composition::mlx::path_instrumentation;
    use eredu_core::BackendProvider as _;
    use safemlx::{Device, DeviceType};

    #[test]
    fn collective_capability_requires_an_attached_world() {
        assert!(!device_capabilities(false).collectives());
        assert!(device_capabilities(true).collectives());
    }

    #[test]
    fn ordinary_backend_device_report_is_fail_closed_for_collectives() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let backend = MlxBackend::new(execution.stream(), execution.stream());
        let devices = backend.devices().unwrap();
        assert_eq!(devices.len(), 1);
        assert!(!devices[0].1.collectives());
    }

    #[test]
    fn planned_backend_rejects_a_stream_that_differs_from_its_realized_device() {
        let realized = Device::new(DeviceType::Cpu, 0);
        let identity = MlxDeviceIdentity::from_realized_device(&realized, None).unwrap();
        let other = ExecutionContext::new(Device::new(DeviceType::Cpu, 1));
        let backend = MlxBackend::for_execution_plan(other.stream(), other.stream(), identity);

        let error = backend.devices().unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match backend stream device"));
    }

    #[test]
    fn missing_world_rejects_before_payload_or_architecture_construction() {
        path_instrumentation::reset();
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let backend = MlxBackend::new(execution.stream(), execution.stream());
        let manifest = eredu_runtime::CommunicationManifest::new(2, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let rank =
            super::MlxRankContext::new(2, 0, super::DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();

        let error = match backend.materialize_after_communication(
            eredu_core::SessionCapabilities::default(),
            Some(&manifest),
            Some(rank),
            |_| -> Result<MlxModel, super::Error> {
                path_instrumentation::payload_open();
                path_instrumentation::architecture_construction();
                unreachable!("missing world must reject before materialization")
            },
        ) {
            Ok(_) => panic!("missing world unexpectedly reached materialization"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("distributed model preparation requires native::distributed_backend"));
        assert_eq!(
            path_instrumentation::communication_realization_attempts(),
            1
        );
        assert_eq!(path_instrumentation::snapshot(), Default::default());
    }

    #[test]
    fn mismatched_world_rejects_before_payload_or_architecture_construction() {
        path_instrumentation::reset();
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        assert_eq!(world.size(), 1);
        let backend =
            MlxBackend::with_distributed_world(execution.stream(), execution.stream(), &world);
        let manifest = eredu_runtime::CommunicationManifest::new(2, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let rank =
            super::MlxRankContext::new(2, 0, super::DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();

        let error = match backend.materialize_after_communication(
            eredu_core::SessionCapabilities::default(),
            Some(&manifest),
            Some(rank),
            |_| -> Result<MlxModel, super::Error> {
                path_instrumentation::payload_open();
                path_instrumentation::architecture_construction();
                unreachable!("mismatched world must reject before materialization")
            },
        ) {
            Ok(_) => panic!("mismatched world unexpectedly reached materialization"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("communication projection has 1 manifests, expected 2"),
            "unexpected world mismatch: {error}"
        );
        assert_eq!(
            path_instrumentation::communication_realization_attempts(),
            1
        );
        assert_eq!(path_instrumentation::snapshot(), Default::default());
    }

    #[test]
    fn selected_manifest_rejects_before_payload_or_architecture_construction() {
        path_instrumentation::reset();
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let world = safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        assert_eq!(world.size(), 1);
        let backend =
            MlxBackend::with_distributed_world(execution.stream(), execution.stream(), &world);
        let manifest = eredu_runtime::CommunicationManifest::new(2, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );

        let error = match backend.materialize_after_communication(
            eredu_core::SessionCapabilities::default(),
            Some(&manifest),
            None,
            |_| -> Result<MlxModel, super::Error> {
                path_instrumentation::payload_open();
                path_instrumentation::architecture_construction();
                unreachable!("mismatched manifest world must reject before materialization")
            },
        ) {
            Ok(_) => panic!("mismatched manifest world unexpectedly reached materialization"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("communication manifest has no MLX rank/device context"));
        assert_eq!(
            path_instrumentation::communication_realization_attempts(),
            0
        );
        assert_eq!(path_instrumentation::snapshot(), Default::default());
    }
}

impl ModelLoadingBackend for MlxBackend<'_> {
    type LoadOptions = MlxLoadRequest;
    type SelectedPreparation = crate::composition::mlx::loading::MlxSelectedPreparation;
    type ConfigurationResolver = eredu_architectures::configuration::ModelConfigurations;

    fn configuration_resolver(&self) -> &Self::ConfigurationResolver {
        &eredu_architectures::configuration::MODEL_CONFIGURATIONS
    }

    fn preparation_policy(
        &self,
        options: &Self::LoadOptions,
    ) -> Result<eredu_core::PreparationPolicy, Self::Error> {
        options.preparation_policy()
    }

    fn select_preparation(
        &self,
        inspection: &eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        options: &Self::LoadOptions,
        policy: eredu_core::PreparationPolicy,
    ) -> Result<Self::SelectedPreparation, Self::Error> {
        crate::composition::mlx::loading::select_preparation(inspection, options.clone(), policy)
    }

    fn session_capabilities(
        &self,
        inspection: &eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        policy: eredu_core::PreparationPolicy,
    ) -> Result<SessionCapabilities, Self::Error> {
        crate::composition::mlx::structural::inspected_session_capabilities(inspection, policy)
    }

    fn model_config(
        &self,
        selected: eredu_core::SelectedModelPreparation<Self>,
    ) -> Result<Self::ModelConfig, Self::Error> {
        Ok(crate::composition::mlx::loading::MlxModelConfig::new(
            selected,
        ))
    }
}

/// Exact MLX event plus retained output arrays.
pub struct MlxCompletion {
    event: Event,
    retained: Vec<Array>,
}

impl Completion for MlxCompletion {
    type Error = Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.event.is_complete().map_err(Into::into)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize().map_err(Into::into)
    }
}

impl Drop for MlxCompletion {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}

impl MlxCompletion {
    pub(crate) fn submission(output: Array) -> Result<Submission<Array, Self>, Error> {
        Self::submission_retaining(output, std::iter::empty())
    }

    pub(crate) fn submission_retaining(
        output: Array,
        additional: impl IntoIterator<Item = Array>,
    ) -> Result<Submission<Array, Self>, Error> {
        let retained = std::iter::once(output.clone())
            .chain(additional)
            .collect::<Vec<_>>();
        let event = async_eval_with_event(retained.iter())?;
        Ok(Submission {
            output,
            completion: Self { event, retained },
        })
    }

    /// Number of arrays held until exact completion.
    pub fn retained_resources(&self) -> usize {
        self.retained.len()
    }
}
