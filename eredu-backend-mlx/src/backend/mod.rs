//! MLX backend adapter.

/// Prompt-cache topology conversion for MLX distributed execution.
pub mod cache;
mod compaction;
pub mod config;
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
/// MLX process-local device binding for a canonical core rank topology.
pub mod topology;
pub(crate) use config::ModelLoadOptions;
pub(crate) use distributed::MlxDistributedConfig;
pub(crate) use distributed::MlxDistributedSession;
pub use execution::ExecutionContext;
#[cfg(test)]
pub(crate) use topology::DeviceAssignment;
pub(crate) use topology::MlxParallelContext;

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
    composition::mlx::{distributed::pipeline::PipelineModel, Executable, MlxModelSession},
};

fn device_capabilities(has_world: bool) -> DeviceCapabilities {
    DeviceCapabilities {
        exact_completion: true,
        transfers: true,
        collectives: has_world,
    }
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
        DeviceDescriptor {
            id: format!("{}:{}", self.family, self.index),
            name: format!("MLX {} {}", self.family, self.index),
            family: self.family.into(),
            memory_bytes: None,
        }
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
/// Replicated, tensor-, pipeline-, and expert-parallel materializations share
/// this type. Architecture-specific rank-local executables are deliberately
/// not exposed through the public loading API.
pub struct MlxModel {
    inner: MlxModelKind,
    floating_state_dtype_bytes: NonZeroU8,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
}

pub(crate) enum MlxModelKind {
    Complete(Executable),
    Pipeline(PipelineModel),
}

impl MlxModel {
    pub(crate) const fn complete(model: Executable, floating_state_dtype_bytes: NonZeroU8) -> Self {
        Self {
            inner: MlxModelKind::Complete(model),
            floating_state_dtype_bytes,
            #[cfg(any(feature = "image", feature = "audio"))]
            processor: None,
        }
    }

    pub(crate) const fn pipeline(
        model: PipelineModel,
        floating_state_dtype_bytes: NonZeroU8,
    ) -> Self {
        Self {
            inner: MlxModelKind::Pipeline(model),
            floating_state_dtype_bytes,
            #[cfg(any(feature = "image", feature = "audio"))]
            processor: None,
        }
    }

    /// Wraps a directly constructed replicated model for backend integration tests.
    #[cfg(test)]
    pub const fn complete_for_test(
        model: Executable,
        floating_state_dtype_bytes: NonZeroU8,
    ) -> Self {
        Self::complete(model, floating_state_dtype_bytes)
    }

    pub(crate) const fn floating_state_dtype_bytes(&self) -> NonZeroU8 {
        self.floating_state_dtype_bytes
    }

    /// Reports speculative-weight readiness to backend integration tests.
    #[cfg(test)]
    pub fn speculative_capability_for_test(&self) -> eredu_core::SpeculativeCapability {
        match &self.inner {
            MlxModelKind::Complete(model) => model.speculative_capability(),
            MlxModelKind::Pipeline(model) => model.speculative_capability(),
        }
    }

    pub(crate) fn into_kind(self) -> MlxModelKind {
        self.inner
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
    /// Extracts a complete executable, rejecting a pipeline model.
    pub fn into_complete(self) -> Result<Executable, Error> {
        match self.inner {
            MlxModelKind::Complete(model) => Ok(model),
            MlxModelKind::Pipeline(_) => Err(Error::Parallel(
                "the tokenizer/generation facade requires a replicated model; execute distributed models through MlxModelSession"
                    .into(),
            )),
        }
    }

    /// Returns the selected model's canonical architecture family.
    pub fn model_family(&self) -> eredu_architectures::ModelKind {
        match &self.inner {
            MlxModelKind::Complete(model) => model.model_family(),
            MlxModelKind::Pipeline(model) => model.model_family(),
        }
    }

    /// Returns the effective model type preserved from the parsed configuration.
    pub fn effective_model_type(&self) -> &str {
        match &self.inner {
            MlxModelKind::Complete(model) => model.effective_model_type(),
            MlxModelKind::Pipeline(model) => model.effective_model_type(),
        }
    }

    /// Returns the rank-local topology for a distributed executable.
    pub fn topology(&self) -> Option<crate::backend::MlxParallelContext> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.parallel_info().map(|info| info.topology()),
            MlxModelKind::Pipeline(model) => Some(model.stage_info().topology),
        }
        .filter(|topology| !topology.is_replicated())
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.residency_report(),
            MlxModelKind::Pipeline(model) => model.parameter_residency_report(),
        }
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        match &self.inner {
            MlxModelKind::Complete(model) => model.dense_stream_report(),
            MlxModelKind::Pipeline(model) => model.dense_stream_report(),
        }
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn expert_cache_report(
        &self,
    ) -> Result<Option<crate::backend::runtime::residency::expert_cache::ExpertCacheReport>, Error>
    {
        match &self.inner {
            MlxModelKind::Complete(model) => model.expert_cache_report(),
            MlxModelKind::Pipeline(model) => model.expert_cache_report(),
        }
    }
}

/// Request to prepare any facade-supported model on MLX.
#[derive(Debug, Clone)]
pub struct MlxModelConfig {
    /// Backend-neutral inspected artifact and materialization route.
    pub plan: eredu_core::ModelPreparationPlan<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    /// MLX materialization details for the selected neutral route.
    pub options: ModelLoadOptions,
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

    /// Waits for all work submitted to this backend's execution queue.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.stream.synchronize().map_err(Into::into)
    }

    #[cfg(test)]
    /// Creates test communication for a topology using the backend stream.
    pub fn communication_for_topology(
        &self,
        topology: crate::backend::MlxParallelContext,
        world: &'a safemlx::distributed::Group,
    ) -> Result<MlxDistributedSession<'a>, Error> {
        MlxDistributedSession::new(MlxDistributedConfig { topology, world }, &self.stream)
    }
}

impl<'a> BackendProvider for MlxBackend<'a> {
    type ModelConfig = MlxModelConfig;
    type Model = MlxModel;
    type Session = MlxModelSession<'a>;
    type Error = Error;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "mlx".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
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
        let capabilities = config.plan.admitted_session_capabilities();
        crate::composition::mlx::loading::materialize_model_plan(
            config.plan,
            config.options,
            &self.stream,
            &self.weights_stream,
        )
        .map(|model| PreparedModel::new(model, capabilities))
    }

    fn create_session(
        &self,
        model: PreparedModel<Self::Model>,
    ) -> Result<Self::Session, Self::Error> {
        let admitted = model.capabilities();
        let distributed = match model.topology() {
            Some(topology) => {
                let world = self.world.ok_or_else(|| {
                    Error::Parallel(
                        "distributed model session creation requires native::distributed_backend"
                            .into(),
                    )
                })?;
                Some(MlxDistributedSession::new(
                    MlxDistributedConfig { topology, world },
                    &self.stream,
                )?)
            }
            None => None,
        };
        MlxModelSession::from_model(model.into_inner(), distributed, admitted)
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
    use super::{device_capabilities, MlxBackend, MlxDeviceIdentity};
    use crate::backend::ExecutionContext;
    use eredu_core::BackendProvider as _;
    use safemlx::{Device, DeviceType};

    #[test]
    fn collective_capability_requires_an_attached_world() {
        assert!(!device_capabilities(false).collectives);
        assert!(device_capabilities(true).collectives);
    }

    #[test]
    fn ordinary_backend_device_report_is_fail_closed_for_collectives() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let backend = MlxBackend::new(execution.stream(), execution.stream());
        let devices = backend.devices().unwrap();
        assert_eq!(devices.len(), 1);
        assert!(!devices[0].1.collectives);
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
}

impl ModelLoadingBackend for MlxBackend<'_> {
    type LoadOptions = ModelLoadOptions;
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

    fn validate_preparation(
        &self,
        inspection: &eredu_core::ArtifactInspection<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        policy: eredu_core::PreparationPolicy,
    ) -> Result<(), Self::Error> {
        crate::composition::mlx::structural::validate_inspected_preparation(inspection, policy)
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
        plan: eredu_core::ModelPreparationPlan<
            eredu_architectures::processor_plan::ArtifactArchitecturePlan,
        >,
        options: Self::LoadOptions,
    ) -> Result<Self::ModelConfig, Self::Error> {
        Ok(MlxModelConfig { plan, options })
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
