//! Portable preparation admission over architecture and backend mechanism facts.

use serde::{Deserialize, Serialize};

use crate::{
    ArtifactFormat, InputModalities, LoadingProtocol, MaterializationRoute, ParallelAxis,
    PreparationPolicy, ResidencyRequest, SessionCapabilities,
};

/// Preparation facts supplied by one exact normalized architecture.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArchitecturePreparationCapabilities {
    independently_addressable_parameters: bool,
    nonresident_safetensors_quantization: bool,
    parallel_axes: [bool; 3],
    input_modalities: InputModalities,
}

impl ArchitecturePreparationCapabilities {
    /// Creates an exact architecture capability report.
    pub const fn new(
        independently_addressable_parameters: bool,
        nonresident_safetensors_quantization: bool,
        tensor_parallel: bool,
        pipeline_parallel: bool,
        expert_parallel: bool,
        input_modalities: InputModalities,
    ) -> Self {
        Self {
            independently_addressable_parameters,
            nonresident_safetensors_quantization,
            parallel_axes: [tensor_parallel, pipeline_parallel, expert_parallel],
            input_modalities,
        }
    }

    /// Whether parameters may be admitted as independently addressable banks.
    pub const fn independently_addressable_parameters(self) -> bool {
        self.independently_addressable_parameters
    }

    /// Whether SafeTensors transformation is valid under nonresident execution.
    pub const fn nonresident_safetensors_quantization(self) -> bool {
        self.nonresident_safetensors_quantization
    }

    /// Whether the architecture defines semantics for one parallel axis.
    pub const fn supports_parallel_axis(self, axis: ParallelAxis) -> bool {
        match axis {
            ParallelAxis::Tensor => self.parallel_axes[0],
            ParallelAxis::Pipeline => self.parallel_axes[1],
            ParallelAxis::Expert => self.parallel_axes[2],
            ParallelAxis::Data => true,
        }
    }

    /// Input modalities supported by the normalized architecture.
    pub const fn input_modalities(self) -> InputModalities {
        self.input_modalities
    }
}

/// Cold mechanism facts supplied by a concrete backend adapter.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparationMechanismCapabilities {
    safetensors: bool,
    gguf: bool,
    safetensors_quantization: bool,
    nonresident_safetensors_quantization: bool,
    gguf_quantized_loading: bool,
    residency: [bool; 4],
    parallel_axes: [bool; 4],
    input_modalities: InputModalities,
    exact_completion: bool,
    session: SessionCapabilities,
}

impl PreparationMechanismCapabilities {
    /// Creates a fail-closed mechanism report for supported artifact containers.
    pub const fn new(safetensors: bool, gguf: bool) -> Self {
        Self {
            safetensors,
            gguf,
            safetensors_quantization: false,
            nonresident_safetensors_quantization: false,
            gguf_quantized_loading: false,
            residency: [false; 4],
            parallel_axes: [false; 4],
            input_modalities: InputModalities {
                text: false,
                image: false,
                audio: false,
                video: false,
            },
            exact_completion: false,
            session: SessionCapabilities::new(false, false, false),
        }
    }

    /// Reports SafeTensors transformation support.
    pub const fn with_safetensors_quantization(
        mut self,
        resident: bool,
        nonresident: bool,
    ) -> Self {
        self.safetensors_quantization = resident;
        self.nonresident_safetensors_quantization = nonresident;
        self
    }

    /// Reports whether packed GGUF values may satisfy quantized loading requests.
    pub const fn with_gguf_quantized_loading(mut self, supported: bool) -> Self {
        self.gguf_quantized_loading = supported;
        self
    }

    /// Reports support for one residency family.
    pub const fn with_residency(mut self, residency: ResidencyRequest, supported: bool) -> Self {
        self.residency[residency_index(residency)] = supported;
        self
    }

    /// Reports support for one logical parallel axis.
    pub const fn with_parallel_axis(mut self, axis: ParallelAxis, supported: bool) -> Self {
        self.parallel_axes[axis_index(axis)] = supported;
        self
    }

    /// Reports native input conversion mechanisms.
    pub const fn with_input_modalities(mut self, modalities: InputModalities) -> Self {
        self.input_modalities = modalities;
        self
    }

    /// Reports deterministic completion of all native preparation work.
    pub const fn with_exact_completion(mut self, supported: bool) -> Self {
        self.exact_completion = supported;
        self
    }

    /// Reports facilities of the exact session that would be realized.
    pub const fn with_session(mut self, session: SessionCapabilities) -> Self {
        self.session = session;
        self
    }

    /// Exact session facilities available after realization.
    pub const fn session(self) -> SessionCapabilities {
        self.session
    }

    const fn supports_format(self, format: ArtifactFormat) -> bool {
        match format {
            ArtifactFormat::SafeTensors => self.safetensors,
            ArtifactFormat::Gguf => self.gguf,
        }
    }

    const fn supports_residency(self, residency: ResidencyRequest) -> bool {
        self.residency[residency_index(residency)]
    }

    const fn supports_parallel_axis(self, axis: ParallelAxis) -> bool {
        self.parallel_axes[axis_index(axis)]
    }

    /// Native input conversion mechanisms reported by the backend.
    pub const fn input_modalities(self) -> InputModalities {
        self.input_modalities
    }
}

const fn residency_index(residency: ResidencyRequest) -> usize {
    match residency {
        ResidencyRequest::FullyResident => 0,
        ResidencyRequest::LayerwiseHost => 1,
        ResidencyRequest::DenseDiskStream => 2,
        ResidencyRequest::AddressableParameterBanks => 3,
    }
}

const fn axis_index(axis: ParallelAxis) -> usize {
    match axis {
        ParallelAxis::Tensor => 0,
        ParallelAxis::Pipeline => 1,
        ParallelAxis::Expert => 2,
        ParallelAxis::Data => 3,
    }
}

/// Exact immutable inputs to portable preparation admission.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparationAdmissionRequest {
    protocol: LoadingProtocol,
    format: ArtifactFormat,
    policy: PreparationPolicy,
    architecture: ArchitecturePreparationCapabilities,
    required_input_modalities: InputModalities,
    exact_completion: bool,
}

impl PreparationAdmissionRequest {
    /// Creates an admission request for ordinary text execution.
    pub const fn new(
        protocol: LoadingProtocol,
        format: ArtifactFormat,
        policy: PreparationPolicy,
        architecture: ArchitecturePreparationCapabilities,
    ) -> Self {
        Self {
            protocol,
            format,
            policy,
            architecture,
            required_input_modalities: InputModalities::TEXT,
            exact_completion: false,
        }
    }

    /// Selects the input mechanisms required from this exact preparation.
    pub const fn with_required_input_modalities(mut self, modalities: InputModalities) -> Self {
        self.required_input_modalities = modalities;
        self
    }

    /// Requires deterministic completion before prepared-session publication.
    pub const fn with_exact_completion(mut self, required: bool) -> Self {
        self.exact_completion = required;
        self
    }
}

/// Retained result of portable architecture/policy/mechanism admission.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparationAdmission {
    request: PreparationAdmissionRequest,
    mechanisms: PreparationMechanismCapabilities,
    route: MaterializationRoute,
}

impl PreparationAdmission {
    /// Immutable facts admitted by the neutral selector.
    pub const fn request(self) -> PreparationAdmissionRequest {
        self.request
    }

    /// Selected portable materialization route.
    pub const fn route(self) -> MaterializationRoute {
        self.route
    }

    /// Exact session facilities admitted before native realization.
    pub const fn session_capabilities(self) -> SessionCapabilities {
        self.mechanisms.session()
    }

    /// Exact backend facts used by this selection.
    pub const fn mechanisms(self) -> PreparationMechanismCapabilities {
        self.mechanisms
    }
}

/// Stable structured reason for rejecting preparation before native work.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PreparationAdmissionError {
    /// The artifact selects a different lifecycle contract.
    #[error("loading protocol {0:?} is not supported by whole-model preparation")]
    UnsupportedLoadingProtocol(LoadingProtocol),
    /// The backend cannot open the admitted artifact format.
    #[error("backend mechanisms do not support {0:?} artifacts")]
    UnsupportedArtifactFormat(ArtifactFormat),
    /// The requested residency mechanism is unavailable.
    #[error("backend mechanisms do not support {0:?} weight residency")]
    UnsupportedResidency(ResidencyRequest),
    /// The architecture cannot expose independently addressable parameters.
    #[error("architecture does not declare independently addressable parameter banks")]
    ArchitectureParameterBanks,
    /// The requested quantization route is unavailable.
    #[error("{0}")]
    UnsupportedQuantization(&'static str),
    /// Architecture semantics do not define an active topology axis.
    #[error("architecture does not declare {0:?} parallel semantics")]
    ArchitectureParallelAxis(ParallelAxis),
    /// Backend mechanisms cannot realize an active topology axis.
    #[error("backend mechanisms do not support {0:?} parallel execution")]
    BackendParallelAxis(ParallelAxis),
    /// The architecture does not accept a required input modality.
    #[error("architecture does not support required {0} input")]
    ArchitectureInputModality(&'static str),
    /// The backend does not provide a required native input conversion.
    #[error("backend mechanisms do not support required {0} input")]
    BackendInputModality(&'static str),
    /// The backend cannot deterministically complete preparation work.
    #[error("backend mechanisms do not support exact preparation completion")]
    BackendCompletion,
    /// The exact realized session would omit a required facility.
    #[error("prepared session does not support required capability {0}")]
    SessionCapability(&'static str),
}

/// Intersects exact architecture, request, artifact, and backend facts once.
pub fn admit_preparation(
    request: PreparationAdmissionRequest,
    mechanisms: PreparationMechanismCapabilities,
) -> Result<PreparationAdmission, PreparationAdmissionError> {
    if request.protocol != LoadingProtocol::Model {
        return Err(PreparationAdmissionError::UnsupportedLoadingProtocol(
            request.protocol,
        ));
    }
    if !mechanisms.supports_format(request.format) {
        return Err(PreparationAdmissionError::UnsupportedArtifactFormat(
            request.format,
        ));
    }
    let residency = request.policy.residency();
    if !mechanisms.supports_residency(residency) {
        return Err(PreparationAdmissionError::UnsupportedResidency(residency));
    }
    if residency == ResidencyRequest::AddressableParameterBanks
        && !request.architecture.independently_addressable_parameters()
    {
        return Err(PreparationAdmissionError::ArchitectureParameterBanks);
    }
    if request.policy.quantization().is_some() {
        let distributed = request
            .policy
            .topology()
            .is_some_and(|topology| !topology.is_replicated());
        match request.format {
            ArtifactFormat::SafeTensors if !mechanisms.safetensors_quantization => {
                return Err(PreparationAdmissionError::UnsupportedQuantization(
                    "backend mechanisms do not support SafeTensors load-time quantization",
                ));
            }
            ArtifactFormat::SafeTensors
                if !distributed
                    && residency != ResidencyRequest::FullyResident
                    && !request.architecture.nonresident_safetensors_quantization() =>
            {
                return Err(PreparationAdmissionError::UnsupportedQuantization(
                    "architecture does not support nonresident SafeTensors quantization",
                ));
            }
            ArtifactFormat::SafeTensors
                if !distributed
                    && residency != ResidencyRequest::FullyResident
                    && !mechanisms.nonresident_safetensors_quantization =>
            {
                return Err(PreparationAdmissionError::UnsupportedQuantization(
                    "backend mechanisms do not support nonresident SafeTensors quantization",
                ));
            }
            ArtifactFormat::Gguf if !mechanisms.gguf_quantized_loading => {
                return Err(PreparationAdmissionError::UnsupportedQuantization(
                    "backend mechanisms do not support quantized GGUF loading",
                ));
            }
            _ => {}
        }
    }
    if let Some(topology) = request.policy.topology() {
        for axis in [
            ParallelAxis::Tensor,
            ParallelAxis::Pipeline,
            ParallelAxis::Expert,
            ParallelAxis::Data,
        ] {
            if topology.is_axis_active(axis) {
                if !request.architecture.supports_parallel_axis(axis) {
                    return Err(PreparationAdmissionError::ArchitectureParallelAxis(axis));
                }
                if !mechanisms.supports_parallel_axis(axis) {
                    return Err(PreparationAdmissionError::BackendParallelAxis(axis));
                }
            }
        }
    }
    validate_modalities(
        request.required_input_modalities,
        request.architecture.input_modalities(),
        true,
    )?;
    if request.exact_completion && !mechanisms.exact_completion {
        return Err(PreparationAdmissionError::BackendCompletion);
    }
    validate_modalities(
        request.required_input_modalities,
        mechanisms.input_modalities(),
        false,
    )?;
    request
        .policy
        .required_session_capabilities()
        .validate(&mechanisms.session())
        .map_err(|error| PreparationAdmissionError::SessionCapability(error.capability()))?;
    let route = match residency {
        ResidencyRequest::FullyResident => MaterializationRoute::Resident,
        ResidencyRequest::LayerwiseHost | ResidencyRequest::DenseDiskStream => {
            MaterializationRoute::Layerwise
        }
        ResidencyRequest::AddressableParameterBanks => {
            MaterializationRoute::AddressableParameterBanks
        }
    };
    Ok(PreparationAdmission {
        request,
        mechanisms,
        route,
    })
}

fn validate_modalities(
    required: InputModalities,
    available: InputModalities,
    architecture: bool,
) -> Result<(), PreparationAdmissionError> {
    for (required, available, name) in [
        (required.text, available.text, "text"),
        (required.image, available.image, "image"),
        (required.audio, available.audio, "audio"),
        (required.video, available.video, "video"),
    ] {
        if required && !available {
            return Err(if architecture {
                PreparationAdmissionError::ArchitectureInputModality(name)
            } else {
                PreparationAdmissionError::BackendInputModality(name)
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ParallelTopology, QuantizationRequest};
    use std::cell::Cell;

    fn architecture() -> ArchitecturePreparationCapabilities {
        ArchitecturePreparationCapabilities::new(
            true,
            true,
            true,
            true,
            false,
            InputModalities {
                text: true,
                image: true,
                audio: false,
                video: false,
            },
        )
    }

    fn mechanisms() -> PreparationMechanismCapabilities {
        PreparationMechanismCapabilities::new(true, true)
            .with_safetensors_quantization(true, true)
            .with_gguf_quantized_loading(true)
            .with_residency(ResidencyRequest::FullyResident, true)
            .with_residency(ResidencyRequest::LayerwiseHost, true)
            .with_residency(ResidencyRequest::DenseDiskStream, true)
            .with_residency(ResidencyRequest::AddressableParameterBanks, true)
            .with_parallel_axis(ParallelAxis::Tensor, true)
            .with_parallel_axis(ParallelAxis::Pipeline, true)
            .with_parallel_axis(ParallelAxis::Data, true)
            .with_input_modalities(InputModalities {
                text: true,
                image: true,
                audio: false,
                video: false,
            })
            .with_exact_completion(true)
            .with_session(SessionCapabilities::new(true, true, false))
    }

    #[test]
    fn neutral_selector_retains_the_exact_intersection() {
        let policy = PreparationPolicy::new(
            Some(QuantizationRequest::Affine {
                group_size: 32,
                bits: 4,
            }),
            ResidencyRequest::LayerwiseHost,
        )
        .with_topology(ParallelTopology::new(2, 2, 1, 1).unwrap())
        .with_required_session_capabilities(SessionCapabilities::new(true, true, false));
        let request = PreparationAdmissionRequest::new(
            LoadingProtocol::Model,
            ArtifactFormat::SafeTensors,
            policy,
            architecture(),
        )
        .with_exact_completion(true)
        .with_required_input_modalities(InputModalities {
            text: true,
            image: true,
            audio: false,
            video: false,
        });
        let admitted = admit_preparation(request, mechanisms()).unwrap();
        assert_eq!(admitted.request(), request);
        assert_eq!(admitted.route(), MaterializationRoute::Layerwise);
        assert_eq!(
            admitted.session_capabilities(),
            SessionCapabilities::new(true, true, false)
        );
    }

    #[test]
    fn partial_independent_backend_rejects_before_native_work() {
        struct IndependentBackend {
            native_allocations: Cell<usize>,
            payload_reads: Cell<usize>,
        }

        impl IndependentBackend {
            fn capabilities(&self) -> PreparationMechanismCapabilities {
                PreparationMechanismCapabilities::new(true, false)
                    .with_residency(ResidencyRequest::FullyResident, true)
                    .with_input_modalities(InputModalities::TEXT)
            }

            fn realize(&self, _: PreparationAdmission) {
                self.native_allocations
                    .set(self.native_allocations.get() + 1);
                self.payload_reads.set(self.payload_reads.get() + 1);
            }
        }

        let backend = IndependentBackend {
            native_allocations: Cell::new(0),
            payload_reads: Cell::new(0),
        };
        let request = PreparationAdmissionRequest::new(
            LoadingProtocol::Model,
            ArtifactFormat::SafeTensors,
            PreparationPolicy::default()
                .with_required_session_capabilities(SessionCapabilities::new(false, true, false)),
            architecture(),
        );
        let error = admit_preparation(request, backend.capabilities()).unwrap_err();
        assert_eq!(
            error,
            PreparationAdmissionError::SessionCapability("output_observation")
        );
        assert_eq!(backend.native_allocations.get(), 0);
        assert_eq!(backend.payload_reads.get(), 0);

        let media_request = request.with_required_input_modalities(InputModalities {
            text: true,
            image: true,
            audio: false,
            video: false,
        });
        assert_eq!(
            admit_preparation(media_request, backend.capabilities()).unwrap_err(),
            PreparationAdmissionError::BackendInputModality("image")
        );
        assert_eq!(backend.native_allocations.get(), 0);
        assert_eq!(backend.payload_reads.get(), 0);
        let _ = IndependentBackend::realize;
    }
}
