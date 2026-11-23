use eredu_core::{
    backend::TokenOutput,
    eager_completion::EagerCompletion,
    preparation::{ReplicatedTextMechanismSupport, StateComponentPlacement},
    BackendDescriptor, BackendFailure, BackendFailureKind, BackendProvider, DeviceCapabilities,
    DeviceDescriptor, SessionAdmission, SessionCapabilities, SubmissionLease,
    TextGenerationBackend,
};
use eredu_nn::Tensor;

use crate::{CandleContext, CandleNeuralBackend, CandlePlacement, CandleTensor};

/// A committed token produced by local execution.
#[derive(Clone, Copy, Debug)]
pub struct CandleToken(pub(crate) u32);

impl TokenOutput for CandleToken {
    type Error = BackendFailure;

    fn token_id(&self) -> Result<u32, BackendFailure> {
        Ok(self.0)
    }
}

impl BackendProvider for CandleContext {
    type Error = BackendFailure;

    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("candle", env!("CARGO_PKG_VERSION"))
    }

    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, BackendFailure> {
        let placement = self.placement();
        let name = match placement {
            CandlePlacement::Cpu => "CPU",
            CandlePlacement::Metal(_) => "Metal",
            CandlePlacement::Cuda(_) => "CUDA",
        };
        Ok(vec![(
            DeviceDescriptor::new(
                placement.device_plan().device(),
                name,
                std::env::consts::ARCH,
                None,
            ),
            DeviceCapabilities::new(true, false, false),
        )])
    }

    fn session_capability_mismatch(
        &self,
        admitted: SessionCapabilities,
        realized: SessionCapabilities,
    ) -> BackendFailure {
        let mismatch = SessionAdmission::new(admitted)
            .validate(realized)
            .expect_err("mismatch requires different session reports");
        BackendFailure::new(BackendFailureKind::InvalidSession, mismatch)
    }
}

impl TextGenerationBackend for CandleContext {
    type InputTensor = CandleTensor;
    type Token = CandleToken;
    type TextCompletion = EagerCompletion<BackendFailure, SubmissionLease>;
}

impl CandleContext {
    pub(crate) fn tokens(&self, ids: &[u32]) -> Result<CandleTensor, BackendFailure> {
        let length = i32::try_from(ids.len()).map_err(BackendFailure::from_error)?;
        let ids = ids
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(BackendFailure::from_error)?;
        CandleTensor::from_i32_slice(&ids, &[1, length], self).map_err(BackendFailure::from_error)
    }
}

pub(crate) fn invalid(message: &'static str) -> BackendFailure {
    BackendFailure::new(
        BackendFailureKind::InvalidInput,
        std::io::Error::other(message),
    )
}

/// Side-effect-free mechanism facts for an exact Candle placement.
pub struct CandlePreparation(CandlePlacement, crate::rows::CandleRowLookupSupport);

impl CandlePreparation {
    pub(crate) fn new(placement: CandlePlacement) -> Self {
        Self(
            placement,
            crate::rows::CandleRowLookupSupport::new(placement),
        )
    }

    pub(crate) fn media_features(&self) -> eredu_core::MediaFeatureAvailability {
        eredu_core::MediaFeatureAvailability {
            image: self.0.supports_local_text_extensions() && cfg!(feature = "image"),
            audio: self.0.supports_local_text_extensions() && cfg!(feature = "audio"),
        }
    }

    pub(crate) fn operators(&self) -> eredu_nn::NeuralOperatorCapabilities {
        use eredu_nn::NeuralBackend;
        if self.0.is_cpu() {
            CandleNeuralBackend::OPERATOR_CAPABILITIES
        } else if self.0.is_metal() {
            CandleNeuralBackend::LOCAL_OPERATORS
        } else {
            CandleNeuralBackend::ACCELERATOR_OPERATORS
        }
    }

    fn supports_state_placement(placement: StateComponentPlacement) -> bool {
        matches!(
            placement,
            StateComponentPlacement::Device | StateComponentPlacement::Paged
        )
    }

    fn supports_state_dtype(&self, dtype: eredu_core::preparation::StateStorageDtype) -> bool {
        use eredu_core::preparation::StateStorageDtype as D;
        match dtype {
            D::F16 | D::Bf16 | D::F32 | D::I32 | D::U32 => true,
            D::F64 => self.0.is_cpu(),
            _ => false,
        }
    }
}

impl CandlePreparation {
    fn grouped_operations(&self) -> &'static [eredu_nn::GroupedOperationRequirement] {
        use eredu_nn::GroupedOperationRequirement::{GatedProduct, Linear, Relu2};
        if self.0.supports_local_text_extensions() {
            &[GatedProduct, Linear, Relu2]
        } else {
            &[GatedProduct]
        }
    }
}

impl ReplicatedTextMechanismSupport for CandlePreparation {
    fn facts(
        &self,
        placement: eredu_core::preparation::StateComponentPlacement,
    ) -> eredu_core::preparation::BackendMechanismFacts {
        use eredu_core::preparation::{
            rows::NativeRowLookupSupport, BackendMechanismFacts, StateLifecycleCapabilities,
            WeightResidencyMechanism,
        };
        let local = Self::supports_state_placement(placement);
        let residency = [
            WeightResidencyMechanism::Resident,
            WeightResidencyMechanism::Windowed,
            WeightResidencyMechanism::DiskStreamed,
        ];
        BackendMechanismFacts::new(
            self.operators(),
            residency,
            StateLifecycleCapabilities::new()
                .with_transactions(local, local)
                .with_reset(local)
                .with_prompt_cache(local)
                .with_observation_retention(local),
        )
        .with_grouped_operations(self.grouped_operations().iter().copied())
        .with_session(eredu_core::SessionCapabilities::new(false, true, true))
        .with_exact_completion(true)
        .with_indexed_movement(self.0.supports_local_text_extensions())
        .with_addressable_storage(self.1.row_storage())
        .with_prompt_cache(local)
        .with_chunked_prefill(true)
    }

    fn supports_direct(
        &self,
        descriptor: &eredu_checkpoint::lowering::WeightLoweringDescriptor,
    ) -> bool {
        use eredu_checkpoint::{Endian, GgmlType, LinearFormat, SourceTensorEncoding, StoredDtype};
        let supported = match (descriptor.source(), descriptor.executable()) {
            (
                SourceTensorEncoding::Safetensors(StoredDtype::F64)
                | SourceTensorEncoding::RecipeOutput(StoredDtype::F64),
                LinearFormat::Dense,
            ) => self.0.is_cpu(),
            (
                SourceTensorEncoding::Safetensors(
                    StoredDtype::F32
                    | StoredDtype::U8
                    | StoredDtype::F8E8M0
                    | StoredDtype::F16
                    | StoredDtype::BF16
                    | StoredDtype::I32
                    | StoredDtype::I64,
                )
                | SourceTensorEncoding::RecipeOutput(
                    StoredDtype::F32
                    | StoredDtype::U8
                    | StoredDtype::F8E8M0
                    | StoredDtype::F16
                    | StoredDtype::BF16
                    | StoredDtype::I32
                    | StoredDtype::I64,
                ),
                LinearFormat::Dense,
            ) => true,
            (
                SourceTensorEncoding::Safetensors(StoredDtype::U32)
                | SourceTensorEncoding::RecipeOutput(StoredDtype::U32),
                LinearFormat::Affine(_) | LinearFormat::MxFp4,
            ) => self
                .0
                .supports_word_matrix(descriptor.logical_shape().len()),
            (
                SourceTensorEncoding::Safetensors(StoredDtype::F4)
                | SourceTensorEncoding::RecipeOutput(StoredDtype::F4),
                LinearFormat::MxFp4,
            ) => self
                .0
                .supports_word_matrix(descriptor.logical_shape().len()),
            (
                SourceTensorEncoding::Safetensors(StoredDtype::U32)
                | SourceTensorEncoding::RecipeOutput(StoredDtype::U32),
                LinearFormat::NvFp4,
            )
            | (
                SourceTensorEncoding::Safetensors(StoredDtype::F8E4M3 | StoredDtype::U8)
                | SourceTensorEncoding::RecipeOutput(StoredDtype::F8E4M3 | StoredDtype::U8),
                LinearFormat::E4M3BlockFp8(_),
            ) => {
                self.0.supports_local_text_extensions()
                    && matches!(descriptor.logical_shape().len(), 2 | 3)
            }
            (
                SourceTensorEncoding::Gguf {
                    ggml_type: GgmlType::MxFp4,
                    ..
                },
                LinearFormat::MxFp4,
            ) => self
                .0
                .supports_word_matrix(descriptor.logical_shape().len()),
            (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Affine(format)) => {
                self.0
                    .supports_word_matrix(descriptor.logical_shape().len())
                    && eredu_gguf::affine_layout(*ggml_type).is_some_and(|(bits, group)| {
                        i32::from(bits) == format.bits && group as i32 == format.group_size
                    })
            }
            (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Dense) => {
                matches!(
                    ggml_type,
                    GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 | GgmlType::I32
                ) || (self.0.supports_local_text_extensions()
                    && (*ggml_type == GgmlType::MxFp4
                        || eredu_gguf::affine_layout(*ggml_type).is_some()))
            }
            (
                SourceTensorEncoding::Gguf { ggml_type, endian },
                LinearFormat::GgufIQuant {
                    ggml_type: execution_type,
                    endian: execution_endian,
                },
            ) => {
                crate::quantized::supports_packed_matrix(
                    self.0,
                    *ggml_type,
                    descriptor.logical_shape().len(),
                ) && *ggml_type == execution_type
                    && *endian == Endian::Little
                    && *endian == execution_endian
                    && descriptor.physical_shape() == descriptor.logical_shape()
            }
            (
                SourceTensorEncoding::RecipeOutput(StoredDtype::U8),
                LinearFormat::GgufIQuant {
                    ggml_type,
                    endian: Endian::Little,
                },
            ) => {
                crate::quantized::supports_packed_matrix(
                    self.0,
                    ggml_type,
                    descriptor.logical_shape().len(),
                ) && matches!(descriptor.logical_shape(), [_, width] | [_, _, width]
                        if ggml_type.block_and_bytes().is_ok_and(|(block, _)| width.is_multiple_of(block as usize)))
            }
            _ => false,
        };
        cfg!(target_endian = "little") && supported && descriptor.has_valid_direct_geometry()
    }

    fn prefers_original_encoding(
        &self,
        descriptor: &eredu_checkpoint::lowering::WeightLoweringDescriptor,
    ) -> bool {
        matches!(
            descriptor.executable(),
            eredu_checkpoint::LinearFormat::GgufIQuant { .. }
        ) && self.supports_direct(descriptor)
    }

    fn supports_transform(
        &self,
        descriptor: &eredu_checkpoint::lowering::WeightLoweringDescriptor,
    ) -> bool {
        self.0.supports_local_text_extensions()
            && matches!(descriptor.logical_shape().len(), 2 | 3)
            && cfg!(target_endian = "little")
            && descriptor.supports_word_transform()
            && descriptor
                .packed_extent()
                .is_some_and(|width| width <= crate::bounded_quantization::LIMITS.elements.get())
    }

    fn floating_state_dtype(
        &self,
        source: &eredu_core::checkpoint::TensorDtype,
    ) -> Option<eredu_core::preparation::StateStorageDtype> {
        use eredu_core::{checkpoint::TensorDtype, preparation::StateStorageDtype};
        StateStorageDtype::from_floating_source(source, |packed| match packed {
            TensorDtype::U32 => self.0.supports_word_matrix(2),
            TensorDtype::Encoded(encoding) => {
                crate::quantized::supports_activation_for_encoding(self.0, encoding)
            }
            _ => false,
        })
        .filter(|&dtype| self.supports_state_dtype(dtype))
    }

    fn supports_state_component(
        &self,
        shape: &[eredu_core::cache::StateTensorDimension],
        dtype: eredu_core::preparation::StateStorageDtype,
        placement: eredu_core::preparation::StateComponentPlacement,
    ) -> bool {
        use eredu_core::cache::StateTensorDimension;
        Self::supports_state_placement(placement)
            && self.supports_state_dtype(dtype)
            && shape.iter().all(|dimension| match dimension {
                StateTensorDimension::Fixed(size) => i32::try_from(size.get()).is_ok(),
                _ => true,
            })
    }
}

impl eredu_core::ModelLoadingBackend for CandleContext {
    type LoadPlacement = std::convert::Infallible;
    type Binding = CandlePlacement;
    type LoadMechanisms = CandlePreparation;

    fn load_mechanisms(&self) -> CandlePreparation {
        CandlePreparation::new(self.placement())
    }

    fn bind_load_placement(
        &self,
        rank: Option<eredu_core::ParallelRankTopology>,
        _: Option<Self::LoadPlacement>,
    ) -> Result<CandlePlacement, BackendFailure> {
        if rank.is_some() {
            return Err(BackendFailure::new(
                BackendFailureKind::Unsupported,
                std::io::Error::other("Candle does not realize distributed placement"),
            ));
        }
        self.check_healthy()?;
        Ok(self.placement())
    }
}

impl eredu_core::preparation::NativePreparation for CandlePreparation {
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        let media = self.media_features();
        let word_conversion = self.0.supports_local_text_extensions();
        eredu_core::PreparationMechanismCapabilities::new(true, true)
            .with_safetensors_quantization(word_conversion, word_conversion)
            .with_gguf_quantized_loading(self.0.is_cpu() || self.0.is_metal())
            .with_input_modalities(eredu_core::InputModalities {
                image: media.image,
                video: media.image,
                audio: media.audio,
                ..eredu_core::InputModalities::TEXT
            })
            .with_replicated_text_facts(&self.facts(StateComponentPlacement::Device))
    }

    fn row_lookup_support(
        &self,
    ) -> Option<&dyn eredu_core::preparation::rows::NativeRowLookupSupport> {
        Some(&self.1)
    }

    fn supports_grouped_operation(
        &self,
        requirement: eredu_nn::GroupedOperationRequirement,
    ) -> bool {
        self.grouped_operations().contains(&requirement)
    }

    fn replicated_text_support(
        &self,
    ) -> Option<&dyn eredu_core::preparation::ReplicatedTextMechanismSupport> {
        Some(self)
    }

    fn observation_mechanisms(&self) -> eredu_core::ObservationMechanisms {
        eredu_core::ObservationMechanisms {
            activation_tensors: self.0.supports_local_text_extensions(),
            routing_tensors: self.0.supports_local_text_extensions(),
            routed_unit_tensors: self.0.supports_local_text_extensions(),
            floating_to_f32: self.0.supports_local_text_extensions(),
        }
    }

    fn capture_capabilities(&self) -> eredu_core::capture::CaptureCapabilities {
        if self.0.supports_local_text_extensions() {
            crate::capture::capabilities()
        } else {
            Default::default()
        }
    }

    fn processor_capabilities(&self) -> eredu_core::media::MediaPrimitiveCapabilities {
        use eredu_core::{
            media::{MediaPrimitiveCapabilities, ProcessorPrimitive},
            InputModality,
        };
        let mut media = Vec::new();
        let mut primitives = vec![
            ProcessorPrimitive::TensorU32,
            ProcessorPrimitive::TensorF32,
            ProcessorPrimitive::TensorI32,
            ProcessorPrimitive::MetadataInspection,
        ];
        let features = self.media_features();
        if features.image {
            media.extend([InputModality::Image, InputModality::Video]);
            primitives.extend([
                ProcessorPrimitive::RgbResizeBicubic,
                ProcessorPrimitive::RgbResizeLanczos3,
                ProcessorPrimitive::RgbNormalize,
                ProcessorPrimitive::VideoSampling,
            ]);
        }
        if features.audio {
            media.push(InputModality::Audio);
            primitives.extend([
                ProcessorPrimitive::TensorBool,
                ProcessorPrimitive::AudioWindow,
                ProcessorPrimitive::AudioSpectrum,
                ProcessorPrimitive::AudioMelFilter,
                ProcessorPrimitive::AudioLogarithm,
            ]);
        }
        if media.is_empty() {
            return MediaPrimitiveCapabilities::new([], [], [], [], 0);
        }
        let prepared = media
            .iter()
            .copied()
            .chain([InputModality::Text])
            .collect::<Vec<_>>();
        MediaPrimitiveCapabilities::new(
            media,
            prepared.clone(),
            prepared,
            primitives,
            i32::MAX as u64,
        )
    }

    fn speculative_capabilities(
        &self,
    ) -> eredu_core::speculative::SpeculativeMechanismCapabilities {
        use eredu_core::speculative::SpeculativeMechanism as M;
        let mechanisms = [
            M::TensorOperations,
            M::NeuralOperations,
            M::PayloadMaterialization,
            M::LogitsProcessing,
            M::Sampling,
            M::Randomness,
            M::StateStorage,
            M::StorageResidency,
            M::ExactCompletion,
            M::Observation,
            M::QueueBinding,
            M::SameDeviceHandoff,
            M::CrossDeviceTransfer,
            M::Agreement,
            M::Publication,
        ];
        eredu_core::speculative::SpeculativeMechanismCapabilities::new(
            mechanisms
                .into_iter()
                .filter(|_| self.0.supports_local_text_extensions()),
        )
    }

    fn communication_capabilities(&self) -> eredu_core::communication::CommunicationCapabilities {
        eredu_core::communication::CommunicationCapabilities::new([])
            .expect("empty communication capabilities")
    }
}
