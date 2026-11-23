use candle_core::{DType, Tensor as NativeTensor};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeDtype},
    store::{CheckpointLease, CheckpointSource, EncodedTensorLease, StoreError, TensorSelection},
    Endian, GgmlType, SourceTensorEncoding,
};
use eredu_core::{
    checkpoint_mechanisms::{
        bounded_quantization::BoundedQuantizationBackend,
        recipe_execution::{self, RecipeExecution},
        CheckpointParameterBackend, ExactParameterMaterializer, ParameterBindingTarget,
    },
    BackendFailure, BackendFailureKind,
};
use eredu_nn::Error;

use crate::{CandleContext, CandleNeuralBackend, CandleTensor};

pub(crate) fn native_dtype(dtype: &RecipeDtype) -> Result<DType, Error> {
    match dtype {
        RecipeDtype::F16 => Ok(DType::F16),
        RecipeDtype::BF16 => Ok(DType::BF16),
        RecipeDtype::F32 => Ok(DType::F32),
        RecipeDtype::F64 => Ok(DType::F64),
        RecipeDtype::I16 => Ok(DType::I16),
        RecipeDtype::I32 => Ok(DType::I32),
        RecipeDtype::I64 => Ok(DType::I64),
        RecipeDtype::U8 | RecipeDtype::F8E4M3 | RecipeDtype::F8E8M0 | RecipeDtype::F4 => {
            Ok(DType::U8)
        }
        RecipeDtype::U32 => Ok(DType::U32),
        _ => Err(Error::backend(format!(
            "unsupported checkpoint scalar type {dtype:?}"
        ))),
    }
}

fn recipe_dtype(dtype: DType) -> Option<RecipeDtype> {
    Some(match dtype {
        DType::F16 => RecipeDtype::F16,
        DType::BF16 => RecipeDtype::BF16,
        DType::F32 => RecipeDtype::F32,
        DType::F64 => RecipeDtype::F64,
        DType::I16 => RecipeDtype::I16,
        DType::I32 => RecipeDtype::I32,
        DType::I64 => RecipeDtype::I64,
        DType::U8 => RecipeDtype::U8,
        DType::U32 => RecipeDtype::U32,
        _ => return None,
    })
}

fn check_recipe_shape(shape: &[usize]) -> Result<(), Error> {
    for &dim in shape {
        i32::try_from(dim).map_err(Error::backend_source)?;
    }
    Ok(())
}

pub(crate) fn import_bytes(
    bytes: &[u8],
    dtype: &RecipeDtype,
    shape: &[usize],
    context: &CandleContext,
) -> Result<CandleTensor, Error> {
    if cfg!(target_endian = "big") {
        return Err(Error::backend(
            "checkpoint byte import requires a little-endian host",
        ));
    }
    let packed;
    let shape = if *dtype == RecipeDtype::F4 {
        let mut dimensions = shape.to_vec();
        let last = dimensions
            .last_mut()
            .filter(|width| **width > 0 && width.is_multiple_of(2))
            .ok_or_else(|| Error::backend("F4 source rows require full bytes"))?;
        *last /= 2;
        packed = dimensions;
        packed.as_slice()
    } else {
        shape
    };
    let dtype = native_dtype(dtype)?;
    check_recipe_shape(shape)?;
    let expected = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |length, &dim| {
            length.checked_mul(dim)
        });
    if expected != Some(bytes.len()) {
        return Err(Error::backend(
            "checkpoint byte extent differs from its shape and dtype",
        ));
    }
    CandleTensor::wrap(NativeTensor::from_raw_buffer(
        bytes,
        dtype,
        shape,
        &context.device,
    ))
}

fn validate_recipe_node(
    recipe: &DerivedWeightRecipe,
    encoding: &impl Fn(&str) -> Result<SourceTensorEncoding, StoreError>,
) -> Result<(), Error> {
    match recipe {
        DerivedWeightRecipe::Source { key, .. } => {
            match encoding(key).map_err(Error::backend_source)? {
                SourceTensorEncoding::Safetensors(dtype) => native_dtype(&dtype.into()).map(|_| ()),
                SourceTensorEncoding::Gguf {
                    ggml_type: GgmlType::F16 | GgmlType::Bf16 | GgmlType::F32 | GgmlType::I32,
                    ..
                } => Ok(()),
                SourceTensorEncoding::Gguf { ggml_type, endian }
                    if eredu_gguf::BlockDecoder::new(ggml_type, endian).is_ok() =>
                {
                    Ok(())
                }
                _ => Err(Error::backend(
                    "checkpoint source requires an unsupported encoding",
                )),
            }
        }
        DerivedWeightRecipe::Reshape { .. }
        | DerivedWeightRecipe::Transpose { .. }
        | DerivedWeightRecipe::SubtractOne { .. }
        | DerivedWeightRecipe::Concatenate { .. }
        | DerivedWeightRecipe::Stack { .. }
        | DerivedWeightRecipe::Select { .. }
        | DerivedWeightRecipe::NegLog { .. }
        | DerivedWeightRecipe::DecodeTensorFp8 { .. }
        | DerivedWeightRecipe::ScaleF32 { .. } => Ok(()),
        DerivedWeightRecipe::Cast { dtype, .. }
            if !matches!(dtype, RecipeDtype::F8E4M3 | RecipeDtype::F8E8M0) =>
        {
            native_dtype(dtype).map(|_| ())
        }
        DerivedWeightRecipe::View {
            dtype: RecipeDtype::U8 | RecipeDtype::U32,
            ..
        } => Ok(()),
        _ => Err(Error::backend(
            "checkpoint recipe requires an unsupported transformation",
        )),
    }
}

pub(crate) fn preflight_recipe_catalog(
    recipe: &DerivedWeightRecipe,
    catalog: &(impl RecipeCatalog + ?Sized),
    encoding: &impl Fn(&str) -> Result<SourceTensorEncoding, StoreError>,
) -> Result<(), Error> {
    recipe_execution::preflight(
        recipe,
        catalog,
        &|node, output| {
            validate_recipe_node(node, encoding)?;
            // Native byte reinterprets have an exact input and output storage type.
            if let DerivedWeightRecipe::View { input, .. } = node {
                let input = input.infer(catalog).map_err(Error::backend_source)?;
                if !matches!(native_dtype(input.dtype())?, DType::U8 | DType::U32) {
                    return Err(Error::backend(
                        "byte reinterpretation requires U8 or U32 storage",
                    ));
                }
            }
            if let DerivedWeightRecipe::Cast { input, .. } = node {
                let input = input.infer(catalog).map_err(Error::backend_source)?;
                if matches!(
                    input.dtype(),
                    RecipeDtype::F4 | RecipeDtype::F8E4M3 | RecipeDtype::F8E8M0
                ) {
                    return Err(Error::backend(
                        "encoded FP8 values require an exact decoder",
                    ));
                }
            }
            if output.dtype() == &RecipeDtype::F4
                && (!matches!(node, DerivedWeightRecipe::Source { .. })
                    || !output
                        .shape()
                        .last()
                        .is_some_and(|width| width.is_multiple_of(2)))
            {
                return Err(Error::backend(
                    "F4 requires a byte-aligned source followed by word lowering",
                ));
            }
            if matches!(node, DerivedWeightRecipe::NegLog { .. })
                && !matches!(
                    output.dtype(),
                    RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32 | RecipeDtype::F64
                )
            {
                return Err(Error::backend(
                    "negative logarithm requires floating checkpoint values",
                ));
            }
            native_dtype(output.dtype())?;
            check_recipe_shape(output.shape())
        },
        &Error::backend_source,
        &|_, validate| validate(),
    )
}

pub(crate) fn decode_scaled_e4m3(
    weight: &NativeTensor,
    scale: &NativeTensor,
) -> Result<NativeTensor, Error> {
    let scale = scale
        .to_dtype(DType::F32)
        .and_then(|scale| scale.reshape(()))
        .and_then(|scale| scale.to_scalar::<f32>())
        .map_err(Error::backend_source)?;
    if !scale.is_finite() || scale <= 0. {
        return Err(Error::backend(
            "checkpoint scale must be positive and finite",
        ));
    }
    let bytes = weight
        .flatten_all()
        .and_then(|data| data.to_vec1::<u8>())
        .map_err(Error::backend_source)?;
    let decoded: Vec<_> = bytes
        .into_iter()
        .map(|byte| eredu_checkpoint::fp8::decode_e4m3(byte) * scale)
        .collect();
    NativeTensor::from_vec(decoded, weight.shape(), weight.device()).map_err(Error::backend_source)
}

fn reinterpret_bytes(
    input: &NativeTensor,
    dtype: &RecipeDtype,
    shape: &[usize],
    context: &CandleContext,
) -> Result<CandleTensor, Error> {
    let output = native_dtype(dtype)?;
    if input.dtype() == output {
        return CandleTensor::wrap(input.reshape(shape));
    }
    let flat = input.flatten_all().map_err(Error::backend_source)?;
    let bytes = match input.dtype() {
        DType::U8 => flat.to_vec1::<u8>().map_err(Error::backend_source)?,
        DType::U32 => flat
            .to_vec1::<u32>()
            .map_err(Error::backend_source)?
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect(),
        _ => {
            return Err(Error::backend(
                "byte reinterpretation requires U8 or U32 storage",
            ))
        }
    };
    import_bytes(&bytes, dtype, shape, context)
}

impl RecipeExecution for CandleContext {
    type Value = CandleTensor;
    type Retained = std::convert::Infallible;
    type Error = Error;

    fn import(
        &self,
        lease: CheckpointLease,
        _: &mut Vec<Self::Retained>,
    ) -> Result<CandleTensor, Error> {
        // Recipe operators must address the actual bytes. CUDA's direct
        // QTensor binding exposes only a shape placeholder as a dense tensor.
        match lease {
            CheckpointLease::Gguf(lease) => materialize_gguf(&lease, self, false),
            lease => CandleNeuralBackend::materialize(lease, self),
        }
    }

    fn apply(
        &self,
        recipe: &DerivedWeightRecipe,
        operands: Vec<CandleTensor>,
        _: &mut Vec<Self::Retained>,
    ) -> Result<CandleTensor, Error> {
        let unary = || {
            let [input] = operands.as_slice() else {
                return Err(Error::backend("expected one recipe operand"));
            };
            Ok(&input.value)
        };
        match recipe {
            DerivedWeightRecipe::Reshape { shape, .. } => {
                CandleTensor::wrap(unary()?.reshape(shape.as_slice()))
            }
            DerivedWeightRecipe::Transpose { axes, .. } => {
                CandleTensor::wrap(unary()?.permute(axes.clone()))
            }
            DerivedWeightRecipe::Concatenate { axis, .. } => {
                let tensors: Vec<_> = operands.iter().map(|tensor| &tensor.value).collect();
                CandleTensor::wrap(NativeTensor::cat(&tensors, *axis))
            }
            DerivedWeightRecipe::Stack { axis, .. } => {
                let tensors: Vec<_> = operands.iter().map(|tensor| &tensor.value).collect();
                CandleTensor::wrap(NativeTensor::stack(&tensors, *axis))
            }
            DerivedWeightRecipe::Select { selection, .. } => {
                let input = unary()?;
                CandleTensor::wrap(match selection {
                    TensorSelection::Full => Ok(input.clone()),
                    TensorSelection::Range { axis, start, end } => {
                        input.narrow(*axis, *start, end - start)
                    }
                    TensorSelection::Indices { axis, indices } => {
                        let indices = indices
                            .iter()
                            .copied()
                            .map(u32::try_from)
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(Error::backend_source)?;
                        NativeTensor::new(indices, &self.device)
                            .and_then(|indices| input.index_select(&indices, *axis))
                    }
                    TensorSelection::Contiguous {
                        offset_elements,
                        shape,
                    } => input
                        .flatten_all()
                        .and_then(|input| {
                            input.narrow(0, *offset_elements, shape.iter().product::<usize>())
                        })
                        .and_then(|input| input.reshape(shape.as_slice())),
                })
            }
            DerivedWeightRecipe::View { dtype, shape, .. } => {
                reinterpret_bytes(unary()?, dtype, shape, self)
            }
            DerivedWeightRecipe::NegLog { .. } => {
                let input = unary()?;
                CandleTensor::wrap((|| {
                    let wide = if input.dtype() == DType::F64 {
                        input.clone()
                    } else {
                        input.to_dtype(DType::F32)?
                    };
                    if wide.ge(0.)?.max_all()?.to_scalar::<u8>()? != 0
                        || wide.ne(&wide)?.max_all()?.to_scalar::<u8>()? != 0
                    {
                        candle_core::bail!("negative logarithm recipe requires negative input");
                    }
                    wide.neg()?.log()?.to_dtype(input.dtype())
                })())
            }
            DerivedWeightRecipe::DecodeTensorFp8 { .. } | DerivedWeightRecipe::ScaleF32 { .. } => {
                let [weight, scale] = operands.as_slice() else {
                    return Err(Error::backend("expected weight and scalar recipe operands"));
                };
                if matches!(recipe, DerivedWeightRecipe::DecodeTensorFp8 { .. }) {
                    return CandleTensor::wrap(
                        decode_scaled_e4m3(&weight.value, &scale.value)
                            .map_err(candle_core::Error::wrap),
                    );
                }
                CandleTensor::wrap((|| {
                    let scalar = scale.value.reshape(())?.to_scalar::<f32>()?;
                    if !scalar.is_finite() || scalar <= 0. {
                        candle_core::bail!("checkpoint scale must be positive and finite");
                    }
                    weight.value.affine(f64::from(scalar), 0.)
                })())
            }
            DerivedWeightRecipe::Cast { dtype, .. } => {
                CandleTensor::wrap(unary()?.to_dtype(native_dtype(dtype)?))
            }
            DerivedWeightRecipe::SubtractOne { .. } => CandleTensor::wrap(unary()?.affine(1., -1.)),
            _ => Err(Error::backend(
                "checkpoint recipe requires an unsupported transformation",
            )),
        }
    }

    fn store_error(&self, error: StoreError) -> Error {
        Error::backend_source(error)
    }
}

impl CheckpointParameterBackend for CandleNeuralBackend {
    type Parameter = CandleTensor;
    type MaterializedWeight = CandleTensor;
    type MaterializationContext = CandleContext;
    type Materialization = CandleTensor;
    type ParameterError = Error;

    fn preflight_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<(), Error> {
        preflight_recipe_catalog(recipe, source, &|key| {
            source
                .source_provenance(key)
                .map(|source| source.source_encoding)
        })
    }

    fn materialize(lease: CheckpointLease, context: &CandleContext) -> Result<CandleTensor, Error> {
        if let CheckpointLease::Gguf(lease) = lease {
            return materialize_gguf(&lease, context, true);
        }
        let bytes = lease.encoded_bytes().ok_or_else(|| {
            Error::backend("checkpoint lease did not expose an encoded byte span")
        })?;
        import_bytes(
            bytes,
            &lease.metadata().stored_dtype.clone().into(),
            lease.output_shape(),
            context,
        )
    }

    fn materialize_recipe(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
        context: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        Self::preflight_recipe(recipe, source)?;
        recipe_execution::execute(recipe, source, context, &mut Vec::new())
    }

    fn materialized_weight(materialization: &CandleTensor) -> &CandleTensor {
        materialization
    }

    fn finish_materialization(materialization: CandleTensor) -> Result<CandleTensor, Error> {
        Ok(materialization)
    }

    fn share_materialized_weight(weight: &CandleTensor) -> Result<CandleTensor, Error> {
        Ok(weight.clone())
    }

    fn validate_bind(parameter: &CandleTensor, weight: &CandleTensor) -> Result<(), Error> {
        let target = Self::parameter_target(parameter)
            .ok_or_else(|| Error::backend("unsupported destination parameter representation"))?;
        let dtype = recipe_dtype(weight.value.dtype())
            .ok_or_else(|| Error::backend("unsupported materialized parameter representation"))?;
        if target.shape != weight.value.dims()
            || (target.dtype != dtype && !target.permitted_source_dtypes.contains(&dtype))
            || !parameter.value.device().same_device(weight.value.device())
            || !crate::quantized::matches_binding(parameter, weight)
        {
            return Err(Error::backend(
                "materialized weight differs from its binding target",
            ));
        }
        Ok(())
    }

    fn bind(parameter: &mut CandleTensor, weight: CandleTensor) {
        *parameter = weight;
    }
}

fn materialize_gguf(
    lease: &eredu_checkpoint::gguf_store::GgufLease,
    context: &CandleContext,
    allow_cuda_qtensor: bool,
) -> Result<CandleTensor, Error> {
    use eredu_gguf::{ConvertedTensor, DenseDtype};
    if !lease.selection_is_materialized() {
        return Err(Error::backend(
            "GGUF read did not realize the selected tensor view",
        ));
    }
    let (_, names, converted) = lease
        .materialize_portable()
        .map_err(Error::backend_source)?
        .into_parts();
    let output = names
        .iter()
        .position(|name| name == lease.logical_output_name())
        .ok_or_else(|| Error::backend("GGUF conversion lacks the leased output"))?;
    let (shape, dtype, data, format) = match converted {
        ConvertedTensor::Dense(value) => {
            let dtype = match value.dtype {
                DenseDtype::F32 => RecipeDtype::F32,
                DenseDtype::F16 => RecipeDtype::F16,
                DenseDtype::Bf16 => RecipeDtype::BF16,
                DenseDtype::I32 => RecipeDtype::I32,
                _ => return Err(Error::backend("unsupported dense GGUF dtype")),
            };
            (value.shape, dtype, value.data, None)
        }
        ConvertedTensor::IQuant(value)
            if eredu_gguf::BlockDecoder::new(value.ggml_type, value.endian).is_ok() =>
        {
            (
                value.packed_shape().map_err(Error::backend_source)?,
                RecipeDtype::U8,
                value.data,
                (value.endian == Endian::Little).then_some(value.ggml_type),
            )
        }
        ConvertedTensor::MxFp4(value) => match output {
            0 => (
                value.weight_shape,
                RecipeDtype::U32,
                value
                    .weights
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
                None,
            ),
            1 => (value.scale_shape, RecipeDtype::U8, value.scales, None),
            _ => return Err(Error::backend("invalid MXFP4 companion identity")),
        },
        ConvertedTensor::Affine(value) => match output {
            0 => (
                value.weight_shape,
                RecipeDtype::U32,
                value
                    .weights
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect(),
                None,
            ),
            1 => (
                value.scale_shape,
                RecipeDtype::F16,
                value
                    .scales
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                None,
            ),
            2 => (
                value.scale_shape,
                RecipeDtype::F16,
                value
                    .biases
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                None,
            ),
            _ => return Err(Error::backend("invalid affine companion identity")),
        },
        _ => return Err(Error::backend("unsupported native GGUF representation")),
    };
    let shape = shape
        .into_iter()
        .map(usize::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Error::backend_source)?;
    if shape != lease.output_shape() || dtype != lease.metadata().stored_dtype.clone().into() {
        return Err(Error::backend(
            "GGUF parameter differs from its admitted shape or dtype",
        ));
    }
    if let Some(format) = format.filter(|_| {
        shape.len() == 2
            && !context.device.is_metal()
            && (allow_cuda_qtensor || context.device.is_cpu())
    }) {
        return crate::quantized::import_matrix_bytes(&data, &shape, format, &context.device);
    }
    import_bytes(&data, &dtype, &shape, context)
}

impl ExactParameterMaterializer<Self> for CandleNeuralBackend {
    fn parameter_target(parameter: &CandleTensor) -> Option<ParameterBindingTarget> {
        let dtype = recipe_dtype(parameter.value.dtype())?;
        let mut permitted_source_dtypes = if dtype == RecipeDtype::F32 {
            vec![RecipeDtype::F16, RecipeDtype::BF16]
        } else if dtype == RecipeDtype::U8 {
            vec![RecipeDtype::F8E8M0, RecipeDtype::F8E4M3]
        } else {
            Vec::new()
        };
        if dtype == RecipeDtype::F32 && parameter.value.device().is_cpu() {
            permitted_source_dtypes.push(RecipeDtype::F64);
        }
        Some(ParameterBindingTarget {
            shape: parameter.value.dims().to_vec(),
            dtype,
            permitted_source_dtypes,
        })
    }

    fn lower_mxfp4(
        recipe: DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<DerivedWeightRecipe, BackendFailure> {
        recipe
            .lower_mxfp4_words(source)
            .map_err(BackendFailure::from_error)
    }

    fn quantizer(
        context: &CandleContext,
    ) -> Result<Box<dyn BoundedQuantizationBackend + '_>, BackendFailure> {
        if !context.placement().supports_local_text_extensions() {
            return Err(unsupported(
                "checkpoint quantization is unavailable on this placement",
            ));
        }
        Ok(Box::new(
            crate::bounded_quantization::CandleBoundedQuantizer::new(),
        ))
    }
}

fn unsupported(message: &'static str) -> BackendFailure {
    BackendFailure::new(BackendFailureKind::Unsupported, Error::backend(message))
}

/// Failure while reading an admitted parameter into a Candle tensor.
#[derive(Debug, thiserror::Error)]
pub enum ParameterReadError {
    /// Portable binding rejected the operation.
    #[error(transparent)]
    Binding(#[from] BackendFailure),
    /// The prepared source could not supply the selected bytes.
    #[error(transparent)]
    Store(#[from] eredu_checkpoint::store::StoreError),
    /// Tensor materialization failed.
    #[error(transparent)]
    Tensor(#[from] Error),
    /// Portable residency rejected the copy.
    #[error(transparent)]
    Residency(#[from] eredu_core::residency::ResidencyLedgerError),
    /// Conversion retention could not be observed or changed.
    #[error(transparent)]
    Retention(#[from] eredu_core::residency::conversion_claim::ConversionRetentionError),
    /// Resource description is invalid.
    #[error(transparent)]
    Resources(#[from] eredu_core::resources::ResourceDescriptionError),
}

impl eredu_core::checkpoint_mechanisms::ParameterReadScope for CandleContext {
    type DirectRead = eredu_checkpoint::recipe::EncodedRecipeRead;
    type Value = CandleTensor;
    type Pending = Vec<CandleTensor>;
    type Error = ParameterReadError;

    fn slots(&self) -> std::num::NonZeroUsize {
        std::num::NonZeroUsize::MIN
    }

    fn direct_read(
        &self,
        source: &dyn CheckpointSource,
        binding: &eredu_core::residency::parameter_binding::WeightBinding,
    ) -> Result<Option<Self::DirectRead>, ParameterReadError> {
        let direct;
        let recipe = match binding.recipe() {
            Some(recipe) => recipe,
            None => {
                direct = DerivedWeightRecipe::source(
                    binding.checkpoint_key(),
                    binding.selection().clone(),
                );
                &direct
            }
        };
        CandleNeuralBackend::preflight_recipe(recipe, source)?;
        let read = recipe
            .prepare_encoded_read(source)
            .map_err(Error::backend_source)?;
        if let Some(read) = &read {
            if read.output().byte_len != binding.expected_bytes() {
                return Err(Error::backend("direct read extent differs from its binding").into());
            }
        }
        Ok(read)
    }

    fn submit_direct(
        &mut self,
        reads: Vec<(Self::DirectRead, eredu_nn::ParameterStorageLayout)>,
    ) -> Result<Self::Pending, ParameterReadError> {
        // Candle imports consume these bytes into native-owned tensor storage.
        reads
            .into_iter()
            .map(|(read, _)| {
                let output = read.output().clone();
                let byte_len = usize::try_from(output.byte_len).map_err(Error::backend_source)?;
                let mut bytes = vec![0; byte_len];
                read.read_into(&mut bytes)?;
                import_bytes(&bytes, &output.dtype, &output.shape, self).map_err(Into::into)
            })
            .collect()
    }

    fn submit_ordinary(
        &mut self,
        _: &str,
        input: eredu_core::checkpoint_mechanisms::ParameterReadInput<'_>,
    ) -> Result<Self::Pending, ParameterReadError> {
        use eredu_core::checkpoint_mechanisms::ParameterReadInput;
        let tensor = match input {
            ParameterReadInput::Lease(lease) => CandleNeuralBackend::materialize(lease, self),
            ParameterReadInput::Recipe(recipe, source) => {
                CandleNeuralBackend::materialize_recipe(recipe, source, self)
            }
        }?;
        Ok(vec![tensor])
    }

    fn take_values(
        &mut self,
        pending: Self::Pending,
    ) -> Result<Vec<CandleTensor>, ParameterReadError> {
        Ok(pending)
    }

    fn checkpoint_capacity_exhausted(error: &ParameterReadError) -> bool {
        let mut cause: Option<&dyn std::error::Error> = Some(error);
        while let Some(error) = cause {
            if matches!(
                error.downcast_ref::<eredu_checkpoint::store::StoreError>(),
                Some(eredu_checkpoint::store::StoreError::CapacityExhausted { .. })
            ) {
                return true;
            }
            cause = error.source();
        }
        false
    }
}
