fn shape(values: &[usize]) -> Result<Vec<i32>, Error> {
    values
        .iter()
        .map(|&value| i32::try_from(value).map_err(Error::backend))
        .collect()
}

pub(super) fn decode(
    bytes: &[u8],
    dtype: &StoredDtype,
    dimensions: &[usize],
) -> Result<NumericTensor, Error> {
    let (width, values): (usize, Vec<f32>) = match dtype {
        StoredDtype::F32 => (
            4,
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect(),
        ),
        StoredDtype::F16 => (
            2,
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| half::f16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                .collect(),
        ),
        StoredDtype::BF16 => (
            2,
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                .collect(),
        ),
        StoredDtype::I32 => (
            4,
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| i32::from_le_bytes(*chunk) as f32)
                .collect(),
        ),
        StoredDtype::U8 => (1, bytes.iter().map(|&value| f32::from(value)).collect()),
        dtype => {
            return Err(Error::backend(format!(
                "numeric dense payload cannot decode {dtype:?}"
            )))
        }
    };
    let dimensions = shape(dimensions)?;
    if !bytes.len().is_multiple_of(width) || elements(&dimensions) != values.len() {
        return Err(Error::backend(
            "numeric payload length does not match the admitted shape",
        ));
    }
    Ok(NumericTensor::new(dimensions, values))
}

fn select(value: NumericTensor, selection: &TensorSelection) -> Result<NumericTensor, Error> {
    match selection {
        TensorSelection::Full => Ok(value),
        TensorSelection::Range { axis, start, end } => select_parameter(
            &value,
            &TensorPlacement::Range {
                axis: *axis,
                start: *start,
                end: *end,
            },
        ),
        TensorSelection::Indices { axis, indices } => select_parameter(
            &value,
            &TensorPlacement::Indices {
                axis: *axis,
                indices: indices.clone(),
            },
        ),
        TensorSelection::Contiguous {
            offset_elements,
            shape: dimensions,
        } => {
            let dimensions = shape(dimensions)?;
            let end = offset_elements
                .checked_add(elements(&dimensions))
                .ok_or_else(|| Error::backend("numeric payload selection overflowed"))?;
            let values = value
                .data
                .get(*offset_elements..end)
                .ok_or_else(|| Error::backend("numeric payload selection exceeds source"))?;
            Ok(NumericTensor::new(dimensions, values.to_vec()))
        }
    }
}

pub(super) fn recipe_value(
    recipe: &DerivedWeightRecipe,
    checkpoint: &dyn CheckpointSource,
    context: &NumericContext,
) -> Result<NumericTensor, Error> {
    match recipe {
        DerivedWeightRecipe::Source { key, selection } => {
            let lease = checkpoint
                .acquire_lease(TensorReadRequest {
                    key: key.clone(),
                    selection: selection.clone(),
                    policy: ReadPolicy::RequireBounded,
                })
                .map_err(Error::backend)?;
            record_reference_payload_reads(vec![ReferencePayloadRead {
                task: key.clone(),
                source: key.clone(),
                selection: selection.clone(),
                output_shape: lease.output_shape().to_vec(),
                encoded_bytes: lease.bounded_read_proof().length_bytes,
                physically_bounded: lease.bounded_read_proof().physically_bounded,
            }]);
            match &lease {
                eredu_checkpoint::store::CheckpointLease::Gguf(gguf) => {
                    let converted = gguf.materialize_portable().map_err(Error::backend)?;
                    match converted.converted() {
                        eredu_gguf::ConvertedTensor::Dense(tensor) => {
                            let dtype = match tensor.dtype {
                                eredu_gguf::DenseDtype::F32 => StoredDtype::F32,
                                eredu_gguf::DenseDtype::F16 => StoredDtype::F16,
                                eredu_gguf::DenseDtype::Bf16 => StoredDtype::BF16,
                                _ => {
                                    return Err(Error::backend(
                                        "numeric GGUF payload dtype is not admitted",
                                    ))
                                }
                            };
                            decode(&tensor.data, &dtype, lease.output_shape())
                        }
                        _ => Err(Error::backend(
                            "numeric dense payload binder does not admit packed GGUF",
                        )),
                    }
                }
                _ => decode(
                    lease
                        .encoded_bytes()
                        .ok_or_else(|| Error::backend("numeric lease has no scalar bytes"))?,
                    &lease.metadata().stored_dtype,
                    lease.output_shape(),
                ),
            }
        }
        DerivedWeightRecipe::Select { input, selection } => {
            select(recipe_value(input, checkpoint, context)?, selection)
        }
        DerivedWeightRecipe::Concatenate { axis, inputs } => {
            let inputs = inputs
                .iter()
                .map(|input| recipe_value(input, checkpoint, context))
                .collect::<Result<Vec<_>, _>>()?;
            NumericTensor::concatenate(
                &inputs,
                i32::try_from(*axis).map_err(Error::backend)?,
                context,
            )
        }
        DerivedWeightRecipe::Stack { axis, inputs } => {
            let inputs = inputs
                .iter()
                .map(|input| {
                    let value = recipe_value(input, checkpoint, context)?;
                    let mut dimensions = value.shape.clone();
                    if *axis > dimensions.len() {
                        return Err(Error::backend("numeric stack axis exceeds rank"));
                    }
                    dimensions.insert(*axis, 1);
                    value.reshape(&dimensions, context)
                })
                .collect::<Result<Vec<_>, _>>()?;
            NumericTensor::concatenate(
                &inputs,
                i32::try_from(*axis).map_err(Error::backend)?,
                context,
            )
        }
        DerivedWeightRecipe::Reshape {
            input,
            shape: dimensions,
        } => recipe_value(input, checkpoint, context)?.reshape(&shape(dimensions)?, context),
        DerivedWeightRecipe::Transpose { input, axes } => {
            recipe_value(input, checkpoint, context)?.transpose_axes(&shape(axes)?, context)
        }
        DerivedWeightRecipe::Cast { input, dtype } => {
            let value = recipe_value(input, checkpoint, context)?;
            match dtype {
                RecipeDtype::F32 => Ok(value),
                RecipeDtype::F16 => Ok(value.map(|x| half::f16::from_f32(x).to_f32())),
                RecipeDtype::BF16 => Ok(value.map(|x| half::bf16::from_f32(x).to_f32())),
                _ => Err(Error::backend("numeric dense payload cast is not admitted")),
            }
        }
        DerivedWeightRecipe::NegLog { input } => {
            let value = recipe_value(input, checkpoint, context)?;
            if value.data.iter().any(|x| !(*x < 0.0)) {
                return Err(Error::backend(
                    "negative-log source must be strictly negative",
                ));
            }
            Ok(value.map(|x| (-x).ln()))
        }
        DerivedWeightRecipe::SubtractOne { input } => {
            Ok(recipe_value(input, checkpoint, context)?.map(|x| x - 1.0))
        }
        DerivedWeightRecipe::View { .. } => Err(Error::backend(
            "numeric dense payload reinterpretation is not admitted",
        )),
    }
}
