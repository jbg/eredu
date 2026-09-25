//! Allocation-free descriptions of reusable MLX mechanisms.
//!
//! Kernel-private workspace and allocator capacity are deliberately unknown.
//! Named tensors below are individual payload facts, not simultaneous peaks.
use eredu_checkpoint::LinearFormat;
use eredu_nn::{mechanism_memory::*, Error, TensorElementType};

/// Describes the selected ordinary MLX invocation without a device or stream.
/// Cache-specific allocation reuse is refined by the cache instance hook.
pub fn describe(invocation: &MechanismInvocation) -> Result<MechanismMemoryContract, Error> {
    describe_selected(invocation, None, None)
}

/// Refines native path facts using the already-selected device kind, without
/// constructing a device, stream or tensor. Omitting it leaves device-dependent
/// conversions unknown in [`describe`].
pub fn describe_for_device(
    invocation: &MechanismInvocation,
    device: safemlx::DeviceType,
) -> Result<MechanismMemoryContract, Error> {
    describe_selected(invocation, Some(device), None)
}

/// Describe an ordinary projection against its actual bound weight, without
/// evaluating it. Unknown/lazy layouts retain the same conservative contract.
/// Exact partial payload and a layout-dependent activation-copy envelope replace
/// the generic promotion allowance only when the shared selector is covered.
pub fn describe_bound_projection(
    invocation: &MechanismInvocation,
    weight: &safemlx::Array,
    stream: &safemlx::Stream,
) -> Result<MechanismMemoryContract, Error> {
    let device = stream
        .get_device()
        .and_then(|d| d.get_type())
        .map_err(Error::backend)?;
    let mut workspace = None;
    if let MechanismInvocation::Projection {
        rows,
        input,
        output,
        format: LinearFormat::Dense,
        element: TensorElementType::F32,
        weight_element: Some(element),
        ..
    } = invocation
    {
        let expected = match element {
            TensorElementType::F16 => Some(safemlx::Dtype::Float16),
            TensorElementType::Bf16 => Some(safemlx::Dtype::Bfloat16),
            _ => None,
        };
        if expected == Some(weight.dtype()) {
            if let Some(facts) =
                super::mixed_projection::storage_facts(weight, stream).map_err(Error::backend)?
            {
                if let Some(range) = facts.for_invocation(
                    *input,
                    *output,
                    &eredu_core::checkpoint::TensorDtype::F32,
                    *rows,
                ) {
                    workspace = Some((
                        bytes(&[*rows, range.partial_bytes_per_row], 1)?,
                        bytes(&[*rows, facts.activation_copy_bytes_per_row], 1)?,
                    ));
                }
            }
        }
    }
    describe_selected(invocation, Some(device), workspace)
}

fn describe_selected(
    invocation: &MechanismInvocation,
    device: Option<safemlx::DeviceType>,
    projection_workspace: Option<(u64, u64)>,
) -> Result<MechanismMemoryContract, Error> {
    use MechanismInvocation::*;
    let mut result = MechanismMemoryContract {
        values: invocation.logical_values()?,
        storage: Vec::new(),
        missing: Vec::new(),
    };
    // Native lowering can promote the portable nominal result representation.
    if let Some(output) = result
        .values
        .iter_mut()
        .find(|value| value.name == "output")
    {
        match invocation {
            Projection {
                format: LinearFormat::Dense,
                element,
                weight_element: Some(weight),
                ..
            } => {
                output.element = promoted_element(*element, *weight);
            }
            Projection {
                format: LinearFormat::GgufIQuant { .. },
                ..
            } if device
                .is_some_and(|device| !super::native_quantization::uses_metal_device(device)) =>
            {
                output.element = TensorElementType::F32
            }
            Recurrent {
                kind: RecurrentKind::GatedDelta,
                ..
            } => output.element = TensorElementType::F32,
            _ => {}
        }
    }
    // Output shapes do not prove alias-free storage for cache updates or causal
    // history views. Those are refined by the ordinary cache/convolution owner.
    for value in result.values.clone() {
        if value.kind == LogicalValueKind::Input {
            continue;
        }
        if value.name == "adaptive_state" {
            let mut state = allocation(
                "adaptive_state",
                4,
                MechanismStorageRole::State,
                StorageRetention::Returned,
            );
            state.backing = MechanismBacking::Unknown;
            state.placement = MechanismPlacement::Host;
            state.detail =
                "one mu scalar in the existing Mirostat sampler; not a fresh device tensor".into();
            result.storage.push(state);
            continue;
        }
        let aliases = matches!(invocation, CacheUpdate { .. })
            || matches!(invocation, Convolution { .. }) && value.kind == LogicalValueKind::State;
        let mut storage = allocation(
            &value.name,
            value.logical_bytes()?,
            if value.kind == LogicalValueKind::State {
                MechanismStorageRole::State
            } else {
                MechanismStorageRole::Output
            },
            StorageRetention::Returned,
        );
        if aliases {
            storage.backing = MechanismBacking::Unknown;
            storage.detail = "logical result can alias input/state or an implementation temporary; backing requires owner facts".into();
        }
        if matches!(
            invocation,
            Projection {
                format: LinearFormat::GgufIQuant { .. },
                ..
            }
        ) && device.is_none()
            || matches!(
                invocation,
                Projection {
                    format: LinearFormat::Dense,
                    weight_element: None,
                    ..
                } | ExpertDispatch { .. }
                    | Convolution { .. }
            ) && value.kind == LogicalValueKind::Output
        {
            storage.payload = MechanismBytes::unknown(0);
            storage.capacity = MechanismBytes::unknown(0);
            storage.detail =
                "result representation depends on selected device or bound parameter dtypes".into();
        }
        result.storage.push(storage);
    }
    match *invocation {
        Projection {
            rows,
            input,
            output,
            format,
            element,
            weight_element,
            bias,
        } => {
            match format {
                LinearFormat::Dense => {
                    if projection_workspace.is_none()
                        && element == TensorElementType::F32
                        && matches!(
                            weight_element,
                            Some(TensorElementType::Bf16 | TensorElementType::F16)
                        )
                    {
                        let mut promotion = allocation(
                            "promoted_weight",
                            bytes(&[input, output], 4)?,
                            MechanismStorageRole::ParameterConversion,
                            StorageRetention::Unknown,
                        );
                        promotion.backing = MechanismBacking::Unknown;
                        promotion.detail = "ordinary F32 activation promotion; residency registry decides whether conversion is retained by a parameter owner or invocation-local".into();
                        result.storage.push(promotion);
                        result.missing.push("converted weight backing/retention requires the actual resident parameter owner".into());
                    } else if weight_element.is_none() {
                        result.missing.push(
                            "dense weight dtype and any promotion storage are unknown".into(),
                        );
                    }
                }
                LinearFormat::Affine(_) | LinearFormat::MxFp4 => {
                    result.missing.push("native packed matmul's internal activation conversions/workspace are opaque; packed parameters are borrowed".into());
                }
                LinearFormat::GgufIQuant { .. } => match device {
                    Some(device) if super::native_quantization::uses_metal_device(device) => {
                        result.missing.push("GGUF direct Metal kernel register/threadgroup scratch is native-private; no full dense weight conversion".into());
                    }
                    Some(_) => {
                        let mut row = allocation(
                            "decoded_weight_row",
                            bytes(&[input], 4)?,
                            MechanismStorageRole::Scratch,
                            StorageRetention::NativeCompletion,
                        );
                        row.placement = MechanismPlacement::Host;
                        row.detail =
                            "ordinary GGUF fallback decodes one F32 weight row at a time".into();
                        result.storage.push(row);
                        let mut output = allocation(
                            "host_projection_output",
                            bytes(&[rows, output], 4)?,
                            MechanismStorageRole::Scratch,
                            StorageRetention::NativeCompletion,
                        );
                        output.placement = MechanismPlacement::Host;
                        result.storage.push(output);
                        if element != TensorElementType::F32 {
                            result.storage.push(allocation(
                                "float32_activation",
                                bytes(&[rows, input], 4)?,
                                MechanismStorageRole::Scratch,
                                StorageRetention::Evaluation,
                            ));
                        }
                    }
                    None => result.missing.push(
                        "GGUF device selection is unknown (direct Metal vs host row-wise decoding)"
                            .into(),
                    ),
                },
                LinearFormat::E4M3BlockFp8(_) => {
                    result.storage.push(allocation(
                        "fp8_activations",
                        bytes(&[rows, input], 1)?,
                        MechanismStorageRole::Scratch,
                        StorageRetention::Evaluation,
                    ));
                    result.storage.push(allocation(
                        "fp8_activation_scales",
                        bytes(&[rows, input.div_ceil(super::fp8::SCALE_BLOCK as u64)], 4)?,
                        MechanismStorageRole::Scratch,
                        StorageRetention::Evaluation,
                    ));
                    if device == Some(safemlx::DeviceType::Cpu) {
                        result.storage.push(allocation(
                            "dequantized_weight",
                            bytes(&[output, input], 4)?,
                            MechanismStorageRole::Scratch,
                            StorageRetention::Evaluation,
                        ));
                        result.storage.push(allocation(
                            "dequantized_activations",
                            bytes(&[rows, input], 4)?,
                            MechanismStorageRole::Scratch,
                            StorageRetention::Evaluation,
                        ));
                    } else if device.is_none() {
                        result.missing.push(
                            "FP8 CPU fallback dequantized weight/input depends on selected device"
                                .into(),
                        );
                    }
                }
            }
            if bias {
                let scalar = result
                    .values
                    .iter()
                    .find(|value| value.name == "output")
                    .expect("projection output")
                    .element;
                result.storage.push(allocation(
                    "pre_bias_output",
                    bytes(&[rows, output], element_bytes(scalar))?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                ));
                if weight_element.is_none()
                    || matches!(format, LinearFormat::GgufIQuant { .. }) && device.is_none()
                {
                    let pre_bias = result.storage.last_mut().expect("pre-bias output");
                    pre_bias.payload = MechanismBytes::unknown(0);
                    pre_bias.capacity = MechanismBytes::unknown(0);
                }
                result.missing.push("ordinary bias dtype can further promote the final output; inspect its bound parameter metadata".into());
                if let Some(output) = result
                    .storage
                    .iter_mut()
                    .find(|storage| storage.name == "output")
                {
                    output.payload.upper = None;
                }
            }
        }
        Attention {
            batch,
            query_heads,
            kv_heads,
            queries,
            keys,
            key_width,
            value_width,
            element,
            arithmetic,
            softcap,
            sinks,
        } => {
            let path = super::attention::memory_path(queries, keys, arithmetic, softcap);
            if path != super::attention::AttentionMemoryPath::Fused {
                let score_bytes = if arithmetic == eredu_nn::AttentionArithmetic::InputScores {
                    element_bytes(element)
                } else {
                    4
                };
                let tile = super::attention::memory_tile_geometry(queries, keys, path);
                let score_columns = tile
                    .1
                    .checked_add(u64::from(sinks))
                    .ok_or_else(|| Error::backend("attention sink column overflowed"))?;
                let score_count = bytes(&[batch, query_heads, tile.0, score_columns], 1)?;
                result.storage.push(allocation(
                    "score_tile",
                    bytes(&[score_count], score_bytes)?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                ));
                result.storage.push(allocation(
                    "softmax_float32_tile",
                    bytes(&[score_count], 4)?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                ));
                let mut probabilities = allocation(
                    "probability_tile",
                    bytes(
                        &[batch, query_heads, tile.0, tile.1],
                        element_bytes(element),
                    )?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                );
                if element == TensorElementType::F32 {
                    probabilities.backing = MechanismBacking::Unknown;
                    probabilities.detail = "F32 probability result is a view of softmax output, including sink removal; no second independent allocation".into();
                }
                result.storage.push(probabilities);
                for (name, width, scalar) in [
                    ("prepared_keys", key_width, score_bytes),
                    ("prepared_values", value_width, element_bytes(element)),
                ] {
                    let mut storage = allocation(
                        name,
                        bytes(&[batch, query_heads, tile.1, width], scalar)?,
                        MechanismStorageRole::Scratch,
                        StorageRetention::Evaluation,
                    );
                    // Contiguous no-op and dtype-preserving casts can retain the
                    // borrowed input. We cannot assign fresh backing cold.
                    storage.backing = MechanismBacking::Unknown;
                    storage.detail = format!("selected {:?} prepared K/V layout; grouped repetition {} and contiguous conversion may alias input", path, query_heads / kv_heads);
                    result.storage.push(storage);
                }
                result.missing.push(format!("selected {path:?}: lazy graph retention follows tile evaluation policy; tile count and overlap require lifetime composition"));
            }
        }
        Convolution {
            batch,
            tokens,
            channels,
            kernel,
            element,
        } => {
            let padded = tokens
                .checked_add(kernel - 1)
                .ok_or_else(|| Error::backend("convolution history concatenation overflowed"))?;
            let mut history = allocation(
                "history_and_input",
                bytes(&[batch, padded, channels], element_bytes(element))?,
                MechanismStorageRole::Scratch,
                StorageRetention::Evaluation,
            );
            if kernel == 1 {
                history.backing = MechanismBacking::Unknown;
            }
            result.storage.push(history);
            result.missing.push("convolution native kernel workspace and activation/bias intermediates depend on implementation lowering".into());
        }
        Recurrent {
            batch,
            tokens,
            heads,
            value_width,
            state_width,
            kind,
            ..
        } => {
            result.storage.push(allocation(
                "float32_input",
                bytes(&[batch, tokens, heads, value_width], 4)?,
                MechanismStorageRole::Scratch,
                StorageRetention::Evaluation,
            ));
            // A cast can be a no-op. Describe its logical payload without a
            // fictitious independent allocation when actual dtype is F32.
            result.storage.last_mut().unwrap().backing = MechanismBacking::Unknown;
            result.missing.push(format!("{kind:?} native scan selection (device, vector/scalar decay and chunk kernels), intermediate state matrices [{batch}, {heads}, {value_width}, {state_width}] and retention require selected kernel facts"));
        }
        ExpertDispatch {
            rows,
            selected,
            input,
            output,
            element,
            ..
        } => {
            for name in ["selection_order", "sorted_group_ids", "token_indices"] {
                result.storage.push(allocation(
                    name,
                    bytes(&[rows, selected], 4)?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                ));
            }
            result.storage.push(allocation(
                "gathered_input_rows",
                bytes(&[rows, selected, input], element_bytes(element))?,
                MechanismStorageRole::Scratch,
                StorageRetention::Evaluation,
            ));
            result.storage.push(allocation(
                "selected_projection_output",
                bytes(&[rows, selected, output], element_bytes(element))?,
                MechanismStorageRole::Scratch,
                StorageRetention::Evaluation,
            ));
            result.storage.push(allocation(
                "weighted_routes",
                bytes(&[rows, selected, output], element_bytes(element))?,
                MechanismStorageRole::Scratch,
                StorageRetention::Evaluation,
            ));
            // Bound expert parameter/bias dtypes and reduction policy can
            // promote these results; nominal activation bytes cannot bound them.
            for storage in result.storage.iter_mut().filter(|storage| {
                matches!(
                    storage.name.as_str(),
                    "selected_projection_output" | "weighted_routes"
                )
            }) {
                storage.payload = MechanismBytes::unknown(0);
                storage.capacity = MechanismBytes::unknown(0);
                storage.detail =
                    "selected expert result dtype depends on bound parameters and reduction policy"
                        .into();
            }
            result.missing.push("expert gather/sort and quantized projection scratch vary by selected format/device; indices and coefficients are borrowed".into());
        }
        CacheUpdate { .. } => {
            result.missing.push("cache append backing, capacity growth, sliding views and compression require the cache instance's update_memory_contract".into());
        }
        Sampling {
            rows,
            vocabulary,
            history,
            element,
            mode,
            top_k,
            top_p,
            min_p,
            penalties,
        } => {
            if penalties && history > 0 {
                for (name, scalar) in [("penalty_mask", 1), ("penalty_additive", 4)] {
                    let mut storage = allocation(
                        name,
                        bytes(&[rows, vocabulary], scalar)?,
                        MechanismStorageRole::Scratch,
                        StorageRetention::NativeCompletion,
                    );
                    storage.placement = MechanismPlacement::Host;
                    storage.detail = "actual apply_penalties host vector payload; Vec allocation capacity is unspecified".into();
                    result.storage.push(storage);
                }
                result.missing.push(
                    "host penalty count map allocation capacity depends on distinct history tokens"
                        .into(),
                );
            }
            if top_k > 0 && top_k < vocabulary {
                result.storage.push(allocation(
                    "top_k_values",
                    bytes(&[rows, top_k], element_bytes(element))?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                ));
            }
            if top_p {
                result.storage.push(allocation(
                    "top_p_sorted_indices",
                    bytes(&[rows, vocabulary], 4)?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                ));
                for name in [
                    "top_p_sorted_logits",
                    "top_p_probabilities",
                    "top_p_cumulative",
                    "top_p_before",
                ] {
                    result.storage.push(allocation(
                        name,
                        bytes(&[rows, vocabulary], element_bytes(element))?,
                        MechanismStorageRole::Scratch,
                        StorageRetention::Evaluation,
                    ));
                }
            }
            if min_p || mode == SamplingMode::MirostatV2 {
                result.storage.push(allocation(
                    "filter_probabilities",
                    bytes(&[rows, vocabulary], element_bytes(element))?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                ));
            }
            result.missing.push("sampling random state, filtered logits, reductions and sorting workspace depend on native lowering and owned sampler state".into());
        }
    }
    if let Some((partials, activation_copy)) = projection_workspace {
        result.storage.push(allocation(
            "split_k_partials",
            partials,
            MechanismStorageRole::Scratch,
            StorageRetention::NativeCompletion,
        ));
        let mut copy = allocation(
            "activation_layout_copy",
            activation_copy,
            MechanismStorageRole::Scratch,
            StorageRetention::NativeCompletion,
        );
        copy.payload = MechanismBytes {
            lower: 0,
            upper: Some(activation_copy),
        };
        copy.capacity = MechanismBytes::unknown(0);
        copy.detail = "native activation layout is resolved at evaluation; logical copy upper payload, no full-weight promotion".into();
        result.storage.push(copy);
        result.missing.push("allocator rounding/capacity and GPU-private registers/threadgroup scratch are not exposed; native partial payload is exact".into());
        return Ok(result);
    }
    result.storage.push(MechanismStorage {
        name: "native_workspace".into(),
        role: MechanismStorageRole::Scratch,
        payload: MechanismBytes::unknown(0),
        capacity: MechanismBytes::unknown(0),
        backing: MechanismBacking::Invocation,
        placement: MechanismPlacement::Execution,
        retention: StorageRetention::NativeCompletion,
        detail: "MLX does not expose a per-invocation upper bound for native kernel workspace"
            .into(),
    });
    result.missing.push(
        "native kernel workspace and allocator alignment/capacity are not exposed by MLX".into(),
    );
    result.validate()?;
    Ok(result)
}

pub(crate) fn allocation(
    name: &str,
    payload: u64,
    role: MechanismStorageRole,
    retention: StorageRetention,
) -> MechanismStorage {
    MechanismStorage {
        name: name.into(),
        role,
        payload: MechanismBytes::exact(payload),
        capacity: MechanismBytes::unknown(payload),
        backing: MechanismBacking::Invocation,
        placement: MechanismPlacement::Execution,
        retention,
        detail: "MLX selected mechanism tensor payload; physical allocator capacity is not exposed"
            .into(),
    }
}

pub(crate) fn bytes(shape: &[u64], scalar: u64) -> Result<u64, Error> {
    shape
        .iter()
        .chain(std::iter::once(&scalar))
        .try_fold(1u64, |total, value| total.checked_mul(*value))
        .ok_or_else(|| Error::backend("MLX mechanism byte geometry overflowed"))
}

fn promoted_element(left: TensorElementType, right: TensorElementType) -> TensorElementType {
    use TensorElementType::*;
    match (left, right) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) | (F16, Bf16) | (Bf16, F16) => F32,
        _ => left,
    }
}

/// Converts scalar metadata without inspecting tensor contents.
pub(crate) fn element_type(dtype: safemlx::Dtype) -> Option<TensorElementType> {
    use safemlx::Dtype as D;
    use TensorElementType as T;
    Some(match dtype {
        D::Bool => T::Bool,
        D::Float16 => T::F16,
        D::Bfloat16 => T::Bf16,
        D::Float32 => T::F32,
        D::Float64 => T::F64,
        D::Int8 => T::I8,
        D::Int16 => T::I16,
        D::Int32 => T::I32,
        D::Int64 => T::I64,
        D::Uint8 => T::U8,
        D::Uint16 => T::U16,
        D::Uint32 => T::U32,
        D::Uint64 => T::U64,
        D::Complex64 => T::Complex64,
    })
}

#[cfg(test)]
mod contract_validation_tests {
    use super::*;
    #[test]
    fn mechanism_attention_sink_extent_overflow_is_rejected() {
        let request = MechanismInvocation::Attention {
            batch: 1,
            query_heads: 1,
            kv_heads: 1,
            queries: 1,
            keys: u64::MAX,
            key_width: 1,
            value_width: 1,
            element: TensorElementType::U8,
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
            softcap: true,
            sinks: true,
        };
        // Logical input byte extents fit; the added sink column does not.
        assert!(request.logical_values().is_ok());
        assert!(describe(&request).is_err());
    }
}
