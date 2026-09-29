//! Allocation-free descriptions of reusable MLX mechanisms.
//!
//! Kernel-private workspace and unproven allocation capacities remain unknown.
//! Named tensors below are individual payload facts, not simultaneous peaks.
use eredu_checkpoint::LinearFormat;
use eredu_nn::{mechanism_memory::*, Error, TensorElementType};

/// Describes the selected ordinary MLX invocation without a device or stream.
/// Cache-specific allocation reuse is refined by the cache instance hook.
pub fn describe(invocation: &MechanismInvocation) -> Result<MechanismMemoryContract, Error> {
    describe_selected(invocation, None, false)
}

/// Refines native path facts using the already-selected device kind, without
/// constructing a device, stream or tensor. Omitting it leaves device-dependent
/// conversions unknown in [`describe`].
pub fn describe_for_device(
    invocation: &MechanismInvocation,
    device: safemlx::DeviceType,
) -> Result<MechanismMemoryContract, Error> {
    describe_selected(invocation, Some(device), false)
}

/// The ordinary segmented primitive always submits unmasked native SDPA.
/// This describes one segment, excluding preparation and final concatenation.
pub(crate) fn describe_unmasked_segment(
    invocation: &MechanismInvocation,
    device: safemlx::DeviceType,
) -> Result<MechanismMemoryContract, Error> {
    if !matches!(invocation, MechanismInvocation::Attention {
        batch: 1, query_heads, kv_heads, queries, keys,
        arithmetic: eredu_nn::AttentionArithmetic::Fused, softcap: false, sinks: false, ..
    } if query_heads == kv_heads && queries == keys)
    {
        return Err(Error::backend(
            "segmented attention memory requires unmasked equal-length self-attention geometry",
        ));
    }
    describe_selected(invocation, Some(device), true)
}

fn describe_selected(
    invocation: &MechanismInvocation,
    device: Option<safemlx::DeviceType>,
    unmasked_attention: bool,
) -> Result<MechanismMemoryContract, Error> {
    use MechanismInvocation::*;
    let mut result = MechanismMemoryContract {
        values: invocation.logical_values()?,
        storage: Vec::new(),
        missing: Vec::new(),
    };
    let mut projection_pre_bias_element = None;
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
        if let Projection {
            bias: true,
            bias_element,
            ..
        } = invocation
        {
            projection_pre_bias_element = Some(output.element);
            if let Some(bias) = bias_element {
                output.element = promoted_element(output.element, *bias);
            }
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
        if unmasked_attention {
            storage.backing = MechanismBacking::Unknown;
            storage.capacity = MechanismBytes::unknown(0);
            storage.detail = "selected segment output payload; native layout, output views and assembled-result backing are not established by this report".into();
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
                } | Projection {
                    format: LinearFormat::Affine(_) | LinearFormat::MxFp4,
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
        LayerNormalization { .. } => {
            return super::layer_norm_memory::describe(invocation, device);
        }
        MultiAxisRotary {
            ref position_shape,
            position_element,
            ref spec,
        } => {
            describe_multi_axis_rotary(&mut result, position_shape, position_element, spec)?;
        }
        Projection {
            rows,
            input,
            output,
            format,
            element,
            weight_element,
            bias,
            bias_element,
        } => {
            match format {
                LinearFormat::Dense => {
                    if element == TensorElementType::F32
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
                    result.missing.push("packed matmul result promotion from scale and quantization-bias representations is not established; logical output representation remains conditional".into());
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
                let scalar = projection_pre_bias_element.unwrap_or_else(|| {
                    result
                        .values
                        .iter()
                        .find(|value| value.name == "output")
                        .expect("projection output")
                        .element
                });
                result.storage.push(allocation(
                    "pre_bias_output",
                    bytes(&[rows, output], element_bytes(scalar))?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                ));
                if format == LinearFormat::Dense && weight_element.is_none()
                    || matches!(format, LinearFormat::Affine(_) | LinearFormat::MxFp4)
                    || matches!(format, LinearFormat::GgufIQuant { .. }) && device.is_none()
                {
                    let pre_bias = result.storage.last_mut().expect("pre-bias output");
                    pre_bias.payload = MechanismBytes::unknown(0);
                    pre_bias.capacity = MechanismBytes::unknown(0);
                }
                if bias_element.is_none() {
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
            if path == super::attention::AttentionMemoryPath::Fused {
                if device == Some(safemlx::DeviceType::Cpu) {
                    // MLX fast.cpp's CPU SDPA fallback is ordinary full-score
                    // attention, not the tiled InputScores implementation below.
                    // Precise softmax accumulates internally in F32 but returns
                    // the input dtype. Native kernel workspace remains unknown.
                    let scalar = element_bytes(element);
                    let scores = bytes(&[batch, query_heads, queries, keys], scalar)?;
                    for (name, payload) in [
                        (
                            "sdpa_scaled_queries",
                            bytes(&[batch, query_heads, queries, key_width], scalar)?,
                        ),
                        ("sdpa_full_scores", scores),
                    ] {
                        let mut storage = allocation(
                            name,
                            payload,
                            MechanismStorageRole::Scratch,
                            StorageRetention::Evaluation,
                        );
                        if unmasked_attention && name == "sdpa_scaled_queries" {
                            storage.backing = MechanismBacking::Unknown;
                            storage.capacity = MechanismBytes::unknown(0);
                            storage.detail = "CPU SDPA scales queries in the promoted input representation; elementwise lowering may donate or reuse input backing".into();
                        }
                        result.storage.push(storage);
                    }
                    // Mask presence is not part of MechanismInvocation. The
                    // backend receives an optional caller-owned array mask;
                    // allow its where/add result without inventing a mask copy.
                    if !unmasked_attention {
                        let mut masked = allocation(
                            "sdpa_masked_scores",
                            scores,
                            MechanismStorageRole::Scratch,
                            StorageRetention::Evaluation,
                        );
                        masked.payload.lower = 0;
                        masked.capacity = MechanismBytes::unknown(0);
                        masked.detail = "CPU SDPA optional array-mask where/add result; mask presence is not described by this invocation".into();
                        result.storage.push(masked);
                    }
                    let columns = keys
                        .checked_add(u64::from(sinks))
                        .ok_or_else(|| Error::backend("attention sink column overflowed"))?;
                    let probability_bytes = bytes(&[batch, query_heads, queries, columns], scalar)?;
                    if sinks {
                        result.storage.push(allocation(
                            "sdpa_sink_scores",
                            probability_bytes,
                            MechanismStorageRole::Scratch,
                            StorageRetention::Evaluation,
                        ));
                    }
                    let mut probabilities = allocation(
                        "sdpa_probabilities",
                        probability_bytes,
                        MechanismStorageRole::Scratch,
                        StorageRetention::Evaluation,
                    );
                    probabilities.backing = MechanismBacking::Unknown;
                    probabilities.detail = if unmasked_attention {
                        "CPU SDPA precise softmax returns promoted-dtype full probabilities; native donation may reuse score storage"
                    } else {
                        "CPU SDPA precise softmax returns input-dtype full probabilities; native donation may reuse score storage, and removing the sink column is a view"
                    }.into();
                    result.storage.push(probabilities);
                    result.missing.push(if unmasked_attention {
                        "CPU SDPA fallback uses unmasked full score matrices; native buffer donation, layout and lazy graph overlap remain unresolved"
                    } else {
                        "CPU SDPA fallback uses full score matrices; mask presence, native buffer donation and lazy graph overlap require bound invocation/lifetime facts"
                    }.into());
                } else {
                    result.missing.push("native SDPA fused/fallback selection and scratch depend on device, layout and kernel geometry; full CPU score facts cannot be assumed".into());
                }
            } else {
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
        IndexedAttention {
            batch,
            query_heads,
            kv_heads,
            queries,
            selected,
            local,
            key_width,
            value_width,
            element,
            arithmetic,
            sinks,
        } => {
            let slots = selected
                .checked_add(local)
                .ok_or_else(|| Error::backend("selected attention extent overflowed"))?;
            // Only one batch/query's selection and score graph is live. The
            // concatenation of completed outputs is separately returned.
            for (name, width) in [
                ("selected_keys", key_width),
                ("selected_values", value_width),
            ] {
                let mut storage = allocation(
                    name,
                    bytes(&[kv_heads, selected, width], element_bytes(element))?,
                    MechanismStorageRole::Scratch,
                    StorageRetention::Evaluation,
                );
                storage.payload = MechanismBytes {
                    lower: 0,
                    upper: Some(bytes(&[kv_heads, selected, width, 8], 4)?),
                };
                storage.capacity = MechanismBytes::unknown(0);
                storage.detail = "bounded unique page copies, concatenation, slot remapping, invalid-slot clearing and dtype casts for one query; actual duplicates/invalids may reduce the payload".into();
                result.storage.push(storage);
            }
            let mut host = allocation(
                "selected_host_positions",
                bytes(&[selected], 32)?,
                MechanismStorageRole::Scratch,
                StorageRetention::NativeCompletion,
            );
            host.placement = MechanismPlacement::Host;
            host.detail = "one query's host positions, validity, deduplication and slot remapping; no history-sized catalog vector".into();
            result.storage.push(host);
            result.storage.push(allocation(
                "selected_score_mask",
                bytes(&[query_heads, slots, 4], 4)?,
                MechanismStorageRole::Scratch,
                StorageRetention::Evaluation,
            ));
            result.storage.push(allocation(
                "completed_output_rows",
                bytes(
                    &[batch, query_heads, queries, value_width],
                    element_bytes(element),
                )?,
                MechanismStorageRole::Scratch,
                StorageRetention::Evaluation,
            ));
            if slots > 0 {
                let nested = describe_selected(
                    &Attention {
                        batch: 1,
                        query_heads,
                        kv_heads,
                        queries: 1,
                        keys: slots,
                        key_width,
                        value_width,
                        element,
                        arithmetic,
                        softcap: false,
                        sinks,
                    },
                    device,
                    false,
                )?;
                for mut storage in nested.storage {
                    storage.name = format!("selected_query_{}", storage.name);
                    storage.role = MechanismStorageRole::Scratch;
                    storage.retention = StorageRetention::Evaluation;
                    result.storage.push(storage);
                }
                result.missing.extend(nested.missing);
            }
            result.missing.push("source cache pages, transfer buffers and metadata capacity remain owned by cache residency; native allocator capacity and fused-kernel private workspace are not exposed".into());
        }
        Convolution {
            batch,
            tokens,
            channels,
            kernel,
            dilation,
            element,
        } => {
            let history_len = (kernel - 1)
                .checked_mul(dilation)
                .ok_or_else(|| Error::backend("convolution history length overflowed"))?;
            let padded = tokens
                .checked_add(history_len)
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

// Mirrors generic Tensor::multi_axis_rotary_embeddings graph construction. These
// records describe individual logical payloads; none asserts simultaneous liveness.
fn describe_multi_axis_rotary(
    result: &mut MechanismMemoryContract,
    position_shape: &[u64],
    position_element: TensorElementType,
    spec: &eredu_nn::multimodal::MultiAxisRotarySpec,
) -> Result<(), Error> {
    use eredu_nn::multimodal::MultiAxisRotaryLayout;
    let rows = bytes(&position_shape[..position_shape.len() - 1], 1)?;
    if rows > i32::MAX as u64 || position_shape.iter().any(|d| *d > i32::MAX as u64) {
        return Err(Error::backend("MLX rotary position geometry exceeds i32"));
    }
    let dimensions = spec.dimensions()? as u64;
    let mut record = |name: String, extent: u64, placement, aliases: bool, detail: &str| {
        let mut storage = allocation(
            &name,
            extent,
            MechanismStorageRole::Scratch,
            StorageRetention::Evaluation,
        );
        storage.placement = placement;
        // Graph views/casts/concatenations can alias or fuse; logical byte extents
        // do not prove a separately allocated native buffer or minimum capacity.
        storage.backing = if aliases {
            MechanismBacking::Unknown
        } else {
            MechanismBacking::Invocation
        };
        storage.capacity = MechanismBytes::unknown(0);
        storage.detail = detail.into();
        if placement == MechanismPlacement::Host {
            storage.retention = StorageRetention::Unknown;
        }
        result.storage.push(storage);
    };
    for (index, axis) in spec.axes.iter().enumerate() {
        let half = axis.dimensions as u64 / 2;
        for name in [
            "host_inverse_frequencies",
            "host_inverse_tensor",
            "inverse_frequencies",
        ] {
            let host = name.starts_with("host_");
            record(format!("axis_{index}_{name}"), bytes(&[half], 4)?,
                if host { MechanismPlacement::Host } else { MechanismPlacement::Execution },
                true, "F32 inverse frequency Vec, source tensor or copied tensor from native rotary construction; host copying and lifetime are not exposed");
        }
        // I32 IDs match the constructor's I32 offset and minimum scalars exactly.
        // Other ID representations need MLX promotion facts, kept unknown below.
        if position_element == TensorElementType::I32 {
            for name in [
                "selected_positions",
                "offset_positions",
                "clamped_positions",
            ] {
                record(format!("axis_{index}_{name}"), bytes(&[rows], 4)?,
                    MechanismPlacement::Execution, true,
                    "I32 per-axis position slice, offset sum or clamped result; views and lazy fusion lack independent backing identity");
            }
        }
        record(
            format!("axis_{index}_float_positions"),
            bytes(&[rows], 4)?,
            MechanismPlacement::Execution,
            true,
            "F32 position cast and expanded view; aliasing depends on source representation",
        );
        record(format!("axis_{index}_angles"), bytes(&[rows, half], 4)?,
            MechanismPlacement::Execution, true, "F32 position/inverse-frequency product, including the unused axis graph built for round-robin layout");
        if spec.layout == MultiAxisRotaryLayout::IndependentAxes {
            record(
                format!("axis_{index}_expanded_angles"),
                bytes(&[rows, half, 2], 4)?,
                MechanismPlacement::Execution,
                true,
                "F32 doubled axis angle concatenation",
            );
        }
    }
    match spec.layout {
        MultiAxisRotaryLayout::IndependentAxes => {}
        MultiAxisRotaryLayout::SplitHalves => {
            record(
                "half_angles".into(),
                bytes(&[rows, dimensions / 2], 4)?,
                MechanismPlacement::Execution,
                true,
                "F32 concatenation of axis angles before repeating the half",
            );
        }
        MultiAxisRotaryLayout::RoundRobinSections => {
            for name in [
                "round_robin_host_inverse_frequencies",
                "round_robin_host_inverse_tensor",
                "round_robin_inverse_frequencies",
            ] {
                record(name.into(), bytes(&[dimensions / 2], 4)?,
                    if name.contains("host_") { MechanismPlacement::Host } else { MechanismPlacement::Execution },
                    true, "F32 global inverse frequency Vec, source tensor or copied tensor for round-robin layout");
            }
            if position_element == TensorElementType::I32 {
                record(
                    "round_robin_selected_positions".into(),
                    bytes(&[rows, dimensions / 2], 4)?,
                    MechanismPlacement::Execution,
                    true,
                    "I32 concatenation of per-frequency selected, offset and clamped positions",
                );
            }
            for name in ["round_robin_float_positions", "half_angles"] {
                record(
                    name.into(),
                    bytes(&[rows, dimensions / 2], 4)?,
                    MechanismPlacement::Execution,
                    true,
                    "F32 round-robin selected-position cast or inverse-frequency product",
                );
            }
        }
    }
    record("angles".into(), bytes(&[rows, dimensions], 4)?,
        MechanismPlacement::Execution, true, "F32 assembled angle tensor consumed by cosine and sine; concatenation may reuse a single input");
    if position_element != TensorElementType::I32 {
        result.missing.push("rotary position slicing, offset and clamp representation depends on MLX promotion with I32 scalars for non-I32 position inputs".into());
    }
    if spec.layout == MultiAxisRotaryLayout::RoundRobinSections {
        result.missing.push(format!("round-robin constructs {} per-frequency selected position views, offset sums and clamp results before concatenation; their backing, scalar constructors and retirement are undescribed", dimensions / 2));
    }
    result.missing.push("rotary scalar constructors, graph metadata, host Vec capacities, source tensor copying, view backing, fusion and intermediate retirement are not exposed; payloads cannot be summed as a live peak".into());
    Ok(())
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

pub(crate) fn promoted_element(
    left: TensorElementType,
    right: TensorElementType,
) -> TensorElementType {
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
    fn indexed_workspace_is_per_query_and_output_scales_with_prefill() {
        let request = |queries| MechanismInvocation::IndexedAttention {
            batch: 2,
            query_heads: 4,
            kv_heads: 2,
            queries,
            selected: 7,
            local: 3,
            key_width: 8,
            value_width: 6,
            element: TensorElementType::Bf16,
            arithmetic: eredu_nn::AttentionArithmetic::InputScores,
            sinks: true,
        };
        let one = describe(&request(1)).unwrap();
        let many = describe(&request(128)).unwrap();
        one.validate().unwrap();
        many.validate().unwrap();
        for storage in &one.storage {
            let other = many
                .storage
                .iter()
                .find(|other| other.name == storage.name)
                .unwrap();
            if storage.name == "output" || storage.name == "completed_output_rows" {
                assert_eq!(other.payload.lower, storage.payload.lower * 128);
            } else {
                assert_eq!(
                    storage.payload, other.payload,
                    "{} should not retain per-query scratch",
                    storage.name
                );
            }
        }
    }
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
