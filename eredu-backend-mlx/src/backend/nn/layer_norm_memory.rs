//! Storage facts for the ordinary native affine layer-normalization path.
use eredu_nn::{mechanism_memory::*, Error, TensorElementType};
use safemlx::DeviceType;

use super::memory::{allocation, bytes, promoted_element};

fn output_element(invocation: &MechanismInvocation) -> Option<TensorElementType> {
    let MechanismInvocation::LayerNormalization {
        element,
        weight,
        weight_element,
        bias,
        bias_element,
        arithmetic,
        ..
    } = *invocation
    else {
        return None;
    };
    // Pinned fast.cpp deliberately ignores bias promotion if weight is absent:
    // passed_bias is cast to the activation dtype in that case.
    if arithmetic == eredu_nn::LayerNormalizationArithmetic::CascadedMoments || !weight {
        return Some(element);
    }
    let mut output = promoted_element(element, weight_element?);
    if bias {
        output = promoted_element(output, bias_element?);
    }
    Some(output)
}

pub(crate) fn describe(
    invocation: &MechanismInvocation,
    device: Option<DeviceType>,
) -> Result<MechanismMemoryContract, Error> {
    let MechanismInvocation::LayerNormalization {
        rows,
        width,
        element,
        weight,
        weight_element,
        bias,
        bias_element,
        arithmetic,
    } = *invocation
    else {
        return Err(Error::backend(
            "layer normalization memory requires matching invocation",
        ));
    };
    let mut report = MechanismMemoryContract {
        values: invocation.logical_values()?,
        storage: Vec::new(),
        missing: Vec::new(),
    };
    let output_type = output_element(invocation);
    let output = report
        .values
        .iter_mut()
        .find(|v| v.name == "output")
        .expect("layer norm output");
    if let Some(element) = output_type {
        output.element = element;
    }
    let mut storage = allocation(
        "output",
        output.logical_bytes()?,
        MechanismStorageRole::Output,
        StorageRetention::Returned,
    );
    storage.backing = MechanismBacking::Unknown;
    storage.capacity = MechanismBytes::unknown(0);
    storage.detail = "normalization result may donate an existing backing or share its compaction result; logical shape does not establish allocation capacity".into();
    if output_type.is_none() {
        storage.payload = MechanismBytes::unknown(0);
        report.missing.push(
            "normalization output representation depends on unknown affine parameter dtypes".into(),
        );
    }
    report.storage.push(storage);

    if arithmetic == eredu_nn::LayerNormalizationArithmetic::CascadedMoments {
        let chunks = (width / 4).div_ceil(16);
        let depth = (u64::BITS - chunks.saturating_sub(1).leading_zeros()).max(1) as u64;
        for (name, size) in [
            ("cascaded_input_f32", bytes(&[rows, width], 4)?),
            ("cascaded_output_f32", bytes(&[rows, width], 4)?),
            ("cascaded_host_output_vec", bytes(&[rows, width], 4)?),
            ("cascaded_moments_stack", bytes(&[depth], 40)?),
            ("cascaded_gain_f32", bytes(&[u64::from(weight), width], 4)?),
            ("cascaded_bias_f32", bytes(&[u64::from(bias), width], 4)?),
            (
                "cascaded_host_input_copy",
                bytes(&[rows, width], element_bytes(element))?,
            ),
        ] {
            let mut storage = allocation(
                name,
                size,
                MechanismStorageRole::Scratch,
                StorageRetention::Evaluation,
            );
            storage.payload.lower = 0;
            storage.backing = MechanismBacking::Unknown;
            storage.capacity = MechanismBytes::unknown(0);
            storage.placement = MechanismPlacement::Host;
            storage.detail = "bounded CPU moments/affine operation; non-CPU execution uses host activation/parameter copies; aliasing and cast backing remain unknown".into();
            report.storage.push(storage);
        }
        for (name, present, element) in [
            ("cascaded_host_gain_copy", weight, weight_element),
            ("cascaded_host_bias_copy", bias, bias_element),
        ] {
            if !present {
                continue;
            }
            let mut storage = allocation(
                name,
                0,
                MechanismStorageRole::ParameterConversion,
                StorageRetention::Evaluation,
            );
            storage.payload = match element {
                Some(element) => MechanismBytes {
                    lower: 0,
                    upper: Some(bytes(&[width], element_bytes(element))?),
                },
                None => MechanismBytes::unknown(0),
            };
            storage.backing = MechanismBacking::Unknown;
            storage.capacity = MechanismBytes::unknown(0);
            storage.placement = MechanismPlacement::Host;
            storage.detail = "conditional original-precision parameter copy for the selected-device host fallback; bound scalar metadata fixes the payload upper bound".into();
            report.storage.push(storage);
        }
        report.missing.push("selected-device copy backing and result-transfer overlap require bound native storage facts".into());
        return Ok(report);
    }

    let mut graph_value =
        |name: &str, count: u64, dtype: TensorElementType, detail: &str| -> Result<(), Error> {
            let mut storage = allocation(
                name,
                bytes(&[count], element_bytes(dtype))?,
                MechanismStorageRole::Scratch,
                StorageRetention::Unknown,
            );
            storage.backing = MechanismBacking::Unknown;
            storage.capacity = MechanismBytes::unknown(0);
            storage.detail = detail.into();
            report.storage.push(storage);
            Ok(())
        };
    let count = bytes(&[rows, width], 1)?;
    if device == Some(DeviceType::Cpu) {
        // CPU fast::LayerNorm::use_fallback selects the composed fast.cpp graph.
        // These are graph values, not claims of distinct physical allocations.
        if element != TensorElementType::F32 {
            graph_value("input_f32", count, TensorElementType::F32, "CPU fallback activation cast; physical backing and retirement depend on native lowering")?;
        }
        for (name, count) in [
            ("mean_f32", rows),
            ("centered_f32", count),
            ("squared_centered_f32", count),
            ("variance_f32", rows),
            ("stabilized_variance_f32", rows),
            ("inverse_stddev_f32", rows),
        ] {
            graph_value(name, count, TensorElementType::F32, "CPU fallback F32 graph value; broadcasting, donation, fusion and retirement remain unresolved")?;
        }
        if weight || bias || output_type != Some(TensorElementType::F32) {
            graph_value("normalized_f32", count, TensorElementType::F32, "CPU fallback normalized value before result cast and affine operations; backing may be reused")?;
        }
        report.missing.push("CPU normalization fallback also contains scalar construction, affine/cast graph dependencies and native reduction workspace; graph values are not disjoint allocations or a live peak".into());
    } else if device.is_some_and(super::native_quantization::uses_metal_device)
        && output_type.is_some_and(|e| {
            matches!(
                e,
                TensorElementType::F32 | TensorElementType::F16 | TensorElementType::Bf16
            )
        })
    {
        // Metal computes mean/variance inside the kernel. Its only tensor
        // allocation is output (or a compaction sharing that output backing).
        let mut scratch = allocation(
            "reduction_buffer",
            0,
            MechanismStorageRole::Scratch,
            StorageRetention::NativeCompletion,
        );
        scratch.capacity = MechanismBytes::exact(0);
        scratch.detail = "selected fused Metal forward normalization allocates no separate tensor reduction buffer; compaction, if needed, is the returned output backing".into();
        report.storage.push(scratch);
        report.missing.push("Metal kernel-private register/threadgroup storage, native descriptors and aggregate lifetimes remain undescribed".into());
    } else {
        report.missing.push("selected normalization lowering and native workspace are not established for this device/representation".into());
    }
    if output_type != Some(element)
        || !weight
        || !bias
        || weight_element != output_type
        || bias_element != output_type
    {
        report.missing.push("activation/affine dtype conversions and absent-parameter scalar construction may create additional dependencies; their backing and lifetimes remain unresolved".into());
    }
    report.validate()?;
    Ok(report)
}
