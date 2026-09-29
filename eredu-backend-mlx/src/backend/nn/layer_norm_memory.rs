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
        ..
    } = *invocation
    else {
        return None;
    };
    // Pinned fast.cpp deliberately ignores bias promotion if weight is absent:
    // passed_bias is cast to the activation dtype in that case.
    if !weight {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn invocation(weight: bool, bias: bool) -> MechanismInvocation {
        MechanismInvocation::LayerNormalization {
            rows: 3,
            width: 5,
            element: TensorElementType::F16,
            weight,
            weight_element: weight.then_some(TensorElementType::Bf16),
            bias,
            bias_element: bias.then_some(TensorElementType::F32),
        }
    }

    #[test]
    fn cpu_layer_norm_graph_facts_preserve_native_promotion_and_unknown_backing() {
        for (weight, bias, expected) in [
            (false, false, TensorElementType::F16),
            (false, true, TensorElementType::F16),
            (true, false, TensorElementType::F32),
            (true, true, TensorElementType::F32),
        ] {
            let invocation = invocation(weight, bias);
            let report =
                super::super::memory::describe_for_device(&invocation, DeviceType::Cpu).unwrap();
            report.validate().unwrap();
            assert_eq!(
                report
                    .values
                    .iter()
                    .find(|v| v.name == "output")
                    .unwrap()
                    .element,
                expected
            );
            assert_eq!(
                report
                    .storage
                    .iter()
                    .find(|s| s.name == "mean_f32")
                    .unwrap()
                    .payload,
                MechanismBytes::exact(3 * 4)
            );
            assert_eq!(
                report
                    .storage
                    .iter()
                    .find(|s| s.name == "centered_f32")
                    .unwrap()
                    .payload,
                MechanismBytes::exact(3 * 5 * 4)
            );
            assert!(report.storage.iter().all(|s| s.capacity.upper.is_none()));
            assert!(report
                .storage
                .iter()
                .all(|s| s.backing == MechanismBacking::Unknown));
            assert!(!report.missing.is_empty());
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn fused_metal_layer_norm_does_not_duplicate_output_as_scratch() {
        let report = describe(&invocation(true, true), Some(DeviceType::Gpu)).unwrap();
        assert_eq!(report.storage.len(), 2);
        let output = report.storage.iter().find(|s| s.name == "output").unwrap();
        assert_eq!(output.capacity.upper, None);
        assert_eq!(output.backing, MechanismBacking::Unknown);
        let scratch = report
            .storage
            .iter()
            .find(|s| s.name == "reduction_buffer")
            .unwrap();
        assert_eq!(scratch.payload, MechanismBytes::exact(0));
        assert_eq!(scratch.capacity, MechanismBytes::exact(0));
        assert_eq!(scratch.retention, StorageRetention::NativeCompletion);
        assert!(!report.missing.is_empty());
    }
}
