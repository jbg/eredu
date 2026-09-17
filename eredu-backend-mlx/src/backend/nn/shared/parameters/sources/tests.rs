use super::*;
use crate::module::{PhysicalParam, PhysicalParameter};
use safemlx::{Device, DeviceType};

#[derive(Default)]
struct Rows<'a> {
    values: Vec<(ParameterMetadataView<'a>, &'a MlxTensor)>,
    auxiliary: Vec<&'a MlxTensor>,
}
impl<'a> ParameterSourceVisitor<'a, MlxTensor> for Rows<'a> {
    fn parameter(&mut self, metadata: ParameterMetadataView<'a>, value: &'a MlxTensor) {
        self.values.push((metadata, value));
    }
    fn retained(&mut self, value: &'a MlxTensor) {
        self.auxiliary.push(value);
    }
}
fn norm() -> nn::RmsNorm {
    nn::RmsNorm {
        weight: PhysicalParam::new(Array::from_slice(&[2_f32, 3.0], &[2])),
        eps: 1e-5,
    }
}
#[test]
fn strict_topology_refuses_missing_extra_and_wrong_keys_before_callbacks() {
    let native = norm();
    for topology in [
        BTreeMap::new(),
        BTreeMap::from([("other".into(), ParameterSpec::trainable("x").unwrap())]),
        BTreeMap::from([
            ("weight".into(), ParameterSpec::trainable("w").unwrap()),
            ("other".into(), ParameterSpec::trainable("x").unwrap()),
        ]),
    ] {
        let mut rows = Rows::default();
        assert!(matches!(
            visit_module_parameter_sources(&native, &topology, &mut rows),
            Err(ParameterSourceError::TopologyMismatch { .. })
        ));
        assert!(rows.values.is_empty());
        assert!(rows.auxiliary.is_empty());
    }
}
#[test]
fn native_borrowed_rows_follow_actual_freeze_state_and_preserve_ordinary_metadata() {
    let mut native = norm();
    native.weight.freeze(false);
    let topology = BTreeMap::from([(
        "weight".into(),
        ParameterSpec::trainable("norm.échelle").unwrap(),
    )]);
    let mut rows = Rows::default();
    visit_module_parameter_sources(&native, &topology, &mut rows).unwrap();
    assert!(!rows.values[0].0.trainable());
    assert!(std::ptr::eq(
        rows.values[0].1.as_array(),
        &native.weight.value
    ));
    struct Ordinary(Vec<eredu_nn::ParameterMetadata>);
    impl<'a> eredu_nn::ParameterVisitor<'a, MlxTensor> for Ordinary {
        fn visit(&mut self, m: eredu_nn::ParameterMetadata, _: &'a MlxTensor) {
            self.0.push(m);
        }
    }
    let mut ordinary = Ordinary(Vec::new());
    visit_module_parameters(&native, &topology, &mut ordinary);
    assert_eq!(ordinary.0, [rows.values[0].0.to_owned()]);
    drop(rows);
    native.weight.unfreeze(false);
    let mut rows = Rows::default();
    visit_module_parameter_sources(&native, &topology, &mut rows).unwrap();
    assert!(rows.values[0].0.trainable());
}
#[derive(Debug, Clone, eredu_backend_mlx_macros::PhysicalParameters)]
#[module(root = crate)]
struct Legacy {
    #[param]
    weight: PhysicalParam<Array>,
}
impl NativeRetainedValues for Legacy {
    fn visit_native_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        visitor(MlxTensor::ref_cast(&self.weight.value));
        false
    }
}
#[test]
fn custom_native_keeps_ordinary_adapter_and_strict_default_refusal() {
    let native = Legacy {
        weight: norm().weight,
    };
    let topology = BTreeMap::from([(
        "weight".into(),
        ParameterSpec::trainable("legacy.weight").unwrap(),
    )]);
    let mut rows = Rows::default();
    assert_eq!(
        visit_module_parameter_sources(&native, &topology, &mut rows),
        Err(ParameterSourceError::Unavailable)
    );
    assert!(rows.values.is_empty());
    struct Ordinary(usize);
    impl<'a> eredu_nn::ParameterVisitor<'a, MlxTensor> for Ordinary {
        fn visit(&mut self, _: eredu_nn::ParameterMetadata, _: &'a MlxTensor) {
            self.0 += 1;
        }
    }
    let mut ordinary = Ordinary(0);
    visit_module_parameters(&native, &topology, &mut ordinary);
    assert_eq!(ordinary.0, 1);
}
#[test]
fn lazy_source_traversal_preserves_unknown_without_native_work() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[2_f32, 3.0], &[2]);
    let native = nn::RmsNorm {
        weight: PhysicalParam::new(root.square(&stream).unwrap()),
        eps: 1e-5,
    };
    let topology = BTreeMap::from([("weight".into(), ParameterSpec::trainable("lazy").unwrap())]);
    let mut rows = Rows::default();
    visit_module_parameter_sources(&native, &topology, &mut rows).unwrap();
    let mut runtime = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    assert_eq!(
        runtime
            .descriptor(rows.values[0].1.as_array())
            .unwrap()
            .facts()
            .allocation(),
        None
    );
    drop(runtime);
    drop(rows);
    assert_eq!(native.weight.value.allocation_info().unwrap(), None);
}

#[derive(Debug, Clone, eredu_backend_mlx_macros::PhysicalParameters)]
#[module(root = crate)]
struct RepeatedKey {
    #[param]
    weight: PhysicalParam<Array>,
    #[param]
    other: PhysicalParam<Array>,
}
impl NativeRetainedValues for RepeatedKey {
    fn native_parameter_source_count(&self) -> Option<usize> {
        Some(2)
    }
    fn visit_native_parameter_sources<'a>(
        &'a self,
        visitor: &mut dyn NativeParameterSourceVisitor<'a>,
    ) -> Result<(), ParameterSourceError> {
        visitor.parameter("weight", &self.weight.value, true);
        visitor.parameter("weight", &self.other.value, true);
        Ok(())
    }
    fn visit_native_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        visitor(MlxTensor::ref_cast(&self.weight.value));
        visitor(MlxTensor::ref_cast(&self.other.value));
        true
    }
}
#[test]
fn repeated_physical_key_cannot_hide_a_missing_topology_member() {
    let native = RepeatedKey {
        weight: norm().weight,
        other: norm().weight,
    };
    let topology = BTreeMap::from([
        ("weight".into(), ParameterSpec::trainable("w").unwrap()),
        ("other".into(), ParameterSpec::trainable("o").unwrap()),
    ]);
    let mut rows = Rows::default();
    assert!(matches!(
        visit_module_parameter_sources(&native, &topology, &mut rows),
        Err(ParameterSourceError::TopologyMismatch { .. })
    ));
    assert!(rows.values.is_empty());
    assert!(rows.auxiliary.is_empty());
}
#[test]
fn callback_unwind_keeps_actual_source_and_existing_runtime_owner() {
    struct Panics;
    impl<'a> ParameterSourceVisitor<'a, MlxTensor> for Panics {
        fn parameter(&mut self, _: ParameterMetadataView<'a>, _: &'a MlxTensor) {
            std::panic::panic_any(73_u32);
        }
        fn retained(&mut self, _: &'a MlxTensor) {
            unreachable!();
        }
    }
    let native = norm();
    let topology = BTreeMap::from([("weight".into(), ParameterSpec::trainable("w").unwrap())]);
    let mut runtime = safemlx::RuntimeCallDeadline::new(std::time::Duration::from_secs(5))
        .unwrap()
        .enter()
        .unwrap();
    let before = runtime.descriptor(&native.weight.value).unwrap().facts();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        visit_module_parameter_sources(&native, &topology, &mut Panics)
    }));
    assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 73);
    assert_eq!(
        runtime.descriptor(&native.weight.value).unwrap().facts(),
        before
    );
}

#[test]
fn weightless_normalization_is_complete_only_with_empty_topology() {
    let mut norm = MlxRmsNorm {
        groups: None,
        module: None,
        topology: BTreeMap::new(),
        offset: None,
        dimensions: 2,
        epsilon: 1e-5,
    };
    let mut rows = Rows::default();
    norm.visit_parameter_sources(&mut rows).unwrap();
    assert!(rows.values.is_empty());
    assert!(rows.auxiliary.is_empty());
    drop(rows);
    norm.topology
        .insert("weight".into(), ParameterSpec::trainable("absent").unwrap());
    let mut rows = Rows::default();
    assert_eq!(
        norm.visit_parameter_sources(&mut rows),
        Err(ParameterSourceError::TopologyMismatch { slot: 0 })
    );
    assert!(rows.values.is_empty());
}
