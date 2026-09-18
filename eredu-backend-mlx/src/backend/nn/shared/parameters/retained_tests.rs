use super::*;
use crate::module::PhysicalParam;

#[derive(Debug, Clone, eredu_backend_mlx_macros::PhysicalParameters)]
#[module(root = crate)]
struct OpaqueNativeLeaf {
    #[param]
    weight: PhysicalParam<Array>,
    hidden: Array,
}

impl NativeRetainedValues for OpaqueNativeLeaf {
    fn native_parameter_source_count(&self) -> Option<usize> { Some(1) }
    fn visit_native_parameter_sources<'a>(&'a self, visitor: &mut dyn NativeParameterSourceVisitor<'a>) -> Result<(),ParameterSourceError> {
        visitor.parameter("weight", &self.weight.value, true);
        Err(ParameterSourceError::UnclassifiedRetainedField)
    }
    fn visit_native_parameter_sources_mut<'a>(&'a mut self, visitor: &mut dyn NativeParameterSourceVisitorMut<'a>) -> Result<(),ParameterSourceError> {
        visitor.parameter("weight", &mut self.weight.value, true);
        Err(ParameterSourceError::UnclassifiedRetainedField)
    }
}

#[test]
fn retained_wrappers_preserve_incomplete_native_inventory_without_evaluation() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let weight = Array::from_slice(&[2_f32, 3.0], &[2]);
    let hidden = weight.square(&stream).unwrap();
    let module = MlxNamedModule::with_exact_topology(
        OpaqueNativeLeaf {
            weight: PhysicalParam::new(weight.clone()),
            hidden: hidden.clone(),
        },
        [("weight", ParameterSpec::trainable("module.weight").unwrap())],
    )
    .unwrap();
    let wrapped = MlxModule::new(module);
    let before = eredu_nn::validate_parameter_topology(&wrapped).unwrap_err();
    let mut values = Vec::new();
    assert!(!wrapped.visit_retained_values(&mut |value| values.push(value.clone())));
    assert_eq!(values.len(), 1);
    assert_eq!(
        values[0].as_array().allocation_info().unwrap(),
        weight.allocation_info().unwrap()
    );
    assert_eq!(wrapped.inner.inner.hidden.allocation_info().unwrap(), None);
    assert_eq!(hidden.allocation_info().unwrap(), None);
    assert_eq!(
        eredu_nn::validate_parameter_topology(&wrapped).unwrap_err(),
        before
    );
}

mod slot_bounds;

mod topology_bounds;
