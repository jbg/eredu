use super::*;
use eredu_runtime::{ArchitectureParameters, StaticParameterVisitorMut};
use std::convert::Infallible;

struct Populate;
impl<'a> eredu_nn::ParameterVisitorMut<'a, MlxTensor> for Populate {
    fn visit_mut(&mut self, _: eredu_nn::ParameterMetadataView<'_>, value: &'a mut MlxTensor) {
        let shape = value.as_array().shape().to_vec();
        let count = shape.iter().map(|&n| n as usize).product::<usize>();
        assert!(count > 0);
        *value = MlxTensor::from_array(Array::from_slice(&vec![1.25_f32; count], &shape));
    }
}
impl StaticParameterVisitorMut<MlxNeuralBackend> for Populate {
    type Error = Infallible;
    fn visit_mut<M: Parameterized<MlxTensor>>(
        &mut self,
        _: &str,
        module: &mut M,
    ) -> Result<(), Self::Error> {
        module.visit_parameters_mut(self);
        Ok(())
    }
}

#[test]
fn actual_tied_and_untied_dense_modules_have_finite_bounds_with_absent_optional_children() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for tied in [false, true] {
        let args = eredu_architectures::llama::model_args_from_config_value(&serde_json::json!({
            "model_type":"llama", "hidden_size":8, "intermediate_size":16,
            "num_hidden_layers":1, "num_attention_heads":4, "num_key_value_heads":2,
            "head_dim":2, "vocab_size":16, "rms_norm_eps":1e-5,
            "max_position_embeddings":32, "tie_word_embeddings":tied
        }))
        .unwrap();
        let mut model =
            eredu_architectures::llama::LayeredModel::<MlxNeuralBackend>::new(args, &stream)
                .unwrap();
        model.visit_static_parameters_mut(&mut Populate).unwrap();
        let mut unit = model.construct_unit(0, &stream).unwrap();
        unit.visit_parameters_mut(&mut Populate);
        let static_bound = model
            .retained_static_value_slot_bound()
            .expect("actual static topology");
        let unit_bound = unit
            .retained_value_slot_bound()
            .expect("actual unit topology");
        let mut statics = 0;
        assert!(model.visit_retained_static_values(&mut |value| {
            statics += 1;
            assert!(value
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v != 0.0));
        }));
        let mut units = 0;
        assert!(unit.visit_retained_values(&mut |value| {
            units += 1;
            assert!(value
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v != 0.0));
        }));
        assert!(statics > 0 && statics <= static_bound);
        assert!(units > 0 && units <= unit_bound);
    }
}
