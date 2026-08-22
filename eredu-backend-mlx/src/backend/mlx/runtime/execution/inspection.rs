//! MLX recorders and adapters for the backend-neutral inspection contract.

use crate::MlxTensor;
use safemlx::error::Exception;

pub use eredu_runtime::{NoopObserver, RoutingObservation as MoeRoutingObservation};

/// A cloned activation captured by [`ActivationRecorder`].
#[derive(Debug, Clone)]
pub struct RecordedActivation {
    /// Stable path-like name of the tensor within the model forward pass.
    pub name: String,
    /// Lazy MLX tensor handle for the observed tensor.
    pub value: MlxTensor,
}

/// Simple observer that records cloned array handles.
#[derive(Debug, Default, Clone)]
pub struct ActivationRecorder {
    activations: Vec<RecordedActivation>,
}

impl ActivationRecorder {
    /// Creates an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the recorded activations.
    pub fn activations(&self) -> &[RecordedActivation] {
        &self.activations
    }

    /// Consumes the recorder and returns the recorded activations.
    pub fn into_activations(self) -> Vec<RecordedActivation> {
        self.activations
    }

    /// Removes all recorded activations.
    pub fn clear(&mut self) {
        self.activations.clear();
    }
}

impl eredu_runtime::ActivationObserver<MlxTensor, Exception> for ActivationRecorder {
    fn observe(&mut self, name: &str, value: &MlxTensor) -> Result<(), Exception> {
        self.activations.push(RecordedActivation {
            name: name.to_string(),
            value: value.clone(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ActivationRecorder, NoopObserver};
    use crate::MlxTensor;
    use eredu_runtime::ActivationObserver;
    use safemlx::{error::Exception, Array};

    struct ReplacingObserver {
        intervention_name: Option<String>,
    }

    impl ActivationObserver<MlxTensor, Exception> for ReplacingObserver {
        fn observe(&mut self, _name: &str, _value: &MlxTensor) -> Result<(), Exception> {
            Ok(())
        }

        fn intervene(
            &mut self,
            name: &str,
            _value: &MlxTensor,
        ) -> Result<Option<MlxTensor>, Exception> {
            self.intervention_name = Some(name.to_string());
            Ok(Some(MlxTensor::from_array(Array::from_slice(
                &[9.0f32, 8.0],
                &[2],
            ))))
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn recorder_clones_observed_array_handles() {
        let array = MlxTensor::from_array(Array::from_slice(&[1.0f32, 2.0], &[2]));
        let mut recorder = ActivationRecorder::new();

        recorder.observe("layer.output", &array).unwrap();

        let activations = recorder.activations();
        assert_eq!(activations.len(), 1);
        assert_eq!(activations[0].name, "layer.output");
        assert_eq!(activations[0].value.as_array().shape(), &[2]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn intervention_defaults_to_passthrough_and_can_replace_an_activation() {
        let array = MlxTensor::from_array(Array::from_slice(&[1.0f32, 2.0], &[2]));
        assert!(
            <NoopObserver as ActivationObserver<MlxTensor, Exception>>::intervene(
                &mut NoopObserver,
                "model.layers.0.output",
                &array,
            )
            .unwrap()
            .is_none()
        );

        let mut observer = ReplacingObserver {
            intervention_name: None,
        };
        let replacement = observer
            .intervene("model.layers.0.output", &array)
            .unwrap()
            .unwrap();
        assert_eq!(replacement.as_array().shape(), &[2]);
        assert_eq!(
            observer.intervention_name.as_deref(),
            Some("model.layers.0.output")
        );
    }
}
