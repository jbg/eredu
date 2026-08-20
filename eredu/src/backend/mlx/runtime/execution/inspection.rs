//! MLX recorders and adapters for the backend-neutral inspection contract.

use safemlx::{error::Exception, Array};

pub use eredu_runtime::{NoopObserver, RoutingObservation as MoeRoutingObservation};

/// Adapts a dynamically selected MLX observer to generic instrumented code.
pub(crate) struct ActivationObserverProxy<'a>(
    pub &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
);

impl eredu_runtime::ActivationObserver<Array, Exception> for ActivationObserverProxy<'_> {
    fn observe(&mut self, name: &str, value: &Array) -> Result<(), Exception> {
        self.0.observe(name, value)
    }

    fn intervene(&mut self, name: &str, value: &Array) -> Result<Option<Array>, Exception> {
        self.0.intervene(name, value)
    }

    fn observe_routing(
        &mut self,
        routing: MoeRoutingObservation<'_, Array>,
    ) -> Result<(), Exception> {
        self.0.observe_routing(routing)
    }
}

/// A cloned activation captured by [`ActivationRecorder`].
#[derive(Debug, Clone)]
pub struct RecordedActivation {
    /// Stable path-like name of the tensor within the model forward pass.
    pub name: String,
    /// Lazy MLX array handle for the observed tensor.
    pub value: Array,
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

impl eredu_runtime::ActivationObserver<Array, Exception> for ActivationRecorder {
    fn observe(&mut self, name: &str, value: &Array) -> Result<(), Exception> {
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
    use eredu_runtime::ActivationObserver;
    use safemlx::{error::Exception, Array};

    struct ReplacingObserver {
        intervention_name: Option<String>,
    }

    impl ActivationObserver<Array, Exception> for ReplacingObserver {
        fn observe(&mut self, _name: &str, _value: &Array) -> Result<(), Exception> {
            Ok(())
        }

        fn intervene(&mut self, name: &str, _value: &Array) -> Result<Option<Array>, Exception> {
            self.intervention_name = Some(name.to_string());
            Ok(Some(Array::from_slice(&[9.0f32, 8.0], &[2])))
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn recorder_clones_observed_array_handles() {
        let array = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let mut recorder = ActivationRecorder::new();

        recorder.observe("layer.output", &array).unwrap();

        let activations = recorder.activations();
        assert_eq!(activations.len(), 1);
        assert_eq!(activations[0].name, "layer.output");
        assert_eq!(activations[0].value.shape(), &[2]);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn intervention_defaults_to_passthrough_and_can_replace_an_activation() {
        let array = Array::from_slice(&[1.0f32, 2.0], &[2]);
        assert!(
            <NoopObserver as ActivationObserver<Array, Exception>>::intervene(
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
        assert_eq!(replacement.shape(), &[2]);
        assert_eq!(
            observer.intervention_name.as_deref(),
            Some("model.layers.0.output")
        );
    }
}
