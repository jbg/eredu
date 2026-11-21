//! Passive recording of observed tensor handles.

/// A cloned activation captured by [`ActivationRecorder`].
#[derive(Debug, Clone)]
pub struct RecordedActivation<T> {
    /// Stable path-like name of the tensor within the model forward pass.
    pub name: String,
    /// Retained handle for the observed tensor.
    pub value: T,
}

/// Records cloned tensor handles without materializing tensor data.
#[derive(Debug, Clone)]
pub struct ActivationRecorder<T> {
    activations: Vec<RecordedActivation<T>>,
}

impl<T> ActivationRecorder<T> {
    /// Creates an empty recorder.
    pub fn new() -> Self {
        Self {
            activations: Vec::new(),
        }
    }

    /// Returns the recorded activations.
    pub fn activations(&self) -> &[RecordedActivation<T>] {
        &self.activations
    }

    /// Consumes the recorder and returns the recorded activations.
    pub fn into_activations(self) -> Vec<RecordedActivation<T>> {
        self.activations
    }

    /// Removes all recorded activations.
    pub fn clear(&mut self) {
        self.activations.clear();
    }
}

impl<T: Clone, E> super::ActivationObserver<T, E> for ActivationRecorder<T> {
    fn observe(&mut self, name: &str, value: &T) -> Result<(), E> {
        self.activations.push(RecordedActivation {
            name: name.to_string(),
            value: value.clone(),
        });
        Ok(())
    }
}

impl<T> Default for ActivationRecorder<T> {
    fn default() -> Self {
        Self::new()
    }
}
