//! Borrowed capture slots retained by the same native recovery payload.
use super::*;
use std::cell::RefCell;

pub(crate) struct Capture {
    paths: Vec<String>,
    values: RefCell<Vec<Option<MlxTensor>>>,
    metadata: Option<WorkspaceContext>,
}
impl Capture {
    pub(crate) fn prepare<A>(
        request: &ExternalPredictionCaptureRequest,
        metadata: Option<&WorkspaceContext>,
    ) -> Result<Self, Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>,
    {
        if let Some(metadata) = metadata {
            metadata
                .charge_metadata(size_of::<(
                    Self,
                    Result<Self, Error>,
                    Option<WorkspaceContext>,
                )>())
                .map_err(|e| Error::Neural(e.into()))?;
        }
        let paths = match metadata {
            Some(metadata) => A::external_prediction_capture_paths_with_metadata(request, metadata),
            None => A::external_prediction_capture_paths(request),
        }?
        .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        if paths.is_empty()
            || paths
                .iter()
                .enumerate()
                .any(|(i, p)| paths[..i].contains(p))
        {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let mut values = match metadata {
            Some(metadata) => metadata.metadata_vec(paths.len())?,
            None => Vec::with_capacity(paths.len()),
        };
        values.resize_with(paths.len(), || None);
        Ok(Self {
            paths,
            values: RefCell::new(values),
            metadata: metadata.cloned(),
        })
    }
    pub(crate) fn observer(&self) -> Observer<'_> {
        Observer(self)
    }
    pub(crate) fn values(&self) -> Result<Vec<MlxTensor>, eredu_nn::Error> {
        let mut values = self.values.borrow_mut();
        let mut result = match &self.metadata {
            Some(metadata) => {
                metadata.charge_metadata(size_of::<(
                    Vec<MlxTensor>,
                    Result<Vec<MlxTensor>, eredu_nn::Error>,
                    std::cell::RefMut<'_, Vec<Option<MlxTensor>>>,
                )>())?;
                metadata.metadata_vec(values.len())?
            }
            None => Vec::with_capacity(values.len()),
        };
        // A failed capture retains every already produced alias in Q.
        if values.iter().any(Option::is_none) {
            return Err(self.error("external target omitted a selected capture path"));
        }
        for value in values.iter_mut() {
            result.push(value.take().expect("complete capture checked"));
        }
        Ok(result)
    }
    fn error(&self, message: &'static str) -> eredu_nn::Error {
        match &self.metadata {
            Some(metadata) => metadata.metadata_error(format_args!("{message}")),
            None => eredu_nn::Error::backend(message),
        }
    }
}
pub(crate) struct Observer<'a>(&'a Capture);
impl ActivationObserver<MlxTensor, eredu_nn::Error> for Observer<'_> {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn requires_sequence_readout(&self) -> bool {
        self.0
            .paths
            .iter()
            .any(|p| p == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
    }
    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), eredu_nn::Error> {
        if let Some(index) = self.0.paths.iter().position(|p| p == path) {
            let mut values = self.0.values.borrow_mut();
            if values[index].is_some() {
                return Err(self.0.error("duplicate external capture path"));
            }
            if let Some(metadata) = &self.0.metadata {
                metadata.charge_metadata(size_of::<Option<MlxTensor>>())?;
            }
            // The same clone is present in the ordinary workspace equation;
            // its original native handle is charged by that active scope.
            values[index] = Some(MlxTensor::from_array(
                value
                    .as_array()
                    .try_clone_handle()
                    .map_err(eredu_nn::Error::backend_retained_source)?,
            ));
        }
        Ok(())
    }
    fn observe_generated(
        &mut self,
        path: &str,
        _: &MlxTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<MlxTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        if self.0.paths.iter().any(|p| p == path) {
            self.observe(path, &generate()?)?;
        }
        Ok(())
    }
}
