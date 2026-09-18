use super::*;
use common::linear::NativeProjectionInputObserver;
use eredu_nn::{
    BlockFp8InputReconstructionPlan, GeneratedTensorProgram, GeneratedTensorSourceRole,
    RetainedGeneratedTensorFactory,
};
use std::{error::Error as _, sync::Arc};
#[derive(Debug)]
struct Sentinel(Arc<()>);
impl std::fmt::Display for Sentinel {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        panic!("original retention error must not be formatted for a propagation signal")
    }
}
impl std::error::Error for Sentinel {}
struct Factory<'a> {
    value: &'a Array,
    scale: &'a Array,
    stream: &'a Stream,
    calls: usize,
}
impl RetainedGeneratedTensorFactory<Array, Exception> for Factory<'_> {
    fn program(&self) -> GeneratedTensorProgram<'_> {
        GeneratedTensorProgram::BlockFp8Input(
            BlockFp8InputReconstructionPlan::new(self.value.shape()).unwrap(),
        )
    }
    fn visit_sources(
        &mut self,
        sink: &mut dyn FnMut(GeneratedTensorSourceRole, &Array) -> Result<(), Exception>,
    ) -> Result<(), Exception> {
        sink(GeneratedTensorSourceRole::CompactValues, self.value)?;
        sink(GeneratedTensorSourceRole::BlockScales, self.scale)
    }
    fn generate(
        &mut self,
        sink: &mut dyn FnMut(&Array) -> Result<(), Exception>,
    ) -> Result<Array, Exception> {
        self.calls += 1;
        let value = self.value.as_dtype(Dtype::Float32, self.stream)?;
        sink(&value)?;
        Ok(value)
    }
}
struct Observer {
    original: Arc<()>,
    roots: Vec<MlxTensor>,
}
impl eredu_nn::ProjectionInputObserver<MlxTensor> for Observer {
    fn observe(&mut self, _: &MlxTensor) -> Result<(), ComputeError> {
        panic!("retained forwarding")
    }
    fn observe_generated(
        &mut self,
        _: &MlxTensor,
        _: &eredu_nn::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<MlxTensor, ComputeError>,
    ) -> Result<(), ComputeError> {
        panic!("factory erased")
    }
    fn observe_generated_retained(
        &mut self,
        _: &MlxTensor,
        _: &eredu_nn::GeneratedTensorSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<MlxTensor, ComputeError>,
    ) -> Result<(), ComputeError> {
        factory.visit_sources(&mut |_, value| {
            self.roots.push(value.clone());
            Ok(())
        })?;
        factory.generate(&mut |value| {
            self.roots.push(value.clone());
            Err(ComputeError::backend_retained_source(RetainedInputFailure(
                Sentinel(self.original.clone()),
            )))
        })?;
        Ok(())
    }
}
#[test]
fn native_tensor_factory_bridge_keeps_original_retention_error_and_source_aliases() {
    let stream = crate::test_stream();
    let values = Array::from_slice(&[1u8, 2], &[1, 2]);
    let scales = Array::from_slice(&[0.5f32], &[1, 1]);
    let prototype = Array::from_slice(&[1f32, 2.0], &[1, 2]);
    let original = Arc::new(());
    let mut observer = Observer {
        original: original.clone(),
        roots: vec![],
    };
    let mut factory = Factory {
        value: &values,
        scale: &scales,
        stream,
        calls: 0,
    };
    let mut bridge = NativeInputObserver {
        inner: &mut observer,
        failure: None,
    };
    let source = eredu_nn::GeneratedTensorSource {
        creation_bytes: 32,
        element_type: Some(eredu_nn::TensorElementType::F32),
    };
    assert!(bridge
        .observe_generated_retained(&prototype, &source, &mut factory)
        .is_err());
    let error = bridge.failure.take().unwrap();
    assert_eq!(
        error.to_string(),
        "retained generated input observation failed"
    );
    let mut cause = error.source();
    let sentinel = loop {
        let current = cause.expect("typed original retention failure");
        if let Some(sentinel) = current.downcast_ref::<Sentinel>() {
            break sentinel;
        }
        cause = current.source();
    };
    assert!(Arc::ptr_eq(&sentinel.0, &original));
    drop(bridge);
    assert_eq!(factory.calls, 1);
    drop(factory);
    assert_eq!(observer.roots.len(), 3);
    assert_eq!(
        observer.roots[0].as_array().allocation_info().unwrap(),
        values.allocation_info().unwrap()
    );
    drop(values);
    drop(scales);
    drop(prototype);
    let evaluated = observer.roots[2].as_array().evaluated().unwrap();
    assert_eq!(evaluated.as_slice::<f32>(), &[1.0, 2.0]);
}
