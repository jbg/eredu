use super::NeutralActivationObserver;
use crate::{backend::error::Error as MlxError, MlxTensor};
use eredu_nn::{
    BlockFp8InputReconstructionPlan, GeneratedTensorProgram, GeneratedTensorSourceRole,
    RetainedGeneratedTensorFactory,
};
use eredu_runtime::ActivationObserver;
use safemlx::Array;
use std::{fmt, sync::Arc};

#[derive(Debug)]
struct Original(Arc<()>);
impl fmt::Display for Original {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("retention error text must not be duplicated for a propagation signal")
    }
}
impl std::error::Error for Original {}
// Protocol fixture: actual handles and borrowed plan, no numerical quote claim.
struct Factory<'a> {
    values: &'a MlxTensor,
    scales: &'a MlxTensor,
    output: &'a MlxTensor,
    calls: usize,
}
impl RetainedGeneratedTensorFactory<MlxTensor, eredu_nn::Error> for Factory<'_> {
    fn program(&self) -> GeneratedTensorProgram<'_> {
        GeneratedTensorProgram::BlockFp8Input(
            BlockFp8InputReconstructionPlan::new(self.values.as_array().shape()).unwrap(),
        )
    }
    fn visit_sources(
        &mut self,
        retain: &mut dyn FnMut(
            GeneratedTensorSourceRole,
            &MlxTensor,
        ) -> Result<(), eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        retain(GeneratedTensorSourceRole::CompactValues, self.values)?;
        retain(GeneratedTensorSourceRole::BlockScales, self.scales)
    }
    fn generate(
        &mut self,
        retain: &mut dyn FnMut(&MlxTensor) -> Result<(), eredu_nn::Error>,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        self.calls += 1;
        retain(self.output)?;
        Ok(self.output.clone())
    }
}
struct Observer<'a> {
    values: &'a Array,
    scales: &'a Array,
    output: &'a Array,
    roles: Vec<GeneratedTensorSourceRole>,
    outputs: usize,
    selected: bool,
    fail: bool,
    original: Arc<()>,
}
impl ActivationObserver<Array, MlxError> for Observer<'_> {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn transactional(&self) -> bool {
        true
    }
    fn observe(&mut self, _: &str, _: &Array) -> Result<(), MlxError> {
        panic!("bare observation fallback")
    }
    fn observe_generated(
        &mut self,
        _: &str,
        _: &Array,
        _: &eredu_core::capture::GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<Array, MlxError>,
    ) -> Result<(), MlxError> {
        panic!("bare factory fallback loses retained source protocol")
    }
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &Array,
        _: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<Array, MlxError>,
    ) -> Result<(), MlxError> {
        assert_eq!(path, "projection.input");
        assert!(std::ptr::eq(prototype, self.output));
        if !self.selected {
            return Ok(());
        }
        let GeneratedTensorProgram::BlockFp8Input(plan) = factory.program();
        assert_eq!(plan.shape(), &[1, 2]);
        factory.visit_sources(&mut |role, value| {
            self.roles.push(role);
            let actual = match role {
                GeneratedTensorSourceRole::CompactValues => self.values,
                GeneratedTensorSourceRole::BlockScales => self.scales,
            };
            assert!(std::ptr::eq(value, actual));
            Ok(())
        })?;
        factory.generate(&mut |value| {
            assert!(std::ptr::eq(value, self.output));
            self.outputs += 1;
            if self.fail {
                Err(MlxError::Other(Box::new(Original(self.original.clone()))))
            } else {
                Ok(())
            }
        })?;
        Ok(())
    }
}
#[test]
fn managed_neutral_adapter_forwards_retained_roots_and_preserves_original_error_without_formatting()
{
    let values = MlxTensor::from_array(Array::from_slice(&[1u8, 2], &[1, 2]));
    let scales = MlxTensor::from_array(Array::from_slice(&[0.5f32], &[1, 1]));
    let output = MlxTensor::from_array(Array::from_slice(&[5.0f32, 7.0], &[1, 2]));
    let source = eredu_core::capture::GeneratedCaptureSource {
        creation_bytes: 32,
        source_dtype: Some(eredu_core::checkpoint::TensorDtype::F32),
    };
    for (selected, fail) in [(false, false), (true, false), (true, true)] {
        let original = Arc::new(());
        let mut observer = Observer {
            values: values.as_array(),
            scales: scales.as_array(),
            output: output.as_array(),
            roles: vec![],
            outputs: 0,
            selected,
            fail,
            original: original.clone(),
        };
        let mut factory = Factory {
            values: &values,
            scales: &scales,
            output: &output,
            calls: 0,
        };
        let mut adapter = NeutralActivationObserver::new(&mut observer);
        assert!(adapter.requires_prepared_traversal());
        assert!(!adapter.requires_sequence_readout());
        assert!(adapter.transactional());
        let result =
            adapter.observe_generated_retained("projection.input", &output, &source, &mut factory);
        if fail {
            let error = match result {
                Err(e) => e,
                Ok(()) => panic!("original retention failure must propagate"),
            };
            assert_eq!(
                error.to_string(),
                "retained generated activation observation failed"
            );
            let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
            let sentinel = loop {
                let current = cause.expect("typed original cause");
                if let Some(sentinel) = current.downcast_ref::<Original>() {
                    break sentinel;
                }
                cause = current.source();
            };
            assert!(Arc::ptr_eq(&sentinel.0, &original));
        } else {
            assert!(result.is_ok());
        }
        drop(adapter);
        assert_eq!(factory.calls, usize::from(selected));
        assert_eq!(observer.outputs, usize::from(selected));
        assert_eq!(
            observer.roles,
            if selected {
                vec![
                    GeneratedTensorSourceRole::CompactValues,
                    GeneratedTensorSourceRole::BlockScales,
                ]
            } else {
                vec![]
            }
        );
    }
}
