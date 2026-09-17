use super::*;
use eredu_nn::{
    BlockFp8InputReconstructionPlan, GeneratedTensorProgram, GeneratedTensorSourceRole,
    RetainedGeneratedTensorFactory,
};
use std::rc::Rc;
#[derive(Debug)]
struct Original(Rc<u32>);
struct Factory {
    inputs: [i32; 2],
    calls: usize,
}
impl RetainedGeneratedTensorFactory<i32, u32> for Factory {
    fn program(&self) -> GeneratedTensorProgram<'_> {
        GeneratedTensorProgram::BlockFp8Input(
            BlockFp8InputReconstructionPlan::new(&[1, 4]).unwrap(),
        )
    }
    fn visit_sources(
        &mut self,
        sink: &mut dyn FnMut(GeneratedTensorSourceRole, &i32) -> Result<(), u32>,
    ) -> Result<(), u32> {
        sink(GeneratedTensorSourceRole::CompactValues, &self.inputs[0])?;
        sink(GeneratedTensorSourceRole::BlockScales, &self.inputs[1])
    }
    fn generate(&mut self, sink: &mut dyn FnMut(&i32) -> Result<(), u32>) -> Result<i32, u32> {
        self.calls += 1;
        sink(&7)?;
        Ok(7)
    }
}
struct Observer {
    failure: Rc<u32>,
    selected: bool,
    roots: Vec<i32>,
    addresses: Vec<usize>,
}
impl ActivationObserver<i32, Original> for Observer {
    fn requires_prepared_traversal(&self) -> bool {
        true
    }
    fn requires_sequence_readout(&self) -> bool {
        false
    }
    fn transactional(&self) -> bool {
        true
    }
    fn observe(&mut self, _: &str, _: &i32) -> Result<(), Original> {
        panic!("retained callback must survive both wrappers")
    }
    fn observe_generated_retained(
        &mut self,
        path: &str,
        _: &i32,
        _: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<i32, Original>,
    ) -> Result<(), Original> {
        assert_eq!(path, "projection.input");
        if !self.selected {
            return Ok(());
        }
        factory.visit_sources(&mut |_, v| {
            self.addresses.push(v as *const _ as usize);
            self.roots.push(*v);
            Ok(())
        })?;
        factory.generate(&mut |v| {
            self.roots.push(*v);
            Err(Original(self.failure.clone()))
        })?;
        Ok(())
    }
}
#[test]
fn retained_factory_survives_borrowed_and_error_bridges_with_original_sink_failure() {
    for selected in [false, true] {
        let failure = Rc::new(73);
        let mut observer = Observer {
            failure: failure.clone(),
            selected,
            roots: vec![],
            addresses: vec![],
        };
        let mut factory = Factory {
            inputs: [3, 5],
            calls: 0,
        };
        let addresses = factory
            .inputs
            .iter()
            .map(|v| v as *const _ as usize)
            .collect::<Vec<_>>();
        let source = eredu_core::capture::GeneratedCaptureSource {
            creation_bytes: 32,
            source_dtype: None,
        };
        {
            let mut borrowed = BorrowedActivationObserver(&mut observer);
            let mut bridge = ObserverErrorBridge::new(
                &mut borrowed,
                |n| Original(Rc::new(n)),
                |_: &Original| 91u32,
            );
            assert!(bridge.requires_prepared_traversal());
            assert!(!bridge.requires_sequence_readout());
            assert!(bridge.transactional());
            let result =
                bridge.observe_generated_retained("projection.input", &0, &source, &mut factory);
            if selected {
                assert_eq!(result, Err(91));
                let error = bridge.resolve(Ok(())).unwrap_err();
                assert!(Rc::ptr_eq(&error.0, &failure));
            } else {
                assert!(result.is_ok());
                assert!(bridge.resolve(Ok(())).is_ok());
            }
        }
        assert_eq!(factory.calls, usize::from(selected));
        assert_eq!(
            observer.roots,
            if selected { vec![3, 5, 7] } else { vec![] }
        );
        assert_eq!(
            observer.addresses,
            if selected { addresses } else { vec![] }
        );
    }
}
