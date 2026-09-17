use super::*;
use eredu_nn::{
    workspace::*, BlockFp8InputReconstructionPlan, Error, GeneratedTensorProgram,
    GeneratedTensorSourceRole, RetainedGeneratedTensorFactory, Tensor,
};
#[derive(Debug)]
struct Unknown;
impl WorkspaceMechanisms for Unknown {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}
// A borrowed protocol spy: no operations or allocation/funding proof is claimed.
struct Factory<'a> {
    input: &'a WorkspaceTensor,
    calls: usize,
}
impl RetainedGeneratedTensorFactory<WorkspaceTensor, Error> for Factory<'_> {
    fn program(&self) -> GeneratedTensorProgram<'_> {
        GeneratedTensorProgram::BlockFp8Input(
            BlockFp8InputReconstructionPlan::new(self.input.shape()).unwrap(),
        )
    }
    fn visit_sources(
        &mut self,
        sink: &mut dyn FnMut(GeneratedTensorSourceRole, &WorkspaceTensor) -> Result<(), Error>,
    ) -> Result<(), Error> {
        sink(GeneratedTensorSourceRole::CompactValues, self.input)
    }
    fn generate(
        &mut self,
        sink: &mut dyn FnMut(&WorkspaceTensor) -> Result<(), Error>,
    ) -> Result<WorkspaceTensor, Error> {
        self.calls += 1;
        sink(self.input)?;
        Ok(self.input.clone())
    }
}
struct Observer {
    path: usize,
    source: usize,
    events: Vec<&'static str>,
}
impl ActivationObserver<WorkspaceTensor, Error> for Observer {
    fn observe(&mut self, _: &str, _: &WorkspaceTensor) -> Result<(), Error> {
        panic!("retained callback erased")
    }
    fn observe_generated_retained(
        &mut self,
        path: &str,
        _: &WorkspaceTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        assert_eq!(path.as_ptr() as usize, self.path);
        factory.visit_sources(&mut |_, value| {
            assert_eq!(value as *const _ as usize, self.source);
            self.events.push("source");
            Ok(())
        })?;
        factory.generate(&mut |_| {
            self.events.push("output");
            Ok(())
        })?;
        Ok(())
    }
}
#[test]
fn prepared_borrowed_hook_preserves_factory_and_original_semantic_path() {
    let context = WorkspaceContext::new(Unknown);
    let input = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[1, 4], WorkspaceDtype::Uint8).unwrap(),
        &context,
    )
    .unwrap();
    let paths = source();
    let path = &paths.0.payload[0].units[0].input;
    let mut observer = Observer {
        path: path.as_ptr() as usize,
        source: &input as *const _ as usize,
        events: vec![],
    };
    let mut factory = Factory {
        input: &input,
        calls: 0,
    };
    let mut hook = BorrowedHook {
        observer: &mut observer,
        paths: &paths,
    };
    <BorrowedHook<'_,Observer> as LayeredTraversalHook<WorkspaceBackend,(),Error>>::observe_generated_activation_retained(
        &mut hook,path,&input,&eredu_core::capture::GeneratedCaptureSource{creation_bytes:32,source_dtype:None},&mut factory,
    ).unwrap();
    assert_eq!(factory.calls, 1);
    assert_eq!(observer.events, ["source", "output"]);
}
