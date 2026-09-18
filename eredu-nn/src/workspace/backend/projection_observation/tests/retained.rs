use super::*;
use crate::{GeneratedTensorProgram, GeneratedTensorSourceRole, RetainedGeneratedTensorFactory};
struct Retaining {
    selected: bool,
    stop: Option<usize>,
    roots: Vec<WorkspaceTensor>,
    roles: Vec<GeneratedTensorSourceRole>,
    outputs: usize,
}
impl crate::ProjectionInputObserver<WorkspaceTensor> for Retaining {
    fn observe(&mut self, _: &WorkspaceTensor) -> Result<(), Error> {
        panic!("FP8 generated route")
    }
    fn observe_generated(
        &mut self,
        _: &WorkspaceTensor,
        _: &crate::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        panic!("retained route must not erase factory")
    }
    fn observe_generated_retained(
        &mut self,
        prototype: &WorkspaceTensor,
        _: &crate::GeneratedTensorSource,
        factory: &mut dyn RetainedGeneratedTensorFactory<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        if !self.selected {
            return Ok(());
        }
        let GeneratedTensorProgram::BlockFp8Input(plan) = factory.program();
        assert_eq!(plan.shape(), prototype.shape());
        factory.visit_sources(&mut |role, value| {
            assert!(value.same_context(prototype));
            self.roles.push(role);
            self.roots.push(value.clone());
            Ok(())
        })?;
        let value = factory.generate(&mut |value| {
            self.outputs += 1;
            assert!(value.same_context(prototype));
            self.roots.push(value.clone());
            if self.stop == Some(self.outputs) {
                Err(Error::backend_retained_source(std::io::Error::other(
                    "retained sentinel",
                )))
            } else {
                Ok(())
            }
        })?;
        assert_eq!(value.shape(), prototype.shape());
        assert_eq!(value.layout().dtype(), WorkspaceDtype::Float32);
        Ok(())
    }
}
#[test]
fn workspace_retains_actual_compact_inputs_and_every_output_in_original_context() {
    for selected in [false, true] {
        let context = WorkspaceContext::new(Facts {
            selected: true,
            decode: true,
            host: true,
        });
        let mut linear = linear(&context, true, 259);
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 259], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_state_span([&input]).unwrap();
        let mut observer = Retaining {
            selected,
            stop: None,
            roots: vec![],
            roles: vec![],
            outputs: 0,
        };
        let output = linear
            .forward_with_input_observer(&input, &context, Some(&mut observer))
            .unwrap();
        if selected {
            assert_eq!(
                observer.roles,
                vec![
                    GeneratedTensorSourceRole::CompactValues,
                    GeneratedTensorSourceRole::BlockScales
                ]
            );
            assert_eq!(observer.roots[0].shape(), &[2, 259]);
            assert_eq!(observer.roots[0].layout().dtype(), WorkspaceDtype::Uint8);
            assert_eq!(observer.roots[1].shape(), &[2, 3]);
            assert_eq!(observer.roots[1].layout().dtype(), WorkspaceDtype::Float32);
            assert_eq!(observer.outputs, 7);
        } else {
            assert!(observer.roots.is_empty());
            assert_eq!(observer.outputs, 0);
        }
        observer.roots.push(output);
        let report = context.report(&observer.roots).unwrap();
        assert_eq!(report.operations.len(), if selected { 9 } else { 2 });
        assert!(report.total_bytes.is_some());
        assert!(matches!(
            report.operations.last().unwrap().kind,
            WorkspaceOperationKind::ProjectionFinish(_)
        ));
    }
}
#[test]
fn retention_failure_stops_before_projection_finish_and_keeps_partial_roots() {
    for stop in 1..=7 {
        let context = WorkspaceContext::new(Facts {
            selected: true,
            decode: true,
            host: true,
        });
        let mut linear = linear(&context, true, 259);
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 259], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_state_span([&input]).unwrap();
        let mut observer = Retaining {
            selected: true,
            stop: Some(stop),
            roots: vec![],
            roles: vec![],
            outputs: 0,
        };
        assert!(linear
            .forward_with_input_observer(&input, &context, Some(&mut observer))
            .is_err());
        drop(linear);
        drop(input);
        assert_eq!(observer.outputs, stop);
        assert_eq!(observer.roots.len(), 2 + stop);
        let report = context.report(&observer.roots).unwrap();
        assert_eq!(report.operations.len(), 1 + stop);
        assert!(!report
            .operations
            .iter()
            .any(|op| matches!(op.kind, WorkspaceOperationKind::ProjectionFinish(_))));
        assert!(report.total_bytes.is_some());
    }
}
