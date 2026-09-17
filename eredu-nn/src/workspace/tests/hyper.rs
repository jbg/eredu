use super::*;
use crate::{
    HyperConnectionOperator, HyperConnectionSpec, HyperHeadOperator, HyperHeadSpec,
    HyperNeuralBackend, TensorValueObserver,
};
fn connection(streams: i32, hidden_size: i32) -> HyperConnectionSpec {
    HyperConnectionSpec {
        streams,
        hidden_size,
        sinkhorn_iterations: 7,
        epsilon: 1e-6,
        function: ParameterSpec::trainable("mix.function").unwrap(),
        base: ParameterSpec::trainable("mix.base").unwrap(),
        scale: ParameterSpec::trainable("mix.scale").unwrap(),
    }
}
fn head(streams: i32, hidden_size: i32) -> HyperHeadSpec {
    HyperHeadSpec {
        streams,
        hidden_size,
        norm_epsilon: 1e-5,
        epsilon: 1e-6,
        function: ParameterSpec::trainable("head.function").unwrap(),
        base: ParameterSpec::trainable("head.base").unwrap(),
        scale: ParameterSpec::trainable("head.scale").unwrap(),
    }
}
#[test]
fn multi_stream_state_and_parameters_retain_the_full_geometry_across_residual_cycles() {
    for streams in [1, 3, 4] {
        for tokens in [1, 7] {
            let context = context();
            let mut mixing =
                WorkspaceBackend::hyper_connection(connection(streams, 32), &context).unwrap();
            let mut head = WorkspaceBackend::hyper_head(head(streams, 32), &context).unwrap();
            let mut residual = existing_f32(&[2, tokens, streams, 32], &context).unwrap();
            context.begin_span();
            for _ in 0..2 {
                let state = mixing.collapse(&residual, 1e-5, &context).unwrap();
                assert_eq!(state.collapsed.shape(), [2, tokens, 32]);
                assert_eq!(state.pre.shape(), [2, tokens, streams]);
                assert_eq!(state.post.shape(), [2, tokens, streams]);
                assert_eq!(state.combination.shape(), [2, tokens, streams, streams]);
                let sublayer = state.collapsed.square(&context).unwrap();
                residual = mixing
                    .expand(&sublayer, &residual, &state, &context)
                    .unwrap();
            }
            let output = head.forward(&residual, &context).unwrap();
            assert_eq!(output.shape(), [2, tokens, 32]);
            let report = context.report(&[]).unwrap();
            let collapse = &report.operations[0];
            assert_eq!(
                collapse.inputs[1].shape(),
                [(streams + 2) * streams, streams * 32]
            );
            assert_eq!(collapse.inputs[2].shape(), [(streams + 2) * streams]);
            assert_eq!(collapse.inputs[3].shape(), [3]);
            assert!(
                matches!(&collapse.kind,WorkspaceOperationKind::HyperCollapse(s,e) if s.streams==streams && s.sinkhorn_iterations==7 && *e==1e-5)
            );
            let coefficients = &report.operations[6];
            assert_eq!(coefficients.inputs[1].shape(), [streams, streams * 32]);
            assert_eq!(coefficients.inputs[2].shape(), [streams]);
            assert_eq!(coefficients.inputs[3].shape(), [1]);
            assert_eq!(report.operations.len(), 8);
        }
    }
}
struct Coefficients {
    calls: usize,
    fail: bool,
}
impl TensorValueObserver<WorkspaceTensor> for Coefficients {
    fn observe_generated(
        &mut self,
        _: &WorkspaceTensor,
        _: &crate::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        panic!("hyper head borrows ordinary coefficients");
    }
    fn observe(&mut self, values: &WorkspaceTensor) -> Result<(), Error> {
        assert_eq!(values.shape(), [2, 7, 3]);
        self.calls += 1;
        if self.fail {
            Err(Error::backend("coefficient observer failed"))
        } else {
            Ok(())
        }
    }
}
#[test]
fn coefficient_observer_precedes_sum_and_invalid_geometry_precedes_all_work() {
    for fail in [false, true] {
        let context = context();
        let mut head = WorkspaceBackend::hyper_head(head(3, 32), &context).unwrap();
        let input = existing_f32(&[2, 7, 3, 32], &context).unwrap();
        context.begin_span();
        let mut observer = Coefficients { calls: 0, fail };
        assert_eq!(
            head.forward_with_coefficients_observer(&input, &context, Some(&mut observer))
                .is_err(),
            fail
        );
        assert_eq!(observer.calls, 1);
        assert_eq!(
            context.report(&[]).unwrap().operations.len(),
            if fail { 1 } else { 2 }
        );
        context.begin_span();
        let invalid = existing_f32(&[2, 7, 4, 32], &context).unwrap();
        assert!(head
            .forward_with_coefficients_observer(&invalid, &context, Some(&mut observer))
            .is_err());
        let mut mixing = WorkspaceBackend::hyper_connection(connection(3, 32), &context).unwrap();
        context.begin_span();
        assert!(mixing.collapse(&invalid, 1e-5, &context).is_err());
        assert!(mixing.collapse(&input, f32::NAN, &context).is_err());
        assert!(WorkspaceBackend::hyper_connection(connection(i32::MAX, 1), &context).is_err());
        assert!(WorkspaceBackend::hyper_head(super::hyper::head(3, i32::MAX), &context).is_err());
        assert!(context.report(&[]).unwrap().operations.is_empty());
        assert_eq!(observer.calls, 1);
    }
}
#[test]
fn mixed_trace_hyper_state_is_rejected_before_expansion() {
    let context = context();
    let other = WorkspaceContext::new(AllocatingMechanism);
    let mut mixing = WorkspaceBackend::hyper_connection(connection(3, 32), &context).unwrap();
    let input = existing_f32(&[2, 1, 3, 32], &context).unwrap();
    context.begin_span();
    let mut state = mixing.collapse(&input, 1e-5, &context).unwrap();
    state.post = existing_f32(&[2, 1, 3], &other).unwrap();
    assert!(mixing
        .expand(&state.collapsed, &input, &state, &context)
        .is_err());
    assert_eq!(context.report(&[]).unwrap().operations.len(), 1);
}
