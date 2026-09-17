//! Entered through the existing genuine Source fixture. The same Tensor formula
//! supplies metadata declarations and actual MLX work; no separate interpreter.
use super::*;
use crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms;
use eredu_nn::{
    workspace::{
        WorkspaceContext, WorkspaceDtype, WorkspaceLayout, WorkspaceOperation,
        WorkspaceOperationKind, WorkspaceTensor,
    },
    Tensor,
};

fn equation<T: Tensor>(
    left: &T,
    right: &T,
    scale: &T,
    context: &T::Context,
) -> Result<T, eredu_nn::Error> {
    left.add(right, context)?.multiply(scale, context)
}

impl PreparedNeuralSubmission {
    pub(crate) fn exercise_pointwise_traversal(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        inputs: &[&MlxTensor; 3],
        stream: &Stream,
        plan: PointwiseSubmissionPlan,
        unsupported: crate::backend::nn::workspace::PointwiseTraversalFactError,
        lazy: &MlxTensor,
    ) -> MlxTensor {
        let prologue = plan.gpu_prologue();
        assert_eq!(prologue.evaluations(), 11); // two operations plus Synchronizer
        assert_eq!(prologue.input_edges(), 13); // 2*6 + one root, not one root per entry
        assert_eq!(prologue.sibling_slots(), 0);
        assert_eq!(prologue.output_array_slots(), 11);
        assert_eq!(prologue.tracer_array_slots(), 13);
        let shape = NeuralSubmissionShape::new(1, 0).unwrap();
        let prepared = Self::try_new(shape, controls.clone()).unwrap();
        let identity = Rc::as_ptr(&prepared.shared);
        let buffer = prepared
            .shared
            .payload
            .borrow()
            .as_ref()
            .unwrap()
            .arrays
            .as_ptr();
        let clone_slots = prepared
            .shared
            .payload
            .borrow()
            .as_ref()
            .unwrap()
            .clone_slots
            .as_ptr();
        // The unsupported cold plan never received or spent this slot.
        assert!(matches!(
            unsupported,
            crate::backend::nn::workspace::PointwiseTraversalFactError::UnknownOperation
        ));
        assert_eq!(Rc::as_ptr(&prepared.shared), identity);
        let failed = match prepared.prepare_pointwise(plan, &inputs[..2], observer, stream) {
            Ok(_) => panic!("wrong input count acquired a prepared traversal"),
            Err(error) => error,
        };
        assert!(matches!(
            failed.cause,
            pointwise::PointwisePreparationCause::ShapeMismatch
        ));
        assert_eq!(Rc::as_ptr(&failed.prepared.shared), identity);
        assert!(failed.prepared.primary.is_some());
        {
            let loan = failed.prepared.shared.payload.borrow();
            let payload = loan.as_ref().unwrap();
            assert!(payload.event.is_none());
            assert!(payload.traversal.is_none());
            assert!(payload.arrays.is_empty());
            assert_eq!(payload.arrays.as_ptr(), buffer);
            assert_eq!(payload.clone_slots.as_ptr(), clone_slots);
        }
        let invalid_inputs = [inputs[0], inputs[1], lazy];
        let failed =
            match failed
                .prepared
                .prepare_pointwise(failed.plan, &invalid_inputs, observer, stream)
            {
                Ok(_) => panic!("lazy input acquired a detached-leaf traversal"),
                Err(error) => error,
            };
        assert!(matches!(
            failed.cause,
            pointwise::PointwisePreparationCause::Native(_)
        ));
        assert_eq!(Rc::as_ptr(&failed.prepared.shared), identity);
        assert!(failed.prepared.primary.is_some());
        {
            let loan = failed.prepared.shared.payload.borrow();
            let payload = loan.as_ref().unwrap();
            assert!(payload.event.is_none() && payload.traversal.is_none());
            assert!(payload.arrays.is_empty());
            assert_eq!(payload.arrays.as_ptr(), buffer);
            assert_eq!(payload.clone_slots.as_ptr(), clone_slots);
        }
        let prepared = failed
            .prepared
            .prepare_pointwise(failed.plan, inputs, observer, stream)
            .unwrap();
        let plan = prepared.layout();
        assert_eq!(plan.record_allocations(), 11);
        assert_eq!(
            plan.limits(),
            safemlx::OperationEvalTraversalLimits {
                roots: 1,
                arrays: 14,
                tape_entries: 11,
                input_edges: 13,
                output_slots: 11,
                streams: 1,
                captures: 8,
            }
        );
        let values = prepared.inputs();
        let result = equation(values[0], values[1], values[2], prepared.stream()).unwrap();
        prepared.submit(&[&result]).unwrap().finish().unwrap();
        result
    }

    pub(crate) fn pointwise_declarations() -> (
        PointwiseSubmissionPlan,
        crate::backend::nn::workspace::PointwiseTraversalFactError,
    ) {
        let mechanisms = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(mechanisms);
        let left = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 1], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let right = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[1, 3], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let scale = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 3], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_span();
        let result = equation(&left, &right, &scale, &context).unwrap();
        let report = context.report(&[result]).unwrap();
        assert_eq!(report.operations.len(), 2);
        let shape = NeuralSubmissionShape::new(1, 0).unwrap();
        let plan = PointwiseSubmissionPlan::prepare(
            shape,
            3,
            report.operations.iter().map(WorkspaceOperation::as_view),
            &mechanisms,
        )
        .unwrap();
        let ordinary = Self::control_bytes(shape).unwrap();
        let layout = plan.layout();
        let graph = plan.graph_layout();
        assert_eq!(graph.operations(), 2);
        assert_eq!(graph.maximum_rank(), 2);
        assert_eq!(graph.blocks(), 44);
        let base = safemlx::OperationEvent::eval_record_layout(11, 1, 11).unwrap();
        assert_eq!(
            plan.control_bytes(),
            ordinary
                + u64::try_from(
                    Self::pointwise_control_bytes().unwrap()
                        + base.query_control_bytes().unwrap()
                        + layout.query_control_bytes().unwrap()
                        + graph.control_bytes().unwrap()
                )
                .unwrap()
        );
        assert!(plan.control_bytes() > ordinary);
        let unsupported = WorkspaceOperation {
            kind: WorkspaceOperationKind::Elementwise("square"),
            inputs: vec![WorkspaceLayout::new(&[2, 1], WorkspaceDtype::Float32).unwrap()],
            outputs: vec![WorkspaceLayout::new(&[2, 1], WorkspaceDtype::Float32).unwrap()],
        };
        let refused =
            PointwiseSubmissionPlan::prepare(shape, 3, [unsupported.as_view()], &mechanisms)
                .unwrap_err();
        (plan, refused)
    }
}
