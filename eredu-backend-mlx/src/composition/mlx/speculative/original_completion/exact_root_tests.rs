use super::*;
use std::error::Error as _;

impl PreparedOriginalSpeculativeCompletion {
    pub(in crate::composition::mlx) fn exercise_exact_native_root_reserve_failure(
        controls: &OriginalTextControlGuard,
        observer: &OriginalScopeObserver,
        root: &Array,
        stream: &Stream,
        graph_capacity: usize,
    ) -> Error {
        let element = safemlx::OperationEvent::root_storage_layout(1)
            .unwrap()
            .roots_bytes();
        let count = graph_capacity.checked_div(element).unwrap() + 1;
        let fit = Self::native_fit(count).unwrap();
        assert_eq!(fit.roots, count);
        assert_eq!(fit.evaluation_frontiers, 1);
        assert_eq!(fit.consumer_waits, 0);
        assert!(fit.root_storage.roots_bytes() > graph_capacity);
        assert_eq!(fit.root_storage.graph_blocks(), 2);
        assert_eq!(
            fit.outer_array_handle_bytes,
            u64::try_from(count * Array::inspection_clone_handle_bytes()).unwrap()
        );
        assert_eq!(
            fit.non_object_operation_controls
                + u64::try_from(fit.root_storage.object_bytes()).unwrap(),
            u64::try_from(safemlx::OperationEvent::control_bytes().unwrap()).unwrap()
        );
        assert_eq!(fit.missing().len(), 3); // prepared handles do not close Graph/Record fit
        let prepared = Self::try_new(count, controls.clone()).unwrap();
        let error = match prepared.submit(
            std::iter::repeat(root).take(count),
            observer.clone(),
            stream,
        ) {
            Ok(_) => panic!("native root reserve exceeded its actual Graph capacity"),
            Err(error) => error,
        };
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<OnceFailure>()
            .unwrap();
        let OnceCause::Submission(OriginalArraySubmissionCause::Native(cause)) = &source.cause
        else {
            panic!("root reserve lost its actual native cause");
        };
        assert_eq!(
            cause.scoped_evaluation_cause(),
            Some(safemlx::error::ScopedEvaluationCause::Failed)
        );
        assert_eq!(
            cause.source().unwrap().source().unwrap().to_string(),
            "graph metadata capacity exhausted"
        );
        assert!(!observer.status().failed()); // reserve precedes Eval acceptance
        error
    }
}
