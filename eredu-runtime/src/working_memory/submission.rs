//! Immediate host-metadata completion for the ordinary runtime traversal.
//! This is never evidence of native execution: WorkspaceBackend creates only
//! descriptors, and unknown allocation facts remain unknown after submission.

use crate::SubmissionBackend;
use eredu_core::{BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion};
use eredu_nn::{
    workspace::{WorkspaceBackend, WorkspaceContext, WorkspaceTensor},
    Error,
};

/// Completion of synchronous metadata recording, tied to its exact trace.
/// It owns no native device, queue, event or completion object and grants no
/// inference authority. The runtime can therefore inspect the same traversal
/// it later uses for execution without introducing a second equation driver.
#[derive(Debug, Clone)]
pub struct WorkspaceCompletion {
    context: WorkspaceContext,
}

impl WorkspaceCompletion {
    pub(super) fn recorded(context: &WorkspaceContext) -> Result<Self, Error> {
        context.charge_metadata(std::mem::size_of::<(Self, Result<Self, Error>, &WorkspaceContext)>())?;
        Ok(Self { context: context.clone() })
    }
}

impl Completion for WorkspaceCompletion {
    type Error = Error;
    fn is_complete(&self) -> Result<bool, Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl BoundedCompletion for WorkspaceCompletion {
    fn wait_bounded(self, _: BoundedCompletionWait) -> Result<BoundedCompletionOutcome, Error> {
        Ok(BoundedCompletionOutcome::Completed)
    }
}

impl SubmissionBackend for WorkspaceBackend {
    type Executor = WorkspaceContext;
    type OwnedExecutor = WorkspaceContext;
    type Completion = WorkspaceCompletion;

    fn fork_executors(
        executor: &WorkspaceContext,
        count: usize,
    ) -> Result<Vec<WorkspaceContext>, Error> {
        executor.charge_metadata(std::mem::size_of::<(
            &WorkspaceContext, usize, Vec<WorkspaceContext>, Result<Vec<WorkspaceContext>,Error>,
        )>())?;
        let mut values=executor.metadata_vec(count)?;
        values.resize_with(count,||executor.clone());
        Ok(values)
    }

    fn submit<'a, I>(executor: &WorkspaceContext, values: I) -> Result<WorkspaceCompletion, Error>
    where
        I: IntoIterator<Item = &'a WorkspaceTensor>,
    {
        executor.validate_values(values)?;
        Ok(WorkspaceCompletion {
            context: executor.clone(),
        })
    }

    fn order_after(
        completion: &WorkspaceCompletion,
        executor: &WorkspaceContext,
    ) -> Result<(), Error> {
        if !executor.shares_trace(&completion.context) {
            return Err(Error::backend(
                "workspace completion belongs to another equation trace",
            ));
        }
        Ok(())
    }

    fn retain_until_complete<T: Send + 'static>(
        executor: &WorkspaceContext,
        completion: &WorkspaceCompletion,
        value: T,
    ) -> Result<(), Error> {
        Self::order_after(completion, executor)?;
        // Host metadata recording is already finished. Dropping this payload
        // does not reset the shared allocation ledger or its retained roots.
        drop(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::{workspace::*, Tensor};

    // Submission tests begin with resident input storage. Their spans cover
    // the issued equations and completion, not parameter construction.
    fn existing_f32(shape: &[i32], context: &WorkspaceContext) -> WorkspaceTensor {
        WorkspaceTensor::existing(
            WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
            context,
        )
        .unwrap()
    }

    #[derive(Debug)]
    struct Facts;
    impl WorkspaceMechanisms for Facts {
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 5,
                assumptions:
                    "test mechanism reserves five disjoint host staging bytes per operation".into(),
            }))
        }

        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(Some(WorkspaceOperationBound {
                outputs: operation
                    .outputs
                    .iter()
                    .map(|output| output.bytes().map(WorkspaceOutputStorage::Allocate))
                    .collect::<Result<_, _>>()?,
                scratch_bytes: 7,
                assumptions: "test mechanism: independent output and seven scratch bytes".into(),
            }))
        }
    }

    #[test]
    fn metadata_lanes_share_charges_and_completion_never_refunds_the_span() {
        let context = WorkspaceContext::new(Facts);
        let forks = WorkspaceBackend::fork_executors(&context, 2).unwrap();
        let input = existing_f32(&[2, 3], &context);
        let left = input.square(&forks[0]).unwrap();
        let right = input.square(&forks[1]).unwrap();
        let completion = WorkspaceBackend::submit(&forks[1], [&left, &right]).unwrap();
        assert!(completion.is_complete().unwrap());
        assert!(completion.resources_releasable());
        WorkspaceBackend::order_after(&completion, &forks[0]).unwrap();
        WorkspaceBackend::retain_until_complete(&context, &completion, vec![17_u8]).unwrap();
        for lane in [&context, &forks[0], &forks[1]] {
            let report = lane.report(&[left.clone(), right.clone()]).unwrap();
            assert_eq!(report.total_bytes, Some(72));
            assert_eq!(report.retained_bytes, Some(48));
            assert_eq!(report.transient_bytes, Some(24));
        }
        drop(completion);
        assert_eq!(context.report(&[]).unwrap().total_bytes, Some(72));
        context.begin_span();
        assert_eq!(
            forks[0].report(&[left, right]).unwrap().total_bytes,
            Some(0)
        );
    }

    #[test]
    fn metadata_submission_cannot_import_dependencies_or_completion_from_another_trace() {
        let first = WorkspaceContext::new(Facts);
        let second = WorkspaceContext::new(Facts);
        let value = existing_f32(&[3], &first);
        assert!(WorkspaceBackend::submit(&second, [&value]).is_err());
        let completion = WorkspaceBackend::submit(&first, [&value]).unwrap();
        assert!(WorkspaceBackend::order_after(&completion, &second).is_err());
        assert!(WorkspaceBackend::retain_until_complete(&second, &completion, ()).is_err());
        assert!(second.report(&[]).unwrap().operations.is_empty());
    }

    #[test]
    fn successful_metadata_completion_does_not_upgrade_unknown_native_workspace() {
        #[derive(Debug)]
        struct Unknown;
        impl WorkspaceMechanisms for Unknown {
            fn host_workspace_bound(
                &self,
                _: &WorkspaceOperation,
            ) -> Result<Option<WorkspaceHostBound>, Error> {
                Ok(Some(WorkspaceHostBound {
                    bytes: 0,
                    assumptions: "test mechanism has no disjoint host workspace".into(),
                }))
            }

            fn operation_bound(
                &self,
                _: &WorkspaceOperation,
            ) -> Result<Option<WorkspaceOperationBound>, Error> {
                Ok(None)
            }
        }
        let context = WorkspaceContext::new(Unknown);
        let input = existing_f32(&[2, 3], &context);
        let output = input.square(&context).unwrap();
        let completion = WorkspaceBackend::submit(&context, [&output]).unwrap();
        completion.wait().unwrap();
        assert!(completion.resources_releasable());
        let report = context.report(&[output]).unwrap();
        assert_eq!(report.total_bytes, None);
        assert_eq!(report.transient_bytes, None);
        assert_eq!(report.unpriced_operations, [0]);
    }
}
