//! MLX execution primitives for speculative model sessions.

/// External assistant adapters over neutral family equations and state.
pub mod external;
/// MLX semantic-generation resource adapter over the portable runtime driver.
pub mod scheduler;

pub use scheduler::SpeculativeComponentTimingGuard;

use eredu_core::{
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    CompletionCancellationMode, SamplingPlacement, SpeculativeDraftRandomPosition,
    SpeculativeExecutionTopology, SpeculativeSampling, TokenizerCompatibilityProof,
};
use eredu_runtime::{
    SelectedSpeculativeRealization, SpeculativeMechanism, SpeculativeMechanismCapabilities,
    SpeculativeSampler,
};
use safemlx::{
    error::Exception,
    ops::{indexing::TryIndexOp, maximum, softmax_axis},
    random,
    transforms::{async_eval_with_event, eval},
    Array, Event, Stream,
};

use crate::{
    backend::error::Error,
    backend::nn::shared::{MlxModule, MlxNeuralBackend},
    backend::random::RandomState,
    backend::runtime::generation::MlxSamplingBackend,
    MlxTensor,
};

mod assistant;
pub(super) mod memory_observation;
mod parameter_policy;
pub(crate) mod autoregressive;
mod completion;
mod execution_streams;
mod sampling;
pub(crate) mod state_snapshot;

pub use assistant::MlxDrafter;
pub(crate) use assistant::{
    speculative_mechanism_capabilities, MlxAssistantPreparationVisitor, MlxExternalAssistant,
};
pub use completion::MlxSpeculativeCompletion;
pub use execution_streams::SpeculativeExecutionStreams;
pub(in crate::composition::mlx) use execution_streams::BorrowedTargetObserver;
#[cfg(test)]
use sampling::array_probability_at;
pub(crate) use sampling::validate_control_capture;
pub use sampling::MlxSpeculativeSampling;

#[cfg(test)]
mod test_fixture;
#[cfg(test)]
mod completion_tests;

/// Exercises invocation-group binding against a real selected target in native tests.
#[cfg(test)]
pub(crate) fn verify_retained_assistant_tensor_group(
    session: &mut super::MlxModelSession,
    stream: &Stream,
    rank: eredu_core::ParallelRankTopology,
) {
    use eredu_architectures::external_assistant::{
        ExternalAssistantExecutionMechanisms, MuseAssistantArchitecture,
    };
    use eredu_core::checkpoint::TensorDtype;
    use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
    use eredu_runtime::{
        CommunicationGroupRequirements, CommunicationOperation, CommunicationOperationRequirement,
        CommunicationTensorLimits,
    };
    let requirements = |elements| {
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, elements, None).unwrap(),
            true,
        )
        .unwrap()])
        .unwrap()
    };
    let binding = session.speculative_partition_binding();
    let context =
        SpeculativeExecutionStreams::single(stream).with_capture_binding(binding.as_ref());
    let entered = std::cell::Cell::new(false);
    let refused = session.with_prepared_model_operation(
        |local| {
            let local = local.and_then(|()| {
                if rank.global_rank() == 1 {
                    Err(Error::Speculative(
                        "injected target ownership refusal".into(),
                    ))
                } else {
                    Ok(())
                }
            });
            context.finish_preparation(
                eredu_core::run_preparation::TextPreparationStage::Request,
                local,
                |error| Error::Exception(Exception::from_source(error)),
            )
        },
        |_, ()| {
            entered.set(true);
            Ok(())
        },
    );
    assert!(refused.is_err());
    assert!(!entered.get(), "peer refusal precedes native model access");
    session.ensure_no_submission_in_flight().unwrap();
    let group =
        <external::MlxExternalAssistantMechanisms as ExternalAssistantExecutionMechanisms<
            MuseAssistantArchitecture,
        >>::assistant_parallel_context(context, &requirements(16))
        .unwrap();
    if rank.tensor_parallel_size() == 1 {
        assert!(group.is_none());
    } else {
        let rejected =
            <external::MlxExternalAssistantMechanisms as ExternalAssistantExecutionMechanisms<
                MuseAssistantArchitecture,
            >>::assistant_parallel_context(context, &requirements(usize::MAX));
        assert!(
            rejected.is_err(),
            "assistant bounds must not enlarge the retained target group"
        );
        let group = group.expect("target's retained tensor communicator");
        assert_eq!(
            MlxNeuralBackend::parallel_size(group),
            rank.tensor_parallel_size()
        );
        assert_eq!(
            MlxNeuralBackend::parallel_rank(group),
            rank.tensor_parallel_rank()
        );
        let value = MlxTensor::from_array(Array::from_slice(
            &[rank.tensor_parallel_rank() as f32 + 1.; 16],
            &[1, 1, 16],
        ));
        let reduced = MlxNeuralBackend::sum_parallel(value, group, stream).unwrap();
        let values = reduced.to_f32_vec(stream).unwrap();
        let size = rank.tensor_parallel_size();
        let expected = (size * (size + 1) / 2) as f32;
        assert!(values.iter().all(|value| *value == expected));
    }
}
