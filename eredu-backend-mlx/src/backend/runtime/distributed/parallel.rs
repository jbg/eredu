//! MLX-native sampling and synchronization for selected tensor-parallel plans.
//!
//! Logical rank placement and planning remain in neutral crates; this module
//! only realizes their native operations.

use eredu_runtime::{
    BroadcastBackend, DistributedExecutionPhase, PartitionCommunicationAuthority, Sampler,
};

use safemlx::{
    ops::{indexing::TryIndexOp, ones, zeros},
    Array, Dtype, Stream,
};

use crate::{
    backend::error::Error, backend::runtime::distributed::Group,
    backend::runtime::generation::MlxSamplingBackend, MlxTensor,
};

/// Token selected on one distributed rank together with synchronized stop state.
#[derive(Debug)]
pub struct SynchronizedToken {
    /// Selected token ids with shape `[batch, 1]` on every rank.
    pub token: Array,
    /// Whether every rank should terminate generation.
    pub finished: bool,
}

/// Samples on one designated rank and synchronizes only token ids and stop state.
///
/// `logits` is required on `sampling_rank` and ignored elsewhere. Accepting an
/// optional value lets pipeline stages avoid retaining full logits while TP and
/// EP callers may pass their identical complete logits on every rank.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_and_synchronize_bounded<S: Sampler<MlxSamplingBackend>>(
    logits: Option<&MlxTensor>,
    batch_size: i32,
    sampler: &mut S,
    temperature: f32,
    prng_state: Option<&mut crate::backend::random::RandomState>,
    finished: bool,
    sampling_rank: usize,
    group: &Group,
    authority: &PartitionCommunicationAuthority,
    stream: &Stream,
) -> Result<SynchronizedToken, Error> {
    authority
        .ensure_active()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    if sampling_rank >= group.size() {
        return Err(Error::Parallel(format!(
            "sampling rank {sampling_rank} is outside distributed group size {}",
            group.size()
        )));
    }
    if batch_size <= 0 {
        return Err(Error::Parallel(format!(
            "distributed sampling batch size must be positive, got {batch_size}"
        )));
    }
    let local_token = if group.rank() == sampling_rank {
        let logits = logits.ok_or_else(|| {
            Error::Parallel(format!(
                "sampling rank {sampling_rank} requires complete logits"
            ))
        })?;
        if logits.as_array().dim(0) != batch_size {
            return Err(Error::Parallel(format!(
                "sampling logits batch {} does not match declared batch {batch_size}",
                logits.as_array().dim(0)
            )));
        }
        let logits = if logits.as_array().ndim() == 3 {
            MlxTensor::from_array(logits.as_array().try_index_device((.., -1, ..), stream)?)
        } else {
            logits.clone()
        };
        Sampler::<MlxSamplingBackend>::sample(sampler, &logits, temperature, prng_state, stream)?
            .into_array()
            .reshape(&[batch_size, 1], stream)?
            .as_dtype(Dtype::Float32, stream)?
    } else {
        zeros::<f32>(&[batch_size, 1], stream)?
    };
    let phase = DistributedExecutionPhase::SamplingSynchronization;
    let operation = eredu_runtime::CommunicationOperation::Broadcast;
    let token_submission =
        <crate::backend::nn::shared::MlxNeuralBackend as BroadcastBackend>::broadcast(
            MlxTensor::from_array(local_token),
            sampling_rank,
            group,
            stream,
        )
        .map_err(|error| authority.submission_error(error, operation, phase, None))
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let token = authority
        .wait(token_submission, operation, phase, None)
        .map_err(|error| Error::Parallel(error.to_string()))?
        .into_array()
        .as_dtype(Dtype::Uint32, stream)?;

    let local_finished = if group.rank() == sampling_rank && finished {
        ones::<f32>(&[], stream)?
    } else {
        zeros::<f32>(&[], stream)?
    };
    let finished_submission =
        <crate::backend::nn::shared::MlxNeuralBackend as BroadcastBackend>::broadcast(
            MlxTensor::from_array(local_finished),
            sampling_rank,
            group,
            stream,
        )
        .map_err(|error| authority.submission_error(error, operation, phase, None))
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let eredu_core::Submission { output, completion } = finished_submission;
    let (finished, completion) = completion.with_f32_flag(output.into_array());
    let finished = authority
        .wait(
            eredu_core::Submission {
                output: finished,
                completion,
            },
            operation,
            phase,
            None,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?
        .resolve()?;
    Ok(SynchronizedToken { token, finished })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::nn::shared::MlxCommunicationTensorMetadata;
    use crate::backend::runtime::distributed::{
        completion::{force_next_communication_pending, release_forced_pending_orphans},
        group::{native_collective_submissions, reset_native_collective_submissions},
        topology::CommunicationRouteRealization,
    };
    use eredu_runtime::{
        CommunicationCompletionPolicy, CommunicationGroupDescriptor,
        CommunicationGroupRequirements, CommunicationOperationRequirement,
        CommunicationTensorLimits, PartitionCommunication, RealizedCommunicationGroup,
    };
    use safemlx::{distributed::Backend, Device, DeviceType};

    #[derive(Default)]
    struct FixedTokenSampler;

    impl Sampler<MlxSamplingBackend> for FixedTokenSampler {
        fn sample(
            &mut self,
            _logits: &MlxTensor,
            _temperature: f32,
            _random: Option<&mut crate::backend::random::RandomState>,
            _context: &Stream,
        ) -> Result<MlxTensor, safemlx::error::Exception> {
            Ok(MlxTensor::from_array(Array::from_slice(&[0u32], &[1])))
        }
    }

    #[test]
    fn bounded_sampling_timeout_poisons_retry_before_another_collective() {
        release_forced_pending_orphans();
        reset_native_collective_submissions();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let native = safemlx::distributed::init(false, Backend::Ring).unwrap();
        let requirement = CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::Broadcast,
            [eredu_core::checkpoint::TensorDtype::F32],
            CommunicationTensorLimits::new(1, 2, 1024, None).unwrap(),
            true,
        )
        .unwrap();
        let descriptor = CommunicationGroupDescriptor::new(
            eredu_core::CollectiveGroupId::new(1),
            0,
            vec![0],
            Some(0),
            CommunicationGroupRequirements::new([requirement]).unwrap(),
        )
        .unwrap();
        let completion_policy = CommunicationCompletionPolicy::new(
            std::time::Duration::from_millis(25),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let group = Group::uncontracted(&native)
            .with_manifest_contract(&descriptor, completion_policy)
            .unwrap();
        let manifest =
            eredu_runtime::CommunicationManifest::new(1, 0, vec![descriptor], Vec::new())
                .unwrap()
                .with_completion_policy(completion_policy);
        let communication = PartitionCommunication::<
            crate::backend::nn::shared::MlxNeuralBackend,
            Group,
            CommunicationRouteRealization,
            MlxCommunicationTensorMetadata,
        >::new(
            manifest,
            vec![RealizedCommunicationGroup::new(
                eredu_core::CollectiveGroupId::new(1),
                group.clone(),
            )],
            Vec::new(),
            MlxCommunicationTensorMetadata,
        )
        .unwrap();
        let authority = communication.authority();
        let logits = MlxTensor::from_array(Array::from_slice(&[1.0f32], &[1, 1]));
        force_next_communication_pending();
        let first = sample_and_synchronize_bounded(
            Some(&logits),
            1,
            &mut FixedTokenSampler,
            0.0,
            None,
            false,
            0,
            &group,
            &authority,
            &stream,
        )
        .unwrap_err();
        assert!(first.to_string().contains("deadline"));
        assert_eq!(native_collective_submissions(), 1);

        let retry = sample_and_synchronize_bounded(
            Some(&logits),
            1,
            &mut FixedTokenSampler,
            0.0,
            None,
            false,
            0,
            &group,
            &authority,
            &stream,
        )
        .unwrap_err();
        assert!(retry.to_string().contains("poisoned"));
        assert_eq!(native_collective_submissions(), 1);
        release_forced_pending_orphans();
    }
}
