use super::*;

/// Mechanism-only topology-wide consensus transport for realtime scheduling.
///
/// This transport owns only the selected native group and stream. Schedule,
/// completion, and publication policy remain in the neutral runtime scheduler.
#[derive(Clone)]
pub struct MlxRealtimeConsensusTransport {
    group: Group,
    stream: Stream,
}

impl MlxRealtimeConsensusTransport {
    /// Binds scheduler consensus to one already selected world group.
    pub fn new(group: &NativeGroup, stream: &Stream) -> Self {
        Self {
            group: Group::uncontracted(group),
            stream: stream.clone(),
        }
    }
}

impl eredu_core::consensus::ConsensusTransport for MlxRealtimeConsensusTransport {
    type Error = Error;

    fn participant_count(&self) -> usize {
        self.group.size()
    }

    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
        let submission =
            <Self as eredu_core::consensus::BoundedConsensusTransport>::submit_all_gather_words(
                self, local,
            )?;
        let output = submission.wait()?;
        <Self as eredu_core::consensus::BoundedConsensusTransport>::resolve_all_gather_words(
            self, output,
        )
    }
}

impl eredu_core::consensus::BoundedConsensusTransport for MlxRealtimeConsensusTransport {
    type Completion = MlxCommunicationCompletion;
    type GatherOutput = MlxTensor;

    fn submit_all_gather_words(
        &self,
        local: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Self::Completion>, Self::Error> {
        let length = i32::try_from(local.len())
            .map_err(|_| Error::Parallel("realtime consensus metadata exceeds i32".into()))?;
        let local = Array::from_slice(local, &[length]);
        let gathered = distributed::all_gather(&local, &self.group, &self.stream)?;
        let completion = MlxCommunicationCompletion::submit(
            [&gathered],
            vec![local, gathered.clone()],
            Vec::new(),
            vec![self.group.clone()],
            Vec::new(),
            vec![self.stream.clone()],
        )?;
        Ok(Submission {
            output: MlxTensor::from_array(gathered),
            completion,
        })
    }

    fn resolve_all_gather_words(
        &self,
        output: Self::GatherOutput,
    ) -> Result<Vec<u32>, Self::Error> {
        Ok(output.as_array().evaluated()?.as_slice::<u32>().to_vec())
    }
}
