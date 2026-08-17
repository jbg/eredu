//! MLX execution primitives for speculative model sessions.

use safemlx::{error::Exception, transforms::async_eval_with_event, Array, Event, Stream};
use safemlx_lm_core::{Completion, SpeculativeExecutionTopology};

/// Target and assistant streams assigned to one speculative session.
#[derive(Debug, Clone, Copy)]
pub struct MtpExecutionStreams<'a> {
    target: &'a Stream,
    draft: &'a Stream,
    topology: SpeculativeExecutionTopology,
}

impl<'a> MtpExecutionStreams<'a> {
    /// Creates an execution assignment and classifies its device topology.
    pub fn new(target: &'a Stream, draft: &'a Stream) -> Result<Self, Exception> {
        let topology = if target == draft {
            SpeculativeExecutionTopology::Single
        } else if target.get_device()? == draft.get_device()? {
            SpeculativeExecutionTopology::SameDeviceSplit
        } else {
            SpeculativeExecutionTopology::CrossDeviceSplit
        };
        Ok(Self {
            target,
            draft,
            topology,
        })
    }

    /// Creates an assignment in which all speculative work uses one stream.
    pub const fn single(stream: &'a Stream) -> Self {
        Self {
            target: stream,
            draft: stream,
            topology: SpeculativeExecutionTopology::Single,
        }
    }

    /// Stream used for target prefill and verification.
    pub const fn target(self) -> &'a Stream {
        self.target
    }

    /// Stream used for proposal generation.
    pub const fn draft(self) -> &'a Stream {
        self.draft
    }

    /// Relationship between the target and assistant streams.
    pub const fn topology(self) -> SpeculativeExecutionTopology {
        self.topology
    }

    /// Whether target and assistant work use different streams.
    pub const fn is_split(self) -> bool {
        !matches!(self.topology, SpeculativeExecutionTopology::Single)
    }

    /// Whether values must be physically transferred between devices.
    pub const fn crosses_devices(self) -> bool {
        matches!(
            self.topology,
            SpeculativeExecutionTopology::CrossDeviceSplit
        )
    }

    /// Submits target outputs and orders subsequent assistant work after them.
    pub(crate) fn wait_for_target_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
    ) -> Result<Event, Exception> {
        self.wait_for_same_device_outputs(outputs, self.draft, "target-to-draft")
    }

    /// Submits assistant outputs and orders subsequent target work after them.
    pub(crate) fn wait_for_draft_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
    ) -> Result<Event, Exception> {
        self.wait_for_same_device_outputs(outputs, self.target, "draft-to-target")
    }

    fn wait_for_same_device_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
        consumer: &Stream,
        direction: &str,
    ) -> Result<Event, Exception> {
        if self.topology != SpeculativeExecutionTopology::SameDeviceSplit {
            return Err(Exception::custom(format!(
                "MTP {direction} event handoff requires distinct streams on one device, got {}",
                self.topology
            )));
        }
        let completion = async_eval_with_event(outputs)?;
        completion.wait_on(consumer)?;
        Ok(completion)
    }
}

/// Exact completion for one retained MLX speculative verification.
pub struct MlxSpeculativeCompletion {
    event: Event,
    retained: Vec<Array>,
}

impl MlxSpeculativeCompletion {
    /// Submits all retained verification outputs as one exact completion.
    pub(crate) fn submit<'a>(
        outputs: impl IntoIterator<Item = &'a Array>,
    ) -> Result<Self, Exception> {
        let retained = outputs.into_iter().cloned().collect::<Vec<_>>();
        let event = async_eval_with_event(retained.iter())?;
        Ok(Self { event, retained })
    }

    /// Number of output handles retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self.retained.len()
    }
}

impl Completion for MlxSpeculativeCompletion {
    type Error = Exception;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.event.is_complete()
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.event.synchronize()
    }
}

impl Drop for MlxSpeculativeCompletion {
    fn drop(&mut self) {
        match self.event.is_complete() {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                let _ = self.event.synchronize();
            }
        }
    }
}
