use super::*;

/// Target and assistant streams assigned to one speculative session.
#[derive(Debug, Clone, Copy)]
pub struct SpeculativeExecutionStreams<'a> {
    target: &'a Stream,
    draft: &'a Stream,
    topology: SpeculativeExecutionTopology,
}

impl<'a> SpeculativeExecutionStreams<'a> {
    /// Binds queues to topology already selected by portable composition.
    pub fn bind(
        target: &'a Stream,
        draft: &'a Stream,
        topology: SpeculativeExecutionTopology,
    ) -> Result<Self, Exception> {
        let matches = match topology {
            SpeculativeExecutionTopology::Single => target == draft,
            SpeculativeExecutionTopology::SameDeviceSplit => {
                target != draft && target.get_device()? == draft.get_device()?
            }
            SpeculativeExecutionTopology::CrossDeviceSplit => {
                target.get_device()? != draft.get_device()?
            }
            _ => {
                return Err(Exception::custom(
                    "selected speculative topology is unsupported by the MLX queue binder",
                ))
            }
        };
        if !matches {
            return Err(Exception::custom(format!(
                "selected speculative topology {topology:?} does not match bound MLX queues"
            )));
        }
        Ok(Self {
            target,
            draft,
            topology,
        })
    }

    #[cfg(test)]
    pub(super) fn for_test(target: &'a Stream, draft: &'a Stream) -> Result<Self, Exception> {
        let topology = if target == draft {
            SpeculativeExecutionTopology::Single
        } else if target.get_device()? == draft.get_device()? {
            SpeculativeExecutionTopology::SameDeviceSplit
        } else {
            SpeculativeExecutionTopology::CrossDeviceSplit
        };
        Self::bind(target, draft, topology)
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
    pub fn wait_for_target_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Array>,
    ) -> Result<Event, Exception> {
        self.wait_for_same_device_outputs(outputs, self.draft, "target-to-draft")
    }

    /// Submits assistant outputs and orders subsequent target work after them.
    pub fn wait_for_draft_outputs<'b>(
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
                "speculative {direction} event handoff requires distinct streams on one device, got {}",
                self.topology
            )));
        }
        let completion = async_eval_with_event(outputs)?;
        completion.wait_on(consumer)?;
        Ok(completion)
    }
}
