//! Portable queue assignment and observation binding for one speculative invocation.

use crate::capture::SpeculativePartitionBinding;
use eredu_core::{speculative::SpeculativeQueueMechanisms, SpeculativeExecutionTopology};

/// Portable assignment of exact queues, capture binding, and target observer.
pub struct SpeculativeExecutionContext<'a, Q: SpeculativeQueueMechanisms, T, O: ?Sized> {
    target: &'a Q::Queue,
    draft: &'a Q::Queue,
    topology: SpeculativeExecutionTopology,
    capture: Option<&'a SpeculativePartitionBinding<T>>,
    target_observer: Option<&'a O>,
}

impl<Q: SpeculativeQueueMechanisms, T, O: ?Sized> Copy
    for SpeculativeExecutionContext<'_, Q, T, O>
{
}
impl<Q: SpeculativeQueueMechanisms, T, O: ?Sized> Clone
    for SpeculativeExecutionContext<'_, Q, T, O>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Q: SpeculativeQueueMechanisms, T, O: ?Sized> std::fmt::Debug
    for SpeculativeExecutionContext<'_, Q, T, O>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpeculativeExecutionContext")
            .field("topology", &self.topology)
            .field("target_observer", &self.target_observer.is_some())
            .finish_non_exhaustive()
    }
}

impl<'a, Q: SpeculativeQueueMechanisms, T, O: ?Sized> SpeculativeExecutionContext<'a, Q, T, O> {
    /// Binds queues to topology already selected by portable composition.
    pub fn bind(
        target: &'a Q::Queue,
        draft: &'a Q::Queue,
        topology: SpeculativeExecutionTopology,
    ) -> Result<Self, Q::Error> {
        let matches = match topology {
            SpeculativeExecutionTopology::Single => Q::same_queue(target, draft),
            SpeculativeExecutionTopology::SameDeviceSplit => {
                !Q::same_queue(target, draft) && Q::same_device(target, draft)?
            }
            SpeculativeExecutionTopology::CrossDeviceSplit => !Q::same_device(target, draft)?,
            _ => {
                return Err(Q::invalid_assignment(
                    "selected speculative topology is unsupported by the queue binder".into(),
                ))
            }
        };
        if !matches {
            return Err(Q::invalid_assignment(format!(
                "selected speculative topology {topology:?} does not match the bound queues"
            )));
        }
        Ok(Self::paired(target, draft, topology))
    }

    const fn paired(
        target: &'a Q::Queue,
        draft: &'a Q::Queue,
        topology: SpeculativeExecutionTopology,
    ) -> Self {
        Self {
            target,
            draft,
            topology,
            capture: None,
            target_observer: None,
        }
    }

    /// Inspects the physical relationship between two already opened queues.
    pub fn infer_queue_assignment(
        target: &'a Q::Queue,
        draft: &'a Q::Queue,
    ) -> Result<Self, Q::Error> {
        let topology = if Q::same_queue(target, draft) {
            SpeculativeExecutionTopology::Single
        } else if Q::same_device(target, draft)? {
            SpeculativeExecutionTopology::SameDeviceSplit
        } else {
            SpeculativeExecutionTopology::CrossDeviceSplit
        };
        Self::bind(target, draft, topology)
    }

    /// Creates an assignment in which all speculative work uses one stream.
    pub const fn single(stream: &'a Q::Queue) -> Self {
        Self {
            target: stream,
            draft: stream,
            topology: SpeculativeExecutionTopology::Single,
            capture: None,
            target_observer: None,
        }
    }

    /// Attaches the observer already mapped for this target invocation.
    pub fn with_target_observer(mut self, observer: &'a O) -> Self {
        self.target_observer = Some(observer);
        self
    }

    /// Borrows the exact target observer supplied by the driver.
    pub fn target_observer(self) -> Option<&'a O> {
        self.target_observer
    }

    pub(crate) fn with_capture_binding(
        mut self,
        binding: Option<&'a SpeculativePartitionBinding<T>>,
    ) -> Self {
        self.capture = binding;
        self
    }

    /// Returns the exact installed capture binding for this invocation.
    pub(crate) fn capture_binding(self) -> Option<&'a SpeculativePartitionBinding<T>> {
        self.capture
    }

    /// Queue used for target prefill and verification.
    pub const fn target(self) -> &'a Q::Queue {
        self.target
    }

    /// Queue used for proposal generation.
    pub const fn draft(self) -> &'a Q::Queue {
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
        outputs: impl IntoIterator<Item = &'b Q::Root>,
    ) -> Result<Q::Handoff, Q::Error>
    where
        Q::Root: 'b,
    {
        self.wait_for_same_device_outputs(outputs, self.draft, "target-to-draft")
    }

    /// Submits assistant outputs and orders subsequent target work after them.
    pub fn wait_for_draft_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Q::Root>,
    ) -> Result<Q::Handoff, Q::Error>
    where
        Q::Root: 'b,
    {
        self.wait_for_same_device_outputs(outputs, self.target, "draft-to-target")
    }

    fn wait_for_same_device_outputs<'b>(
        self,
        outputs: impl IntoIterator<Item = &'b Q::Root>,
        consumer: &Q::Queue,
        direction: &str,
    ) -> Result<Q::Handoff, Q::Error>
    where
        Q::Root: 'b,
    {
        if self.topology != SpeculativeExecutionTopology::SameDeviceSplit {
            return Err(Q::invalid_assignment(format!(
                "speculative {direction} event handoff requires distinct streams on one device, got {}",
                self.topology
            )));
        }
        Q::order_roots(outputs, consumer)
    }
}
