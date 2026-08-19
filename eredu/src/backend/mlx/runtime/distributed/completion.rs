//! Backend-independent completion ownership for distributed operations.
//!
//! Distributed MLX primitives are lazy just like ordinary array operations.
//! This module couples a submitted result with the exact SafeMLX event which
//! completes it, so pipeline and collective callers do not need to duplicate
//! `eval` plus whole-stream synchronization sequences.

use safemlx::{transforms::async_eval_with_event, Array, Event, EventBackend, Stream};

use crate::backend::mlx::error::Error;

/// A submitted distributed result and its exact backend completion.
///
/// Construction explicitly evaluates the supplied MLX outputs. Merely
/// constructing a send, receive, collective, or pipeline graph does not create
/// a completion. The retained arrays keep every submitted endpoint alive until
/// this value is dropped; MLX additionally retains backend resources while
/// producer work or consumer waits remain outstanding.
///
/// A completion is single-shot but may be queried, host-waited, or waited on by
/// multiple compatible streams. [`Self::wait_on`] is a backend stream
/// dependency and does not block the host. Dropping the value is safe while
/// work remains outstanding, but asynchronous errors are observable only
/// through [`Self::is_complete`], [`Self::synchronize`], or
/// [`Self::into_value`].
///
/// This type is intentionally neither `Send` nor `Sync`, matching SafeMLX's
/// thread-affine [`Event`] contract.
#[derive(Debug)]
#[must_use = "distributed work has been submitted; retain, wait on, or synchronize its completion"]
pub struct DistributedCompletion<T> {
    value: T,
    event: Event,
    _retained: Vec<Array>,
}

impl<T> DistributedCompletion<T> {
    pub(crate) fn submit<'a>(
        value: T,
        outputs: impl IntoIterator<Item = &'a Array>,
    ) -> Result<Self, Error> {
        let retained = outputs.into_iter().cloned().collect::<Vec<_>>();
        let event = async_eval_with_event(retained.iter())?;
        Ok(Self {
            value,
            event,
            _retained: retained,
        })
    }

    /// Returns the submitted value without waiting for its backend completion.
    ///
    /// Host access still requires synchronization. Work evaluated later on a
    /// different compatible stream must first call [`Self::wait_on`].
    pub const fn value(&self) -> &T {
        &self.value
    }

    /// Orders later evaluation on `stream` after this completion.
    ///
    /// Because MLX graphs are lazy, the consumer graph must be evaluated after
    /// this call. Constructing it before or after the call does not submit it.
    pub fn wait_on(&self, stream: &Stream) -> Result<(), Error> {
        self.event.wait_on(stream)?;
        Ok(())
    }

    /// Returns whether the exact distributed operation has completed.
    pub fn is_complete(&self) -> Result<bool, Error> {
        Ok(self.event.is_complete()?)
    }

    /// Returns the backend which owns this exact completion.
    pub fn backend(&self) -> Result<EventBackend, Error> {
        Ok(self.event.backend()?)
    }

    /// Returns the arrays explicitly retained through exact completion.
    pub fn retained_resources(&self) -> usize {
        self._retained.len()
    }

    /// Blocks the host for this exact completion, not the remainder of a stream.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.event.synchronize()?;
        Ok(())
    }

    /// Waits for exact completion and returns the owned result.
    pub fn into_value(self) -> Result<T, Error> {
        self.event.synchronize()?;
        Ok(self.value)
    }
}

impl<T> eredu_core::Completion for DistributedCompletion<T> {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        self.is_complete()
    }

    fn wait(&self) -> Result<(), Self::Error> {
        self.synchronize()
    }
}

pub(crate) fn synchronize_outputs<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
) -> safemlx::error::Result<()> {
    async_eval_with_event(outputs)?.synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use safemlx::{
        transforms::{async_eval, async_eval_with_event},
        Device, DeviceType,
    };

    fn cpu_stream() -> Stream {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }

    #[test]
    fn distributed_completion_orders_multiple_cpu_consumers() {
        let producer = cpu_stream();
        let consumer_a = cpu_stream();
        let consumer_b = cpu_stream();
        let blocker_lhs = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
        let blocker_rhs = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
        let blocker = blocker_lhs.matmul(&blocker_rhs, &producer).unwrap();
        async_eval([&blocker]).unwrap();

        let value = Array::ones::<f32>(&[1, 1024], &producer).unwrap();
        let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();
        assert!(!completion.is_complete().unwrap());
        completion.wait_on(&consumer_a).unwrap();
        completion.wait_on(&consumer_b).unwrap();
        let consumed_a = completion.value().square(&consumer_a).unwrap();
        let consumed_b = completion.value().square(&consumer_b).unwrap();
        let completion_a = async_eval_with_event([&consumed_a]).unwrap();
        let completion_b = async_eval_with_event([&consumed_b]).unwrap();

        let value = completion.into_value().unwrap();
        assert_eq!(value.shape(), [1, 1024]);
        completion_a.synchronize().unwrap();
        completion_b.synchronize().unwrap();
    }

    #[test]
    fn dropping_distributed_completion_preserves_a_queued_cpu_wait() {
        let producer = cpu_stream();
        let consumer = cpu_stream();
        let value = Array::ones::<f32>(&[8, 8], &producer).unwrap();
        let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();
        completion.wait_on(&consumer).unwrap();
        let consumed = completion
            .value()
            .add(Array::from_int(1), &consumer)
            .unwrap();
        let consumed_completion = async_eval_with_event([&consumed]).unwrap();
        drop(completion);

        consumed_completion.synchronize().unwrap();
        assert_eq!(consumed.evaluated().unwrap().as_slice::<f32>(), &[2.0; 64]);
    }

    #[test]
    #[ignore = "explicit Metal distributed completion test; run on a Metal host"]
    fn distributed_completion_metal_wait_does_not_block_the_host() {
        let producer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let consumer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let blocker_lhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
        let blocker_rhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
        let blocker = blocker_lhs.matmul(&blocker_rhs, &producer).unwrap();
        async_eval([&blocker]).unwrap();
        let value = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
        let completion = DistributedCompletion::submit(value.clone(), [&value]).unwrap();

        assert!(!completion.is_complete().unwrap());
        completion.wait_on(&consumer).unwrap();
        assert!(!completion.is_complete().unwrap());
        let consumed = completion.value().square(&consumer).unwrap();
        let consumed_completion = async_eval_with_event([&consumed]).unwrap();
        drop(completion);
        consumed_completion.synchronize().unwrap();
    }
}
