use super::*;

impl eredu_core::speculative::SpeculativeQueueMechanisms
    for super::autoregressive::MlxAutoregressiveMechanisms
{
    type Queue = Stream;
    type Root = Array;
    type Handoff = Event;
    type Error = Exception;

    fn same_queue(left: &Stream, right: &Stream) -> bool {
        left == right
    }

    fn same_device(left: &Stream, right: &Stream) -> Result<bool, Exception> {
        Ok(left.get_device()? == right.get_device()?)
    }

    fn order_roots<'a>(
        roots: impl IntoIterator<Item = &'a Array>,
        consumer: &Stream,
    ) -> Result<Event, Exception> {
        let completion = async_eval_with_event(roots)?;
        completion.wait_on(consumer)?;
        Ok(completion)
    }

    fn invalid_assignment(message: String) -> Exception {
        Exception::custom(message)
    }
}
