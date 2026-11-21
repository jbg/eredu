//! Member-only status reduction on a connected segment of the native Ring.
//! Model tensor collectives retain their independently selected wave protocol.
use super::*;

fn chain_start(world: usize, members: &[usize]) -> Option<usize> {
    if world == 0
        || members.is_empty()
        || members.iter().any(|rank| *rank >= world)
        || members.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let mut start = None;
    for index in 0..members.len() {
        let next = (index + 1) % members.len();
        if (members[index] + 1) % world != members[next] {
            if start.is_some() {
                return None;
            }
            start = Some(next);
        }
    }
    Some(start.unwrap_or(0))
}

/// Exact cold membership fact used by both admission and native execution.
pub(crate) fn independent_status_members(world: usize, members: &[usize]) -> bool {
    chain_start(world, members).is_some()
}

pub(super) fn independent_status_sum(
    input: &Array,
    group: &Group,
    stream: &Stream,
) -> Result<Option<Array>> {
    let Some(logical) = group.logical.as_ref() else {
        return Ok(None);
    };
    let Some(start) = chain_start(group.native.size(), &logical.global_ranks) else {
        return Ok(None);
    };
    let count = logical.global_ranks.len();
    if count == 1 {
        return Ok(Some(input.clone()));
    }
    let policy = group.completion_policy().ok_or_else(|| {
        Exception::custom("independent status agreement requires bounded completion")
    })?;
    let started = std::time::Instant::now();
    let ordinal = (logical.rank + count - start) % count;
    let left = (ordinal > 0).then(|| logical.global_ranks[(start + ordinal - 1) % count]);
    let right = (ordinal + 1 < count).then(|| logical.global_ranks[(start + ordinal + 1) % count]);
    // Complete each send before submitting the receive in the opposite direction.
    // This avoids a lazy graph scheduling a blocking broadcast receive ahead of
    // its own reduction send. Intermediate submissions retain the same exact
    // arrays/groups/streams and quarantine policy as final communication work.
    let mut total = match left {
        Some(peer) => input.add(
            native::recv_like(
                input,
                peer,
                &group.native,
                group.communication_stream(stream.as_ref())?,
            )?,
            stream,
        )?,
        None => input.clone(),
    };
    if let Some(peer) = right {
        let sent = native::send(
            &total,
            peer,
            &group.native,
            group.communication_stream(stream.as_ref())?,
        )?;
        complete_output(&sent, group, stream, policy, started)?;
        total = native::recv_like(
            input,
            peer,
            &group.native,
            group.communication_stream(stream.as_ref())?,
        )?;
    }
    if let Some(peer) = left {
        let sent = native::send(
            &total,
            peer,
            &group.native,
            group.communication_stream(stream.as_ref())?,
        )?;
        complete_output(&sent, group, stream, policy, started)?;
    }
    Ok(Some(total))
}

/// Retains exact native resources until an output completes or its timed-out
/// submission is quarantined. Both status and guarded data traffic use this
/// boundary before another operation may reuse the transport.
pub(super) fn complete_output(
    output: &Array,
    group: &Group,
    stream: &Stream,
    policy: CommunicationCompletionPolicy,
    started: std::time::Instant,
) -> Result<()> {
    use eredu_core::{BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait};
    let remaining = policy.timeout().saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(Exception::custom(
            "communication output deadline exceeded before submission",
        ));
    }
    let completion =
        crate::backend::runtime::distributed::completion::MlxCommunicationCompletion::submit(
            [output],
            vec![output.clone()],
            vec![],
            vec![group.clone()],
            vec![],
            vec![stream.clone()],
        )?;
    let remaining = policy
        .timeout()
        .saturating_sub(started.elapsed())
        .max(std::time::Duration::from_nanos(1));
    let wait = BoundedCompletionWait::new(remaining, policy.cancellation())
        .map_err(|error| Exception::custom(error.to_string()))?;
    match completion.wait_bounded(wait)? {
        BoundedCompletionOutcome::Completed => Ok(()),
        BoundedCompletionOutcome::DeadlineExceeded { .. } => Err(Exception::custom("communication output exceeded its deadline; native resources retained until completion")),
    }
}
