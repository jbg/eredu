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
        finish_send(&sent, group, stream, policy, started)?;
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
        finish_send(&sent, group, stream, policy, started)?;
    }
    Ok(Some(total))
}

fn finish_send(
    sent: &Array,
    group: &Group,
    stream: &Stream,
    policy: CommunicationCompletionPolicy,
    started: std::time::Instant,
) -> Result<()> {
    use eredu_core::{BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait};
    let remaining = policy.timeout().saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(Exception::custom(
            "independent status setup deadline exceeded before submission",
        ));
    }
    let completion =
        crate::backend::runtime::distributed::completion::MlxCommunicationCompletion::submit(
            [sent],
            vec![sent.clone()],
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
        BoundedCompletionOutcome::DeadlineExceeded { .. } => Err(Exception::custom("independent status send exceeded its deadline; native resources retained until completion")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires native MLX CPU execution; run explicitly"]
    fn status_intermediate_timeout_retains_and_fences_native_group() {
        use crate::backend::runtime::distributed::completion::{
            ensure_group_available, force_next_communication_pending,
            release_forced_pending_orphans,
        };
        let native = native::init(false, native::Backend::Ring).unwrap();
        let group = Group::uncontracted(&native);
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let value = Array::ones::<i32>(&[1], &stream).unwrap();
        let policy = CommunicationCompletionPolicy::new(
            std::time::Duration::from_millis(5),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        force_next_communication_pending();
        let result = finish_send(&value, &group, &stream, policy, std::time::Instant::now());
        assert!(result
            .unwrap_err()
            .what()
            .contains("native resources retained"));
        assert!(ensure_group_available(&group).is_err());
        release_forced_pending_orphans();
        assert!(ensure_group_available(&group).is_ok());
        finish_send(&value, &group, &stream, policy, std::time::Instant::now()).unwrap();
    }
    #[test]
    fn independent_status_fact_matches_connected_native_ring_membership() {
        for world in 1..=10 {
            for mask in 1usize..1 << world {
                let members = (0..world)
                    .filter(|rank| mask & (1 << rank) != 0)
                    .collect::<Vec<_>>();
                let mut reached = std::collections::BTreeSet::from([members[0]]);
                loop {
                    let previous = reached.len();
                    for rank in reached.clone() {
                        for neighbor in [(rank + 1) % world, (rank + world - 1) % world] {
                            if members.contains(&neighbor) {
                                reached.insert(neighbor);
                            }
                        }
                    }
                    if reached.len() == previous {
                        break;
                    }
                }
                assert_eq!(
                    independent_status_members(world, &members),
                    reached.len() == members.len(),
                    "{world}: {members:?}"
                );
            }
        }
        for (world, members) in [
            (0, vec![0]),
            (4, vec![]),
            (4, vec![0, 4]),
            (4, vec![1, 1]),
            (4, vec![2, 1]),
        ] {
            assert!(!independent_status_members(world, &members));
        }
    }
}
