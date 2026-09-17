use super::handle::record_native_collective_submission;
use super::*;
use crate::backend::nn::logical_collective::{self,Operations};

fn depends_on(value: &Array, dependency: &Array) -> Result<Array> {
    safemlx::transforms::depends([value], [dependency])?
        .pop()
        .ok_or_else(|| Exception::custom("MLX depends returned no output"))
}

/// Completes one slot in a logical subgroup's shared-world participation wave.
/// Independent lazy branches can otherwise evaluate in a different order on
/// active and zero-work pipeline stages, matching unrelated native collectives.
pub(super) fn ordered_world_sum(input: &Array, group: &Group, stream: &Stream) -> Result<Array> {
    use eredu_core::{
        BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    };
    let started = std::time::Instant::now();
    let output = native::all_sum(
        input,
        &group.native,
        group.communication_stream(stream.as_ref())?,
    )?;
    let completion =
        crate::backend::runtime::distributed::completion::MlxCommunicationCompletion::submit(
            [&output],
            vec![input.clone(), output.clone()],
            vec![],
            vec![group.clone()],
            vec![],
            vec![stream.clone()],
        )?;
    if let Some(policy) = group.completion_policy() {
        let remaining = policy
            .timeout()
            .saturating_sub(started.elapsed())
            .max(std::time::Duration::from_nanos(1));
        let wait = BoundedCompletionWait::new(remaining, policy.cancellation())
            .map_err(|error| Exception::custom(error.to_string()))?;
        if matches!(
            completion.wait_bounded(wait)?,
            BoundedCompletionOutcome::DeadlineExceeded { .. }
        ) {
            return Err(Exception::custom("logical world collective exceeded its deadline; native resources retained until completion"));
        }
    } else {
        completion.wait()?;
    }
    Ok(output)
}

pub(super) fn pack_logical_value(
    input: &Array,
    slot: usize,
    native_size: usize,
    stream: &Stream,
) -> Result<Array> {
    logical_collective::packed::pack(&logical_collective::Native(stream),input,slot,native_size)
}

fn native_send(input: &Array, destination: usize, group: &Group, stream: &Stream) -> Result<Array> {
    native::send(
        input,
        destination,
        &group.native,
        group.communication_stream(stream.as_ref())?,
    )
}

fn native_recv_like(like: &Array, source: usize, group: &Group, stream: &Stream) -> Result<Array> {
    native::recv_like(
        like,
        source,
        &group.native,
        group.communication_stream(stream.as_ref())?,
    )
}

pub(super) fn logical_direct_exchange(
    input: &Array,
    group: &Group,
    stream: &Stream,
) -> Result<Option<Array>> {
    let Some(plan) = group.logical_exchange_plan().map_err(Exception::from_source)? else {
        return Ok(None);
    };
    let ops=logical_collective::Native(stream);
    let zero = ops.zero(input)?;
    let exchanged = plan.fold_rounds(input.clone(), |_, exchanged| {
        let (destination, source) = plan.peers();
        let sent = native_send(&exchanged, destination, group, stream)?;
        let received = native_recv_like(&exchanged, source, group, stream)?;
        let exchanged = logical_collective::peer(&ops,&sent,&received,&zero)?;
        safemlx::transforms::async_eval_with_event([&exchanged])?.synchronize()?;
        Ok::<_,Exception>(exchanged)
    })?;
    Ok(Some(exchanged))
}

fn logical_routed_values(
    input: &Array,
    group: &Group,
    stream: &Stream,
) -> Result<Option<Vec<(usize, Array)>>> {
    let Some(plan) = group.logical_routed_plan().map_err(Exception::from_source)? else {
        return Ok(None);
    };
    let zero = zeros_dtype(&[], input.dtype(), stream)?;
    let mut values = Vec::with_capacity(plan.len());
    for route in plan.values() {
        let routed = route.fold_exchanges(input.clone(), |_, exchange, routed| {
            let (destination, source) = exchange.peers();
            let sent = native_send(&routed, destination, group, stream)?;
            let received = native_recv_like(&routed, source, group, stream)?;
            let routed = logical_collective::peer(&logical_collective::Native(stream), &sent, &received, &zero)?;
            safemlx::transforms::async_eval_with_event([&routed])?.synchronize()?;
            Ok::<_, Exception>(routed)
        })?;
        values.push((route.source_rank(), routed));
    }
    Ok(Some(values))
}

fn logical_all_sum(input: &Array, group: &Group, stream: &Stream) -> Result<Array> {
    if let Some(values) = logical_routed_values(input, group, stream)? {
        return logical_collective::routed::sum(&logical_collective::Native(stream), values);
    }
    if let Some(peer) = logical_direct_exchange(input, group, stream)? {
        return logical_collective::sum(&logical_collective::Native(stream),input,&peer);
    }
    let plan=group.logical_packed_world_plan().map_err(Exception::from_source)?
        .ok_or_else(||Exception::custom("logical subgroup has no selected packed-world plan"))?;
    let packed=pack_logical_value(input,plan.representative(),plan.world_size(),stream)?;
    logical_collective::packed::sum_result(&logical_collective::Native(stream),
        &ordered_world_sum(&packed,group,stream)?,plan.representative())
}

fn logical_all_gather_stacked(input: &Array, group: &Group, stream: &Stream) -> Result<Array> {
    if let Some(values) = logical_routed_values(input, group, stream)? {
        return logical_collective::routed::stacked(&logical_collective::Native(stream), values);
    }
    if let Some(peer) = logical_direct_exchange(input, group, stream)? {
        let stacked = logical_collective::stacked(&logical_collective::Native(stream),input,&peer,group.rank()==0);
        drop(peer);
        return stacked;
    }
    let plan=group.logical_packed_world_plan().map_err(Exception::from_source)?
        .ok_or_else(||Exception::custom("logical subgroup has no selected packed-world plan"))?;
    let packed=pack_logical_value(input,plan.world_rank(),plan.world_size(),stream)?;
    let gathered=ordered_world_sum(&packed,group,stream)?;
    logical_collective::packed::gather_stacked(&logical_collective::Native(stream),
        &gathered,plan.members())
}

fn all_sum_unchecked(input: &Array, group: &Group, stream: &Stream) -> Result<Array> {
    record_native_collective_submission(group);
    match group.logical {
        Some(_) => logical_all_sum(input, group, stream),
        None => native::all_sum(
            input,
            &group.native,
            group.communication_stream(stream.as_ref())?,
        ),
    }
}

/// Executes one tensor-carrying sum-based collective under its exact contract.
pub(crate) fn all_sum_for(
    operation: CommunicationOperation,
    input: &Array,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.validate_tensor(operation, input, false)?;
    group.validate_expected_output(operation, input.dtype(), input.ndim(), input.size())?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let output = all_sum_unchecked(input, group, stream.as_ref())?;
    group.validate_tensor(operation, &output, true)?;
    if output.shape() != input.shape() {
        return Err(Exception::custom(format!(
            "{operation:?} completed with shape {:?}, expected {:?}",
            output.shape(),
            input.shape()
        )));
    }
    Ok(output)
}

/// Executes one payload-free sum-based collective under its exact contract.
pub(crate) fn payload_free_all_sum_for(
    operation: CommunicationOperation,
    token: &Array,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.validate_payload_free(operation)?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let output = if operation == CommunicationOperation::FailureAgreement {
        match super::status::independent_status_sum(token, group, stream.as_ref())? {
            Some(output) => {
                record_native_collective_submission(group);
                output
            }
            None => all_sum_unchecked(token, group, stream.as_ref())?,
        }
    } else {
        all_sum_unchecked(token, group, stream.as_ref())?
    };
    if output.shape() != token.shape() {
        return Err(Exception::custom(format!(
            "{operation:?} completed with shape {:?}, expected {:?}",
            output.shape(),
            token.shape()
        )));
    }
    Ok(output)
}

/// Sums `input` element-wise across a native or logical group.
pub fn all_sum(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    all_sum_for(CommunicationOperation::AllReduceSum, input, group, stream)
}

/// Gathers `input` from every rank, concatenating along axis zero.
pub(crate) fn all_gather_for(
    operation: CommunicationOperation,
    input: &Array,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.validate_tensor(operation, input, false)?;
    let output_elements = input
        .size()
        .checked_mul(group.size())
        .ok_or_else(|| Exception::custom("all-gather output elements overflow usize"))?;
    let output_rank = if input.ndim() == 0 { 1 } else { input.ndim() };
    group.validate_expected_output(operation, input.dtype(), output_rank, output_elements)?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let output = all_gather_unchecked(input, group, stream)?;
    group.validate_tensor(operation, &output, true)?;
    Ok(output)
}

pub(crate) fn all_gather_unchecked(
    input: &Array,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    record_native_collective_submission(group);
    let stream = stream.as_ref();
    let output = if group.logical.is_none() {
        native::all_gather(
            input,
            &group.native,
            group.communication_stream(stream.as_ref())?,
        )?
    } else {
        let stacked = logical_all_gather_stacked(input, group, stream)?;
        if input.ndim() == 0 {
            stacked
        } else {
            let mut shape = input.shape().to_vec();
            shape[0] = shape[0]
                .checked_mul(
                    i32::try_from(group.size())
                        .map_err(|_| Exception::custom("logical group size does not fit in i32"))?,
                )
                .ok_or_else(|| Exception::custom("logical all-gather shape exceeds i32"))?;
            stacked.reshape(&shape, stream)?
        }
    };
    let mut expected = input.shape().to_vec();
    if input.ndim() == 0 {
        expected = vec![i32::try_from(group.size())
            .map_err(|_| Exception::custom("group size does not fit in i32"))?];
    } else {
        expected[0] = expected[0]
            .checked_mul(
                i32::try_from(group.size())
                    .map_err(|_| Exception::custom("group size does not fit in i32"))?,
            )
            .ok_or_else(|| Exception::custom("all-gather output shape exceeds i32"))?;
    }
    if output.shape() != expected {
        return Err(Exception::custom(format!(
            "all-gather completed with shape {:?}, expected {expected:?}",
            output.shape()
        )));
    }
    Ok(output)
}

/// Gathers `input` from every rank, concatenating along axis zero.
pub fn all_gather(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    all_gather_for(CommunicationOperation::AllGatherEven, input, group, stream)
}

fn validate_all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
) -> Result<()> {
    if input.ndim() == 0 {
        return Err(Exception::custom(
            "all_to_all_v input must have a leading row dimension",
        ));
    }
    if send_counts.len() != group.size() || recv_counts.len() != group.size() {
        return Err(Exception::custom(format!(
            "all_to_all_v requires {} send counts and receive counts, got {} and {}",
            group.size(),
            send_counts.len(),
            recv_counts.len()
        )));
    }
    let send_rows = send_counts.iter().try_fold(0usize, |total, count| {
        total
            .checked_add(*count)
            .ok_or_else(|| Exception::custom("all_to_all_v send count sum overflowed usize"))
    })?;
    if usize::try_from(input.dim(0)).ok() != Some(send_rows) {
        return Err(Exception::custom(format!(
            "all_to_all_v send count sum {send_rows} does not match input row count {}",
            input.dim(0)
        )));
    }
    if send_counts[group.rank()] != recv_counts[group.rank()] {
        return Err(Exception::custom(format!(
            "all_to_all_v self send count {} does not match self receive count {}",
            send_counts[group.rank()],
            recv_counts[group.rank()]
        )));
    }
    // Native row geometry is I32; the shared route worker derives each actual
    // selected interval from these already-checked counts.
    i32::try_from(send_rows).map_err(|_| Exception::custom("all_to_all_v row offset exceeds i32"))?;
    Ok(())
}

fn concatenate_leading_blocks(
    input: &Array,
    counts: &[usize],
    logical_order: impl Iterator<Item = usize>,
    stream: &Stream,
) -> Result<Array> {
    logical_collective::blocks::concatenate(&logical_collective::Native(stream), input, counts, logical_order)
}

fn logical_world_all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
    stream: &Stream,
) -> Result<Array> {
    let plan = group.logical_variable_world_plan()
        .ok_or_else(|| Exception::custom("variable exchange has no retained world participation wave"))?;
    let world_size = plan.world_size();
    let mut world_send = vec![0_usize; world_size];
    let mut world_recv = vec![0_usize; world_size];
    for (logical_rank, global_rank) in plan.members().iter().copied().enumerate() {
        world_send[global_rank] = send_counts[logical_rank];
        world_recv[global_rank] = recv_counts[logical_rank];
    }
    let canonical_order = plan.canonical_order();
    let world_input = if canonical_order {
        input.clone()
    } else {
        concatenate_leading_blocks(
            input,
            send_counts,
            plan.input_order(),
            stream,
        )?
    };
    let world_output = native::all_to_all_v(
        &world_input,
        &world_send,
        &world_recv,
        &group.native,
        group.communication_stream(stream.as_ref())?,
    )?;
    if canonical_order {
        Ok(world_output)
    } else {
        let world_receive_counts = (0..world_size)
            .map(|global_rank| world_recv[global_rank])
            .collect::<Vec<_>>();
        concatenate_leading_blocks(
            &world_output,
            &world_receive_counts,
            plan.output_order(),
            stream,
        )
    }
}

/// Exchanges variable-sized leading-axis blocks across a native or logical group.
pub fn all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.validate_tensor(CommunicationOperation::VariableAllToAll, input, false)?;
    group.validate_peer_counts(
        CommunicationOperation::VariableAllToAll,
        send_counts,
        recv_counts,
    )?;
    let expected_rows = recv_counts.iter().try_fold(0usize, |total, count| {
        total
            .checked_add(*count)
            .ok_or_else(|| Exception::custom("all_to_all_v receive count sum overflowed usize"))
    })?;
    let trailing_elements = input.shape()[1..]
        .iter()
        .try_fold(1usize, |total, dimension| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| Exception::custom("all_to_all_v input has a negative dimension"))?;
            total
                .checked_mul(dimension)
                .ok_or_else(|| Exception::custom("all_to_all_v output elements overflow usize"))
        })?;
    let output_elements = expected_rows
        .checked_mul(trailing_elements)
        .ok_or_else(|| Exception::custom("all_to_all_v output elements overflow usize"))?;
    group.validate_expected_output(
        CommunicationOperation::VariableAllToAll,
        input.dtype(),
        input.ndim(),
        output_elements,
    )?;
    validate_all_to_all_v(input, send_counts, recv_counts, group)?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let stream = stream.as_ref();
    record_native_collective_submission(group);
    if group.logical.is_none() {
        let output = native::all_to_all_v(
            input,
            send_counts,
            recv_counts,
            &group.native,
            group.communication_stream(stream.as_ref())?,
        )?;
        validate_all_to_all_output(&output, input, expected_rows, group)?;
        return Ok(output);
    }
    if group.size() == 1 {
        let output = input.clone();
        validate_all_to_all_output(&output, input, expected_rows, group)?;
        return Ok(output);
    }
    let logical = group.logical.as_ref().expect("logical group");
    if logical.world_collective_wave {
        let output = logical_world_all_to_all_v(input, send_counts, recv_counts, group, stream)?;
        validate_all_to_all_output(&output, input, expected_rows, group)?;
        return Ok(output);
    }
    let plan = group.logical_variable_route_plan().map_err(Exception::from_source)?
        .ok_or_else(|| Exception::custom("all_to_all_v logical subgroup has no retained local route itinerary"))?;
    let output = logical_collective::variable::execute(&mut LocalVariable { group, stream },
        input, send_counts, recv_counts, plan)?;
    validate_all_to_all_output(&output, input, expected_rows, group)?;
    Ok(output)
}

struct LocalVariable<'a> { group: &'a Group, stream: &'a Stream }
impl LocalVariable<'_> {
    fn complete_pair(&self, exchange: LogicalExchangePlan<'_>, sent: &Array, received: &Array)
        -> Result<()> {
        // Both peers consume the same selected endpoint ordering as the counted
        // pair source. Blocking payload sends must not be first on both ranks.
        let roots = if exchange.sends_first() { [sent, received] } else { [received, sent] };
        safemlx::transforms::async_eval_with_event(roots)?.synchronize()
    }
}
impl logical_collective::variable::LocalVariableOperations for LocalVariable<'_> {
    type Value = Array;
    type Error = Exception;
    fn invalid(&self) -> Exception { Exception::custom("variable route differs from its exact row counts") }
    fn charge(&self, _: usize) -> Result<()> { Ok(()) }
    fn values(&self, capacity: usize) -> Result<Vec<(usize, Array)>> { Ok(Vec::with_capacity(capacity)) }
    fn rows(&self, value: &Array) -> Option<usize> { value.shape().first().and_then(|value| usize::try_from(*value).ok()) }
    fn slice(&mut self, input: &Array, counts: &[usize], destination: usize) -> Result<Array> {
        let start = counts[..destination].iter().copied().try_fold(0usize, usize::checked_add)
            .and_then(|value| i32::try_from(value).ok()).ok_or_else(|| self.invalid())?;
        let end = i32::try_from(counts[destination]).ok().and_then(|value| start.checked_add(value))
            .ok_or_else(|| self.invalid())?;
        input.try_index_device(start..end, self.stream)
    }
    fn exchange(&mut self, _: usize, _: usize, exchange: LogicalExchangePlan<'_>, routed: Array) -> Result<Array> {
        let (destination, source) = exchange.peers();
        let group = self.group; let stream = self.stream;
        // Array dimensions and native route geometry are I32. Both ordinary
        // and counted transports use this exact scalar wire representation.
        let count = Array::from_slice(&[routed.dim(0)], &[1]).copy(stream)?;
        let sent_count = native_send(&count, destination, group, stream)?;
        let received_count = native_recv_like(&count, source, group, stream)?;
        self.complete_pair(exchange, &sent_count, &received_count)?;
        let incoming_rows = received_count.evaluated()?.as_slice::<i32>()[0];
        if incoming_rows < 0 { return Err(self.invalid()); }
        let mut shape = routed.shape().to_vec(); shape[0] = incoming_rows;
        let empty = zeros_dtype(&shape, routed.dtype(), stream)?;
        let sent = native_send(&routed, destination, group, stream)?;
        let incoming = native_recv_like(&empty, source, group, stream)?;
        self.complete_pair(exchange, &sent, &incoming)?;
        Ok(incoming)
    }
    fn concatenate(&mut self, values: Vec<(usize, Array)>) -> Result<Array> {
        let arrays = values.iter().map(|(_, array)| array).collect::<Vec<_>>();
        concatenate_axis(&arrays, 0, self.stream)
    }
}

fn validate_all_to_all_output(
    output: &Array,
    input: &Array,
    expected_rows: usize,
    group: &Group,
) -> Result<()> {
    group.validate_tensor(CommunicationOperation::VariableAllToAll, output, true)?;
    let mut expected = input.shape().to_vec();
    expected[0] = i32::try_from(expected_rows)
        .map_err(|_| Exception::custom("all_to_all_v output rows exceed i32"))?;
    if output.shape() != expected {
        return Err(Exception::custom(format!(
            "VariableAllToAll completed with shape {:?}, expected {expected:?}",
            output.shape()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod ordered_wave_tests {
    use super::*;

    #[test]
    #[ignore = "requires native MLX CPU execution; run explicitly"]
    fn ordered_world_timeout_retains_and_fences_native_group() {
        use crate::backend::runtime::distributed::completion::{
            ensure_group_available, force_next_communication_pending,
            release_forced_pending_orphans,
        };
        let native = native::init(false, native::Backend::Ring).unwrap();
        let policy = CommunicationCompletionPolicy::new(
            std::time::Duration::from_millis(5),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let group = Group::uncontracted(&native).with_completion_policy(policy);
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let value = Array::from_slice(&[2.0_f32, -3.0], &[2]);
        force_next_communication_pending();
        assert!(ordered_world_sum(&value, &group, &stream)
            .unwrap_err()
            .what()
            .contains("native resources retained"));
        assert!(ensure_group_available(&group).is_err());
        release_forced_pending_orphans();
        assert!(ensure_group_available(&group).is_ok());
        let result = ordered_world_sum(&value, &group, &stream).unwrap();
        assert_eq!(result.evaluated().unwrap().as_slice::<f32>(), &[2.0, -3.0]);
    }
}
