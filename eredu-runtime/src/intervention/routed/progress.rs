//! Same full-invocation progress for ordinary and original sparse edits.
use super::*;
use RoutedInterventionLoweringError as E;
pub(crate) fn prepare(
    previous: Option<RoutedUnitInterventionReceipt>,
    source_tokens: u64,
    range: [u64; 2],
) -> Result<RoutedUnitInterventionReceipt, E> {
    let current = previous.unwrap_or(RoutedUnitInterventionReceipt {
        source_tokens,
        completed_tokens: 0,
        affected_values: 0,
    });
    if source_tokens == 0
        || current.source_tokens != source_tokens
        || range[0] != current.completed_tokens
        || range[0] >= range[1]
        || range[1] > source_tokens
    {
        return Err(E::Coordinates);
    }
    Ok(current)
}
pub(crate) fn advance(
    previous: RoutedUnitInterventionReceipt,
    range: [u64; 2],
    affected: u64,
) -> Result<RoutedUnitInterventionReceipt, E> {
    let mut current = prepare(Some(previous), previous.source_tokens, range)?;
    current.affected_values = current
        .affected_values
        .checked_add(affected)
        .ok_or(E::Overflow)?;
    current.completed_tokens = range[1];
    Ok(current)
}
pub(crate) fn outcome(receipt: RoutedUnitInterventionReceipt) -> Result<InterventionOutcome, E> {
    if receipt.source_tokens == 0 || receipt.completed_tokens != receipt.source_tokens {
        return Err(E::Coordinates);
    }
    Ok(if receipt.affected_values == 0 {
        InterventionOutcome::Unmatched
    } else {
        InterventionOutcome::Applied
    })
}
pub(crate) fn successful(record: &InterventionRecord) -> bool {
    match (&record.outcome, record.routed_units) {
        (InterventionOutcome::Inactive, _) => true,
        (InterventionOutcome::Applied, None) => true,
        (InterventionOutcome::Applied, Some(receipt)) => {
            outcome(receipt) == Ok(InterventionOutcome::Applied)
        }
        (InterventionOutcome::Unmatched, Some(receipt)) => {
            outcome(receipt) == Ok(InterventionOutcome::Unmatched)
        }
        _ => false,
    }
}
