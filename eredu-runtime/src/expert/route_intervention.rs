//! Selection remains in the architecture-configured selector. These adapters
//! only connect portable control/evidence hooks before provider execution.
use super::RoutedExpertProvider;
use eredu_nn::{Error, GroupSelection, GroupSelectionOperator, GroupedNeuralBackend, Tensor};

fn token_rows<T: Tensor>(input: &T) -> Result<u64, Error> {
    let shape = input.shape();
    let rows = shape
        .get(..shape.len().saturating_sub(1))
        .ok_or_else(|| Error::backend("router input has no token axes"))?;
    rows.iter()
        .try_fold(1u64, |n, value| {
            u64::try_from(*value)
                .ok()
                .and_then(|value| n.checked_mul(value))
        })
        .ok_or_else(|| Error::backend("routing token-row shape overflow"))
}

/// Selects through an architecture-owned selector before an observed provider's
/// expert call. Ordinary providers take the original lightweight select path.
pub fn select_routes_with_provider<B, P>(
    selector: &mut B::Selector,
    input: &B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
    provider: &mut P,
    bank: crate::RoutedBankId,
) -> Result<GroupSelection<B::Tensor>, Error>
where
    B: GroupedNeuralBackend,
    P: RoutedExpertProvider<B>,
    P::Error: std::fmt::Display,
{
    let Some(control) = provider
        .routing_control(bank, token_rows(input)?)
        .map_err(Error::backend)?
    else {
        return selector.select(input, context);
    };
    let result: Result<_, Error> = (|| {
        let selection = selector.select_intervened(input, &control, context)?;
        provider
            .routing_applied(
                bank,
                selection.original.as_ref().map(Into::into),
                (&selection.effective).into(),
            )
            .map_err(Error::backend)?;
        Ok(selection.effective)
    })();
    if let Err(error) = &result {
        provider.routing_failed(bank, &error.to_string());
    }
    result
}

/// Selects with a direct architecture observation hook, preserving shared-expert
/// work outside routed control. No unmodified model forward pass is performed.
pub fn select_routes_with_observer<T, S, O>(
    selector: &mut S,
    input: &T,
    context: &T::Context,
    path: &str,
    observer: &mut O,
) -> Result<GroupSelection<T>, Error>
where
    T: Tensor,
    S: GroupSelectionOperator<T>,
    O: crate::ActivationObserver<T, Error> + ?Sized,
{
    let Some(control) = observer.routing_control(path, token_rows(input)?)? else {
        return selector.select(input, context);
    };
    let result: Result<_, Error> = (|| {
        let selection = selector.select_intervened(input, &control, context)?;
        observer.routing_applied(
            path,
            selection.original.as_ref().map(Into::into),
            (&selection.effective).into(),
        )?;
        Ok(selection.effective)
    })();
    if let Err(error) = &result {
        observer.routing_failed(path, &error.to_string());
    }
    result
}
