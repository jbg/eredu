//! Same macro positional/keyword/caller policy for ordinary and paid storage.
//! Views own lookup and destinations own cloning; this worker allocates nothing.
pub(crate) trait View<'a> {
    type Value;
    fn parameters(&self) -> usize;
    fn parameter(&self, index: usize) -> Option<&'a str>;
    fn positionals(&self) -> usize;
    fn positional(&self, index: usize) -> Option<Self::Value>;
    fn keyword(&self, name: &str) -> Option<Self::Value>;
    fn keywords(&self) -> usize;
    fn keyword_name(&self, index: usize) -> Option<&'a str>;
    fn undefined(&self) -> Self::Value;
    fn caller(&self) -> bool;
}
#[derive(Debug)]
pub(crate) enum Failure<'a, E> {
    Positionals,
    Duplicate(&'a str),
    Unknown(&'a str),
    Destination(E),
}
pub(crate) fn bind<'a, V: View<'a>, E>(
    view: &V,
    mut push: impl FnMut(V::Value) -> Result<(), E>,
) -> Result<Option<V::Value>, Failure<'a, E>> {
    if view.positionals() > view.parameters() {
        return Err(Failure::Positionals);
    }
    for index in 0..view.parameters() {
        let value = match view.parameter(index) {
            None => view.undefined(),
            Some(name) => match (view.positional(index), view.keyword(name)) {
                (Some(_), Some(_)) => return Err(Failure::Duplicate(name)),
                (Some(value), None) | (None, Some(value)) => value,
                (None, None) => view.undefined(),
            },
        };
        push(value).map_err(Failure::Destination)?;
    }
    let caller = if view.caller() {
        Some(view.keyword("caller").unwrap_or_else(|| view.undefined()))
    } else {
        None
    };
    for index in 0..view.keywords() {
        if let Some(name) = view.keyword_name(index) {
            if !(view.caller() && name == "caller")
                && !(0..view.parameters()).any(|i| view.parameter(i) == Some(name))
            {
                return Err(Failure::Unknown(name));
            }
        }
    }
    Ok(caller)
}
pub(crate) fn control_bytes<V>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<V>(),
        size_of::<Option<V>>(),
        size_of::<(Option<V>, Option<V>)>(),
        size_of::<Option<&str>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<[usize; 2]>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
