//! Plain join's shared separator/iteration protocol. Formatting policy and
//! iterator/source authority remain with the concrete caller.
#![forbid(unsafe_code)]
use std::fmt::{self, Write};
pub(crate) fn plain<W: Write, I: Iterator, F: FnMut(&mut W, I::Item) -> fmt::Result>(
    output: &mut W,
    iter: I,
    separator: &str,
    mut item: F,
) -> fmt::Result {
    for (index, value) in iter.enumerate() {
        if index > 0 {
            output.write_str(separator)?;
        }
        item(output, value)?;
    }
    Ok(())
}
pub(crate) fn control_bytes<W: Write, I: Iterator, F>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<(&mut W, &str)>(),
        size_of::<I>(),
        size_of::<std::iter::Enumerate<I>>(),
        size_of::<Option<(usize, I::Item)>>(),
        size_of::<I::Item>(),
        size_of::<F>(),
        size_of::<usize>(),
        size_of::<fmt::Result>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
