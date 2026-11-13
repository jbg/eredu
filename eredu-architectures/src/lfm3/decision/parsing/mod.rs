//! Interpretation of numbers and dates written in a selected source span.

mod date;
mod number;

pub(super) use date::date;
pub(super) use number::{number, unit};
