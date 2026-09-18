//! Portable ordered collection storage with prospective allocation callbacks.
//!
//! This crate owns collection mechanics only. Callers retain admission policy,
//! allocation accounts and the custody required by their own payloads.
#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod ordered_map;
