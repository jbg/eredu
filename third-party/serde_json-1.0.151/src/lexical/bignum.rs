// Adapted from https://github.com/Alexhuszagh/rust-lexical.

//! Big integer type definition.

use super::math::*;
#[allow(unused_imports)]
use alloc::vec::Vec;

/// Storage for a big integer type.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Bigint {
    /// Internal storage for the Bigint, in little-endian order.
    pub(crate) data: Vec<Limb>,
}

impl Default for Bigint {
    fn default() -> Self {
        Bigint {
            data: Vec::with_capacity(super::bounded::bigint_limbs()),
        }
    }
}

impl Math for Bigint {
    #[inline]
    fn imul_pow5(&mut self, n: u32) {
        // Same exact multiplication worker as the ordinary small-power branch.
        // Float conversion needs only two Bigints; no recursive product vectors
        // are created, and the owning float limb bound covers every growth.
        imul_pow5_iterative(&mut self.data, n);
    }

    #[inline]
    fn data(&self) -> &Vec<Limb> {
        &self.data
    }

    #[inline]
    fn data_mut(&mut self) -> &mut Vec<Limb> {
        &mut self.data
    }
}
