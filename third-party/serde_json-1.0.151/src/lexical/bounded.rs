//! Storage of the existing round-trip float conversion, from its owning bounds.
use super::{
    bignum::Bigint,
    cached::{ModeratePathCache, ModeratePathPowers},
    float::ExtendedFloat,
    math::Limb,
    num::Float,
};
use core::{
    iter::Chain,
    mem::{size_of, size_of_val},
    slice::Iter,
};

fn decimal_digits(mut value: u64) -> usize {
    let mut digits = 1;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}

fn limb_bound() -> Option<usize> {
    let powers = ExtendedFloat::get_powers();
    let digits = <f64 as Float>::MAX_DIGITS;
    let bias = usize::try_from(powers.bias).ok()?;
    let step = usize::try_from(powers.step).ok()?;
    // Outside this cached range moderate_path returns a final zero/infinity,
    // so it never invokes bhcomp. Its retained u64 mantissa has at most these
    // significant decimal digits; leading fraction zeros do not contribute.
    let mantissa_digits = decimal_digits(u64::MAX);
    let positive = step
        .checked_mul(powers.large.len())?
        .checked_sub(bias)?
        .checked_sub(1)?
        .checked_add(mantissa_digits)?
        .checked_sub(1)?;
    // bhcomp scales scientific_exp + 1 - min(significant digits, MAX_DIGITS).
    let negative = bias.checked_add(digits)?.checked_sub(1)?;
    // 10 < 2^4 and 5 < 2^3. These are integer bit inequalities, not byte
    // multipliers. All operations are positive multiplication/addition/shift,
    // so intermediate magnitudes never exceed their final branch envelope.
    let mantissa_bits = digits.checked_mul(4)?;
    let theor_bits = usize::try_from(<f64 as Float>::MANTISSA_SIZE)
        .ok()?
        .checked_add(2)?;
    let smallest_theor =
        usize::try_from(1i32.checked_sub(<f64 as Float>::DENORMAL_EXPONENT)?).ok()?;
    let largest_theor = usize::try_from(<f64 as Float>::MAX_EXPONENT.checked_sub(2)?).ok()?;
    let positive_bits = mantissa_bits.checked_add(positive.checked_mul(4)?)?;
    let real_bits = mantissa_bits.checked_add(smallest_theor)?;
    let theoretical_bits = theor_bits
        .checked_add(negative.checked_mul(3)?)?
        .checked_add(largest_theor.checked_add(negative)?)?;
    let bits = positive_bits.max(real_bits).max(theoretical_bits);
    let limb_bits = size_of::<Limb>().checked_mul(8)?;
    bits.checked_add(limb_bits - 1)?.checked_div(limb_bits)
}

/// The same reserve used by each actual float Bigint constructor. Larger
/// arbitrary-precision math outside float parsing retains ordinary Vec growth;
/// the numeric profile proves every actual float operation stays within this.
pub(crate) fn bigint_limbs() -> usize {
    limb_bound().expect("compiled float/cached-power geometry fits usize")
}

/// parse_mantissa constructs one Bigint; small_atof constructs one theoretical
/// Bigint. Bigint's power worker is iterative and creates no product vectors.
/// large_atof consumes the first Bigint, and no selected float path clones one.
pub(crate) fn temporary_bytes() -> Option<usize> {
    limb_bound()?.checked_mul(size_of::<Limb>())?.checked_mul(2)
}

pub(crate) fn control_bytes() -> Option<usize> {
    let parts = [
        // The same pure limb-layout calculation also runs at Bigint birth.
        size_of::<(&ModeratePathPowers, usize, usize, usize, usize)>(),
        size_of::<(usize, usize, usize, usize, usize, usize, usize, usize)>(),
        size_of::<(usize, usize, usize, usize, u64)>(),
        // parse_concise_float's formatting is inline; the de formatter is
        // independently included by the owning deserializer control query.
        size_of::<itoa::Buffer>(),
        size_of::<(u64, i32, bool, ExtendedFloat, f64)>(),
        // fast_path exponent limits, mantissa masks, checked product and return.
        size_of::<(
            u64,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            u64,
            u64,
            f64,
            Option<f64>,
        )>(),
        // moderate_path and multiply_exponent_extended's cache/index/error loans.
        size_of::<(u64, i32, bool, ExtendedFloat, bool)>(),
        size_of::<(
            &mut ExtendedFloat,
            i32,
            bool,
            &ModeratePathPowers,
            i32,
            i32,
            i32,
            u32,
            u64,
            bool,
            u32,
        )>(),
        size_of::<(ExtendedFloat, ExtendedFloat)>(),
        // ExtendedFloat::mul's four halves, four products, carry and result.
        size_of::<(
            &ExtendedFloat,
            &ExtendedFloat,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            u64,
            ExtendedFloat,
        )>(),
        // error_is_accurate and nearest_error_is_accurate's exact scalar controls.
        size_of::<(u32, &ExtendedFloat, i32, i32, i32, u64, u64)>(),
        size_of::<(u64, &ExtendedFloat, u64, u64, u64, u64, bool, bool)>(),
        // parse_truncated_float: input loans, iterator, mantissa and counters.
        size_of::<(&[u8], &[u8], usize, u64, i32, bool)>(),
        size_of::<Chain<Iter<'static, u8>, Iter<'static, u8>>>(),
        // parse_mantissa result, bhcomp transfer and large_atof's consumed value.
        size_of::<Bigint>(),
        size_of::<Bigint>(),
        size_of::<Bigint>(),
        size_of::<(&[Limb], usize, usize, Limb, usize)>(),
        // small_atof's real/theoretical values and from_u64 construction.
        size_of::<Bigint>(),
        size_of::<Bigint>(),
        size_of::<Bigint>(),
        size_of::<(i32, i32, i32, i32, ExtendedFloat)>(),
        size_of::<[Limb; 2]>(),
        // Exact iterative small multiply/add/shift and output rounding frames.
        size_of::<(&mut alloc::vec::Vec<Limb>, &[Limb], usize, usize, Limb)>(),
        size_of::<(Limb, Limb, Limb, u128, bool)>(),
        size_of::<(usize, usize, usize, Limb, Limb, Limb)>(),
        size_of::<(ExtendedFloat, u64, i32, bool, bool, bool, f64)>(),
        // Native rounding: normalize, round_to_float, nearest/downward and ties.
        size_of::<(&mut ExtendedFloat, u32, i32, i32, i32)>(),
        size_of::<(&mut ExtendedFloat, i32, u64, u64, u64, bool, bool)>(),
        size_of::<(&mut ExtendedFloat, bool, bool, bool)>(),
        size_of::<(&mut ExtendedFloat, i32, u64, u64, bool)>(),
        // Overflow avoidance, bit-mask helpers, shifts and final bit conversion.
        size_of::<(&mut ExtendedFloat, i32, u64, u64, u64, i32)>(),
        size_of::<(u64, u64, u64, u64)>(),
        size_of::<(&mut ExtendedFloat, i32, u64)>(),
        size_of::<(ExtendedFloat, u64, u64, f64)>(),
        size_of::<core::cmp::Ordering>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
