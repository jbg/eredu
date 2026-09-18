use crate::fraction::{GenericFraction, Sign};
use crate::{Integer, Ratio, Zero};
use std::ops::Div;

impl<T, O> Div<O> for GenericFraction<T>
where
    T: Clone + Integer,
    O: Into<GenericFraction<T>>,
{
    type Output = Self;

    fn div(self, other: O) -> Self {
        self.div_with_operations(other.into(), &crate::operations::Ordinary)
            .unwrap_or_else(|error| match error {})
    }
}

impl<T, O> Div<O> for &GenericFraction<T>
where
    T: Clone + Integer,
    O: Into<GenericFraction<T>>,
{
    type Output = GenericFraction<T>;

    fn div(self, other: O) -> GenericFraction<T> {
        let other = other.into();
        match *self {
            GenericFraction::NaN => self.clone(),
            GenericFraction::Infinity(sign) => match other {
                GenericFraction::NaN => other,
                GenericFraction::Infinity(_) => GenericFraction::NaN,
                GenericFraction::Rational(osign, _) => {
                    GenericFraction::Infinity(if sign == osign {
                        Sign::Plus
                    } else {
                        Sign::Minus
                    })
                }
            },
            GenericFraction::Rational(sign, ref l) => match other {
                GenericFraction::NaN => other,
                GenericFraction::Infinity(_) => {
                    GenericFraction::Rational(Sign::Plus, Ratio::zero())
                }
                GenericFraction::Rational(osign, ref r) => {
                    if l.is_zero() && r.is_zero() {
                        GenericFraction::NaN
                    } else if r.is_zero() {
                        GenericFraction::Infinity(sign)
                    } else if l.is_zero() {
                        GenericFraction::Rational(Sign::Plus, l.clone())
                    } else {
                        GenericFraction::Rational(
                            if sign == osign {
                                Sign::Plus
                            } else {
                                Sign::Minus
                            },
                            l.div(r),
                        )
                    }
                }
            },
        }
    }
}

impl<T> Div for &GenericFraction<T>
where
    T: Clone + Integer,
{
    type Output = GenericFraction<T>;

    fn div(self, other: Self) -> GenericFraction<T> {
        match *self {
            GenericFraction::NaN => self.clone(),
            GenericFraction::Infinity(sign) => match *other {
                GenericFraction::NaN => other.clone(),
                GenericFraction::Infinity(_) => GenericFraction::NaN,
                GenericFraction::Rational(osign, _) => {
                    GenericFraction::Infinity(if sign == osign {
                        Sign::Plus
                    } else {
                        Sign::Minus
                    })
                }
            },
            GenericFraction::Rational(sign, ref l) => match *other {
                GenericFraction::NaN => other.clone(),
                GenericFraction::Infinity(_) => {
                    GenericFraction::Rational(Sign::Plus, Ratio::zero())
                }
                GenericFraction::Rational(osign, ref r) => {
                    if l.is_zero() && r.is_zero() {
                        GenericFraction::NaN
                    } else if r.is_zero() {
                        GenericFraction::Infinity(sign)
                    } else if l.is_zero() {
                        GenericFraction::Rational(Sign::Plus, l.clone())
                    } else {
                        GenericFraction::Rational(
                            if sign == osign {
                                Sign::Plus
                            } else {
                                Sign::Minus
                            },
                            l.div(r),
                        )
                    }
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GenericFraction;
    // use crate::{One, Zero};

    type F = GenericFraction<u8>;

    #[test]
    fn div_scalar() {
        assert_eq!(F::from(3) / 2, F::from(1.5));
        assert_eq!(F::from(3) / 1.5, F::from(2));

        assert_eq!(&F::from(3) / 2, F::from(1.5));
        assert_eq!(&F::from(3) / 1.5, F::from(2));

        assert_eq!(F::from(3) / F::from(2), F::from(1.5));
        assert_eq!(F::from(3) / F::from(1.5), F::from(2));

        assert_eq!(&F::from(3) / &F::from(2), F::from(1.5));
        assert_eq!(&F::from(3) / &F::from(1.5), F::from(2));
    }
}

impl<T: Clone + Integer> GenericFraction<T> {
    pub fn div_with_operations<P: crate::operations::IntegerOperations<T>>(
        self,
        other: Self,
        operations: &P,
    ) -> Result<Self, P::Error> {
        operations.reserve_controls(std::mem::size_of::<(
            Self,
            Self,
            &P,
            Sign,
            Sign,
            Result<Self, P::Error>,
        )>())?;

        Ok(match self {
            GenericFraction::NaN => self,
            GenericFraction::Infinity(sign) => match other {
                GenericFraction::NaN => other,
                GenericFraction::Infinity(_) => GenericFraction::NaN,
                GenericFraction::Rational(osign, _) => {
                    GenericFraction::Infinity(if sign == osign {
                        Sign::Plus
                    } else {
                        Sign::Minus
                    })
                }
            },
            GenericFraction::Rational(sign, l) => match other {
                GenericFraction::NaN => other,
                GenericFraction::Infinity(_) => GenericFraction::Rational(
                    Sign::Plus,
                    Ratio::new_raw(operations.zero()?, operations.one()?),
                ),
                GenericFraction::Rational(osign, r) => {
                    if l.is_zero() && r.is_zero() {
                        GenericFraction::NaN
                    } else if r.is_zero() {
                        GenericFraction::Infinity(sign)
                    } else if l.is_zero() {
                        GenericFraction::Rational(Sign::Plus, l)
                    } else {
                        GenericFraction::Rational(
                            if sign == osign {
                                Sign::Plus
                            } else {
                                Sign::Minus
                            },
                            l.div_with_operations(r, operations)?,
                        )
                    }
                }
            },
        })
    }
}
